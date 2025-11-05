import logging
import os
from datetime import datetime
import torch
from torch import optim
from torch.cuda.amp import GradScaler
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from open_clip.loss import ClipLoss, SigLipLoss
from open_clip.tokenizer import HFTokenizer

from videoechoclip.utils import random_seed, get_latest_checkpoint, pt_load, setup_logging
from videoechoclip.distributed import is_master, init_distributed_device, broadcast_object
from videoechoclip.scheduler import cosine_lr, const_lr, const_lr_cooldown
from videoechoclip.dataset import get_data
from videoechoclip.train import train_one_epoch, evaluate
from videoechoclip.model import create_model_and_transforms


LATEST_CHECKPOINT_NAME = "epoch_latest.pt"


@hydra.main(config_name="config", config_path="config", version_base="1.1")
def main(args: DictConfig):
    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # Fully initialize distributed device environment
    device = init_distributed_device(args)

    if args.distributed:
        logging.info(
            f"Running in distributed mode with multiple processes. Device: {args.device}.Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}."
        )
    else:
        logging.info(f"Running with a single process. Device {args.device}.")

    # Set the name of the experiments if not be set (set name when you want to resume the past training)
    if args.name is None:
        date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        if args.distributed:
            # sync date_str from master to all ranks
            date_str = broadcast_object(args, date_str)
        args.name = "-".join(
            [
                date_str,
                f"lr_{args.lr}",
                f"b_{args.batch_size}",
                f"j_{args.workers}",
                f"p_{args.precision}",
            ]
        )

    # Create log file
    args.log_path = None
    log_base_path = os.path.join(args.logs, args.name)
    if is_master(args, local=args.log_local):
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f"out-{args.rank}" if args.log_local else "out.log"
        args.log_path = os.path.join(log_base_path, log_filename)
        if os.path.exists(args.log_path) and not args.resume:
            logging.error("Error. Experiment already exists. Use --name {} to specify a new experiment.")
            exit(1)

    # Setup text logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    # Setup checkpoint logging
    args.checkpoint_path = os.path.join(log_base_path, "checkpoints")
    if is_master(args):
        os.makedirs(args.checkpoint_path, exist_ok=True)

    # Determine if this worker should save checkpoints. only do so if it is rank == 0
    args.save_checkpoints = args.logs and args.logs.lower() != "none" and is_master(args)

    # Load latest checkpoint if resume training from latest
    if args.resume:
        resume_from = None
        checkpoint_path = args.checkpoint_path
        if is_master(args):
            # Checking for existing checkpoint via master rank only. It is possible for
            # different rank processes to see different files if a shared file-system is under
            # stress, however it's very difficult to fully work around such situations.
            if args.save_most_recent:
                # if --save-most-recent flag is set, look for latest at a fixed filename
                resume_from = os.path.join(checkpoint_path, LATEST_CHECKPOINT_NAME)
                if not os.path.exists(resume_from):
                    # If no latest checkpoint has been saved yet, don't try to resume
                    resume_from = None
            else:
                # otherwise, list checkpoint dir contents and pick the newest checkpoint
                resume_from = get_latest_checkpoint(checkpoint_path, epoch=args.checkpoint_epoch)

            if resume_from:
                logging.info(f"Found latest resume checkpoint at {resume_from}.")
            else:
                logging.error(f"No latest resume checkpoint found in {checkpoint_path}.")
                exit(1)

        if args.distributed:
            # sync found checkpoint path to all ranks
            resume_from = broadcast_object(args, resume_from)

    if args.precision == "fp16":
        logging.warning(
            "It is recommended to use AMP mixed-precision instead of FP16. FP16 support needs further verification and tuning, especially for train."
        )

    random_seed(args.seed, rank=0)

    # Create model & preprocess functions
    model, preprocess_train, preprocess_val = create_model_and_transforms(args)

    # Freeze model parameters
    if args.lock_text:
        model.lock_text_tower(unlocked_layers=args.lock_text_unlocked_layers, freeze_layer_norm=args.lock_text_freeze_layer_norm)

    random_seed(args.seed, args.rank)

    # Log model name & config
    if is_master(args):
        logging.info("Model:")
        logging.info(f"{str(model)}")
        logging.info("Params:")
        logging.info(f"{OmegaConf.to_yaml(args)}")
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            f.write(f"{OmegaConf.to_yaml(args)}")

    # Create DDP model
    if args.distributed:
        if args.ddp_static_graph:
            # this doesn't exist in older PyTorch, arg added indirectory to avoid errors
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], **dict(static_graph=True))
        else:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])

    # Create optimizer and scaler (when amp)
    optimizer = None
    scaler = None
    if args.train_data:

        def exclude(n, p):
            return p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or "logit_scale" in n

        def include(n, p):
            return not exclude(n, p)

        named_parameters = list(model.named_parameters())
        gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
        rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

        optimizer = optim.AdamW(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.0},
                {"params": rest_params, "weight_decay": args.wd},
            ],
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
        )
        scaler = GradScaler() if args.precision == "amp" else None

    # Optionally resume from a checkpoint
    start_epoch = 0
    if args.resume:
        checkpoint = pt_load(resume_from, map_location="cpu")
        if "epoch" in checkpoint:
            # resuming a train checkpoint w/ epoch and optimizer state
            start_epoch = checkpoint["epoch"]

            sd = checkpoint["state_dict"]
            if not args.distributed and next(iter(sd.items()))[0].startswith("module"):
                sd = {k[len("module.") :]: v for k, v in sd.items()}
            model.load_state_dict(sd)

            if optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if scaler is not None and "scaler" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler"])

            logging.info(f"=> resuming checkpoint '{resume_from}' (epoch {start_epoch})")
        else:
            # loading a bare (model only) checkpoint for evaluation only
            model.load_state_dict(checkpoint)
            logging.info(f"=> loaded checkpoint '{resume_from}' (epoch {start_epoch})")

    # Create tokenizer
    tokenizer = HFTokenizer(args.model.text.hf_tokenizer_name, context_length=args.model.text.context_length)
    if tokenizer.tokenizer.name_or_path == "rinna/japanese-roberta-base":
        tokenizer.tokenizer.do_lower_case = True  # NOTE due to the bugs in T5TokenizerFast

    # Create dataset
    data = get_data(args, preprocess_train, preprocess_val, epoch=start_epoch, tokenizer=tokenizer)
    assert len(data), "At least one train or val dataset must be specified."

    args.train_sz = data["train"].dataloader.num_samples if args.train_data is not None else 0
    args.val_sz = data["val"].dataloader.num_samples if args.val_data is not None else 0

    # Create scheduler if train
    scheduler = None
    if "train" in data and optimizer is not None:
        total_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs
        if args.lr_scheduler == "cosine":
            scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)
        elif args.lr_scheduler == "const":
            scheduler = const_lr(optimizer, args.lr, args.warmup, total_steps)
        elif args.lr_scheduler == "const-cooldown":
            assert args.epochs_cooldown is not None, "Please specify the number of cooldown epochs for this lr schedule."
            cooldown_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs_cooldown
            scheduler = const_lr_cooldown(optimizer, args.lr, args.warmup, total_steps, cooldown_steps, args.lr_cooldown_power, args.lr_cooldown_end)
        else:
            logging.error(f"Unknown scheduler, {args.lr_scheduler}. Available options are: cosine, const, const-cooldown.")
            exit(1)

    # Init wandb
    if args.wandb and is_master(args):
        assert wandb is not None, "Please install wandb."
        logging.debug("Starting wandb.")
        wandb.init(
            dir=os.path.join(args.logs, "wandb"),
            project=args.project,
            name=args.name,
            id=args.name,
            tags=[],
            resume="auto" if args.resume else None,
            config=dict(args),
        )
        if args.debug:
            wandb.watch(model, log="all")
        wandb.save(params_file)
        logging.debug("Finished loading wandb.")

    # Compile model to speed up training
    #   Pytorch 2.0 adds '_orig_mod.' prefix to keys of state_dict() of compiled models.
    #   For compatibility, we save state_dict() of the original model, which shares the
    #   weights without the prefix.
    original_model = model
    if args.torchcompile:
        logging.info("Compiling model...")
        model = torch.compile(original_model)

    # Evaluate only if train data is not given
    if "train" not in data:
        evaluate(model, data, start_epoch, args)
        return

    # Create Loss
    if args.siglip:
        loss = SigLipLoss(
            rank=args.rank,
            world_size=args.world_size,
        )
    else:
        loss = ClipLoss(
            local_loss=args.local_loss,
            gather_with_grad=args.gather_with_grad,
            cache_labels=True,
            rank=args.rank,
            world_size=args.world_size,
        )

    # Train
    for epoch in range(start_epoch, args.epochs):
        if is_master(args):
            logging.info(f"Start epoch {epoch}")

        train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, args)
        completed_epoch = epoch + 1

        if "val" in data:
            evaluate(model, data, completed_epoch, args)

        # saving checkpoints
        if args.save_checkpoints:
            checkpoint_dict = {
                "epoch": completed_epoch,
                "name": args.name,
                "state_dict": original_model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            if scaler is not None:
                checkpoint_dict["scaler"] = scaler.state_dict()

            if completed_epoch == args.epochs or (args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0):
                torch.save(
                    checkpoint_dict,
                    os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt"),
                )
            if args.delete_previous_checkpoint:
                previous_checkpoint = os.path.join(args.checkpoint_path, f"epoch_{completed_epoch - 1}.pt")
                if os.path.exists(previous_checkpoint):
                    os.remove(previous_checkpoint)

            if args.save_most_recent:
                # try not to corrupt the latest checkpoint if save fails
                tmp_save_path = os.path.join(args.checkpoint_path, "tmp.pt")
                latest_save_path = os.path.join(args.checkpoint_path, LATEST_CHECKPOINT_NAME)
                torch.save(checkpoint_dict, tmp_save_path)
                os.replace(tmp_save_path, latest_save_path)

    # Finish wandb
    if args.wandb and is_master(args):
        wandb.finish()


if __name__ == "__main__":
    main()
