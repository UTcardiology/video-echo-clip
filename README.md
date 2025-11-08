# Video CLIP Model for Multi-View Echocardiography Interpretation
#### [Ryo Takizawa](https://crumbyrobotics.github.io), Satoshi Kodera, Tempei Kabayama, Ryo Matsuoka, Yuta Ando, Yuto Nakamura, Haruki Settai, Norihiko Takeda
NeurIPS 2025 Workshop on Learning from Time Series for Health (TS4H)

📄 **Preprint:** [arxiv.org/abs/2504.18800](https://arxiv.org/abs/2504.18800)

<img width="1657" height="701" alt="Screenshot from 2025-11-05 23-17-04" src="https://github.com/user-attachments/assets/508b5d5d-bc04-41fe-8d29-420ce74fc963" />


## Installation

We recommend using a virtual environment (e.g. `venv` or `conda`) to avoid package conflicts.

```bash
git clone <this-repo-url>
cd <this-repo>
pip install -r requirements.txt
````


## Directory Structure

Key files and directories:

* **`run_train.py`**
  Training script for the CLIP model on multi-view echocardiography data.

  * Supports DDP and other distributed training setups.
  * Reads paths and options from `config/config.yaml` and related config files.

* **`eval.ipynb`**
  Jupyter notebook for evaluating a trained CLIP model.

  * Shows how to load a checkpoint.
  * Demonstrates basic evaluation workflows.

* **`config/config.yaml`**
  Main configuration file for:

  * Dataset paths
  * Training hyperparameters
  * Model settings

* **`config/model/*.yaml`**
  Configuration files for different video encoder backbones.
  You can select a model in `config/config.yaml` via:

  ```yaml
  defaults:
    - model: <model_name>
  ```

  where `<model_name>` matches one of the YAML filenames under `config/model/`.

* **`example1.mp4`, `example2.mp4`**
  Sample echocardiography videos for testing.

* **`example1.txt`, `example2.txt`**
  Sample Japanese report texts corresponding to the example videos.

* **`example_dataset.csv`** 
  Example dataset in CSV format.

* **`csv2wds.py`** 
  Utility script for converting the dataset from CSV format to WebDataset format.

## Pretrained Models

We provide pretrained model weights for different video encoder backbones on [Hugging Face](https://huggingface.co/r-takizawa/).

| Model Name | Pretraining Source | Description | Download |
|---|---|---|---|
| `videomae2` | VideoMAE v2 pretrained | VideoMAE2 encoder further fine-tuned on multi-view echocardiography. | **[checkpoint](https://huggingface.co/r-takizawa/VideoEchoCLIP-videomae2/resolve/main/checkpoint_videomae2.pt?download=true)** |
| `vivit-kinetics400` | ViViT pretrained on Kinetics-400 | ViViT encoder initialized from Kinetics-400 weights and further fine-tuned. | **[checkpoint](https://huggingface.co/r-takizawa/VideoEchoCLIP-vivit-kinetics400/resolve/main/checkpoint_vivit-kinetics400.pt?download=true)** |
| `vivit-scratch` | None (from scratch) | ViViT variant trained **from scratch** on multi-view echocardiography. | **[checkpoint](https://huggingface.co/r-takizawa/VideoEchoCLIP-vivit-scratch/resolve/main/checkpoint_vivit-scratch.pt?download=true)** |

---

### How to Use Pretrained Weights
Place downloaded `.pt` weight files under the `weights/` directory:
```bash
mkdir -p weights
mv <downloaded>.pt ./weights/
```


## Training
Before training, edit **`config/config.yaml`** to:

* Set dataset paths (e.g. training / validation data).
* Adjust batch size, learning rate, and other training hyperparameters.
* Choose the video encoder backbone via the `defaults.model` entry.

For backbone-specific options, refer to the corresponding file in `config/model/*.yaml`.


Run the training script as:

```bash
python run_train.py
```

To use distributed training (e.g. DDP), launch with your preferred launcher (such as `torchrun`).

```bash
torchrun --nproc_per_node=4 -m run_train
```

## Citation

If you use this repository or the associated model in your research, please cite:

```bibtex
@inproceedings{takizawa2025videoclip,
  title     = {Video CLIP Model for Multi-View Echocardiography Interpretation},
  author    = {Takizawa, Ryo and Kodera, Satoshi and Kabayama, Tempei and 
               Matsuoka, Ryo and Ando, Yuta and Nakamura, Yuto and 
               Settai, Haruki and Takeda, Norihiko},
  booktitle = {NeurIPS 2025 Workshop TS4H},
  year      = {2025}
}
```
