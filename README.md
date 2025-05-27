# Diffusion Based Ambiguous Image Segmentation

The official repository for the paper "Diffusion Based Ambiguous Image Segmentation". This repository contains the code and model checkpoints to reproduce the central results of the paper. The main functionalities of the code are:

- **Training with train.py**: Train the diffusion model from scratch.
- **Evaluation with evaluate.py**: Evaluate a saved diffusion model or checkpoint on the data.
- **Basic visualization with visualize.ipynb**: Plot training losses or visualize the samples produced by the diffusion model.

## Environment Setup
After cloning the repository, create a conda environment and install the required packages:
```bash
conda create --name amb-env python=3.12 pip --no-default-packages --yes
conda activate amb-env
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

## Data Preparation
Simply download the dataset from [Google Drive](https://drive.google.com/drive/folders/1xKfKCQo8qa6SAr3u7qWNtQjIphIrvmd5) and place it the pickle file in the `data` folder. The dataset is from the [Probibalistic UNet repo](https://github.com/stefanknegt/Probabilistic-Unet-Pytorch?tab=readme-ov-file).

## Training and Evaluation
We include training and evaluation setups for our best performing models, in both the default setting (`configs/config_default.yaml`) and random cropping mode (`configs/config_crop64.yaml`). To train a model with e.g. `configs/config_default.yaml`, run:
```bash
python src/amb_diff_seg/train.py configs/config_default.yaml
```
and similarly, to evaluate run
```bash
python src/amb_diff_seg/evaluate.py configs/config_default.yaml
```

You can change any config arguments in the command line. If you want to train a model with 1e-3 learning rate instead of the default 1e-4, run:
```bash
python src/amb_diff_seg/train.py configs/config_default.yaml training.lr=1e-3
```

After training a model, you can evaluate it with
```bash
python src/amb_diff_seg/evaluate.py configs/config_default.yaml
```
## Evaluation With Our Checkpoints
You can also specify our checkpoints and evaluate these. We supply our best checkpoints in the `checkpoints` folder for both the default 128x128 setting and random cropping 64x64 setting. To evaluate these from our checkpoints, run:
```bash
python src/amb_diff_seg/evaluate.py configs/config_default.yaml sampling.load_ckpt=checkpoints/default_ckpt.pt
python src/amb_diff_seg/evaluate.py configs/config_crop64.yaml sampling.load_ckpt=checkpoints/crop64_ckpt.pt
```


Please cite our work if you use this code or the knowledge from the paper in your own work:
```
@inproceedings{christensen2025diffusionbasedambiguousimage,
      title={Diffusion Based Ambiguous Image Segmentation}, 
      author={Jakob Lønborg Christensen and Morten Rieger Hannemose and Anders Bjorholm Dahl and Vedrana Andersen Dahl},
      year={2025},
      journal={Scandinavian Conference on Image Analysis (SCIA)}, 
}
```