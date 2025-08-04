<div align="center">

# Debris Segmentation using Post-Hurricane Aerial Imagery

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

## Description

This is the official implementation of the paper *Debris Segmentation using Post-Hurricane Aerial Imagery*.

## Installation

#### Pip

```bash
# clone project
git clone https://github.com/Way-Yuhao/CLIPSeg-debris
cd CLIPSeg-debris

# [OPTIONAL] create conda environment
conda create -n clipseg python=3.10
conda activate clipseg

# install pytorch according to instructions: 
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

## How to run

Navigate to Design-Safe to download the annotated debris dataset proposed in this work.

### Train & Fine-tune
Fine-tune CLIPSeg model on the dataset using the provided training script.

```bash
# train on GPU, e.g., GPU 0
python ./src/train.py experiment=clipseg_finetune trainer.devices=[0]
```
Experiment configuration can be found from [configs/experiment/](configs/experiment/)

### Test & Evaluate
Test the fine-tuned model on the dataset using the provided testing script.

```bash
python ./src/eval.py experiment=retest trainer.devices=[0] name=DIR_NAME ckpt_path=CKPT_PATH
```
Replace `DIR_NAME` with the name of the directory where the results will be saved, and `CKPT_PATH` with the path to the 
checkpoint file of the fine-tuned model.

### Prediction
Predict the segmentation mask for a single image using the fine-tuned model.

```bash
python ./src/eval.py trainer.devices=[0] data.query_images_dir=INPUT_DIR ckpt_path=CKPT_PATH name=DIR_NAME
```
Replace `INPUT_DIR` with the directory containing the input images, `CKPT_PATH` with the path to the checkpoint file 
of the fine-tuned model, and `DIR_NAME` with the name of the directory where the results will be saved.


