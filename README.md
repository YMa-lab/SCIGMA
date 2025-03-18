# SCIGMA
Spatially informed, Contrastive learning-based Integration with Graph neural 
networks for Multi-sample multi-modal Analysis

**Seowon Chang, Ying Ma**


## Overview
We present SCIGMA,a deep learning framework for integrating multi modal spatial omics data.
Using uncertainty-based contrastive learning that accounts for intra- and inter-modality
alignment, SCIGMA can accurately align multiple modalities. 
SCIGMA has been evaluated on a variety of modalities and technologies, including
spatial ATAC-seq, SPOTS, 10xXenium and 10xXenium Prime 5K, 10x VisiumHD, 
Stereo-CITE-seq, CUT&Tag seq, and spatial metabolomics. 


## Installation
- Clone the repository
```bash
git clone https://github.com/YMa-lab/SCIGMA.git
```
- Create a virtual environment (python or conda) with Python 3.8
```bash
conda create -n SCIGMA python=3.8
```
- Activate the environment
```bash
conda activate SCIGMA
```
- Install R packages
```bash
conda install -c conda-forge r-base=4.0.5```, ```conda install -c conda-forge r-mclust==5.4.9
```
- Install base python packages 
```bash
pip install -r /path/to/requirements.txt
```
- Install CUDA related packages
```bash
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2+cu117 -f https://download.pytorch.org/whl/torch_stable.html
```
- For Jupyter notebook: install ipykernel
```bash
conda install -c anaconda ipykernel
```
```python
python -m ipykernel install --user --name=SCIGMA
```

## Tutorial
For running SCIGMA on a dataset, refer to our tutorial: https://github.com/YMa-lab/SCIGMA/blob/main/tutorial/SCIGMA_Tutorial.ipynb 
