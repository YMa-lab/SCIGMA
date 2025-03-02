# SCIGMA
Spatially informed, Contrastive learning-based Integration with Graph neural 
networks for Multi-sample multi-modal Analysis

## Environment Steup
- Create a virtual environment (python or conda)
```conda create -n SCIGMA python=3.8```
- Activate the environment
```conda activate SCIGMA```
- Install R packages
```conda install -c conda-forge r-base=4.0.5```, ```conda install -c conda-forge r-mclust==5.4.9```
- Install base python packages 
```pip install -r /path/to/requirements.txt```
- Install CUDA related packages
```pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2+cu117 -f https://download.pytorch.org/whl/torch_stable.html```
- For Jupyter notebook: install ipykernel
```conda install -c anaconda ipykernel```
```python -m ipykernel install --user --name=SCIGMA```
