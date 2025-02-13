#!/usr/bin/env python
"""
# Author: ChangXu
# Created Time : Mon 23 Apr 2021 08:26:32 
# File Name: his_feat.py
# Description:`

"""
"""


"""
import os
import math
import anndata as ad
import numpy as np 
import scanpy as sc
import pandas as pd 
from PIL import Image
from pathlib import Path
from scipy.sparse import issparse, isspmatrix_csr, csr_matrix, spmatrix
from sklearn.metrics import pairwise_distances
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from tqdm import tqdm
import random

import torch
import torch.nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights, ResNet18_Weights
from torch.autograd import Variable 
import torchvision.transforms as transforms


class ImageFeature:
    def __init__(
        self,
        adata,
        pca_components=50,
        cnnType='ResNet18',
        seeds=2025,
        crop_size=64,
        target_size=224,
    ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.adata = adata
        self.pca_components = pca_components
        self.seeds = seeds
        self.cnnType = cnnType
        self.target_size=target_size
        self.crop_size = crop_size

    def load_cnn_model(
        self,
        ):
        
        if self.cnnType == 'EfficientNetB1':
            cnn_pretrained_model = models.efficientnet_b1(weights='EfficientNet_B1_Weights.IMAGENET1K_V2')
            cnn_pretrained_model.to(self.device) 
        if self.cnnType == 'ResNet18':
            cnn_pretrained_model = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
            cnn_pretrained_model.to(self.device)           
        elif self.cnnType == 'Vgg19':
            cnn_pretrained_model = models.vgg19(pretrained=True)
            cnn_pretrained_model.to(self.device)
        elif self.cnnType == 'Vgg16':
            cnn_pretrained_model = models.vgg16(pretrained=True)
            cnn_pretrained_model.to(self.device)
        return cnn_pretrained_model

    def extract_image_feat(
        self,
        ):
        weights = ResNet18_Weights.IMAGENET1K_V1
        preprocess = weights.transforms(antialias=True)
        

        feat_df = []
        model = self.load_cnn_model()
        model.eval()
        
        image = self.adata.uns['image']
        
        if len(image.shape) > 2:
            print("multi-channel image")
            img_pillow = Image.fromarray(image)
        else:
            print("single-channel image")
            img_pillow = Image.fromarray(image)


        with tqdm(total=len(self.adata),
              desc="Extract image feature",
              bar_format="{l_bar}{bar} [ time left: {remaining} ]",) as pbar:
            for imagerow, imagecol in self.adata.obsm["spatial"]:
                # crop
                imagerow_down = imagerow - self.crop_size / 2
                imagerow_up = imagerow + self.crop_size / 2
                imagecol_left = imagecol - self.crop_size / 2
                imagecol_right = imagecol + self.crop_size / 2
                tile = img_pillow.crop(
                    (imagecol_left, imagerow_down, imagecol_right, imagerow_up))
                
                spot_slice = np.asarray(tile, dtype="float32")
                if len(spot_slice.shape) > 2:
                    spot_slice = spot_slice.astype(np.float32)
                    spot_slices = np.moveaxis(spot_slice, -1, 0)
                else:  
                    spot_slice = np.reshape(spot_slice, (self.crop_size,self.crop_size))
                    spot_slice = spot_slice.astype(np.float32)
                    spot_slices = np.stack((spot_slice,spot_slice,spot_slice))
                if np.max(spot_slices) > 0:
                    spot_slices = spot_slices / 255
                tensor = preprocess(torch.from_numpy(spot_slices))
                tensor = tensor.resize_(1,3,self.target_size,self.target_size)
                tensor = tensor.to(self.device)
                result = model(Variable(tensor))
                result_npy = result.data.cpu().numpy().ravel()
                feat_df.append(result_npy)
                pbar.update(1)
        feat_df = sc.pp.scale(np.array(feat_df))
        self.adata.obsm["image_feat"] = feat_df

        pca = PCA(n_components=self.pca_components, random_state=self.seeds)
        pca.fit(feat_df)
        self.adata.obsm["image_feat_pca"] = pca.transform(feat_df)
        return self.adata 