import os
import anndata
import torch
import numpy as np
import scanpy as sc
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
import scipy.sparse as sp
import networkx as nx
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph 
from pathlib import Path
from img_feat import ImageFeature
from scipy.sparse import coo_matrix
from scipy import sparse
from utils import lsi
import gc


def read_10xVis(path, quality='full'):
    adata = sc.read_visium(path)
    adata.var_names_make_unique()
    print(adata.uns["spatial"])
    
    library_id = list(adata.uns["spatial"].keys())[0]
    if quality == "full":
        image_coor = adata.obsm["spatial"]
        img = plt.imread(path+'/spatial/tissue_hires_image.png', 0)
        adata.uns["spatial"][library_id]["images"]["full"] = img
    else:
        scale = adata.uns["spatial"][library_id]["scalefactors"][
            "tissue_" + quality + "_scalef"]
        image_coor = adata.obsm["spatial"] * scale
    adata.obs["imagecol"] = image_coor[:, 0]
    adata.obs["imagerow"] = image_coor[:, 1]
    adata.uns["spatial"][library_id]["use_quality"] = quality
    
    
    # image feats
    save_path_image_crop = Path(os.path.join('DLFPC_out', 'Image_crop', 'library_id'))
    save_path_image_crop.mkdir(parents=True, exist_ok=True)
    adata = image_crop(adata, save_path=save_path_image_crop)
    adata = ImageFeature(adata, pca_components = 30).extract_image_feat()
    
    # image adata
    img_adata = anndata.AnnData(adata.obsm['image_feat_pca'])
    img_adata.obsm['spatial'] = adata.obsm['spatial']
    img_adata.obs_names = adata.obs_names
    
    return adata, img_adata

def read_10xXenium(path, radius=16):
    print("Loading expression")
    adata = sc.read_10x_h5(path+'/cell_feature_matrix.h5')
    adata.var_names_make_unique()
    cell_info = pd.read_csv(path+'/cells.csv.gz', sep=",")
    adata.obsm['spatial'] = cell_info[['x_centroid','y_centroid']].to_numpy()
    
    # get image
    print("Loading image")
    DAPI = []
    X_hvg = pd.DataFrame(adata.X.toarray(), 
                         index=adata.obs.index,
                         columns=adata.var.index)
    num_nodes = X_hvg.shape[0]
    im1 = tifffile.TiffFile(path+'/morphology.ome.tif')
    for i in range(len(im1.pages)):
        temp = im1.pages[i].asarray().astype(dtype=float)
        if len(DAPI) == 0:
            DAPI = temp
        else:
            DAPI = DAPI+temp
    DAPI = np.expand_dims(DAPI,axis=-1)
    print(DAPI.shape)

    
    extracted_feats = []
    indices = []
    DAPI[:,:,0] = DAPI[:,:,0] / np.max(DAPI[:,:,0])
    i = 0
    r = radius
    for x,y in adata.obsm['spatial']:
        x = int(x)
        y = int(y)
        if (DAPI[x-r:x+r,y-r:y+r,:].shape[0] == r*2) and (DAPI[x-r:x+r,y-r:y+r,:].shape[1] == r*2):
            indices.append(i)
            extracted_feats.append(DAPI[x-r:x+r,y-r:y+r,:].flatten())
        i += 1
    print(len(extracted_feats))
    img_X = np.stack(extracted_feats)
    
    #align the anndatas
    image = anndata.AnnData(img_X)
    image.obsm['spatial'] = cell_info[['x_centroid','y_centroid']].to_numpy()[indices]
    shifted_obs = []
    for x in image.obs_names:
        shifted_obs.append(str(int(x)+1))
    image.obs_names = shifted_obs
    adata = adata[image.obs_names]
    
    return adata, image


def read_10xXeniumPrime(path, radius=16):
    print("reading rna")
    # gene expression
    adata = sc.read_10x_h5(path+'/cell_feature_matrix.h5')
    adata.var_names_make_unique()
    cell_info = pd.read_csv(path+'/cells.csv.gz', sep=",")
    adata.obsm['spatial'] = cell_info[['x_centroid','y_centroid']].to_numpy()
    adata.X = sparse.csr_matrix(adata.X)
    
    # image
    print("reading image")
    fullres_multich_img = tifffile.imread(path+'/morphology_focus/morphology_focus_0000.ome.tif', is_ome=False, level=0)
    im = fullres_multich_img.astype(np.float32)
    del fullres_multich_img
    im = (im-np.min(im))/(np.max(im)-np.min(im))
    im = np.expand_dims(im,axis=0)
    im = np.concatenate((im,im,im),axis=0)
    adata.uns['image'] = im
    del im
    
    print("Processing image with pretrained model")
    adata = ImageFeature(adata, pca_components=50).extract_image_feat()
    
    img_adata = anndata.AnnData(adata.obsm['image_feat_pca'])
    img_adata.obsm['spatial'] = adata.obsm['spatial']
    img_adata.obs_names = adata.obs_names

    return adata, image


def load_data(input='SPOTS', paths=None, multisample=False, rep = 0):
    """
    Reading adata
    """
    ## mouse spleen (SPOTS)
    if input == 'SPOTS':
        adata_omics1 = sc.read_h5ad('SpatialGlue/Mouse_Spleen/adata_RNA.h5ad')
        adata_omics2 = sc.read_h5ad('SpatialGlue/Mouse_Spleen/adata_Pro.h5ad')  
    
    ## mouse thymus (Stereo-CITE-seq)
    if input == 'Stereo-CITE-seq':
        adata_omics1 = sc.read_h5ad('SpatialGlue/Mouse_Thymus/adata_RNA.h5ad')
        adata_omics2 = sc.read_h5ad('SpatialGlue/Mouse_Thymus/adata_ADT.h5ad')
    
    ## mouse brain (Spatial-ATAC-RNA-seq)
    if input == 'Spatial-ATAC-RNA-seq':
        adata_omics1 = sc.read_h5ad('SpatialGlue/Mouse_Brain/adata_RNA.h5ad')
        adata_omics2 = sc.read_h5ad('SpatialGlue/Mouse_Brain/adata_peaks_normalized.h5ad')
        
    ## breast cancer panel (10xXenium)
    if input == 'Xenium':
        if paths:
            rep1_path = paths[0]
            rep2_path = paths[1]
        else: 
            rep1_path ='10xXenium_Rep1'
            rep2_path = '10xXenium_Rep2'
        if multisample:
            adata_omics1, img_adata1 = read_10xXenium(rep1_path)
            adata_omics2, img_adata2 = read_10xXenium(rep2_path)
            return adata_omics1, img_adata1, adata_omics2, img_adata2
        else:
            if rep == 0:
                adata_omics1, img_adata1 = read_10xXenium(rep1_path)
                return adata_omics1, img_adata1
            else:
                adata_omics2, img_adata2 = read_10xXenium(rep2_path)
                return adata_omics2, img_adata2
        
    ## mouse brain panel (10xXeniumPrime)
    if input == 'XeniumPrime':
        if not paths:
            paths = '/oscar/data/yma16/Project/SCIGMA/data/10xXenium5K/'
        adata_omics1, img_adata1 = read_10xXeniumPrime(paths)
        return adata_omics1, img_adata1
        
    ## DLPFC
    if input == 'DLPFC':
        adata_omics1, img_adata1 = read_10xVis('DLPFC/151673')
        adata_omics2, img_adata2 = read_10xVis('DLPFC/151674')
        return adata_omics1, adata_omics2, img_adata1, img_adata2
    
    # make feature names unique
    adata_omics1.var_names_make_unique()
    adata_omics2.var_names_make_unique()

    return adata_omics1, adata_omics2
    
def preprocessing(adata_omics1, adata_omics2, datatype='SPOTS', n_neighbors=8, feat_neighbors=8): 
    if datatype not in ['SPOTS', 'Stereo-CITE-seq', 'Spatial-ATAC-RNA-seq', 'Xenium', 'XeniumPrime','DLPFC', 'MSI', 'VisiumHD', 'Multiplex']:
        raise ValueError("The datatype is not supported.") 
    
    print("Preprocessing anndatas")
    if datatype == 'SPOTS':  # mouse spleen 
      # RNA
      sc.pp.filter_genes(adata_omics1, min_cells=10)
        
      sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
      sc.pp.normalize_total(adata_omics1, target_sum=1e6)
      sc.pp.log1p(adata_omics1)
      adata_omics1.layers['logcounts'] = adata_omics1.X
      sc.pp.scale(adata_omics1)
      
      adata_omics1_high =  adata_omics1[:, adata_omics1.var['highly_variable']]
      sc.pp.pca(adata_omics1_high, n_comps=adata_omics2.n_vars-1)
      adata_omics1.obsm['feat'] = adata_omics1_high.obsm['X_pca']
    
      # Protein
      adata_omics2 = adata_omics2[adata_omics1.obs_names].copy()
      adata_omics2 = clr_normalize_each_cell(adata_omics2)
      adata_omics2.layers['logcounts'] = adata_omics2.X
      sc.pp.scale(adata_omics2)
      sc.pp.pca(adata_omics2, n_comps=adata_omics2.n_vars-1)
      adata_omics2.obsm['feat'] = adata_omics2.obsm['X_pca'].copy()
    
    elif datatype == 'Xenium':  # Xenium data
      # RNA
      sc.pp.filter_cells(adata_omics1, min_counts=100)
      sc.pp.filter_genes(adata_omics1, min_cells=10)
    
      sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
      sc.pp.normalize_total(adata_omics1, target_sum=1e6)
      sc.pp.log1p(adata_omics1)
      sc.pp.scale(adata_omics1)
      
      adata_omics1_high =  adata_omics1[:, adata_omics1.var['highly_variable']]
      sc.pp.pca(adata_omics1_high, n_comps=30)
      adata_omics1.obsm['feat'] = adata_omics1_high.obsm['X_pca'].copy()
    
      # Image
      adata_omics2 = adata_omics2[adata_omics1.obs_names].copy()
      adata_omics2.obsm['feat'] = adata_omics2.X.copy()
        
    elif datatype == 'XeniumPrime':  # Xenium data
      # RNA
      sc.pp.filter_cells(adata_omics1, min_counts=50)
      sc.pp.filter_genes(adata_omics1, min_cells=10)
    
      sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
      sc.pp.normalize_total(adata_omics1, target_sum=1e6)
      sc.pp.log1p(adata_omics1)
      adata_omics1.layers['logcounts'] = adata_omics1.X
      sc.pp.scale(adata_omics1)
      
      adata_omics1_high =  adata_omics1[:, adata_omics1.var['highly_variable']]
      sc.pp.pca(adata_omics1_high, n_comps=30)
      adata_omics1.obsm['feat'] = adata_omics1_high.obsm['X_pca'].copy()
    
      # Image
      adata_omics2 = adata_omics2[adata_omics1.obs_names].copy()
      adata_omics2.obsm['feat'] = adata_omics2.X.copy()
        
    elif datatype == 'VisiumHD':  # VisiumHD data
      # RNA
      sc.pp.filter_cells(adata_omics1, min_counts=50)
      sc.pp.filter_genes(adata_omics1, min_cells=10)
    
      print(adata_omics1.shape)
    
      sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
      sc.pp.normalize_total(adata_omics1, target_sum=1e6)
      sc.pp.log1p(adata_omics1)
      adata_omics1.layers['logcounts'] = adata_omics1.X
      sc.pp.scale(adata_omics1)
    
      gc.collect()
      
      adata_omics1_high =  adata_omics1[:, adata_omics1.var['highly_variable']]
      sc.pp.pca(adata_omics1_high, n_comps=40)
      adata_omics1.obsm['feat'] = adata_omics1_high.obsm['X_pca'].copy()
        
      gc.collect()
    
      # Image
      adata_omics2 = adata_omics2[adata_omics1.obs_names].copy()
      adata_omics2.obsm['feat'] = adata_omics2.X.copy()
        
      gc.collect()
              
    elif datatype == 'Stereo-CITE-seq':  
      # RNA
      sc.pp.filter_genes(adata_omics1, min_cells=10)
      sc.pp.filter_cells(adata_omics1, min_genes=80)

      sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
      sc.pp.normalize_total(adata_omics1, target_sum=1e6)
      sc.pp.log1p(adata_omics1)
      sc.pp.scale(adata_omics1)
      
      adata_omics1_high =  adata_omics1[:, adata_omics1.var['highly_variable']]
      sc.pp.pca(adata_omics1_high, n_comps=adata_omics2.n_vars-1)
      adata_omics1.obsm['feat'] = adata_omics1_high.obsm['X_pca']
      
      # Protein
      #sc.pp.filter_genes(adata_omics2, min_cells=50)
      
      adata_omics2 = adata_omics2[adata_omics1.obs_names].copy()
      adata_omics2 = clr_normalize_each_cell(adata_omics2)
      sc.pp.scale(adata_omics2)
      sc.pp.pca(adata_omics2, n_comps=adata_omics2.n_vars-1)
      adata_omics2.obsm['feat'] = adata_omics2.obsm['X_pca'].copy()
      
    elif datatype == 'Spatial-ATAC-RNA-seq':  
      # RNA
      sc.pp.filter_genes(adata_omics1, min_cells=10)
      sc.pp.filter_genes(adata_omics2, min_cells=10)
      sc.pp.filter_cells(adata_omics1, min_genes=200)
      
      adata_omics1.layers['counts'] = adata_omics1.X.copy()
      sc.pp.normalize_total(adata_omics1, target_sum=1e6)
      sc.pp.log1p(adata_omics1)
      sc.pp.scale(adata_omics1)
      sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000, layer="counts")
      
      adata_omics1_high =  adata_omics1[:, adata_omics1.var['highly_variable']]
      sc.pp.pca(adata_omics1_high, n_comps=50)  
      adata_omics1.obsm['feat'] = adata_omics1_high.obsm['X_pca']
      
      # ATAC
      adata_omics2 = adata_omics2[adata_omics1.obs_names].copy()
      sc.pp.highly_variable_genes(adata_omics2, flavor="seurat_v3", n_top_genes=3000)
      if 'X_lsi' not in adata_omics2.obsm:
          lsi(adata_omics2, n_components=50)
      adata_omics2.obsm['feat'] = adata_omics2.obsm['X_lsi'].copy()
    
    elif datatype == 'Multiplex':  
      # RNA
      sc.pp.filter_genes(adata_omics1, min_cells=10)
      sc.pp.filter_cells(adata_omics1, min_genes=80)
      sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
      sc.pp.normalize_total(adata_omics1, target_sum=1e6)
      sc.pp.log1p(adata_omics1)
      adata_omics1.layers['log1p'] = adata_omics1.X
      sc.pp.scale(adata_omics1)
      
      adata_omics1_high =  adata_omics1[:, adata_omics1.var['highly_variable']]
      sc.pp.pca(adata_omics1_high, n_comps=30)  
      adata_omics1.obsm['feat'] = adata_omics1_high.obsm['X_pca']
      
      # ATAC
      adata_omics2 = adata_omics2[adata_omics1.obs_names].copy()
      adata_omics2.obsm['feat'] = np.nan_to_num(adata_omics2.obsm['X_lsi'],nan=0).copy()
    
    if datatype == 'DLPFC':
      # RNA
      sc.pp.filter_genes(adata_omics1, min_cells=10)
      sc.pp.filter_cells(adata_omics1, min_genes=80)
    
      sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
      sc.pp.normalize_total(adata_omics1, target_sum=1e6)
      sc.pp.log1p(adata_omics1)
      sc.pp.scale(adata_omics1)

      adata_omics1_high =  adata_omics1[:, adata_omics1.var['highly_variable']]
      sc.pp.pca(adata_omics1_high, n_comps=adata_omics2.n_vars)
      adata_omics1.obsm['feat'] = adata_omics1_high.obsm['X_pca']
    
      # Image
      adata_omics2 = adata_omics2[adata_omics1.obs_names].copy()
      sc.pp.scale(adata_omics2)
      adata_omics2.obsm['feat'] = adata_omics2.X.copy()
    
    if datatype == 'MSI':
      # RNA
      sc.pp.filter_cells(adata_omics1, min_genes=10)
    
      sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
      sc.pp.normalize_total(adata_omics1, target_sum=1e6)
      sc.pp.log1p(adata_omics1)
      sc.pp.scale(adata_omics1)

      adata_omics1_high =  adata_omics1[:, adata_omics1.var['highly_variable']]
      sc.pp.pca(adata_omics1_high, n_comps=30)
      adata_omics1.obsm['feat'] = adata_omics1_high.obsm['X_pca']
    
      # MSI
      adata_omics2 = adata_omics2[adata_omics1.obs_names].copy()
      adata_omics2 = clr_normalize_each_cell(adata_omics2)
      sc.pp.scale(adata_omics2)
      sc.pp.pca(adata_omics2, n_comps=30)
      adata_omics2.obsm['feat'] = adata_omics2.obsm['X_pca'].copy()
    

    print("Constructing spatial graphs")
    # modality 1
    cell_position_omics1 = adata_omics1.obsm['spatial']
    adj_omics1 = build_network(cell_position_omics1, n_neighbors=n_neighbors)
    adata_omics1.uns['adj_spatial'] = adj_omics1
    
    gc.collect()
    
    # modality 2
    cell_position_omics2 = adata_omics2.obsm['spatial']
    adj_omics2 = build_network(cell_position_omics2, n_neighbors=n_neighbors)
    adata_omics2.uns['adj_spatial'] = adj_omics2
    
    gc.collect()
    
    print("Constructing feature graphs")
    feature_graph_omics1, feature_graph_omics2 = feature_graph(adata_omics1, adata_omics2, k=feat_neighbors)
    adata_omics1.obsm['adj_feature'], adata_omics2.obsm['adj_feature'] = feature_graph_omics1, feature_graph_omics2
    
    gc.collect()
    
    data = {'adata_omics1': adata_omics1, 'adata_omics2': adata_omics2}
    
    return data


def clr_normalize_each_cell(adata, inplace=True):
    import numpy as np
    import scipy

    def seurat_clr(x):
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)

    if not inplace:
        adata = adata.copy()
    
    adata.X = np.apply_along_axis(
        seurat_clr, 1, (adata.X.A if scipy.sparse.issparse(adata.X) else np.array(adata.X))
    )
    return adata     


def feature_graph(adata_omics1, adata_omics2, k=6, mode= "connectivity", metric="correlation", include_self=False):
    print("Feature graph 1")
    feature_graph_omics1=kneighbors_graph(adata_omics1.obsm['feat'], k, mode=mode, metric=metric, include_self=include_self, n_jobs=-1)
    print("Feature graph 1 finished")
    gc.collect()
    print("Feature graph 2")
    feature_graph_omics2=kneighbors_graph(adata_omics2.obsm['feat'], k, mode=mode, metric=metric, include_self=include_self, n_jobs=-1)
    print("Feature graph 2 finished")
    gc.collect()

    return feature_graph_omics1, feature_graph_omics2


def build_network(cell_position, n_neighbors=6):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(cell_position)  
    _ , indices = nbrs.kneighbors(cell_position)
    x = indices[:, 0].repeat(n_neighbors)
    y = indices[:, 1:].flatten()
    adj = pd.DataFrame(columns=['x', 'y', 'value'])
    adj['x'] = x
    adj['y'] = y
    adj['value'] = np.ones(x.size)
    return adj


def construct_graph(adjacent):
    n_spot = adjacent['x'].max() + 1
    adj = coo_matrix((adjacent['value'], (adjacent['x'], adjacent['y'])), shape=(n_spot, n_spot))
    return adj


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def construct_combined_graph(adata):
    adj_spatial_omics = construct_graph(adata.uns['adj_spatial']) 
    adj_feature_omics = adata.obsm['adj_feature']
    # combined graph
    combined_graph_1 = (adj_spatial_omics+adj_feature_omics).sign()
    adj_spatial_omics = None
    adj_feature_omics = None
    gc.collect()
    # normalize the combined graph
    combined_graph_1 = combined_graph_1 + combined_graph_1.transpose()
    indices = combined_graph_1.data > 1
    combined_graph_1.data[indices] = 1
    combined_graph_1 = normalize_adj(combined_graph_1)
    combined_graph_1_sp = nx.from_scipy_sparse_array(sparse.coo_matrix((combined_graph_1.coalesce().values().numpy(), combined_graph_1.coalesce().indices().numpy()), shape=combined_graph_1.shape))
    gc.collect()
    
    # return combined graph and sparse graph
    return combined_graph_1, combined_graph_1_sp

def construct_ablation_graphs(adata1, adata2):
    adj_spatial_omics1 = construct_graph(adata1.uns['adj_spatial']) 
    adj_spatial_omics2 = construct_graph(adata2.uns['adj_spatial'])
    adj_feature_omics1 = adata1.obsm['adj_feature']
    adj_feature_omics2 = adata2.obsm['adj_feature']
        
    # combined graph
    combined_graph_1 = (adj_feature_omics1).sign()
    adj_spatial_omics1 = None
    adj_feature_omics1 = None
    gc.collect()

    combined_graph_2 = (adj_feature_omics2).sign()
    adj_spatial_omics2 = None 
    adj_feature_omics2 = None
    gc.collect()

    # normalize the combined graph
    combined_graph_1 = combined_graph_1 + combined_graph_1.transpose()
    indices = combined_graph_1.data > 1
    combined_graph_1.data[indices] = 1
    combined_graph_1 = normalize_adj(combined_graph_1)
    #final_combined_graph_1 = combined_graph_1.to_dense()
    combined_graph_1_sp = nx.from_scipy_sparse_array(sparse.coo_matrix((combined_graph_1.coalesce().values().numpy(), combined_graph_1.coalesce().indices().numpy()), shape=combined_graph_1.shape))
    #combined_graph_1 = None
    gc.collect()

    combined_graph_2 = combined_graph_2 + combined_graph_2.transpose()
    indices = combined_graph_2.data > 1 
    combined_graph_2.data[indices] = 1
    combined_graph_2 = normalize_adj(combined_graph_2)
    #final_combined_graph_2 = combined_graph_2.to_dense()
    combined_graph_2_sp = nx.from_scipy_sparse_array(sparse.coo_matrix((combined_graph_2.coalesce().values().numpy(), combined_graph_2.coalesce().indices().numpy()), shape=combined_graph_2.shape))
    #combined_graph_2 = None
    gc.collect()
    
    data = {'final_combined_graph_1': combined_graph_1, 'final_combined_graph_2': combined_graph_2, 'combined_graph_1_sp': combined_graph_1_sp, 'combined_graph_2_sp': combined_graph_2_sp}
    
    return data


def construct_combined_graphs(adata1, adata2):
    adj_spatial_omics1 = construct_graph(adata1.uns['adj_spatial']) 
    adj_spatial_omics2 = construct_graph(adata2.uns['adj_spatial'])
    adj_feature_omics1 = adata1.obsm['adj_feature']
    adj_feature_omics2 = adata2.obsm['adj_feature']
        
    # combined graph
    combined_graph_1 = (adj_spatial_omics1+adj_feature_omics1).sign()
    adj_spatial_omics1 = None
    adj_feature_omics1 = None
    gc.collect()

    combined_graph_2 = (adj_spatial_omics2+adj_feature_omics2).sign()
    adj_spatial_omics2 = None 
    adj_feature_omics2 = None
    gc.collect()

    # normalize the combined graph
    combined_graph_1 = combined_graph_1 + combined_graph_1.transpose()
    indices = combined_graph_1.data > 1
    combined_graph_1.data[indices] = 1
    combined_graph_1 = normalize_adj(combined_graph_1)
    #final_combined_graph_1 = combined_graph_1.to_dense()
    combined_graph_1_sp = nx.from_scipy_sparse_array(sparse.coo_matrix((combined_graph_1.coalesce().values().numpy(), combined_graph_1.coalesce().indices().numpy()), shape=combined_graph_1.shape))
    #combined_graph_1 = None
    gc.collect()

    combined_graph_2 = combined_graph_2 + combined_graph_2.transpose()
    indices = combined_graph_2.data > 1 
    combined_graph_2.data[indices] = 1
    combined_graph_2 = normalize_adj(combined_graph_2)
    #final_combined_graph_2 = combined_graph_2.to_dense()
    combined_graph_2_sp = nx.from_scipy_sparse_array(sparse.coo_matrix((combined_graph_2.coalesce().values().numpy(), combined_graph_2.coalesce().indices().numpy()), shape=combined_graph_2.shape))
    #combined_graph_2 = None
    gc.collect()
    
    data = {'final_combined_graph_1': combined_graph_1, 'final_combined_graph_2': combined_graph_2, 'combined_graph_1_sp': combined_graph_1_sp, 'combined_graph_2_sp': combined_graph_2_sp}
    
    return data

    
def construct_combined_multisample_graphs(adata1_rep1, adata2_rep1, adata1_rep2, adata2_rep2):
    
    adj_spatial_omics1_rep1 = construct_graph(adata1_rep1.uns['adj_spatial'].copy())
    adj_spatial_omics2_rep1 = construct_graph(adata2_rep1.uns['adj_spatial'].copy())
    adj_feature_omics1_rep1 = adata1_rep1.obsm['adj_feature'].copy() 
    adj_feature_omics2_rep1 = adata2_rep1.obsm['adj_feature'].copy()
    adj_spatial_omics1_rep2 = construct_graph(adata1_rep2.uns['adj_spatial'].copy())
    adj_spatial_omics2_rep2 = construct_graph(adata2_rep2.uns['adj_spatial'].copy())
    adj_feature_omics1_rep2 = adata1_rep2.obsm['adj_feature'].copy() 
    adj_feature_omics2_rep2 = adata2_rep2.obsm['adj_feature'].copy()
    gc.collect()


    adj_spatial_omics1 = sparse.block_diag((adj_spatial_omics1_rep1, adj_spatial_omics1_rep2))
    adj_spatial_omics1_rep1 = None
    adj_spatial_omics1_rep2 = None
    adj_spatial_omics2 = sparse.block_diag((adj_spatial_omics2_rep1, adj_spatial_omics2_rep2))
    adj_spatial_omics2_rep1 = None
    adj_spatial_omics2_rep2 = None
    adj_feature_omics1 = sparse.block_diag((adj_feature_omics1_rep1, adj_feature_omics1_rep2))
    adj_feature_omics1_rep1 = None
    adj_feature_omics1_rep2 = None
    adj_feature_omics2 = sparse.block_diag((adj_feature_omics2_rep1, adj_feature_omics2_rep2))
    adj_feature_omics2_rep1 = None
    adj_feature_omics2_rep2 = None
    gc.collect()


    # combined graph
    combined_graph_1 = (adj_spatial_omics1+adj_feature_omics1).sign()
    adj_spatial_omics1 = None
    adj_feature_omics1 = None
    gc.collect()

    combined_graph_2 = (adj_spatial_omics2+adj_feature_omics2).sign()
    adj_spatial_omics2 = None 
    adj_feature_omics2 = None
    gc.collect()


    # normalize the combined graph
    combined_graph_1 = combined_graph_1 + combined_graph_1.transpose()
    indices = combined_graph_1.data > 1
    combined_graph_1.data[indices] = 1
    combined_graph_1 = normalize_adj(combined_graph_1)
    #final_combined_graph_1 = combined_graph_1.to_dense()
    combined_graph_1_sp = nx.from_scipy_sparse_array(sparse.coo_matrix((combined_graph_1.coalesce().values().numpy(), combined_graph_1.coalesce().indices().numpy()), shape=combined_graph_1.shape))
    #combined_graph_1 = None
    gc.collect()

    combined_graph_2 = combined_graph_2 + combined_graph_2.transpose()
    indices = combined_graph_2.data > 1 
    combined_graph_2.data[indices] = 1
    combined_graph_2 = normalize_adj(combined_graph_2)
    #final_combined_graph_2 = combined_graph_2.to_dense()
    combined_graph_2_sp = nx.from_scipy_sparse_array(sparse.coo_matrix((combined_graph_2.coalesce().values().numpy(), combined_graph_2.coalesce().indices().numpy()), shape=combined_graph_2.shape))
    #combined_graph_2 = None
    gc.collect()
    
    data = {'final_combined_graph_1': combined_graph_1, 'final_combined_graph_2': combined_graph_2, 'combined_graph_1_sp': combined_graph_1_sp, 'combined_graph_2_sp': combined_graph_2_sp}
    
    return data



