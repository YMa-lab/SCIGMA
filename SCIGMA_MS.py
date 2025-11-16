import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from models import Model
from data import construct_combined_multisample_graphs
from losses import CrossCLR_onlyIntraModality
from scipy import sparse
from graph_sampling import ForestFire

eps = 1e-8


def connectivity(data, gamma):
    '''
    Calculates connectivity value for each sample
    
    Returns a list of indices of noninfluential samples
    '''
    mask = (sparse.eye(data.shape[0])*-1).data + 1
    prox_vid = (data@data.transpose()).multiply(mask).mean(axis=1)
    scores_v = prox_vid.data / prox_vid.max()
    scores_v = scores_v.todense()
    connectivities = []
    i = 0
    for x in scores_v:
        if x < gamma:
            connectivities.append(i)
        i += 1

    return connectivities


class SCIGMA:
    def __init__(self, 
        data1,
        data2, 
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        learning_rate=1e-4,
        weight_decay=1e-4,
        clr_weight=0.3,
        contrastive_weight1=0.2,
        contrastive_weight2=0.2,
        gamma=0.1,
        wass_weight=1,
        connectivities=None,
        epochs=1000, 
        dim_output=64,
        batch_size = 6000,
        ):
        '''\

        Parameters
        ----------
        data : dict
            dict object of spatial multi-omics data.
        datatype : string, optional
            Data type of input, Our current model supports 'SPOTS', 'Stereo-CITE-seq', and 'Spatial-ATAC-RNA-seq'. We plan to extend our model for more data types in the future.  
            The default is 'SPOTS'.
        device : string, optional
            Using GPU or CPU? The default is 'cpu'.  
        learning_rate : float, optional
            Learning rate for ST representation learning. The default is 0.001.
        weight_decay : float, optional
            Weight factor to control the influence of weight parameters. The default is 0.0001.
        epochs : int, optional
            Epoch for model training. The default is 1000.
        dim_output : int, optional
            Dimension of latent representation. The default is 64.
    
        Returns
        -------
        The learned representation 'self.emb_combined'.

        '''
        self.data1 = data1.copy()
        self.data2 = data2.copy()
        self.device = device
        self.learning_rate=learning_rate
        self.weight_decay=weight_decay
        self.epochs=epochs
        self.dim_output = dim_output
        self.contrastive_weight1 = contrastive_weight1
        self.contrastive_weight2 = contrastive_weight2
        self.batch_size = batch_size
        self.wass_weight = wass_weight

        self.adata_omics1_rep1 = self.data1['adata_omics1']
        self.adata_omics2_rep1 = self.data1['adata_omics2']
        self.adata_omics1_rep2 = self.data2['adata_omics1']
        self.adata_omics2_rep2 = self.data2['adata_omics2']
        
        
        # feature
        
        features_omics1_rep1 = torch.FloatTensor(self.adata_omics1_rep1.obsm['feat'].copy()).to(self.device)
        features_omics2_rep1 = torch.FloatTensor(self.adata_omics2_rep1.obsm['feat'].copy()).to(self.device)
        features_omics1_rep2 = torch.FloatTensor(self.adata_omics1_rep2.obsm['feat'].copy()).to(self.device)
        features_omics2_rep2 = torch.FloatTensor(self.adata_omics2_rep2.obsm['feat'].copy()).to(self.device)
        
        self.features_omics1 = torch.cat((features_omics1_rep1, features_omics1_rep2),0)
        self.features_omics2 = torch.cat((features_omics2_rep1, features_omics2_rep2),0)
        if not connectivities:
            connectivities = [i for i in range(len(self.features_omics1))] 
        self.connectivities = set(connectivities)
        
        
        # adj
        print("Creating adjacency matrices")
        adj_data = construct_combined_multisample_graphs(self.adata_omics1_rep1, self.adata_omics2_rep1, self.adata_omics1_rep2, self.adata_omics2_rep2)
        self.combined_graph_1 = adj_data['final_combined_graph_1']
        self.combined_graph_2 = adj_data['final_combined_graph_2']
        self.combined_graph_1_sp = adj_data['combined_graph_1_sp']
        self.combined_graph_2_sp = adj_data['combined_graph_2_sp']
        
    
        self.n_cell_omics1 = self.adata_omics1_rep1.n_obs + self.adata_omics1_rep1.n_obs
        self.n_cell_omics2 = self.adata_omics2_rep1.n_obs + self.adata_omics2_rep2.n_obs
        
        # dimension of input feature
        self.dim_input1 = self.features_omics1.shape[1]
        self.dim_input2 = self.features_omics2.shape[1]
        self.dim_output1 = self.dim_output
        self.dim_output2 = self.dim_output
        
        # contrastive loss
        self.criterion = CrossCLR_onlyIntraModality(temperature=0.03, negative_weight=clr_weight)
        self.criterion2 = CrossCLR_onlyIntraModality(temperature=0.03, negative_weight=clr_weight) 
        
            
        print("Model ready for training!")

        
    def train(self, epochs):
        self.model = Model(self.dim_input1, self.dim_output1, self.dim_input2, self.dim_output2, layer='GCAT').to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate, 
                                          weight_decay=self.weight_decay)

        sampler = ForestFire()

        for epoch in tqdm(range(epochs)): 
            ##### get batch ####  
            if epoch%2 == 0:
                sample = sampler.forestfire(self.combined_graph_1_sp, self.batch_size)
                indices = torch.tensor(list(set([n for n in sample]).intersection(self.connectivities)))
                edge1 = torch.index_select(torch.index_select(self.combined_graph_1, 0, indices), 1, indices).coalesce()
                feat1 = self.features_omics1[indices]
                edge2 = torch.index_select(torch.index_select(self.combined_graph_2, 0, indices), 1, indices).coalesce()
                feat2 = self.features_omics2[indices]     
            else:
                sample = sampler.forestfire(self.combined_graph_2_sp, self.batch_size)
                indices = torch.tensor(list(set([n for n in sample]).intersection(self.connectivities)))
                edge2 = torch.index_select(torch.index_select(self.combined_graph_2, 0, indices), 1, indices).coalesce()
                feat2 = self.features_omics2[indices]
                edge1 = torch.index_select(torch.index_select(self.combined_graph_1, 0, indices), 1, indices).coalesce()
                feat1 = self.features_omics1[indices] 
            sample = None 
            
            self.model.train()
            results = self.model(feat1, feat2, edge1, edge2)
          
            # reconstruction loss
            loss_recon_omics1 = F.mse_loss(feat1, results['emb_recon_omics1'])
            loss_recon_omics2 = F.mse_loss(feat2, results['emb_recon_omics2'])
            
                            
            loss_contrast_cross_mod =  self.contrastive_weight1*self.criterion(results['emb_latent_omics1'], results['emb_latent_combined']) +  self.contrastive_weight2*self.criterion2(results['emb_latent_omics2'], results['emb_latent_combined'])
            
            # full loss function
            loss = loss_recon_omics1 + loss_recon_omics2 + loss_contrast_cross_mod
 
            if (epoch%50) == 0:
                print(loss.item(), loss_recon_omics1.item(), loss_recon_omics2.item(), loss_contrast_cross_mod.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            torch.cuda.empty_cache()
            
            feat1 = None
            feat2 = None
            edge1 = None
            edge2 = None
        
        print("Model training finished!\n")  
    
        with torch.no_grad():
            self.model.eval()
              
            results = self.model(self.features_omics1, self.features_omics2, self.combined_graph_1.coalesce(), self.combined_graph_2.coalesce())
            self.combined_graph_1 = None
            self.combined_graph_2 = None
            
        emb_omics1 = F.normalize(results['emb_latent_omics1'], p=2, eps=1e-12, dim=1)  
        emb_omics2 = F.normalize(results['emb_latent_omics2'], p=2, eps=1e-12, dim=1)
        emb_combined = F.normalize(results['emb_latent_combined'], p=2, eps=1e-12, dim=1)
        
        output = {'emb_latent_omics1': emb_omics1.detach().cpu().numpy(),
                  'emb_latent_omics2': emb_omics2.detach().cpu().numpy(),
                  'emb_latent_combined': emb_combined.detach().cpu().numpy(),
                 }
        
        return output
    

    
    
      

    
        
    
    
