import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from models import Model
from data import construct_combined_graphs, construct_ablation_graphs
from losses import CrossCLR_onlyIntraModality, MaxMargin_coot
from graph_sampling import ForestFire
import gc
from utils import seed

def connectivity(data, gamma):
    '''
    Calculates connectivity value for each sample
    
    Returns a list of indices of noninfluential samples
    '''
    mask = 1-(np.eye(data.shape[0]))
    prox_vid = np.mean((data@data.T)*mask, axis=1)
    scores_v = prox_vid / np.max(prox_vid)
    connectivities = []
    i = 0
    for x in scores_v:
        if x < gamma:
            connectivities.append(i)
        i += 1

    return connectivities

def weighted_mse(pred, target, weights):
    return (torch.square(pred-target)*weights).mean()


class SCIGMA:
    def __init__(self, 
        data,
        seed_num,
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        inference_device = torch.device('cpu'),
        learning_rate=1e-4,
        weight_decay=1e-2,
        clr_weight=0.3,
        contrastive_weight1=0.1,
        contrastive_weight2=0.1,
        recon_weight1 = 1,
        recon_weight2 = 1,
        connectivities=None,
        epochs=1000, 
        dim_output=64,
        batch_size = 6000,
        downweight = 100,
        architecture = 'GAT',
        ):
        '''\

        Parameters
        ----------
        data : dict
            dict object of spatial multi-omics data.
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
        seed(seed_num)
        self.data = data
        self.device = device
        self.inference_device = inference_device
        self.learning_rate=learning_rate
        self.weight_decay=weight_decay
        self.epochs=epochs
        self.dim_output = dim_output
        self.contrastive_weight1 = contrastive_weight1
        self.contrastive_weight2 = contrastive_weight2
        self.recon_weight1 = recon_weight1
        self.recon_weight2 = recon_weight2 
        self.batch_size = batch_size
        self.downweight = downweight

        self.adata_omics1 = self.data['adata_omics1']
        self.adata_omics2 = self.data['adata_omics2']
        
        # feature
        if not connectivities:
            connectivities = [i for i in range(len(self.adata_omics1.obsm['feat']))] 
        self.connectivities = set(connectivities)
        self.features_omics1 = torch.FloatTensor(self.adata_omics1.obsm['feat'].copy()).to(self.device)
        self.features_omics2 = torch.FloatTensor(self.adata_omics2.obsm['feat'].copy()).to(self.device)
        print(f"Num samples pruned: {len(self.adata_omics1.obsm['feat'])-len(connectivities)}")
        
        # adj
        gc.collect()
        print("Creating adjacency matrices")
        adj_data = construct_combined_graphs(self.adata_omics1, self.adata_omics2)
        self.combined_graph_1 = adj_data['final_combined_graph_1']
        self.combined_graph_2 = adj_data['final_combined_graph_2']
        self.combined_graph_1_sp = adj_data['combined_graph_1_sp']
        self.combined_graph_2_sp = adj_data['combined_graph_2_sp']
    
        self.n_cell_omics1 = self.adata_omics1.n_obs
        self.n_cell_omics2 = self.adata_omics2.n_obs
        
        # dimension of input feature
        self.dim_input1 = self.features_omics1.shape[1]
        self.dim_input2 = self.features_omics2.shape[1]
        self.dim_output1 = self.dim_output
        self.dim_output2 = self.dim_output
        
        # contrastive loss
        self.criterion = CrossCLR_onlyIntraModality(temperature=0.03, negative_weight=clr_weight, device=self.device)  
        self.criterion2 = CrossCLR_onlyIntraModality(temperature=0.03, negative_weight=clr_weight, device=self.device) 
        
        
        # model + optim
        self.model = Model(self.dim_input1, self.dim_output1, self.dim_input2, self.dim_output2, uncertainty_size=self.features_omics1.shape[0], layer=architecture, device=self.device).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), self.learning_rate, 
                                          weight_decay=self.weight_decay)
            
        print("Model ready for training!")
        
    
        
    def train(self, epochs):
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
            # get log temperature (numerical stability)
            invtau = self.model.get_temp(results['emb_latent_combined']) 
            tau = torch.exp(invtau)     

            # reconstruction loss
            loss_recon_omics1 = self.recon_weight1*F.mse_loss(results['emb_recon_omics1'], feat1)
            loss_recon_omics2 = self.recon_weight2*F.mse_loss(results['emb_recon_omics2'], feat2)

            # contrastive loss
            loss_contrast_mod1 = self.criterion(results['emb_latent_combined'], results['emb_latent_omics1'], tau=tau)

            loss_contrast_mod2 = self.criterion2(results['emb_latent_combined'], results['emb_latent_omics2'], tau=tau)


            # full loss function
            loss = loss_recon_omics1 + loss_recon_omics2 + self.contrastive_weight1*loss_contrast_mod1 + self.contrastive_weight2*loss_contrast_mod2
            if (epoch%50) == 0:
                print(loss.item(), loss_recon_omics1.item(), loss_recon_omics2.item(), loss_contrast_mod1.item(), loss_contrast_mod2.item())
                print(tau)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            torch.cuda.empty_cache()
            gc.collect()

            feat1 = None
            feat2 = None
            edge1 = None
            edge2 = None
        
        print("Model training finished!\n") 
        torch.cuda.empty_cache()
    

        self.model.eval()
        self.optimizer.zero_grad()
        # use cpu as device for inference to bypass memory issues
        self.model.device = self.inference_device
        gc.collect()
        for param in self.model.parameters():
            param.requires_grad=False
            param.data = param.to(self.inference_device)
        with torch.no_grad():
            results = self.model(self.features_omics1.detach().cpu(), self.features_omics2.detach().cpu(), self.combined_graph_1.coalesce().detach().cpu(), self.combined_graph_2.coalesce().detach().cpu())
            invtau = self.model.get_temp(results['emb_latent_combined']) 
            tau = torch.exp(invtau)
        gc.collect()
 
        emb_omics1 = F.normalize(results['emb_latent_omics1'], p=2, eps=1e-12, dim=1)  
        emb_omics2 = F.normalize(results['emb_latent_omics2'], p=2, eps=1e-12, dim=1)
        emb_combined = F.normalize(results['emb_latent_combined'], p=2, eps=1e-12, dim=1)
        emb_recon_omics1 = results['emb_recon_omics1']
        emb_recon_omics2 = results['emb_recon_omics2']
        
        output = {'emb_latent_omics1': emb_omics1.detach().cpu().numpy(),
                  'emb_latent_omics2': emb_omics2.detach().cpu().numpy(),
                  'emb_latent_combined': emb_combined.detach().cpu().numpy(),
                  'emb_recon_omics1': emb_recon_omics1.detach().cpu().numpy(),
                  'emb_recon_omics2': emb_recon_omics2.detach().cpu().numpy(),
                  'tau': tau.detach().cpu().numpy(),
                  'invtau': invtau.detach().cpu().numpy(),
                  'attention': results['attention_weights'].detach().cpu().numpy(),
                 }
        
        return output
    

