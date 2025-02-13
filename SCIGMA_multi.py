import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from models_multi import Model
from data import construct_combined_graph
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


class SCIGMA_multi:
    def __init__(self, 
        data,
        seed_num,
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        inference_device = torch.device('cpu'),
        learning_rate=1e-3,
        weight_decay=1e-2,
        clr_weight=0.3,
        contrastive_weights = {},
        recon_weights = {},
        connectivities=None,
        epochs=1000, 
        dim_output=30,
        batch_size = 6000,
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
        self.adatas = data
        self.device = device
        self.inference_device = inference_device
        self.learning_rate=learning_rate
        self.weight_decay=weight_decay
        self.epochs=epochs
        self.dim_output = dim_output
        self.contrastive_weights = contrastive_weights
        self.recon_weights = recon_weights
        self.batch_size = batch_size
        

        self.num_samples = self.adatas[list(self.adatas.keys())[0]].shape[0]
        
        # feats
        self.features = {}
        for key in self.adatas:
            self.features[key] = torch.FloatTensor(self.adatas[key].obsm['feat'].copy()).to(self.device)
        
        # connectivities
        self.connectivities = [i for i in range(self.num_samples)] 
        
        # adj
        gc.collect()
        print("Creating adjacency matrices")
        self.combined_graphs = {}
        self.combined_graphs_sp = {}
        for key in self.adatas:
            combined_graph, combined_graph_sp = construct_combined_graph(self.adatas[key])
            self.combined_graphs[key] = combined_graph
            self.combined_graphs_sp[key] = combined_graph_sp
            
        # weights
        if len(self.recon_weights.keys()) == 0:
            for key in self.adatas:
                self.recon_weights[key] = 1.0
        if len(self.contrastive_weights.keys()) == 0:
            for key in self.adatas:
                self.contrastive_weights[key] = 1.0
    
        
        # dimension of input feature
        self.dim_ins = {}
        self.dim_outs = {}
        for key in self.features:
            self.dim_ins[key] = self.features[key].shape[1]
            self.dim_outs[key] = self.dim_output
 
        
        # contrastive loss
        self.criterions = {}
        for key in self.adatas:
            self.criterions[key] = CrossCLR_onlyIntraModality(temperature=0.03, negative_weight=clr_weight, device=self.device)  
        
        # uncertainty model
        self.dum_model = None
        
        # model + optim
        self.model = Model(dim_ins = self.dim_ins, dim_hid = self.dim_outs, dim_outs = self.dim_outs, layer='GAT', device=self.device).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), self.learning_rate, 
                                          weight_decay=self.weight_decay)
            
        print("Model ready for training!")
        
    
        
    def train(self, epochs):
        sampler = ForestFire()
        for epoch in tqdm(range(epochs)): 
            ##### get batch ####     
            to_sample = epoch%len(self.adatas)
            to_sample = list(self.adatas.keys())[to_sample]
            sample = sampler.forestfire(self.combined_graphs_sp[to_sample], self.batch_size)
            indices = torch.tensor(list(set([n for n in sample]).intersection(self.connectivities)))
            edges = {}
            feats = {}
            for key in self.features:
                edges[key] = torch.index_select(torch.index_select(self.combined_graphs[key], 0, indices), 1, indices).coalesce()
                feats[key] = self.features[key][indices]     

            sample = None

            self.model.train()
            
            
            results = self.model(feats, edges)
            # get log temperature (numerical stability)
            invtau = self.model.get_temp(results['emb_latent_combined']) 
            tau = torch.exp(invtau)  
            
            loss = 0

            # reconstruction loss
            for key in results['emb_recon']:
                recon_loss = self.recon_weights[key] * F.mse_loss(results['emb_recon'][key], feats[key])
                loss += recon_loss
                if (epoch%50) == 0:
                    print(f"{key} recon loss: {recon_loss.item()}")
   
            # contrastive loss
            for key in results['emb_latents']:
                contrast_loss = self.contrastive_weights[key] * self.criterions[key](results['emb_latent_combined'], results['emb_latents'][key], tau=tau)
                loss += contrast_loss
                if (epoch%50) == 0:
                    print(f"{key} contrastive loss: {contrast_loss.item()}")

            if (epoch%50) == 0:
                print(loss.item())
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
        for param in self.model.parameters():
            param.requires_grad=False
            param.data = param.to(self.inference_device)
        for key in self.model.encoders:
            self.model.encoders[key].requires_grad = False
            self.model.encoders[key].data = self.model.encoders[key].to(self.inference_device)
            self.model.decoders[key].requires_grad = False
            self.model.decoders[key].data = self.model.decoders[key].to(self.inference_device)
        
        edges = {}
        feats = {}
        for key in self.features:
            feats[key] = self.features[key].detach().cpu()
            edges[key] = self.combined_graphs[key].coalesce().detach().cpu()
            
        with torch.no_grad():
            results = self.model(feats, edges)
            invtau = self.model.get_temp(results['emb_latent_combined']) 
            tau = torch.exp(invtau)
        edges = None
        feats = None
        
        embs = {}
        recon = {}
        emb_combined = results['emb_latent_combined'].detach().cpu().numpy()
        for key in self.adatas:
            embs[key] = F.normalize(results['emb_latents'][key], p=2, eps=1e-12, dim=1).detach().cpu().numpy()
            recon[key] = results['emb_recon'][key].detach().cpu().numpy()
        
        output = {'emb_latent': embs,
                  'emb_latent_combined': emb_combined,
                  'emb_recon': recon,
                  'tau': tau.detach().cpu().numpy(),
                  'invtau': invtau.detach().cpu().numpy(),
                  'attention': results['attention'].detach().cpu().numpy(),
                 }
        
        return output
    

