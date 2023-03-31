"""

Network training module
-- details on the loss function
-- data loaders

"""

import time
import random
import numpy as np
import torch
import pandas as pd
import scanpy as sc
from progress.bar import Bar
import torch.nn.modules.loss
import torch.nn.functional as F
from progress.bar import Bar
from scDREAMER_model import VAE, Batch_classifier, Discriminator
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torch.distributions import kl_divergence as kl
from torch.distributions import Normal

# Data Loader
class AnndataLoader1(Dataset):
    def __init__(self, adata, batch, cell_type):
        
        self.batch = batch  
        self.cell_type = cell_type
        self.Ann = adata
        
    def __len__(self):
        return len(self.Ann)
        
    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        x = self.Ann.X[idx, :]
        batch_encoded = self.Ann.obsm[self.batch + '_encoded'][idx, :]
        
        sample = {'X' : x, 
                  self.batch + '_encoded' : batch_encoded} 
        
        return sample

def cap(inp):
    
    capL = torch.ones_like(inp)*1e-4
    capU = torch.ones_like(inp)*1e4
    
    return torch.minimum(torch.maximum(inp, capL), capU)


def zinb(x, mu, theta, pi, eps = 1e-8):
    
    """
    
    Note: All inputs are torch Tensors
    log likelihood (scalar) of a minibatch according to a zinb model.
    Notes: We parametrize the bernoulli using the logits, hence the softplus functions appearing

    Variables:
    mu: mean of the negative binomial (has to be positive support) (shape: minibatch x genes)
    theta: inverse dispersion parameter (has to be positive support) (shape: minibatch x genes)
    pi: logit of the dropout parameter (real support) (shape: minibatch x genes)
    eps: numerical stability constant
    
    """

    # theta is the dispersion rate. If .ndimension() == 1, it is shared for all cells (regardless of batch or labels)
   
    if theta.ndimension() == 1:
        theta = theta.view(
            1, theta.size(0)
        )  # In this case, we reshape theta for broadcasting
    
    
    softplus_pi = F.softplus(-pi)  #  uses log(sigmoid(x)) = -softplus(-x)
    log_theta_eps = torch.log(cap(theta + eps))

    log_theta_mu_eps = torch.log(cap(theta + mu + eps))
    pi_theta_log = -pi + theta * (log_theta_eps - log_theta_mu_eps)

    case_zero = F.softplus(pi_theta_log) - softplus_pi
    mul_case_zero = torch.mul((x < eps).type(torch.float32), case_zero)

    case_non_zero = (
        -softplus_pi
        + pi_theta_log
        + x * (torch.log(cap(mu + eps)) - log_theta_mu_eps)
        + torch.lgamma(cap(x + theta))
        - torch.lgamma(cap(theta))
        - torch.lgamma(cap(x + 1))
    )
    
    mul_case_non_zero = torch.mul((x > eps).type(torch.float32), case_non_zero)

    res = mul_case_zero + mul_case_non_zero

    #res = torch.sum(res) # TODO: returns sum of each row of the 2D tensor
    
    return torch.mean(res.sum(-1))

def kl_div_l(mu1, var1, mu2, var2):
    
    """
    l_post = torch.normal(mu1, var1)
    l_prior = torch.normal(mu2, var2)
    
    kl_loss = torch.nn.functional.kl_div(l_post, l_prior) 
    # mu2, var2 = prior
    """
    
    kl_loss = kl(Normal(mu1, torch.sqrt(var1)), Normal(mu2, torch.sqrt(var2))).sum(dim = 1)

    #kl_loss = 2*torch.log(cap(var2/var1)) + torch.square(var1/var2) + torch.square((mu1 - mu2)/var2) - 1
    #return 0.5 * torch.mean(torch.sum(kl_loss))
    
    return torch.mean(kl_loss)

def kl_div(mu, var, n_obs):
    
    """
    Computes KL divergence between posterior sample ~ N(mu, sigma) and prior sample ~ N(0, 1) 
    """    
    #kl_loss = torch.nn.functional.kl_div(posterior, prior)       
    #kl_loss = -0.5 * torch.mean(torch.sum(1 - torch.pow(mu, 2) - torch.pow(var, 2) + 2 * torch.log(cap(var))))
    # return kl_loss
    
    kl_loss = kl(Normal(mu, torch.sqrt(var)), Normal(0, 1)).sum(dim = 1)
    
    return torch.mean(kl_loss)
    
        
def Bhattacharyya_loss(p, q):
    
    """
    Discriminator maximizes the Bhattachryya loss i.e. B.D (x, x')
    i.e. minimizes -ve Bhattachryya loss i.e. log (B.C.)
    assert: tmp >= 0, square root of -ve = nan; log(0) - undefined.
    """
    p[p < 0] *= -1
    q[q < 0] *= -1
    
    tmp = torch.sum(torch.mul(p, q), 1)
    b_loss = torch.log(cap(torch.sum(torch.sqrt(tmp)))) 
    
    return 0.1*b_loss
    
    
def crossentropy_loss(predict_labels, label):
    
    """
    Classifier network such as batch-classifier and 
    cell-type classifier network are trained by minimizing cross-entropy loss
    assert: tensorflow automatically takes softmax
    """
    
    loss = torch.nn.CrossEntropyLoss()
    c_loss = loss(predict_labels, label)
    
    return c_loss
    
    
        
class scDREAMER_Train:
    
    def __init__(self, adata, params):
        

#         self.device = params.device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.batch_size = params.batch_size
        self.epochs = params.epochs
        self.batch = params.batch
        self.cell_type = params.cell_type
        self.z_dim = params.z_dim
        self.X_dim = params.X_dim
        self.lr = params.lr
        self.name = params.name
        self.num_batches = len(adata.obs[params.batch].unique())
        
        #self.data = torch.FloatTensor(data.copy()).to(self.device)
        
        self.adata = adata
        self.total_size = len(self.adata)
        self.num_minibatches = int(self.total_size/self.batch_size)        
        params.num_batches = self.num_batches
        self.params = params
        
        self.vae = VAE(self.params).to(self.device)
#         self.vae.to(self.device)
        
        print("*"*30)
        print(self.device)
#         print('vae.device: {}'.format(self.vae.device))
        
        self.batch_classifier = Batch_classifier(self.params).to(self.device)
        self.discriminator = Discriminator(self.params).to(self.device)
        self.kl_scale = 0.001 # 0.001 (actual) earlier...0.01 tried so far

        lib_real = torch.log(torch.sum(torch.FloatTensor(self.adata.X), axis = 1))
        self.lib_mu = torch.mean(lib_real)
        self.lib_var = torch.var(lib_real)
                
        # Sample from Gaussian N(0, 1): Retuns a normal dist with mea 0 and variance 1        
        #self.real_dis = np.random.randn(self.batch_size, self.z_dim).to(self.device)
        self.real_dis = torch.randn_like(torch.zeros(self.batch_size, self.z_dim)).to(self.device)
        self.real_dis_logit = torch.randn_like(torch.zeros(self.batch_size, self.X_dim)).to(self.device)
        
        #betas = (0.5, 0.99)
        parameters = filter(lambda p: p.requires_grad, self.vae.parameters())
        
        self.vae_optimizer = torch.optim.Adam(params = parameters, lr = 0.0002, betas = (0.5, 0.99))
#         self.vae_adv_optimizer = torch.optim.Adam(params = list(self.vae.parameters()), lr = 0.0002, betas = (0.5, 0.99))
        self.vae_adv_optimizer = torch.optim.Adam(params = list(self.vae.z_encoder.parameters()), lr = 0.0002, betas = (0.5, 0.99))
        self.bc_optimizer = torch.optim.Adam(params = list(self.batch_classifier.parameters()), lr = self.params.lr, betas = (0.5, 0.99))
        self.dis_optimizer = torch.optim.Adam(params = list(self.discriminator.parameters()), lr = self.params.lr, betas = (0.5, 0.99))
        
    def train_vae(self, epoch, flag):
        
        ann_dataset = AnndataLoader1(self.adata, self.batch, self.cell_type)
        dataloader = DataLoader(ann_dataset, batch_size = self.batch_size, shuffle = True, num_workers = 4, drop_last = True)

        auto_encoder_loss_sum = 0
        auto_encoder_loss_sum_ = 0
        c_loss_sum = 0
        b_loss_sum = 0
        n_batches = 0
        
        for i_batch, sample_batched in enumerate(dataloader):
    
            data = sample_batched['X'].to(self.device)
            batch = sample_batched[self.batch + '_encoded'].to(self.device)
            batch = batch.to(torch.float32)

            z, mu_z, var_z, mu_l, var_l, mu_x, pi_x, x_recon = self.vae(data, batch)

            theta_x = torch.exp(self.real_dis_logit)                
            zinb_term =  zinb(data, mu_x, theta_x, pi_x)

            kl_term =  self.kl_scale * kl_div(mu_z, var_z, self.batch_size) + \
            self.kl_scale * kl_div_l(mu_l, var_l, self.lib_mu, self.lib_var)

            ELBO = zinb_term - kl_term 
                           
                        
            #################################
            # Adversarial loss discriminators
            #################################
            
            predict_labels = self.batch_classifier(z)            
            classifier_loss = crossentropy_loss(predict_labels, batch)
          
            p_x, p_x_recon = self.discriminator(data, x_recon)
            bhattachryya_loss = Bhattacharyya_loss(p_x, p_x_recon)
            
            auto_encoder_loss = - ELBO - classifier_loss - bhattachryya_loss
            self.vae_optimizer.zero_grad()
            auto_encoder_loss.backward()
            self.vae_optimizer.step()
            
            #################################
            # Training discriminators
            #################################
            
            with torch.no_grad():
                z, _, _, _, _, _, _, x_recon = self.vae(data, batch)

            predict_labels = self.batch_classifier(z)

            classifier_loss = crossentropy_loss(predict_labels, batch)

            self.bc_optimizer.zero_grad()          
            classifier_loss.backward()               
            self.bc_optimizer.step()

            p_x, p_x_recon = self.discriminator(data, x_recon)
            bhattachryya_loss = Bhattacharyya_loss(p_x, p_x_recon)

            self.dis_optimizer.zero_grad()
            bhattachryya_loss.backward()   
            self.dis_optimizer.step()
            
            c_loss_sum += classifier_loss.detach()
            b_loss_sum += bhattachryya_loss.detach()
            auto_encoder_loss_sum += auto_encoder_loss.detach()
            
            n_batches += 1
                
            
        print ("Epoch :", epoch, "c loss:", c_loss_sum/n_batches, "b loss:", b_loss_sum/n_batches)                   
        print ("Epoch :", epoch, "a loss:", auto_encoder_loss_sum/n_batches)                       
   
    
    def train_network(self):
                        
        for epoch in range(self.epochs):
            
            self.vae.train()
            self.batch_classifier.train()
            self.discriminator.train()
            
            self.train_vae(epoch, True)
            
            """    
            if (epoch in [1, 100, 150, 200, 250]):
                z = self.process(epoch)
                self.plot_embeddings(z, epoch)
                
            """

    # interchange the underscore sign when we want to use below function.
    def train_network_(self):
        
        self.vae.train()
        self.batch_classifier.train()
        self.discriminator.train()
        
        """    
        model.zero_grad() and optimizer.zero_grad() are the same only if all your model 
        parameters are in that optimizer. Else, it is safer to use model.zero_grad() to make 
        sure all the grads are zero e.g. if you  have 2 or more optimizers for one model.
        """
  
        torch.autograd.set_detect_anomaly(True)
    
        for epoch in range(self.epochs):
            ann_dataset = AnndataLoader1(self.adata, self.batch, self.cell_type)
            dataloader = DataLoader(ann_dataset, batch_size = self.batch_size, shuffle = True, num_workers = 4,drop_last = True)

            auto_encoder_loss_sum = 0
            n_batches = 0

            for i_batch, sample_batched in enumerate(dataloader):
         

                data = sample_batched['X'].to(self.device)
                batch = sample_batched[self.batch + '_encoded'].to(self.device)
                batch = batch.to(torch.float32)
                
                self.vae.zero_grad()
                self.vae_optimizer.zero_grad()

                z, mu_z, var_z, mu_l, var_l, mu_x, pi_x, x_recon = self.vae(data, batch)

                theta_x = torch.exp(self.real_dis_logit)                
                zinb_term =  zinb(data, mu_x, theta_x, pi_x)

                kl_term =  self.kl_scale * kl_div(mu_z, var_z, self.batch_size) + \
                self.kl_scale * kl_div_l(mu_l, var_l, self.lib_mu, self.lib_var)

                ELBO = zinb_term - kl_term #- penalty_term
                vae_loss = -ELBO
                
                #######################################
                # (1) Train VAE network
                #######################################
                
                # self.vae.zero_grad()
                vae_loss.backward(retain_graph = True)
                self.vae_optimizer.step()
        
                auto_encoder_loss_sum += vae_loss.item()
                n_batches += 1
            
                #######################################
                # (1) Update Discriminator network
                #######################################
                """
                for ep in range(5):

                    self.discriminator.zero_grad()
                    p_x, p_x_recon = self.discriminator(data, x_recon)
                    bhattachryya_loss = Bhattacharyya_loss(p_x, p_x_recon)
                                        
                    bhattachryya_loss.backward(retain_graph = True)
                    self.dis_optimizer.step()
                    
                    print (bhattachryya_loss.item())
            
                #######################################
                # (1) Adversarial training Auto-Encoder
                #######################################  
                
                self.vae.zero_grad()
                adv_loss_dis = -1*bhattachryya_loss
                adv_loss_dis.backward(retain_graph = True)
                self.vae_adv_optimizer.step()
   
                auto_encoder_loss_sum += auto_encoder_loss.detach()
                n_batches += 1
        
                #######################################
                # (1) Update Batch-classifier
                #######################################
                """
#                 for ep in range(5):
#                     self.batch_classifier.zero_grad()
#                     predict_labels = self.batch_classifier(z)            
#                     classifier_loss = crossentropy_loss(predict_labels, batch)
                    
#                     classifier_loss.backward(retain_graph = True)
#                     self.bc_optimizer.step()
                    
#                     print (classifier_loss.item())
                   
               
                #######################################
                # (1) Adversarial training Auto-Encoder
                #######################################
                
#                 self.vae.zero_grad()
#                 adv_loss = -1*classifier_loss
#                 adv_loss.backward(retain_graph = True)
#                 self.vae_adv_optimizer.step()
                              
                             


            print ("Epoch :", epoch, "a loss:", auto_encoder_loss_sum/n_batches)   
        
        
    def save_model(self, save_model_file):
        
        torch.save({'state_dict': self.vae.state_dict()}, save_model_file)
        print('Saving model to %s' % save_model_file)

    def load_model(self, save_model_file):
        
        saved_state_dict = torch.load(save_model_file)
        self.model.load_state_dict(saved_state_dict['state_dict'])
        print('Loading model from %s' % save_model_file)

    def process(self, epoch):
        
        self.vae.eval()
        
        data = torch.FloatTensor(self.adata.X).to(self.device)
        batch = torch.Tensor(self.adata.obsm[self.batch + '_encoded']).to(self.device)
        
        z, _, _, _, _, _, _, _ = self.vae(data, batch)
         
        z = z.data.cpu().numpy()
        np.savetxt(self.name + "_z_" + str(epoch) + ".csv", z, delimiter = ",")
        return z
    
    def plot_embeddings(self, z, epoch):

        self.adata.obsm['final_embeddings'] = pd.DataFrame(z,index = self.adata.obs_names).to_numpy()

        sc.pp.neighbors(self.adata, use_rep = 'final_embeddings') #use_rep = 'final_embeddings'
        sc.tl.umap(self.adata)
        
        sc.pl.umap(self.adata, color = self.cell_type, frameon = False, show = False) # cells
        plt.savefig(self.name + "_cell_" + str(epoch))
        sc.pl.umap(self.adata, color = self.batch, frameon = False, show = False)
        plt.savefig(self.name + "_batch_" + str(epoch))

        
        
        
        
        
        
        
    
    
    
    
    
    




