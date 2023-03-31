"""

Network training module
-- details on the loss function
-- data loaders

"""

import time
import numpy as np
import torch
import scanpy as sc
from progress.bar import Bar
import torch.nn.modules.loss
import torch.nn.functional as F
from progress.bar import Bar
from scDREAMER_model import VAE, Batch_classifier, Discriminator
from torch.utils.data import Dataset, DataLoader

# Data Loader
class AnndataLoader1(Dataset):
    def __init__(self, adata, batch):
        
        self.batch = batch    
        self.Ann = adata
        
    def __len__(self):
        return len(self.Ann)
        
    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        x = self.Ann.X[idx, :]
        batch_encoded = self.Ann.obsm[self.batch + '_encoded'][idx, :]
        
        sample = {'X' : x, 
                  'batch_encoded' : batch_encoded} 
        
        return sample
    

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
    log_theta_eps = torch.log(theta + eps)

    log_theta_mu_eps = torch.log(theta + mu + eps)
    pi_theta_log = -pi + theta * (log_theta_eps - log_theta_mu_eps)

    case_zero = F.softplus(pi_theta_log) - softplus_pi
    mul_case_zero = torch.mul((x < eps).type(torch.float32), case_zero)

    case_non_zero = (
        -softplus_pi
        + pi_theta_log
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
    )
    mul_case_non_zero = torch.mul((x > eps).type(torch.float32), case_non_zero)

    res = mul_case_zero + mul_case_non_zero

    res = torch.sum(res) # TODO: returns sum of each row of the 2D tensor
    
    return torch.mean(res)

def kl_div_l(mu1, var1, mu2, var2):
    
    """
    l_post = torch.normal(mu1, var1)
    l_prior = torch.normal(mu2, var2)
    
    kl_loss = torch.nn.functional.kl_div(l_post, l_prior) 
    # mu2, var2 = prior
    """

    kl_loss = 2*torch.log(var2/var1) + torch.square(var2/var1) + torch.square((mu1 - mu2)/var2) - 1
    #kl_loss = 2 * (var2/var1.exp()) + torch.square(var2/var1.exp()) + torch.square((mu1 - mu2)/var2) - 1
    
    return 0.5 * torch.mean(torch.sum(kl_loss))

def kl_div(mu, var, n_obs):
    
    """
    Computes KL divergence between posterior sample ~ N(mu, sigma) and prior sample ~ N(0, 1) 
    """    
    #kl_loss = torch.nn.functional.kl_div(posterior, prior)    
   
    kl_loss = -0.5 * torch.mean(torch.sum(1 - torch.pow(mu, 2) - torch.pow(var, 2) + 2 * torch.log(var)))
    #kl_loss = -0.5 * torch.mean(torch.sum(1 - torch.pow(mu, 2) - torch.pow(var.exp(), 2) + 2 * var))
    
    return kl_loss
    
    
    
def Bhattacharyya_loss(p, q):
    
    """
    Discriminator maximizes the Bhattachryya loss i.e. B.D (x, x')
    i.e. minimizes -ve Bhattachryya loss i.e. log (B.C.)
    assert: tmp >= 0, square root of -ve = nan; log(0) - undefined.
    """
    p[p < 0] *= -1
    q[q < 0] *= -1
    
    tmp = torch.sum(torch.mul(p, q), 1)
    b_loss = torch.log(torch.sum(torch.sqrt(tmp)))
    
    return b_loss
    
    
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
        

        self.device = params.device
        self.batch_size = params.batch_size
        self.epochs = params.epochs
        self.batch = params.batch
        self.cell_type = params.cell_type
        self.z_dim = params.z_dim
        self.X_dim = params.X_dim
        self.lr = params.lr
        self.num_batches = len(adata.obs[params.batch].unique())
        
        #self.data = torch.FloatTensor(data.copy()).to(self.device)
        
        self.adata = adata
        self.total_size = len(self.adata)
        self.num_minibatches = int(self.total_size/self.batch_size)        
        params.num_batches = self.num_batches
        self.params = params
        
        self.vae = VAE(self.params).to(self.device)
        #self.batch_classifier = Batch_classifier(self.params).to(self.device)
        #self.discriminator = Discriminator(self.params).to(self.device)
        
        self.kl_scale = 0.01

        lib_real = torch.log(torch.sum(torch.FloatTensor(self.adata.X), axis = 1))
        self.lib_mu = torch.mean(lib_real)
        self.lib_var = torch.var(lib_real)
                
        # Sample from Gaussian N(0, 1): Retuns a normal dist with mea 0 and variance 1        
        #self.real_dis = np.random.randn(self.batch_size, self.z_dim).to(self.device)
        self.real_dis = torch.randn_like(torch.zeros(self.batch_size, self.z_dim)).to(self.device)
        self.real_dis_logit = torch.randn_like(torch.zeros(self.batch_size, self.X_dim)).to(self.device)
        
        self.vae_optimizer = torch.optim.Adam(params = list(self.vae.parameters()), lr = self.params.lr)
        #self.bc_optimizer = torch.optim.Adam(params = list(self.batch_classifier.parameters()), lr = self.params.lr)
        #self.dis_optimizer = torch.optim.Adam(params = list(self.discriminator.parameters()), lr = self.params.lr)
        
    def train_vae(self, epoch):
        
        for it in range(self.num_minibatches):
            
            # mini batch for network training                                           
            indx = it * self.batch_size

            data = torch.FloatTensor(self.adata.X[indx : indx + self.batch_size,:]).to(self.device)
            batch = self.adata.obsm[self.batch + '_encoded'][indx : indx + self.batch_size,:]
            batch = torch.Tensor(batch).to(self.device)

            #z, mu_z, var_z, mu_l, var_l, mu_x, pi_x, x_recon, predict_labels, p_x, p_x_recon = self.model(data, batch)
            z, mu_z, var_z, mu_l, var_l, mu_x, pi_x, x_recon = self.vae(data, batch)

            theta_x = torch.exp(self.real_dis_logit)                
            zinb_term =  zinb(data, mu_x, theta_x, pi_x)

            kl_term =  self.kl_scale * kl_div(mu_z, var_z, self.batch_size) + \
            self.kl_scale * kl_div_l(mu_l, var_l, self.lib_mu, self.lib_var)

            ELBO = zinb_term - kl_term

            #predict_labels = self.batch_classifier(z)
            #p_x, p_x_recon = self.discriminator(data, x_recon)

            #classifier_loss = crossentropy_loss(predict_labels, batch)
            #bhattachryya_loss = Bhattacharyya_loss(p_x, p_x_recon)

            # max ELBO i.e. min -ve ELBO & adversarial training BC and D networks

            auto_encoder_loss = - ELBO #- classifier_loss #- bhattachryya_loss

            self.vae_optimizer.zero_grad()
            
            # clip the gradients
            torch.nn.utils.clip_grad_norm_(self.vae.parameters(), max_norm = 1)
            
            auto_encoder_loss.backward()
            self.vae_optimizer.step()
            
        print ("Epoch :", epoch, "a loss:", auto_encoder_loss.item())                       

    def train_discriminators(self, epoch):
        
        for it in range(self.num_minibatches):
                                                      
            indx = it * self.batch_size

            data = torch.FloatTensor(self.adata.X[indx : indx + self.batch_size,:]).to(self.device)
            batch = self.adata.obsm[self.batch + '_encoded'][indx : indx + self.batch_size,:]
            batch = torch.Tensor(batch).to(self.device)

            # Batch_classifier step
            with torch.no_grad():
                z, _, _, _, _, _, _, x_recon = self.vae(data, batch)

            #print ("z", z.data.cpu().numpy())
            predict_labels = self.batch_classifier(z)

            classifier_loss = crossentropy_loss(predict_labels, batch)

            self.bc_optimizer.zero_grad()
            classifier_loss.backward()
            self.bc_optimizer.step()

            # Discriminator step
            """
            p_x, p_x_recon = self.discriminator(data, x_recon)
            bhattachryya_loss = Bhattacharyya_loss(p_x, p_x_recon)

            self.dis_optimizer.zero_grad()
            bhattachryya_loss.backward()                
            self.dis_optimizer.step()
            """
        print ("Epoch :", epoch, "c loss:", classifier_loss.item())\
               #, "b loss:", bhattachryya_loss.item())   
        
    def train_network(self):
        
        self.vae.train()
        #self.batch_classifier.train()
        #self.discriminator.train()
                
        #sc.pp.subsample(self.adata, fraction = 1) # inplace operation
        
        for epoch in range(self.epochs):

            start_time = time.time()
            a_loss = b_loss = c_loss = 0
                                  
            # VAE Step          
            for vae_epoch in range(1):
                self.train_vae(epoch)
                            
            #for ep in range(1):
            #    self.train_discriminators(epoch)
       

            """
            #a_loss += auto_encoder_loss
            #b_loss += bhattachryya_loss
            #c_loss += classifier_loss                
            end_time = time.time()
            batch_time = end_time - start_time
            print ("Epoch :", epoch, "a loss:", auto_encoder_loss.item(), " bc loss ", classifier_loss.item(), \
               " dis loss ", bhattachryya_loss.item())
            """


    def save_model(self, save_model_file):
        
        torch.save({'state_dict': self.vae.state_dict()}, save_model_file)
        print('Saving model to %s' % save_model_file)

    def load_model(self, save_model_file):
        
        saved_state_dict = torch.load(save_model_file)
        self.model.load_state_dict(saved_state_dict['state_dict'])
        print('Loading model from %s' % save_model_file)

    def process(self, Ann):
        
        self.vae.eval()
        
        data = torch.FloatTensor(Ann.X).to(self.device)
        batch = torch.Tensor(Ann.obsm[self.batch + '_encoded']).to(self.device)
        
        #z, _, _, _, _, _, _, _, _, _, _ = self.model(data, batch)
        z, _, _, _, _, _, _, _ = self.vae(data, batch)
         
        z = z.data.cpu().numpy()
        
        return z
    
    
        
        
        
        
        
        
        
        
    
    
    
    
    
    




