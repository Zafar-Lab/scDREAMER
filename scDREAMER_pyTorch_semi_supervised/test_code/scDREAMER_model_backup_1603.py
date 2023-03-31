"""
scDREAMER's network architecture
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


def full_block(in_features, out_features, p_drop):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        #nn.BatchNorm1d(out_features, momentum=0.01, eps=0.001),
        #nn.ReLU(),
        nn.ELU(),
        nn.Dropout(p = p_drop),
    )


class scDREAMER(nn.Module):
    def __init__(self, params):
        
        super(scDREAMER, self).__init__()

        
        # Encoder
        self.encoder = nn.Sequential()
        self.encoder.add_module('encoder_L0', full_block(params.X_dim + params.num_batches, 512, params.p_drop))
        self.encoder.add_module('encoder_L1', full_block(512, 256, params.p_drop))
                
        self.enc_z_mu = full_block(256, params.z_dim, params.p_drop)
        self.enc_z_var = full_block(256, params.z_dim, params.p_drop)
        
        self.enc_hidden = full_block(256, 10, params.p_drop)
        self.enc_l_mu = full_block(10, 1, params.p_drop)
        self.enc_l_var = full_block(10, 1, params.p_drop)
        
        #Decoder
        self.decoder = nn.Sequential()
        self.decoder.add_module('decoder_L0', full_block(params.z_dim + params.num_batches, 256, params.p_drop))
        self.decoder.add_module('decoder_L1', full_block(256, 512, params.p_drop))
        
        self.decoder_x_mu = nn.Sequential(nn.Linear(512, params.X_dim), nn.Softmax())
        self.decoder_logit = nn.Linear(512, params.X_dim)
        self.decoder_x_recon =  nn.Sequential(nn.Linear(512, params.X_dim), nn.Sigmoid())
    
        # Batch Classifier
        self.batch_classifier = nn.Sequential()
        self.batch_classifier.add_module('batch_classifier_L0', full_block(params.z_dim, 256, params.p_drop))
        self.batch_classifier.add_module('batch_classifier_L1', full_block(256, 512, params.p_drop))
        self.batch_classifier.add_module('batch_classifier_L2', full_block(512, params.num_batches, params.p_drop))

        # Discriminator
        self.discriminator = nn.Sequential()
        self.discriminator.add_module('discriminator_L0', full_block(params.X_dim, 256, params.p_drop))
        self.discriminator.add_module('discriminator_L1', full_block(256, 512, params.p_drop))
        self.discriminator.add_module('discriminator_L2', full_block(512, 1, params.p_drop))

    def reparameterize(self, mu, std):
        
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    
    # Encode Pass
    def encode(self, x_b):
        
        hidden1 = self.encoder(x_b)                              
        mu_z = self.enc_z_mu(hidden1)
        var_z = torch.exp(self.enc_z_var(hidden1))
                                      
        hidden2 = self.enc_hidden(hidden1)
        mu_l = self.enc_l_mu(hidden2)
        var_l = torch.exp(self.enc_l_var(hidden2))
                                      
        latent_z = self.reparameterize(mu_z, var_z)
        latent_l = self.reparameterize(mu_l, var_l)
        
        return latent_z, latent_l, mu_z, var_z, mu_l, var_l
    
    # Decoder Pass
    def decode(self, latent_z_b, latent_l):
        hidden = self.decoder(latent_z_b)
                                      
        mu_x = self.decoder_x_mu(hidden)
        mu_x = torch.mul(mu_x, torch.exp(latent_l)) #TODO
        
        pi_x = self.decoder_logit(hidden)
        
        x_recon = self.decoder_x_recon(hidden)
        return mu_x, pi_x, x_recon
        
    # Forward Pass
    def forward(self, x, batches):
        
        # Encoder: concat x with batches                                         
        x_b = torch.cat((x, batches), 1)
        latent_z, latent_l, mu_z, var_z, mu_l, var_l = self.encode(x_b)                         

        # Decoder: concat latent_z with batches                                      
        latent_z_b = torch.cat((latent_z, batches), 1) 
        mu_x, pi_x, x_recon = self.decode(latent_z_b, latent_l)
            
        """
        # Batch-classifier
        predict_labels = self.batch_classifier(latent_z)
        
        # Discriminator
        p = self.discriminator(x_recon)
        q = self.discriminator(x)
        """
        
        #return latent_z, mu_z, var_z, mu_l, var_l, mu_x, pi_x, x_recon, predict_labels, p, q
        return latent_z, mu_z, var_z, mu_l, var_l, mu_x, pi_x, x_recon





