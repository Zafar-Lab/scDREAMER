"""
scDREAMER's network architecture
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

def init_weights(m):
    
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.00) #0.0 used in scDREAMER_tensorflow version
        
def full_block(in_features, out_features, p_drop):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.BatchNorm1d(out_features, momentum=0.01, eps=0.001),
        #nn.ReLU(),
        nn.ELU(),
        nn.Dropout(p = p_drop, inplace = False),
    )

def reparameterize(mu, std):

    eps = torch.randn_like(std)
    return eps.mul(std).add_(mu)
    
class Batch_classifier(nn.Module):
    def __init__(self, params):
        
        super(Batch_classifier, self).__init__()
        
        # Batch Classifier
        self.batch_classifier = nn.Sequential()
        self.batch_classifier.add_module('batch_classifier_L0', full_block(params.z_dim, 256, params.p_drop))
        self.batch_classifier.add_module('batch_classifier_L1', full_block(256, 512, params.p_drop))
        self.batch_classifier.add_module('batch_classifier_L2', full_block(512, params.num_batches, params.p_drop))
        
        self.batch_classifier.apply(init_weights)
        
    # Forward Pass
    def forward(self, latent_z):
        
        # Batch-classifier
        predict_labels = self.batch_classifier(latent_z)
        return predict_labels 
    
class Encoder(nn.Module):
    
    def __init__(self, params):
        
        super(Encoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential()
        self.encoder.add_module('encoder_L0', full_block(params.X_dim + params.num_batches, 512, params.p_drop))
        self.encoder.add_module('encoder_L1', full_block(512, 256, params.p_drop))
                
        self.enc_z_mu = full_block(256, params.z_dim, params.p_drop)
        self.enc_z_var = full_block(256, params.z_dim, params.p_drop)
        
        self.enc_hidden = full_block(256, 10, params.p_drop)
        self.enc_l_mu = full_block(10, 1, params.p_drop)
        self.enc_l_var = full_block(10, 1, params.p_drop)
        
        # initialize weights..encoder..
        self.encoder.apply(init_weights)
        self.enc_z_mu.apply(init_weights)
        self.enc_z_var.apply(init_weights)
        self.enc_hidden.apply(init_weights)
        self.enc_l_mu.apply(init_weights)
        self.enc_l_var.apply(init_weights)
    
    # Encoder Forward Pass
    def forward(self, x_b):
        
        hidden1 = self.encoder(x_b)                              
        mu_z = self.enc_z_mu(hidden1)
        var_z = torch.exp(self.enc_z_var(hidden1))
                                      
        hidden2 = self.enc_hidden(hidden1)
        mu_l = self.enc_l_mu(hidden2)
        var_l = torch.exp(self.enc_l_var(hidden2))
                                      
        latent_z = reparameterize(mu_z, var_z)
        latent_l = reparameterize(mu_l, var_l)
        
        return latent_z, latent_l, mu_z, var_z, mu_l, var_l
    

class EncoderY(nn.Module):
    
    def __init__(self, params):
        
        super(EncoderY, self).__init__()
        
        # Encoder
        self.encoderY = nn.Sequential()
        self.encoderY.add_module('encoderY_L0', full_block(params.z_dim + params.num_cell_type, 512, params.p_drop))
        self.encoderY.add_module('encoderY_L1', full_block(512, 256, params.p_drop))
                
        self.enc_y_mu = full_block(256, params.y_dim, params.p_drop)
        self.enc_y_var = full_block(256, params.y_dim, params.p_drop)
        
        # initialize weights..encoder Y..
        self.encoderY.apply(init_weights)
        self.enc_y_mu.apply(init_weights)
        self.enc_y_var.apply(init_weights)
        
    # Encoder Forward Pass
    def forward(self, z_c):
        
        hidden = self.encoderY(z_c)                              
        mu_y = self.enc_y_mu(hidden)
        var_y = torch.exp(self.enc_y_var(hidden))
                                      
        latent_y = reparameterize(mu_y, var_y)
        
        return latent_y, mu_y, var_y
    
    
class Decoder(nn.Module):
    def __init__(self, params):

        super(Decoder, self).__init__()


        #Decoder
        self.decoder = nn.Sequential()
        self.decoder.add_module('decoder_L0', full_block(params.z_dim + params.num_batches, 256, params.p_drop))
        self.decoder.add_module('decoder_L1', full_block(256, 512, params.p_drop))

        self.decoder_x_mu = nn.Sequential(nn.Linear(512, params.X_dim), nn.Softmax())
        self.decoder_x_logit = nn.Linear(512, params.X_dim)
        #self.decoder_x_theta = nn.Linear(512, params.X_dim)
        self.decoder_x_recon =  nn.Sequential(nn.Linear(512, params.X_dim), nn.Sigmoid())
        #self.decoder_x_recon =  nn.Linear(512, params.X_dim)

        # initialize weights..decoder..
        self.decoder.apply(init_weights)
        self.decoder_x_mu.apply(init_weights)
        self.decoder_x_logit.apply(init_weights)
        self.decoder_x_recon.apply(init_weights)
        
        
    # Decoder Forward Pass
    def forward(self, latent_z_b, latent_l):
        hidden = self.decoder(latent_z_b)
                                      
        mu_x = self.decoder_x_mu(hidden)
        mu_x = torch.mul(mu_x, torch.exp(latent_l)) #TODO
        
        pi_x = self.decoder_x_logit(hidden)
        #theta_x = self.decoder_x_theta(hidden)
        
        x_recon = self.decoder_x_recon(hidden)
        
        return mu_x, pi_x, x_recon

class DecoderY(nn.Module):
    
    def __init__(self, params):
        
        super(DecoderY, self).__init__()
        
        # Encoder
        self.decoderY = nn.Sequential()
        self.decoderY.add_module('encoderY_L0', full_block(params.y_dim + params.num_cell_type, 128, params.p_drop))
                
        self.dec_z_mu = full_block(128, params.z_dim, params.p_drop)
        self.dec_z_var = full_block(128, params.z_dim, params.p_drop)
        
        # initialize weights..decoder Y..
        self.decoderY.apply(init_weights)
        self.dec_z_mu.apply(init_weights)
        self.dec_z_var.apply(init_weights)
        
        
    # Encoder Forward Pass
    def forward(self, latent_y_c):
        
        hidden = self.decoderY(latent_y_c)                              
        mu_z_pr = self.dec_z_mu(hidden)
        var_z_pr = torch.exp(self.dec_z_var(hidden))
                                              
        return mu_z_pr, var_z_pr
    
class Discriminator(nn.Module):
    def __init__(self, params):
        
        super(Discriminator, self).__init__()
        
        # Discriminator
        self.discriminator = nn.Sequential()
        self.discriminator.add_module('discriminator_L0', full_block(params.X_dim, 256, params.p_drop))
        self.discriminator.add_module('discriminator_L1', full_block(256, 128, params.p_drop))
        self.discriminator.add_module('discriminator_L2', full_block(128, 10, params.p_drop)) # earlier 10
    
        self.discriminator.apply(init_weights)
        
    # Forward Pass
    def forward(self, x, x_recon):
                
        # Discriminator        
        p = self.discriminator(x)
        q = self.discriminator(x_recon)
        
        return p, q
    
class VAE(nn.Module):
    def __init__(self, params):
        
        super(VAE, self).__init__()
        
        self.z_encoder = Encoder(params)
        self.z_decoder = Decoder(params)
        self.y_encoder = EncoderY(params)
        self.y_decoder = DecoderY(params)
        
        self.cell_classifier = nn.Sequential()
        self.cell_classifier.add_module('cell_type_classifier', full_block(params.z_dim, params.num_cell_type, params.p_drop))
        #self.cell_classifier = nn.Linear(params.z_dim, params.num_cell_type,)
        self.cell_classifier.apply(init_weights)
            
    # Forward Pass
    def forward(self, x, batches, cell_types):
        
        # Encoder: concat x with batches                                         
        x_b = torch.cat((x, batches), 1)
        latent_z, latent_l, mu_z, var_z, mu_l, var_l = self.z_encoder(x_b)                         

        # cell type classifier
        cell_labels_predict = self.cell_classifier(latent_z)
            
        # Decoder: concat latent_z with batches                                      
        latent_z_b = torch.cat((latent_z, batches), 1) 
        mu_x, pi_x, x_recon = self.z_decoder(latent_z_b, latent_l)
        
        # EncoderY      
        m = nn.Softmax(dim = 0)
        cell_types_softmax = m(cell_types)
        
        z_c = torch.cat((latent_z, cell_types_softmax), 1)
        latent_y, mu_y, var_y = self.y_encoder(z_c)
               
        # DecoderY
        latent_y_c = torch.cat((latent_y, cell_types_softmax), 1)
        mu_z_pr, var_z_pr = self.y_decoder(latent_y_c)
        
        #, mu_z_pr, var_z_pr,
        return latent_z, mu_z, var_z, mu_l, var_l, mu_x, pi_x, x_recon, mu_z_pr, var_z_pr, cell_labels_predict





