"""
scDREAMER.py: running parameters
"""

import os
import torch
import argparse
import warnings
import numpy as np
import anndata
import scanpy as sc
import matplotlib.pyplot as plt
import pandas as pd
from scDREAMER_train import scDREAMER_Train
from utils import *

torch.cuda.empty_cache()
warnings.filterwarnings('ignore')
torch.cuda.cudnn_enabled = False

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

print(torch.cuda.device_count())

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('===== Using device: ' + device)

# ################ Parameter setting ################# 
#Pancreas, Immune_Human, Lung, Macaque_Retina,Human_Mouse, Heart

parser = argparse.ArgumentParser()

batch_key_dic = {'Immune_Human' : 'batch',
                 'Lung' : 'batch',
                 'Pancreas' : 'tech',
                 'Human_Mouse' : 'batch',
                 'Human_Retina': "Batch"                 
                }

cell_type_key_dic = {'Immune_Human' : 'final_annotation', # 16
                 'Lung' : 'cell_type', # 
                 'Pancreas' : 'celltype', #
                 'Human_Mouse' : "celltype", #
                 "Human_Retina":"Subcluster" #  
                  }      

parser.add_argument("--epochs", type = int, default = 100, help = "number of epochs")
parser.add_argument("--lr", type = float, default = 0.0007, help = "learning rate") # 0.0007, 0.001 - I
parser.add_argument("--batch_size", type = int, default = 128, help = "batch size for training the network")
parser.add_argument("--X_dim", type = int, default = 2000, help = "highly variable genes")
parser.add_argument("--z_dim", type = int, default = 10, help = "dim for latent space representation")
parser.add_argument("--y_dim", type = int, default = 6, help = "dim for latent space representation of Hierarchical VAE")
parser.add_argument("--name", type = str, default = "Lung", help = "dataset name") #Pancreas
parser.add_argument("--actv", type = str, default = "sig", help = "dataset name")
parser.add_argument("--save_path", type = str, default = "./output/", help = "dataset name")
parser.add_argument('--p_drop', type=float, default = 0.1, help='Dropout rate.')

params = parser.parse_args()
params.device = device
params.batch = batch_key_dic[params.name]
params.cell_type = cell_type_key_dic[params.name] #None

                     
# ################ Data Loading and preprocessing ################# 

data_path = "../Pan/Pancreas.h5ad"
data_path = "../Lung/Lung_NA_0.5.h5ad" #Lung_NA_0.5, Lung_atlas_public

adata = read_data(data_path, params.batch, params.cell_type, params.name)


# ################ Model Training ################# 

scdreamer_net = scDREAMER_Train(adata, params)
scdreamer_net.train_network()


# Evaluation and plotting ################# 


z = scdreamer_net.process(adata, 300) # de_feat
#adata.obsm['final_embeddings'] = pd.DataFrame(z,index = adata.obs_names).to_numpy()

#plot_embeddings(adata, z, params) # No call: as plots not visible in terminal

#pd.DataFrame(batch_info_enc).to_numpy()
scdreamer_net.save_model("./output/" + params.name + "/model")

# ################ saving latent space  ################

#pd.DataFrame(z).to_csv(os.path.join(params.save_path, params.name + "/" + params.name + "_latent_matrix.csv"))
#np.savez(os.path.join(params.save_path, name + "/" + name + "_latent_matrix.npz"), x = x)
np.savetxt(os.path.join(params.save_path, params.name + "/" + params.name + "_latent_matrix.csv"), z, delimiter = ",")
































