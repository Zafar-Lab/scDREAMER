"""
utils.py has all the utility functions for running scDREAMER
"""


import scanpy as sc
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi


def read_data(data_path, batch, cell_type, name, hvg=2000):

    Ann = sc.read_h5ad(data_path)
    

    if type(Ann.X) != type(np.array([])):
        Ann.X = Ann.X.todense()
        
    Ann.layers["counts"] = Ann.X.copy()

    sc.pp.normalize_total(Ann, target_sum=1e4)
    sc.pp.log1p(Ann)
    Ann.raw = Ann 
    
    sc.pp.highly_variable_genes(
        Ann, 
        flavor="seurat", 
        n_top_genes=hvg,
        batch_key=batch,
        subset=True)
    
    b = Ann.obs[batch] #.to_list()
    batch_info = np.array([[i] for i in b]) 
    enc = OneHotEncoder(handle_unknown = 'ignore')

    enc.fit(batch_info.reshape(-1, 1))
    batch_info_enc = enc.transform(batch_info.reshape(-1, 1)).toarray()
    Ann.obsm[batch + '_encoded'] = pd.DataFrame(batch_info_enc).to_numpy()


    labels = Ann.obs[cell_type].to_list()

    c = Ann.obs[cell_type]
    cell_info = np.array([[i] for i in c])

    enc.fit(cell_info.reshape(-1, 1))
    labels_enc = enc.transform(cell_info.reshape(-1, 1)).toarray()

    Ann.obsm[cell_type + "_encoded"] = pd.DataFrame(labels_enc).to_numpy()
    

    # Ann data scaling as earlier    
    Ann.X = np.log2(Ann.X + 1)/np.max(Ann.X)
    
    
    """
    data = np.log2(data+1)
    scale = np.max(data)
    data /= scale 
    """

    return Ann
    



