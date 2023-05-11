
"""
For Louvain clustering and computing cluster connectivity metric
"""

from annoy import AnnoyIndex
import pandas as pd
import numpy as np
import scanpy as sc

"""
##############################
Computing Cluster Connectivity
##############################
"""

def compute_undirected_cluster_connectivity(
    communities, adj, z_threshold=1.0, conn_threshold=None
):
    N = communities.shape[0]
    n_communities = np.unique(communities).shape[0]

    # Create cluster index
    clusters = {}
    for idx in np.unique(communities):
        cluster_idx = communities == idx
        clusters[idx] = cluster_idx

    undirected_cluster_connectivity = pd.DataFrame(
        np.zeros((n_communities, n_communities)),
        index=np.unique(communities),
        columns=np.unique(communities),
    )
    undirected_z_score = pd.DataFrame(
        np.zeros((n_communities, n_communities)),
        index=np.unique(communities),
        columns=np.unique(communities),
    )
    cluster_outgoing_edges = {}
    for i in np.unique(communities):
        cluster_i = clusters[i]

        # Compute the outgoing edges from the ith cluster
        adj_i = adj[cluster_i, :]
        adj_ii = adj_i[:, cluster_i]
        e_i = np.sum(adj_i) - np.sum(adj_ii)
        n_i = np.sum(cluster_i)
        cluster_outgoing_edges[i] = e_i

        for j in np.unique(communities):
            if i == j:
                continue
            # Compute the outgoing edges from the jth cluster
            cluster_j = clusters[j]
            adj_j = adj[cluster_j, :]
            adj_jj = adj_j[:, cluster_j]
            e_j = np.sum(adj_j) - np.sum(adj_jj)
            n_j = np.sum(cluster_j)

            # Compute the number of inter-edges from the ith to jth cluster
            adj_ij = adj_i[:, cluster_j]
            e_ij = np.sum(adj_ij)

            # Compute the number of inter-edges from the jth to ith cluster
            adj_ji = adj_j[:, cluster_i]
            e_ji = np.sum(adj_ji)
            e_sym = e_ij + e_ji

            # Compute the random assignment of edges from the ith to the jth
            # cluster under the PAGA binomial model
            e_sym_random = (e_i * n_j + e_j * n_i) / (N - 1)

            # Compute the cluster connectivity measure
            std_sym = (e_i * n_j * (N - n_j - 1) + e_j * n_i * (N - n_i - 1)) / (
                N - 1
            ) ** 2
            undirected_z_score.loc[i, j] = (e_sym - e_sym_random) / std_sym

            # Only add non-spurious edges based on a threshold
            undirected_cluster_connectivity.loc[i, j] = (e_sym - e_sym_random) / (
                e_i + e_j - e_sym_random
            )
            if conn_threshold is not None:
                if undirected_cluster_connectivity.loc[i, j] < conn_threshold:
                    undirected_cluster_connectivity.loc[i, j] = 0
            elif undirected_z_score.loc[i, j] < z_threshold:
                undirected_cluster_connectivity.loc[i, j] = 0
    return undirected_cluster_connectivity, undirected_z_score

"""
# ....Computing connectivity....
from sklearn.neighbors import kneighbors_graph
A = kneighbors_graph(emb.X, 2, mode='connectivity', include_self=True)
A = A.toarray()

clus_conn, undirected_z_score = compute_undirected_cluster_connectivity(emb.obs['pseudo_cell_types'], A)
"""

def farthest_point(clusters, from_points):
    
    distances = np.zeros(clusters.shape[0])
    for point_ind,point in enumerate(clusters):
        
        dist = from_points.copy()
        dist = ((dist-point)**2).sum()
        distances[point_ind] = dist
        
    far_ind = distances.argmax()
    far_point = clusters[far_ind]
    
    return far_point


def find_resolution(adata, n_clusters):
    
    obtained_clusters = -1
    iteration = 0
    resolutions = [0., 1000.]
    
    while obtained_clusters != n_clusters and iteration < 50:
        
        current_res = sum(resolutions)/2
        sc.tl.louvain(adata, resolution = current_res, random_state = 0)
        
        labels = adata.obs['louvain']
        obtained_clusters = len(np.unique(labels))
        
        if obtained_clusters < n_clusters:
            resolutions[0] = current_res
        else:
            resolutions[1] = current_res
            
        iteration = iteration + 1
        
    print('iteration {}, obtained number of clusters {}'.format(iteration,obtained_clusters))
    return current_res

def nn_annoy(mean, emb, no_of_nn):
    
    tree = AnnoyIndex(emb.shape[1], metric = "angular") #euclidean
    tree.set_seed(0)
    
    for i in range(emb.shape[0]):
        tree.add_item(i, emb[i, :])
        
    tree.build(50)#n_trees=50
    ind = tree.get_nns_by_vector(mean, no_of_nn, search_k=-1) 
    #search_k=-1 means extract search neighbors
    # ind = np.array(ind)
    
    return ind

def update_sudo_lables_mean(adata, conn_clusters, label_percent):
    
    #adata.obs['celltype_NA'] = adata.obs['celltype_NA'].cat.add_categories(['NA'])
    
    for label in adata.obs['celltype_NA'].unique():
        
        if str(label) not in conn_clusters:
            
            print ('hi')
            is_label = adata.obs['celltype_NA'] == label
            
            no_of_nn = int(sum(is_label)*label_percent)
            names = adata.obs['celltype_NA'][is_label].index

            data = adata[is_label].X 
            mean = data.mean(axis=0)

            ind = nn_annoy(mean,data,no_of_nn)
            adata.obs['celltype_NA'][names] = 'NA'
            adata.obs['celltype_NA'][names[ind]] = label
        
    return adata

def update_sudo_lables(adata,conn_clusters, label_percent):
    
    adata.obs['celltype_NA'] = adata.obs['celltype_NA'].cat.add_categories(['NA'])
    from_points = np.zeros((len(adata.obs['celltype_NA'].unique()),adata.X.shape[1]))
    from_points_ind = 0
    
    # compute from_points data struture for farthest point assignment...
    for label in adata.obs['celltype_NA'].unique():

        if label in conn_clusters:

            is_label = adata.obs['celltype_NA'] == label
            names = adata.obs['celltype_NA'][is_label].index
            data = adata[is_label].X 
            mean = data.mean(axis = 0)
            from_points[from_points_ind] = mean
            from_points_ind += 1 
    
    
    for label in adata.obs['celltype_NA'].unique():

        #print ('label', label)
        if str(label) in conn_clusters:

            is_label = adata.obs['celltype_NA'] == label
            no_of_nn = int(sum(is_label)*label_percent) #no of nearest neighbours

            names = adata.obs['celltype_NA'][is_label].index
            data = adata[is_label].X 
            # mean = data.mean(axis = 0)

            mean = farthest_point(data, from_points)
            ind = nn_annoy(mean,data,no_of_nn)
            
            adata.obs['celltype_NA'][names] = 'NA'
            adata.obs['celltype_NA'][names[ind]] = label 
            
    return adata

def get_reference_labels(adata, ref_labels):
    
    old_dict = {}
    new_dict = {}
    
    ref_labels = np.array(ref_labels)
    curr_labels =  np.array(adata.obs['celltype_NA'].values.copy())
    
    adata.obs['celltype_NA_copy'] = adata.obs['celltype_NA'].values.copy()
    #adata.obs['celltype_NA'] = adata.obs['celltype_NA'].cat.add_categories(['NA'])
    # get indices for all clusters in both the labels array.
    
    
    for label in np.unique(ref_labels):
        
        if label == "NA":
            continue
            
        old_dict[label] = np.where(ref_labels == label)[0]

    for label in np.unique(curr_labels):
        
        new_dict[label] = np.where(curr_labels == label)[0]
        
        highest_match = label
        maxm = 0
        
        if label == "NA":
            continue
            
        for old_label in np.unique(ref_labels):
            
            if old_label == "NA":
                continue
                
            # Compute match percent with the label
            #print ('old_label', old_label)
            
            #print ('intersection', len(np.intersect1d(old_dict[old_label], new_dict[label])))
            
            val = len(np.intersect1d(old_dict[old_label], new_dict[label]))#/curr_labels.shape[0]
            
            if (val > maxm):
                #print ('old_label', old_label)
                maxm = val
                highest_match = old_label
    
        #print ('highest_match', highest_match)
        
        if (highest_match != label):
            adata.obs['celltype_NA_copy'][new_dict[label]] = highest_match
    
    # Replace with the new value
    adata.obs['celltype_NA'] = adata.obs['celltype_NA_copy'].values.copy()
    
    return adata
    








