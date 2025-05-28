import copy, random
import numpy as np
import pandas as pd
import sklearn
import sklearn.linear_model as lm
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csr_matrix, csc_matrix

CLUSTERING_AGO = 'lv'
SKNETWORK = True
try:
    from sknetwork.clustering import Louvain
except ImportError:
    print('WARNING: sknetwork not installed. GMM will be used for clustering.')
    CLUSTERING_AGO = 'km'
    SKNETWORK = False

    
def get_neighbors(adj, n_neighbors):
    
    rows = adj.tocoo().row
    cols = adj.tocoo().col
    data = adj.data    
    
    N = int(np.max(rows) + 1)
    neighbors = np.full([N, n_neighbors], -1)
    distances = np.full([N, n_neighbors], -1, dtype = float)
    cnts = np.zeros(N, dtype = int)
    
    for r, c, d in zip(rows, cols, data):
        neighbors[r, cnts[r]] = c
        distances[r, cnts[r]] = d
        cnts[r] += 1
        
    for i in range(N):
        odr = distances[i,:].argsort()
        distances[i,:] = distances[i,odr]
        neighbors[i,:] = neighbors[i,odr]
        
    return neighbors, distances
    

def convert_adj_mat_dist_to_conn(adjacency_matrix, threshold = 0):
    # Create a new CSR matrix for the "connectivity" graph
    connectivity_matrix = csr_matrix(((adjacency_matrix.data >= threshold).astype(int),
                                      adjacency_matrix.indices, adjacency_matrix.indptr),
                                      shape=adjacency_matrix.shape)
    return connectivity_matrix


def Louvain_clustering( X, n_neighbors = 10, clustering_resolution = 1, 
                        mode = 'distance', adj_mat = None, n_cores = 4):

    if adj_mat is None:
        adj_mat = kneighbors_graph(X, n_neighbors, mode=mode, n_jobs = n_cores, include_self=True)
    louvain = Louvain(resolution = clustering_resolution, shuffle_nodes = True)
    cluster_label = louvain.fit_predict(adj_mat)   
    
    return adj_mat, cluster_label


def clustering_alg(X_pca, clust_algo = 'lv', N_clusters = 25, resolution = 1, 
                   N_neighbors = 10, n_cores = 4, adj_dist = None,
                   n_resampling_runs = 15, resampling_rate = 0.95, 
                   consensus_thresold = 0.5, min_cluster_size = 20,
                   random_state = 0):
                   # mode='distance', n_cores = 4, mode='connectivity'):
    
    if clust_algo[:2] == 'gm':
        gmm = mixture.GaussianMixture(n_components = int(N_clusters), random_state = 0)
        cluster_label = gmm.fit_predict(np.array(X_pca))
        return cluster_label, gmm, adj_dist
    elif clust_algo[:2] == 'km':
        km = cluster.KMeans(n_clusters = int(N_clusters), random_state = 0)
        km.fit(X_pca)
        cluster_label = km.labels_
        return cluster_label, km, adj_dist
    else:
        #'''
        if adj_dist is None:
            adj_dist = kneighbors_graph(X_pca, int(N_neighbors), mode='distance', include_self=True, 
                                   n_jobs = n_cores)
            
        ## Assumed to be in "distance" mode
        adj_in = convert_adj_mat_dist_to_conn(adj_dist, threshold = 0)

        for j in range(10):
            louvain = Louvain(resolution = resolution, random_state = random_state)
            if hasattr(louvain, 'fit_predict'):
                cluster_label = louvain.fit_predict(adj_in)        
            else:
                cluster_label = louvain.fit_transform(adj_in)        
            if len(list(set(list(cluster_label)))) <= 100:
                break
            else:
                resolution = resolution - 0.2
            
        return cluster_label, louvain, adj_dist

    
def get_cluster_lst_and_sizes(cluster_label):

    clust_lst = list(set(cluster_label))
    clust_lst.sort()

    clust_size = [] 
    for c in clust_lst:
        b = cluster_label == c
        clust_size.append(np.sum(b))

    return clust_lst, clust_size


def get_cluster_adj_mat_from_adj_dist(adj_dist, cluster_label):

    ## adj_dist: sparse adj matrix
    ## cluster_label: cluster label for each node in the adj matrix
    ## return a dense aggregated adj matrix

    rows = adj_dist.tocoo().row
    cols = adj_dist.tocoo().col
    vals = adj_dist.data
    
    ## Compute Cluster size, Aggregated Adj.Mat and Merge Small clusters         
    clust_lst = list(set(cluster_label))
    clust_lst.sort()
    
    adj_agg_mat = np.zeros([len(clust_lst), len(clust_lst)], dtype = int)
    for r, c, v in zip(rows, cols, vals):
        adj_agg_mat[cluster_label[r],cluster_label[c]] += 1
    
    adj_agg_mat = adj_agg_mat - np.diag(np.diag(adj_agg_mat))
    adj_agg_mat = adj_agg_mat + adj_agg_mat.transpose()

    return adj_agg_mat


def extend_major_clusters( adj_agg_mat, seed_clusters, 
                           cluster_size, n_neighbors, alpha = 0.08, 
                           mode = 'sum', verbose = False ):

    selected_clusters = copy.deepcopy(seed_clusters)
    maj_clusters = copy.deepcopy(seed_clusters)
    pair_clusters = list(np.zeros(len(seed_clusters)))
    metrics = list(np.zeros(len(seed_clusters)))
    thresholds = list(np.zeros(len(seed_clusters)))
    
    csz_lst = [cluster_size[n] for n in maj_clusters]
    odr = (-np.array(csz_lst)).argsort()
    maj_clusters = [maj_clusters[o] for o in odr]
    
    core_mat = adj_agg_mat[maj_clusters, :][:,maj_clusters]

    for j, n in enumerate(maj_clusters):
        cnt_n = adj_agg_mat[maj_clusters,n]                
        odr = np.array(cnt_n).argsort()
        p = maj_clusters[int(odr[-1])]
        if mode == 'max':
            met = (adj_agg_mat[p,n]/n_neighbors)
            csz = met/min(cluster_size[n], max(cluster_size[p], cluster_size[n]/2))
        else:
            met = (np.sum(cnt_n)/n_neighbors)
            csz = met/min(cluster_size[n], np.sum(np.array(cluster_size)[maj_clusters])-cluster_size[n])
            
        pair_clusters[j] = p
        metrics[j] = met
        thresholds[j] = csz # met/min(cluster_size[n], cluster_size[p])
    
    flag = True
    for a in range(adj_agg_mat.shape[0] - len(seed_clusters)):
        
        core_mat = adj_agg_mat[maj_clusters, :][:,maj_clusters]
        if mode == 'max':
            core_mxs = core_mat.max(axis = 1)
        else:
            core_mxs = core_mat.sum(axis = 1)
        
        core_dm = np.mean(core_mxs)
        core_ds = np.std(core_mxs)
        
        met = []
        nodes = []
        pair = []
        csz_lst = []
        for n in range(adj_agg_mat.shape[0]):

            if n not in maj_clusters:
                cnt_n = adj_agg_mat[maj_clusters,n]                
                odr = np.array(cnt_n).argsort()
                nodes.append(n)
                p = maj_clusters[int(odr[-1])]
                pair.append(p)
                csz_lst.append(cluster_size[n])
                if mode == 'max':
                    met.append(adj_agg_mat[p,n]/n_neighbors)
                else:
                    met.append(np.sum(cnt_n)/n_neighbors)
              
        cnt = 0
        med_cluster_size = 0 # np.median(csz_lst)
        met2 = []
        cond_lst = []
        for md, cn, pp, cs in zip(met, nodes, pair, csz_lst):

            if mode == 'max':
                csz = md/min(cluster_size[cn], max(cluster_size[pp], cluster_size[cn]/2))
            else:
                csz = md/min(cluster_size[cn], np.sum(np.array(cluster_size)[maj_clusters]))
            condition = (csz >= (alpha))

            met2.append(csz)
            cond_lst.append(condition)

        odr = np.array(met2).argsort()
        idx = odr[-1]
        condition = cond_lst[idx]
        cn = nodes[idx]
        pp = pair[idx]
        md = met[idx]
        csz = met2[idx]
        
        if flag & condition:
            maj_clusters.append(cn)
            pair_clusters.append(pp)
            metrics.append(md)
            thresholds.append(csz)
            selected_clusters = copy.deepcopy(maj_clusters)
            cnt += 1
            if verbose: 
                print('A', cn, pp, 'csz: %5.3f, alpha: %5.2f, md: %4i, csize: %4i, %4i' \
                      % (csz, alpha, int(md), cluster_size[cn], cluster_size[pp])) 
                    
        if len(selected_clusters) == len(cluster_size): break
        
        if cnt == 0:
            flag = False
            for md, cn, pp, cs in zip(met, nodes, pair, csz_lst):

                if mode == 'max':
                    csz = md/min(cluster_size[cn], max(cluster_size[pp], cluster_size[cn]/2))
                else:
                    csz = md/min(cluster_size[cn], np.sum(np.array(cluster_size)[maj_clusters]))
                condition = (csz >= (alpha))

                # if verbose: 
                #     print('B', cn, pp, 'md: %4i, csize: %4i, selected: ' % (int(md), cluster_size[cn]), selected_clusters) 
                # break
                maj_clusters.append(cn)
                pair_clusters.append(pp)
                metrics.append(md)
                thresholds.append(csz)
                pass
                    
        if len(maj_clusters) >= len(cluster_size): break
        
    core_mat = adj_agg_mat[maj_clusters, :][:,maj_clusters]
    if mode == 'max':
        core_mxs = core_mat.max(axis = 1)
    else:
        core_mxs = core_mat.sum(axis = 1)
    
    return (np.array(selected_clusters), 
           np.array(maj_clusters), np.array(pair_clusters), metrics, thresholds)


def get_connectivity_seq( cluster_adj_mat, clust_size, cluster_sel, n_neighbors = 14, net_search_mode = 'max', verbose = True ):

    cluster_adj_mat_sel = cluster_adj_mat[cluster_sel,:][:,cluster_sel]
    clust_size_sel = list(np.array(clust_size)[cluster_sel])
    
    cluster_adj_mat_nrom = get_normalized_agg_adj_mat( cluster_adj_mat_sel, clust_size_sel, n_neighbors = n_neighbors)
    
    j = cluster_adj_mat_nrom.max(axis = 1).argmax()
    seed = [j] #  [cluster_sel[j]]
    
    merged_cluster_sel, added_sel, conns_sel = merge_clusters_with_seed( cluster_adj_mat_sel, clust_size_sel, seed = seed,
                              n_neighbors = n_neighbors,
                              connectivity_thresh = 0.0,  
                              net_search_mode = net_search_mode,
                              verbose = True)
    
    merged_cluster_sel = [cluster_sel[j] for j in merged_cluster_sel]
    added_sel = [cluster_sel[j] for j in added_sel]
    
    merged_cluster, added, conns = merge_clusters_with_seed( cluster_adj_mat, clust_size, seed = cluster_sel,
                              n_neighbors = n_neighbors,
                              connectivity_thresh = 0.0,  
                              net_search_mode = net_search_mode,
                              verbose = True)
    
    merged_clusters =  merged_cluster_sel + merged_cluster
    added = added_sel + added
    conns = conns_sel + conns

    return conns, merged_clusters


def calculate_connectivity_threshold( connectivity_seq, no_cluster_sel, spf = 1/3 ):

    conns = connectivity_seq
    p2 = no_cluster_sel
    p1 = int(p2*spf)
    
    c_odr = np.arange(p1, p2)
    conns_sel = np.array(conns[p1:p2])
    
    z = np.polyfit(c_odr, conns_sel, 1)
    p = np.poly1d(z)
    
    conns_est = p(c_odr)
    conns_sd = np.abs(conns_est - conns_sel).mean()
    
    c_odr = np.arange(len(conns))
    conns_est = p(c_odr)
    
    conn_th = conns_est[p2] - conns_sd

    return conn_th, conns_est, conns_sd


def update_cluster_sel_and_threshold(conns, conn_th, cluster_sel, cluster_seq):
    
    cluster_sel_new = copy.deepcopy(cluster_sel)
    N_cluster_sel = len(cluster_sel)
    update = 0
    
    ## Check if any added clusters has connectivity below threshold
    b = np.array(conns)[N_cluster_sel:] < conn_th
    if np.sum(b) > 0:
        ## use conn_th and cluster_sel as is
        pass
    else:
        ## find the last cluster (in cluster_sel) that has its connectivity below threshold
        b = np.array(conns) < conn_th
        if np.sum(b) > 0:
            last = np.nonzero(b)[0][-1]
            if last > N_cluster_sel*2/3:
                for i in range(last):
                    if conns[last-(i+1)] >= conn_th:
                        break
                
                cluster_sel_new = cluster_seq[:(last - i)]
                cluster_sel_new.sort()
                ## use conn_th as is     
                update = 1
            else:
                update = 2      
                ## No tumor cells
        else:
            update = 2
            ## No tumor cells
            pass

    return update, conn_th, cluster_sel_new


def initially_detect_major_clusters_old( adj_agg_mat, 
                           cluster_size, n_neighbors, alpha = 0.08, 
                           mode = 'sum', verbose = False ):

    cluster_lst = list(np.arange(len(cluster_size)))
    
    selected_clusters = []
    for c in cluster_lst:
        seed_clusters = [c]
        selected, maj, pairs, mets, threshs = \
            extend_major_clusters( adj_agg_mat, seed_clusters, 
                           cluster_size, n_neighbors, alpha = alpha, 
                           mode = mode, verbose = verbose )
        
        if len(selected) > len(selected_clusters):
            selected_clusters = copy.deepcopy(selected)
            
    return selected_clusters


def get_normalized_agg_adj_mat( cluster_adj_mat, clust_size, n_neighbors = 10):

    cluster_adj_mat_nrom = cluster_adj_mat*1.0
    for i in range(cluster_adj_mat.shape[0]):
        vec = replace_with_min(clust_size, clust_size[i])
        cluster_adj_mat_nrom[i,:] = cluster_adj_mat[i,:]/(vec*n_neighbors)

    return cluster_adj_mat_nrom


def merge_clusters_with_seed( cluster_adj_mat, clust_size, seed = None,
                              n_neighbors = 10,
                              connectivity_thresh = 0.13, 
                              net_search_mode = 'sum',
                              verbose = False):

    clust_lst = list(np.arange(len(clust_size)))
    cluster_adj_mat_nrom = get_normalized_agg_adj_mat( cluster_adj_mat, clust_size, n_neighbors)

    ## Merge clusters
    merged_clusters = {}
    clust_lst_tmp = copy.deepcopy(clust_lst)
    agg_adj_mat_tmp = copy.deepcopy(cluster_adj_mat)
    clust_size_tmp = copy.deepcopy(clust_size)

    if seed is None:
        cluster_sel = initially_detect_major_clusters( agg_adj_mat_tmp, 
                           cluster_size = clust_size_tmp, n_neighbors = n_neighbors, 
                           alpha = connectivity_thresh, 
                           mode = net_search_mode, verbose = verbose )
    else:
        cluster_sel = copy.deepcopy(seed)
    
    s = ''
    for c in cluster_sel:
        s = s + '%i,' % c
    s = s[:-1]
        
    cluster_added = []
    connectivities = []
    for k in range(len(clust_lst_tmp)):
        c_update = {}
        for j in range(len(clust_lst_tmp)):
            if j not in cluster_sel:
                if net_search_mode == 'sum':
                    c = max(cluster_adj_mat_nrom[cluster_sel,j].sum(), cluster_adj_mat_nrom[j,cluster_sel].sum())
                else:
                    c = max(cluster_adj_mat_nrom[cluster_sel,j].max(), cluster_adj_mat_nrom[j,cluster_sel].max())
                    
                c_update[j] = c
            
        others = list(c_update.keys())
        if len(others) == 0:
            break
        else:
            vals = np.array(list(c_update.values()))
            j = others[ vals.argmax() ]
            if c_update[j] >= connectivity_thresh:
                cluster_sel.append(j)
                cluster_added.append(j)
                connectivities.append(c_update[j])
                # c_history.append(c_update[j])
                s = s + ' -(%4.2f)- %i' % (c_update[j], j)
            else:
                connectivities.append(c_update[j])
                break

    if verbose: print(s)

    return cluster_sel, cluster_added, connectivities


def initially_detect_major_clusters( adj_agg_mat, cluster_size, 
                                     n_neighbors = 14, connectivity_thresh = 0.1, 
                                     net_search_mode = 'max', verbose = False ):

    cluster_lst = list(np.arange(len(cluster_size)))
    
    selected_clusters = []
    for c in cluster_lst:
        if c not in selected_clusters:
            seed_clusters = [c]
            merged_clusters, addeded_clusters, connectivities = \
                merge_clusters_with_seed( adj_agg_mat, cluster_size, seed = seed_clusters,
                                          n_neighbors = n_neighbors,
                                          connectivity_thresh = connectivity_thresh, 
                                          net_search_mode = net_search_mode,
                                          verbose = False)
            
            if len(merged_clusters) > len(selected_clusters):
                selected_clusters = copy.deepcopy(merged_clusters)
            
    return selected_clusters

    
def merge_clusters_with_seed_one_step_only( cluster_adj_mat, clust_size, seed = None,
                              n_neighbors = 10,
                              connectivity_thresh = 0.13, 
                              net_search_mode = 'sum',
                              cluster_dist_mat = None, 
                              verbose = False):

    clust_lst = list(np.arange(len(clust_size)))
    # clust_lst, clust_size = get_cluster_lst_and_sizes(cluster_label)
    # cluster_adj_mat = get_cluster_adj_mat_from_adj_dist(adj_dist, cluster_label)

    cluster_adj_mat_nrom = cluster_adj_mat*1.0
    for i in range(cluster_adj_mat.shape[0]):
        vec = replace_with_min(clust_size, clust_size[i])
        cluster_adj_mat_nrom[i,:] = cluster_adj_mat[i,:]/(vec*n_neighbors)

    ## Merge clusters
    merged_clusters = {}
    clust_lst_tmp = copy.deepcopy(clust_lst)
    agg_adj_mat_tmp = copy.deepcopy(cluster_adj_mat)
    clust_size_tmp = copy.deepcopy(clust_size)

    if seed is None:
        cluster_sel = initially_detect_major_clusters( agg_adj_mat_tmp, 
                           cluster_size = clust_size_tmp, n_neighbors = n_neighbors, 
                           alpha = connectivity_thresh, 
                           mode = net_search_mode, verbose = verbose )
    else:
        cluster_sel = copy.deepcopy(seed)
    
    s = ''
    for c in cluster_sel:
        s = s + '%i,' % c
    s = s[:-1]
    
    # c_history = [c]
    c_update = {}
    for j in range(len(clust_lst_tmp)):
        if j not in cluster_sel:
            if net_search_mode == 'sum':
                c = min(cluster_adj_mat_nrom[cluster_sel,j].sum(), cluster_adj_mat_nrom[j,cluster_sel].sum())
            else:
                c = min(cluster_adj_mat_nrom[cluster_sel,j].max(), cluster_adj_mat_nrom[j,cluster_sel].max())
                
            c_update[j] = c
        
    vals = np.array(list(c_update.values()))
    keys = np.array(list(c_update.keys()))
    b = vals >= connectivity_thresh
    if np.sum(b) > 0:
        others = keys[b]
        for c in others:
            cluster_sel.append(c)
            # c_history.append(c_update[j])
            s = s + ' -(%4.2f)- %i' % (c_update[c], c)
    
        if verbose: print(s)

    return cluster_sel
    

def replace_with_min(clust_size, mv):
    clust_size_ary = np.array(clust_size)
    b = clust_size_ary > mv
    clust_size_ary[b] = mv
    return clust_size_ary


def merge_clusters(cluster_adj_mat, clust_size, n_neighbors = 10, 
                   connectivity_thresh = 0.13, verbose = False):

    merged_clusters = []
    cluster_adj_mat_nrom = cluster_adj_mat*1.0
    for i in range(cluster_adj_mat.shape[0]):
        vec = replace_with_min(clust_size, clust_size[i])
        cluster_adj_mat_nrom[i,:] = cluster_adj_mat[i,:]/(vec*n_neighbors)
    
    clust_lst_tmp = np.arange(len(clust_size))
    for kk in range(len(clust_lst_tmp)):

        if len(clust_lst_tmp) < 2:
            for c in clust_lst_tmp:
                merged_clusters.append([c])
            break
        else:
            i = cluster_adj_mat_nrom.max(axis = 1).argmax()
            j = cluster_adj_mat_nrom[i,:].argmax()
            c = cluster_adj_mat_nrom[i,:].max()
            
            if (c < connectivity_thresh):
                for c in clust_lst_tmp:
                    merged_clusters.append([c])
                break
            else:
                seed = [i, j]
                s = '%i -(%4.2f)- %i ' % (i, c, j)
                # display((seed, round(c,3)))

                c_history = [c]
                for k in range(len(clust_lst_tmp)):
                    c_update = {}
                    for j in range(len(clust_lst_tmp)):
                        if j not in seed:
                            c = max(cluster_adj_mat_nrom[seed,j].sum(), cluster_adj_mat_nrom[j,seed].sum())
                            c_update[j] = c
                        
                    others = list(c_update.keys())
                    if len(others) == 0:
                        break
                    else:
                        vals = np.array(list(c_update.values()))
                        j = others[ vals.argmax() ]
                        if c_update[j] > connectivity_thresh:
                            seed.append(j)
                            c_history.append(c_update[j])
                            s = s + '-(%4.2f)- %i ' % (c_update[j], j)
                        else:
                            break
        
                # display( np.array(seed) )
                # display( np.array(c_history) )
                if verbose: print(s)
            
                seed = list(np.array(clust_lst_tmp)[seed])
                merged_clusters.append(seed)
                if len(others) == 0:
                    break
                elif len(others) == 1:
                    merged_clusters.append(others)
                    break
                else:
                    cluster_adj_mat_nrom = cluster_adj_mat_nrom[others,:][:,others]
                    clust_lst_tmp = list(np.array(clust_lst_tmp)[others])
    
    return merged_clusters    


def merge_clusters_loop(adj_mat, cluster_label, loop = 1,
                        n_neighbors = 10, connectivity_thresh = 0.13, 
                        verbose = False):
    
    cluster_label_updated = copy.deepcopy(cluster_label)
    n_clusters = cluster_label_updated.max()+1
    
    for i in range(loop):
        clust_lst, clust_size = get_cluster_lst_and_sizes(cluster_label_updated)
        cluster_adj_mat = get_cluster_adj_mat_from_adj_dist(adj_mat, cluster_label_updated)
        clusters_new = merge_clusters(cluster_adj_mat, clust_size, n_neighbors = n_neighbors,
                                      connectivity_thresh = connectivity_thresh, verbose = verbose)
        
        cluster_map = {}
        for k, lst in enumerate(clusters_new):
            for c in lst:
                cluster_map[c] = k
        
        cluster_label_updated = np.array( [cluster_map[c] for c in cluster_label_updated] )
        # display(clusters_new)
        if (cluster_label_updated.max() + 1) == n_clusters:
            print('N_loops: %i' % (i+1))
            break
        else:
            n_clusters = cluster_label_updated.max()+1

    return cluster_label_updated


def set_cluster_for_others( ilst, labels, neighbors, distances ):
    
    label_lst = list(set(labels))
    label_lst.sort()
    label_array = np.array(labels)
    iary = np.array(ilst)
    label_all = np.full(neighbors.shape[0], -1)
    label_all[ilst] = labels
    
    b1 = label_all < 0
    nn = np.sum(b1)
    
    for j in range(neighbors.shape[1]):
        for k in range(len(label_all)):
            nlst = list(neighbors[b1,j])
            label_sel = label_all[nlst]
            label_all[b1] = label_sel

            b1 = label_all < 0
            # print(np.sum(b1))
            if (np.sum(b1) == 0) | (nn == np.sum(b1)):
                break
            else:
                nn = np.sum(b1)

        b1 = label_all < 0
        # print(np.sum(b1), 'AA')
        if (np.sum(b1) == 0):
            break
        
    return label_all


def clustering_subsample( X_vec, adj = None, neighbors = None, distances = None, clust_labels = None,
                          clust_algo = 'lv', N_clusters = 25, resolution = 1, N_neighbors = 10, 
                          n_cores = 4, N_cells_max = 10000 ): # mode='connectivity', 

    Ns = N_cells_max
    method = clust_algo
    
    if adj is None:
        adj = kneighbors_graph(X_vec, int(N_neighbors), mode = 'distance', # 'connectivity', 
                           include_self=False, n_jobs = 4)
    
    if (neighbors is None) | (distances is None):
        # start = time.time()
        neighbors, distances = get_neighbors(adj, N_neighbors)
        # lapsed = time.time() - start
        # print(lapsed, len(list(set(list(labels)))))
    
    if X_vec.shape[0] <= Ns:
        Xs = X_vec
        adj_sel = adj
    else:
        lst_full_array = np.arange(X_vec.shape[0])
        lst_full = list(lst_full_array)
        if clust_labels is None:
            lst_sel = random.sample(lst_full, k= Ns)
        else:
            clust_labels_array = np.array(clust_labels)
            clust_labels_lst = list(set(clust_labels))
            lst_sel = []
            r = Ns/X_vec.shape[0]
            for c in clust_labels_lst:
                b = clust_labels_array == c
                Nst = int(np.sum(b)*r)
                lst_full_c = list(lst_full_array[b])
                lst_sel_c = random.sample(lst_full_c, k= Nst)
                lst_sel = lst_sel + lst_sel_c

        for k in range(3):
            label_all = set_cluster_for_others( lst_sel, [0]*len(lst_sel), neighbors, distances )
            b = label_all < 0
            if np.sum(b) == 0:
                break
            elif (np.sum(b) < 20) | (k == 2):
                lst_sel2 = list(np.array(lst_full)[b])
                lst_sel = lst_sel + lst_sel2
                break;
            else:
                lst_sel2 = list(np.array(lst_full)[b])
                lst_sel2 = random.sample(lst_sel2, k= int(len(lst_sel2)*Ns/X_vec.shape[0]))
                lst_sel = lst_sel + lst_sel2

        Xs = X_vec[lst_sel,:]        
        adj_sel = adj[lst_sel,:][:,lst_sel]

    labels, obj, adj_tmp = clustering_alg(Xs, clust_algo = method, N_clusters = N_clusters, 
                                          resolution = resolution, N_neighbors = N_neighbors, 
                                          n_cores = n_cores, adj_dist = adj_sel) # mode = mode, 

    if X_vec.shape[0] > Ns:
        label_all = set_cluster_for_others( lst_sel, labels, neighbors, distances )
    else:
        label_all = labels

    return label_all, obj, adj

