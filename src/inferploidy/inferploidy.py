import warnings, math, time, copy, random, os
from contextlib import redirect_stdout, redirect_stderr
import logging, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cluster, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.decomposition import PCA, IncrementalPCA, TruncatedSVD
from scipy import stats
from scipy.sparse import csr_matrix, csc_matrix
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import NearestNeighbors
import sklearn.linear_model as lm
import anndata
from scipy.signal import medfilt

from inferploidy.clustering import get_neighbors, clustering_alg, merge_clusters_with_seed
from inferploidy.clustering import get_cluster_lst_and_sizes, get_cluster_adj_mat_from_adj_dist
from inferploidy.clustering import initially_detect_major_clusters, convert_adj_mat_dist_to_conn
from inferploidy.clustering import clustering_subsample, get_normalized_agg_adj_mat

CLUSTERING_AGO = 'lv'
SKNETWORK = True
try:
    from sknetwork.clustering import Louvain
except ImportError:
    print('WARNING: sknetwork not installed. GMM will be used for clustering.')
    CLUSTERING_AGO = 'gm'
    SKNETWORK = False

INFERCNVPY = True
try:
    import infercnvpy as cnv
except ImportError:
    print('ERROR: infercnvpy not installed. Please install infercnvpy first to run inferploidy.')
    INFERCNVPY = False

#'''
UMAP_INSTALLED = True
try:
    import umap
except ImportError:
    print('WARNING: umap-learn not installed.')
    UMAP_INSTALLED = False
#'''

MIN_ABS_VALUE = 1e-8
ANEUPLOID = 'Aneuploid'
DIPLOID = 'Diploid'
UNCLEAR = 'Unclear'

import warnings

def get_ref_clusters(cluster_label, ref_ind, ref_pct_min, 
                     ref_score = None, cs_quantile = 0.5):

    y_clust = cluster_label
    cnv_clust_lst = list(set(cluster_label))
    cnv_clust_lst.sort()

    b_inc = []
    ref_pct = []
    cs_ave = []
    
    b = ref_ind
    for idx in cnv_clust_lst:
        b = y_clust == idx
        bt = b & ref_ind
        cnt = np.sum(bt)

        if ref_score is not None: 
            cs_ave.append(ref_score[b].mean())
            
        ref_pct.append(cnt/np.sum(b))
        if (cnt >= ref_pct_min*np.sum(b)):
            b_inc.append(True)
        else:
            b_inc.append(False)

    df_tmp = pd.DataFrame( {'b_inc': b_inc, 'ref_pct': ref_pct }, 
                           index = cnv_clust_lst )
    
    if (ref_score is not None) & (cs_quantile > 0):
        b_tmp = df_tmp['b_inc'] == True
        qv = df_tmp.loc[b_tmp, 'cs_ave'].quantile(cs_quantile)
        b_tmp = df_tmp['cs_ave'] <= qv
        df_tmp.loc[b_tmp, 'b_inc'] = True
        # display(df_tmp)
        b_inc = list(df_tmp['b_inc'])

    if np.sum(b_inc) > 0:
        cluster_sel = list(np.array(cnv_clust_lst)[b_inc]) 
    else:
        odr = np.array(ref_pct).argsort()
        cluster_sel = [ cnv_clust_lst[odr[-1]] ]
        # print('ERROR: No reference cell types found.')
    
    return cluster_sel, df_tmp
    

def calculate_tumor_score_gmm( X_cnv_df, b_ind_sel, b_ind_merged, b_ind_others,
                               uc_margin = 0.2, z_th = 0, gmm_ncomp_n = 2, gmm_ncomp_t = 3,
                               reg_covar = 1e-3, cov_type = 'diag', aneup_wgt = 1):

    X_cnv = X_cnv_df
    
    b1 = b_ind_sel 
    b2 = b_ind_merged 
    b3 = b_ind_others 

    gmm_ref = mixture.GaussianMixture(n_components = int(gmm_ncomp_n), covariance_type = cov_type, 
                                      random_state = 0, reg_covar = reg_covar)
    gmm_ref.fit( X_cnv.loc[b1,:] )

    dist_from_ref = -gmm_ref.score_samples( X_cnv ) ## score_samples: log-likelihood
    dist_from_ref = pd.Series(dist_from_ref, index = X_cnv.index)
    
    b_stop = False
    if b3.sum() < 2:
        
        dist_from_other = dist_from_ref  
        maj_for_other = ( dist_from_ref - dist_from_other )
        
        ref_maj_mean = maj_for_other[b1].mean()
        ref_maj_sd = maj_for_other[b1].std() + 1e-10
        
        sf = (ref_maj_mean)/(ref_maj_sd)
        th_maj = ref_maj_mean + ref_maj_sd*sf
        uc_maj_lower = th_maj - ref_maj_sd*sf*uc_margin
        uc_maj_upper = th_maj + ref_maj_sd*sf*uc_margin
    
        ref_maj_mean = 0
        other_maj_mean = 0
        
        b_stop = True
        
    else:        
        gmm_other = mixture.GaussianMixture(n_components = int(gmm_ncomp_t), covariance_type = cov_type, 
                                            random_state = 0, reg_covar = reg_covar)
        gmm_other.fit( X_cnv.loc[b3,:] )
        
        dist_from_other = -gmm_other.score_samples( X_cnv ) ## score_samples: log-likelihood
        dist_from_other = pd.Series(dist_from_other, index = X_cnv.index)

        maj_for_other = ( dist_from_ref - dist_from_other )
        
        ref_maj_mean = maj_for_other[b1].mean()
        ref_maj_sd = maj_for_other[b1].std()
        
        other_maj_mean = maj_for_other[b3].mean()
        other_maj_sd = maj_for_other[b3].std()
        
        sf = (other_maj_mean - ref_maj_mean)/(ref_maj_sd + other_maj_sd)
        th_maj = ref_maj_mean + ref_maj_sd*sf*(1/aneup_wgt)
        uc_maj_lower = th_maj - ref_maj_sd*sf*uc_margin
        uc_maj_upper = th_maj + other_maj_sd*sf*uc_margin


    thresholds = {'th': th_maj, 'lower': uc_maj_lower, 'upper': uc_maj_upper}

    return maj_for_other, b_stop, sf, ref_maj_mean, other_maj_mean, thresholds


def get_connectivity_seq( cluster_adj_mat, clust_size, cluster_sel, n_neighbors = 14, 
                          net_search_mode = 'max', verbose = True ):

    cluster_adj_mat_sel = cluster_adj_mat[cluster_sel,:][:,cluster_sel]
    clust_size_sel = list(np.array(clust_size)[cluster_sel])
    
    cluster_adj_mat_nrom = get_normalized_agg_adj_mat( cluster_adj_mat_sel, clust_size_sel, 
                                                       n_neighbors = n_neighbors)
    
    j = cluster_adj_mat_nrom.sum(axis = 1).argmax()
    seed = [j] #  [cluster_sel[j]]
    
    merged_cluster_sel, added_sel, conns_sel = merge_clusters_with_seed( cluster_adj_mat_sel, 
                              clust_size_sel, seed = seed,
                              n_neighbors = n_neighbors,
                              connectivity_thresh = 0.0,  
                              net_search_mode = net_search_mode,
                              verbose = verbose)
    
    merged_cluster_sel = [cluster_sel[j] for j in merged_cluster_sel]
    added_sel = [cluster_sel[j] for j in added_sel]
    
    merged_cluster, added, conns = merge_clusters_with_seed( cluster_adj_mat, clust_size, 
                              seed = cluster_sel, n_neighbors = n_neighbors,
                              connectivity_thresh = 0.0,  
                              net_search_mode = net_search_mode,
                              verbose = verbose)
    
    merged_clusters =  merged_cluster_sel + merged_cluster
    added = added_sel + added
    conns = conns_sel + conns

    return conns, merged_clusters


def calculate_connectivity_threshold( connectivity_seq, no_cluster_sel, wgt = None,
                                      spf = 1/3, sd_mul = 23, conn_th_min = 0.05,
                                      conn_th_max = 0.3 ):

    conns = copy.deepcopy(connectivity_seq)
    
    p2 = int(no_cluster_sel)
    p1 = int(p2*spf)

    # print(p1, p2, len(conns))
    if p2 >= len(conns):
        conns_sel = np.array(conns[p1:])
        p2 = len(conns) - 1
        c_odr = np.arange(p1, len(conns))
    else:
        conns_sel = np.array(conns[p1:p2])
        c_odr = np.arange(p1, p2)
        
    # print(len(c_odr), len(conns_sel))
    z = np.polyfit(c_odr, conns_sel, 1, w = wgt)
    p = np.poly1d(z)

    conns_est = p(c_odr)
    conns_sd = sd_mul*np.sqrt( ((conns_est - conns_sel)**2).mean() )
    
    c_odr = np.arange(len(conns))
    conns_est = p(c_odr)
    
    if p[1] > 0:
        conns_est[:] = np.mean(conns_sel)
    
    conn_th = conns_est[p2] - conns_sd
    # conn_th = conns[p2]
    if conn_th < conn_th_min:
        conn_th = conn_th_min        
        # conns_sd = conns_est[p2] - conn_th
    elif conn_th > conn_th_max:
        conn_th = conn_th_max            

    return conn_th, conns_est, conns_sd


def update_cluster_sel(conns, conn_th, cluster_sel, cluster_seq, spf = 0.1 ):
    
    cluster_sel_new = copy.deepcopy(cluster_sel)
    N_cluster_sel = len(cluster_sel)
    update = 0

    if N_cluster_sel >= len(conns):
        update = 2
        ## No tumor cells
    else:    
        ## Check if any added clusters has connectivity below threshold
        min_conn2 = np.array(conns)[N_cluster_sel:].min()
        min_conn1 = np.array(conns)[int(spf*N_cluster_sel):N_cluster_sel].min()
        
        b = np.array(conns)[N_cluster_sel:] < conn_th
        if min_conn1 > min_conn2: # (np.sum(b) > 3):
            ## use conn_th and cluster_sel as is
            pass
        else:
            ## find the last cluster (in cluster_sel) that has its connectivity below threshold
            min_conn1_pos = np.array(conns)[int(spf*N_cluster_sel):N_cluster_sel].argmin() + int(spf*N_cluster_sel)
            if min_conn1_pos > N_cluster_sel*(1 - spf):
                last = min_conn1_pos
                for i in range(int(last)):
                    if conns[last-(i+1)] > conn_th:
                        break
                
                cluster_sel_new = cluster_seq[:(last - (i+2))]
                cluster_sel_new.sort()
                ## use conn_th as is     
                update = 1
            else:
                update = 0
                ## Skip
                pass

    return update, conn_th, cluster_sel_new


def find_num_clusters( N, Clustering_resolution = 1 ):
    return int(max(((N*(Clustering_resolution**2))**(1/6))*5, 10))


def pca_subsample(Xx, N_components_pca, N_cells_max_for_pca = 100000):
    
    pca = TruncatedSVD(n_components = int(N_components_pca)) # , algorithm = 'arpack')
    
    if Xx.shape[0] <= N_cells_max_for_pca:
        X_pca = pca.fit_transform(Xx)
    else:
        if isinstance(Xx, pd.DataFrame):
            lst_full = list(Xx.index.values)
            lst_sel = random.sample(lst_full, k= N_cells_max_for_pca)
            pca.fit(Xx.loc[lst_sel, :])
        else:
            lst_full = list(np.arange(Xx.shape[0]))
            lst_sel = random.sample(lst_full, k= N_cells_max_for_pca)
            pca.fit(Xx[lst_sel, :])
            
        # X_pca = Xx.dot(pca.components_.transpose()) 
        X_pca = pca.transform(Xx)
        
    return X_pca


def inferploidy( X_cnv, X_pca = None, adj_dist = None, ref_ind = None, ## should be provided
                 Clustering_algo = 'lv', Clustering_resolution = 6, 
                 ref_pct_min = 0.25, dec_margin = 0.2, dec_margin_adj = 0.5, 
                 n_neighbors = 14, N_loops = 5, N_runs = 7, 
                 n_cores = 4, connectivity_min = 0.18, connectivity_max = 0.32, 
                 net_search_mode = 'sum', spf = 0.1, connectivity_std_scale_factor = 1.5, 
                 plot_connection_profile = False, suffix = '', verbose = False, 
                 gmm_ncomp_n = 3, gmm_ncomp_t = 3, 
                 n_pca_comp = 15, use_umap = False, cs_comp_method = 0, cs_ref_quantile = 0, 
                 N_cells_max_for_clustering = 60000, N_cells_max_for_pca = 60000, 
                 N_clusters = 30, clust_labels = None, cnv_score = None, 
                 force_ref_to_diploid = True, reg_covar = 1e-6, cov_type = 'diag', 
                 print_prefix = '   ', log_lines = '', aneup_wgt = 1 ):

    connectivity_thresh_org = connectivity_min
    connectivity_thresh = connectivity_min
    refp_min = ref_pct_min
    
    uc_margin = dec_margin
    N_clusters = find_num_clusters( X_cnv.shape[0], Clustering_resolution )
    if verbose: 
        if (Clustering_algo != 'lv') & (Clustering_algo != 'cc'):
            print('Clustering using %s with N_clusters = %i. ' % (Clustering_algo.upper(), N_clusters))
    
    ## Remove all zero X_cnv
    X_cnv_mean = np.array(X_cnv.sum(axis = 1))
    b = X_cnv_mean == 0
    if np.sum(b) > 0:
        # print(np.sum(b))
        odr = np.array(X_cnv_mean).argsort()
        o_min = odr[int(np.sum(b))]
        x_cnv = X_cnv[o_min,:]
        idxs = np.arange(X_cnv.shape[0])[list(b)]
        for i in idxs:
            X_cnv[i,:] = x_cnv
            
    ref_addon = None
    score_key = 'ploidy_score' + suffix
    cluster_key = 'cnv_cluster' 
    
    ##########
    ## PCA ###
    start_time = time.time()
    start_time_a = start_time
    if verbose: 
        # print('Running inferPloidy .. ', end = '', flush = True)
        s = 'Running inferPloidy .. '
        
    
    if not isinstance(X_cnv, pd.DataFrame):
        X_cnv = pd.DataFrame(X_cnv)
    df = pd.DataFrame(index = X_cnv.index.values)
    
    if X_pca is None: 
        # X_pca = pca_obj.fit_transform(X_cnv)
        X_pca = pca_subsample(X_cnv, N_components_pca = n_pca_comp, 
                              N_cells_max_for_pca = N_cells_max_for_pca)
        
        etime = round(time.time() - start_time) 
        start_time = time.time()           
        
    X_vec = np.array(copy.deepcopy(X_pca))   
    
    ## Get neighbor lst and dst    
    clust_labels_all = clust_labels
    # adj_dist = None
    
    if adj_dist is None:
        adj_dist = kneighbors_graph(X_vec, int(n_neighbors), mode = 'distance', # 'connectivity', 
                           include_self=True, n_jobs = 4)
        
    neighbors, distances = get_neighbors(adj_dist, n_neighbors)

    ####################
    #### Outer Loop ####

    uc_lst = {}
    sf_lst = {}
    df_lst = {}
    res_lst = {}
    
    #'''
    if clust_labels_all is None:
        clust_labels_all, cobj, adj_dist = clustering_alg(X_vec, clust_algo = Clustering_algo, 
                                                        N_clusters = N_clusters, 
                                                        resolution = Clustering_resolution, 
                                                        N_neighbors = n_neighbors, 
                                                        n_cores = n_cores, adj_dist = adj_dist)  
    else:
        cobj = None

    etime = round(time.time() - start_time) 
    start_time = time.time()

    #######################
    ## Compute CNV score ##
    if cnv_score is not None:
        y_conf = np.array(cnv_score)*100
    else:
        if cs_comp_method <= 0:
            y_conf = (np.sqrt(X_cnv**2).mean(axis = 1))*100
        else:
            # y_conf = (np.sqrt(X_cnv**2).mean(axis = 1))*100
            q = cs_comp_method
            if q > 0.9: q = 0.9
            sq_X_cnv = X_cnv**2
            qv = sq_X_cnv[sq_X_cnv > 0].quantile(q).quantile(q)
            y_conf = np.log10( ((sq_X_cnv >= qv)).sum(axis = 1) + 1 )
        
    
    for orun in range(N_runs):
    
        cnv_clust_col = cluster_key
        tumor_dec_col = 'ploidy_dec'
        score_col = score_key
        ref_ind_col = 'cnvref_ind'
        
        uc_margin = dec_margin    
        uc_margin_max = 0.4
        if uc_margin > uc_margin_max:
            uc_margin = uc_margin_max
        if N_loops < 2:
            N_loops = 2
        
        a = np.arange(N_loops)
        b = (uc_margin_max-uc_margin)/(N_loops-1)
        uc_margin_lst = uc_margin_max - a*b
            
        ####################
        #### Inner Loop ####
        if orun > 0:
            clust_labels_all, cobj, adj_dist_t = clustering_alg( X_vec, clust_algo = Clustering_algo, 
                                                                N_clusters = N_clusters, 
                                                                resolution = Clustering_resolution, 
                                                                N_neighbors = n_neighbors, 
                                                                n_cores = n_cores, adj_dist = adj_dist,
                                                                random_state = orun)  

            clust_labels_all, cobj, adj_dist_t = clustering_subsample( X_vec, adj_dist, neighbors, distances, 
                                                     clust_labels = clust_labels_all,
                                                     clust_algo = Clustering_algo, N_clusters = N_clusters, 
                                                     resolution = Clustering_resolution, N_neighbors = n_neighbors, 
                                                     n_cores = n_cores,  
                                                     N_cells_max = int(X_vec.shape[0]*0.95) ) # N_cells_max_for_clustering )  

        ## Compute Cluster size, Aggregated Adj.Mat and Merge Small clusters     
        y_clust = copy.deepcopy( clust_labels_all )
        cnv_clust_lst, cluster_size = get_cluster_lst_and_sizes(y_clust)
        
        ## Generate agg_adj_mat
        cluster_adj_mat = get_cluster_adj_mat_from_adj_dist(adj_dist, y_clust)
        
        df_stat = None
        if ref_ind is None: 
            cluster_sel_org = initially_detect_major_clusters( cluster_adj_mat, cluster_size, 
                                             n_neighbors = n_neighbors, connectivity_thresh = connectivity_thresh, 
                                             net_search_mode = net_search_mode, verbose = verbose )
            ref_ind = pd.Series(y_clust, index = X_cnv.index).isin(list(cluster_sel_org)) 
            force_ref_to_diploid = False
        else:
            cluster_sel_org, df_stat = get_ref_clusters(y_clust, np.array(ref_ind), refp_min, 
                                           ref_score = y_conf, cs_quantile = cs_ref_quantile)
        
        thresholds = []    
        df = pd.DataFrame( index = X_cnv.index )
        df[cnv_clust_col] = list(y_clust)
        df[ref_ind_col] = list(ref_ind)
        df[tumor_dec_col] = 'Normal'
        df[score_col] = 0

        #######################################
        #### Adjust connectivity_threshold ####

        sf = 10
        z_th = 0
        cluster_sel = copy.deepcopy(cluster_sel_org)
        b_stop = False
        ref_ind_tmp = copy.deepcopy(ref_ind)
                
        conns, cluster_seq = get_connectivity_seq( cluster_adj_mat, cluster_size, cluster_sel, n_neighbors, 
                                                      net_search_mode = net_search_mode, verbose = False )
        # conns = medfilt(conns, 5 )
        N_cluster_sel = len(cluster_sel)            
        conn_th, conns_est, conns_sd = calculate_connectivity_threshold( conns, N_cluster_sel, spf = spf, 
                                                                         sd_mul = connectivity_std_scale_factor, 
                                                                         conn_th_min = connectivity_thresh_org,
                                                                         conn_th_max = connectivity_max )
        update = 3
        if False:  # conns_est[0] < conns_est[-1]:
            ## No tumor cells
            b_stop = True
            pass
        else:
            update, conn_th_new, cluster_sel_new = update_cluster_sel(conns, conn_th, cluster_sel, cluster_seq, spf = spf)
        
            # display(update)
            if update == 0:
                connectivity_thresh = conn_th_new
                pass
            elif update == 1:
                connectivity_thresh = conn_th
                cluster_sel = cluster_sel_new
                ref_ind_tmp = np.array( pd.Series(y_clust).isin(cluster_sel) )
            else:
                b_stop = True
            
        
        #### Adjust connectivity_threshold ####
        #######################################
        
        if b_stop: 
            N_loops = 0
            
        loop_cnt = 0    
        pct_uc = 100
        s_csel = ''
        for crun in range(N_loops):
    
            # start_time = time.time()
            ## Find seed clusters
            df_stat = None
            if crun > 0:
                cluster_sel = None
                refp_min_tmp = refp_min # min( refp_min + crun*0.01, 0.5 )
                if ref_ind_tmp is None: 
                    cluster_sel = initially_detect_major_clusters( cluster_adj_mat, cluster_size, 
                                                                   n_neighbors = n_neighbors, 
                                                                   connectivity_thresh = connectivity_thresh, 
                                                                   net_search_mode = net_search_mode, 
                                                                   verbose = verbose )
                else:
                    cluster_sel, df_stat = get_ref_clusters(y_clust, np.array(ref_ind_tmp), refp_min_tmp, 
                                                   ref_score = y_conf, cs_quantile = cs_ref_quantile)

            ## Find major group of clusters
            merged_cluster, added, connectivities = merge_clusters_with_seed( cluster_adj_mat, cluster_size, 
                                      seed = cluster_sel,
                                      n_neighbors = n_neighbors,
                                      connectivity_thresh = connectivity_thresh, 
                                      net_search_mode = net_search_mode,
                                      verbose = False)

            s_csel = s_csel + '%i-' % len(cluster_sel)
            cluster_added = list(set(merged_cluster) - set(cluster_sel))
            cluster_other = list(set(cnv_clust_lst) - set(merged_cluster))
            
            b0 = (df[cnv_clust_col].astype(int)).isin( cluster_sel_org )
            b1 = (df[cnv_clust_col].astype(int)).isin( cluster_sel )
            b2 = (df[cnv_clust_col].astype(int)).isin( cluster_added )
            b3 = (df[cnv_clust_col].astype(int)).isin( cluster_other )
                    
            maj_for_other, b_stop, sf, ref_maj_mean, other_maj_mean, th_dct = \
                calculate_tumor_score_gmm( X_cnv, b1, b2, b3, uc_margin = uc_margin, z_th = 0,
                                           gmm_ncomp_n = gmm_ncomp_n, gmm_ncomp_t = gmm_ncomp_t,
                                           reg_covar = reg_covar, cov_type = cov_type, aneup_wgt = aneup_wgt )
            
            th_maj = th_dct['th'] 
            uc_maj_lower = th_dct['lower']
            uc_maj_upper = th_dct['upper']
            
            loop_cnt += 1
            thresholds.append((th_maj, uc_maj_lower, uc_maj_upper))    
            
            df[score_col] = list(maj_for_other)

            ### Only for CNV estimates other than InferCNV ###
            score_min = df[score_col].min()
            bs = df[score_col] == score_min
            ##################################################

            p_tmp = df[tumor_dec_col].copy(deep = True)
            df[tumor_dec_col] = ANEUPLOID # 'Tumor'  
            
            if False: # b_stop | (crun == (N_loops - 1)):
                df.loc[b1, tumor_dec_col] = DIPLOID # 'Normal' ######

                lower_adj = (th_maj - uc_maj_lower)/4
                upper_adj = (uc_maj_upper - th_maj)/4
                print( 'Adj - L: %f, %f, U: %f, %f ' % (uc_maj_lower, lower_adj, uc_maj_upper, upper_adj) )
                
                b = maj_for_other <= uc_maj_lower + lower_adj # uc_maj_lower 
                df.loc[b&(~b1), tumor_dec_col] = DIPLOID # 'Normal' ######
                b = (maj_for_other < uc_maj_upper - upper_adj) & (maj_for_other > (uc_maj_lower + lower_adj))
                
                if force_ref_to_diploid:
                    df.loc[np.array(ref_ind), tumor_dec_col] = DIPLOID # 'Normal' ######       
            else:
                df.loc[b1, tumor_dec_col] = DIPLOID # 'Normal' ######
                
                b = maj_for_other <= uc_maj_lower 
                df.loc[b&(~b1), tumor_dec_col] = DIPLOID # 'Normal'   
                b = (maj_for_other < uc_maj_upper) & (maj_for_other > uc_maj_lower)
                df.loc[b&(~b1), tumor_dec_col] = 'Unclear'
                if force_ref_to_diploid:
                    df.loc[np.array(ref_ind), tumor_dec_col] = DIPLOID # 'Normal' ######                                

            ### Only for CNV estimates other than InferCNV ###
            if np.sum(bs) > 10:
                df.loc[bs&b1, tumor_dec_col] = DIPLOID  
                df.loc[bs&b2, tumor_dec_col] = 'Unclear'  
                df.loc[bs&b3, tumor_dec_col] = ANEUPLOID  
            ##################################################
                
            n_changed = (p_tmp != df[tumor_dec_col]).sum()
            if (n_changed == 0) | b_stop | (crun == (N_loops - 1)):
                
                df.loc[b1, tumor_dec_col] = DIPLOID # 'Normal' ######

                lower_adj = (th_maj - uc_maj_lower)*dec_margin_adj
                upper_adj = (uc_maj_upper - th_maj)*dec_margin_adj
                
                b = maj_for_other <= uc_maj_lower + lower_adj # uc_maj_lower 
                df.loc[b&(~b1), tumor_dec_col] = DIPLOID # 'Normal' ######
                b = (maj_for_other < uc_maj_upper - upper_adj) & (maj_for_other > (uc_maj_lower + lower_adj))
                
                if force_ref_to_diploid:
                    df.loc[np.array(ref_ind), tumor_dec_col] = DIPLOID # 'Normal' ######       
                
                ### Only for CNV estimates other than InferCNV ###
                if np.sum(bs) > 10:
                    df.loc[bs&b1, tumor_dec_col] = DIPLOID  
                    df.loc[bs&b2, tumor_dec_col] = 'Unclear'  
                    df.loc[bs&b3, tumor_dec_col] = ANEUPLOID  

            
            df['%s_%i' % (ref_ind_col, crun+1)] = 'Unreachables'
            df.loc[b1, '%s_%i' % (ref_ind_col, crun+1)] = 'Normal_Ref'
            df.loc[b2, '%s_%i' % (ref_ind_col, crun+1)] = 'Reachables'
            
            df['%s_%i' % (score_col, crun+1)] = list(maj_for_other)
            df['%s_%i' % (tumor_dec_col, crun+1)] = df[tumor_dec_col].copy(deep = True)
            
            ref_ind_tmp = (df[tumor_dec_col] == DIPLOID) # 'Normal') 
                
            pct_uc = 100*(df[tumor_dec_col] == 'Unclear').sum()/df.shape[0]
            if b_stop:
                pct_uc = 100
                break   
                
            if n_changed == 0:               
                break
        
        #### Inner Loop ####
        ####################
    
        if verbose: 
            if 'ref_maj_mean' not in locals():
                ref_maj_mean = 0
                other_maj_mean = 0

            if not b_stop:
                s = '   %i/%i' % (orun+1, N_runs)
                sa = s # '%s%s' % (print_prefix, s)
                print(sa, flush = True)                
            else:
                s = '   %i/%i' % (orun+1, N_runs)
                sa = s # '%s%s' % (print_prefix, s)
                # log_lines = log_lines + '%s\n' % s
                print(sa, flush = True)                
                
        if df is not None:
            uc_lst[str(orun+1)] = (pct_uc)
            sf_lst[str(orun+1)] = (sf)
            df_lst[str(orun+1)] = (df)
            res_lst[str(orun+1)] = { 'cluster_adj_matrix': cluster_adj_mat,
                                     'TN_decision_thresholds': thresholds,
                                     'TN_t_statistics': sf, 
                                     'unclear_pct': pct_uc,
                                     'tid_summary_df': df }
        
    #### Outer Loop ####
    ####################

    odr = np.array(list(sf_lst.values())).argsort()
    # odr = np.array(uc_lst).argsort()
    o = odr[0]
    key = list(sf_lst.keys())[o]

    df = df_lst[key]
    sf = sf_lst[key]
    pct_uc = uc_lst[key]

    df_a = pd.DataFrame(index = df.index)
    df_t = pd.DataFrame(index = df.index)
    df_s = pd.DataFrame(index = df.index)
    for k in df_lst.keys():
        df_t[k] = list(df_lst[k][tumor_dec_col])
        df_s[k] = list(df_lst[k][score_col])
        df_a[k] = list(df_lst[k]['%s_%i' % (ref_ind_col, 1)])
    
    maj = df_t.mode(axis = 1)
    b = maj[0].isna()
    maj.loc[b,0] = 'Unclear'
    
    df_t[tumor_dec_col] = list(maj[0])
    df_t[score_col] = list(df_s.mean(axis = 1))

    maj = df_a.mode(axis = 1)
    df_a['majority'] = list(maj[0])    
    df_t['init_group'] = list(maj[0])    
    
    # if verbose: print( 'Best run: %i with SF/UC: %5.3f, %4.2f ' % (o+1, sf, pct_uc) )
    
    etime = round(time.time() - start_time) 
    # if verbose: print('D(%i) .. ' % etime, end = '', flush = True) 

    summary = {}
    summary['connectivity_threshold'] = connectivity_thresh
    summary['selected run'] = key
    summary['run summary'] = res_lst

    etime = round(time.time() - start_time_a) 
        
    return df_t, summary, cobj, X_pca, adj_dist, df_a


import scanpy as sc
import pkg_resources
import collections

GTF_line = collections.namedtuple('GTF_line', 'chr, src, feature, start, end, score, strand, frame, attr, gid, gname, tid, tname, eid, biotype')
CHR, SRC, FEATURE, GSTART, GEND, SCORE, STRAND, FRAME, ATTR, GID, GNAME, TID, TNAME, EID, BIOTYPE = [i for i in range(15)]

def get_id_and_name_from_gtf_attr(str_attr):
    
    gid = ''
    gname = ''
    tid = ''
    tname = ''
    biotype = ''
    eid = ''
    
    items = str_attr.split(';')
    for item in items[:-1]:
        sub_item = item.strip().split()
        if sub_item[0] == 'gene_id':
            gid = sub_item[1].replace('"','')
        elif sub_item[0] == 'gene_name':
            gname = sub_item[1].replace('"','')
        elif sub_item[0] == 'transcript_id':
            tid = sub_item[1].replace('"','')
        elif sub_item[0] == 'transcript_name':
            tname = sub_item[1].replace('"','')
        elif sub_item[0] == 'exon_id':
            eid = sub_item[1].replace('"','')
        elif sub_item[0] == 'gene_biotype':
            biotype = sub_item[1].replace('"','')
        elif sub_item[0] == 'transcript_biotype':
            biotype = sub_item[1].replace('"','')
    
    return gid, gname, tid, tname, eid, biotype


def load_gtf( fname, verbose = True, ho = False ):
    
    gtf_line_lst = []
    hdr_lines = []
    if verbose: print('Loading GTF ... ', end='', flush = True)

    f = open(fname,'r')
    if ho:
        for line in f:
            
            if line[0] == '#':
                # line.replace('#','')
                cnt = 0
                for m, c in enumerate(list(line)):
                    if c != '#': break
                    else: cnt += 1
                hdr_lines.append(line[cnt:-1])
            else:
                break
    else:
        for line in f:
            
            if line[0] == '#':
                # line.replace('#','')
                cnt = 0
                for m, c in enumerate(list(line)):
                    if c != '#': break
                    else: cnt += 1
                hdr_lines.append(line[cnt:-1])
            else:
                items = line[:-1].split('\t')
                if len(items) >= 9:
                    chrm = items[0]
                    src = items[1]
                    feature = items[2]
                    start = int(items[3])
                    end = int(items[4])
                    score = items[5]
                    strand = items[6]
                    frame = items[7]
                    attr = items[8]
                    gid, gname, tid, tname, eid, biotype = get_id_and_name_from_gtf_attr(attr)
                    gl = GTF_line(chrm, src, feature, start, end, score, strand, frame, attr, gid, gname, tid, tname, eid, biotype)
                    gtf_line_lst.append(gl)
        
    f.close()
    if verbose: print('done %i lines. ' % len(gtf_line_lst))
    
    return(gtf_line_lst, hdr_lines)


def load_gtf_as_df( gtf_file ):
    gtf_line_lst, hdr_lines = load_gtf( gtf_file, verbose = False, ho = False )
    df_gtf = pd.DataFrame(gtf_line_lst)
    
    pcnt = df_gtf['gname'].value_counts()
    b = pcnt > 1
    gm = pcnt.index.values[b]
    
    df_gtf.set_index('gname', inplace = True)
    df_gtf = df_gtf[~df_gtf.index.duplicated(keep='first')]
    df_gtf = df_gtf.drop(columns = ['attr'])
    df_gtf = df_gtf.drop(index = gm)
    return df_gtf

def set_chrom_and_pos( adata, gtf_file ):

    df_gtf = load_gtf_as_df( gtf_file )  
    glst1 = list(adata.var.index.values)
    glst2 = list(df_gtf.index.values)
    glstc = list(set(glst1).intersection(glst2))

    adata.var['chromosome'] = ''
    adata.var['start'] = 0
    adata.var['end'] = 0
    adata.var['gene_name'] = ''
    adata.var['gene_id'] = ''
    
    adata.var.loc[glstc, 'chromosome'] = df_gtf.loc[glstc, 'chr']
    adata.var.loc[glstc, 'start'] = df_gtf.loc[glstc, 'start']
    adata.var.loc[glstc, 'end'] = df_gtf.loc[glstc, 'end']
    adata.var.loc[glstc, 'gene_name'] = list(df_gtf.loc[glstc].index.values)
    adata.var.loc[glstc, 'gene_id'] = df_gtf.loc[glstc, 'gid']
    
    return adata
    

def run_infercnv( adata, ref_key = None, ref_cat = None, 
                  gtf_file = None, species = 'hs', 
                  window_size = 100, n_cores = 4, log_transformed = False ):

    """
    Run infercnv py with the following parameters

    Parameters:
    adata: AnnData object for which the infercnvpy is run.
    ref_key: reference_key(a column name in adata.obs) passed to infercnvpy.
    ref_cat: reference_cat passed to infercnvpy. A list of categories in adata.obs[ref_key] to be used for indication of reference cells.
    gtf_file: GTF files passed to infercnvpy. If none, 'species' must be specified.
    species: must be either 'hs' (for human) or 'mm' (for mouse). If gtf_file is none, the function use the GTF files the package provides.
    window_size: It is the window size passed to infercnvpy.
    n_cores: The number of cores used to run infercnvpy.

    Returns:
    AnnData objects with 'cnv' added to adata.uns and 'X_cnv' to adata.obsm
    """
    
    if gtf_file is None:
        default_file_path = pkg_resources.resource_filename('inferploidy', 'default_optional_files')
        if species.lower() == 'hs':
            gtf_file = '%s/hg38_gene_only.gtf' % (default_file_path)
        elif species.lower() == 'mm':
            gtf_file = '%s/mm10_gene_only.gtf' % (default_file_path)
        else:
            print('ERROR: species must be either hs or mm.')
            return 

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        ref_ind = adata.obs[ref_key].isin(ref_cat)   
        adata.obs['cnv_ref_ind'] = ref_ind
        if np.sum(ref_ind) == 0:
            ref_ind_key = None
            ref_types = None
        else:
            ref_ind_key = 'cnv_ref_ind'
            ref_types = [True]
        
        adata_tmp = adata[:,:]
        if log_transformed:
            pass
        else:
            if 'log1p' not in list(adata.uns.keys()):
                sc.pp.normalize_total(adata_tmp, target_sum=1e4)
                sc.pp.log1p(adata_tmp, base = 2)
            elif adata_tmp.uns['log1p']['base'] is None:
                adata_tmp.X.data = adata_tmp.X.data /np.log(2)
            else:
                adata_tmp.X.data = adata_tmp.X.data * (np.log(adata_tmp.uns['log1p']['base'])/np.log(2))
        
        adata_tmp = set_chrom_and_pos( adata_tmp, gtf_file )

        cnv.tl.infercnv( adata_tmp, reference_key = ref_ind_key, reference_cat = ref_types, 
                            window_size = window_size, n_jobs = n_cores)

        for key in ['chromosome', 'start', 'end', 'gene_id', 'gene_name']:
            adata.var[key] = adata_tmp.var[key]
        adata.uns['cnv'] = adata_tmp.uns['cnv']
        adata.obsm['X_cnv'] = adata_tmp.obsm['X_cnv']
        adata.obs['cnv_ref_ind'] = adata.obs[ref_key].isin(ref_cat)
        
        return adata


def run_inferploidy( adata, X_cnv_key = 'X_cnv', 
                     ref_key = None, ref_cat = None, 
                     n_cores = 4, verbose = True, 
                     N_runs = 7,
                     n_pca_comp = 15, 
                     n_neighbors = 14,  
                     clustering_resolution = 5 ):

    """
    Run infercnv py with the following parameters

    Parameters:
    adata: AnnData object for which the inferploidy is run.
    X_cnv_key (string): The key to retrieve CNVs (matrix) from adata.obsm.
    ref_key (string): reference_key(a column name in adata.obs) passed to infercnvpy.
    ref_cat (list of strings): reference_cat passed to infercnvpy. A list of categories in adata.obs[ref_key] to be used for indication of reference cells.
    n_cores (int): The number of cores used to run infercnvpy.
    verbose (bool): level of verbosity (True or False)
    N_runs (int): The number of inferploidy component runs with difference clustering seeds.
    n_pca_comp (int): The number of PCA components for dimension reduction of CNV matrix.
    n_neighbors (int): The number of neighbors in the neighbor graph for Louvain clustering.
    clustering_resolution (positive real number): Clustering resolution for Louvain clustering.
    
    Returns:
    AnnData objects with 'cnv' added to adata.uns and 'X_cnv' to adata.obsm
    """
    
    ref_ind = None
    if isinstance(ref_key, str):
        if ref_key in list(adata.obs.columns.values):
            ref_ind = adata.obs[ref_key].isin(ref_cat)
            if np.sum(ref_ind) == 0:
                ref_ind = None

    if ref_ind is None:
        ref_cat = None
        s = 'WARNING: No reference cells exist -> InferCNV performed without reference.'
        print(s)
            
    X_cnv = np.array(adata.obsm[X_cnv_key].todense())
    
    s = 'InferPloidy .. ' 
    if verbose: 
        print(s, flush = True)
        
    start_time_t = time.time()    
    df_res, summary, cobj, X_pca, adj_dist, df_reachability_stat = \
         inferploidy(  X_cnv, ref_ind = ref_ind, 
                       N_runs = N_runs,
                       n_cores = n_cores,                      
                       n_pca_comp = n_pca_comp, 
                       n_neighbors = n_neighbors, 
                       Clustering_resolution = clustering_resolution, 
                       force_ref_to_diploid = True,
                       verbose = verbose, dec_margin = 0.2, 
                       connectivity_min = 0.2 )
    
    etime = round(time.time() - start_time_t) 
    if verbose: 
        s = 'InferPloidy .. done. (%i) ' % etime
        print(s, flush = True)

    adata.obsm['%s_pca' % X_cnv_key] = X_pca    
    if ref_ind is not None:
        adata.obs['cnv_ref_ind'] = list(ref_ind)
    else: 
        adata.obs['cnv_ref_ind'] = False
        
    adata.obs['iploidy_score'] = list(df_res['ploidy_score'])
    adata.obs['iploidy_dec'] = list(df_res['ploidy_dec'])
    adata.obs['iploidy_init_group'] = list(df_res['init_group'])

    rend = dict(zip(df_res.index.values, adata.obs.index.values))
    df_res.rename(index = rend, inplace = True)

    adata.obsm['inferploidy_results'] = df_res 
    adata.uns['inferploidy_summary'] = summary

    return adata
    
