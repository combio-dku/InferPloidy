import time, os, copy, datetime, math, random, warnings
import numpy as np
import pandas as pd
import anndata
from contextlib import redirect_stdout, redirect_stderr
import logging, sys
from scipy.sparse import csr_matrix, csc_matrix
import sklearn.linear_model as lm
import sklearn

from scoda.icnv import run_icnv, identify_tumor_cells, X_preprocessing, convert_adj_mat_dist_to_conn
from scoda.cpdb import cpdb_run, cpdb_get_results, cpdb4_run, cpdb4_get_results
from scoda.cpdb import cpdb_plot, cpdb_get_gp_n_cp #, plot_circ
from scoda.gsea import select_samples, run_gsa, run_prerank, run_gsea
from scoda.deg import deg_multi, get_population, plot_population, deg_multi_ext, deg_multi_ext_check_if_skip
from scoda.misc import plot_sankey_e, get_opt_files_path
from scoda.hicat import HiCAT, pca_subsample, X_variable_gene_sel

from scoda.hicat import HiCAT, get_markers_major_type
from scoda.hicat import get_markers_cell_type, get_markers_minor_type2, load_markers_all

GSEAPY = True
try:
    import gseapy as gp
except ImportError:
    print('WARNING: gseapy not installed or not available. ')
    GSEAPY = False

def load_gmt_file(file):
    dct = {}
    with open(file, 'r') as f:
        for line in f:
            items = line.split('\t')
            dct[items[0]] = items[2:]
    return dct

import collections

##########################################################################
## Functions and objects to handle GTF file
##########################################################################

# GTF_line = collections.namedtuple('GTF_line', 'chr, src, feature, start, end, score, strand, frame, attr')
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


def set_genomic_spot_no( adata_t, gtf_file, verbose = False ):

    ## Load GTF file
    gtf_lines, hdr_lines = load_gtf(gtf_file, verbose = verbose)
    df_gtf = pd.DataFrame(gtf_lines)
    
    # b = ~df_gtf['chr'].isin(['chrX', 'chrY', 'chrM'])
    # glst1 = list(set(list(df_gtf.loc[b, 'gname'])))
    glst1 = list(set(list(df_gtf['gname'])))
    glst2 = list(adata_t.var.index.values)
    glstc = list(set(glst1).intersection(glst2))
    
    b = df_gtf['gname'].isin(glstc)
    df_gtf = df_gtf.loc[b,:]
    df_gtf.set_index('gname', inplace = True)
    df_gtf = df_gtf[~df_gtf.index.duplicated(keep='first')]
    
    ## Set chromosome for the genes that are annotated in GTF file 
    ## Other genes not annotated are set '-'
    adata_t.var['chr'] = '-'
    adata_t.var.loc[glstc, 'chr'] = df_gtf.loc[glstc, 'chr']
    
    ## Get CNV info from InferCNV result
    n = adata_t.obsm['X_cnv'].shape[1]
    chr_pos = adata_t.uns['cnv']['chr_pos']
    df_chr_pos = pd.DataFrame( {'start': chr_pos.values()}, 
                               index = chr_pos.keys() )
    df_chr_pos.sort_values(by = 'start', inplace = True)
    endp = []
    for i in range(df_chr_pos.shape[0]):
        if i < (df_chr_pos.shape[0]-1):
            endp.append( df_chr_pos.iloc[i+1]['start'] )
        else:
            endp.append( n )
    df_chr_pos['end'] = endp
    df_chr_pos['len'] = df_chr_pos['end'] - df_chr_pos['start']
    
    ## Set genomic spot number in the GTF table.
    ## The spot_no for unspecified chromosome are set to -1
    clst = df_chr_pos.index.values.tolist()
    df_gtf['pos10'] = -1
    n = 0
    for c in clst:
        b = df_gtf['chr'] == c
        df_tmp = df_gtf.loc[b,:]
        df_tmp.sort_values(by = 'start', ascending = True, inplace = True)
        df_tmp['pos10'] = list(np.arange(np.sum(b)))
        L = (np.round(np.sum(b)/(df_chr_pos.loc[c, 'end'] - df_chr_pos.loc[c, 'start']),3))
        if verbose: print( 'L = %f ' % L )
        df_tmp['pos10'] = (df_tmp['pos10']/L).astype(int) + df_chr_pos.loc[c, 'start']
        idx = df_tmp.index.values.tolist()
        df_gtf.loc[idx,'pos10'] = df_tmp.loc[idx, 'pos10']
        # print( '%s: %i ' % (c, np.sum(b)) )
        n += int( np.round(np.sum(b)/L) )
    
    if verbose: print('L: %4.2f, n: %i, X_cnv.shape[1]: %i ' % (L, n, adata_t.obsm['X_cnv'].shape[1]))
    
    ## Use the GTF table to assign chromosome and genomic spot number to adata.var
    df_gtf['pos10'] = df_gtf['pos10'].astype(int)
    # adata_t.var['chr'] = 'chrMXY'
    adata_t.var['spot_no'] = -1
    genes = df_gtf.index.values.tolist()
    # adata_t.var.loc[genes, 'chr'] = df_gtf.loc[genes, 'chr']
    adata_t.var.loc[genes, 'spot_no'] = df_gtf.loc[genes, 'pos10']
    ## For the genes belonging to the chromosomes not shown in InferCNV results,
    ## the chromosome name will be '-' and the spot_no be -1

    return df_chr_pos


def scoda_icnv_addon( adata_t, gtf_file, ref_condition, ref_types, 
                      ref_key, use_ref_only, n_neighbors,
                      clustering_algo, clustering_resolution, 
                      connectivity_threshold, connectivity_threshold2, 
                      n_cores, verbose, print_prefix, 
                      tumor_dec_margin, uc_cor_margin, tumor_dec_th_max, 
                      N_loops, net_search_mode, cmd_cutoff, gcm, 
                      N_cells_max_for_clustering, N_cells_max_for_pca,
                      use_umap, n_pca_comp, ref_pct_min, use_cnv_score,
                      gmm_N_comp = 0, N_cells_max_for_gmmfit = 20000,
                      group_cell_size = 10000, cnv_window_size = 100, 
                      cs_comp_method = 0, cs_ref_quantile = 0.5,
                      cnv_filter_quantile = 0, cond_col = 'condition',
                      logreg_correction = True, split_run = False ):

    log_lines = ''
    adata = adata_t[:,:]
    if hasattr(adata_t, 'X_log1p'):
        adata.X_log1p = adata_t.X_log1p

    s = 'InferCNV .. ' 
    log_lines = log_lines + '%s\n' % s
    if verbose: 
        sa = '%s%s' % (print_prefix, s)
        print(sa, flush = True)

    ref_key2 = 'cnv_ref_ind'
    adata.obs[ref_key2] = False
    ref_ind = adata.obs[ref_key2]
    
    sx = ''
    if isinstance(ref_types, list):
        # if len(ref_types) > 0:
        #     ref_types2 = list(set(ref_types).intersection(list(adata.obs[ref_key].unique())))
        ref_ind = ref_ind | adata.obs[ref_key].isin(ref_types)

        if len(ref_types) > 0:
            sx = ref_types[0]
            if len(ref_types) > 1:
                for ct in ref_types[1:]:
                    sx = sx + ',%s' % ct

    if isinstance(ref_condition, str):
        ref_condition = [ref_condition]

    sy = ''
    if isinstance(ref_condition, list):
        ref_ind = ref_ind | adata.obs[cond_col].isin(ref_condition)
        if len(ref_condition) > 0:
            sy = ref_condition[0]
            if len(ref_condition) > 1:
                for ct in ref_condition[1:]:
                    sy = sy + ',%s' % ct

    s = '   using celltypes [%s] and conditions [%s] as normal references. (N_ref_cells: %i) ' % (sx, sy, np.sum(ref_ind))
    log_lines = log_lines + '%s\n' % s
    if verbose: 
        sa = '%s%s' % (print_prefix, s)
        print(sa, flush = True)
        
    if np.sum(ref_ind) == 0:
        ref_ind = None
        ref_types = None
        ref_types2 = ref_types
        s = 'WARNING: No reference cells exist -> InferCNV performed without reference.'
        log_lines = log_lines + '%s\n' % s
        print(s, flush = True)
    else:
        adata.obs[ref_key2] = ref_ind
        ref_types2 = [True]
        
    #''' 
    
    if clustering_resolution == 0:
        n_samples = len(adata.obs['sample'].unique())
        clustering_resolution = max( np.log2(n_samples), 1 )

    '''
    if 'X_cnv' in list(adata.obsm.keys()):
        del adata.obsm['X_cnv']
    if 'cnv' in list(adata.uns.keys()):
        del adata.uns['cnv']
    '''    
    adata = run_icnv(adata, ref_key2, ref_types2, gtf_file, 
                     clust_algo = 'lv', clust_resolution = clustering_resolution, 
                     N_pca = n_pca_comp, n_neighbors = n_neighbors,
                     cluster_key = 'cnv_cluster', scoring = True, # use_cnv_score, 
                     pca = False, N_cells_max_for_pca = N_cells_max_for_pca, 
                     window_size = cnv_window_size, n_cores = n_cores, 
                     cnv_filter_quantile = cnv_filter_quantile, verbose = verbose )

    '''
    X_cnv = np.array(adata.obsm['X_cnv'].todense())
    xv = np.abs(np.array(X_cnv)).std(axis = 0)   
    qv = cnv_filter_quantile
    odr = xv.argsort()
    X_cnv = X_cnv[:,odr[int(len(odr)*qv):]]    
    X_pca = pca_subsample( X_cnv, N_components_pca = n_pca_comp, 
                           N_cells_max_for_pca = N_cells_max_for_pca)
    adata.obsm['X_cnv_pca'] = X_pca
    '''
    
    # if verbose: print('%sInferCNVpy .. done. ' % (print_prefix), flush = True)
    #'''
    if adata is None:
        s = 'WARNING: InferCNV failed -> Skip tumor identification. '
        print(s, flush = True)
        log_lines = log_lines + '%s\n' % s        
        return log_lines
    else:
        s = 'InferCNV addon .. ' 
        log_lines = log_lines + '%s\n' % s
        sa = '%s%s' % (print_prefix, s)
        if verbose: print(sa, flush = True)
           
        X_cnv = np.array(adata.obsm['X_cnv'].todense())
        
        pca = False
        if use_cnv_score:
            cnv_score = adata.obs['cnv_score']
            # X_pca = adata.obsm['X_cnv_pca']            
            # adj_dist = adata.obsp['cnv_neighbor_graph_distance']
            clust_labels = np.array(list(adata.obs['cnv_cluster']))
            X_pca = X_cnv
            adj_dist = None
        else:
            cnv_score = None
            # X_pca = adata.obsm['X_cnv_pca']            
            # adj_dist = adata.obsp['cnv_neighbor_graph_distance']
            clust_labels = np.array(list(adata.obs['cnv_cluster']))
            X_pca = X_cnv
            adj_dist = None
            
            '''
            start_time = time.time()           
            X_pca = pca_subsample(X_cnv, N_components_pca = n_pca_comp, 
                                  N_cells_max_for_pca = N_cells_max_for_pca)

            etime = round(time.time() - start_time) 
            if verbose: print('P(%i) .. ' % etime, end = '', flush = True)
            #'''
            
        df_res, summary, cobj, X_pca, adj_dist = \
             identify_tumor_cells( X_cnv, ref_ind, X_pca = X_pca, adj_dist = adj_dist, clust_labels = clust_labels,
                                   cnv_score = cnv_score, Clustering_algo = clustering_algo, 
                                   Clustering_resolution = clustering_resolution, N_clusters = 30,
                                   gmm_N_comp = gmm_N_comp, th_max = tumor_dec_th_max, refp_min = ref_pct_min, p_exc = 0.1, 
                                   dec_margin = tumor_dec_margin, cor_margin = uc_cor_margin, n_neighbors = n_neighbors, 
                                   cmd_cutoff = cmd_cutoff, N_loops = N_loops, gcm = gcm, 
                                   plot_stat = False, use_ref = use_ref_only, 
                                   N_cells_max_for_clustering = N_cells_max_for_clustering,
                                   N_cells_max_for_pca = N_cells_max_for_pca,
                                   N_cells_max_for_gmmfit = N_cells_max_for_gmmfit,
                                   connectivity_thresh = connectivity_threshold, 
                                   connectivity_thresh2 = connectivity_threshold2,
                                   net_search_mode = net_search_mode, n_pca_comp = n_pca_comp, 
                                   use_umap = use_umap, cs_comp_method = cs_comp_method,
                                   cs_ref_quantile = cs_ref_quantile,
                                   suffix = '', Data = None, n_cores = n_cores, verbose = True )
        
        # if verbose: print('%sInferCNV addon .. done. ' % (print_prefix), flush = True)
        s = 'InferCNV addon .. done. '
        log_lines = log_lines + '%s\n' % s
        sa = '%s%s' % (print_prefix, s)
        if verbose: print(sa, flush = True)

        adj_conn = convert_adj_mat_dist_to_conn(adj_dist, threshold = 0)
        adata_t.obsp['cnv_neighbor_graph_distance'] = adj_dist
        adata_t.obsp['cnv_neighbor_graph_connectivity'] = adj_conn
        
        adata_t.obsm['X_cnv'] = adata.obsm['X_cnv']
        adata_t.obsm['X_cnv_pca'] = adata.obsm['X_cnv_pca'] 
        adata_t.uns['cnv'] = adata.uns['cnv']
        
        adata_t.uns['cnv_neighbors_info'] = {'connectivities_key': 'cnv_neighbor_graph_connectivity',
                                             'distances_key': 'cnv_neighbor_graph_distance',
                                             'params': {'n_neighbors': n_neighbors,
                                              'method': 'umap',
                                              'random_state': 0,
                                              'metric': 'euclidean',
                                              'use_rep': 'X_cnv_pca',
                                              'n_pcs': n_pca_comp}}        

        if use_cnv_score:
            adata_t.obs['cnv_score'] = list(adata.obs['cnv_score'])
        else:
            adata_t.obs['cnv_score'] = list(df_res['y_conf'])
            
        adata_t.obs['cnv_cluster'] = list(df_res['cnv_cluster'].astype(str))
        # adata_t.obs['cnv_score'] = list(df_res['tumor_score'])
        if ref_ind is not None:
            adata_t.obs['ref_ind'] = list(ref_ind)
        else: 
            adata_t.obs['ref_ind'] = False
            
        adata_t.obs['tumor_score'] = list(df_res['tumor_score'])
        adata_t.obs['tumor_dec'] = list(df_res['tumor_dec'])
            
        if ref_types is None:
            adata_t.uns['cnv_ref_celltypes'] = []
        else:
            adata_t.uns['cnv_ref_celltypes'] = ref_types
            
        # summary['clustering_obj'] = cobj
        summary['cnv_ref_conditions'] = ref_condition
        summary['cnv_ref_celltypes'] = ref_types
        adata_t.uns['cnv_addon_summary'] = summary

        lst1 = list(df_res.index.values)
        lst2 = list(adata_t.obs.index.values)
        rend = dict(zip(lst1, lst2))
        df_res.rename(index = rend, inplace = True)

        adata_t.obsm['cnv_addon_results'] = df_res 

        ## Set genomic spot ##
        df_chr_pos = set_genomic_spot_no( adata_t, gtf_file, verbose = False )        
        
        ## Use linear Classifier to correct tumor_decision
        if logreg_correction:
            
            y = adata_t.obs['tumor_dec'].astype(str)
            X = pd.DataFrame( adata_t.obsm['X_cnv_pca'], index = adata_t.obs.index )
            # X = pd.DataFrame( adata_t.obsm['X_pca'], index = adata_t.obs.index )
            
            b = adata_t.obs['tumor_dec'].isin(['Tumor', 'Normal'])
            y_train = y[b]
            X_train = X.loc[b,:]
            
            ## (1) Define model/set model parameters
            if sklearn.__version__ < '1.2':
                penalty_ = 'none'
            else:
                penalty_ = None
            
            classifier = lm.LogisticRegression(C = 1, #l1_ratio = None, 
                                               multi_class = 'multinomial',
                                               solver = 'saga', max_iter = 1000, 
                                               n_jobs = 6,
                                               penalty = penalty_, # 'elasticnet', 
                                               class_weight = 'balanced',
                                               verbose = 0) # verbose)
            
            classifier.fit(X_train,y_train)
            y_pred = classifier.predict(X)
    
            y_prob = classifier.predict_proba(X)
            y_prob = pd.DataFrame(y_prob, index = adata_t.obs.index)
            if uc_cor_margin > 0.25: uc_cor_margin = 0.25
            B = (y_prob > (0.5-uc_cor_margin)) & (y_prob < (0.5+uc_cor_margin))
            b = B.sum(axis = 1) > 0
            
            adata_t.obs['tumor_dec'] = list(y_pred)
            if uc_cor_margin > 0:
                adata_t.obs.loc[b, 'tumor_dec'] = 'unclear'

        '''        
        if ref_ind is not None: 
            b = adata_t.obs['ref_ind'] == True
            adata_t.obs.loc[b, 'tumor_dec'] = 'Normal'

        #'''
        adata_t.obs['celltype_minor_rev'] = adata_t.obs['celltype_minor'].copy(deep = True).astype(str)
        b = (adata_t.obs['tumor_dec'] == 'Tumor')
        if np.sum(b) > 0:
            b = b & (~adata_t.obs['ref_ind'])
            adata_t.obs.loc[b, 'celltype_minor_rev'] = 'Tumor cell'
        #'''
        return log_lines


def scoda_icnv_addon_split_run( adata_t, gtf_file, ref_condition, ref_types, 
                                ref_key, n_neighbors,  
                                clustering_algo, 
                                clustering_resolution, 
                                connectivity_threshold_min,  
                                connectivity_threshold_max,  
                                n_cores, verbose, print_prefix, 
                                tumor_dec_margin, N_tid_runs,
                                N_tid_loops, net_search_mode, 
                                N_cells_max_for_clustering, N_cells_max_for_pca,
                                n_pca_comp, ref_pct_min, use_cnv_score,
                                gmm_ncomp_n = 2, gmm_ncomp_t = 3, 
                                group_cell_size = 10000, cnv_window_size = 100, 
                                cs_comp_method = 0, cs_ref_quantile = 0.5,
                                cnv_filter_quantile = 0, cond_col = 'condition',
                                logreg_correction = False, split_run = False,
                                connectivity_std_scale_factor = 3, spf = 0.3,
                                plot_connection_profile = False,
                                cnv_suffix_to_use = None ):

    log_lines = ''
    adata = adata_t[:,:]
    if hasattr(adata_t, 'X_log1p'):
        adata.X_log1p = adata_t.X_log1p

    s = 'InferCNV .. ' 
    log_lines = log_lines + '%s\n' % s
    if verbose: 
        sa = '%s%s' % (print_prefix, s)
        print(sa, flush = True)

    ref_key2 = 'cnv_ref_ind'
    adata.obs[ref_key2] = False
    ref_ind = adata.obs[ref_key2]
    
    sx = ''
    if isinstance(ref_types, list):
        # if len(ref_types) > 0:
        #     ref_types2 = list(set(ref_types).intersection(list(adata.obs[ref_key].unique())))
        ref_ind = ref_ind | adata.obs[ref_key].isin(ref_types)

        if len(ref_types) > 0:
            sx = ref_types[0]
            if len(ref_types) > 1:
                for ct in ref_types[1:]:
                    sx = sx + ',%s' % ct

    if isinstance(ref_condition, str):
        ref_condition = [ref_condition]

    sy = ''
    if isinstance(ref_condition, list):
        ref_ind = ref_ind | adata.obs[cond_col].isin(ref_condition)
        if len(ref_condition) > 0:
            sy = ref_condition[0]
            if len(ref_condition) > 1:
                for ct in ref_condition[1:]:
                    sy = sy + ',%s' % ct

    s = '   using celltypes [%s] and conditions [%s] as normal references. (N_ref_cells: %i) ' % (sx, sy, np.sum(ref_ind))
    log_lines = log_lines + '%s\n' % s
    if verbose: 
        sa = '%s%s' % (print_prefix, s)
        print(sa, flush = True)
        
    if np.sum(ref_ind) == 0:
        ref_ind = None
        ref_types = None
        ref_types2 = ref_types
        s = 'WARNING: No reference cells exist -> InferCNV performed without reference.'
        log_lines = log_lines + '%s\n' % s
        print(s, flush = True)
    else:
        adata.obs[ref_key2] = ref_ind
        ref_types2 = [True]
        
    #''' 
    
    if clustering_resolution == 0:
        n_samples = len(adata.obs['sample'].unique())
        clustering_resolution = max( np.log2(n_samples), 1 )

    '''
    if 'X_cnv' in list(adata.obsm.keys()):
        del adata.obsm['X_cnv']
    if 'cnv' in list(adata.uns.keys()):
        del adata.uns['cnv']
    '''    

    if isinstance(cnv_suffix_to_use, str):
        cnv_to_use = 'X_cnv%s' % cnv_suffix_to_use
        if cnv_to_use in list(adata.obsm.keys()):
            s = 'INFO: %s to be used for tumor cell identification.' % cnv_to_use
            sa = '%s%s' % (print_prefix, s)
            if verbose: print(sa, flush = True)            
            pass
        else:
            s = 'WARNING: %s not found in obsm -> InferCNV performed.' % cnv_to_use
            sa = '%s%s' % (print_prefix, s)
            if verbose: print(sa, flush = True)            
            cnv_suffix_to_use = None
    else:
        cnv_suffix_to_use = None

    start_time_t = time.time()
    
    adata = run_icnv(adata, ref_key2, ref_types2, gtf_file, 
                     clust_algo = 'lv', clust_resolution = clustering_resolution, 
                     N_pca = n_pca_comp, n_neighbors = n_neighbors,
                     cluster_key = 'cnv_cluster', scoring = True, # use_cnv_score, 
                     pca = False, N_cells_max_for_pca = N_cells_max_for_pca, 
                     window_size = cnv_window_size, n_cores = n_cores, 
                     cnv_filter_quantile = cnv_filter_quantile, 
                     verbose = verbose, cnv_suffix_to_use = cnv_suffix_to_use )

    if cnv_suffix_to_use is None:
        X_cnv_key = 'X_cnv'
        X_cnv_pca_key = 'X_cnv_pca'
        cnv_score_key = 'cnv_score'
        cnv_key = 'cnv'
    else:
        X_cnv_key = 'X_cnv%s' % cnv_suffix_to_use
        X_cnv_pca_key = 'X_cnv_pca%s' % cnv_suffix_to_use
        cnv_score_key = 'cnv_score%s' % cnv_suffix_to_use
        cnv_key = 'cnv%s' % cnv_suffix_to_use
    
    '''
    X_cnv = np.array(adata.obsm['X_cnv'].todense())
    xv = np.abs(np.array(X_cnv)).std(axis = 0)   
    qv = cnv_filter_quantile
    odr = xv.argsort()
    X_cnv = X_cnv[:,odr[int(len(odr)*qv):]]    
    X_pca = pca_subsample( X_cnv, N_components_pca = n_pca_comp, 
                           N_cells_max_for_pca = N_cells_max_for_pca)
    adata.obsm['X_cnv_pca'] = X_pca
    '''

    etime = round(time.time() - start_time_t) 
    s = 'InferCNVpy done (%i). ' % (etime)
    if verbose: 
        print(s, flush = True)    
    start_time_t = time.time()
    
    # if verbose: print('%sInferCNVpy .. done. ' % (print_prefix), flush = True)
    #'''
    if adata is None:
        s = 'WARNING: InferCNV failed -> Skip tumor identification. '
        print(s, flush = True)
        log_lines = log_lines + '%s\n' % s        
        return log_lines
    else:
        X_cnv = np.array(adata.obsm[X_cnv_key].todense())
        adj_dist = adata.obsp['cnv_neighbor_graph_distance']
        X_pca = adata.obsm[X_cnv_pca_key]
        
        '''
        pca = False
        if use_cnv_score:
            cnv_score = adata.obs['cnv_score']
            X_pca = adata.obsm['X_cnv_pca']            
            adj_dist = adata.obsp['cnv_neighbor_graph_distance']
            clust_labels = np.array(list(adata.obs['cnv_cluster']))
        else:
            cnv_score = None
            X_pca = adata.obsm['X_cnv_pca']            
            adj_dist = adata.obsp['cnv_neighbor_graph_distance']
            clust_labels = np.array(list(adata.obs['cnv_cluster']))
        #'''

        clst_all = adata.obs.index.values.tolist()
        if split_run:
            clst_lst, n_cells_lst = group_cells( adata, group_cell_size, key = 'sample' ) 
        else:
            clst_lst = [clst_all]
            n_cells_lst = [len(clst_all)]
        
        s = 'InferCNV addon (%i) .. ' % len(clst_lst) 
        log_lines = log_lines + '%s\n' % s
        sa = '%s%s' % (print_prefix, s)
        if verbose: print(sa, flush = True)
            
        for k, clst in enumerate(clst_lst):
            adata_s = adata[clst,:]
               
            X_cnv_s = np.array(adata_s.obsm[X_cnv_key].todense())
            '''
            xv = np.abs(np.array(X_cnv_s)).std(axis = 0)   
            qv = cnv_filter_quantile
            odr = xv.argsort()
            X_cnv_s = X_cnv_s[:,odr[int(len(odr)*qv):]]    
            #'''
            if ref_ind is None:
                ref_ind_s = None
            else:
                ref_ind_s = ref_ind[clst]
                if np.sum(ref_ind_s) == 0:
                    ref_ind_s = None
            
            pca = False
            if use_cnv_score:
                cnv_score_s = adata_s.obs[cnv_score_key]
                X_pca_s = adata_s.obsm[X_cnv_pca_key]
                
                adj_dist_s = None # adata.obsp['cnv_neighbor_graph_distance']
                clust_labels_s = None # np.array(list(adata.obs['cnv_cluster']))
                if not split_run: 
                    adj_dist_s = adj_dist
                    clust_labels_s = None # np.array(list(adata.obs['cnv_cluster']))
            else:
                cnv_score_s = None
                X_pca_s = adata_s.obsm[X_cnv_pca_key]            
                adj_dist_s = None
                clust_labels_s = None
                if not split_run: 
                    adj_dist_s = adj_dist
                    clust_labels_s = None # np.array(list(adata.obs['cnv_cluster']))

            df_res_s, summary, cobj, X_pca_s, adj_dist_s = \
                 identify_tumor_cells( X_cnv_s, ref_ind = ref_ind_s, X_pca = X_pca_s, adj_dist = adj_dist_s, 
                                       clust_labels = clust_labels_s, cnv_score = cnv_score_s, 
                                       Clustering_algo = clustering_algo, 
                                       Clustering_resolution = clustering_resolution, N_clusters = 30, 
                                       ref_pct_min = ref_pct_min, 
                                       dec_margin = tumor_dec_margin, n_neighbors = n_neighbors, 
                                       N_loops = N_tid_loops, N_runs = N_tid_runs,
                                       N_cells_max_for_clustering = N_cells_max_for_clustering,
                                       N_cells_max_for_pca = N_cells_max_for_pca,
                                       connectivity_min = connectivity_threshold_min, 
                                       connectivity_max = connectivity_threshold_max, 
                                       net_search_mode = net_search_mode, n_pca_comp = n_pca_comp, 
                                       gmm_ncomp_n = gmm_ncomp_n, gmm_ncomp_t = gmm_ncomp_t, 
                                       use_umap = False, cs_comp_method = cs_comp_method,
                                       cs_ref_quantile = cs_ref_quantile, spf = spf,
                                       connectivity_std_scale_factor = connectivity_std_scale_factor,
                                       plot_connection_profile = plot_connection_profile, 
                                       suffix = '', n_cores = n_cores, verbose = True )
            
            lst1 = list(df_res_s.index.values)
            lst2 = list(adata_s.obs.index.values)
            rend = dict(zip(lst1, lst2))
            df_res_s.rename(index = rend, inplace = True)

            ## Use linear Classifier to correct tumor_decision               
            y = df_res_s['tumor_dec'].astype(str)
            X = pd.DataFrame( X_pca_s, index = adata_s.obs.index )
            # X = pd.DataFrame( adata_s.obsm['X_pca'], index = adata_s.obs.index )
            
            b = df_res_s['tumor_dec'].isin(['Tumor', 'Normal'])
            y_train = y[b]
            X_train = X.loc[b,:]
                
            if logreg_correction & (len(list(set(list(y_train)))) > 1):
                
                ## (1) Define model/set model parameters
                if sklearn.__version__ < '1.2':
                    penalty_ = 'none'
                else:
                    penalty_ = None
                
                classifier = lm.LogisticRegression(C = 1, #l1_ratio = None, 
                                                   multi_class = 'multinomial',
                                                   solver = 'saga', max_iter = 1000, 
                                                   n_jobs = 6,
                                                   penalty = penalty_, # 'elasticnet', 
                                                   class_weight = 'balanced',
                                                   verbose = 0) # verbose)
                
                classifier.fit(X_train,y_train)
                y_pred = classifier.predict(X)
        
                y_prob = classifier.predict_proba(X)
                y_prob = pd.DataFrame(y_prob, index = adata_s.obs.index)
                if uc_cor_margin > 0.25: uc_cor_margin = 0.25
                B = (y_prob > (0.5-uc_cor_margin)) & (y_prob < (0.5+uc_cor_margin))
                b = B.sum(axis = 1) > 0
                
                df_res_s['tumor_dec'] = list(y_pred)
                if uc_cor_margin > 0:
                    df_res_s.loc[b, 'tumor_dec'] = 'unclear'

            if k == 0:
                df_res = df_res_s
            else:
                df_res = pd.concat( [df_res, df_res_s], axis = 0 )

        df_res = df_res.loc[clst_all,:]

        ##################
        ## Save results ##
        
        etime = round(time.time() - start_time_t) 
        s = 'InferCNV addon .. done. (%i) ' % etime
        log_lines = log_lines + '%s\n' % s
        sa = '%s%s' % (print_prefix, s)
        if verbose: print(sa, flush = True)

        if not split_run: adj_dist = adj_dist_s
        adj_conn = convert_adj_mat_dist_to_conn(adj_dist, threshold = 0)
        adata_t.obsp['cnv_neighbor_graph_distance'] = adj_dist
        adata_t.obsp['cnv_neighbor_graph_connectivity'] = adj_conn
        
        adata_t.obsm[X_cnv_key] = adata.obsm[X_cnv_key]
        adata_t.obsm[X_cnv_pca_key] = adata.obsm[X_cnv_pca_key]
        adata_t.uns[cnv_key] = adata.uns[cnv_key]
        
        adata_t.uns['cnv_neighbors_info'] = {'connectivities_key': 'cnv_neighbor_graph_connectivity',
                                             'distances_key': 'cnv_neighbor_graph_distance',
                                             'params': {'n_neighbors': n_neighbors,
                                              'method': 'umap',
                                              'random_state': 0,
                                              'metric': 'euclidean',
                                              'use_rep': X_cnv_pca_key,
                                              'n_pcs': n_pca_comp}}        

        if use_cnv_score:
            adata_t.obs[cnv_score_key] = list(adata.obs[cnv_score_key])
        # else:
        #     adata_t.obs['cnv_score'] = list(df_res['y_conf'])
            
        # adata_t.obs['cnv_cluster'] = list(df_res['cnv_cluster'].astype(str))
        # adata_t.obs['cnv_score'] = list(df_res['tumor_score'])
        if ref_ind is not None:
            adata_t.obs['ref_ind'] = list(ref_ind)
        else: 
            adata_t.obs['ref_ind'] = False
            
        adata_t.obs['tumor_score'] = list(df_res['tumor_score'])
        adata_t.obs['tumor_dec'] = list(df_res['tumor_dec'])
            
        if ref_types is None:
            adata_t.uns['cnv_ref_celltypes'] = []
        else:
            adata_t.uns['cnv_ref_celltypes'] = ref_types
            
        # summary['clustering_obj'] = cobj
        summary['cnv_ref_conditions'] = ref_condition
        summary['cnv_ref_celltypes'] = ref_types
        adata_t.uns['cnv_addon_summary'] = summary

        lst1 = list(df_res.index.values)
        lst2 = list(adata_t.obs.index.values)
        rend = dict(zip(lst1, lst2))
        df_res.rename(index = rend, inplace = True)

        adata_t.obsm['cnv_addon_results'] = df_res 

        ## Set genomic spot ##
        df_chr_pos = set_genomic_spot_no( adata_t, gtf_file, verbose = False )        
        
        '''
        if ref_ind is not None: 
            b = adata_t.obs['ref_ind'] == True
            adata_t.obs.loc[b, 'tumor_dec'] = 'Normal'
        #'''
        adata_t.obs['celltype_minor_rev'] = adata_t.obs['celltype_minor'].copy(deep = True).astype(str)
        b = (adata_t.obs['tumor_dec'] == 'Tumor')
        if np.sum(b) > 0:
            b = b & (~adata_t.obs['ref_ind'])
            adata_t.obs.loc[b, 'celltype_minor_rev'] = 'Tumor cell'
        #'''
        return log_lines
        

import random

def group_cells_random( adata, group_cell_size, shuffle = True ):
    
    clst_all = list(adata.obs.index.values)
    if shuffle:
        random.shuffle(clst_all)
    clst_all = np.array(clst_all)
    
    n_cells = len(clst_all)
    n_batch = int( max( 1, np.round(n_cells/group_cell_size) ) )
    batch_size_new = int( np.round(n_cells/n_batch) )

    clst_lst = [] 
    n_cells_lst = []
    for i in range(n_batch):
        start = i*batch_size_new
        end = (i+1)*batch_size_new
        if i == (n_batch-1):
            end = n_cells
        clst_lst.append( list(clst_all[start:end]) )
        n_cells_lst.append((end - start))

    return clst_lst, n_cells_lst


def group_cells( adata, group_cell_size, key = 'sample', shuffle = True ):

    if key is None:
        return group_cells_random( adata, group_cell_size, shuffle = shuffle )

    elif key in adata.obs.columns.values.tolist():
        n_cells = adata.obs.shape[0]
        n_cells_per_loop = group_cell_size
        n_loop = int(np.round( n_cells/n_cells_per_loop ))
        pcnt = adata.obs[key].value_counts()
        n_samples = pcnt.shape[0]
        slst_all = pcnt.index.values.tolist()
        clst_all = adata.obs.index.values
        
        if (n_samples >= n_loop) & (n_cells > group_cell_size*1.5):
            clst_lst = []
            slst_lst = []
            n_cells_lst = []
            n_cells_lst2 = []
            cnt = 0
            slst = []
            while True:
                if len(slst_all) > 0: 
                    slst.append( slst_all.pop(0) )
                if len(slst_all) > 0:
                    slst.append( slst_all.pop(-1) )
                if (len(slst_all) > 0) & (pcnt.loc[slst_all].sum() < n_cells_per_loop/2):
                    slst = slst + slst_all
                    slst_all = []
                    
                if (pcnt.loc[slst].sum() >= n_cells_per_loop) | (len(slst_all) == 0):
                    slst_lst.append(slst)
                    n_cells_lst.append(pcnt.loc[slst].sum())
                    b = adata.obs[key].isin(slst)
                    clst = clst_all[b].tolist()
                    clst_lst.append( clst )
                    n_cells_lst2.append( len(clst) )
                    slst = []
                
                if len(slst_all) == 0: 
                    break
                    
        elif (n_cells > group_cell_size*1.5):
            clst_lst = []
            slst_lst = []
            n_cells_lst = []
            n_cells_lst2 = []
        
            cnt = 0
            while True:
                if len(clst_all) <= n_cells_per_loop*1.5:
                    clst = copy.deepcopy(clst_all)
                    clst_lst.append( clst )
                    n_cells_lst.append(len(clst))
                    clst_all = []
                else:
                    clst = clst_all[:n_cells_per_loop]
                    clst_lst.append( clst )
                    n_cells_lst.append(len(clst))
                    clst_all = clst_all[n_cells_per_loop:]
        
                if len(clst_all) == 0:
                    n_cells_lst2 = n_cells_lst
                    break
        
        else:
            slst_lst = [slst_all]
            clst_lst = [clst_all.tolist()]
            n_cells_lst = [len(clst_all)]
            
        return clst_lst, n_cells_lst

    else:
        return group_cells_random( adata, group_cell_size, shuffle = shuffle )


def scoda_hicat(adata, mkr_db, print_prefix, verbose = 1,
                clustering_resolution = 2, pct_th_cutoff = 0.5, 
                N_cells_max_for_pca = 80000, N_cells_max_for_gmm = 10000,
                target_tissues = [], target_cell_types = [], N_components_pca = 15,
                mkr_selector = '100000'):
    
    log_lines = ''
    s = 'Celltype annotation (using HiCAT) ..  ' 
    log_lines = log_lines + '%s\n' % s
    if verbose: 
        sa = '%s%s' % (print_prefix, s)
        print(sa, flush = True)
        
    # X1 = adata.to_df()
    # log_transformed = False

    clst_lst, n_cells_lst = group_cells( adata, N_cells_max_for_pca ) 
    clst_all = adata.obs.index.values.tolist()

    for k, clst in enumerate(clst_lst):
        adata_s = adata[clst,:]
        
        Xs = adata_s.X
        Xx = X_preprocessing( Xs, log_transformed = False )
        adata_s.X_log1p = Xx
        log_transformed = True
        X1 = pd.DataFrame( Xx.todense(), index = adata_s.obs.index, columns = adata_s.var.index )
        
        PNSH12 = mkr_selector
        to_exclude = [] 
    
        if len(clst_lst) > 1: 
            s = '   HiCAT %i/%i ' % (k+1, len(clst_lst))
            sa = '%s%s' % (print_prefix, s)
            log_lines = log_lines + '%s\n' % s
            print(sa, flush = True)
            
            pp = print_prefix + '      '
        else:
            pp = print_prefix + '   '
            
        df_pred_tmp, summary, var_genes, X_pca = \
            HiCAT( X1, mkr_db, log_transformed = log_transformed,
                   target_tissues = target_tissues, target_cell_types = target_cell_types, 
                   minor_types_to_exclude = to_exclude, mkr_selector = PNSH12, 
                   N_neighbors_minor = 31, N_neighbors_subset = 1,  
                   Clustering_algo = 'lv', Clustering_resolution = clustering_resolution, 
                   Clustering_base = 'pca', N_pca_components = N_components_pca, 
                   N_cells_max_for_pca = N_cells_max_for_pca, 
                   N_cells_max_for_gmm = N_cells_max_for_gmm, cycling_cell = False, 
                   copy_X = False, verbose = verbose, print_prefix = pp,
                   model = 'gmm', N_gmm_components = 20, cbc_cutoff = 0.01,
                   Target_FPR = 0.05, pval_th = 0.05, pval_th_subset = 1, 
                   pmaj = 0.7, pth_fit_pnt = 0.4, pth_min = pct_th_cutoff, min_logP_diff = 1, 
                   use_minor = True, minor_wgt = 0, use_union = False, use_union_major = True,
                   use_markers_for_pca = False, comb_mkrs = False, 
                   knn_type = 1, knn_pmaj = 0.3, N_max_to_ave = 2,
                   thresholding_minor = False, thresholding_subset = False )

        if k == 0:
            df_pred = df_pred_tmp
        else:
            df_pred = pd.concat([df_pred, df_pred_tmp], axis = 0)

    df_pred = df_pred.loc[clst_all, :]

    adata.obs['celltype_major'] = df_pred['cell_type_major']
    adata.obs['celltype_minor'] = df_pred['cell_type_minor']
    adata.obs['celltype_subset'] = df_pred['cell_type_subset']

    adata.obsm['HiCAT_result'] = df_pred
    
    adata.var['variable_genes'] = False
    genes_org = list(adata.var.index.values)
    genes_new = [g.upper() for g in genes_org]
    rdct = dict(zip(genes_new, genes_org))
    var_genes = list(pd.Series(var_genes).map(rdct))
    adata.var.loc[var_genes, 'variable_genes'] = True

    if len(clst_lst) > 1:
        Xs = adata.X
        Xx = X_preprocessing( Xs, log_transformed = False )
        adata.X_log1p = Xx
        log_transformed = True
        Xx = pd.DataFrame( Xx.todense(), index = adata.obs.index, columns = adata.var.index )
        Xx = X_variable_gene_sel( Xx, N_genes = 2000, N_cells_max = N_cells_max_for_pca, vg_sel = True )
        X_pca = pca_subsample(Xx, N_components_pca, N_cells_max_for_pca = N_cells_max_for_pca) 
        
    adata.obsm['X_pca'] = np.array(X_pca)
    
    ct_lst = list(adata.obs['celltype_major'].unique())
    ct_lst.sort()
    
    ct = ct_lst[0]
    b = adata.obs['celltype_major'] == ct
    ss = '%s(%i)' % (ct, np.sum(b))
    for ct in ct_lst[1:]:
        b = adata.obs['celltype_major'] == ct
        ss = ss + ',%s(%i)' % (ct, np.sum(b))
        # if (len(ss.split('\n')[-1]) > 65) | (len(ss.split('\n')[-1]) > 180):
        #     ss = ss + '\n      '
        
    s = '   %i major-types identified: %s' % (len(ct_lst), ss)
    log_lines = log_lines + '%s\n' % s
    sa = '%s%s' % (print_prefix, s)
    print(sa)
    
    s = 'Celltype annotation (using HiCAT) .. done.  ' 
    log_lines = log_lines + '%s\n' % s

    if verbose: 
        sa = '%s%s' % (print_prefix, s)
        print(sa, flush = True)
    
    ## Store marker info.
    mkr_dict, mkr_dict_neg, mkr_dict_sec, major_dict, minor_dict = load_markers_all(mkr_db, target_cells = [], 
                                                                                 pnsh12 = '111000', 
                                                                                 comb_mkrs = False, to_upper = False)
    #'''
    mkr_dict, mkr_dict_neg = \
    get_markers_minor_type2(mkr_db, target_cells = [], 
                            pnsh12 = '100000', comb_mkrs = False, 
                            rem_common = False, verbose = False, to_upper = False)
    #'''

    mkr_info = {}
    if isinstance(mkr_db, pd.DataFrame):
        mkr_info['marker_db'] = mkr_db
    elif isinstance(mkr_db, str):
        mkr_info['marker_db'] = pd.read_csv(mkr_db, sep = '\t')
        
    mkr_info['subset_markers_dict'] = mkr_dict
    mkr_info['subset_to_major_map'] = major_dict
    mkr_info['subset_to_minor_map'] = minor_dict
    adata.uns['Celltype_marker_DB'] = mkr_info
    
    return log_lines


def scoda_hicat_one_shot(adata, mkr_db, print_prefix, verbose = 1,
                         clustering_resolution = 2, pct_th_cutoff = 0.5, 
                         N_cells_max_for_pca = 60000, N_cells_max_for_gmm = 10000,
                         target_tissues = [], target_cell_types = [],
                         mkr_selector = '100000'):
    
    log_lines = ''
    s = 'Celltype annotation (using HiCAT) ..  ' 
    log_lines = log_lines + '%s\n' % s
    if verbose: 
        sa = '%s%s' % (print_prefix, s)
        print(sa, flush = True)
        
    # X1 = adata.to_df()
    # log_transformed = False
    
    Xs = adata.X
    Xx = X_preprocessing( Xs, log_transformed = False )
    adata.X_log1p = Xx
    log_transformed = True
    X1 = pd.DataFrame( Xx.todense(), index = adata.obs.index, columns = adata.var.index )
    
    PNSH12 = mkr_selector
    to_exclude = [] 

    # if verbose: print('%sCelltype annotation (using HiCAT) .. ' % (print_prefix), flush = True)
    df_pred, summary, var_genes, X_pca = \
        HiCAT( X1, mkr_db, log_transformed = log_transformed,
               target_tissues = target_tissues, target_cell_types = target_cell_types, 
               minor_types_to_exclude = to_exclude, mkr_selector = PNSH12, 
               N_neighbors_minor = 31, N_neighbors_subset = 1,  
               Clustering_algo = 'lv', Clustering_resolution = clustering_resolution, 
               Clustering_base = 'pca', N_pca_components = 15, 
               N_cells_max_for_pca = N_cells_max_for_pca, 
               N_cells_max_for_gmm = N_cells_max_for_gmm, cycling_cell = False, 
               copy_X = False, verbose = verbose, print_prefix = print_prefix + '   ',
               model = 'gmm', N_gmm_components = 20, cbc_cutoff = 0.01,
               Target_FPR = 0.05, pval_th = 0.05, pval_th_subset = 1, 
               pmaj = 0.7, pth_fit_pnt = 0.4, pth_min = pct_th_cutoff, min_logP_diff = 1, 
               use_minor = True, minor_wgt = 0, use_union = False, use_union_major = True,
               use_markers_for_pca = False, comb_mkrs = False, 
               knn_type = 1, knn_pmaj = 0.3, N_max_to_ave = 2,
               thresholding_minor = False, thresholding_subset = False )

    adata.obs['celltype_major'] = df_pred['cell_type_major']
    adata.obs['celltype_minor'] = df_pred['cell_type_minor']
    adata.obs['celltype_subset'] = df_pred['cell_type_subset']

    adata.obsm['HiCAT_result'] = df_pred
    adata.uns['HiCAT_summary'] = summary
    
    adata.var['variable_genes'] = False
    genes_org = list(adata.var.index.values)
    genes_new = [g.upper() for g in genes_org]
    rdct = dict(zip(genes_new, genes_org))
    var_genes = list(pd.Series(var_genes).map(rdct))
    adata.var.loc[var_genes, 'variable_genes'] = True
    
    adata.obsm['X_pca'] = np.array(X_pca)
    
    ct_lst = list(adata.obs['celltype_major'].unique())
    ct_lst.sort()
    
    ct = ct_lst[0]
    b = adata.obs['celltype_major'] == ct
    ss = '%s(%i)' % (ct, np.sum(b))
    for ct in ct_lst[1:]:
        b = adata.obs['celltype_major'] == ct
        ss = ss + ',%s(%i)' % (ct, np.sum(b))
        # if (len(ss.split('\n')[-1]) > 65) | (len(ss.split('\n')[-1]) > 180):
        #     ss = ss + '\n      '
        
    s = '   %i major-types identified: %s' % (len(ct_lst), ss)
    log_lines = log_lines + '%s\n' % s
    sa = '%s%s' % (print_prefix, s)
    print(sa)
    
    s = 'Celltype annotation (using HiCAT) .. done.  ' 
    log_lines = log_lines + '%s\n' % s

    if verbose: 
        sa = '%s%s' % (print_prefix, s)
        print(sa, flush = True)
    
    ## Store marker info.
    mkr_dict, mkr_dict_neg, mkr_dict_sec, major_dict, minor_dict = load_markers_all(mkr_db, target_cells = [], 
                                                                                 pnsh12 = '111000', 
                                                                                 comb_mkrs = False, to_upper = False)
    #'''
    mkr_dict, mkr_dict_neg = \
    get_markers_minor_type2(mkr_db, target_cells = [], 
                            pnsh12 = '100000', comb_mkrs = False, 
                            rem_common = False, verbose = False, to_upper = False)
    #'''

    mkr_info = {}
    if isinstance(mkr_db, pd.DataFrame):
        mkr_info['marker_db'] = mkr_db
    elif isinstance(mkr_db, str):
        mkr_info['marker_db'] = pd.read_csv(mkr_db, sep = '\t')
        
    mkr_info['subset_markers_dict'] = mkr_dict
    mkr_info['subset_to_major_map'] = major_dict
    mkr_info['subset_to_minor_map'] = minor_dict
    adata.uns['Celltype_marker_DB'] = mkr_info
    
    return log_lines


def scoda_cci( adata_t, cpdb_path, cond_col, sample_col, 
               cci_base, unit_of_cci_run, min_n_cells_for_cci, 
               data_dir, pval_max, mean_min, Rth, n_cores,
               print_prefix, verbose = False, cpdb_version = 4 ):
    
    log_lines = ''
    
    s = 'CellphoneDB .. ' # % (print_prefix)
    log_lines = log_lines + '%s\n' % s
    if verbose: 
        sa = '%s%s' % (print_prefix, s)
        print(sa, flush = True)
        
    if cci_base in list(adata_t.obs.columns.values):
        celltype_col = cci_base
    else:
        celltype_col = 'celltype_minor'

    if cond_col not in list(adata_t.obs.columns.values):
        adata_t.obs[cond_col] = 'Not specified'

        s = '   \'condition\' column not specified. Will be set as \'unspecied\''         
        log_lines = log_lines + '%s\n' % s
        if verbose: 
            sa = '%s%s' % (print_prefix, s)
            print(sa, flush = True)
            
    cond_lst = list(adata_t.obs[cond_col].unique())
    
    if sample_col not in list(adata_t.obs.columns.values):
        adata_t.obs[sample_col] = 'Not specified'
            
        s = '   \'sample\' column not specified. Will be set as \'unspecied\'' # % print_prefix)
        log_lines = log_lines + '%s\n' % s
        if verbose: 
            sa = '%s%s' % (print_prefix, s)
            print(sa, flush = True)
            
    # print(cond_lst)
    # print(sample_lst)
    t_col = cond_col
    t_lst = cond_lst

    if unit_of_cci_run is None: unit_of_cci_run = sample_col
        
    if unit_of_cci_run in list(adata_t.obs.columns.values): # == 'sample':        
        sample_lst = list(adata_t.obs[unit_of_cci_run].unique())
        t_col = unit_of_cci_run
        t_lst = sample_lst
        s = '   Unit of CCI RUN is %s ' % unit_of_cci_run # % print_prefix)        
        log_lines = log_lines + '%s\n' % s
        if verbose: 
            sa = '%s%s' % (print_prefix, s)
            print(sa, flush = True)
    else:
        s = 'ERROR: %s not in obs. '         
        log_lines = log_lines + '%s\n' % s
        if verbose: 
            sa = '%s%s' % (print_prefix, s)
            print(sa, flush = True)
        return
        
    df_cnt, df_pct= get_population( adata_t.obs[t_col], 
                                    adata_t.obs[celltype_col], sort_by = [] )

    ## Filter celltype with its number is below the minimum value
    N_cells_min = min_n_cells_for_cci
    
    ## Set output dir
    cpdb_dir = data_dir + '/cpdb'
    if not os.path.isdir(cpdb_dir):
        os.mkdir(cpdb_dir)

    dfv_dct_all = {}
    dfv_dct = {}
    for k, s in enumerate(t_lst):

        # if verbose: print('%s   %2i/%2i - %s' % (print_prefix, k+1, len(t_lst), s))

        sx = '   %2i/%2i' % (k+1, len(t_lst))
        log_lines = log_lines + '%s\n' % sx
        if verbose: 
            sa = '%s%s' % (print_prefix, sx)
            print(sa, flush = True)

        ## If s contains ''/'', directory cannot be created properly
        ss = s.replace('/', '-')
        out_dir = cpdb_dir + '/CPDB_res_%s' % (ss)

        celltype_all = df_cnt.columns.values
        b = df_cnt.loc[s,:] >= N_cells_min
        celltype_lst = list(celltype_all[b])
        
        b1 = adata_t.obs[t_col] == s
        b2 = adata_t.obs[celltype_col].isin(celltype_lst)
        b = b1 & b2

        if (np.sum(b) >= N_cells_min) & (len(celltype_lst) > 1):
            
            adata_s = adata_t[b,:]
            celltype = adata_s.obs[celltype_col]
    
            if cpdb_version == 4:
                df_mn, df_pv = cpdb4_run( adata_s, celltype, db = cpdb_path,
                                          out_dir = out_dir, n_cores = n_cores,
                                          threshold = None, gene_id_type = 'gene_name', 
                                          verbose = verbose )
    
                dfi, dfp, dfm, dfv = cpdb4_get_results( df_pv, df_mn, 
                                                        pval_max = 1,  # pval_max, 
                                                        mean_min = 0 ) # mean_min )
            else:
                '''
                cpdb_run( df_cell_by_gene, cell_types, out_dir,
                          gene_id_type = 'gene_name', db = None, 
                          n_iter = None, pval_th = None, threshold = None, 
                          verbose = False)
                #'''
                    
                cpdb_run( adata_s.to_df(), celltype, out_dir = out_dir, 
                          gene_id_type = 'gene_name', threshold = None, 
                          db = None, verbose = verbose )    
    
                if os.path.isdir(out_dir):
                    dfi, dfp, dfm, dfv = cpdb_get_results( out_dir, 
                                                           pval_max = 1,  # pval_max, 
                                                           mean_min = 0 ) #mean_min )
                
            b = (dfv['pval'] <= pval_max) & (dfv['mean'] >= mean_min) 
            dfv_sel = dfv.loc[b, :]
            
            dfv_dct_all[s] = dfv
            dfv_dct[s] = dfv_sel
        ## End for
            
    if unit_of_cci_run == cond_col:       
        adata_t.uns['CCI'] = dfv_dct
    else: 
        # adata_t.uns['CCI_all'] = dfv_dct_all
        
        s2c_map = {}
        group_lst = []
        for k, s in enumerate(sample_lst):
            b = adata_t.obs[sample_col] == s
            cond = list(adata_t.obs.loc[b, cond_col])[0]
            s2c_map[s] = cond
            group_lst.append(cond)

        ## Load CPDB results, convert them to suitable format
        dfv_per_group = {}
        for s, g in zip(sample_lst, group_lst):
            if s in list(dfv_dct.keys()):
                dfv = dfv_dct[s]            
                if g in dfv_per_group.keys():
                    dfv_per_group[g].append(dfv)
                else:
                    dfv_per_group[g] = [dfv]    
                    
        ## Filter CCIs
        ## Get all gg--cc indices: idx_lst_all

        rth = Rth

        idx_lst_all = []
        idx_dct = {}
        df_dct = {}

        for g in dfv_per_group.keys():
            lst = dfv_per_group[g]

            ## Get list of gg--cc indices (idx_lst) that meets the requirement
            idx_lst = []
            for df in lst:
                idx_lst = idx_lst + list(df.index.values)
                
            idx_lst = list(set(idx_lst))
            # print(len(idx_lst))
            idx_lst.sort()

            ## For each gg--cc index, 
            ## Get # of samples where gg--cc interaction detected
            ps = pd.Series(0, index = idx_lst)

            for df in lst:
                ps[df.index.values] += 1

            ## Get Union of gg--cc indices from all groups
            b = ps >= len(lst)*rth
            idxs = list(ps.index.values[b])
            idx_dct[g] = idxs
            idx_lst_all = idx_lst_all + idxs

            s = 'Group: %s (Ns = %i) - N_valid_interactions: %i among %i.' % (g, len(lst), np.sum(b), len(b))
            log_lines = log_lines + '%s\n' % s
            if verbose: 
                sa = '%s%s' % (print_prefix, s)
                print(sa, flush = True)

            ## Get combined dfv
            ## pval = max(pvals)
            ## mean = mean(means)

            dfv = pd.DataFrame(index = idxs, columns = df.columns)
            dfv['pval'] = 0
            dfv['mean'] = 0
            for k, df in enumerate(lst):

                idxt = list(set(idxs).intersection(list(df.index.values)))
                cols = list(df.columns.values[:-2])
                dfv.loc[idxt, cols] = df.loc[idxt, cols]
                dfv.loc[idxt, 'mean'] = dfv.loc[idxt, 'mean'] + df.loc[idxt, 'mean']
                dft = pd.DataFrame(index = idxt)
                dft['pv1'] = dfv.loc[idxt, 'pval']
                dft['pv2'] = df.loc[idxt, 'pval']
                dfv.loc[idxt, 'pval'] = dft.max(axis = 1)

            dfv['mean'] = dfv['mean']/len(lst) 
            df_dct[g] = dfv

        idx_lst_all = list(set(idx_lst_all))
            
        s = 'Number of Interactions found: %i ' % len(idx_lst_all)    
        log_lines = log_lines + '%s\n' % s
        if verbose: 
            sa = '%s%s' % (print_prefix, s)
            print(sa, flush = True)
        
        adata_t.uns['CCI'] = df_dct
        adata_t.uns['CCI_sample'] = dfv_dct

        '''
        ## Get union of CCIs for all samples
        ## Extend dfv_dct using dfv_dct_all
        cci_lst = []
        lst = list(dfv_dct.keys())
        for j, k in enumerate(lst):
            cci_lst = list(set(cci_lst).union(list(dfv_dct[k].index.values)))
        Lo = len(cci_lst)
            
        lst = list(dfv_dct_all.keys())
        for j, k in enumerate(lst):
            cci_lst = list(set(cci_lst).intersection(list(dfv_dct_all[k].index.values)))
        if len(cci_lst) < Lo:
            print('WARNING: CCI %i < %i ' % (len(cci_lst), Lo))

        dfv_dct_r = {}
        lst = list(dfv_dct_all.keys())
        for j, k in enumerate(lst):
            dfv_dct_r[k] = dfv_dct_all[k].loc[cci_lst, :]
        adata_t.uns['CCI_sample'] = dfv_dct_r
        '''
        
    s = 'CellphoneDB .. done. ' 
    log_lines = log_lines + '%s\n' % s
    if verbose: 
        sa = '%s%s' % (print_prefix, s)
        print(sa, flush = True)
        
    return log_lines


def scoda_deg_gsea( adata_t, pw_db, 
                    cond_col, sample_col, 
                    deg_base, ref_group, 
                    deg_pval_cutoff,
                    gsea_pval_cutoff, 
                    N_cells_min_per_sample, 
                    N_cells_min_per_condition, n_cores, 
                    print_prefix, 
                    deg_pairwise = True,
                    deg_cmp_mode = 'max',
                    uns_key_suffix = '',
                    verbose = False): 
    
    log_lines = ''
    # ref_group_for_deg = ref_group
    ref_group_for_deg = None
    if ref_group is not None:
        if ref_group in list(adata_t.obs[cond_col].unique()):
            ref_group_for_deg = ref_group    
    
    # test_method = 't-test' # 't-test', 'wilcoxon'
    pw_db_sel = pw_db
    R_max = 2
    min_size = 5
    max_size = 1000                
    log_fc_col = 'log2_FC'
    gene_col = 'gene'

    if deg_base in list(adata_t.obs.columns.values):
        celltype_col = deg_base
    else:
        celltype_col = 'celltype_minor'

    if cond_col not in list(adata_t.obs.columns.values):
        adata_t.obs[cond_col] = 'Not specified'
    cond_lst = list(adata_t.obs[cond_col].unique())
    
    if sample_col not in list(adata_t.obs.columns.values):
        adata_t.obs[sample_col] = 'Not specified'
    sample_lst = list(adata_t.obs[sample_col].unique())
    
    if len(cond_lst) <= 1:
        s = 'DEG analysis not performed.' # % print_prefix)
        log_lines = log_lines + '%s\n' % s
        if verbose: 
            sa = '%s%s' % (print_prefix, s)
            print(sa)
            
        s = '(might be single sample or no sample condition provided.)' 
        log_lines = log_lines + '%s\n' % s
        if verbose: 
            sa = '%s%s' % (print_prefix, s)
            print(sa, flush = True)    
    else:
        s = 'DEG/GSA analysis .. ' 
        log_lines = log_lines + '%s\n' % s
        if verbose: 
            sa = '%s%s' % (print_prefix, s)
            print(sa, flush = True)    
        pass
        
        ## Normalize and log-transform
        adata = adata_t[:,:]
        if hasattr(adata_t, 'X_log1p'):
            adata.X_log1p = adata_t.X_log1p
        # if not tumor_identification:    
        
        # sc.pp.normalize_total(adata, target_sum=1e4)
        # sc.pp.log1p(adata)

        if hasattr(adata, 'X_log1p'):
            adata = anndata.AnnData(X = adata.X_log1p, obs = adata.obs, var = adata.var)
        else:
            start_time = time.time()
            # Xs = adata.to_df()
            Xs = adata.X
            Xx = X_preprocessing( Xs, log_transformed = False )
            if isinstance(Xx, csc_matrix):
                adata = anndata.AnnData(X = csc_matrix(Xx), obs = adata.obs, var = adata.var)
            else:
                adata = anndata.AnnData(X = csr_matrix(Xx), obs = adata.obs, var = adata.var)
            # adata.X = csr_matrix(Xx)
            # del Xs
            # sc.pp.highly_variable_genes(adata, n_top_genes = 2000) # , flavor = 'seurat_v3')

            etime = round(time.time() - start_time) 
            s = '   Preprocessing done (%i). ' % (etime)
            log_lines = log_lines + '%s\n' % s
            if verbose: 
                sa = '%s%s' % (print_prefix, s)
                print(sa, flush = True)    
        
        cond_lst = list(adata.obs[cond_col].unique())
        celltype_lst = list(adata.obs[celltype_col].unique()) 
        genes_all = list(adata.var.index.values)

        df_dct_dct_n_cells = {}
        df_dct_dct_deg = {}
        df_dct_dct_enr = {}
        df_dct_dct_enr_up = {}
        df_dct_dct_enr_dn = {}
        df_dct_dct_pr = {}
        
        if 'unassigned' in celltype_lst:
            celltype_lst = list(set(celltype_lst) - {'unassigned'})
        celltype_lst.sort()
        
        if ref_group_for_deg is not None:
            if ref_group_for_deg not in list(adata.obs[cond_col].unique()):
                s = '   WARNING: Ref group %s not present -> will be performed in one-against-the-rest.' % (ref_group_for_deg)
                log_lines = log_lines + '%s\n' % s
                if verbose: 
                    sa = '%s%s' % (print_prefix, s)
                    print(sa, flush = True)    
                ref_group_for_deg = None
                
        for ct in celltype_lst:
            
            b = adata.obs[celltype_col] == ct
            adata_sel = adata[b,:]

            adata_s = select_samples( adata_sel, sample_col, 
                                      N_min = N_cells_min_per_sample, R_max = R_max )

            pcnt = adata_s.obs[cond_col].value_counts()
            sx = ''
            for i in pcnt.index.values:
                sx = sx + '%i(%s), ' % (pcnt[i], i)
            sx = sx[:-2]

            '''
            if ((pcnt.min() < N_cells_min_per_condition) or pcnt.shape[0] <= 1):
                s = '   %s: %s -> insufficient # of cells -> DEG skipped.' % (ct, sx)
                log_lines = log_lines + '%s\n' % s
                if verbose: 
                    sa = '%s%s' % (print_prefix, s)
                    print(sa, flush = True)    
            else:
            '''
            
            df_cbyg_in = adata_s.to_df()
            groups_in = adata_s.obs[cond_col]
            samples_in = adata_s.obs[sample_col] # adata_s.obs.index.values
                
            df_lst = deg_multi_ext_check_if_skip( df_cbyg_in, 
                                groups_in, ref_group = ref_group_for_deg, 
                                samples_in = samples_in, min_exp_frac = 0.05, 
                                exp_only = False, min_frac = 0.05, 
                                N_cells_min_per_condition = N_cells_min_per_condition )            

            b_do_deg = True
            if pcnt.shape[0] <= 1:
                b_do_deg = False
                s = '   %s: %s -> Only one condition (Nothing to compare) -> DEG skipped.' % (ct, sx)
                log_lines = log_lines + '%s\n' % s
                if verbose: 
                    sa = '%s%s' % (print_prefix, s)
                    print(sa, flush = True)    
            elif (ref_group_for_deg is not None): 
                if (ref_group_for_deg not in pcnt.index.values.tolist()):
                    b_do_deg = False
                    s = '   %s: %s -> No ref cells (of condition %s) -> DEG skipped.' % (ct, sx, ref_group_for_deg)
                elif pcnt[ref_group_for_deg] < N_cells_min_per_condition:
                    b_do_deg = False
                    s = '   %s: %s -> insufficient # of ref cells -> DEG skipped.' % (ct, sx)
                '''
                elif np.sum(pcnt >= N_cells_min_per_condition) == 1:
                    b_do_deg = False
                    s = '   %s: %s -> insufficient # of ref cells -> DEG skipped.' % (ct, sx)
                ''' 
                if not b_do_deg:
                    log_lines = log_lines + '%s\n' % s
                    if verbose: 
                        sa = '%s%s' % (print_prefix, s)
                        print(sa, flush = True)           

            if b_do_deg: 
                if (np.sum(pcnt >= N_cells_min_per_condition) < 1 ) \
                     or (len(list(df_lst.keys())) == 0):
                    b_do_deg = False
                    s = '   %s: %s -> insufficient # of cells -> DEG skipped.' % (ct, sx)
                    log_lines = log_lines + '%s\n' % s
                    if verbose: 
                        sa = '%s%s' % (print_prefix, s)
                        print(sa, flush = True)

            if b_do_deg:
                s = '   %s: %s' % (ct, sx)
                log_lines = log_lines + '%s\n' % s
                if verbose: 
                    sa = '%s%s' % (print_prefix, s)
                    print(sa, flush = True)    

                df_lst, nc_lst, df_lst_cbyg, groups_lst, test_group_lst = deg_multi_ext( df_cbyg_in, 
                                    groups_in, ref_group = ref_group_for_deg, 
                                    samples_in = samples_in, min_exp_frac = 0.05, 
                                    exp_only = False, min_frac = 0.05, 
                                    N_cells_min_per_condition = N_cells_min_per_condition,
                                    pairwise = deg_pairwise, cmp_mode = deg_cmp_mode)
                
                df_dct_dct_deg[ct] = df_lst
                df_dct_dct_n_cells[ct] = nc_lst

                '''
                '''
                ## Run gseapy.prerank
                df_lst_enrichr = {}
                df_lst_enrichr_up = {}
                df_lst_enrichr_dn = {}
                df_lst_prerank = {}
                #'''
                for c in df_lst.keys():
                    
                    gene_rank_all = df_lst[c].copy(deep = True)
                    b = gene_rank_all['pval'] <= deg_pval_cutoff

                    df_cbyg = df_lst_cbyg[c]
                    groups = groups_lst[c]
                    group_test = test_group_lst[c]

                    # print(np.sum(b), gene_rank.shape)
                    if np.sum(b) > 1:
                        gene_rank = gene_rank_all.loc[b,:].copy(deep = True)            
                        gene_rank[gene_col] = list(gene_rank.index.values)

                        ## Run gseapy.enrichr
                        df_res_enr_pos = None
                        b = gene_rank[log_fc_col] > 0
                        if np.sum(b) > 0:
                            df_res_enr_pos = run_gsa(gene_rank.loc[b,gene_col], pw_db_sel, genes_all, 
                                                     pval_max = gsea_pval_cutoff, min_size = min_size)
                            df_res_enr_pos['Ind'] = 1
                            # if verbose: print('  Num. of selected pathways in Enrichr (+): ', df_res_enr_pos.shape[0])

                        df_res_enr_neg = None
                        b = gene_rank[log_fc_col] < 0
                        if np.sum(b) > 0:
                            df_res_enr_neg = run_gsa(gene_rank.loc[b,gene_col], pw_db_sel, genes_all, 
                                                     pval_max = gsea_pval_cutoff, min_size = min_size)
                            df_res_enr_neg['Ind'] = -1
                            # if verbose: print('  Num. of selected pathways in Enrichr (-): ', df_res_enr_neg.shape[0])

                        if (df_res_enr_pos is not None) & (df_res_enr_neg is not None):
                            df_res_enr = pd.concat([df_res_enr_pos, df_res_enr_neg], axis = 0)
                        elif (df_res_enr_pos is not None):
                            df_res_enr = df_res_enr_pos
                        elif (df_res_enr_neg is not None):
                            df_res_enr = df_res_enr_neg
                        else:
                            df_res_enr = None

                        if df_res_enr is not None:
                            df_lst_enrichr[c] = df_res_enr

                        if df_res_enr_pos is not None:
                            df_lst_enrichr_up[c] = df_res_enr_pos
                        if df_res_enr_neg is not None:
                            df_lst_enrichr_dn[c] = df_res_enr_neg

                        ## Run gseapy.prerank
                        #'''
                        logfc = gene_rank[[log_fc_col]] ## index must be gene name
                        df_res_pr, pr_res = run_prerank(logfc, pw_db_sel, pval_max = gsea_pval_cutoff,
                                        min_size = min_size, max_size = max_size, n_cores = n_cores )
                        '''
                        cls_lst = list(groups)
                        gene_rank['names'] = gene_rank[gene_col]
                        gene_rank['logfoldchanges'] = gene_rank[log_fc_col]
                        
                        df_res_pr, pr_res = run_gsea( df_cbyg, cls_lst, group_test, 
                                       gene_rank, pw_db_sel, 
                                       pval_max = gsea_pval_cutoff, min_pvals = 1e-20, 
                                       min_size = min_size, max_size = max_size, 
                                       n_cores = n_cores, seed = 7, verbose = verbose)
                        #'''                        
                        if df_res_pr.shape[0] > 0:
                            df_res_pr[['ES', 'NES', 'NOM p-val', 'FDR q-val', 'FWER p-val']] = \
                                df_res_pr[['ES', 'NES', 'NOM p-val', 'FDR q-val', 'FWER p-val']].astype(float)
                            df_lst_prerank[c] = df_res_pr                        
                            
                df_dct_dct_enr_up[ct] = df_lst_enrichr_up
                df_dct_dct_enr_dn[ct] = df_lst_enrichr_dn
                df_dct_dct_enr[ct] = df_lst_enrichr
                df_dct_dct_pr[ct] = df_lst_prerank
                # '''

        adata_t.uns['DEG_stat%s' % uns_key_suffix] = df_dct_dct_n_cells
        adata_t.uns['DEG%s' % uns_key_suffix] = df_dct_dct_deg
        # adata_t.uns['GSA'] = df_dct_dct_enr
        adata_t.uns['GSA_up%s' % uns_key_suffix] = df_dct_dct_enr_up
        adata_t.uns['GSA_down%s' % uns_key_suffix] = df_dct_dct_enr_dn
        adata_t.uns['GSEA%s' % uns_key_suffix] = df_dct_dct_pr
        
    s = 'DEG/GSA analysis .. done. ' 
    log_lines = log_lines + '%s\n' % s
    if verbose: 
        sa = '%s%s' % (print_prefix, s)
        print(sa, flush = True)    
    
    return log_lines


def scoda_all_in_one( adata_t, mkr_db, cpdb_path, gsea_pw_db, cnv_gtf = None, 
                      cond_col = 'condition', 
                      sample_col = 'sample', 
                     
                      N_genes_min = 200,
                      N_cells_min = 10,
                      Pct_cnt_mt_max = 20,
        
                      ###  params for HiCAT  ###
                      hicat_clustering_resolution = 1,
                      hicat_pct_th_cutoff = 0.5,
                      hicat_n_cells_max_for_pca = 60000,
                      hicat_n_cells_max_for_gmm = 10000,
                      hicat_mkr_selector = '100000',

                      cnv_ref_key = 'celltype_major',
                      cnv_ref_list = None, 
                      cnv_ref_pct_min = 0.2,
                      cnv_cmd_cutoff = 0.03, 
                      cnv_use_ref_only = False,
                      cnv_n_cells_max_for_pca = 60000,
                      cnv_connectivity_threshold = 0.4, 
                      cnv_connectivity_threshold2 = 0.6, 
                      cnv_clustering_algo = 'lv', 
                      cnv_clustering_resolution = 1, 
                      cnv_clustering_loop = 5, 
                      cnv_clustering_n_cells = 10000,
                      cnv_tumor_dec_th_max = 5, 
                      cnv_tumor_dec_margin = 0.01, 
                      cnv_gcm = 0.3,
                      cnv_unclear_corr_margin = 0.5,
                     
                      cci_run_unit = 'sample', 
                      cci_n_cells_min = 40, 
                      cci_base = 'celltype_minor',
                      cci_pval_cutoff = 0.1, 
                      cci_mean_cutoff = 0, 
                      cci_rth = 0.5, 
                     
                      deg_ref_group = None, 
                      deg_pval_cutoff = 0.01, 
                      deg_n_cells_min = 100, 
                      deg_base = 'celltype_minor', 
                     
                      gsea_pval_cutoff = 0.1, 
                     
                      n_cores = 4, jump_to = 0,
                      data_dir = '.', 
                      verbose = True, 
                      print_prefix = ''):

    df_mkr_db = mkr_db
    gtf_file = cnv_gtf 
    # cpdb_path = cpdb_path 
    pw_db_for_gsea = gsea_pw_db 
            
    # tumor_id_ref_celltypes = None
    tumor_id_ref_celltypes = cnv_ref_list 

    unit_of_cci_run = cci_run_unit 
    min_n_cells_for_cci = cci_n_cells_min 
    cci_pval_max = cci_pval_cutoff 
    cci_mean_min = cci_mean_cutoff 
    Rth = cci_rth 

    # deg_ref_group = deg_ref # None, 
    # deg_pval_cutoff = 0.01, 
    min_n_cells_for_deg = deg_n_cells_min 
    n_cores_to_use = n_cores 
    # data_dir = '.', 
    # verbose = True, 
    # prefix = ''

    #####################
    ### Filter data #####
    if 'condition' in list(adata_t.obs.columns.values):
        adata_t.obs['condition'] = adata_t.obs['condition'].astype(str) 
    if 'sample' in list(adata_t.obs.columns.values):
        adata_t.obs['sample'] = adata_t.obs['sample'].astype(str)  

    bm = adata_t.var_names.str.startswith('MT-')  
    tc = np.array(adata_t.X.sum(axis = 1))[:,0]
    mc = np.array(adata_t[:, bm].X.sum(axis = 1))[:,0]
    bp = (100*mc/tc) <= Pct_cnt_mt_max
    if np.sum(bp) != len(bp):
        print('%s   N cells: %i -> %i (%i cells dropped as pct_count_mt > %4.1f) ' % (c_prefix, len(bp), np.sum(bp), len(bp)-np.sum(bp), Pct_cnt_mt_max))
        adata_t = adata_t[bp, :]
    
    bc = (np.array((adata_t.X > 0).sum(axis = 1)) >= N_genes_min)[:,0]  
    if np.sum(bc) != len(bc):
        print('%s   N cells: %i -> %i (%i cells dropped as gene_count < %i) ' % (c_prefix, len(bc), np.sum(bc), len(bc)-np.sum(bc), N_genes_min))
        adata_t = adata_t[bc, :]
        
    bg = (np.array(adata_t.X.sum(axis = 0)) >= N_cells_min)[0,:]
    if np.sum(bg) != len(bg):
        print('%s   N genes: %i -> %i (%i genes dropped as cell_count < %i) ' % (c_prefix, len(bg), np.sum(bg), len(bg)-np.sum(bg), N_cells_min))
        adata_t = adata_t[:, bg]
    
    ################################
    ### Cell-type identification ###
    if jump_to < 1:
        # if verbose: print('%sCelltype annotation running .. ' % print_prefix)

        scoda_hicat(adata_t, df_mkr_db, print_prefix = print_prefix, verbose = verbose,
                    clustering_resolution = hicat_clustering_resolution, 
                    pct_th_cutoff = hicat_pct_th_cutoff, 
                    N_cells_max_for_pca = hicat_n_cells_max_for_pca, 
                    N_cells_max_for_gmm = hicat_n_cells_max_for_gmm, 
                    mkr_selector = hicat_mkr_selector )

        # if verbose: print('%sCelltype annotation done.' % print_prefix)

        # adata_t.write(file_h5ad)

        ## For client info.
        ct_lst_maj = list(adata_t.obs['celltype_major'].unique())
        ct_lst_min = list(adata_t.obs['celltype_minor'].unique())
        ct_lst_sub = list(adata_t.obs['celltype_subset'].unique())

        ct_lst_maj.sort()
        s = ''
        for c in ct_lst_maj:
            s = s + '%s, ' % c
        s = s[:-2]

        if verbose: 
            print('%s  %i major type, %i minor type, %i subset identified.' 
                   % (print_prefix, len(ct_lst_maj), len(ct_lst_min), len(ct_lst_sub)))
            print('%s  Major types: %s' % (print_prefix, s))
        
    #################################
    ### tumor cell identification ###
    if jump_to <= 1:
        if cnv_gtf is not None:

            ## Test without Reference 
            # if verbose: print('%sIdentifying tumor cells .. ' % print_prefix)

            ref_types = tumor_id_ref_celltypes

            ## clustering_algo = GMM is not suitable for this work
            df = scoda_icnv_addon( adata_t, gtf_file, 
                       ref_types = cnv_ref_list, 
                       ref_key = cnv_ref_key, 
                       use_ref_only = cnv_use_ref_only, 
                       n_neighbors = 15,
                       n_pca_comp = 15, 
                       clustering_algo = cnv_clustering_algo,  
                       clustering_resolution = cnv_clustering_resolution, 
                       N_loops = cnv_clustering_loop,
                       N_cells_max_for_clustering = cnv_clustering_n_cells,
                       connectivity_threshold = cnv_connectivity_threshold, 
                       connectivity_threshold2 = cnv_connectivity_threshold2, 
                       N_cells_max_for_pca = cnv_n_cells_max_for_pca, 
                       ref_pct_min = cnv_ref_pct_min,
                       use_umap = False, 
                       tumor_dec_th_max = cnv_tumor_dec_th_max, 
                       tumor_dec_margin = cnv_tumor_dec_margin, 
                       uc_cor_margin = cnv_unclear_corr_margin, 
                       net_search_mode = 'sum', 
                       cmd_cutoff = cnv_cmd_cutoff, 
                       gcm = cnv_gcm,
                       use_cnv_score = True, 
                       print_prefix = print_prefix, 
                       n_cores = n_cores_to_use, 
                       verbose = verbose,
                       group_cell_size = 15000 )
            
            if verbose: print('%sTumor cells identification done. ' % print_prefix)
            # adata_t.write(file_h5ad)
           
    #'''
    #############################
    ### Cell-cell interaction ###
    if jump_to <= 2:
        # if verbose: print('%sInfering cell-cell interactions .. ' % print_prefix)

        scoda_cci( adata_t, cpdb_path, cond_col = cond_col, sample_col = sample_col, 
                   cci_base = cci_base, unit_of_cci_run = unit_of_cci_run, n_cores = n_cores_to_use, 
                   min_n_cells_for_cci = min_n_cells_for_cci, Rth = Rth,
                   pval_max = cci_pval_max, mean_min = cci_mean_min, data_dir = data_dir,
                   print_prefix = print_prefix, cpdb_version = 4, verbose = verbose )

        # if verbose: print('%sInfering cell-cell interactions done. ' % print_prefix)    
        # adata_t.write(file_h5ad)
        
    ####################
    ### DEG analysis ###
    
    if jump_to <= 3:
        # if verbose: print('%sDEG/GSEA analysis ..       ' % print_prefix)
        scoda_deg_gsea( adata_t, pw_db = pw_db_for_gsea, 
                        cond_col = cond_col, sample_col = sample_col, 
                        deg_base = deg_base, 
                        ref_group = deg_ref_group, 
                        deg_pval_cutoff = deg_pval_cutoff, 
                        gsea_pval_cutoff = gsea_pval_cutoff,
                        N_cells_min_per_sample = min_n_cells_for_deg, 
                        N_cells_min_per_condition = min_n_cells_for_deg, 
                        n_cores = n_cores_to_use, 
                        print_prefix = print_prefix, 
                        verbose = verbose)
        
        ## Overwrite 
        # adata_t.write(file_h5ad)        
        if verbose: print('%sDEG/GSEA analysis done.     ' % print_prefix)
    #'''
        
    return

#####################################
######### Tissue detection ##########

from scoda.hicat import get_markers_major_type, GSA_cell_subtyping

def detect_tissue( adata_t, df_mkr_db_all, gtf_file,
                   tissue_lst, tissue_score_means = None,
                   n_cells_to_use = 3000, non_ref_n_th = 100,
                   non_ref_p_th = 0.05, score_th = 6, 
                   taxo_level = 'cell_type_major',
                   confidence = 'Confidence',
                   ident_level = 3, N_mkrs_max = 50, 
                   verbose = True, ref_celltypes_add = [] ):
    '''
    idx = list(adata_t.obs.index.values)
    idx_sel = random.sample(idx, min( n_cells_to_use*2, len(idx) ))
    adata = adata_t[idx_sel, :]
    df_gene_odr = get_gene_order( adata )
    '''
    
    idx = list(adata_t.obs.index.values)
    idx_sel = random.sample(idx, min( max(n_cells_to_use, int(len(idx)*0.08)), len(idx) ))
    adata = adata_t[idx_sel, :]
    
    ref_tissues = ['Immune', 'Immune_ext', 'Generic']
    b = df_mkr_db_all['tissue'].isin(ref_tissues)
    ref_celltypes = list( df_mkr_db_all.loc[b,'cell_type_major'].unique() )
    ref_celltypes = ref_celltypes + ref_celltypes_add # ['Stellate cell', 'Fibroblast']

    X_cell_by_gene = adata.to_df()
    
    score_a_lst = []
    score_b_lst = []
    score_c_lst = []
    score_d_lst = []
    unr_lst = []
    fnref_lst = []
    for tissue in tissue_lst:
        '''
        if (tissue in tissue_lst) & (tissue != 'Generic'):
            # target_tissues = [tissue]
            target_tissues = ['Immune', 'Immune_ext', 'Generic', tissue]
        else:
            # target_tissues = ['Epithelium']
            target_tissues = ['Immune', 'Immune_ext', 'Generic', 'Epithelium']
        '''
        if tissue == 'Generic':
            target_tissues = ['Immune', 'Generic', 'Epithelium']
        elif tissue == 'Blood':
            target_tissues = ['Immune', 'Generic', 'Immune_ext', tissue] # 
        elif tissue in tissue_lst:
            target_tissues = ['Immune', 'Generic', tissue] # , 'Immune_ext'
        else:
            target_tissues = ['Immune', 'Generic', 'Epithelium']
        
        b = df_mkr_db_all['tissue'].isin(target_tissues)
        df_mkr_db = df_mkr_db_all.loc[b,:].copy(deep = True)
        # df_mkr_db = select_highly_variable_markers( df_mkr_db, df_gene_odr, N_mkrs = N_mkrs_max )

        '''
        df_pred = scoda_hicat(adata, df_mkr_db, print_prefix = 'USER INFO: ', verbose = -1, 
                    clustering_resolution = 2, pct_th_cutoff = 0.5, 
                    N_cells_max_for_pca = 60000, N_cells_max_for_gmm = 10000,
                    target_tissues = [], target_cell_types = [], 
                    ident_level = ident_level, N_markers_max = [N_mkrs_max]*3 )
        #'''

        mkr_lst, mkr_lst_neg = get_markers_major_type( df_mkr_db, target_cells = [], # pnsh12 = PNSH12,
                                rem_common = False, to_upper = False, verbose = False)
        df_pred, df_score, dfn = GSA_cell_subtyping( X_cell_by_gene, mkr_lst, mkrs_neg = None, verbose = False )
        df_pred[taxo_level] = df_pred['cell_type(1st)']
        df_pred[confidence] = df_pred['-logP']

        '''
        adata.obs['celltype_major'] = df_pred['cell_type(1st)'].astype(str)
        ref_types = ['T cell', 'B cell', 'Myeloid cell', 'Fibroblast']
        
        ## clustering_algo = GMM is not suitable for this work
        df = scoda_icnv_addon( adata, gtf_file, 
                           ref_types = ref_types, 
                           ref_key = "celltype_major", 
                           use_ref_only = False, 
                           clustering_algo = 'lv', # CLUSTERING_AGO,  
                           clustering_resolution = 0, #cnv_clustering_resolution, 
                           N_loops = 7, # cnv_clustering_loop,
                           N_cells_max_for_clustering = 20000, # cnv_clustering_n_cells,
                           n_neighbors = 15,
                           connectivity_threshold = 0.4, # cnv_connectivity_threshold, 
                           connectivity_threshold2 = 0.6,
                           n_pca_comp = 15, 
                           N_cells_max_for_pca = 60000, # cnv_clustering_n_cells,
                           use_umap = False, 
                           ref_pct_min = 0.2,
                           tumor_dec_th_max = 5, # cnv_tumor_dec_th_max, 
                           tumor_dec_margin = 0.2, # cnv_tumor_dec_margin, 
                           uc_cor_margin = 0.5, 
                           net_search_mode = 'sum', 
                           cmd_cutoff = 0.03, gcm = 0.3, # cnv_gcm,
                           use_cnv_score = True, 
                           print_prefix = None, 
                           n_cores = 3, 
                           verbose = verbose )
        #'''

        b = df_pred[taxo_level] == 'unassigned'
        unr = np.mean(b)
        # unr_lst[tissue] = unr
        unr_lst.append(unr)
    
        b_non_ref = ~df_pred[taxo_level].isin(ref_celltypes)
        N_th = min(non_ref_n_th, non_ref_p_th*len(b_non_ref))
        if np.sum(b_non_ref) >= N_th:
            df_pred_sel = df_pred.loc[b_non_ref,:]
        else:
            df_pred_sel = df_pred
            df_pred_sel[confidence] = 0
            # print('WARNING: %i < %i' % (np.sum(b_non_ref), N_th))
        frac_non_ref = np.sum(b_non_ref)/len(b_non_ref)
        fnref_lst.append(frac_non_ref)
        
        smed = df_pred_sel[confidence].median()
        bmed = df_pred_sel[confidence] >= smed
        if len(bmed) > 0:
            score_a = df_pred_sel.loc[bmed, confidence].mean()
        else:
            score_a = 0
        
        score_b = df_pred_sel[confidence].mean()
        score_c = df_pred[confidence].mean()

        if tissue_score_means is not None:
            score_d = (score_b - tissue_score_means[tissue]) 
        else:             
            score_d = score_b
            
        score_a_lst.append(score_a)
        score_b_lst.append(score_b)
        score_c_lst.append(score_c)
        score_d_lst.append(score_d)
        
        if verbose: 
            print('%16s: %5.2f, %5.2f, %5.2f' % (tissue, score_d, 
                                                 score_b, frac_non_ref))
            # print('%14s: %5.2f, %5.2f, %5.2f - %5.2f, %5.2f' % (tissue, score_c, score_a, score, frac_non_ref, unr))

    df = pd.DataFrame( {'sa': score_a_lst, 'sb': score_b_lst, 
                        'sc': score_c_lst, 'sd': score_d_lst,
                        'ur' : unr_lst, 'fnr': fnref_lst},
                       index = tissue_lst)

    return df


def detect_tissue_get_stats( default_files_path, non_ref_p_th = 0.05, verbose = False ): 

    if len(default_files_path) == 0:
        d = 'tissue_score_stats/'
    elif default_files_path[-1] == '/':
        d = default_files_path + 'tissue_score_stats/'
    else:
        d = default_files_path + '/tissue_score_stats/'
    df_sb = pd.read_csv(d + 'res_score_b.csv', index_col = 0)
    df_fr = pd.read_csv(d + 'res_frac_non_ref.csv', index_col = 0)
    
    # df_s = df_s.astype(float)
    
    lst = list(df_sb.columns.values)
    targets = [s.split('_')[0] for s in lst]
    # display(targets)
    
    tissue_score_means = {}
    df_s = df_sb.copy(deep = True)
    tgt = np.array(targets)
    for i in list(df_s.index.values):
        b = tgt != i 
        mn = df_s.loc[i,b].mean()
        tissue_score_means[i] = mn
        if verbose: print('%s: %5.2f ' % (i, mn))
        df_s.loc[i,:] = df_s.loc[i,:] - mn
    
    df_s = df_sb.sub(df_sb.mean(axis = 1), axis = 0) * (df_fr >= non_ref_p_th)
    
    # tissue_score_means
    decision = list(df_s.idxmax(axis = 0))
    
    tgt = np.array(targets)
    dec = np.array(decision)

    if verbose:
        b = (tgt != dec) & (dec != 'Generic')
        print( 'ErrorRate: %4.2f <- %i/%i' % ((np.sum(b)/len(b)).round(3)*100, np.sum(b), len(b)) )
        print(dict(zip(tgt,dec)))
    
    return tissue_score_means
