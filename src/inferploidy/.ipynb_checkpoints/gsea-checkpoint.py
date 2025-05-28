import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, warnings, random
import matplotlib as mpl
from scipy.stats import hypergeom

'''
import scanpy as sc
import gseapy as gp
import seaborn as sns
'''

SCANVPY = True
try:
    import scanpy as sc
except ImportError:
    print('WARNING: scanpy not installed.')
    SCANVPY = False

SEABORN = True
try:
    import seaborn as sns
except ImportError:
    print('WARNING: seaborn not installed or not available. ')
    SEABORN = False

GSEAPY = True
try:
    import gseapy as gp
except ImportError:
    print('WARNING: gseapy not installed or not available. ')
    GSEAPY = False


### Example
# data = 'BRL_mouse'
# adata_t = sc.read_h5ad('mmColon_85K_%s_ann.h5ad' % data)

# ## Normalize and log-transform
# sc.pp.normalize_total(adata_t, target_sum=1e4)
# sc.pp.log1p(adata_t)

# # sample_col = 'sid'
# test_method = 't-test' # 't-test', 'wilcoxon'
# test_method_gsea = 't_test' # 'signal_to_noise', 't_test'
# pw_db_sel = 'KEGG_2019_Human'

# ## Set analysis paramters
# test_sel = 't-test'
# log2fc_min = 0.25
# pval_cutoff = 0.05
# pv_maxs_pw = [0.01, 0.1, 0.1]
# min_pvals = 1e-300
# log1p = True
# n_cells_min = 40
# min_size = 5
# max_size = 1000

# sample_col = 'sid'
# cond_col = 'condition'
# celltype_col = 'HiCAT_minor'

# cell = 'Macrophage'
# group_control = 'Healthy'
# group_test = 'AcuteColitis'

# ## select target groups
# b1 = adata_t.obs[celltype_col] == cell
# b2 = adata_t.obs[cond_col].isin([group_test, group_control])
# adata_s = adata_t[b1&b2,:]
# adata_s.obs[cond_col].value_counts()

# adata_s = select_samples( adata_s, sample_col, N_min = 100, R_max = 2.5 )

# ## Run GSEApy
# df_res_enr, df_res_pr, df_gsea = \
#     run_gsea_all(adata_s, cond_col, group_test, pw_db_sel, 
#                  pvmins = [0.05, 0.1, 0.1], log1p = log1p,
#                  log2fc_min = log2fc_min, pval_cutoff = pval_cutoff, 
#                  test_sel = test_sel, n_cells_min = n_cells_min,
#                  min_size = min_size, max_size = max_size, verbose = True )

# ## Plot prerank results
# df_res = df_res_pr.copy(deep = True)
# df_res = df_res.sort_values(by = 'NES', ascending = False)
# terms_sel = list(df_res.index.values)

# items_to_plot = ['-log(p-val)', 'NES', '-log(q-val)']
# lims = [[0, 2], [-5, 5], [0, 2]]

# plot_gsea_res( df_res, terms_sel, items_to_plot, lims)


def get_gene_rank(adata_s, cond_col, group_test, 
                  log2fc_min = 0.25, pval_cutoff = 0.05, 
                  test_sel = 't-test', n_cells_min = 40, 
                  log1p = True):
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc.tl.rank_genes_groups(adata_s, cond_col, method=test_sel, 
                                key_added = test_sel, pts = True)

        # calculate_qc_metrics will calculate number of cells per gene
        sc.pp.calculate_qc_metrics(adata_s, percent_top=None, log1p=log1p, inplace=True)
        # filter for genes expressed in at least 30 cells.
        genes = adata_s.var_names[adata_s.var.n_cells_by_counts >= n_cells_min]

        ## Get gene ranks
        gene_rank = sc.get.rank_genes_groups_df(adata_s, group=group_test, 
                                                key=test_sel)
        
    gene_rank = gene_rank[gene_rank['names'].isin(genes)]

    ## Filter gene ranks
    b1 = np.abs(gene_rank['logfoldchanges']) >= log2fc_min 
    b2 = np.abs(gene_rank['logfoldchanges']) <= -log2fc_min 
    b3 = gene_rank['pvals'] <= pval_cutoff
    b = (b1 | b2) & b3
    gene_rank = gene_rank.loc[b,:]

    return gene_rank


def run_enrich(glist, pw_db_sel, background_genes = None,
                pval_max = 0.05, min_pvals = 1e-20):
    
    ## Run gseapy.enrich (not enrichr)
    enr_res = gp.enrich(gene_list = glist,
                         gene_sets = pw_db_sel,
                         background = background_genes,
                         outdir=None, verbose = True )

    df_res_enr = enr_res.res2d.copy(deep = True)
    df_res_enr['-log(p-val)'] = -np.log10(df_res_enr['P-value'] + min_pvals)
    df_res_enr['-log(q-val)'] = -np.log10(df_res_enr['Adjusted P-value'] + min_pvals)

    terms_sel_enr = list(df_res_enr['Term'])
    df_res_enr = df_res_enr.set_index('Term')
    df_res_enr['Term'] = terms_sel_enr

    b = df_res_enr['-log(p-val)'] >= -np.log10(pval_max)
    df_res_enr = df_res_enr.loc[b,:]
    return df_res_enr
  
    
def run_enrichr(glist, pw_db_sel, background_genes = None,
                pval_max = 0.05, min_pvals = 1e-20):
    
    ## Run gseapy.enrichr (not enrich)
    enr_res = gp.enrichr(gene_list = glist,
                         gene_sets = pw_db_sel,
                         background = background_genes,
                         outdir=None, verbose = True )

    df_res_enr = enr_res.res2d.copy(deep = True)
    df_res_enr['-log(p-val)'] = -np.log10(df_res_enr['P-value'] + min_pvals)
    df_res_enr['-log(q-val)'] = -np.log10(df_res_enr['Adjusted P-value'] + min_pvals)

    terms_sel_enr = list(df_res_enr['Term'])
    df_res_enr = df_res_enr.set_index('Term')
    df_res_enr['Term'] = terms_sel_enr

    b = df_res_enr['-log(p-val)'] >= -np.log10(pval_max)
    df_res_enr = df_res_enr.loc[b,:]
    return df_res_enr
  

def load_gmt(file):
    dct = {}
    with open(file, 'r') as f:
        for line in f:
            items = line.split('\t')
            dct[items[0]] = items[2:]
    return dct

    
def run_gsa(glist, pw_db_sel, background_genes = None,
                pval_max = 0.05, min_pvals = 1e-20, min_size = 5):
    
    if isinstance(pw_db_sel, str):
        if os.path.isfile(pw_db_sel):
            pw_db_dict = load_gmt(pw_db_sel)
        else:
            print('ERROR: the pathway DB must be a path to .gmt file or dict.')
            return None
    elif isinstance(pw_db_sel, dict):
        pw_db_dict = pw_db_sel
    else:
        print('ERROR: the pathway DB must be a path to .gmt file or dict.')
        return None
        
    df_res = pd.DataFrame(# index = list(pw_db_dict.keys()), 
                          columns = ['pval', 'pval_adj', 'genes', 'overlap', 'pw_size', '-log(p-val)', '-log(q-val)' ])
    
    neg_log_pval_max = -np.log10(min_pvals)
    N = len(background_genes)
    nlogN = -np.log10(N)
    k = len(glist)
    for key in pw_db_dict.keys():
        mkrs = pw_db_dict[key]
        overlap = list(set(mkrs).intersection(list(glist)))
        n = len(overlap)
        if n >= min_size:
            M = len(mkrs)
            neg_log_pval = min(-hypergeom.logsf(n-1, N, M, k)*np.log10(np.exp(1)), neg_log_pval_max)
            pval = 10**(-neg_log_pval)
            df_res.loc[key, 'pval'] = float(pval)
            df_res.loc[key, 'pval_adj'] = float(min(pval*N, 1))
            df_res.loc[key, 'overlap'] = int(n)
            df_res.loc[key, 'pw_size'] = int(M)
            df_res.loc[key, '-log(p-val)'] = float(neg_log_pval)
            df_res.loc[key, '-log(q-val)'] = float(max( neg_log_pval + nlogN, 0 ))
            genes_str = ''
            for g in overlap:
                genes_str = genes_str + '%s;' % g
            genes_str = genes_str[:-1]
            df_res.loc[key, 'genes'] = genes_str
    
    ## Run gseapy.enrichr (not enrich)
    terms_sel = list(df_res.index.values)
    df_res['Term'] = terms_sel
    df_res['pval'] = df_res['pval'].astype(float)
    df_res['pval_adj'] = df_res['pval_adj'].astype(float)
    df_res['overlap'] = df_res['overlap'].astype(int)
    df_res['pw_size'] = df_res['pw_size'].astype(int)
    df_res['-log(p-val)'] = df_res['-log(p-val)'].astype(float)
    df_res['-log(q-val)'] = df_res['-log(q-val)'].astype(float)

    b = df_res['-log(p-val)'] >= -np.log10(pval_max)
    df_res = df_res.loc[b,:]
    return df_res
  
        
def run_prerank(gene_rank, pw_db_sel, background_genes = None,
                pval_max = 0.05, min_pvals = 1e-20, 
                min_size = 3, max_size = 1000, n_cores = 4):
    
    res = gp.prerank(rnk = gene_rank, 
                         gene_sets = pw_db_sel,
                         threads = n_cores,
                         min_size = min_size,
                         max_size = max_size,
                         permutation_num = 1000, # reduce number to speed up testing
                         background = background_genes,
                         outdir = None, # don't write to disk
                         seed = 6) # see what's going on behind the scenes)

    df_res_pr = res.res2d.copy(deep = True)
    df_res_pr['-log(p-val)'] = -np.log10(df_res_pr['NOM p-val'].astype(float) + min_pvals)
    df_res_pr['-log(q-val)'] = -np.log10(df_res_pr['FDR q-val'].astype(float) + min_pvals)
    terms_sel_pr = list(df_res_pr['Term'].copy(deep = True))
    df_res_pr.set_index('Term', inplace = True)
    df_res_pr['Term'] = terms_sel_pr
    # df_res_pr = df_res_pr.sort_values(by = 'NOM p-val', ascending = True)

    ## Filter GSEA results
    b = df_res_pr['-log(p-val)'] >= -np.log10(pval_max)
    df_res_pr = df_res_pr.loc[b,:]
    return df_res_pr, res
  
    
def revise_gsea_res( gs_res, df_gep, cls_lst, target_group, verbose = True ):
    
    lfc = gs_res.ranking[0]
    gene = gs_res.ranking.index.values[0]

    if target_group not in list(set(cls_lst)):
        print('ERROR: %s not in the class names. You have ' % target_group, set(cls_lst))
    else:
        b = np.array(cls_lst) == target_group
        mr = df_gep.loc[~b, gene].mean()
        mo = df_gep.loc[b, gene].mean()
        if (df_gep < 0).sum().sum() > 0:
            fc = mo - mr
        else:
            fc = np.log(mo/mr)

        s = 1
        if fc*lfc < 1:
            s = -1
            
    if verbose: print('  Revise_gsea_res: Sign = %i' % s)
            
    gs_res.ranking = gs_res.ranking*s
    gs_res.res2d['ES'] = gs_res.res2d['ES']*s
    gs_res.res2d['NES'] = gs_res.res2d['NES']*s

    for key in gs_res.results.keys():
        gs_res.results[key]['es'] = gs_res.results[key]['es']*s
        gs_res.results[key]['nes'] = gs_res.results[key]['nes']*s
        gs_res.results[key]['RES'] = list(np.array(gs_res.results[key]['RES'])*s)
        
    return gs_res


def run_gsea(  df_gep, cls_lst, group_test, 
               gene_rank, pw_db_sel, 
               pval_max = 0.05, min_pvals = 1e-20, 
               min_size = 3, max_size = 1000, verbose = True):
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        ## Run gseapy.gsea
        df_gep2 = df_gep.transpose()
        df_gep2.insert(0, 'Gene', list(df_gep2.index.values))
        df_gep2.insert(1, 'NAME', list(df_gep2.index.values))

        gs_res = gp.gsea(data = df_gep2, # or data='./P53_resampling_data.txt'
                         gene_sets = pw_db_sel, # or enrichr library names
                         cls = cls_lst, # cls=class_vector
                         # set permutation_type to phenotype if samples >=15
                         min_size = min_size,
                         max_size = max_size,
                         permutation_type = 'phenotype',
                         permutation_num = 1000, # reduce number to speed up test
                         outdir = None,  # do not write output to disk
                         method = 't_test',
                         # ascending = False,
                         threads = 4, seed = 7)

        gs_res = revise_gsea_res( gs_res, df_gep, cls_lst, group_test, verbose )

    df_gsea = gs_res.res2d.copy(deep = True)
    df_gsea['-log(p-val)'] = -np.log10(df_gsea['NOM p-val'].astype(float) + min_pvals)
    df_gsea['-log(q-val)'] = -np.log10(df_gsea['FDR q-val'].astype(float) + min_pvals)
    terms_sel_gsea = list(df_gsea['Term'])
    df_gsea.set_index('Term', inplace = True)
    df_gsea['Term'] = terms_sel_gsea

    b = df_gsea['-log(p-val)'] >= -np.log10(pval_max)
    df_gsea = df_gsea.loc[b,:]
    return df_gsea, gs_res
  
    
def plot_gsea_res( df_res, terms_sel, items_to_plot, lims = None, title = None, 
                   title_pos = (0.5, 1), title_fs = 16, title_ha = 'center', Ax = None):

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ## draw result
        sc.settings.set_figure_params(figsize = (4,(len(terms_sel) + 8)/9), dpi=100)

        nr, nc = 1, len(items_to_plot)
        fig, axes = plt.subplots(nrows=nr, ncols=nc, constrained_layout=True)
        fig.tight_layout() # Or equivalently,  "plt.tight_layout()"
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, 
                            wspace=0.3, hspace=0.25)
        if title is not None: 
            fig.suptitle(title, x = title_pos[0], y = title_pos[1], 
                         fontsize = title_fs, ha = title_ha)

        ylabel = 'Term'

        for k, x in enumerate(items_to_plot): # , '-log(q-val)-enr', 'NES-pr']):
            plt.subplot(1,nc,k+1)
            xlabel = x
            ax = sns.barplot(data = df_res, y = ylabel, x = xlabel, 
                             facecolor = 'firebrick', orient = 'h', 
                             edgecolor = 'black', linewidth = 0.8, ax = Ax )
            ax = plt.xticks(fontsize = 8)
            plt.xlabel(xlabel, fontsize = 10)
            if k == 0:
                ax = plt.yticks(fontsize = 8)
                plt.ylabel(ylabel, fontsize = 10)
            else:
                ax = plt.yticks([])
                plt.ylabel(None)
            if lims is not None: plt.xlim(lims[k])

        plt.show()
    return True


def run_gsea_all(adata_s, cond_col, group_test, pw_db_sel, 
                 pvmins = [0.01, 0.05, 0.05], log1p = True,
                 log2fc_min = 0.25, pval_cutoff = 0.01, 
                 test_sel = 't-test', n_cells_min = 40,
                 min_size = 3, max_size = 1000, verbose = True ):
    
    gene_rank = get_gene_rank(adata_s, cond_col, group_test, 
                              log2fc_min = log2fc_min, 
                              pval_cutoff = pval_cutoff, 
                              test_sel = test_sel, 
                              n_cells_min = n_cells_min, 
                              log1p = log1p)
    if verbose: print('  Num DEGs selected: ', gene_rank.shape)

    ## Run gseapy.enrichr
    b = gene_rank['logfoldchanges'] > 0
    df_res_enr_pos = run_enrichr(gene_rank.loc[b,'names'], pw_db_sel, pval_max = pvmins[0])
    df_res_enr_pos['NES'] = 1
    if verbose: print('  Num. of selected pathways in Enrichr (+): ', df_res_enr_pos.shape[0])

    b = gene_rank['logfoldchanges'] < 0
    df_res_enr_neg = run_enrichr(gene_rank.loc[b,'names'], pw_db_sel, pval_max = pvmins[0])
    df_res_enr_neg['NES'] = -1
    if verbose: print('  Num. of selected pathways in Enrichr (-): ', df_res_enr_neg.shape[0])

    df_res_enr = pd.concat([df_res_enr_pos, df_res_enr_neg], axis = 0)

    ## Run gseapy.prerank
    df_res_pr, pr_res = run_prerank(gene_rank, pw_db_sel, pval_max = pvmins[1],
                            min_size = min_size, max_size = max_size )
    if verbose: print('  Num. of selected pathways in Prerank: ', df_res_pr.shape[0])

    ## Get gene expression matrix (cell x gene) and group vector
    df_gep = adata_s.to_df()
    cls_lst = adata_s.obs[cond_col].astype(str)
    # b = cls_lst != group_test
    # cls_lst[b] = 'Others'
    cls_lst = list(cls_lst)
    genes = list(gene_rank['names'])
    df_gep = df_gep[genes]

    ## Run gseapy.gsea
    df_gsea, gs_res = run_gsea( df_gep, cls_lst, group_test, 
                        gene_rank, pw_db_sel, pval_max = pvmins[2],
                        min_size = min_size, max_size = max_size, 
                        verbose = verbose )

    if verbose: print('  Num. of selected pathways in GSEA: ', df_gsea.shape[0])
    
    return gene_rank, df_res_enr, df_res_pr, df_gsea


def select_samples( adata_s, sample_col, N_min = 100, R_max = 2.5, verbose = False ):

    pcnt = adata_s.obs[sample_col].value_counts()
    b = pcnt >= N_min
    plst = list(pcnt.index.values[b])

    N_min = int(max(pcnt.min(), N_min))
    N_max = int(N_min*R_max)

    ## Check basic stats.
    pcnt = adata_s.obs[sample_col].value_counts()

    psel = []
    cnt = 0
    for p in list(pcnt.index):
        b = adata_s.obs[sample_col] == p
        pids = list(adata_s.obs.index.values[b])
        if len(pids) > N_max:
            pids = random.sample(pids, N_max)
        psel = psel + pids
        cnt += 1

    adata_t = adata_s[psel,:]

    if verbose:
        print('N_min/max: %i/%i, N_cells: %i -> %i, N_samples: %i -> %i, N_cells/sample: %4.2f' % 
              (pcnt.min(), N_max, pcnt.sum(), len(psel), len(pcnt), cnt, len(psel)/cnt))
        if pcnt.min() < N_min:
            pcnt = adata_t.obs[sample_col].value_counts()
            print(pcnt)
    
    return adata_t
