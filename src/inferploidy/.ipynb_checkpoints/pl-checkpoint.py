import time, os, copy, datetime, math, random, warnings
import anndata
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import cm, colormaps
import matplotlib as mpl
from matplotlib.pyplot import figure
from matplotlib.patches import Rectangle
# from scipy.cluster.hierarchy import linkage, leaves_list
# import sys
# sys.setrecursionlimit(100000)

from importlib_resources import files

SCANVPY = True
try:
    import scanpy as sc
except ImportError:
    print('WARNING: scanpy not installed.')
    SCANVPY = False

STATANNOT = True
try:
    from statannot import add_stat_annotation
except ImportError:
    print('WARNING: statannot not installed or not available. ')
    STATANNOT = False

SEABORN = True
try:
    import seaborn as sns
except ImportError:
    print('WARNING: seaborn not installed or not available. ')
    SEABORN = False

try:
    import plotly.graph_objects as go
except ImportError:
    print('WARNING: plotly not installed.')
    

def get_sample_to_group_dict( samples, conditions ):
    
    samples = np.array(samples)
    conditions = np.array(conditions)
    
    slst = list(set(list(samples)))
    slst.sort()
    glst = []
    for s in slst:
        b = samples == s
        g = conditions[b][0]
        glst.append(g)
        
    dct = dict(zip(slst, glst))
    return dct


def get_sample_to_group_map( samples, conditions ):
    return get_sample_to_group_dict( samples, conditions )


# df_cnt, df_pct= get_population( adata_s.obs['sid'], 
#                                 adata_s.obs['minor'], sort_by = [] )
def get_population( pids, cts, sort_by = [] ):
    
    pid_lst = list(set(list(pids)))
    pid_lst.sort()
    celltype_lst = list(set(list(cts)))
    celltype_lst.sort()

    df_celltype_cnt = pd.DataFrame(index = pid_lst, columns = celltype_lst)
    df_celltype_cnt.loc[:,:] = 0

    for pid in pid_lst:
        b = np.array(pids) == pid
        ct_sel = np.array(cts)[b]

        for ct in celltype_lst:
            bx = ct_sel == ct
            df_celltype_cnt.loc[pid, ct] = np.sum(bx)

    df_celltype_pct = (df_celltype_cnt.div(df_celltype_cnt.sum(axis = 1), axis = 0)*100).astype(float)
    
    if len(sort_by) > 0:
        df_celltype_pct.sort_values(by = sort_by, inplace = True)

    return df_celltype_cnt, df_celltype_pct


def get_gene_expression_mean( adata, genes = [], group_col = 'sample' ): 

    sample_col = group_col
    slst = adata.obs[sample_col].unique().tolist()
    slst.sort()

    if genes is None:
        Xs = adata.to_df()
        df = pd.DataFrame(index = slst, columns = adata.var.index)
    elif len(genes) == 0:
        Xs = adata.to_df()
        df = pd.DataFrame(index = slst, columns = adata.var.index)
    else:
        Xs = adata[:, genes].to_df()
        df = pd.DataFrame(index = slst, columns = genes)
        
    for s in slst:
        b = adata.obs[sample_col] == s
        mns = (Xs.loc[b,:] > 0).mean(axis = 0)
        df.loc[s,:] = list(mns)
    
    df = df.astype(float)
    return df


def cci_get_means( cci_df_dct, cci_idx_lst = None, 
                   cells = [], genes = [], pval_cutoff = 0.05 ):

    if cci_idx_lst is None:
        ## get coomon cci index
        for j, key in enumerate(cci_df_dct.keys()):
            b = cci_df_dct[key]['pval'] <= pval_cutoff
            if len(cells) > 0:
                b = b & (cci_df_dct[key]['cell_A'].isin(cells) | cci_df_dct[key]['cell_B'].isin(cells))
            if len(genes) > 0:
                b = b & (cci_df_dct[key]['gene_A'].isin(genes) | cci_df_dct[key]['gene_B'].isin(genes))
                         
            if j == 0:
                idx_union = list(cci_df_dct[key].index.values[b])
            else:
                idx_union = list(set(idx_union).union(list(cci_df_dct[key].index.values[b])))
        cci_idx_lst = idx_union
    
    tlst = cci_idx_lst
    slst = list(cci_df_dct.keys())
    
    df = pd.DataFrame(index = slst, columns = tlst)
    df.loc[:,:] = None
    
    for s in slst:
        dfv = cci_df_dct[s]
        idx = list(dfv.index.values)
        clst = list(set(tlst).intersection(idx))
        df.loc[s, clst] = dfv.loc[clst, 'mean'].astype(float)   
                
    df = df.astype(float)
    return df
    

def get_cci_means( cci_df_dct, cci_idx_lst = None, 
                   cells = [], genes = [], pval_cutoff = 0.05 ):

    return cci_get_means( cci_df_dct, cci_idx_lst, 
                          cells, genes, pval_cutoff)


def cci_get_diff_interactions( df_cci_sample, sample_group_map, 
                               ref_group = None, pval_cutoff = 0.05 ):
    df = df_cci_sample
    group = []
    for k in list(df.index.values):
        group.append( sample_group_map[k] )
    glst = list(set(group))
    glst.sort()
    group = pd.Series(group, index = df.index)
    
    df_res = pd.DataFrame(index = list(df.columns.values))
    
    if ref_group is None:
        for i, g1 in enumerate(glst):
            
            b_r = (group == g1) # & (~df[item].isnull())
            n_r = np.sum(b_r)
            if n_r > 0:
                for j, g2 in enumerate(glst):
                    if i > j:
                        b_o = (group == g2) # & (~df[item].isnull())
                        n_o = np.sum(b_o)
                        if n_o > 0:
                            res = stats.ttest_ind(df.loc[b_r,:], 
                                                  df.loc[b_o,:], axis=0, 
                                                  equal_var=False, # (n_r == n_o), 
                                                  nan_policy='omit', # 'propagate', 
                                                  permutations=None, 
                                                  random_state=None, 
                                                  alternative='two-sided', 
                                                  trim=0)
                            df_res.loc[:, '%s_vs_%s' % (g1, g2)] = res.pvalue
            
    elif ref_group in glst:
    
        b_ref = group == ref_group
        for g in glst:
            if g != ref_group:
                b_r = (group == g) # & (~df[item].isnull()) 
                n_r = np.sum(b_r)
    
                b_o = b_ref # & (~df[item].isnull())
                n_o = np.sum(b_o)
    
                res = stats.ttest_ind(df.loc[b_r,:], 
                                      df.loc[b_o,:], axis=0, 
                                      equal_var=False, # (n_r == n_o), 
                                      nan_policy='omit', # 'propagate', 
                                      permutations=None, 
                                      random_state=None, 
                                      alternative='two-sided', 
                                      trim=0)
                df_res.loc[:, '%s_vs_%s' % (g, ref_group)] = res.pvalue
    
    b = df_res.isnull()
    df_res[b] = 1
    
    pv_min = df_res.min(axis = 1)
    df_res = df_res.loc[pv_min <= pval_cutoff,:]

    df_res['ss'] = (df_res <= pval_cutoff).sum(axis = 1)
    df_res['pp'] = -np.log10(df_res.prod(axis = 1))
    df_res.sort_values(by = ['ss', 'pp'], ascending = False, inplace = True)
       
    return df_res.iloc[:,:-2]


def test_group_diff( df_sample_by_items, sample_group_map, 
                     ref_group = None, pval_cutoff = 0.05 ):
    
    return cci_get_diff_interactions( df_sample_by_items, sample_group_map, 
                               ref_group, pval_cutoff )


### Remove common CCI
def cci_remove_common( df_dct ):
    
    idx_dct = {}
    idxo_dct = {}
    celltype_lst = []

    for j, g in enumerate(df_dct.keys()):
        idx_dct[g] = list(df_dct[g].index.values)
        if j == 0:
            idx_c = idx_dct[g]
        else:
            idx_c = list(set(idx_c).intersection(idx_dct[g]))

        ctA = list(df_dct[g]['cell_A'].unique())
        ctB = list(df_dct[g]['cell_B'].unique())
        celltype_lst = list(set(celltype_lst).union(ctA + ctB))

    for g in df_dct.keys():
        idxo_dct[g] = list(set(idx_dct[g]) - set(idx_c))

    dfs_dct = {}
    for g in df_dct.keys():
        dfs_dct[g] = df_dct[g].loc[idxo_dct[g],:]

    celltype_lst.sort()
    # len(idx_c), celltype_lst
    
    return dfs_dct


## Get matrices summarizing the num CCIs for each condition
def cci_get_ni_mat( df_dct, remove_common = True ):
    
    if remove_common: 
        dfs_dct = cci_remove_common( df_dct )
    else:
        dfs_dct = df_dct
        
    celltype_lst = []
    for j, g in enumerate(dfs_dct.keys()):
        ctA = list(dfs_dct[g]['cell_A'].unique())
        ctB = list(dfs_dct[g]['cell_B'].unique())
        celltype_lst = list(set(celltype_lst).union(ctA + ctB))

    celltype_lst.sort()
    
    df_lst = {} 
    for g in dfs_dct.keys():
        b = dfs_dct[g]['cell_A'].isin(celltype_lst) & (dfs_dct[g]['cell_B'].isin(celltype_lst))
        dfs = dfs_dct[g].loc[b,:]
        df = pd.DataFrame(index = celltype_lst, columns = celltype_lst)
        df.loc[:] = 0
        for a, b in zip(dfs['cell_A'], dfs['cell_B']):
            df.loc[a,b] += 1

        df_lst[g] = df
                
    return df_lst

def get_cci_ni_mat( df_dct, remove_common = True ):
    return cci_get_ni_mat( df_dct, remove_common )


def cci_get_df_to_plot( df_dct, pval_cutoff = 0.01, mean_cutoff = 1, target_cells = None ):

    idx_lst_all = []
    for k in df_dct.keys():
        b = df_dct[k]['pval'] <= pval_cutoff
        b = b & df_dct[k]['mean'] >= mean_cutoff
        if target_cells is not None:
            if isinstance(target_cells, list):
                b1 = df_dct[k]['cell_A'].isin(target_cells)
                b2 = df_dct[k]['cell_B'].isin(target_cells)
                b = b & (b1 | b2)
                
        idx_lst_all = idx_lst_all + list(df_dct[k].index.values[b])

    idx_lst_all = list(set(idx_lst_all))
    display('Union of Interactions: %i' % len(idx_lst_all))    

    df_dct_to_plot = {}
    for k in df_dct.keys():
        df = df_dct[k]
        dfv = pd.DataFrame(index = idx_lst_all, columns = df.columns)
        dfv['mean'] = 0
        dfv['pval'] = 1

        idxt = list(set(idx_lst_all).intersection(list(df.index.values)))
        cols = list(df.columns.values)
        cols.remove('mean')
        cols.remove('pval')
        dfv.loc[idxt, cols] = df.loc[idxt, cols]
        dfv.loc[idxt, 'mean'] = df.loc[idxt, 'mean']
        dfv.loc[idxt, 'pval'] = df.loc[idxt, 'pval']

        gp, cp, ga, gb, ca, cb = [], [], [], [], [], []
        for s in list(dfv.index.values):
            gpt, cpt, gat, gbt, cat, cbt = cpdb_get_gp_n_cp(s)

            gp = gp + [gpt]
            cp = cp + [cpt]
            ga = ga + [gat]
            gb = gb + [gbt]
            ca = ca + [cat]
            cb = cb + [cbt]

        dfv['gene_pair'] = gp
        dfv['cell_pair'] = cp
        dfv['gene_A'] = ga
        dfv['gene_B'] = gb
        dfv['cell_A'] = ca
        dfv['cell_B'] = cb
        
        df_dct_to_plot[k] = dfv
        # print(dfv.shape)
        
    return df_dct_to_plot


def get_markers_from_deg( df_dct, ref_col = 'score',  N_mkrs = 30, 
                          nz_pct_test_min = 0.5, nz_pct_ref_max = 0.1,
                          rem_common = True ):
## Get markers from DEG results

    df_deg = df_dct
    mkr_dict = {}
    b = True
    for key in df_deg.keys():
        if ref_col not in list(df_deg[key].columns.values):
            b = False
            break
    
    if not b:
        print('ERROR: %s not found in column name of DEG results.' % ref_col)
        return None

    for key in df_deg.keys():

        g = key.split('_vs_')[0]
        df = df_deg[key].copy(deep = True)
        b1 = df['nz_pct_test'] >= nz_pct_test_min
        b2 = df['nz_pct_ref'] <= nz_pct_ref_max
        df = df.loc[b1&b2, : ]
        df = df.sort_values([ref_col], ascending = False)

        mkr_dict[g] = list(df.iloc[:N_mkrs].index.values)

    ## Remove common markers
    if rem_common:
        lst = list(mkr_dict.keys())
        cmm = []
        for j, k1 in enumerate(lst):
            for i, k2 in enumerate(lst):
                if (k1 != k2) & (j < i):
                    lc = list(set(mkr_dict[k1]).intersection(mkr_dict[k2]))
                    cmm = cmm + lc
        cmm = list(set(cmm))

        for j, k in enumerate(lst):
            mkr_dict[k] = list(set(mkr_dict[k]) - set(cmm))

    return mkr_dict


def filter_gsa_result( dct, neg_log_p_cutoff ):
    
    dct_t = {}
    for kk in dct.keys():
        dft = dct[kk]
        b = dft['-log(p-val)'] >= neg_log_p_cutoff
        dct_t[kk] = dft.loc[b,:]
    return dct_t


### Example 
# lst = [adata_t.obs['cell_type_major_pred'], adata_t.obs['tumor_dec'], adata_t.obs['subtype']] #, 
# plot_sankey_e( lst, title = None, fs = 12, WH = (800, 600), th = 0, title_y = 0.85 )

def plot_sankey_e( lst, title = None, fs = 12, WH = (700, 800), th = 0, title_y = 0.85 ):
    
    if len(lst) < 2:
        print('ERROR: Input must have length of 2 or more.')
        return
    
    W = WH[0]
    H = WH[1]

    sall = []
    tall = []
    vall = []
    lbl_all = []
    label_lst = []
    
    nn_cnt = 0
    for nn in range(len(lst)-1):
        source_in = ['L%i_%s' % (nn+1, a) for a in list(lst[nn])]
        target_in = ['L%i_%s' % (nn+2, a)  for a in list(lst[nn+1])]
                
        source = pd.Series(source_in).astype(str)
        b = source.isna()
        source[b] = 'N/A'
        target = pd.Series(target_in).astype(str)
        b = target.isna()
        target[b] = 'N/A'

        src_lst = list(set(source))
        tgt_lst = list(set(target))
        src_lst.sort()
        tgt_lst.sort()

        if th > 0:
            bx = np.full(len(source), True)
            for k, s in enumerate(src_lst):
                bs = source == s
                for m, t in enumerate(tgt_lst):
                    bt = target == t
                    b = bs & bt
                    if np.sum(b) < th:
                        bx[b] = False
            source = source[bx]
            target = target[bx]

            src_lst = list(set(source))
            tgt_lst = list(set(target))
            src_lst.sort()
            tgt_lst.sort()

        src = []
        tgt = []
        val = []
        sl_lst = []
        tl_lst = []
        Ns = len((src_lst))
        for k, s in enumerate(src_lst):
            bs = source == s
            for m, t in enumerate(tgt_lst):
                bt = target == t
                b = bs & bt
                if (np.sum(b) > 0):
                    if s not in sl_lst:
                        sn = len(sl_lst)
                        sl_lst.append(s)
                    else:
                        for n, lbl in enumerate(sl_lst):
                            if s == lbl:
                                sn = n
                                break

                    if t not in tl_lst:
                        tn = len(tl_lst) + Ns
                        tl_lst.append(t)
                    else:
                        for n, lbl in enumerate(tl_lst):
                            if t == lbl:
                                tn = n + Ns
                                break

                    src.append(sn)
                    tgt.append(tn)
                    val.append(np.sum(b))
                    label_lst = sl_lst + tl_lst
                    nn_cnt += 1

        if (nn == 0): # | (nn_cnt == 0):
            src2 = src
            tgt2 = tgt
        else:
            lbl_ary = np.array(lbl_all)
            sseq = np.arange(len(lbl_ary))
            src2 = []
            for a in src:
                s = sl_lst[a]
                b = lbl_ary == s
                if np.sum(b) == 1:
                    m = sseq[b][0]
                    src2.append(m)
                else:
                    print('ERROR ... S')
            
            # src2 = [(a + len(lbl_all) - len(sl_lst)) for a in src]
            tgt2 = [(a + len(lbl_all) - len(sl_lst)) for a in tgt]
        
        sall = sall + copy.deepcopy(src2)
        tall = tall + copy.deepcopy(tgt2)
        vall = vall + copy.deepcopy(val)
        if nn == 0:
            lbl_all = copy.deepcopy(label_lst)
        else:
            lbl_all = lbl_all + copy.deepcopy(tl_lst)
    '''
    mat = np.array([sall,tall,vall])
    print(mat)
    print(lbl_all)
    '''      
        
    link = dict(source = sall, target = tall, value = vall)
    node = dict(label = lbl_all, pad=50, thickness=5)
    data = go.Sankey(link = link, node=node)
    layout = go.Layout(height = H, width = W)
    # plot
    fig = go.Figure(data, layout)
    if title is not None:
        fig.update_layout(
            title={
                'text': '<span style="font-size: ' + '%i' % fs + 'px;">' + title + '</span>',
                'y':title_y,
                'x':0.5,
                # 'font': 12,
                'xanchor': 'center',
                'yanchor': 'top'})    
    fig.show()   
    return


### Example 
# plot_population(df_pct, figsize=(6, 4), dpi = 80, return_fig = False)
# df_cnt
    
def plot_population(df_pct, bar_width = 0.6, title = None, title_fs = 12, ylabel = None,
                    label_fs = 11, tick_fs = 10, xtick_rot = 45, xtick_ha = 'center',
                    legend_fs = 10, legend_loc = 'upper left', bbox_to_anchor = (1,1), 
                    legend_ncol = 1, cmap_name = None, figsize=(5, 3), dpi = 100):    

    if cmap_name is None:
        cmap_name = 'Spectral'
    cmap = plt.get_cmap(cmap_name)
    color = cmap(np.arange(df_pct.shape[1])/df_pct.shape[1])

    fig = plt.figure(dpi = dpi)
    ax = df_pct.plot.bar(stacked = True, rot = xtick_rot, figsize = figsize, 
                         color = color, width = bar_width)
    ax.legend( list(df_pct.columns.values), bbox_to_anchor=bbox_to_anchor, 
               loc = legend_loc, fontsize = legend_fs, ncol = legend_ncol )
    plt.xticks(fontsize=tick_fs, ha = xtick_ha)
    plt.yticks(fontsize=tick_fs)
    if isinstance(ylabel, str):
        plt.ylabel(ylabel, fontsize = label_fs)
    plt.title(title, fontsize = title_fs)
    plt.show()
    return


def plot_population_grouped(df_pct, sample_group_map, sort_by = [], bar_width = 0.6, 
                            title = None, title_y_pos = 1.1, title_fs = 14, ylabel = None,
                            label_fs = 11, tick_fs = 10, xtick_rot = 45, xtick_ha = 'center',
                            legend_fs = 10, legend_loc = 'upper left', bbox_to_anchor = (1,1), 
                            legend_ncol = 1, cmap_name = None, figsize=(5, 3), dpi = 100,
                            wspace=0.06, hspace=0.25 ):    

    if cmap_name is None:
        cmap_name = 'Spectral'
    cmap = plt.get_cmap(cmap_name)
    color = cmap(np.arange(df_pct.shape[1])/df_pct.shape[1])
    
    df = df_pct.copy(deep = True)

    items = list(df.columns.values)

    df['Group'] = list(df.index.values)
    df['Group'].replace(sample_group_map, inplace = True)

    num_p = []

    glst = list(df['Group'].unique())
    glst.sort()

    cnt = df['Group'].value_counts()
    for g in glst:
        num_p.append(cnt.loc[g])

    nr, nc = 1, len(glst)
    fig, axes = plt.subplots(nrows=nr, ncols=nc, constrained_layout=True, 
                             gridspec_kw={'width_ratios': num_p}, dpi = dpi)
    fig.tight_layout() 
    if title is not None: 
        fig.suptitle(title, x = 0.5, y = title_y_pos, fontsize = title_fs, ha = 'center')
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=wspace, hspace=hspace)

    cnt = 0
    for j, g in enumerate(glst):
        b = df['Group'] == g
        dft = df.loc[b,:]
        if len(sort_by) > 0:
            dft = dft.sort_values(sort_by)

        if nc <= 1: ax_sel = axes
        else: ax_sel = axes[j+cnt]
            
        ax = dft.plot.bar(width = bar_width, stacked = True, ax = ax_sel, 
                          figsize = figsize, legend = None, color = color)
        ax.set_title('%s' % (g), fontsize = title_fs-2)
        if j != 0:
            # ax.set_yticks([])
            pass
        else:
            if isinstance(ylabel, str):
                ax.set_ylabel(ylabel, fontsize = label_fs)

        ax.tick_params(axis='x', labelsize=tick_fs) # , rotation = xtick_rot)
        ax.tick_params(axis='y', labelsize=tick_fs)
        labels = ax.get_xticklabels()
        ax.set_xticklabels(labels, rotation=xtick_rot, ha=xtick_ha)
            
        if g == glst[-1]: 
            ax.legend(dft.columns.values, loc = legend_loc, bbox_to_anchor = bbox_to_anchor, 
                       fontsize = legend_fs, ncol = legend_ncol)  
        else:
            pass
    plt.show()
    
    return


def plot_pct_box( df_pct, sg_map, ncols, figsize = None, dpi = 100,
                  title = None, title_y_pos = 1.05, title_fs = 14, 
                  label_fs = 11, tick_fs = 10, xtick_rot = 0, xtick_ha = 'center', group_order = None,
                  annot = True, annot_ref = None, annot_fmt = 'simple', annot_fs = 10, 
                  ws_hs = (0.3, 0.3), ylabel = None, stripplot = True ):

    title2_fs = None
    if title2_fs is None:
        title2_fs = title_fs - 2
    df = df_pct.copy(deep = True)
    # nr, nc = nr_nc
    nc = ncols
    nr = int(np.ceil(df.shape[1]/nc))
    if figsize is None:
        figsize = (3*nc,3*nr)
    else:
        figsize = (figsize[0]*nc,figsize[1]*nr)
        
    ws, hs = ws_hs
    fig, axes = plt.subplots(figsize = figsize, dpi=dpi, nrows=nr, ncols=nc, constrained_layout=True)
    fig.tight_layout() # Or equivalently,  "plt.tight_layout()"
    if title is not None:
        fig.suptitle('%s' % title, x = 0.5, y = title_y_pos, fontsize = title_fs, ha = 'center')
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=ws, hspace=hs)

    items = list(df.columns.values)

    df['Group'] = list(df.index.values)
    df['Group'].replace(sg_map, inplace = True)

    lst = df['Group'].unique()

    for kk, item in enumerate(items):
        plt.subplot(nr,nc,kk+1)
        b = df[item].isnull()
        pcnt = df.loc[~b,'Group'].value_counts()
        b1 = pcnt > 1
        group_sel = pcnt.index.values[b1].tolist()

        if len(group_sel) > 1:
            b = (~b) & df['Group'].isin(group_sel)
            
            lst = df.loc[b, 'Group'].unique()
            lst_pair = []
            if annot_ref in lst:
                for k, l1 in enumerate(lst):
                    if l1 != annot_ref:
                        lst_pair.append((annot_ref, l1))
            else:
                for k, l1 in enumerate(lst):
                    for j, l2 in enumerate(lst):
                        if j <  k:
                            lst_pair.append((l1, l2))
        
            ax = sns.boxplot(data = df.loc[b,:], x = 'Group', y = item, order = group_order)
            if stripplot:
                sns.stripplot(x='Group', y=item, data=df.loc[b,:], jitter=True, 
                              color="black", order = group_order)
            if kk%nc == 0: 
                ax.set_ylabel(ylabel, fontsize = label_fs)                
            else: 
                ax.set_ylabel(None)
                
            if kk >= nc*(nr-1): 
                ax.set_xlabel('Condition', fontsize = label_fs)
            else: 
                ax.set_xlabel(None)
            ax.set_title(item, fontsize = title2_fs)
            
            if kk < (nr*nc - nc):
                # plt.xticks([])
                plt.xticks(rotation = xtick_rot, ha = xtick_ha, fontsize = tick_fs)
                plt.yticks(fontsize = tick_fs)
            else:
                plt.xticks(rotation = xtick_rot, ha = xtick_ha, fontsize = tick_fs)
                plt.yticks(fontsize = tick_fs)
    
            if annot:
                add_stat_annotation(ax, data=df.loc[b,:], x = 'Group', y = item, 
                                    order = group_order,
                            box_pairs=lst_pair, loc='inside', fontsize = annot_fs,
                            test='t-test_ind', text_format=annot_fmt, verbose=0)
            #'''
    
    if (len(items) < (nr*nc)) & (nr > 1):
        for k in range(nr*nc - len(items)):
            axes[nr-1][len(items)%nc + k].axis("off")
        
    plt.show()
    return 


def plot_violin( df, genes_lst, group_col = 'group', scale = 'width', group_order = None, 
                 inner = 'box', width = 0.9, linewidth = 0.3, bw = 'scott', 
                 figsize = (3,2), dpi = 100, text_fs = 10, title = 'Title', title_fs = 14, title_y_pos = 1, 
                 label_fs = 11, tick_fs = 10, xtick_rot = 0, xtick_ha = 'center', 
                 ncols = 2, wspace = 0.15, hspace = 0.4 ):

    nr, nc = int(np.ceil(len(genes_lst)/ncols)), int(ncols) # len(df_deg_dct.keys())
    fig, axes = plt.subplots(figsize = (figsize[0]*nc,figsize[1]*nr), nrows=nr, ncols=nc, # constrained_layout=True, 
                             gridspec_kw={'width_ratios': [1]*nc}, dpi = dpi)
    fig.tight_layout() 
    if title is not None:
        fig.suptitle(title, x = 0.5, y = title_y_pos, fontsize = title_fs, ha = 'center')
        
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, 
                        wspace=wspace, hspace=hspace)
    cnt = 0
    for j, gene in enumerate(genes_lst): 
    
        plt.subplot(nr,nc, int(j+1))
        sns.violinplot(x = group_col, y = gene, data = df, scale = scale, 
                       order = group_order, width = width, inner = inner, bw = bw)
        '''
        plt.title('%s' % (gene), fontsize = label_fs + 1)
        plt.yticks(fontsize = tick_fs)
        plt.xticks(None, rotation = xtick_rot, ha = xtick_ha, fontsize = tick_fs)
        plt.xlabel(None)
        if j%nc == 0: plt.ylabel('Log exp.', fontsize = label_fs)
        else: plt.ylabel(None)
        '''
        plt.title('%s' % (gene), fontsize = label_fs + 1)
        if j%nc == 0: plt.ylabel('Log exp.', fontsize = label_fs)
        else: plt.ylabel(None)
            
        if j >= nc*(nr-1): plt.xlabel('Condition', fontsize = label_fs)
        else: plt.xlabel(None)
        if j < (nr*nc - nc):
            plt.xticks([])
            plt.yticks(fontsize = tick_fs)
        else:
            plt.xticks(rotation = xtick_rot, ha = xtick_ha, fontsize = tick_fs)
            plt.yticks(fontsize = tick_fs)
        
        cnt += 1

    if nc*nr > len(genes_lst):
        for j in range(nc*nr-len(genes_lst)):
            k = j + len(genes_lst) + 1
            ax = plt.subplot(nr,nc,k)
            ax.axis('off')

    plt.show()
    return


## CPDB plot

###################################
### Plot functions for CellPhoneDB

def cci_get_all_pairs( cci_df_dct ):
    
    lst = list(cci_df_dct.keys())
    lst.sort()
    
    gps = []
    cps = []
    for g in lst:
        dfv = cci_df_dct[g]
        gps = gps + list(dfv['gene_pair'])
        cps = cps + list(dfv['cell_pair'])
    
    gps = list(set(gps))
    cps = list(set(cps))

    return gps, cps


def plot_cci_dot( dfv, n_gene_pairs = None, 
                  pval_cutoff = 0.05, mean_cutoff = 0.1, 
                  target_cells = [], target_genes = [], 
                  legend_fs = 11, legend_mkr_sz = 6, 
                  tick_fs = 6, xtick_rot = 90, xtick_ha = 'center', 
                  title = None, title_fs = 14, dot_size_max = 1,
                  dpi = 100, swap_ax = False, cmap = 'Reds',
                  gene_pair_lst = None, cell_pair_lst = None ):
                  # plot_all_gene_pairs = False, plot_all_cell_pairs = False ):

    '''
    gene_pair_lst = None
    cell_pair_lst = None
    if plot_all_gene_pairs | plot_all_cell_pairs:
        gps, cps = cci_get_all_pairs( cci_df_dct )
        if plot_all_gene_pairs: gene_pair_lst = gps
        if plot_all_cell_pairs: cell_pair_lst = cps
    #'''
    
    dfs = dfv.copy(deep = True)
    
    if isinstance(target_cells, list):
        if (len(target_cells) > 0):
            b = (dfs['cell_A'].isin(target_cells) & dfs['cell_B'].isin(target_cells))
            dfs = dfs.loc[b,:]
        
    if isinstance(target_genes, list):
        if (len(target_genes) > 0):
            b = (dfs['gene_A'].isin(target_genes) | dfs['gene_B'].isin(target_genes))
            dfs = dfs.loc[b,:]
    
    if n_gene_pairs is None:
        dfs = dfs.sort_values(by = ['gene_pair', 'cell_pair'])
        b = dfs['pval'] <= pval_cutoff
        b = b & dfs['mean'] >= mean_cutoff    
        dfp = dfs.loc[b,:]
    else:
        dfs['neg_mean'] = -dfs['mean']
        dfp = dfs.sort_values(by = ['pval', 'neg_mean'], ascending = True)
        ilst = dfp['gene_pair'].tolist()
        ilst2 = []
        j = 0
        for j, i in enumerate(ilst):
            if len(list(set(ilst[:j]))) > n_gene_pairs:
                break                
        dfp = dfp.iloc[:j]
    
    if dfp.shape[0] == 0:
        print('WARNING: no interactions met the conditions you provide. ')
        return
    else:
        dfp = dfp.sort_values(by = ['gene_pair', 'cell_pair'])
        if swap_ax == False:
            a = 'gene_pair'
            b = 'cell_pair'
            apl = gene_pair_lst
            bpl = cell_pair_lst
        else:
            b = 'gene_pair'
            a = 'cell_pair'
            apl = cell_pair_lst 
            bpl = gene_pair_lst

        if apl is None:
            y = len(set(dfp[a]))
        else:
            y = len(apl)
            
        if bpl is None:
            x = len(set(dfp[b]))
        else:
            x = len(bpl)
        
        print('%i %ss, %i %ss found' % (y, a, x, b))
        
        pv = -np.log10(dfp['pval']+1e-10).round()
        pvmn, pvmx = int(np.min(pv)), int(np.max(pv))
        
        mn = np.log2((1+dfp['mean']))
        np.min(mn), np.max(mn)    
        
        w = x/6
        sc.settings.set_figure_params(figsize=(w, w*(y/x)), 
                                      dpi=dpi, facecolor='white')
        fig, ax = plt.subplots()

        if apl is None:
            a_lst = list(dfp[a].unique())
        else:
            a_lst = apl
        a_lst.sort()
        
        if bpl is None:
            b_lst = list(dfp[b].unique())
        else:
            b_lst = bpl
        b_lst.sort()
        
        ax.autoscale(False)
        # Set the sorted y-ticks.
        La = len(a_lst)
        ax.set_yticks(np.arange(La))
        ax.set_yticklabels(a_lst)
        ax.set_ylim([-1, La])
        y_map = dict(zip(a_lst, list(np.arange(La))))
        
        # Set the x-ticks.
        Lb = len(b_lst)
        ax.set_xticks(np.arange(Lb))
        ax.set_xticklabels(b_lst)  # `ax.set_xticklabels("abcde")` would work too.
        ax.set_xlim([-1, Lb])
        x_map = dict(zip(b_lst, list(np.arange(Lb))))
        
        mul = legend_mkr_sz*dot_size_max*0.6
        lst_b = [x_map[x] for x in list(dfp[b])]
        lst_a = [y_map[y] for y in list(dfp[a])]
        
        scatter = ax.scatter(lst_b, lst_a, s = pv*mul, c = mn, 
                             linewidth = 0, cmap = cmap)
    
        kw = dict(prop="colors", num=5, fmt="{x:.1f}", # color=scatter.cmap(0.7), 
                  func=lambda s: (s))
        legend1 = ax.legend(*scatter.legend_elements(**kw),
                            loc='upper left', 
                            bbox_to_anchor=(1+1/x, 0.5), 
                            title=' log2(m) ', 
                            fontsize = legend_fs)
        legend1.get_title().set_fontsize(legend_fs)
        ax.add_artist(legend1)
    
        # produce a legend with a cross section of sizes from the scatter
        kw = dict(prop="sizes", num=4, fmt="{x:.0f}", # color=scatter.cmap(0.7), 
                  func=lambda s: (s/mul))
        legend2 = ax.legend(*scatter.legend_elements(**kw),
                            loc='lower left', 
                            bbox_to_anchor=(1+1/x, 0.5), 
                            title='-log10(p)', 
                            fontsize = legend_fs)        
        legend2.get_title().set_fontsize(legend_fs)
    
        if title is not None: plt.title(title, fontsize = title_fs)
        plt.yticks(fontsize = tick_fs)
        plt.xticks(rotation = xtick_rot, ha=xtick_ha, fontsize = tick_fs)
        plt.margins(x=0.6/x, y=0.6/y)
        plt.show()   
    return


def cpdb_get_gp_n_cp(idx):
    
    items = idx.split('--')
    gpt = items[0]
    cpt = items[1]
    gns = gpt.split('_')
    ga = gns[0]
    gb = gns[1]
    cts = cpt.split('|')
    ca = cts[0]
    cb = cts[1]
    
    return gpt, cpt, ga, gb, ca, cb
    
###########################    
### Circle plot for CCI ###
    
def center(p1, p2):
    return (p1[0]+p2[0])/2, (p1[1]+p2[1])/2

def norm( p ):
    n = np.sqrt(p[0]**2 + p[1]**2)
    return n

def vrot( p, s ):
    v = (np.array([[0, -1], [1, 0]]).dot(np.array(p)))
    v = ( v/norm(v) )
    return v #, v2
    
def vrot2( p, t ):
    v = (np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]).dot(p))
    return v #, v2
    
def get_arc_pnts( cp, R, t1, t2, N):
    
    a = (t1 - t2)
    if a >= math.pi:
        t2 = t2 + 2*math.pi
    elif a <= -math.pi:
        t1 = t1 + 2*math.pi
    
    N1 = (N*np.abs(t1 - t2)/(2*math.pi))
    # print(N1)
    s = np.sign(t1 - t2)
    t = t2 + np.arange(N1+1)*(s*2*math.pi/N)
    # if t.max() > (math.pi*2): t = t - math.pi*2
    
    x = np.sin(t)*R + cp[0]
    y = np.cos(t)*R + cp[1]
    x[-1] = np.sin(t1)*R + cp[0]
    y[-1] = np.cos(t1)*R + cp[1]
        
    return x, y, a

def get_arc( p1, p2, R, N ):
    
    A = norm(p1 - p2)
    pc = center(p1, p2)
    
    a = np.sqrt((R*A)**2 - norm(p1 - pc)**2)
    c = pc + vrot(p1 - pc, +1)*a

    d1 = p1 - c
    t1 = np.arctan2(d1[0], d1[1])
    d2 = p2 - c
    t2 = np.arctan2(d2[0], d2[1])

    x, y, t1 = get_arc_pnts( c, R*A, t2, t1, N)
    
    return x, y, c


def get_circ( p1, R, N ):
    
    pp = np.arange(N)*(2*math.pi/N)
    px = np.sin(pp)*0.5
    py = np.cos(pp)
    pnts = np.array([px, py])
    
    t = -np.arctan2(p1[0], p1[1])
    pnts = vrot2( pnts, t+math.pi )*R
    pnts[0,:] = pnts[0,:] + p1[0]*(1+R)
    pnts[1,:] = pnts[1,:] + p1[1]*(1+R)
    x = pnts[0,:]
    y = pnts[1,:]
    c = np.array([0,0])
    
    return x, y, c


def plot_cci_circ( df_in, figsize = (10, 10), dpi = 10, title = None, title_fs = 16, 
                   text_fs = 14, num_fs = 12, margin = 0.08, alpha = 0.5, 
                   linewidth_max = 10, linewidth_log = False, 
                   R_curvature = 4, node_size = 10, rot = True, 
                   cmap = 'Spectral', ax = None, show = True):
            
    linewidth_scale = 0
    log_lw = linewidth_log
    lw_max = linewidth_max
    lw_scale = linewidth_scale
    
    Rs = 0.1
    N = 500
    R = R_curvature
    df = df_in.copy(deep = True)
    mxv = df_in.max().max()
    
    if ax is None: 
        plt.figure(figsize = figsize, dpi = dpi)
        ax = plt.gca()
        
    # color_lst = ['orange', 'navy', 'green', 'gold', # 'lime', 'magenta', \
    #         'turquoise', 'red', 'royalblue', 'firebrick', 'gray']
    # color_map = cm.get_cmap(cmap)
    color_map = colormaps[cmap]
    if df.shape[0] > 1:
        color_lst = [color_map(i/(df.shape[0]-1)) for i in range(df.shape[0])]
    else:
        mmm = 10
        color_lst = [color_map(i/(mmm-1)) for i in range(mmm)]

    clst = list(df.index.values) 

    M = df.shape[0]
    if M > 1:
        pp = np.arange(M)*(2*math.pi/M)
    else:
        pp = np.arange(mmm)*(2*math.pi/mmm)
    px = np.sin(pp)
    py = np.cos(pp)
    pnts = np.array([px, py])
    
    for j in range(M):
        p1 = pnts[:,j]
        for k in range(M):
            p2 = pnts[:,k]
            
            val = df.loc[clst[j], clst[k]]
            if lw_scale > 0:
                lw = val*lw_scale
            elif lw_max > 0:
                lw = val*lw_max/mxv
            else:
                lw = val
            if log_lw: lw = np.log2(lw)                    
                    
            if (df.loc[clst[j], clst[k]] != 0): # & (j!= k):

                if j == k:
                    x, y, c = get_circ( p1, 0.1, N )
                    K = int(len(x)*0.5)
                    d = vrot(p1, 1)
                    d = d*0.05/norm(d)
                elif (j != k) :
                    x, y, c = get_arc( p1, p2, R, N )

                    K = int(len(x)*0.5)
                    d = (p2 - p1)
                    d = d*0.05/norm(d)

                q2 = np.array([x[K], y[K]])
                q1 = np.array([x[K] - d[0], y[K] - d[1]])

                s = norm(q1 - q2)
                
                ha = 'center'
                if c[0] < -1:
                    ha = 'left'
                elif c[0] > 1:
                    ha = 'right'
                    
                va = 'center'
                if c[1] < -1:
                    va = 'bottom'
                elif c[1] > 1:
                    va = 'top'
                    
                if norm(q2) <= 0.7: # mnR*2:
                    ha = 'center'
                    va = 'center'
                    
                if ax is None: 
                    plt.plot(x, y, c = color_lst[j], linewidth = lw, alpha = alpha)
                    if j != k:
                        plt.arrow(q1[0], q1[1], q2[0]-q1[0], q2[1]-q1[1], linewidth = lw,
                              head_width=s/2, head_length=s, 
                              fc=color_lst[j], ec=color_lst[j])
                    plt.text( q2[0], q2[1], ' %i ' % val, fontsize = num_fs, 
                              va = va, ha = ha)
                else:
                    ax.plot(x, y, c = color_lst[j], linewidth = lw, alpha = alpha)
                    if j != k:
                        ax.arrow(q1[0], q1[1], q2[0]-q1[0], q2[1]-q1[1], linewidth = lw,
                              head_width=0.05*lw/lw_max, head_length=s, alpha = alpha, 
                              fc=color_lst[j], ec=color_lst[j])
                    ax.text( q2[0], q2[1], '%i' % val, fontsize = num_fs, 
                             va = va, ha = ha)

            elif (df.loc[clst[j], clst[k]] != 0) & (j== k):
                x, y, c = get_circ( p1, Rs, N )
                K = int(len(x)*0.5)
                d = vrot(p1, 1)
                d = d*0.05/norm(d)

                q2 = np.array([x[K], y[K]])
                q1 = np.array([x[K] - d[0], y[K] - d[1]])

                s = norm(q1 - q2)
                
                ha = 'center'
                if c[0] < -1:
                    ha = 'left'
                elif c[0] > 1:
                    ha = 'right'
                    
                va = 'center'
                if c[1] < -1:
                    va = 'bottom'
                elif c[1] > 1:
                    va = 'top'
                    
                if norm(q2) <= 0.7: # mnR*2:
                    ha = 'center'
                    va = 'center'
                    
                if ax is None: 
                    plt.plot(x, y, c = color_lst[j], linewidth = lw, alpha = alpha)
                    plt.arrow(q1[0], q1[1], q2[0]-q1[0], q2[1]-q1[1], linewidth = lw,
                              head_width=s/2, head_length=s, 
                              fc=color_lst[j], ec=color_lst[j])
                    plt.text( q2[0], q2[1], ' %i ' % val, fontsize = num_fs, 
                              va = va, ha = ha)
                else:
                    ax.plot(x, y, c = color_lst[j], linewidth = lw, alpha = alpha)
                    ax.arrow(q1[0], q1[1], q2[0]-q1[0], q2[1]-q1[1], linewidth = lw,
                              head_width=0.05*lw/lw_max, head_length=s, alpha = alpha, 
                              fc=color_lst[j], ec=color_lst[j])
                    ax.text( q2[0], q2[1], '%i' % val, fontsize = num_fs, 
                             va = va, ha = ha)
    if rot:
        rotation = 90 - 180*np.abs(pp)/math.pi
        b = rotation < -90
        rotation[b] = 180+rotation[b]
    else:
        rotation = np.zeros(M)
        
    for j in range(pnts.shape[1]):
        (x, y) = (pnts[0,j], pnts[1,j])
        
        ha = 'center'
        if x < 0:
            ha = 'right'
        else: 
            ha = 'left'
        va = 'center'
        if y == 0:
            pass
        elif y < 0:
            va = 'top'
        else: 
            va = 'bottom'
            
        if j < M:
            ct = clst[j]
            a = (df.loc[ct, ct] != 0)*(Rs*2)
            rot = rotation[j]
            cc = color_lst[j]
        else:
            ct = ''
            a = 0
            rot = 0
            cc = 'gray'
            
        if ax is None: 
            plt.plot( x, y, 'o', ms = node_size, c = cc)
            plt.text( x, y, ' %s ' % ct, fontsize = text_fs, 
                      ha = ha, va = va, rotation = rot)
        else:
            ax.plot( x, y, 'o', ms = node_size, c = cc)
            ax.text( x*(1+a), y*(1+a), '  %s  ' % ct, fontsize = text_fs, 
                     ha = ha, va = va, rotation = rot)

    if ax is None: 
        plt.xticks([])
        plt.yticks([])
        plt.margins(x=margin, y=margin)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.margins(x=margin, y=margin)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False) 
        
    if title is not None: ax.set_title(title, fontsize = title_fs )
        
    if show: plt.show()
    return


def plot_cci_circ_group( df_dct, target = [], 
                         rename = None, remove_common = False,
                         ncol = 3, figsize = (8,7), dpi = 100,
                         title = None, title_y_pos = 1, title_fs = 18, 
                         text_fs = 16, num_fs = 12, margin = 0.08, alpha = 0.5, 
                         linewidth_max = 10, linewidth_log = False,  
                         R_curvature = 3, node_size = 10, rot = False, 
                         cmap = 'Spectral', wspace = 0.35, hspace = 0.2 ):

    df_lst = get_cci_ni_mat( df_dct, remove_common )
    if isinstance(rename, dict):
        for k in df_lst.keys():
            df_lst[k].rename(columns = rename, index = rename, inplace = True)

    if target is None:
        df_lst2 = df_lst
    elif not isinstance(target, list):
        print('ERROR: targte must be a list of celltypes.')
        return 
    elif len(target) == 0:
        df_lst2 = df_lst
    else:
        ## check if target exists in the data
        b = True
        for key in df_lst.keys():
            celltypes = df_lst[key].columns.values.tolist()
            b = len(list(set(target).intersection(celltypes))) == len(target)
            if not b: break

        if not b:
            s = ''
            for t in target:
                s = s + '%s,' % t
            s = s[:-1]
            print('ERROR: %s not exists in the data.' % s)
            return
        else:
            df_lst2 = {}
            for key in df_lst.keys():
                df = df_lst[key]
                msk = df.copy(deep = True)
                msk.loc[:,:] = 0
                msk.loc[target,:] = 1
                msk.loc[:,target] = 1
                df_lst2[key] = df*msk
   
    n_max = 0
    for k in df_lst2.keys():
        n_max = max( df_lst2[k].max().max(), n_max )
    linewidth_scale = linewidth_max/(n_max + 0.01)
    lw_max_dict = {}
    for k in df_lst2.keys():
        lw_max_dict[k] = df_lst2[k].max().max()*linewidth_scale
    
    cond_lst = list(df_lst2.keys())
    cond_lst.sort()
    ws_hs = [wspace, hspace]
    
    R = R_curvature
    N = 500
    nc = ncol
    nr = int(np.ceil(len(cond_lst)/nc))
    # nr, nc = 1, int(len(cond_lst))
    fig, axes = plt.subplots(figsize = (figsize[0]*nc,figsize[1]*nr), nrows=nr, ncols=nc, # constrained_layout=True, 
                             gridspec_kw={'width_ratios': [1]*nc}, dpi = dpi)
    fig.tight_layout() 
    plt.subplots_adjust(left=0.025, bottom=0.025, right=0.975, top=0.975, wspace=ws_hs[0], hspace=ws_hs[1])
    if title is not None: plt.suptitle(title, fontsize = title_fs, y = title_y_pos)

    cnt = 0
    for j, k in enumerate(cond_lst):

        df = df_lst2[k]*(df_lst2[k] > 0).copy(deep = True)
        if df.shape[0] > 0:
            cnt += 1
            plt.subplot(nr,nc,cnt)

            if nr == 1: ax = axes[int(j)]
            else: ax = axes[int(j/nc)][j%nc]

            plot_cci_circ( df, title = k, title_fs = text_fs + 2, 
                           text_fs = text_fs, num_fs = num_fs, margin = margin, alpha = alpha, 
                           R_curvature = R_curvature, 
                           linewidth_max = lw_max_dict[k], # linewidth_max, 
                           # linewidth_scale = 0, # linewidth_scale, 
                           linewidth_log = linewidth_log, 
                           node_size = node_size, rot = rot, cmap = cmap, 
                           ax = ax, show = False)

    if cnt < (nr*nc):
        for k in range(int(nr*nc - cnt)):
            j = k + cnt
            ax = plt.subplot(nr,nc,j+1)
            ax.axis('off')

    plt.show()
    return


def plot_cnv( adata, groupby = 'tumor_dec_rev', 
              title = 'log2(CNR)', title_fs = 14, title_y_pos = 1.1, 
              label_fs = 12, tick_fs = 12, figsize = (12, 6), swap_axes = False, 
              var_group_rotation = 45, cmap='RdBu_r', vmax = 1, 
              cnv_obsm_key = 'X_cnv', cnv_uns_key = 'cnv', 
              xlabel = None):
    
    X_cnv = adata.obsm[cnv_obsm_key]
    chrs = np.array([' ']*X_cnv.shape[1])

    if isinstance(cnv_uns_key, str):
        chr_pos = adata.uns[cnv_uns_key]['chr_pos']
        df_chr_pos = pd.DataFrame({'chr_pos': list(chr_pos.values())}, index = chr_pos.keys())
        df_chr_pos = df_chr_pos.sort_values('chr_pos')
    
        ## Get chromosome name for each loci in X_cnv
        chrs = []
        for i in range(df_chr_pos.shape[0]):
            if i < (df_chr_pos.shape[0]-1):
                start = df_chr_pos.iloc[i][0]
                end = df_chr_pos.iloc[i+1][0]
            else:
                start = df_chr_pos.iloc[i][0]
                end = X_cnv.shape[1]
    
            chrs = chrs + list([df_chr_pos.index.values[i]]*int(end-start))
    
        df_chr = pd.DataFrame({'chr': chrs})
    
        lst = list(df_chr['chr'].unique())
    
        ## Get vg_pos & vg_labels
        cnt = 0
        vg_pos = []
        vg_labels = []
        for i in lst:
            b = df_chr['chr'] == i
            vg_pos.append((cnt, cnt+np.sum(b)-10))
            vg_labels.append(i)
            cnt += np.sum(b)
    
        ## Create AnnData for plot
        ad = anndata.AnnData(X = X_cnv, var = df_chr, obs = adata.obs)
    else:
        vg_pos = None
        vg_labels = None
        ad = anndata.AnnData(X = X_cnv, obs = adata.obs)
        

    X = np.abs(ad.to_df())
    vmax = X.quantile([0.99]).mean(axis = 1)*2

    ax_dict = sc.pl.heatmap(ad, var_names = ad.var.index.values, groupby = groupby, 
                            show = False, figsize = figsize, swap_axes = swap_axes, 
                            var_group_positions = vg_pos, var_group_labels = vg_labels, 
                            var_group_rotation = var_group_rotation, 
                            cmap=cmap, vmax = vmax, vmin = -vmax)

    ax_dict['heatmap_ax'].set_title(title, fontsize = title_fs, y = title_y_pos)
    if swap_axes:
        ax_dict['groupby_ax'].set_xlabel(groupby, fontsize = label_fs)
        ticks = ax_dict['groupby_ax'].get_xticklabels()
        ax_dict['groupby_ax'].set_xticklabels(ticks, fontsize = tick_fs)
        
        if xlabel is not None:
            ax_dict['heatmap_ax'].set_ylabel(xlabel, fontsize = label_fs)
        pass
    else:
        ax_dict['groupby_ax'].set_ylabel(groupby, fontsize = label_fs)
        ticks = ax_dict['groupby_ax'].get_yticklabels()
        ax_dict['groupby_ax'].set_yticklabels(ticks, fontsize = tick_fs)
        
        if xlabel is not None:
            ax_dict['heatmap_ax'].set_xlabel(xlabel, fontsize = label_fs)
            
    plt.show()
    return ax_dict


def get_normal_pdf( x, mu, var, nbins):
    
    MIN_ABS_VALUE = 1e-8
    y = np.array(x)
    mn_x = y.min()
    mx_x = y.max()
    dx = mx_x - mn_x
    mn_x -= dx/4
    mx_x += dx/4
    L = 100
    # dx = len(y)*(mx_x-mn_x)/L
    dx = (mx_x-mn_x)/nbins
    xs = np.arange(mn_x,mx_x, dx )
    pdf = (dx*len(y))*np.exp(-((xs-mu)**2)/(2*var+MIN_ABS_VALUE))/(np.sqrt(2*math.pi*var)+MIN_ABS_VALUE) + MIN_ABS_VALUE
    return pdf, xs


def plot_td_stats( params, n_bins = 30, title = None, title_fs = 14,
                   label_fs = 12, tick_fs = 11, legend_fs = 11, 
                   legend_loc = 'upper left', bbox_to_anchor = (1, 1),
                   figsize = (4,3), log = True, alpha = 0.8 ):
    
    th = params['th']
    m0 = params['m0']
    v0 = params['v0']
    w0 = params['w0']
    m1 = params['m1']
    v1 = params['v1']
    w1 = params['w1']
    df = params['df']
        
    mxv = df['cmean'].max()
    mnv = df['cmean'].min()
    Dv = mxv - mnv
    dv = Dv/200

    x = np.arange(mnv,mxv,dv)
    pdf0, xs0 = get_normal_pdf( x, m0, v0, 100)
    pdf1, xs1 = get_normal_pdf( x, m1, v1, 100)
    
    pr = pdf1/(pdf1 + pdf0) # get_malignancy_prob( xs0, [w0, m0, v0, w1, m1, v1] )
    bx = (xs0 >= m0) & ((xs1 <= m1))

    nn = len(df['cmean'])
    pdf0 = pdf0*(w0*nn*(200/n_bins)/pdf0.sum())
    pdf1 = pdf1*(w1*nn*(200/n_bins)/(pdf1.sum())) 

    max_pdf = max(pdf0.max(), pdf1.max())
    
    plt.figure(figsize = figsize)
    ax = plt.gca()
    
    counts, bins = np.histogram(df['cmean'], bins = n_bins)
    # max_cnt = np.max(counts)

    legend_labels = []
    
    max_cnt = 0
    b = df['tumor_dec'] == 'Normal'
    if np.sum(b) > 0:
        legend_labels.append('Normal')
        counts, bins_t, bar_t = plt.hist(df.loc[b, 'cmean'], bins = bins, alpha = alpha)
        max_cnt = max(max_cnt, np.max(counts))
    b = df['tumor_dec'] == 'Tumor'
    if np.sum(b) > 0:
        legend_labels.append('Tumor')
        counts, bins_t, bar_t = plt.hist(df.loc[b, 'cmean'], bins = bins, alpha = alpha)
        max_cnt = max(max_cnt, np.max(counts))
    b = df['tumor_dec'] == 'unclear'
    if np.sum(b) > 0:
        legend_labels.append('unclear')
        counts, bins_t, bar_t = plt.hist(df.loc[b, 'cmean'], bins = bins, alpha = alpha)
        max_cnt = max(max_cnt, np.max(counts))
    
    sf = 0.9*max_cnt/max_pdf
    plt.plot(xs0, pdf0*sf)
    plt.plot(xs1, pdf1*sf)
    plt.plot([th, th], [0, max_cnt]) # max(pdf0.max()*sf, pdf1.max()*sf)])
    plt.plot(xs0[bx], pr[bx]*max_cnt)

    if title is not None: plt.title(title, fontsize = title_fs)
    plt.xlabel('CNV_score', fontsize = label_fs)
    plt.ylabel('Number of clusters', fontsize = label_fs)
    plt.legend(['Normal distr.', 'Tumor distr.', 'Threshold', 'Tumor Prob.'], #, 'Score hist.'], 
               loc = legend_loc, bbox_to_anchor = bbox_to_anchor, fontsize = legend_fs)
    if log: plt.yscale('log')
    ax.tick_params(axis='x', labelsize=tick_fs)
    ax.tick_params(axis='y', labelsize=tick_fs)
    plt.grid()
    plt.show()
        
    return 
    
### Plot dot

def remove_common( mkr_dict, prn = True ):

    cts = list(mkr_dict.keys())
    mkrs_all = []
    for c in cts:
        mkrs_all = mkrs_all + list(mkr_dict[c])
    mkrs_all = list(set(mkrs_all))
    df = pd.DataFrame(index = mkrs_all, columns = cts)
    df.loc[:,:] = 0

    for c in cts:
        df.loc[mkr_dict[c], c] = 1
    Sum = df.sum(axis = 1)
    
    to_del = []
    s = ''
    for c in cts:
        b = (df[c] > 0) & (Sum == 1)
        mkrs1 = list(df.index.values[b])
        if prn & (len(mkr_dict[c]) != len(mkrs1)):
            s = s + '%s: %i > %i, ' % (c, len(mkr_dict[c]), len(mkrs1))
        
        if len(mkrs1) == 0:
            to_del.append(c)
        else:
            mkr_dict[c] = mkrs1

    if prn & len(s) > 0:
        print(s[:-2])

    if len(to_del) > 0:
        for c in cts:
            if c in to_del:
                del mkr_dict[c]
                
    return mkr_dict


def get_markers_all4(mkr_file, target_lst, genes = None, 
                    rem_cmn = False ):
    
    # target = 'Myeloid cell'
    if isinstance(mkr_file, dict):
        mkr_dict = mkr_file
    else:
        print('ERROR: marker input must be a dictionary.')
        return None
    
    if target_lst is not None:
        if len(target_lst) > 0:
            mkr_dict_new = {}
            for c in target_lst:
                if c in list(mkr_dict.keys()):
                    mkr_dict_new[c] = mkr_dict[c]
            mkr_dict = mkr_dict_new
            
    if rem_cmn:
        mkr_dict = remove_common( mkr_dict, prn = True )
        
    mkrs_all = [] #['SELL']
    mkrs_cmn = []
    for ct in mkr_dict.keys():
        if genes is not None:
            ms = list(set(mkr_dict[ct]).intersection(genes))
        else: 
            ms = mkr_dict[ct]
        mkrs_all = mkrs_all + ms
        if len(mkrs_cmn) == 0:
            mkrs_cmn = ms
        else:
            mkrs_cmn = list(set(mkrs_cmn).intersection(ms))

    mkrs_all = list(set(mkrs_all))
    if genes is not None:
        mkrs_all = list(set(mkrs_all).intersection(genes))
    
    return mkrs_all, mkr_dict


def remove_mac_common_markers(mkrs_dict):   

    lst2 = list(mkrs_dict.keys())
    lst = []
    Mono = None
    for item in lst2:
        if item[:3] == 'Mac':
            lst.append(item)
        if item[:4] == 'Mono':
            Mono = item
            
    if len(lst) > 1:
        mac_common = mkrs_dict[lst[0]]
        for item in lst[1:]:
            mac_common = list(set(mac_common).intersection(mkrs_dict[item]))
            
        for item in lst:
            for mkr in mac_common:
                mkrs_dict[item].remove(mkr)
#         if Mono is not None:
#             mono_lst = mkrs_dict[Mono]
#             del mkrs_dict[Mono]
#             mkrs_dict[Mono] = mono_lst
            
    return mkrs_dict

    
def update_markers_dict2(mkrs_all, mkr_dict, X, y, rend = None, cutoff = 0.3, 
                        Nall = 20, Npt = 20, Npt_tot = 0):
    
    if rend is None:
        lst = list(mkr_dict.keys())
        lst.sort()
        rend = dict(zip(lst, lst))
    else:
        lst = list(rend.keys())
        
    df = pd.DataFrame(index = lst, columns = mkrs_all)
    df.loc[:,:] = 0
        
    for ct in lst:
        if ct in list(mkr_dict.keys()):
            b = y == ct
            ms = list(set(mkr_dict[ct]).intersection(mkrs_all))
            pe = list((X.loc[b,ms] > 0).mean(axis = 0))
            for j, m in enumerate(ms):
                df.loc[ct, m] = pe[j]

    if df.shape[0] == 1:
        mkrs_all = list(df.columns.values)
        mkrs_dict = {}
        
        pe_lst = []
        pex_lst = []
        
        for ct in lst:
            b = y == ct
            ms = list(set(mkr_dict[ct]).intersection(mkrs_all))
            pe = np.array((X.loc[b,ms] > 0).mean(axis = 0))
            pex = np.array((X.loc[~b,ms] > 0).mean(axis = 0))
            odr = np.array(-pe).argsort()
            ms_new = []
            for o in odr:
                if (pe[o] >= cutoff):
                    ms_new.append(ms[o])

            pe = pe[~np.isnan(pe)]
            pex = pex[~np.isnan(pex)]
            pe_lst = pe_lst + list(pe)
            pex_lst = pex_lst + list(pex)
            
            if len(ms_new) > 0:
                mkrs_dict[rend[ct]] = ms_new[:min(Npt,len(ms_new))]
            else:
                mkrs_dict[rend[ct]] = ms[:min(Npt,len(ms))]
                
        mkrs_dict2 = mkrs_dict
    else:
        p1 = df.max(axis = 0)
        p2 = p1.copy(deep = True)
        p2[:] = 0
        idx = df.index.values
        # print(df)
        for m in list(df.columns.values):
            odr = np.array(-df[m]).argsort()
            p2[m] = df.loc[idx[odr[1]], m]
        nn = (df >= 0.5).sum(axis = 0)

        b0 = p1 > 0
        b1 = (p2/(p1 + 0.0001)) < 0.5
        b2 = nn < 4
        b = b0 # & b1 & b2
        df = df.loc[:,b]
        mkrs_all = list(df.columns.values)

        mkrs = [] 
        cts = [] 
        pes = [] 
        mkrs_dict = {}
        pe_dict = {}
        pex_dict = {}
        # mkrs_all = [] 
        for ct in lst:
            b = y == ct
            if ct in list(mkr_dict.keys()):
                ms = list(set(mkr_dict[ct]).intersection(mkrs_all))
                p2t = np.array(p2[ms])
                p1t = np.array(p1[ms])
                pe = np.array((X.loc[b,ms] > 0).mean(axis = 0))
                pex = np.array((X.loc[~b,ms] > 0).mean(axis = 0))
                odr = np.array(-pe).argsort()
                ms_new = []
                pe_new = []
                pex_new = []
                for o in odr:
                    if (pe[o] >= cutoff) & (~np.isnan(pe[o])):
                        ms_new.append(ms[o])

                        pe_new.append(pe[o])
                        pex_new.append(pex[o])
                '''
                pe = pe[~np.isnan(pe)]
                pex = pex[~np.isnan(pex)]
                pe_lst = pe_lst + list(pe)
                pex_lst = pex_lst + list(pex)
                '''

                if len(ms_new) > 0:
                    mkrs_dict[rend[ct]] = ms_new[:min(Npt,len(ms_new))]
                    pe_dict[rend[ct]] = pe_new[:min(Npt,len(ms_new))]
                    pex_dict[rend[ct]] = pex_new[:min(Npt,len(ms_new))]
                else:
                    mkrs_dict[rend[ct]] = ms[:min(Npt,len(ms))]
                    pe_dict[rend[ct]] = list(pe)[:min(Npt,len(ms))]
                    pex_dict[rend[ct]] = list(pex)[:min(Npt,len(ms))]
             
        mkr_lst = []
        pe_lst = []
        pex_lst = []
        cnt2 = 0
        for ct in lst:
            if ct in list(mkr_dict.keys()):
                pe_lst = pe_lst + pe_dict[rend[ct]]
                pex_lst = pex_lst + pex_dict[rend[ct]]
                mkr_lst = mkr_lst + mkrs_dict[rend[ct]]
                cnt2 += len(mkrs_dict[rend[ct]])
            
        if (Npt_tot is not None) & (Npt_tot > 20) & (len(pe_lst) > 0):
            odr = (np.array(pe_lst)).argsort()
            if len(odr) > Npt_tot:
                pe_th = pe_lst[odr[int(len(odr)-Npt_tot)]]
            else:
                pe_th = pe_lst[odr[0]]

            pe_lst = []
            pex_lst = []
            cnt = 0
            for ct in lst:
                if ct in list(mkr_dict.keys()):
                    pe = np.array(pe_dict[rend[ct]]) 
                    b = pe >= pe_th
                    mkrs_dict[rend[ct]] = list(np.array(mkrs_dict[rend[ct]])[b])
                    cnt += len(mkrs_dict[rend[ct]])
                    pe_lst = pe_lst + list(np.array(pe_dict[rend[ct]])[b])
                    pex_lst = pex_lst + list(np.array(pex_dict[rend[ct]])[b])
                
            print('Num markers selected: %i -> %i' % (cnt2, cnt))
            
        mkrs_dict2 = {}
        for m in mkr_dict.keys():
            if (rend is not None) & (m in rend.keys()):
                m = rend[m]                
            if m in list(mkrs_dict.keys()):
                mkrs_dict2[m] = mkrs_dict[m]
            
    return mkrs_dict2, df, pe_lst, pex_lst


def plot_marker_exp_old(adata, markers, var_group_col, group_col = None, 
                    title = None, title_y_pos = 1.1, title_fs = 14, 
                    text_fs = 12, linewidth = 1.5, 
                    var_group_height = 1.2, var_group_rotation = 0, standard_scale = 'var', 
                    nz_frac_max = 0.5, nz_frac_cutoff = 0.1, 
                    rem_mkrs_common_in_N_groups_or_more = 3, N_cells_min = 20,  
                    N_markers_per_group_max = 15, N_markers_total = 200, 
                    figsize = (20, 4), swap_ax = False, legend = False ):

    Npt = N_markers_per_group_max
    Npt_tot = N_markers_total
    show_unassigned = False
    show = True
    rend = None    
    other_genes = []
    
    add_rect = False
    if group_col is None:
        add_rect = True
    
    target_lst = list(adata.obs[var_group_col])
    target_lst.sort()
    
    Lw = linewidth
    hgt = var_group_height
    title_fontsize = title_fs
    
    SCANPY = True
    try:
        import scanpy as sc
    except ImportError:
        SCANPY = False
    
    if (not SCANPY):
        print('ERROR: scanpy not installed. ')   
        return
    
    target = ','.join(target_lst)
    genes = list(adata.var.index.values)
    
    mkrs_all, mkr_dict = get_markers_all4(markers, target_lst, 
                                         genes, rem_cmn = False)
    
    target_lst2 = list(mkr_dict.keys())
    y = adata.obs[var_group_col]
#     b = y == target_lst2[0]
#     for t in target_lst2:
#         b = b | (y == t)
    
    if show_unassigned:
        if np.sum(y.isin(['unassigned'])) > 10:
            mkr_dict['unassigned'] = []
            target_lst2 = list(mkr_dict.keys())
        
    b = y == target_lst2[0]
    nh = 0
    if np.sum(b) > 0: nh = 1
    for t in target_lst2:
        bt = y == t
        if np.sum(bt) > 0:
            b = b | bt
            nh += 1

    adata_t = adata[b, list(set(mkrs_all).union(other_genes))]
    X = adata_t.to_df()
    y = adata_t.obs[var_group_col]

    mkrs_dict, df, pe_lst, pex_lst = update_markers_dict2(mkrs_all, mkr_dict, X, y, rend, 
                                        cutoff = 0, Npt = Npt*2, Npt_tot = 0)
    
    mkrs_dict = remove_mac_common_markers(mkrs_dict)
    if rend is not None: 
        adata_t.obs[var_group_col].replace(rend, inplace = True)
        
    ## Get number of marker genes for each cell type
    mkall = []
    for key in mkrs_dict.keys():
        mkall = mkall + mkrs_dict[key]
    mkall = list(set(mkall))
    nmkr = dict(zip(mkall, [0]*len(mkall)))
    for key in mkrs_dict.keys():
        for m in mkrs_dict[key]:
            nmkr[m] += 1
            
    ## remove the marker genes appering in 3 or more cell types
    if rem_mkrs_common_in_N_groups_or_more > 0:
        to_del = []
        for key in nmkr.keys():
            if nmkr[key] >= rem_mkrs_common_in_N_groups_or_more: 
                to_del.append(key)

        if len(to_del) > 0:
            for m in to_del:
                for key in mkrs_dict.keys():
                    if m in mkrs_dict[key]:
                        mkrs_dict[key].remove(m)
            
    ## Select markers upto Npt
    '''
    for key in mkrs_dict.keys():
        ms = mkrs_dict[key]
        if len(ms) > Npt:
            mkrs_dict[key] = ms[:Npt]
    '''
            
    ## Select only the cell types that exist in the data and the number of cells >= N_cells_min
    ps_cnt = adata_t.obs[var_group_col].value_counts()
    lst_prac = list(ps_cnt.index.values) # list(adata_t.obs[var_group_col].unique())
    mkrs_dict2 = {}
    for m in mkrs_dict.keys():
        if m in lst_prac: 
            if ps_cnt[m] >= N_cells_min:
                mkrs_dict2[m] = mkrs_dict[m]
    mkrs_dict = mkrs_dict2        

    ## Remove cell types for which the number of cells below N_cells_min
    target_lst2 = list(mkrs_dict.keys())
    y = adata_t.obs[var_group_col]
    
    if show_unassigned:
        if np.sum(y.isin(['unassigned'])) > 10:
            mkrs_dict['unassigned'] = []
            target_lst2 = list(mkrs_dict.keys())
    
    if len(target_lst2) == 0:
        return None, None, None
    
    b = y == target_lst2[0]
#     for t in target_lst2:
#         b = b | (y == t)
    nh = 0
    if np.sum(b) > 0: nh = 1
    for t in target_lst2:
        bt = y == t
        if np.sum(bt) > 0:
            b = b | bt
            nh += 1
    
    adata_t = adata_t[b, :]
    if rend is not None: 
        adata_t.obs[var_group_col].replace(rend, inplace = True)
    
    nw = 0
    for key in mkrs_dict.keys():
        nw += len(mkrs_dict[key])
        
    X = adata_t.to_df()
    y = adata_t.obs[var_group_col]
    
    ## Get number of marker genes for each cell type
    mkrs_all = []
    for key in mkrs_dict.keys():
        mkrs_all = mkrs_all + mkrs_dict[key]
    mkrs_all = list(set(mkrs_all))
    
    mkrs_dict, df, pe_lst, pex_lst = update_markers_dict2(mkrs_all, mkrs_dict, X, y, None, 
                                        cutoff = nz_frac_cutoff, Npt = Npt, Npt_tot = Npt_tot)
    
    if (len(other_genes) > 0) & (other_genes is not None):
        for key in mkrs_dict.keys():
            other_genes = list(set(other_genes) - set(mkrs_dict[key]))
        mkrs_dict['Other genes'] = other_genes
        mkrs_all = list(set(other_genes).union(mkrs_all))
    
    '''
    if rend is not None:
        mkrs_dict2 = {}
        for m in rend.values():
            if m in list(mkrs_dict.keys()):
                if len(mkrs_dict[m]) > 0:
                    mkrs_dict2[m] = mkrs_dict[m]
        mkrs_dict = mkrs_dict2        
        # plt.rc('font', size=18)    
    #'''
    
    lst = list(mkrs_dict.keys())
    lst.sort()
    mkrs_dict_new = {}
    for c in lst:
        mkrs_dict_new[c] = mkrs_dict[c]
    mkrs_dict = mkrs_dict_new
        
    if show:
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if group_col is None:
                Type_level = var_group_col
                categ_order = list(mkrs_dict.keys())
                # categ_order.sort()
            else:
                #'''
                categ_order = list(mkrs_dict.keys())
                categ_order.sort()
                mkrs_dict2 = {}
                for c in categ_order:
                    mkrs_dict2[c] = mkrs_dict[c]
                mkrs_dict = mkrs_dict2    
                #'''
                Type_level = group_col
                categ_order = list(adata_t.obs[Type_level].unique())
                categ_order.sort()

            # print(categ_order)
            dp = sc.pl.dotplot(adata_t, mkrs_dict, groupby = Type_level, 
                               categories_order = categ_order, 
                               return_fig = True, log = True, figsize = figsize,
                               var_group_rotation = var_group_rotation, show = False, # cmap = 'firebrick',
                               standard_scale = standard_scale, # mean_only_expressed = True,
                               dot_max = nz_frac_max, swap_axes = swap_ax )
                               # colorbar_title = 'mean expression', size_title = 'percent cells' ) 
            dp.add_totals() # .style(dot_edge_color='black', dot_edge_lw=0.5).show()
        

        ax_dict = dp.get_axes()    
        ax = ax_dict['mainplot_ax']
        
        if title is not None:
            ax.set_title(title, fontsize = title_fontsize, y = title_y_pos) 
        ax.tick_params(labelsize=text_fs)
        
        ## add Rectangles in main plot
        if add_rect:
            cnt = 0
            ylabels = []
            bar_length = 80
            for j, key in enumerate(mkrs_dict.keys()):
                L = len(mkrs_dict[key])
                if swap_ax:
                    ax.add_patch(Rectangle((j, cnt), 1, L, fill = False, edgecolor = 'royalblue', lw = Lw))
                else:
                    ax.add_patch(Rectangle((cnt, j), L, 1, fill = False, edgecolor = 'royalblue', lw = Lw))
                cnt += L
                ylabels = ylabels + mkrs_dict[key]

            if categ_order is not None:
                jj = j
                L_tot = cnt        
                outer_loops = int(len(categ_order)/len(list(mkrs_dict.keys())))
                for kk in range(outer_loops-1):
                    cnt = 0
                    for j1, key in enumerate(mkrs_dict.keys()):
                        j = j + 1
                        L = len(mkrs_dict[key])
                        if swap_ax:
                            ax.add_patch(Rectangle((j, cnt), 1, L, fill = False, 
                                                   edgecolor = 'royalblue', lw = Lw))
                        else:
                            ax.add_patch(Rectangle((cnt, j), L, 1, fill = False, 
                                                   edgecolor = 'royalblue', lw = Lw))
                        cnt += L
        else:
            cnt = 0
            ylabels = []
            for j, key in enumerate(mkrs_dict.keys()):
                ylabels = ylabels + mkrs_dict[key]

        ## color/dot size legend 
        ax2 = ax_dict['color_legend_ax']
        ax3 = ax_dict['size_legend_ax']
        
        if not legend:
            ax2.remove()
            ax3.remove()
        else:
            # pass
            #'''
            if swap_ax == False:
                box = ax2.get_position()
                dx = (box.x1 - box.x0)*0.3
                box.x0 = box.x0 + dx
                box.x1 = box.x1 + dx
                ax2.set_position(box)
                
                box2 = ax3.get_position()
                dx = (box2.x1 - box2.x0)*0.3
                box2.x0 = box2.x0 + dx
                box2.x1 = box2.x1 + dx

                dy = (box2.y1 - box2.y0)
                if box2.y0 > (box.y1 + dy*2):
                    Dy = box2.y0 - (box.y1 + dy*2)
                    box2.y0 = box2.y0 - Dy
                    box2.y1 = box2.y1 - Dy
                else:
                    Dy = (box.y1 + dy*2) - box2.y0
                    box2.y0 = box2.y0 + Dy
                    box2.y1 = box2.y1 + Dy                    
                ax3.set_position(box2)
            #'''

        ## Variable Group plot
        axa = ax_dict['gene_group_ax']
        axa.clear()
        axa.set_frame_on(False)
        axa.grid(False)
        cnt = 0
        gap = 0.2

        n_cells = []
        for ct in mkrs_dict.keys():
            b = adata_t.obs[var_group_col] == ct
            n_cells.append(np.sum(b))
        n_cells = np.array(n_cells)
        p_cells = n_cells/n_cells.sum()

        for j, key in enumerate(mkrs_dict.keys()):
            L = len(mkrs_dict[key])
            
            pct = (1000*p_cells[j])
            if pct < 1:
                pct = '<0.1%'
            elif pct < 10:
                pct = '%3.1f' % (pct/10) + '%'
            else:
                pct = '%i' % int(pct/10) + '%'
                
            if swap_ax:
                axa.plot([hgt, hgt], [cnt+gap, cnt+L-gap], 'k', lw = Lw)
                axa.plot([hgt, 0], [cnt+gap, cnt+gap], 'k', lw = Lw)
                axa.plot([hgt, 0], [cnt+L-gap, cnt+L-gap], 'k', lw = Lw)
                axa.text(hgt*var_group_height, cnt+L/2, ' ' + key, fontsize = text_fs, rotation = 0, ha = 'left', va = 'center')
            else:
                axa.plot([cnt+gap, cnt+L-gap], [hgt, hgt], 'k', lw = Lw)
                axa.plot([cnt+gap, cnt+gap], [hgt, 0], 'k', lw = Lw)
                axa.plot([cnt+L-gap, cnt+L-gap], [hgt, 0], 'k', lw = Lw)
                Ha = 'center'
                if (var_group_rotation > 0) & (var_group_rotation < 90):
                    Ha = 'left'
                elif (var_group_rotation < 0) & (var_group_rotation > -90):
                    Ha = 'right'                    
                axa.text(cnt+L/2, hgt*var_group_height, ' ' + key, fontsize = text_fs, 
                         rotation = var_group_rotation, ha = Ha)
            cnt += L

        ## group extra plot
        axb = ax_dict['group_extra_ax']
        
        if swap_ax == False:
            pass
        else:
            box_main = ax.get_position()
            box = axb.get_position()
            box.y1 = box_main.y0 - (box.y1 - box.y0)
            box.y0 = box_main.y0
            axb.set_position(box)
        
        axb.clear()
        axb.set_frame_on(False)
        axb.get_xaxis().set_visible(False)
        axb.get_yaxis().set_visible(False)
        
        n_cells2 = []
        if categ_order is None:
            categ_order = list(mkrs_dict.keys())
            
        for ct in categ_order:
            b = adata_t.obs[Type_level] == ct
            n_cells2.append(np.sum(b))
        n_cells2 = np.array(n_cells2)
        p_cells2 = np.sqrt(n_cells2)
        # p_cells2 = p_cells2 - np.min(p_cells2*0.9)
        # p_cells2 = n_cells2/n_cells2.sum()
        p_cells2 = p_cells2/np.max(p_cells2)
        
        for j, key in enumerate(categ_order):
            if swap_ax:
                y = p_cells2[j]
                axb.add_patch(Rectangle((j+0.2, 0), 0.6, y, fill = True, 
                                        facecolor = 'firebrick', edgecolor = 'black', lw = Lw))
                axb.text( j+0.5, y, ' %i ' % n_cells2[j], rotation = 90, ha = 'center', va = 'top', fontsize = text_fs)
            else:
                y = p_cells2[j]
                axb.add_patch(Rectangle((0, j+0.2), y, 0.6, fill = True, 
                                        facecolor = 'firebrick', edgecolor = 'black', lw = Lw))
                axb.text( y, j+0.5, ' %i ' % n_cells2[j], rotation = 0, ha = 'left', va = 'center', fontsize = text_fs)
        
        ## Set x/y ticks again
        if swap_ax:
            ax.set_yticks( ticks = list(np.arange(len(ylabels))+0.5))
            ax.set_yticklabels(ylabels)
            ax.xaxis.set_ticks_position('top')
            ax.set_xticks( ticks = list(np.arange(len(categ_order))+0.5))
            ax.set_xticklabels(categ_order)
        else:
            ax.set_xticks( ticks = list(np.arange(len(ylabels))+0.5))
            ax.set_xticklabels(ylabels)
            ax.yaxis.set_ticks_position('left')
            ax.set_yticks( ticks = list(np.arange(len(categ_order))+0.5))
            ax.set_yticklabels(categ_order)
                
    plt.show()
    return 
                                


def get_best_markers(adata, markers, var_group_col, group_col = None, 
                    # title = None, title_y_pos = 1.1, title_fs = 14, 
                    # text_fs = 12, linewidth = 1.5, 
                    # var_group_height = 1.2, var_group_rotation = 0, standard_scale = 'var', 
                    # nz_frac_max = 0.5, 
                    nz_frac_cutoff = 0.1, 
                    rem_mkrs_common_in_N_groups_or_more = 3, N_cells_min = 20,  
                    N_markers_per_group_max = 15, N_markers_total = 200 ): 
                    # figsize = (20, 4), swap_ax = False, legend = False ):

    Npt = N_markers_per_group_max
    Npt_tot = N_markers_total
    show_unassigned = False
    # show = True
    rend = None    
    other_genes = []
    
    # add_rect = False
    # if group_col is None:
    #     add_rect = True
    
    target_lst = list(markers.keys()) # list(adata.obs[var_group_col])
    target_lst.sort()
    
    # Lw = linewidth
    # hgt = var_group_height
    # title_fontsize = title_fs
    
    target = ','.join(target_lst)
    genes = list(adata.var.index.values)
    
    mkrs_all, mkr_dict = get_markers_all4(markers, target_lst, 
                                         genes, rem_cmn = False)
    
    target_lst2 = list(mkr_dict.keys())
    y = adata.obs[var_group_col]
#     b = y == target_lst2[0]
#     for t in target_lst2:
#         b = b | (y == t)
    
    if show_unassigned:
        if np.sum(y.isin(['unassigned'])) > 10:
            mkr_dict['unassigned'] = []
            target_lst2 = list(mkr_dict.keys())
        
    b = y == target_lst2[0]
    nh = 0
    if np.sum(b) > 0: nh = 1
    for t in target_lst2:
        bt = y == t
        if np.sum(bt) > 0:
            b = b | bt
            nh += 1

    adata_t = adata[b, list(set(mkrs_all).union(other_genes))]
    X = adata_t.to_df()
    y = adata_t.obs[var_group_col]

    mkrs_dict, df, pe_lst, pex_lst = update_markers_dict2(mkrs_all, mkr_dict, X, y, rend, 
                                        cutoff = 0, Npt = Npt*2, Npt_tot = 0)
    
    mkrs_dict = remove_mac_common_markers(mkrs_dict)
    if rend is not None: 
        adata_t.obs[var_group_col].replace(rend, inplace = True)
        
    ## Get number of marker genes for each cell type
    mkall = []
    for key in mkrs_dict.keys():
        mkall = mkall + mkrs_dict[key]
    mkall = list(set(mkall))
    nmkr = dict(zip(mkall, [0]*len(mkall)))
    for key in mkrs_dict.keys():
        for m in mkrs_dict[key]:
            nmkr[m] += 1
            
    ## remove the marker genes appering in 3 or more cell types
    if rem_mkrs_common_in_N_groups_or_more > 0:
        to_del = []
        for key in nmkr.keys():
            if nmkr[key] >= rem_mkrs_common_in_N_groups_or_more: 
                to_del.append(key)

        if len(to_del) > 0:
            for m in to_del:
                for key in mkrs_dict.keys():
                    if m in mkrs_dict[key]:
                        mkrs_dict[key].remove(m)
            
    ## Select markers upto Npt
    '''
    for key in mkrs_dict.keys():
        ms = mkrs_dict[key]
        if len(ms) > Npt:
            mkrs_dict[key] = ms[:Npt]
    '''
            
    ## Select only the cell types that exist in the data and the number of cells >= N_cells_min
    #'''
    ps_cnt = adata_t.obs[var_group_col].value_counts()
    lst_prac = list(ps_cnt.index.values) # list(adata_t.obs[var_group_col].unique())
    mkrs_dict2 = {}
    for m in mkrs_dict.keys():
        if m in lst_prac: 
            if ps_cnt[m] >= N_cells_min:
                mkrs_dict2[m] = mkrs_dict[m]
    mkrs_dict = mkrs_dict2        
    #'''
    
    ## Remove cell types for which the number of cells below N_cells_min
    target_lst2 = list(mkrs_dict.keys())
    y = adata_t.obs[var_group_col]
    
    if show_unassigned:
        if np.sum(y.isin(['unassigned'])) > 10:
            mkrs_dict['unassigned'] = []
            target_lst2 = list(mkrs_dict.keys())
    
    if len(target_lst2) == 0:
        return None, None

    #'''
    b = y.isin(target_lst2)
    adata_t = adata_t[b, :]
    if rend is not None: 
        adata_t.obs[var_group_col].replace(rend, inplace = True)
    
    X = adata_t.to_df()
    y = adata_t.obs[var_group_col]
    
    ## Get number of marker genes for each cell type
    mkrs_all = []
    for key in mkrs_dict.keys():
        mkrs_all = mkrs_all + mkrs_dict[key]
    mkrs_all = list(set(mkrs_all))
    
    mkrs_dict, df, pe_lst, pex_lst = update_markers_dict2(mkrs_all, mkrs_dict, X, y, None, 
                                        cutoff = nz_frac_cutoff, Npt = Npt, Npt_tot = Npt_tot)
    #'''
    if group_col is not None:
        lst = list(mkrs_dict.keys())
        for k in lst:
            if len(mkrs_dict[k]) == 0:
                del mkrs_dict[k]
    
    return mkrs_dict, adata_t
                                

def plot_marker_exp_inner(adata, markers, var_group_col, 
                    group_col = None, group_categ_col = None,
                    title = None, title_y_pos = 1.1, title_fs = 14, 
                    text_fs = 12, linewidth = 1.5, 
                    var_group_height = 1.2, var_group_rotation = 0, standard_scale = 'var', 
                    nz_frac_max = 0.5, # nz_frac_cutoff = 0.1, 
                    # rem_mkrs_common_in_N_groups_or_more = 3, N_cells_min = 20,  
                    # N_markers_per_group_max = 15, N_markers_total = 200, 
                    figsize = (20, 4), swap_ax = False, legend = False, add_rect = True ):

    mkrs_dict = markers
    # Npt = N_markers_per_group_max
    # Npt_tot = N_markers_total
    # show_unassigned = False
    show = True
    # rend = None    
    # other_genes = []
    # target_lst = list(markers.keys()) 
    # target_lst.sort()
    
    Lw = linewidth
    hgt = var_group_height
    title_fontsize = title_fs
    
    SCANPY = True
    try:
        import scanpy as sc
    except ImportError:
        SCANPY = False
    
    if (not SCANPY):
        print('ERROR: scanpy not installed. ')   
        return
    
    # target = ','.join(target_lst)
    # genes = list(adata.var.index.values)
        
    mkrs_all = []
    for key in mkrs_dict.keys():
        mkrs_all = mkrs_all + mkrs_dict[key]
    # mkrs_all = list(set(mkrs_all))
    
    adata_t = adata[:, list(set(mkrs_all))]

    sx = len(mkrs_all)
    if group_col is None:
        sy = len(adata_t.obs[var_group_col].unique().tolist())
    else:
        sy = len(adata_t.obs[group_col].unique().tolist())

    if swap_ax:
        figsize = None        
    elif figsize is not None:
        if swap_ax:
            lx = figsize[1]
            ly = figsize[0]
        
            ly = lx*sy/sx
            figsize = (ly, lx)
            title_y_pos = (sx + title_y_pos)/sx
        else:
            lx = figsize[0]
            ly = figsize[1]
        
            ly = lx*sy/sx
            figsize = (lx, ly)
            title_y_pos = (sy + title_y_pos)/sy
    
    if show:
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if group_col is None:
                Type_level = var_group_col
                categ_order = list(mkrs_dict.keys())
                # categ_order.sort()
            else:
                #'''
                var_group_lst = list(mkrs_dict.keys())
                var_group_lst.sort()
                mkrs_dict2 = {}
                for c in var_group_lst:
                    mkrs_dict2[c] = mkrs_dict[c]
                mkrs_dict = mkrs_dict2    
                #'''
                Type_level = group_col
                if group_categ_col is None:
                    categ_order = list(adata_t.obs[Type_level].unique())
                    categ_order.sort()
                else:
                    group_categ_lst = list(adata_t.obs[group_categ_col].unique())
                    group_categ_lst.sort()
                    categ_order = []
                    num_groups_per_group_categ = {}
                    for gc in group_categ_lst:
                        b = adata_t.obs[group_categ_col] == gc
                        lst_tmp = list(adata_t[b,:].obs[Type_level].unique())
                        lst_tmp.sort()

                        # print((gc, np.array(var_group_lst)))
                        if gc in var_group_lst:
                            genes = mkrs_dict[gc]
                        else:
                            genes = mkrs_all
                        Xx = (adata_t[b, genes].to_df() > 0)
                        Xx['group'] = adata_t[b,:].obs[Type_level]                            
                        X = Xx.groupby('group').mean()
                        # display(X)
                        # linked = linkage(X, 'single')
                        # idx_odrd = leaves_list(linked)
                        # lst_tmp = list( np.array(lst_tmp)[idx_odrd][::-1] )
                        X['mean'] = X.mean(axis = 1)
                        lst_tmp = X.sort_values(by = 'mean', ascending = False).index.values.tolist()

                        categ_order = categ_order + lst_tmp
                        num_groups_per_group_categ[gc] = len(lst_tmp)

            # print(categ_order)
            dp = sc.pl.dotplot(adata_t, mkrs_dict, groupby = Type_level, 
                               categories_order = categ_order, 
                               return_fig = True, log = True, figsize = figsize,
                               var_group_rotation = var_group_rotation, show = False, # cmap = 'firebrick',
                               standard_scale = standard_scale, # mean_only_expressed = True,
                               dot_max = nz_frac_max, swap_axes = swap_ax )
                               # colorbar_title = 'mean expression', size_title = 'percent cells' ) 
            dp.add_totals() # .style(dot_edge_color='black', dot_edge_lw=0.5).show()
        

        ax_dict = dp.get_axes()    
        ax = ax_dict['mainplot_ax']
        
        if title is not None:
            ax.set_title(title, fontsize = title_fontsize, y = title_y_pos) 
        ax.tick_params(labelsize=text_fs)
        
        ## add Rectangles in main plot
        ylabels = []
        for j, key in enumerate(mkrs_dict.keys()):
            ylabels = ylabels + mkrs_dict[key]
        
        if add_rect:
            if group_col is None: 
                cnt = 0
                ylabels = []
                bar_length = 80
                for j, key in enumerate(mkrs_dict.keys()):
                    L = len(mkrs_dict[key])
                    if swap_ax:
                        ax.add_patch(Rectangle((j, cnt), 1, L, fill = False, edgecolor = 'royalblue', lw = Lw))
                    else:
                        ax.add_patch(Rectangle((cnt, j), L, 1, fill = False, edgecolor = 'royalblue', lw = Lw))
                    cnt += L
                    ylabels = ylabels + mkrs_dict[key]
    
                if categ_order is not None:
                    jj = j
                    L_tot = cnt        
                    outer_loops = int(len(categ_order)/len(list(mkrs_dict.keys())))
                    for kk in range(outer_loops-1):
                        cnt = 0
                        for j1, key in enumerate(mkrs_dict.keys()):
                            j = j + 1
                            L = len(mkrs_dict[key])
                            if swap_ax:
                                ax.add_patch(Rectangle((j, cnt), 1, L, fill = False, 
                                                       edgecolor = 'royalblue', lw = Lw))
                            else:
                                ax.add_patch(Rectangle((cnt, j), L, 1, fill = False, 
                                                       edgecolor = 'royalblue', lw = Lw))
                            cnt += L
            elif group_categ_col is not None:
                #'''
                cnt_x = 0
                cnt_y = 0
                # ylabels = []
                vg_lst = list(mkrs_dict.keys())
                if set(group_categ_lst).issubset(set(vg_lst)) | set(vg_lst).issubset(set(group_categ_lst)):
                    if set(group_categ_lst).issubset(set(vg_lst)):
                        super_list =vg_lst
                    else:
                        super_list =group_categ_lst
                    for gc in super_list: 
                        if gc in vg_lst:
                            L_x = len(mkrs_dict[gc])
                        else:
                            L_x = 0
                        if gc in group_categ_lst:
                            L_y = num_groups_per_group_categ[gc]
                        else:
                            L_y = 0
                        if swap_ax:
                            ax.add_patch(Rectangle((cnt_y, cnt_x), L_y, L_x, fill = False, edgecolor = 'royalblue', lw = Lw))
                        else:
                            ax.add_patch(Rectangle((cnt_x, cnt_y), L_x, L_y, fill = False, edgecolor = 'royalblue', lw = Lw))
                        cnt_x += L_x
                        cnt_y += L_y
                        # ylabels = ylabels + mkrs_dict[key]
                else:
                    cnt = 0
                    width = len(mkrs_all)
                    for gc in (group_categ_lst[:-1]): 
                        cnt += num_groups_per_group_categ[gc]
                        if swap_ax:
                            ax.plot([cnt, cnt], [0, width], 'royalblue', lw = Lw)
                        else:
                            ax.plot([0, width], [cnt, cnt], 'royalblue', lw = Lw)
    
                    cnt = 0
                    height = len(categ_order)
                    vg_lst = list(mkrs_dict.keys())
                    for vg in vg_lst[:-1]: 
                        cnt += len(mkrs_dict[vg])
                        if swap_ax:
                            ax.plot([0, height], [cnt, cnt], 'royalblue', lw = Lw)
                        else:
                            ax.plot([cnt, cnt], [0, height], 'royalblue', lw = Lw)
                                
        ## color/dot size legend 
        ax2 = ax_dict['color_legend_ax']
        ax3 = ax_dict['size_legend_ax']
        
        if not legend:
            ax2.remove()
            ax3.remove()
        else:
            # pass
            #'''
            if swap_ax == False:
                box = ax2.get_position()
                dx = (box.x1 - box.x0)*0.3
                box.x0 = box.x0 + dx
                box.x1 = box.x1 + dx
                ax2.set_position(box)
                
                box2 = ax3.get_position()
                dx = (box2.x1 - box2.x0)*0.3
                box2.x0 = box2.x0 + dx
                box2.x1 = box2.x1 + dx

                dy = (box2.y1 - box2.y0)
                if box2.y0 > (box.y1 + dy*2):
                    Dy = box2.y0 - (box.y1 + dy*2)
                    box2.y0 = box2.y0 - Dy
                    box2.y1 = box2.y1 - Dy
                else:
                    Dy = (box.y1 + dy*2) - box2.y0
                    box2.y0 = box2.y0 + Dy
                    box2.y1 = box2.y1 + Dy                    
                ax3.set_position(box2)
            #'''

        ## Variable Group plot
        axa = ax_dict['gene_group_ax']
        axa.clear()
        axa.set_frame_on(False)
        axa.grid(False)
        cnt = 0
        gap = 0.2

        n_cells = []
        for ct in mkrs_dict.keys():
            b = adata_t.obs[var_group_col] == ct
            n_cells.append(np.sum(b))
        n_cells = np.array(n_cells)
        p_cells = n_cells/n_cells.sum()

        for j, key in enumerate(mkrs_dict.keys()):
            L = len(mkrs_dict[key])
            '''
            pct = (1000*p_cells[j])
            if pct < 1:
                pct = '<0.1%'
            elif pct < 10:
                pct = '%3.1f' % (pct/10) + '%'
            else:
                pct = '%i' % int(pct/10) + '%'
            #'''    
            if swap_ax:
                axa.plot([hgt, hgt], [cnt+gap, cnt+L-gap], 'k', lw = 1.5) #Lw)
                axa.plot([hgt, 0], [cnt+gap, cnt+gap], 'k', lw = 1.5) #Lw)
                axa.plot([hgt, 0], [cnt+L-gap, cnt+L-gap], 'k', lw = 1.5) #Lw)
                axa.text(hgt*var_group_height, cnt+L/2, ' ' + key, fontsize = text_fs + 1, rotation = 0, ha = 'left', va = 'center')
            else:
                axa.plot([cnt+gap, cnt+L-gap], [hgt, hgt], 'k', lw = 1.5) #Lw)
                axa.plot([cnt+gap, cnt+gap], [hgt, 0], 'k', lw = 1.5) #Lw)
                axa.plot([cnt+L-gap, cnt+L-gap], [hgt, 0], 'k', lw = 1.5) #Lw)
                Ha = 'center'
                if (var_group_rotation > 0) & (var_group_rotation < 90):
                    Ha = 'left'
                elif (var_group_rotation < 0) & (var_group_rotation > -90):
                    Ha = 'right'                    
                axa.text(cnt+L/2, hgt*var_group_height, ' ' + key, fontsize = text_fs + 1, 
                         rotation = var_group_rotation, ha = Ha)
            cnt += L

        ## group extra plot
        axb = ax_dict['group_extra_ax']
        
        if swap_ax == False:
            pass
        else:
            box_main = ax.get_position()
            box = axb.get_position()
            box.y1 = box_main.y0 - (box.y1 - box.y0)
            box.y0 = box_main.y0
            axb.set_position(box)
        
        axb.clear()
        axb.set_frame_on(False)
        axb.get_xaxis().set_visible(False)
        axb.get_yaxis().set_visible(False)
        
        n_cells2 = []
        if categ_order is None:
            categ_order = list(mkrs_dict.keys())
            
        for ct in categ_order:
            b = adata_t.obs[Type_level] == ct
            n_cells2.append(np.sum(b))
        n_cells2 = np.array(n_cells2)
        p_cells2 = np.sqrt(n_cells2)
        # p_cells2 = p_cells2 - np.min(p_cells2*0.9)
        # p_cells2 = n_cells2/n_cells2.sum()
        p_cells2 = p_cells2/np.max(p_cells2)
        
        for j, key in enumerate(categ_order):
            if swap_ax:
                y = p_cells2[j]
                axb.add_patch(Rectangle((j+0.2, 0), 0.6, y, fill = True, 
                                        facecolor = 'firebrick', edgecolor = 'black', lw = 1.5)) #Lw))
                axb.text( j+0.5, y, ' %i ' % n_cells2[j], rotation = 90, ha = 'center', va = 'top', fontsize = text_fs)
            else:
                y = p_cells2[j]
                axb.add_patch(Rectangle((0, j+0.2), y, 0.6, fill = True, 
                                        facecolor = 'firebrick', edgecolor = 'black', lw = 1.5)) #Lw))
                axb.text( y, j+0.5, ' %i ' % n_cells2[j], rotation = 0, ha = 'left', va = 'center', fontsize = text_fs)
        
        ## Set x/y ticks again
        if swap_ax:
            ax.set_yticks( ticks = list(np.arange(len(ylabels))+0.5))
            ax.set_yticklabels(ylabels)
            ax.xaxis.set_ticks_position('top')
            ax.set_xticks( ticks = list(np.arange(len(categ_order))+0.5))
            ax.set_xticklabels(categ_order)
        else:
            ax.set_xticks( ticks = list(np.arange(len(ylabels))+0.5))
            ax.set_xticklabels(ylabels)
            ax.yaxis.set_ticks_position('left')
            ax.set_yticks( ticks = list(np.arange(len(categ_order))+0.5))
            ax.set_yticklabels(categ_order)
                
    plt.show()
    return 
                                                                

def plot_marker_exp(adata, markers, var_group_col, group_col = None, group_categ_col = None,
                    title = None, title_y_pos = 1.1, title_fs = 14, 
                    text_fs = 12, linewidth = 1.5, 
                    var_group_height = 1.2, var_group_rotation = 0, standard_scale = 'var', 
                    nz_frac_max = 0.5, nz_frac_cutoff = 0.1, 
                    rem_mkrs_common_in_N_groups_or_more = 3, N_cells_min = 20,  
                    N_markers_per_group_max = 15, N_markers_total = 200, 
                    figsize = (20, 4), swap_ax = False, legend = False,
                    add_rect = True ):

    if var_group_col is not None: 
        adata.obs[var_group_col] = adata.obs[var_group_col].astype(str)
    if group_categ_col is not None: 
        adata.obs[group_categ_col] = adata.obs[group_categ_col].astype(str)
    if group_col is not None: 
        adata.obs[group_col] = adata.obs[group_col].astype(str)

    selected_mkr_dict, adata_sss = get_best_markers( adata, markers = markers,
                      var_group_col = var_group_col, group_col = group_col, 
                      nz_frac_cutoff = nz_frac_cutoff, N_cells_min = N_cells_min,
                      rem_mkrs_common_in_N_groups_or_more = rem_mkrs_common_in_N_groups_or_more, 
                      N_markers_per_group_max = N_markers_per_group_max, 
                      N_markers_total = N_markers_total )
    
    rv = plot_marker_exp_inner(  adata_sss, markers = selected_mkr_dict, var_group_col = var_group_col, 
                      group_col = group_col, group_categ_col = group_categ_col,
                      title = title, title_y_pos = title_y_pos, title_fs = title_fs,
                      text_fs = text_fs, linewidth = linewidth, standard_scale = standard_scale,
                      var_group_rotation = var_group_rotation, var_group_height = var_group_height,
                      nz_frac_max = nz_frac_max, figsize = figsize, swap_ax = swap_ax, 
                      legend = legend, add_rect = add_rect )
    return rv


def plot_deg( df_deg_dct, reference = 'log2_FC', n_genes_to_show = 30, pval_cutoff = 0.05, 
              figsize = (6,4), dpi = 100, text_fs = 10, title_fs = 12, label_fs = 11, 
              tick_fs = 10, ncols = 2, wspace = 0.15, hspace = 0.2, 
              deg_stat_dct = None, show_log_pv = True ):

    nr, nc = int(np.ceil(len(df_deg_dct.keys())/ncols)), int(ncols) # len(df_deg_dct.keys())
    fig, axes = plt.subplots(figsize = (figsize[0]*nc,figsize[1]*nr), nrows=nr, ncols=nc, # constrained_layout=True, 
                             gridspec_kw={'width_ratios': [1]*nc}, dpi = dpi)
    fig.tight_layout() 
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, 
                        wspace=wspace, hspace=hspace)

    for j, key in enumerate(df_deg_dct.keys()):
        b = df_deg_dct[key]['pval_adj'] <= pval_cutoff
        dfs = df_deg_dct[key].loc[b,:].sort_values(reference, ascending = False)
        dfs = dfs.iloc[:min(n_genes_to_show, np.sum(b))]

        plt.subplot(nr,nc,j+1)
        # plt.figure(figsize = (6,4), dpi = 100)
        X = list(np.arange(dfs.shape[0]))
        Y = list(dfs[reference])
        plt.plot(X, Y)
        m = (np.max(Y)-np.min(Y))
        plt.ylim([np.min(Y)-m*0.4, np.max(Y) + m*0.5])
        tlst = list(dfs.index.values)
        for x, y, t in zip(X, Y, tlst):
            plt.text(x, y, '  %s' % (t), rotation = 90, fontsize = text_fs)
            if show_log_pv:
                lpv =  -np.log10(dfs.loc[t,'pval_adj'])
                plt.text(x, y, '(%3.1f) ' % (lpv), rotation = 90, va = 'top', fontsize = text_fs)
        if deg_stat_dct is None:
            plt.title(key, fontsize = title_fs)
        else:
            s = ' ('
            for kk in deg_stat_dct[key].keys():
                s = s + '%s: %i, ' % (kk, deg_stat_dct[key][kk])
            s = '%s)' % s[:-2]
            plt.title(key + s, fontsize = title_fs)
            
        plt.xlabel('Genes', fontsize = label_fs)
        plt.yticks(fontsize=tick_fs)
        plt.xticks(fontsize=tick_fs)
        if j%nc == 0: plt.ylabel(reference, fontsize = label_fs)
        plt.grid('on')

    if nc*nr > len(df_deg_dct.keys()):
        for j in range(nc*nr-len(df_deg_dct.keys())):
            k = j + len(df_deg_dct.keys()) + 1
            ax = plt.subplot(nr,nc,k)
            ax.axis('off')

    plt.show()
    return


def plot_gsa_result( df_res, terms_sel, items_to_plot, lims = None, dpi = 100,
                  title = None, title_pos = (0.5, 1), title_fs = 16, title_ha = 'center', 
                  label_fs = 10, tick_fs = 8, wspace=0.1, hspace=0.25, Ax = None):

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ## draw result
        sc.settings.set_figure_params(figsize = (4,(len(terms_sel)*1.1 + 8)/9), dpi=100)

        nr, nc = 1, len(items_to_plot)
        fig, axes = plt.subplots(nrows=nr, ncols=nc, constrained_layout=True, dpi = dpi)
        fig.tight_layout() # Or equivalently,  "plt.tight_layout()"
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, 
                            wspace=wspace, hspace=hspace)
        if title is not None: 
            fig.suptitle(title, x = title_pos[0], y = title_pos[1], 
                         fontsize = title_fs, ha = title_ha)

        ylabel = 'Term'

        for k, x in enumerate(items_to_plot): # , '-log(q-val)-enr', 'NES-pr']):
            plt.subplot(1,nc,k+1)
            xlabel = x
            ax = sns.barplot(data = df_res, y = ylabel, x = xlabel, 
                             facecolor = 'firebrick', orient = 'h', 
                             edgecolor = 'black', linewidth = 1, ax = Ax )
            # plt.xticks(fontsize = tick_fs)
            ax.tick_params(axis='both', which='major', labelsize = tick_fs)
            ax.set_xlabel(xlabel, fontsize = label_fs)
            if k == 0:
                ax.set_ylabel(ylabel, fontsize = label_fs)
            else:
                plt.yticks([])
                ax.set_ylabel(None)
            if lims is not None: plt.xlim(lims[k])

        plt.show()
    return True


def plot_gsa_bar( df_res, terms_sel, items_to_plot, lims = None, dpi = 100, 
                  title = None, title_pos = (0.5, 1), title_fs = 16, title_ha = 'center', 
                  label_fs = 10, tick_fs = 8, wspace=0.1, hspace=0.25, Ax = None):

    plot_gsa_result( df_res, terms_sel, items_to_plot, lims, dpi,
                  title, title_pos, title_fs, title_ha, 
                  label_fs, tick_fs, wspace, hspace, Ax)
    return
    

import matplotlib.pyplot as plt
import matplotlib as mpl

def get_gsa_summary( gsa_res, max_log_p = 10 ):
    
    df_gse_all = {}
    for ck in gsa_res.keys():
        df_dct = gsa_res[ck]
        for cs in df_dct.keys():
            case = '%s: %s' % (ck, cs)
            df_gse_all[case] = df_dct[cs]

    case_lst = list(df_gse_all.keys())
    pws = []
    for key in df_gse_all.keys():
        df = df_gse_all[key]
        pws = list(set(pws).union(list(df.index.values)))

    pws.sort(reverse = True)

    dfc_qv = pd.DataFrame(0, index = pws, columns = case_lst)
    dfc_es = pd.DataFrame(0, index = pws, columns = case_lst)
    cnt_es = 0

    for case in df_gse_all.keys():

        df = df_gse_all[case]

        col1 = '%s' % case
        col2 = '%s' % case

        dfc_qv.loc[:,col1] = 0
        dfc_es.loc[:,col2] = 0

        b = ~df['-log(p-val)'].isnull()
        dfc_qv.loc[list(df.index[b]), col1] = list(df.loc[b, '-log(p-val)'])
        if 'NES' in list(df.columns.values):
            b = ~df['NES'].isnull()
            cnt_es += np.sum(b)
            dfc_es.loc[list(df.index[b]), col2] = list(df.loc[b, 'NES'])

    # display(dfc_qv.head())

    pw_sel = pws

    if cnt_es > 0:
        b = dfc_es.max(axis = 1) > 0
        df_qv = dfc_qv.loc[b,:] 
        df_es = dfc_es.loc[b,:] 
    else:
        df_qv = dfc_qv
        df_es = None

    df_qv = df_qv.clip(upper = max_log_p)
    
    return df_qv, df_es


def plot_gsa_dot( gsa_res, 
                  title = 'Test', title_fs = 14, 
                  tick_fs = 10, xtick_rot = 90, xtick_ha = 'center',
                  label_fs = 10, legend_fs = 10, swap_ax = True, 
                  figsize = None, dpi = 100, dot_size = 14, 
                  cbar_frac = 0.05, cbar_aspect = 20, cmap = 'Reds' ):
             
    df_qv, df_es = get_gsa_summary(gsa_res, max_log_p = 10)
    
    df1 = df_qv
    df2 = df_es
    alabel = 'Cases'
    n = df1.shape[0]

    mx_pnt_size = dot_size
    mx_pv = np.ceil(df1.max().max())
    ssf = mx_pnt_size/mx_pv
    
    mn, mx = 0, 1
    if df2 is not None:
        mn, mx = np.floor(df2.min().min()), np.ceil(df2.max().max())
        cbar_title = 'NES'
    else:
        mn, mx = np.floor(df1.min().min()), np.ceil(df1.max().max())
        cbar_title = '-log10(P)'
        
    norm = mpl.colors.Normalize(vmin=mn, vmax=mx)

    # plt.figure(figsize = (2+1, 2*df1.shape[0]/df1.shape[1]))
    if figsize is None:
        sf = 1/6
        if swap_ax:
            xs = (df1.shape[0] + 2) 
            ys = (df1.shape[1] + 2)
        else:
            xs = (df1.shape[1] + 2) 
            ys = (df1.shape[0] + 2)
            
        figsize = (sf*xs, sf*ys)
        fig, ax = plt.subplots(figsize = figsize, dpi = dpi) 
    else:
        fig, ax = plt.subplots(figsize = figsize, dpi = dpi) # figsize = (sf*xs, sf*ys)) 
    # ax = plt.figure(figsize = figsize)

    x = []
    y = []
    ss = []
    cc = []
    for j, c in enumerate(list(df1.columns.values)):
        if swap_ax:
            y = y + [j]*n
            x = x + list(np.arange(n))
        else:
            x = x + [j]*n
            y = y + list(np.arange(n))
        ss = ss + list(df1[c]*ssf)
        if df2 is None:
            cc = cc + list(df1[c])
        else:
            cc = cc + list(df2[c])

    p = ax.scatter( x, y, s = ss, c = cc, cmap = cmap )

    # plt.colorbar(p,cax=ax)
    ax.grid('off')  
    ax.set_title(title, fontsize = title_fs)

    # Set number of ticks for x-axis
    if swap_ax:
        plt.ylabel(alabel, fontsize = label_fs)
        ix, iy, yt_label, xt_label, fr = 0, 1, df1.columns.values, df1.index.values, 1 
    else:
        plt.xlabel(alabel, fontsize = label_fs)
        ix, iy, yt_label, xt_label, fr = 1, 0, df1.index.values, df1.columns.values, 0.1

    plt.ylim([-1, df1.shape[iy]])
    plt.xlim([-1, df1.shape[ix]])

    x = np.arange(df1.shape[ix])
    ax.set_xticks(x)
    # Set ticks labels for x-axis
    x_ticks_labels = list(xt_label)
    ax.set_xticklabels(x_ticks_labels, rotation=xtick_rot, ha = xtick_ha, fontsize=tick_fs)

    # Set number of ticks for x-axis
    y = np.arange(df1.shape[iy])
    ax.set_yticks(y)
    # Set ticks labels for x-axis
    y_ticks_labels = list(yt_label)
    a = ax.set_yticklabels(y_ticks_labels, rotation=0, fontsize=tick_fs)

    # Adding the colorbar
    cmap = p.cmap
    if swap_ax:
        lgap = 0.01
        frac = cbar_frac*0.05*figsize[0]/figsize[1]
        # cbaxes = fig.add_axes([0.91, 0.6, 0.1*df1.shape[1]/df1.shape[0], 0.25]) 
        cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),ax=ax, pad = frac*2, #cbar_gap,
                            fraction = frac, aspect = cbar_aspect, # shrink = cbar_shrink,
                            location = 'top', anchor = (1,1)) #, cax = cbaxes)
            
        # if df2 is not None:
        cbar.set_label(cbar_title, fontsize = legend_fs)
        for t in cbar.ax.get_xticklabels():
            t.set_fontsize(tick_fs)
    else:
        lgap = 0.02 
        frac = cbar_frac*0.05*figsize[1]/figsize[0]
        r = df1.shape[1]/df1.shape[0]
        # cbaxes = fig.add_axes([1, 1, 0.1*(1-df1.shape[1]/30), 0.1]) 
        cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),ax=ax, pad = frac*2, #cbar_gap, 
                            fraction = frac, aspect = cbar_aspect, 
                            location = 'right', anchor = (0,0)) #, cax = cbaxes)
            
        # if df2 is not None:
        cbar.set_label(cbar_title, fontsize = legend_fs)
        for t in cbar.ax.get_yticklabels():
            t.set_fontsize(tick_fs)

    #'''
    if swap_ax:
        nn = 5 # list(np.arange(1,mx_pv,2)) 
        kw = dict(prop="sizes", num=nn, fmt="{x:1.0f}", func=lambda s: s/ssf, color = 'grey')
        legend = ax.legend(*p.legend_elements(**kw), title = '-log(P)',
                       fontsize = legend_fs, title_fontsize=legend_fs,
                       loc = 'upper left', bbox_to_anchor = (1+lgap, 1) )
    else:
        nn = 5 # int(mx_pv)
        kw = dict(prop="sizes", num=nn, fmt="{x:1.0f}", func=lambda s: s/ssf, color = 'grey')
        legend = ax.legend(*p.legend_elements(**kw), title = '-log(P)',
                       fontsize = legend_fs, title_fontsize=legend_fs,
                       loc = 'upper left', bbox_to_anchor = (1+lgap, 1) )
    #'''
    plt.show()
    return


def plot_gsa_all( gsa_res, 
                  title = 'Test', title_fs = 14, 
                  tick_fs = 10, xtick_rot = 90, xtick_ha = 'center',
                  label_fs = 10, legend_fs = 10, swap_ax = True, 
                  figsize = None, dot_size = 14, cbar_frac = 0.2, cbar_aspect = 20,
                  cmap = 'Reds' ):
             
    plot_gsa_dot( gsa_res, 
                  title, title_fs, 
                  tick_fs, xtick_rot, xtick_ha,
                  label_fs, legend_fs, swap_ax, 
                  figsize, dot_size, cbar_frac, cbar_aspect, cmap )    
    return

