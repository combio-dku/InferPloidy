import copy, os, time, warnings, math
from contextlib import redirect_stdout, redirect_stderr
import logging, sys
import numpy as np
import pandas as pd
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
from subprocess import Popen, PIPE
import shlex, anndata

CELLPHONEDB = True
try:
    from cellphonedb.src.core.methods import cpdb_degs_analysis_method
    from cellphonedb.src.core.methods import cpdb_statistical_analysis_method
except ImportError:
    print('WARNING: CellPhoneDB seems not be installed. ')
    CELLPHONEDB = False

SCANVPY = True
try:
    import scanpy as sc
except ImportError:
    print('WARNING: scanpy not installed.')
    SCANVPY = False

    
def run_command(cmd, verbose = True):
    cnt = 0
    with Popen(shlex.split(cmd), stdout=PIPE, bufsize=1, \
               universal_newlines=True ) as p:
        for line in p.stdout:
            if (line[:14] == 'Tool returned:'):                    
                cnt += 1
            elif cnt > 0:
                pass
            else: 
                if verbose:
                    print(line, end='')
                    
        exit_code = p.poll()
    return exit_code


def cpdb_run( df_cell_by_gene, cell_types, out_dir,
              gene_id_type = 'gene_name', db = None, 
              n_iter = None, pval_th = None, threshold = None, verbose = False):
    
    start = time.time()
    if verbose: print('Running CellPhoneDB .. ')    
    X = df_cell_by_gene.astype(int) 
    ## X = (X.div(X.sum(axis = 1), axis=0)*1e6).astype(int)
    
    if out_dir[-1] == '/':
        out_dir = out_dir[:-1]
    
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
        
    file_meta = '%s/meta_tmp.tsv' % out_dir
    file_cpm = '%s/exp_mat_tmp.tsv' % out_dir
    
    X.transpose().to_csv(file_cpm, sep = '\t')
    df_celltype = pd.DataFrame({'cell_type': cell_types}, 
                               index = X.index.values)    
    df_celltype.to_csv(file_meta, sep = '\t')
    
    cmd = 'cellphonedb method statistical_analysis '
    cmd = cmd + '%s %s ' % (file_meta, file_cpm)
    cmd = cmd + '--counts-data=%s ' % gene_id_type
    if pval_th is not None: cmd = cmd + '--pvalue=%f ' % pval_th
    if threshold is not None: cmd = cmd + '--threshold=%f ' % threshold
    if n_iter is not None: cmd = cmd + '--iterations=%i ' % n_iter
    if db is not None: '--database %s ' % db
    cmd = cmd + '--output-path %s ' % out_dir
    if not verbose: cmd = cmd + '--quiet '

    print(cmd)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_command(cmd) 
    
    elapsed = time.time() - start
    if verbose: print('Running CellPhoneDB .. done. %i' % elapsed )    
    
    '''
    if os.path.exists(file_cpm):
        os.remove(file_cpm)
    if os.path.exists(file_meta):
        os.remove(file_meta)
    #'''
    
    return cmd
    
    
def split_cellphonedb_out(df):
    cols = df.columns.values
    items = cols[:10]
    pairs = cols[10:]
    df_items = df[items]
    df_pairs = df[pairs]    
    return df_items, df_pairs    
    
def cpdb_get_res( out_dir ):
    
    ## Load p_values
    df_pval = pd.read_csv('%s/pvalues.txt' % out_dir, sep = '\t', 
                          index_col = 0)    
    df_pval_items, df_pval_pairs = split_cellphonedb_out(df_pval)
    
    ## Load means
    df_mean = pd.read_csv('%s/means.txt' % out_dir, sep = '\t', 
                          index_col = 0)    
    df_mean_items, df_mean_pairs = split_cellphonedb_out(df_mean)
    
    ## Check integrity
    idxp = list(df_pval_items.index.values)
    idxm = list(df_mean_items.index.values)
    idxc = set(idxp).intersection(idxm)
    cnt = 0
    for p, m in zip(idxp, idxm):
        if p != m:
            cnt += 1
    if cnt > 0:
        print( len(idxc), len(df_pval_items.index.values), 
               len(df_mean_items.index.values), cnt )
    
    return df_mean_items, df_mean_pairs, df_pval_items, df_pval_pairs   


def to_vector(df, rows, cname):    
    cols = df.columns.values
    idxs, gps, cps, vals = [], [], [], []
    ga, gb, ca, cb = [], [], [], []
    
    for c in list(cols):
        vt = list(df[c])
        ct = [c]*len(vt)
        gt = list(rows)
        it = []
        for r in gt:
            idx = '%s--%s' % (r,c)
            it.append(idx)            
        idxs = idxs + it
        gps = gps + gt
        cps = cps + ct
        vals = vals + vt
        '''
        ga = ga + [g.split('_')[0] for g in gt]
        gb = gb + [g.split('_')[1] for g in gt]
        ca = ca + [g.split('|')[0] for g in ct]
        cb = cb + [g.split('|')[1] for g in ct]
        #'''
        ga_tmp = []
        gb_tmp = []
        ca_tmp = []
        cb_tmp = []
        for g in gt:
            items = g.split('_')
            ga_tmp.append(items[0])
            s_gb = items[1]
            if len(items) > 2:
                for it in items[2:]:
                    s_gb = s_gb + '_%s' % it
            gb_tmp.append(s_gb)
            
        for g in ct:
            items = g.split('|')
            ca_tmp.append(items[0])
            s_gb = items[1]
            '''
            if len(items) > 2:
                for it in items[2:]:
                    s_gb = s_gb + '_%s' % it
            '''
            cb_tmp.append(s_gb)
            
        ga = ga + ga_tmp
        gb = gb + gb_tmp
        ca = ca + ca_tmp
        cb = cb + cb_tmp
        
    dfo = pd.DataFrame({'gene_pair': gps, 'cell_pair': cps, 
                        'gene_A': ga, 'gene_B': gb,
                        'cell_A': ca, 'cell_B': cb,
                         cname: vals}, index = idxs)    
    return dfo

def cpdb_get_vec( df_info, df_mean_pairs, df_pval_pairs, 
                  pval_max = 0.05, mean_min = 0.01 ):    
    dfp = to_vector(df_pval_pairs, 
                    df_info['interacting_pair'], 'pval')
    dfm = to_vector(df_mean_pairs, 
                    df_info['interacting_pair'], 'mean')
    b = (dfp['pval'] <= pval_max) & (dfm['mean'] >= mean_min) 
    b = b & (~dfp['pval'].isnull()) & (~dfm['mean'].isnull())
    dfp = dfp.loc[b,:].copy(deep=True)
    dfm = dfm.loc[b,:].copy(deep=True)   
    dfp['mean'] = dfm['mean']    
    return dfp    


def cpdb_get_results( out_dir, pval_max = 0.05, mean_min = 0.01 ):
    
    df_mean_info, df_mean_pairs, df_pval_info, df_pval_pairs = \
          cpdb_get_res( out_dir )
    dfv = cpdb_get_vec( df_pval_info, df_mean_pairs, df_pval_pairs, 
                        pval_max = pval_max, mean_min = mean_min )
    '''
    idxs = list(dfv.index.values)
    rend = {}
    for idx in idxs:
        if dfv.loc[idx,'cell_A'] > dfv.loc[idx,'cell_B']:
            ca = dfv.loc[idx,'cell_A']
            cb = dfv.loc[idx,'cell_B']
            ga = dfv.loc[idx,'gene_A']
            gb = dfv.loc[idx,'gene_B']   
            dfv.loc[idx,'cell_A'] = cb
            dfv.loc[idx,'cell_B'] = ca
            dfv.loc[idx,'gene_A'] = gb
            dfv.loc[idx,'gene_B'] = ga
            idx_new = '%s_%s--%s|%s' % (gb, ga, cb, ca)
            dfv.loc[idx,'cell_pair'] = '%s|%s' % (cb, ca)
            dfv.loc[idx,'gene_pair'] = '%s|%s' % (gb, ga)
            rend[idx] = idx_new

    if len(rend.keys()) > 0:
        dfv.rename(index = rend, inplace = True)
    '''        
    return df_pval_info, df_pval_pairs, df_mean_pairs, dfv


def cpdb_add_interaction( file_i, file_c = None, 
                          file_p = None, file_g = None, 
                          out_dir = 'cpdb_out'):
        
    if file_i is None:
        print('ERROR: provide file containing interaction info.')
        return None
    if not os.path.exists(file_i):
        print('ERROR: %s not found' % file_i)
    
    if file_c is not None:
        if not os.path.exists(file_c):
            print('ERROR: %s not found' % file_c)
            return None
    
    if file_p is not None:
        if not os.path.exists(file_p):
            print('ERROR: %s not found' % file_p)
            return None

    if file_g is not None:
        if not os.path.exists(file_g):
            print('ERROR: %s not found' % file_g)
            return None
    
    cmd = 'cellphonedb database generate '
    cmd = cmd + '--user-interactions %s ' % file_i
    if file_c is not None: cmd = cmd + '--user-complex %s ' % file_c
    if file_p is not None: cmd = cmd + '--user-protein %s ' % file_p
    if file_g is not None: cmd = cmd + '--user-gene %s ' % file_g
    cmd = cmd + '--result-path %s ' % out_dir
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_command(cmd) 
        pass
    
    print('Updated DB files saved to %s' % out_dir )    
    return out_dir


###################################
### CellPhoneDB 4.1.0 #############

def cpdb4_run( adata, cell_types, db, out_dir,
               gene_id_type = 'gene_name', n_cores = 4,
               n_iter = None, pval_th = None, threshold = None, verbose = False):
    
    if isinstance(adata, pd.DataFrame):
        X = adata.astype(int).copy(deep = True)
        cols = list(X.columns.values)
        rows = list(X.index.values)
        file_cpm = '%s/cnt_tmp.tsv' % out_dir
    elif isinstance(adata, anndata.AnnData):         
        cols = list(adata.var.index.values)
        rows = list(adata.obs.index.values)
        file_cpm = '%s/adata_tmp.h5ad' % out_dir
    else:
        print('ERROR: input data must be either DataFrame or AnnData.')
        return None
    
    start = time.time()
    if verbose: print('Running CellPhoneDB .. ')    
    
    if out_dir[-1] == '/':
        out_dir = out_dir[:-1]
    
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
            
    cols2 = [s.upper() for s in cols]
    rend = dict(zip(cols, cols2))
    
    if isinstance(adata, pd.DataFrame):
        X.rename(columns = rend, inplace = True)
        X.transpose().to_csv(file_cpm, sep = '\t')
        
    elif isinstance(adata, anndata.AnnData):         
        adata.var.rename(index = rend, inplace = True)    
        adata.write(file_cpm)
        
    file_meta = '%s/meta_tmp.tsv' % out_dir
    df_celltype = pd.DataFrame({'cell_type': cell_types}, 
                               index = rows)    
    df_celltype.to_csv(file_meta, sep = '\t')
    
    # 출력을 숨기기 위해 /dev/null로 리디렉션
    log_level = logging.getLogger().getEffectiveLevel()
    logging.basicConfig(level=logging.ERROR)

    # 출력을 숨기기 위해 /dev/null로 리디렉션
    null_file = open(os.devnull, 'w')
    save_stdout = sys.stdout
    save_stderr = sys.stderr
    sys.stdout = null_file
    sys.stderr = null_file

    with open(os.devnull, 'w') as fnull:
        with redirect_stdout(fnull), redirect_stderr(fnull):
            deconv, means, pvals, signif_means = cpdb_statistical_analysis_method.call(
                    cpdb_file_path = db,
                    meta_file_path = file_meta,
                    counts_file_path = file_cpm,
                    counts_data = gene_id_type,
                    output_path = out_dir, threads = n_cores )
    
    #'''
    # 출력을 복원
    sys.stdout = save_stdout
    sys.stderr = save_stderr
    null_file.close()

    logging.getLogger().setLevel(log_level)
    # logging.getLogger().setLevel(logging.NOTSET)    
    #''' 

    ## if gene_a or gene_b is None, set it with partner's name
    b = means['gene_a'].isnull()
    if np.sum(b) > 0:
        means.loc[b, 'gene_a'] = means.loc[b, 'partner_a']

    b = means['gene_b'].isnull()
    if np.sum(b) > 0:
        means.loc[b, 'gene_b'] = means.loc[b, 'partner_b']

    b = pvals['gene_a'].isnull()
    if np.sum(b) > 0:
        pvals.loc[b, 'gene_a'] = pvals.loc[b, 'partner_a']

    b = pvals['gene_b'].isnull()
    if np.sum(b) > 0:
        pvals.loc[b, 'gene_b'] = pvals.loc[b, 'partner_b']

    elapsed = time.time() - start
    if verbose: print('Running CellPhoneDB .. done. %i' % elapsed )    
    
    if os.path.exists(file_cpm):
        os.remove(file_cpm)
    if os.path.exists(file_meta):
        os.remove(file_meta)
    
    return  means, pvals


def cpdb4_get_results( df_pval, df_mean, pval_max = 0.05, mean_min = 0.01 ):
    
    # df_mean_info, df_mean_pairs, df_pval_info, df_pval_pairs = \
    #       cpdb_get_res( out_dir )
    
    cols = list(df_pval.columns.values)
    df_pval_info = df_pval[cols[:11]]
    df_pval_pairs = df_pval[cols[11:]]
    
    cols = list(df_mean.columns.values)
    df_mean_info = df_mean[cols[:11]]
    df_mean_pairs = df_mean[cols[11:]]
    
    dfv = cpdb_get_vec( df_pval_info, df_mean_pairs, df_pval_pairs, 
                        pval_max = pval_max, mean_min = mean_min )
    
    return df_pval_info, df_pval_pairs, df_mean_pairs, dfv


###################################
### Plot functions for CellPhoneDB

def cpdb_plot( dfp, mkr_sz = 6, tick_sz = 6, 
                             legend_fs = 11, title_fs = 14,
                             dpi = 120, title = None, swap_ax = False ):
    if swap_ax == False:
        a = 'gene_pair'
        b = 'cell_pair'
    else:
        b = 'gene_pair'
        a = 'cell_pair'
    
    y = len(set(dfp[a]))
    x = len(set(dfp[b]))
    
    print('%i %ss, %i %ss found' % (y, a, x, b))
    
    pv = -np.log10(dfp['pval']+1e-10).round()
    np.min(pv), np.max(pv)
    
    mn = np.log2((1+dfp['mean']))
    np.min(mn), np.max(mn)    
    
    w = x/6
    sc.settings.set_figure_params(figsize=(w, w*(y/x)), 
                                  dpi=dpi, facecolor='white')
    fig, ax = plt.subplots()

    mul = mkr_sz
    scatter = ax.scatter(dfp[b], dfp[a], s = pv*mul, c = mn, 
                         linewidth = 0, cmap = 'Reds')

    legend1 = ax.legend(*scatter.legend_elements(),
                        loc='upper left', 
                        bbox_to_anchor=(1+1/x, 0.5), 
                        title=' log2(m) ', 
                        fontsize = legend_fs)
    legend1.get_title().set_fontsize(legend_fs)
    ax.add_artist(legend1)

    # produce a legend with a cross section of sizes from the scatter
    handles, labels = scatter.legend_elements(prop='sizes', alpha=0.6)
    # print(labels)
    labels = [1, 2, 3, 4, 5]
    legend2 = ax.legend(handles, labels, loc='lower left', 
                        bbox_to_anchor=(1+1/x, 0.5), 
                        title='-log10(p)', 
                        fontsize = legend_fs)
    legend2.get_title().set_fontsize(legend_fs)

    if title is not None: plt.title(title, fontsize = title_fs)
    plt.yticks(fontsize = tick_sz)
    plt.xticks(rotation = 90, ha='center', fontsize = tick_sz)
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


def plot_circ( df_in, figsize = (10, 10), title = None, title_fs = 16, 
               text_fs = 14, num_fs = 12, margin = 0.12, alpha = 0.5, 
               N = 500, R = 4, Rs = 0.1, lw_max = 10, lw_scale = 0.1, 
               log_lw = False, node_size = 10, rot = True, 
               cmap = 'Spectral', ax = None, show = True):
              
    df = df_in.copy(deep = True)
    mxv = df_in.max().max()
    
    if ax is None: 
        plt.figure(figsize = figsize)
        ax = plt.gca()
        
    # color_lst = ['orange', 'navy', 'green', 'gold', # 'lime', 'magenta', \
    #         'turquoise', 'red', 'royalblue', 'firebrick', 'gray']
    # color_map = cm.get_cmap(cmap)
    color_map = matplotlib.colormaps[cmap]
    color_lst = [color_map(i/(df.shape[0]-1)) for i in range(df.shape[0])]

    clst = list(df.index.values) 

    M = df.shape[0]
    pp = np.arange(M)*(2*math.pi/M)
    px = np.sin(pp)
    py = np.cos(pp)
    pnts = np.array([px, py])
    
    for j in range(pnts.shape[1]):
        p1 = pnts[:,j]
        for k in range(pnts.shape[1]):
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
            
        a = (df.loc[clst[j], clst[j]] != 0)*(Rs*2)
        if ax is None: 
            plt.plot( x, y, 'o', ms = node_size, c = color_lst[j])
            plt.text( x, y, ' %s ' % clst[j], fontsize = text_fs, 
                      ha = ha, va = va, rotation = rotation[j])
        else:
            ax.plot( x, y, 'o', ms = node_size, c = color_lst[j])
            ax.text( x*(1+a), y*(1+a), '  %s  ' % clst[j], fontsize = text_fs, 
                     ha = ha, va = va, rotation = rotation[j])

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


def plot_cci_circ( df_in, figsize = (10, 10), title = None, title_fs = 16, 
               text_fs = 14, num_fs = 12, margin = 0.08, alpha = 0.5, 
               N = 500, R = 4, Rs = 0.1, lw_max = 10, lw_scale = 0.1, 
               log_lw = False, node_size = 10, rot = True, 
               cmap = 'Spectral', ax = None, show = True):
              
    df = df_in.copy(deep = True)
    mxv = df_in.max().max()
    
    if ax is None: 
        plt.figure(figsize = figsize)
        ax = plt.gca()
        
    # color_lst = ['orange', 'navy', 'green', 'gold', # 'lime', 'magenta', \
    #         'turquoise', 'red', 'royalblue', 'firebrick', 'gray']
    # color_map = cm.get_cmap(cmap)
    color_map = colormaps[cmap]
    color_lst = [color_map(i/(df.shape[0]-1)) for i in range(df.shape[0])]

    clst = list(df.index.values) 

    M = df.shape[0]
    pp = np.arange(M)*(2*math.pi/M)
    px = np.sin(pp)
    py = np.cos(pp)
    pnts = np.array([px, py])
    
    for j in range(pnts.shape[1]):
        p1 = pnts[:,j]
        for k in range(pnts.shape[1]):
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
            
        a = (df.loc[clst[j], clst[j]] != 0)*(Rs*2)
        if ax is None: 
            plt.plot( x, y, 'o', ms = node_size, c = color_lst[j])
            plt.text( x, y, ' %s ' % clst[j], fontsize = text_fs, 
                      ha = ha, va = va, rotation = rotation[j])
        else:
            ax.plot( x, y, 'o', ms = node_size, c = color_lst[j])
            ax.text( x*(1+a), y*(1+a), '  %s  ' % clst[j], fontsize = text_fs, 
                     ha = ha, va = va, rotation = rotation[j])

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


def plot_cci_circ_m( df_lst, ncol = 3, figsize = (8,7), title = None, title_y = 1, title_fs = 18, 
                   text_fs = 16, num_fs = 12, margin = 0.08, alpha = 0.5, 
                   N = 500, R = 3, Rs = 0.1, lw_max = 20, lw_scale = 0.2, log_lw = False, 
                   node_size = 10, rot = False, cmap = 'Spectral', ws_hs = (0.2, 0.1) ):
    
    cond_lst = list(df_lst.keys())
    nc = ncol
    nr = int(np.ceil(len(cond_lst)/nc))
    # nr, nc = 1, int(len(cond_lst))
    fig, axes = plt.subplots(figsize = (figsize[0]*nc,figsize[1]*nr), nrows=nr, ncols=nc, # constrained_layout=True, 
                             gridspec_kw={'width_ratios': [1]*nc})
    fig.tight_layout() 
    plt.subplots_adjust(left=0.025, bottom=0.025, right=0.975, top=0.975, wspace=ws_hs[0], hspace=ws_hs[1])
    if title is not None: plt.suptitle(title, fontsize = title_fs, y = title_y)

    for j, k in enumerate(cond_lst):

        df = df_lst[k]*(df_lst[k] > 0).copy(deep = True)

        plt.subplot(nr,nc,j+1)

        if nr == 1: ax = axes[int(j)]
        else: ax = axes[int(j/nc)][j%nc]

        plot_cci_circ( df, title = k, title_fs = text_fs + 2, 
                   text_fs = text_fs, num_fs = num_fs, margin = margin, alpha = alpha, 
                   N = N, R = R, Rs = Rs, lw_max = lw_max, lw_scale = lw_scale, log_lw = log_lw, 
                   node_size = node_size, rot = rot, cmap = cmap, 
                   ax = ax, show = False)

        if len(cond_lst) < (nr*nc):
            for k in range(int(nr*nc - len(cond_lst))):
                j = k + len(cond_lst)
                ax = plt.subplot(nr,nc,j+1)
                ax.axis('off')

    plt.show()
    return


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
def cci_get_ni_mat( dfs_dct, remove_common = True ):
    
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
        
    if remove_common: 
        df_lst = cci_remove_common( df_lst )
        
    return df_lst

