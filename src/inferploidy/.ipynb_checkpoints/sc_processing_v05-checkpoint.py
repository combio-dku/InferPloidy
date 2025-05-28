#!/usr/bin/python3

import random, math
import argparse, time, os, pickle, datetime
import os, warnings, copy
from subprocess import Popen, PIPE
import shlex
import pkg_resources
# from shutil import copyfile
import shutil

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix, csc_matrix
import anndata
# import infercnvpy as cnv

from scoda.icnv import find_condition_specific_markers, generate_hicat_markers_db
from scoda.icnv import run_icnv, identify_tumor_cells, icnv_set_tumor_info
from scoda.cpdb import cpdb4_run, cpdb4_get_results, cpdb_plot, cpdb_get_gp_n_cp #, plot_circ
from scoda.gsea import select_samples, run_enrich, run_enrichr, run_prerank
from scoda.deg import deg_multi, get_population, plot_population
from scoda.misc import plot_sankey_e, get_opt_files_path
from scoda.hicat import HiCAT
from scoda.pipeline import detect_tissue, detect_tissue_get_stats, load_gtf
from scoda.viz import get_sample_to_group_map

from scoda.key_gen import get_dataset_key, recover_int_lst

# from pipeline_gsea import scoda_deg_gsea
from scoda.pipeline import scoda_hicat, scoda_icnv_addon, scoda_cci, scoda_deg_gsea, scoda_icnv_addon_split_run
# from scoda.pipeline import scoda_all_in_one

CLUSTERING_ALGO = 'lv'
SKNETWORK = True
try:
    from sknetwork.clustering import Louvain
except ImportError:
    print('WARNING: sknetwork not installed. GMM will be used for clustering.')
    CLUSTERING_ALGO = 'km'
    SKNETWORK = False

INFERCNVPY = True
try:
    import infercnvpy as cnv
except ImportError:
    print('ERROR: infercnvpy not installed. Tumor cell identification will not be performed.')
    INFERCNVPY = False

GSEAPY = True
try:
    import gseapy as gp
except ImportError:
    print('WARNING: gseapy not installed or not available. ')
    GSEAPY = False

'''
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
'''

def run_command(cmd, verbose = False):
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


def count_dir_in( ddir ):
    lst = os.listdir(ddir)
    dcnt = 0
    path = None
    for i in lst:
        if i[0] != '.':
            if os.path.isdir('%s/%s' % (ddir, i)):
                dcnt += 1
                path = '%s/%s' % (ddir, i)
            
    return dcnt, path
    

def decompress( data_dir, dname = 'data' ):

    lst = os.listdir(data_dir)
    rcode = -1

    error = -1
    d_file = None
    file_name = None
    ddir = ''
    for f in lst:
        d_file = f
        items = f.split('.')
        
        if items[0] == 'cpdb':
            pass
            # sdir = 'cpdb'
        else:
            sdir = dname
            
            if len(items) > 1:
                if items[-1] == 'zip':

                    ddir = '%s/%s' % (data_dir, sdir)
                    if not os.path.isdir(ddir):
                        os.mkdir(ddir)
                    cmd = 'unzip %s/%s -d %s ' % (data_dir, f, ddir)
                    rcode = run_command(cmd)
                    if (rcode is None) | (rcode == 0):
                        dcnt, new_path = count_dir_in( ddir )
                        if dcnt == 1:
                            ddir = new_path

                    error = 0
                    # print('ziped -> %s' % cmd)
                    file_name = ''
                    for item in items[:-1]:
                        file_name = file_name + '%s.' % item
                    file_name = file_name[:-1]


                elif items[-1] == 'gz':
                    if len(items) >= 2: 
                        if (items[-2] == 'tar') & (len(items) >= 3):

                            ddir = '%s/%s' % (data_dir, sdir)
                            if not os.path.isdir(ddir):
                                os.mkdir(ddir)
                            cmd = 'tar -xzvf %s/%s -C %s ' % (data_dir, f, ddir)
                            rcode = run_command(cmd)
                            if (rcode is None) | (rcode == 0):
                                dcnt, new_path = count_dir_in( ddir )
                                if dcnt == 1:
                                    ddir = new_path

                                '''
                                if dcnt >= 1:
                                    pass
                                elif dcnt == 0
                                    ddir_tmp = '%s/%s' % (data_dir, 'dtmp')
                                    os.mkdir(ddir_tmp)
                                    cmd = 'mv %s/* %s' % (ddir, ddir_tmp)

                                else:
                                    dcnt2, new_path2 = count_dir_in( new_path )
                                    if dcnt2 == 0:
                                        ddir = new_path
                                '''
                            error = 0
                            # print('tar.gziped -> %s' % cmd)
                            file_name = ''
                            for item in items[:-2]:
                                file_name = file_name + '%s.' % item
                            file_name = file_name[:-1]

                        else:
                            ddir = '%s/%s' % (data_dir, sdir)
                            if not os.path.isdir(ddir):
                                os.mkdir(ddir)
                            # cmd = 'cp %s/%s %s' % (data_dir, f, ddir) 
                            # rcode = run_command(cmd)
                            shutil.copy( '%s/%s' % (data_dir, f), ddir )


                            cmd = 'gzip -d %s/%s ' % (ddir, f)
                            rcode = run_command(cmd)
                            
                            if (rcode is None) | (rcode == 0):
                                dcnt, new_path = count_dir_in( ddir )
                                if dcnt == 1:
                                    ddir = new_path
                            error = 0
                            # print('gziped -> %s' % cmd)
                            file_name = ''
                            for item in items[:-2]:
                                file_name = file_name + '%s.' % item
                            file_name = file_name[:-1]
                    else:
                        error = 1

                elif items[-1] == 'h5ad':
                    ddir = '%s' % (data_dir)
                    rcode = 0
                    error = 0
                    file_name = ''
                    for item in items[:-1]:
                        file_name = file_name + '%s.' % item
                    file_name = file_name[:-1]
                    # print('h5ad')
                else:
                    error = 2
            else:
                error = 3

            if not error:
                break
            
    return error, rcode, ddir, file_name


META_FILE_NAME = 'meta_data.csv'
META_FILE_NAME2 = 'metadata.csv'
META_FILE_NAME3 = 'meta data.csv'

def check_if_meta_data_exists(lst):
    
    for f in lst:
        pos = f.lower().find(META_FILE_NAME)
        if pos >= 0:
            return f
        else:
            pos = f.lower().find(META_FILE_NAME2)
            if pos >= 0:
                return f
            else:
                pos = f.lower().find(META_FILE_NAME3)
                if pos >= 0:
                    return f
    return None


def get_meta_data(ddir):
    
    lst = os.listdir(ddir)
    meta_file = check_if_meta_data_exists(lst)
    
    if meta_file is not None:
        meta_file_path = '%s/%s' % (ddir, meta_file)
        df = pd.read_csv(meta_file_path, index_col = 0)
        lst.remove(meta_file)
    else:
        df = None
        
    return lst, df

def get_data_from_csv(data_dir, ddir, file_name):

    H5AD_FILE = '%s.h5ad' % file_name
    
    '''
    lst = os.listdir(ddir)
    meta_file = '%s/%s' % (ddir, META_FILE_NAME)

    if META_FILE_NAME in lst:
        df = pd.read_csv(meta_file, index_col = 0)
        lst.remove(META_FILE_NAME)
    else:
        df = None
    '''
    lst, df = get_meta_data(ddir)

    if len(lst) == 0:
        print('WARNING: No data file found')
        return None
    elif len(lst) > 1:
        print('WARNING: More than one data files found: ', lst)
        print('WARNING: The data type you provided might be wrong one. ')
        return None
        
    file_in = ddir + '/%s' % lst[0]
    dfX = pd.read_csv(file_in, index_col = 0)
    genes = list(dfX.columns.values)
    dfv = pd.DataFrame({'gene': genes}, index = genes)
    cells = list(dfX.index.values)
    dfo = pd.DataFrame(index = cells)
    dfo['sample'] = ''
    
    if df is not None:
        if dfX.shape[0] != df.shape[0]:
            print('WARNING: # of cells not match. %i != %i.' % (dfX.shape[0], df.shape[0]))
            return None
        elif dfX.shape[1] == 0:
            print('WARNING: No data found.')
            return None
        else:
            cols = list(df.columns.values)
            dfo[cols] = df[cols]
        
    adata = anndata.AnnData(X = csr_matrix(dfX), obs = dfo, var = dfv)
    
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    #'''
    if 'condition' in list(adata.obs.columns.values):
        adata.obs['condition'] = adata.obs['condition'].astype(str) 
    else:
        print('''WARNING: 'condition' not specified in meta_data.csv''')
        adata.obs['condition'] = 'Not_specified'
        
    if 'sample' in list(adata.obs.columns.values):
        adata.obs['sample'] = adata.obs['sample'].astype(str)  
    else:
        print('''WARNING: 'sample' not specified in meta_data.csv''')
        adata.obs['sample'] = 'Not_specified'
    #'''            
    file_path = data_dir + '/%s' % H5AD_FILE
    adata.write(file_path)

    if not os.path.isfile(file_path):
        print('WARNING: File writing failed.')
        return None
            
    # print('   Data check passed.')
    if os.path.isdir(ddir):
        # cmd = 'rm -r %s' % ddir
        # run_command(cmd)
        shutil.rmtree(ddir)
        
    ddir = '%s/%s' % (data_dir, 'data')
    if os.path.isdir(ddir):
        # cmd = 'rm -r %s' % ddir
        # run_command(cmd) 
        shutil.rmtree(ddir)
    
    return file_path


def get_data_from_h5ad(data_dir, ddir, file_name):

    lst = os.listdir(ddir)

    if len(lst) == 0:
        print('WARNING: No data file found')
        return None
    elif len(lst) > 1:
        print('WARNING: More than one data files found: ', lst)
        print('WARNING: The data type you provided might be wrong one. ')
        return None
        
    file_in = ddir + '/%s' % lst[0]
    try:
        adata = sc.read_h5ad(file_in)
    except:
        print('WARNING: File read failed.')
        return None
    
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    #'''
    if 'condition' in list(adata.obs.columns.values):
        adata.obs['condition'] = adata.obs['condition'].astype(str) 
    else:
        print('''WARNING: 'condition' not specified in meta_data.csv''')
        adata.obs['condition'] = 'Not_specified'
        
    if 'sample' in list(adata.obs.columns.values):
        adata.obs['sample'] = adata.obs['sample'].astype(str)  
    else:
        print('''WARNING: 'sample' not specified in meta_data.csv''')
        adata.obs['sample'] = 'Not_specified'
    #'''
    if not isinstance(adata.X, csr_matrix):
        adata.X = csr_matrix(adata.X)
    
    adata.write(file_in)
    
    if data_dir != ddir:
        # cmd = 'mv %s %s' % (file_in, data_dir)
        # run_command(cmd)
        path, fn, ext = get_path_filename_and_ext_of(file_in)
        shutil.move(file_in, data_dir + '/%s.%s' % (fn, ext) )
        file_in = data_dir + '/%s.%s' % (fn, ext)
        
        if os.path.isdir(ddir):
            # cmd = 'rm -r %s' % ddir
            # run_command(cmd)   
            shutil.rmtree(ddir)

    # print('WARNING:   Data check passed. %s' % (file_in))
    return file_in

'''
def trim_mtx_dir(ddir):
    
    flst = os.listdir(ddir)
    if len(flst) > 0:
        for fn in flst:
            items = fn.split('.')
            if items[0] == 'genes':
                fn_new = 'features'
                for itm in items[1:]:
                    fn_new = fn_new + '.%s' % itm
                cmd = 'mv %s %s' % (ddir + '/%s' % fn, ddir + '/%s' % fn_new )
                run_command(cmd)
    return
#'''

def check_mtx_prefix_old(ddir):
    
    tlst = ['matrix.mtx', 'features.tsv', 'barcodes.tsv']
    flst = os.listdir(ddir)
    
    chk = [0]*len(tlst)
    pfx = None
    for k, t in enumerate(tlst):
        b = False
        for f in flst:
            pos = f.find(t)
            if (t == tlst[1]) & (pos < 0):
                pos = f.find('genes.tsv')
                
            if pos >= 0:
                b = True
                if pos > 0:
                    if pfx is None:
                        pfx = f[:pos]
                    else:
                        if pfx != f[:pos]:
                            print('WARNING: different prefixes were used.')
                chk[k] = 1
                break
                
    return pfx, (np.sum(chk) == 3)


def check_mtx_prefix(ddir):
    
    pfx = None 
    chk = [0]*3
    flst = os.listdir(ddir)
    if len(flst) < 3:
        return None, False
    else:
        for fn in flst:
            items = fn.split('.')
            if len(items) >= 2:
                pos_g = fn.find('genes')
                pos_f = fn.find('features')
                pos_t = fn.find('tsv')
                pos_b = fn.find('barcodes')
                pos_m = fn.find('matrix')
                pos_x = fn.find('mtx')

                ext = items[-1]
                ext2 = items[-2]

                if (pos_m >= 0) & (pos_x >= 0):
                    chk[2] = 1
                    if ext == 'mtx':
                        if fn != 'matrix.mtx':
                            # cmd = 'mv %s %s' % (ddir + '/%s' % fn, ddir + '/matrix.mtx' )
                            # run_command(cmd)
                            shutil.move(ddir + '/%s' % fn, ddir + '/matrix.mtx')
                            
                    elif (ext == 'gz') & (ext2 == 'mtx'):
                        if fn != 'matrix.mtx.gz':
                            # cmd = 'mv %s %s' % (ddir + '/%s' % fn, ddir + '/matrix.mtx.gz' )
                            # run_command(cmd)
                            shutil.move(ddir + '/%s' % fn, ddir + '/matrix.mtx.gz')
                
                elif (pos_b >= 0) & (pos_t >= 0):
                    chk[1] = 1
                    if ext == 'tsv':
                        if fn != 'barcodes.tsv':
                            # cmd = 'mv %s %s' % (ddir + '/%s' % fn, ddir + '/barcodes.tsv' )
                            # run_command(cmd)
                            shutil.move(ddir + '/%s' % fn, ddir + '/barcodes.tsv')
                            
                    elif (ext == 'gz') & (ext2 == 'tsv'):
                        if fn != 'barcodes.tsv.gz':
                            # cmd = 'mv %s %s' % (ddir + '/%s' % fn, ddir + '/barcodes.tsv.gz' )
                            # run_command(cmd)
                            shutil.move(ddir + '/%s' % fn, ddir + '/barcodes.tsv.gz')
                
                elif (pos_g >= 0) & (pos_t >= 0):
                    chk[0] = 1
                    if ext == 'tsv':
                        if fn != 'genes.tsv':
                            # cmd = 'mv %s %s' % (ddir + '/%s' % fn, ddir + '/genes.tsv' )
                            # run_command(cmd)
                            shutil.move(ddir + '/%s' % fn, ddir + '/genes.tsv')
                            
                        df = pd.read_csv(ddir + '/genes.tsv', sep = '\t', header = None)
                        if df.shape[1] == 1:
                            df['gene_name'] = df[0]
                            # cmd = 'rm %s' % (ddir + '/genes.tsv')
                            # run_command(cmd)
                            os.remove(ddir + '/genes.tsv')
                            df.to_csv(ddir + '/genes.tsv', header = False, sep = '\t', index = False)                            
                            
                    elif (ext == 'gz') & (ext2 == 'tsv'):
                        if fn != 'genes.tsv.gz':
                            # cmd = 'mv %s %s' % (ddir + '/%s' % fn, ddir + '/genes.tsv.gz' )
                            # run_command(cmd)
                            shutil.move(ddir + '/%s' % fn, ddir + '/genes.tsv.gz')
                
                elif (pos_f >= 0) & (pos_t >= 0):
                    chk[0] = 1
                    if ext == 'tsv':
                        if fn != 'features.tsv': # 'genes.tsv':
                            # cmd = 'mv %s %s' % (ddir + '/%s' % fn, ddir + '/features.tsv' )
                            # run_command(cmd)
                            shutil.move(ddir + '/%s' % fn, ddir + '/features.tsv')
                            
                        df = pd.read_csv(ddir + '/features.tsv', sep = '\t', header = None)
                        if df.shape[1] == 1:
                            df['gene_name'] = df[0]
                            # cmd = 'rm %s' % (ddir + '/features.tsv')
                            # run_command(cmd)
                            os.remove(ddir + '/features.tsv')
                            df.to_csv(ddir + '/features.tsv', header = False, sep = '\t', index = False) 
                                
                    elif (ext == 'gz') & (ext2 == 'tsv'):
                        if fn != 'features.tsv.gz':
                            # cmd = 'mv %s %s' % (ddir + '/%s' % fn, ddir + '/features.tsv.gz' )
                            # run_command(cmd)
                            shutil.move(ddir + '/%s' % fn, ddir + '/features.tsv.gz')
                
        return pfx, (np.sum(chk) == 3)


def get_data_from_10x_mtx(data_dir, ddir, file_name):

    H5AD_FILE = '%s.h5ad' % file_name
    dcnt, new_ddir = count_dir_in( ddir )
    
    if dcnt <= 1:
        #'''
        if dcnt == 0:
            Dir = ddir
        else:
            Dir = new_ddir            
        try:
            # trim_mtx_dir(Dir)
            pfx, chk = check_mtx_prefix(Dir)
            # print('CLIENT INFO: prefix = %s' % (pfx))
            if chk:
                adata = sc.read_10x_mtx(Dir, 
                                var_names='gene_symbols', 
                                cache=True, prefix = pfx ) 
            else:
                print('WARNING: CellRanger output files not suitably formatted. ', os.listdir(Dir))
                return None                
        except:
            print('WARNING: cannot read data in %s ' % Dir, os.listdir(Dir))
            return None

        adata.var_names_make_unique()
        adata.obs_names_make_unique()

        #'''
        #'''
        if 'condition' in list(adata.obs.columns.values):
            adata.obs['condition'] = adata.obs['condition'].astype(str) 
        else:
            print('''WARNING: 'condition' not specified in meta_data.csv''')
            adata.obs['condition'] = 'Not_specified'

        if 'sample' in list(adata.obs.columns.values):
            adata.obs['sample'] = adata.obs['sample'].astype(str)  
        else:
            print('''WARNING: 'sample' not specified in meta_data.csv''')
            adata.obs['sample'] = 'Not_specified'
        #'''    
        #'''
        file_path = data_dir + '/%s' % H5AD_FILE
        adata.write(file_path)

        if not os.path.isfile(file_path):
            print('WARNING: File writing failed.')
            return None
        #'''
        # print('ERROR: No data folder(s) found.')
        # return None
        
    else:
        ## Get meta data if exists
        #'''
        meta_file = '%s/%s' % (ddir, META_FILE_NAME)

        if os.path.exists(meta_file):
            df = pd.read_csv(meta_file, index_col = 0)
        else:
            df = None
        #'''
        dlst, df = get_meta_data(ddir)
        
        if df is not None:
            dlst2 = list(df.index.values)
            dlstc = list(set(dlst).intersection(dlst2))
            if len(dlstc) < len(dlst):
                print('WARNING: index in the meta_data.csv not match with the folder names containg CellRanger output files.')
                print('ERROR: meta_data.csv not suitably formatted.')
                return None
        
        # dlst = os.listdir(ddir)
        dlst.sort()
        cnt = 0
        for d in dlst:
            d2 = '%s/%s' % (ddir, d)
            if os.path.isdir( d2 ):

                lst2 = os.listdir(d2)
                if len(lst2) == 3:
                    print(d2)                
                    try:
                        # trim_mtx_dir(d2)
                        pfx, chk = check_mtx_prefix(d2)
                        if chk:
                            adata = sc.read_10x_mtx(d2, 
                                            var_names='gene_symbols', 
                                            cache=True, prefix = pfx ) 
                        else:
                            print('WARNING: CellRanger output files not suitably formatted. ', os.listdir(d2))
                            return None                
                    except:
                        print('WARNING: cannot read data in %s ' % d2, os.listdir(d2))
                        return None

                    adata.var_names_make_unique()
                    adata.obs_names_make_unique()

                    bcds = list(adata.obs.index.values)
                    bcds_new = ['%s-%s' % (d, a) for a in bcds]
                    rend = dict(zip(bcds, bcds_new))
                    adata.obs.rename(index = rend, inplace = True)
                    adata.obs['sample'] = d

                    if df is None:
                        adata.obs['condition'] = d
                    else:
                        if d in list(df.index.values):
                            cols = list(df.columns.values)
                            if 'condition' not in cols:
                                adata.obs['condition'] = d
                            else:
                                adata.obs['condition'] = df.loc[d, 'condition']
                                
                            if 'sample' not in cols:
                                adata.obs['sample'] = d
                            else:
                                adata.obs['sample'] = df.loc[d, 'sample']
                                
                            for c in cols:
                                adata.obs[c] = df.loc[d, c]

                    if cnt == 0:
                        adata_a = adata
                    else:
                        adata_a = anndata.concat([adata_a, adata], axis = 0)
                        cols = list(adata.var.columns.values)
                        adata_a.var[cols] = adata.var[cols]
                    cnt += 1
        #'''
        if 'condition' in list(adata_a.obs.columns.values):
            adata_a.obs['condition'] = adata_a.obs['condition'].astype(str) 
        else:
            adata_a.obs['condition'] = 'Not_specified'

        if 'sample' in list(adata_a.obs.columns.values):
            adata_a.obs['sample'] = adata_a.obs['sample'].astype(str)  
        else:
            adata_a.obs['sample'] = 'Not_specified'
        #'''
        file_path = data_dir + '/%s' % H5AD_FILE
        adata_a.write(file_path)

        if not os.path.isfile(file_path):
            print('WARNING: File writing failed.')
            return None
        
    # print('   Data check passed.')
    if os.path.isdir(ddir):
        # cmd = 'rm -r %s' % ddir
        # run_command(cmd)   
        shutil.rmtree(ddir)
        
    ddir = '%s/%s' % (data_dir, 'data')
    if os.path.isdir(ddir):
        # cmd = 'rm -r %s' % ddir
        # run_command(cmd)    
        shutil.rmtree(ddir)
    
    return file_path


def update_log_and_print( s, flog, log_lines = None, c_prefix = None ):
    
    flog.writelines(s + '\n')
    flog.flush()
    if log_lines is not None: 
        log_lines = log_lines + '%s\n' % s
    if c_prefix is not None:
        sa = '%s%s' % (c_prefix, s)
        print(sa, flush = True)
        
    return log_lines


def check_data( data_dir, data_type, species, session_id, 
                tumor_id = 'no', 
                home_dir = '/mnt/HDD2/sc_comp_serv',
                c_prefix = 'CLIENT INFO: '  ):

    s_prefix = 'Checking'

    log_lines = ''
    log_file = data_dir + '/%s.log' % session_id
    with open(log_file, 'w+') as flog:
    
        s = 'session ID = %s ' % (session_id)
        log_lines = update_log_and_print( s, flog, log_lines = log_lines, c_prefix = c_prefix )
        s= '%s data .. Type(%s), Species(%s), TumorId(%s) ' % (s_prefix, data_type, species, tumor_id) 
        log_lines = update_log_and_print( s, flog, log_lines = log_lines, c_prefix = c_prefix )
        
        # print('   Data Dir: %s' % data_dir, flush = True)
        # print('%s   Data Type: %s' % (c_prefix, data_type)) #, flush = True)
        # print('%s   Species: %s' % (c_prefix, species))
        # print('%s   Tumor ident.: %s' % (c_prefix, tumor_id)) #, flush = True)
        # print('   Session ID: %s' % session_id, flush = True)
        
        lst = os.listdir(data_dir)
        ## Check the number of files uploaded
        flst_s = ''
        for f in lst:
            flst_s = flst_s + '%s, ' % f
        flst_s = flst_s[:-2]
        s = 'Uploaded file(s): %s' % (flst_s)
        log_lines = update_log_and_print( s, flog, log_lines = log_lines, c_prefix = c_prefix )
        
        if len(lst) == 0:
            s = 'WARNING: No files uploaded.'
            log_lines = update_log_and_print( s, flog, log_lines = log_lines, c_prefix = '' )
            s = 'ERROR: data check failed. (No valid file not found)'
            log_lines = update_log_and_print( s, flog, log_lines = log_lines, c_prefix = '' )
            return None
        else:
            ## Check if uploaded files are compressed or not
            cnt = 0
            for f in lst:
                ext = f.split('.')[-1]
                if (ext == 'zip') | (ext == 'gz'): cnt += 1
            if cnt == 0:
                s = 'WARNING: Uploaded files are not zip or tar.gz. '
                log_lines = update_log_and_print( s, flog, log_lines = log_lines, c_prefix = '' )
                s = 'WARNING: Please check the instruction to prepare datasets to upload.'
                log_lines = update_log_and_print( s, flog, log_lines = log_lines, c_prefix = '' )
                s = 'ERROR: data check failed. (Upload file must be  zip- or tar.gz-compressed)'
                log_lines = update_log_and_print( s, flog, log_lines = log_lines, c_prefix = '' )
                return None
    
        ecode, rcode, ddir, fn = decompress(data_dir)
        # print(ecode, rcode, ddir, fn)
        
        file_h5ad = None
        ## Check if any error while decompress the uploaded file(s)
        if (ecode != 0) | (fn is None):
            s = 'WARNING: No valid files found.'
            log_lines = update_log_and_print( s, flog, log_lines = log_lines, c_prefix = '' )
            s = 'ERROR: data check failed. (h5ad file not found)'
            log_lines = update_log_and_print( s, flog, log_lines = log_lines, c_prefix = '' )
            return None
        else:
            if (data_type == 'h5ad'):
                file_h5ad = get_data_from_h5ad(data_dir, ddir, file_name = fn)
            elif (data_type == 'csv'):
                file_h5ad = get_data_from_csv(data_dir, ddir, file_name = fn)
            else: ## 10x_mtx
                file_h5ad = get_data_from_10x_mtx(data_dir, ddir, file_name = fn)
            
        if file_h5ad is None:
            s= 'WARNING: The data might not be suitably formatted or the type you provided is wrong one.'
            log_lines = update_log_and_print( s, flog, log_lines = log_lines, c_prefix = '' )
            s = 'WARNING: Please check the instruction to prepare datasets to upload.'
            log_lines = update_log_and_print( s, flog, log_lines = log_lines, c_prefix = '' )
            s = 'ERROR: data check failed.'
            log_lines = update_log_and_print( s, flog, log_lines = log_lines, c_prefix = '' )
        else: 
            s = 'SUCCESS: data check successful.'
            log_lines = update_log_and_print( s, flog, log_lines = log_lines, c_prefix = '' )
            
            try:
                adata = anndata.read_h5ad(file_h5ad)
            except:
                s = 'WARNING: Cannot open the generated file %s.' % file_h5ad
                log_lines = update_log_and_print( s, flog, log_lines = log_lines, c_prefix = '' )
                s = 'ERROR: data processing failed.'
                log_lines = update_log_and_print( s, flog, log_lines = log_lines, c_prefix = '' )
                
            adata.uns['log'] = log_lines
            adata.write(file_h5ad)      
                    
        return file_h5ad


Error_prob = 0.5

def remove_file(file_name):
    if file_name is not None:
        if os.path.isfile(file_name):
            # run_command('rm %s' % (file_name))
            os.remove(file_name)

def remove_dir(dname):
    if dname is not None:
        if os.path.isdir(dname):
            # run_command('rm -r %s' % (dname))
            shutil.rmtree(dname)


def get_path_filename_and_ext_of(path_file_name_ext):

    items = path_file_name_ext.split('/')
    if len(items) == 1:
        path = ''
    else:
        path = items[0]
        for itm in items[1:-1]:
            path = path + '/%s' % itm            
    file_name_ext = items[-1]
    
    items = file_name_ext.split('.')
    file_name = items[0]
    for item in items[1:-1]:
        file_name = file_name + '.%s' % item
    ext = items[-1]
    
    return path, file_name, ext


def load_keys( f_info ):
    
    kpL = None
    kpM = None
    key1 = None
    key2 = None
    N_cells_org = None
    
    with open(f_info, 'r') as f:
        for line in f:
            key_val = line.split(':')
            if key_val[0] == 'Key':
                s = key_val[1]
                items = s.split('/')
                kpL = int(items[0])
                kpM = int(items[1])
                key1 = items[2]
                key2 = items[3]
            elif key_val[0] == 'N_cells_in':
                N_cells_org = int(key_val[1])
                
        f.close()

    return kpL, kpM, key1, key2, N_cells_org

def get_keys(key_val):

    s = key_val
    items = s.split('/')
    kpL = int(items[0])
    kpM = int(items[1])
    key1 = items[2]
    key2 = items[3]
    
    return kpL, kpM, key1, key2
    

def load_info( f_info ):

    tlst = ['Session_ID', 'N_cells_in', 'N_cells_filtered', 'Price', 'Key']
    info = {}
    with open(f_info, 'r') as f:
        for line in f:
            key_val_org = line.split(':')
            if len(key_val_org) > 1:
                key_val = [s.strip() for s in key_val_org]

                if key_val[0] in tlst:
                    if key_val[1].isnumeric():
                        info[key_val[0]] = int(key_val[1])
                    else: 
                        info[key_val[0]] = key_val[1]
                else:
                    info[key_val[0]] = key_val[1]
        f.close()

    for key in tlst:
        if key not in list(info.keys()):
            info[key] = None
        
    return info

    
def save_info( file_hist_expected, info, session_id, N_cells_org, N_cells, unit_price, keyt ):

    tlst = ['Session_ID', 'N_cells_in', 'N_cells_filtered', 'Price', 'Key']
    info_line_lst = []
    for k in info.keys():
        if k not in tlst:
            s = '%s: %s\n' % (k, info[k])
            info_line_lst.append(s)
    
    s = 'Session_ID: %s\n' % session_id
    info_line_lst.append(s)
    s = 'N_cells_in: %i\n' % N_cells_org
    info_line_lst.append(s)
    s = 'N_cells_filtered: %i\n' % N_cells
    info_line_lst.append(s)
    # s = 'N_cells_valid: %i\n' % N_cells_valid
    # info_line_lst.append(s)
    s = 'Price: %i\n' % (int(N_cells*unit_price))
    info_line_lst.append(s)
    s = 'Key: %s' % (keyt)
    info_line_lst.append(s)

    with open(file_hist_expected, 'wt+') as f:
        f.writelines(info_line_lst)
        f.close()
    return


def process_data( data_dir, data_type = 'h5ad', species = 'human', 
                  session_id = None, tumor_id = 'no', 
                  home_dir = '/home/comp_serv/sc_comp_serv',
                  c_prefix = 'CLIENT INFO: ', lim_Ncells = True, 
                  compress_out = True, tissue = None, 
                  file_h5ad_in = None, file_h5ad_suffix = '_scoda',
                  file_cfg_in = None, file_mkr_in = None,
                  file_cpdb_in = None, file_gmt_in = None ):

    ## Fixed param
    sample_col = 'sample'
    cond_col = 'condition'

    #################################################################
    ###                                                           ###
    
    file_h5ad = file_h5ad_in
    if file_h5ad is not None:
        items = file_h5ad.split('.')
        f2 = ''
        if items[-1] == 'h5ad':
            for kx, item in enumerate(items[:-1]):
                f2 = f2 + '%s.' % item
                if kx == (len(items[:-1])-1):
                    f2 = f2[:-1] + '%s' % file_h5ad_suffix
            if os.path.isfile(f2 + '.h5ad'):
                os.remove(f2 + '.h5ad')
            shutil.copyfile( file_h5ad, f2 + '.h5ad' )
        elif data_type == 'h5ad':
            for kx, item in enumerate(items):
                f2 = f2 + '%s.' % item
                if kx == (len(items)-1):
                    f2 = f2[:-1] + '%s' % file_h5ad_suffix    
            if os.path.isfile(f2 + '.h5ad'):
                os.remove(f2 + '.h5ad')
            shutil.copyfile( file_h5ad + '.h5ad', f2 + '.h5ad' )
            
        file_h5ad = f2 + '.h5ad'
        session_id = f2
        
        info_line_lst = []
        log_lines = ''
        log_file = data_dir + '/%s.log' % session_id

        f = open(log_file, 'w')
        f.close()

    else: 
        info_line_lst = []
        log_lines = ''
        log_file = data_dir + '/%s.log' % session_id
    
        if os.path.isfile(log_file):
            f = open(log_file, 'r')
            lines = f.readlines()
            log_lines = log_lines + ''.join(lines)
            f.close()
        else:
            f = open(log_file, 'w')
            f.close()
    
    with open(log_file, 'a') as flog:
    
        start_time = time.time()
        ## Remove original data file
        flst = os.listdir(data_dir)
        fs = ''
        for f in flst:
            if os.path.isfile(data_dir + '/%s' % f):
                items = f.split('.')
                if items[-1] != 'h5ad':
                    fs = fs + '%s, ' % f
        fs = fs[:-2]
    
        #######################################
        ## arguments received from shell script
    
        # session_id = '10x_multi_test'
        # data_dir = 'results'
        # species = 'mouse'
    
        print_prefix = c_prefix
        s_prefix = 'Processing'
        
        s = '%s data .. ' % (s_prefix)
        log_lines = update_log_and_print( s, flog, log_lines = log_lines, c_prefix = c_prefix )
            
        tumor_identification = False
        if isinstance(tumor_id, str):
            if (tumor_id.lower()[0] == 'y') | (tumor_id.lower()[0] == 't'):
                tumor_identification = True
                
        user_param = {}
        user_param['species'] = species
        user_param['data_format'] = data_type
        user_param['tumor_id'] = tumor_identification
        user_param['uploaded_files'] = fs
        
        # home_dir = '/mnt/HDD2/sc_comp_serv'
        default_file_folder = home_dir + '/default_optional_files'
        if not os.path.isdir(default_file_folder):
            default_file_folder = pkg_resources.resource_filename('scoda', 'default_optional_files')
            
        opt_file_path = default_file_folder
    
        ########################
        ## Service config params
    
        max_num_cells_allowed = 1000000
        n_cores_to_use = 6
        unit_price = 20
    
        serv_config_file = home_dir + '/service_config.txt'

        if os.path.isfile(serv_config_file):
            with open(serv_config_file, 'r') as f:
                for line in f:
                    items = line.split(' ')
                    items2 = []
                    for item in items: items2 = items2 + item.split("\t")
                    items = copy.deepcopy(items2)
                    # print(items)
                    if (len(items) >= 2) & (items[0][0] != '#') & (items[0][0] != '\n'):
                        key = items[0].strip()
                        val = items[1].strip()
        
                        if key == 'PATH_TO_WORKSPACE':
                            pass
                        elif key == 'PATH_TO_DEFAULT_OPTIONAL_FILES':
                            default_file_folder = val
                        elif key == 'PATH_TO_ANACONDA':
                            pass
                        elif key == 'CONDA_ENV_TO_USE':
                            pass
                        elif key == 'MAX_NUM_LIVE_SESSIONS':
                            pass
                        elif key == 'MAX_QUEUE_LENGTH':
                            pass
                        elif key == 'NUM_CORES_PER_SESSION':
                            n_cores_to_use = int(val)
                        elif key == 'MAX_NUM_CELLS_ALLOWED':
                            if (lim_Ncells == True) | (lim_Ncells == 'True'):
                                max_num_cells_allowed = int(val)
                        elif key == 'PATH_TO_RESULT_FILES':
                            pass
                        elif key == 'DAYS_TO_DELETE_RESULT_FILES':
                            pass
                        elif key == 'PORT_START':
                            pass
                        elif key == 'UNIT_PRICE':
                            unit_price = int(val)

    
                # s = 'Info: Max # of cells allowed: %i  ' % (max_num_cells_allowed)
                # update_log_and_print( s, flog, log_lines = None, c_prefix = c_prefix )        
        else:
            s = 'Info: service_config.txt not available. ' 
            update_log_and_print( s, flog, log_lines = None, c_prefix = c_prefix )
            
        ###                                                           ###
        #################################################################
        
        lst = os.listdir(data_dir)
        
        file_gmt = None
        file_mkr = None
        file_cfg = None
        file_cpdb = None
        file_gtf = None
        file_hist = None
        info_dict = {}
        file_hist_expected = '%s.info' % (session_id)
        
        for f in lst:
            if f == 'msig.gmt':
                file_gmt = data_dir + '/%s' % f
                
            if f == 'markers.tsv':
                file_mkr = data_dir + '/%s' % f
                
            if (f == 'cpdb.zip'): # | (f == 'cpdb,tar.gz'):
                file_cpdb = data_dir + '/%s' % f
                
            if f == 'analysis_config.py':
                file_cfg = data_dir + '/%s' % f
                
            if f == file_hist_expected:
                info_dict = load_info( data_dir + '/%s' % file_hist_expected )
                file_hist = file_hist_expected
                
                if (info_dict['Key'] is None) | (info_dict['N_cells_in'] is None) | (info_dict['Session_ID'] is None):
                    s = 'New order '
                    update_log_and_print( s, flog, log_lines = None, c_prefix = c_prefix )
                else:
                    if info_dict['Session_ID'] == session_id:
                        s = 'Returning order (%s == %s) ' % (info_dict['Session_ID'], session_id)
                        update_log_and_print( s, flog, log_lines = None, c_prefix = c_prefix )
                        
                    else:
                        s = 'It seems returning order. But %s != %s ' % (info_dict['Session_ID'], session_id)
                        update_log_and_print( s, flog, log_lines = None, c_prefix = '' )
                    
            else:
                if file_h5ad is None:
                    items = f.split('.')
                    if items[-1] == 'h5ad':
                        file_h5ad = data_dir + '/%s' % f
    
        if file_hist is None:
            s = 'Info: Order information not provided. Will process assuming new one.'
            update_log_and_print( s, flog, log_lines = None, c_prefix = c_prefix )
         
        ########################
        ## Check Gene names ####
        if species.lower() == 'mouse':
            mkr_file_all = '%s/markers_mm_all_tissues.tsv' % (opt_file_path)
            gtf_file = '%s/mm10_gene_only.gtf' % (opt_file_path)
        else:
            mkr_file_all = '%s/markers_hs_all_tissues.tsv' % (opt_file_path)
            gtf_file = '%s/hg38_gene_only.gtf' % (opt_file_path)
    
        df_mkr_db_all = pd.read_csv(mkr_file_all, sep = '\t')
        gtf_line_lst, hdr_lines = load_gtf( gtf_file, verbose = False, ho = False )
        df_gtf = pd.DataFrame(gtf_line_lst)
        glst = list(df_gtf['gname'])
    
        try:
            adata = anndata.read_h5ad(file_h5ad)
        except:
            s = 'WARNING: Cannot open the generated file %s.' % file_h5ad
            update_log_and_print( s, flog, log_lines = None, c_prefix = '' )
            s = 'ERROR: data processing failed. (h5ad file not found)'
            update_log_and_print( s, flog, log_lines = None, c_prefix = '' )
            return None
    
        glst2 = list(adata.var.index.values)
        glstc = list(set(glst).intersection(glst2))
        if len(glstc) <= len(glst2)*0.25:
            s = 'WARNING: Only %i out of %i genes were overlapped with %s genes. ' % (len(glstc), len(glst2), species)
            update_log_and_print( s, flog, log_lines = None, c_prefix = '' )
            s = 'WARNING: Maybe you are using Gene ID, not Gene symbol, or provided wrong species.'
            update_log_and_print( s, flog, log_lines = None, c_prefix = '' )
            s = 'ERROR: data processing failed. (GeneName issue)'
            update_log_and_print( s, flog, log_lines = None, c_prefix = '' )
            return None            
            
        ## Check Gene names ####
    
        ## Generate key ########
        if 'Key' not in list(info_dict.keys()):
            kpL = 32
            kpM = 25
        else:
            kpLo, kpMo, key1o, key2o = get_keys(info_dict['Key'])
            kpL = kpLo
            kpM = kpMo   
            
        key1n, int_lst = get_dataset_key(adata, L = kpL, M = kpM)
        
        ## Generate key ########
        if 'Key' not in list(info_dict.keys()):
            key1 = key1n
            N_cells_org = adata.obs.shape[0]
        else:
            kpLo, kpMo, key1o, key2o = get_keys(info_dict['Key'])
            key1 = key1o  
            N_cells_in = info_dict['N_cells_in']
            N_cells_org = N_cells_in
            
            if (key1n != key1o) & (key1n != key2o):
                s = 'WARNING: CheckSum failed. The uploaded data is not the one you used before.'
                update_log_and_print( s, flog, log_lines = None, c_prefix = '' )
                s = 'WARNING: Please use the data you uploaded before or the SCODA result file you downloaded.'
                update_log_and_print( s, flog, log_lines = None, c_prefix = '' )
                s = 'ERROR: data processing failed. (Checksum failed)'
                update_log_and_print( s, flog, log_lines = None, c_prefix = '' )
                return None
            else:
                s = '   CheckSum passed.'
                update_log_and_print( s, flog, log_lines = None, c_prefix = '' )
            
        ## Generate key ########
        ########################            
            
        ss1 = 'markers db, pathways db, CCI db, config'
        ss2 = ''
        user_supplied_cfg = False
        user_supplied_mkr = False
        user_supplied_cpdb = False
        user_supplied_pwdb = False
        
        if file_mkr is not None:
            ss2 = ss2 + 'O, '
            user_supplied_mkr = True
        else:
            ss2 = ss2 + 'X, '
            
            b_mkr = False
            if file_mkr_in is not None:
                if isinstance(file_mkr_in, str):
                    if os.path.isfile(file_mkr_in):
                        file_mkr_org = file_mkr_in    
                        b_mkr = True
                        file_mkr = file_mkr_org
                        
                        s = '   mkr file used: %s' % (file_mkr)
                        log_lines = update_log_and_print( s, flog, log_lines = log_lines, c_prefix = c_prefix )
            
            if not b_mkr:            
                if species.lower() == 'mouse':
                    file_mkr_org = '%s/markers_mm.tsv' % (opt_file_path)
                    file_mkr_new = '%s/markers_mm.tsv' % (data_dir)
                else:
                    file_mkr_org = '%s/markers_hs.tsv' % (opt_file_path)
                    file_mkr_new = '%s/markers_hs.tsv' % (data_dir)
    
                file_mkr = file_mkr_org
                '''
                shutil.copyfile( file_mkr_org, file_mkr_new )            
                file_mkr = '%s/markers.tsv' % (data_dir)
                shutil.move(file_mkr_new, file_mkr)
                '''
            
            if not os.path.isfile(file_mkr):
                pycache_dir = data_dir + '/__pycache__'
                remove_dir( pycache_dir )    
                remove_file( file_cfg )
                if user_supplied_mkr: remove_file( file_mkr )
                if user_supplied_pwdb: remove_file( file_gmt )
                if user_supplied_cpdb: remove_file( file_cpdb )
                print('ERROR: markers.tsv not found.')
                return
    
        if file_gmt is not None:
            ss2 = ss2 + 'O, '
            user_supplied_pwdb = True
        else:
            ss2 = ss2 + 'X, '

            b_gmt = False
            if file_gmt_in is not None:
                if isinstance(file_gmt_in, str):
                    if os.path.isfile(file_gmt_in):
                        file_gmt_org = file_gmt_in    
                        b_gmt = True
                        file_gmt = file_gmt_org
                        
                        s = '   gmt file used: %s' % (file_gmt)
                        log_lines = update_log_and_print( s, flog, log_lines = log_lines, c_prefix = c_prefix )
                        # file_gmt = '%s/msig.gmt' % (data_dir)
                        # shutil.move( '%s' % (file_gmt_org), file_gmt )
            
            if not b_gmt:
                if species.lower() == 'mouse':
                    file_gmt_org = 'wiki_mouse.gmt'
                else:
                    file_gmt_org = 'wiki_human.gmt'

                file_gmt = '%s/%s' % (opt_file_path, file_gmt_org)
                '''
                shutil.copyfile( '%s/%s' % (opt_file_path, file_gmt_org), '%s/%s' % (data_dir, file_gmt_org) )
                file_gmt = '%s/msig.gmt' % (data_dir)
                shutil.move( '%s/%s' % (data_dir, file_gmt_org), file_gmt )
                #'''
                
            if not os.path.isfile(file_gmt):
                pycache_dir = data_dir + '/__pycache__'
                remove_dir( pycache_dir )    
                remove_file( file_cfg )
                if user_supplied_mkr: remove_file( file_mkr )
                if user_supplied_pwdb: remove_file( file_gmt )
                if user_supplied_cpdb: remove_file( file_cpdb )
                remove_file( file_h5ad )
                print('ERROR: msig.gmt not found.')
                return
    
        if file_cpdb is not None:
            ss2 = ss2 + 'O, '
            user_supplied_cpdb = True
        else:
            ss2 = ss2 + 'X, '

            b_cpbd = False
            if file_cpdb_in is not None:
                if isinstance(file_cpdb_in, str):
                    if os.path.isfile(file_cpdb_in):
                        file_cpdb_org = file_cpdb_in    
                        b_cpdb = True
                        
                        file_cpdb = file_cpdb_org
                        
                        s = '   cpdb file used: %s' % (file_cpdb)
                        log_lines = update_log_and_print( s, flog, log_lines = log_lines, c_prefix = c_prefix )
            
            if not b_cpbd:            
                file_cpdb_org = '%s/cellphonedb_v4_nox2_added.zip' % (opt_file_path)            
                file_cpdb = file_cpdb_org
                '''
                file_cpdb = '%s/cpdb.zip' % (data_dir)            
                shutil.copyfile( file_cpdb_org, file_cpdb )
                '''
            
            if not os.path.isfile(file_cpdb):
                pycache_dir = data_dir + '/__pycache__'
                remove_dir( pycache_dir )    
                remove_file( file_cfg )
                if user_supplied_mkr: remove_file( file_mkr )
                if user_supplied_pwdb: remove_file( file_gmt )
                if user_supplied_cpdb: remove_file( file_cpdb )
                remove_file( file_h5ad )
                print('ERROR: cci.db not found.')
                return
    
        if file_cfg is not None:
            ss2 = ss2 + 'O'
            user_supplied_cfg
        else:
            ss2 = ss2 + 'X'

            b_cfg = False
            if file_cfg_in is not None:
                if isinstance(file_cfg_in, str):
                    if os.path.isfile(file_cfg_in):
                        file_cfg_org = file_cfg_in    
                        b_cfg = True
                        
                        # file_cfg = file_cfg_org
                        file_cfg = '%s/analysis_config.py' % (data_dir)
                        shutil.copyfile( file_cfg_org, file_cfg )
                        
                        s = '   config file used: %s' % (file_cfg_in)
                        log_lines = update_log_and_print( s, flog, log_lines = log_lines, c_prefix = c_prefix )
            
            if not b_cfg:
                file_cfg_org = '%s/analysis_config.py' % (opt_file_path)            
                file_cfg = '%s/analysis_config.py' % (data_dir)
                shutil.copyfile( file_cfg_org, file_cfg )
            
            if not os.path.isfile(file_cfg):
                pycache_dir = data_dir + '/__pycache__'
                remove_dir( pycache_dir )    
                remove_file( file_cfg )
                if user_supplied_mkr: remove_file( file_mkr )
                if user_supplied_pwdb: remove_file( file_gmt )
                if user_supplied_cpdb: remove_file( file_cpdb )
                remove_file( file_h5ad )
                print('ERROR: analysis config file not found.')
                return
    
        
        s = 'User supplied: %s = %s ' % (ss1, ss2)
        log_lines = update_log_and_print( s, flog, log_lines = log_lines, c_prefix = c_prefix )
        
        ## Main process
        if file_h5ad is None:
            pycache_dir = data_dir + '/__pycache__'
            remove_dir( pycache_dir )    
            remove_file( file_cfg )
            if user_supplied_mkr: remove_file( file_mkr )
            if user_supplied_pwdb: remove_file( file_gmt )
            if user_supplied_cpdb: remove_file( file_cpdb )
            remove_file( file_h5ad )
            print('ERROR: h5ad file not found.' )
            return None
            
        #################################################################
        ###                                                           ###
        
        ###  common params ###
        # n_cores_to_use = 4
        verbose = True
    
        anal_param = {}
        ###  params for iCNV  ###
        # tumor_id_ref_celltypes = None
        tumor_id_ref_celltypes = ['T cell', 'B cell', 'Myeloid cell', 'Fibroblast']
        tumor_id_ref_condition = ['Normal'] 
    
        ###  params for CCI  ###
        unit_of_cci_run = sample_col    # 'sample'  or 'condition'
        cci_min_OF_to_count = 0.67     # Valid when unit_of_cci_run = 'sample' 
        cci_pval_max = 0.1
        cci_mean_min = 0
        cci_deg_base = 'celltype_minor'
        min_n_cells_for_cci = 40
        cci_min_OF_to_count = 0.5
        
        ###  params for DEG/GSEA  ###
        deg_ref_group = None
        deg_pval_cutoff = 0.1
        gsea_pval_cutoff = 0.1
        min_n_cells_for_deg = 100
        
        ###################################
        ### Load analysis config params ###
    
        # print('importing sys')
        import sys, importlib
        sys.path.append(data_dir)
        # sys.path.append(os.getcwd())
    
        if not os.path.isfile(data_dir + '/analysis_config.py'):
            pycache_dir = data_dir + '/__pycache__'
            remove_dir( pycache_dir )    
            remove_file( file_cfg )
            if user_supplied_mkr: remove_file( file_mkr )
            if user_supplied_pwdb: remove_file( file_gmt )
            if user_supplied_cpdb: remove_file( file_cpdb )
            remove_file( file_h5ad )
            print('ERROR: Cannot find analysis_config.py ...')
            return
        
        # print('Loading anal_config .. ')
        # if 'analysis_config' not in sys.modules:
        import analysis_config as anal_cfg
        # else:
        importlib.reload(anal_cfg)
    
        hicat_clustering_resolution = 1
        hicat_pct_th_cutoff = 0.5
        hicat_n_cells_max_for_pca = 80000
        hicat_n_cells_max_for_gmm = 20000
        user_tissue_sel = tissue
        cnv_clustering_n_cells = 60000
        cnv_connectivity_threshold = 0.18
        cnv_clustering_resolution = 6
        cnv_window_size = 100
        cnv_uc_margin = 0.2
        cnv_filter_quantile = 0
            
        # print('Parsing .. ')
        ##########################
        ###  params for HiCAT  ###
        var_name = 'TISSUE'
        if user_tissue_sel is None:
            if hasattr(anal_cfg, var_name):
                user_tissue_sel = getattr(anal_cfg, var_name)
            else:
                user_tissue_sel = 'Generic'
            anal_param[var_name] = user_tissue_sel
        else:
            anal_param[var_name] = user_tissue_sel
            s = 'User specified tissue: %s' % user_tissue_sel
            log_lines = update_log_and_print( s, flog, log_lines = log_lines, c_prefix = c_prefix )
            
        ##########################
        ###  params for HiCAT  ###
        var_name = 'CNV_ADDON_N_CELLS_MAX_FOR_PROC'
        if hasattr(anal_cfg, var_name):
            cnv_clustering_n_cells = getattr(anal_cfg, var_name)
        else:
            cnv_clustering_n_cells = 60000
        anal_param[var_name] = cnv_clustering_n_cells
            
        var_name = 'CNV_ADDON_CONNECTIVITY_THRESHOLD'
        if hasattr(anal_cfg, var_name):
            cnv_connectivity_threshold = getattr(anal_cfg, var_name)
        else:
            cnv_connectivity_threshold = 0.18
        anal_param[var_name] = cnv_connectivity_threshold
            
        var_name = 'CNV_ADDON_CLUSTERING_RESOLUTION'
        if hasattr(anal_cfg, var_name):
            cnv_clustering_resolution = getattr(anal_cfg, var_name)
        else:
            cnv_clustering_resolution = 6
        anal_param[var_name] = cnv_clustering_resolution

        var_name = 'CNV_ADDON_NORMAL_REF_PCT_MIN'
        if hasattr(anal_cfg, var_name):
            cnv_ref_pct_min = getattr(anal_cfg, var_name)
        else:
            cnv_ref_pct_min = 0.25
        anal_param[var_name] = cnv_ref_pct_min

        var_name = 'CNV_ADDON_N_GMM_COMPONENTS'
        if hasattr(anal_cfg, var_name):
            cnv_gmm_num_components = getattr(anal_cfg, var_name)
        else:
            cnv_gmm_num_components = 3
        anal_param[var_name] = cnv_gmm_num_components
            
        var_name = 'CNV_ADDON_CONNECTIVITY_STD_SCALE_FACTOR'
        if hasattr(anal_cfg, var_name):
            cnv_connectivity_std_sf = getattr(anal_cfg, var_name)
        else:
            cnv_connectivity_std_sf = 1.5
        anal_param[var_name] = cnv_connectivity_std_sf
        
        var_name = 'CNV_ADDON_UNCEAR_MARGIN'
        if hasattr(anal_cfg, var_name):
            cnv_uc_margin = getattr(anal_cfg, var_name)
        else:
            cnv_uc_margin = 0.2
        anal_param[var_name] = cnv_uc_margin
            
        var_name = 'CNV_GENE_FILTER_QUANTILE'
        if hasattr(anal_cfg, var_name):
            cnv_filter_quantile = getattr(anal_cfg, var_name)
        else:
            cnv_filter_quantile = 0
        anal_param[var_name] = cnv_filter_quantile
            
        var_name = 'INFERCNV_WINDOW_SIZE'
        if hasattr(anal_cfg, var_name):
            cnv_window_size = getattr(anal_cfg, var_name)
        else:
            cnv_window_size = 100
        anal_param[var_name] = cnv_window_size
            
        ##########################
        ###  params for HiCAT  ###
        var_name = 'HICAT_CLUSTERING_RESOLUTION'
        if hasattr(anal_cfg, var_name):
            hicat_clustering_resolution = getattr(anal_cfg, var_name)
        else:
            hicat_clustering_resolution = 1
        anal_param[var_name] = hicat_clustering_resolution
            
        var_name = 'HICAT_PCT_TH_CUTOFF'
        if hasattr(anal_cfg, var_name):
            hicat_pct_th_cutoff = getattr(anal_cfg, var_name)
        else:
            hicat_pct_th_cutoff = 0.5
        anal_param[var_name] = hicat_pct_th_cutoff
            
        var_name = 'HICAT_N_CELL_MAX_FOR_PCA'
        if hasattr(anal_cfg, var_name):
            hicat_n_cells_max_for_pca = getattr(anal_cfg, var_name)
        else:
            hicat_n_cells_max_for_pca = 60000
        anal_param[var_name] = hicat_n_cells_max_for_pca
            
        var_name = 'HICAT_N_CELL_MAX_FOR_GMM'
        if hasattr(anal_cfg, var_name):
            hicat_n_cells_max_for_gmm = getattr(anal_cfg, var_name)
        else:
            hicat_n_cells_max_for_gmm = 60000
        anal_param[var_name] = hicat_n_cells_max_for_gmm
            
        var_name = 'HICAT_N_CELLS_MAX_FOR_PROC'
        if hasattr(anal_cfg, var_name):
            hicat_n_cells_max_for_gmm = getattr(anal_cfg, var_name)
            hicat_n_cells_max_for_pca = hicat_n_cells_max_for_gmm
        else:
            hicat_n_cells_max_for_gmm = 60000
            hicat_n_cells_max_for_pca = 60000
        anal_param[var_name] = hicat_n_cells_max_for_gmm
            
        var_name = 'HICAT_MKR_SELECTOR'
        if hasattr(anal_cfg, var_name):
            hicat_mkr_selector = getattr(anal_cfg, var_name)
        else:
            hicat_mkr_selector = '100000'
        anal_param[var_name] = hicat_mkr_selector
            
        ###################################
        ###  params for Cell Filtering  ###
        var_name = 'N_GENES_MIN'
        if hasattr(anal_cfg, var_name):
            N_genes_min = getattr(anal_cfg, var_name)
        else:
            N_genes_min = 200   
            
        ## N_genes_min이 너무 크면 N_cells가 너무 작음 -> 가격이 줄어듦 -> 
        ## 나중에 재주문시 N_genes_min을 작게해서 N_cells이 커져도 추가 과금 불가
        ## 이런 문제 해결하려면 임시방편으로 N_genes_min을 제한
        if N_genes_min > 300:
            N_genes_min = 300
            
        anal_param[var_name] = N_genes_min
            
        var_name = 'N_CELLS_MIN'
        if hasattr(anal_cfg, var_name):
            N_cells_min = getattr(anal_cfg, var_name)
        else:
            N_cells_min = 10
        anal_param[var_name] = N_cells_min
    
        var_name = 'PCT_COUNT_MT_MAX'
        if hasattr(anal_cfg, var_name):
            Pct_cnt_mt_max = getattr(anal_cfg, var_name)
        else:
            Pct_cnt_mt_max = 20
        anal_param[var_name] = Pct_cnt_mt_max
            
        #########################
        ###  params for iCNV  ###
        var_name = 'REF_CELLTYPES_FOR_TUMOR_ID'
        if hasattr(anal_cfg, var_name):
            tumor_id_ref_celltypes = list(getattr(anal_cfg, var_name))
        else:
            tumor_id_ref_celltypes = None
        anal_param[var_name] = tumor_id_ref_celltypes
            
        var_name = 'REF_CONDITION_FOR_TUMOR_ID'
        if hasattr(anal_cfg, var_name):
            if isinstance( getattr(anal_cfg, var_name), np.ndarray ) | isinstance( getattr(anal_cfg, var_name), list ):
                tumor_id_ref_condition = list(getattr(anal_cfg, var_name))
            elif isinstance( getattr(anal_cfg, var_name), str ):
                tumor_id_ref_condition = [getattr(anal_cfg, var_name)]
            else:
                tumor_id_ref_condition = None
        else:
            tumor_id_ref_condition = None # ['Normal']
        anal_param[var_name] = tumor_id_ref_condition
            
        #############################
        ###  params for DEG/GSEA  ###
        var_name = 'MIN_NUM_CELLS_FOR_CCI'
        if hasattr(anal_cfg, var_name):
            min_n_cells_for_cci = getattr(anal_cfg, var_name)
        else:
            min_n_cells_for_cci = 40
        anal_param[var_name] = min_n_cells_for_cci
    
        var_name = 'UNIT_OF_CCI_RUN'
        if hasattr(anal_cfg, var_name):
            unit_of_cci_run = getattr(anal_cfg, var_name)
        else:
            unit_of_cci_run = sample_col
        anal_param[var_name] = unit_of_cci_run
    
        var_name = 'CCI_MIN_OCC_FREQ_TO_COUNT'
        if hasattr(anal_cfg, var_name):
            cci_min_OF_to_count = getattr(anal_cfg, var_name)
        else:
            cci_min_OF_to_count = 0.67
        anal_param[var_name] = cci_min_OF_to_count
    
        var_name = 'CCI_PVAL_CUTOFF'
        if hasattr(anal_cfg, var_name):
            cci_pval_max = getattr(anal_cfg, var_name)
        else:
            cci_pval_max = 0.1
        anal_param[var_name] = cci_pval_max
    
        var_name = 'CCI_MEAN_CUTOFF'
        if hasattr(anal_cfg, var_name):
            cci_mean_min = getattr(anal_cfg, var_name)
        else:
            cci_mean_min = 0.1
        anal_param[var_name] = cci_mean_min
    
        var_name = 'CCI_DEG_BASE'
        if hasattr(anal_cfg, var_name):
            cci_deg_base = getattr(anal_cfg, var_name)
        else:
            cci_deg_base = 'celltype_minor'
        anal_param[var_name] = cci_deg_base
    
        #############################
        ###  params for DEG/GSEA  ###
        var_name = 'MIN_NUM_CELLS_FOR_DEG'
        if hasattr(anal_cfg, var_name):
            min_n_cells_for_deg = getattr(anal_cfg, var_name)
        else:
            min_n_cells_for_deg = 100
        anal_param[var_name] = min_n_cells_for_deg
    
        var_name = 'DEG_PVAL_CUTOFF_FOR_GSEA'
        if hasattr(anal_cfg, var_name):
            deg_pval_cutoff = getattr(anal_cfg, var_name)
        else:
            deg_pval_cutoff = 0.1
        anal_param[var_name] = deg_pval_cutoff
    
        var_name = 'DEG_REF_GROUP'
        if hasattr(anal_cfg, var_name):
            deg_ref_group = getattr(anal_cfg, var_name)
        else:
            deg_ref_group = None
        anal_param[var_name] = deg_ref_group
            
        var_name = 'GSA_PVAL_CUTOFF'
        if hasattr(anal_cfg, var_name):
            gsea_pval_cutoff = getattr(anal_cfg, var_name)
        else:
            gsea_pval_cutoff = 0.1
        anal_param[var_name] = gsea_pval_cutoff
                
        ###################
        cellphone_db = data_dir + '/%s' % 'cpdb.zip'
        # file = '%s/%s.h5ad' % (data_dir, session_id)
        
        ###################
        ### Load data #####
        # print('Loading data ... ', end = '')
        adata_t = anndata.read_h5ad(file_h5ad)
        # print('done')
        
        if cond_col not in list(adata_t.obs.columns.values):
            adata_t.obs[cond_col] = 'Not_specified'
        else:
            b = False
            lst = list(adata_t.obs[cond_col].unique())
            rend = {}
            for c in lst:
                if '/' in c:
                    b = True
                    rend[c] = c.replace('/', '_')
            if b:
                adata_t.obs[cond_col].replace(rend, inplace = True)
                s = 'Info: / in %s names replaced with _ ' % cond_col
                log_lines = update_log_and_print( s, flog, log_lines = log_lines, c_prefix = c_prefix )
                
        if sample_col not in list(adata_t.obs.columns.values):
            adata_t.obs[sample_col] = 'Not_specified'
        else:
            b = False
            lst = list(adata_t.obs[sample_col].unique())
            rend = {}
            for c in lst:
                if '/' in c:
                    b = True
                    rend[c] = c.replace('/', '_')
            if b:
                adata_t.obs[sample_col].replace(rend, inplace = True)
                s = 'Info: / in %s names replaced with _ ' % sample_col
                log_lines = update_log_and_print( s, flog, log_lines = log_lines, c_prefix = c_prefix )
                        
        adata_t.uns['analysis_parameters'] = anal_param
    
        cols = list(adata_t.obs.columns.values)
        if cond_col in cols:
            adata_t.obs[cond_col] = adata_t.obs[cond_col].astype(str) 
        if sample_col in cols:
            adata_t.obs[sample_col] = adata_t.obs[sample_col].astype(str)  
    
        if (cond_col in cols) & (sample_col in cols):
            slst = list(adata_t.obs[sample_col])
            clst = list(adata_t.obs[cond_col])
            
            lstx = []
            for c,s in zip(clst, slst):
                rn = '%s %s' % (c,s) 
                lstx.append(rn)
    
            adata_t.obs[sample_col + '_rev'] = lstx  
        elif (sample_col in cols):
            adata_t.obs[sample_col + '_rev'] = adata_t.obs[sample_col].copy(deep = True)
    
        bm = adata_t.var_names.str.startswith('MT-')  
        tc = np.array(adata_t.X.sum(axis = 1))[:,0]
        mc = np.array(adata_t[:, bm].X.sum(axis = 1))[:,0]
        
        s = 'Filtering cells and genes .. ' 
        log_lines = update_log_and_print( s, flog, log_lines = log_lines, c_prefix = c_prefix )
        bp = (100*mc/tc) <= Pct_cnt_mt_max
        if np.sum(bp) != len(bp):
            adata_t = adata_t[bp, :]
            s = '   N cells: %i -> %i (%i cells dropped as pct_count_mt > %4.1f) ' % \
                (len(bp), np.sum(bp), len(bp)-np.sum(bp), Pct_cnt_mt_max)
            log_lines = update_log_and_print( s, flog, log_lines = log_lines, c_prefix = c_prefix )
        
        bc = (np.array((adata_t.X > 0).sum(axis = 1)) >= N_genes_min)[:,0]  
        if np.sum(bc) != len(bc):
            adata_t = adata_t[bc, :]
            s = '   N cells: %i -> %i (%i cells dropped as gene_count < %i) ' % \
                (len(bc), np.sum(bc), len(bc)-np.sum(bc), N_genes_min)
            log_lines = update_log_and_print( s, flog, log_lines = log_lines, c_prefix = c_prefix )
            
        bg = (np.array(adata_t.X.sum(axis = 0)) >= N_cells_min)[0,:]
        if np.sum(bg) != len(bg):
            adata_t = adata_t[:, bg]
            s = '   N genes: %i -> %i (%i genes dropped as cell_count < %i) ' % \
                (len(bg), np.sum(bg), len(bg)-np.sum(bg), N_cells_min)
            log_lines = update_log_and_print( s, flog, log_lines = log_lines, c_prefix = c_prefix )
    
        #################################
        ### Check the number of cells ###
    
        num_cells = adata_t.obs.shape[0]
        if num_cells > max_num_cells_allowed:
            s = 'WARNING: N cells = %i exceeds the max. number -> down-sampled to %i' % \
                (num_cells, max_num_cells_allowed)
            log_lines = update_log_and_print( s, flog, log_lines = log_lines, c_prefix = '' )
    
            cell_barcodes = list(adata_t.obs.index.values)
            cell_barcodes_sel = random.sample(cell_barcodes, max_num_cells_allowed)
            cell_barcodes_all = pd.Series(list(adata_t.obs.index.values), 
                                          index = adata_t.obs.index.values) 
            b = cell_barcodes_all.isin(cell_barcodes_sel)
            adata_t = adata_t[b,:]
    
        #######################################
        ### Tissue detection & load markers ###
    
        tissue_lst = ['Generic', 'Pancreas', 'Lung', 'Liver', 'Blood', 
                      'Bone', 'Brain', 'Brain_ext', 'Embryo', 'Eye', 'Intestine', 'Breast',
                      'Heart', 'Kidney', 'Stomach', 'Skeletal muscle', 'Skin']
        n_cells_to_use_for_tissue_detection = 4000 
        non_ref_p_th = 0.1
    
        if user_supplied_mkr:
            df_mkr_db = pd.read_csv(file_mkr, sep = '\t')
        else:
            if user_tissue_sel in tissue_lst:
                tissue = user_tissue_sel
            else:
                
                s = 'Selecting marker DB .. ' # % (print_prefix)
                log_lines = update_log_and_print( s, flog, log_lines = log_lines, c_prefix = c_prefix )
                
                ## detecte tissue 
                tissue_score_means = None
                # tissue_score_means = detect_tissue_get_stats( opt_file_path, non_ref_p_th, verbose = False )
            
                df = detect_tissue( adata_t, df_mkr_db_all, gtf_file,
                                tissue_lst, tissue_score_means,
                                n_cells_to_use = n_cells_to_use_for_tissue_detection, # non_ref_n_th = 100,
                                non_ref_p_th = non_ref_p_th, # score_th = 6, 
                                # taxo_level = 'cell_type_major',
                                # confidence = 'Confidence(1st)', 
                                # ident_level = 1, N_mkrs_max = 500, 
                                verbose = True, ref_celltypes_add = [] )
        
                if df['fnr'].max() < non_ref_p_th:
                    tissue = 'Generic'
                else:
                    tissue = (df['sb']*(df['fnr'] >= non_ref_p_th)).idxmax()
    
                s = 'Selecting marker DB .. done.'
                s = s + ' Best match: %s (Non-ref fraction: %4.3f/%4.3f) ' % \
                    (tissue, df.loc[tissue, 'fnr'], df['fnr'].max())
                log_lines = update_log_and_print( s, flog, log_lines = log_lines, c_prefix = c_prefix )

            if tissue == 'Generic':
                target_tissues = ['Immune', 'Generic', 'Epithelium']
            elif tissue == 'Blood':
                target_tissues = ['Immune', 'Generic', 'Immune_ext', tissue] # 
            elif tissue == 'Brain':
                target_tissues = ['Immune', 'Generic', tissue] # 
            elif tissue in tissue_lst:
                target_tissues = ['Immune', 'Generic', tissue] # , 'Immune_ext'
            else:
                target_tissues = ['Immune', 'Generic', 'Epithelium']
        
            b = df_mkr_db_all['tissue'].isin(target_tissues)
            df_mkr_db = df_mkr_db_all.loc[b,:].copy(deep = True)

            if tissue != 'Blood':
                if tissue in ['Brain']:
                    b = df_mkr_db['cell_type_major'].isin( ['Myeloid cell'] )
                    df_mkr_db = df_mkr_db.loc[~b,:]
                else:
                    b = df_mkr_db['cell_type_minor'].isin( ['Monocyte'] )
                    df_mkr_db = df_mkr_db.loc[~b,:]
                        
        ##################################
        if (sample_col + '_rev') in list(adata_t.obs.columns.values):
            sample_col_rev = sample_col + '_rev'
        else: 
            sample_col_rev = sample_col

        sample_group_map = get_sample_to_group_map(adata_t.obs[sample_col_rev], 
                                                   adata_t.obs[cond_col])

        adata_t.uns['lut_sample_to_cond'] = sample_group_map
    
        ##################################
        ###  Cell-type identification  ###
    
        lines = scoda_hicat(adata_t, df_mkr_db, print_prefix = print_prefix, verbose = 1, 
                            clustering_resolution = hicat_clustering_resolution, 
                            pct_th_cutoff = hicat_pct_th_cutoff, 
                            N_cells_max_for_pca = hicat_n_cells_max_for_pca, 
                            N_cells_max_for_gmm = hicat_n_cells_max_for_gmm,
                            mkr_selector = hicat_mkr_selector )

        if lines is not None:
            if lines[-1] == '\n': lines = lines[:-1]
        log_lines = update_log_and_print( lines, flog, log_lines = log_lines, c_prefix = None )

        ## Check unassigned
        b = adata_t.obs['celltype_major'] != 'unassigned'
        if np.sum(b) == 0:
            if user_supplied_mkr:
                s = 'WARNING: Celltypes not assigned at all with the user-specified markers db.'
                log_lines = update_log_and_print( s, flog, log_lines = log_lines, c_prefix = '' )
                
                s = 'WARNING: Rerunning HiCAT using the default markers db .. '
                log_lines = update_log_and_print( s, flog, log_lines = log_lines, c_prefix = '' )
    
                ## Get default markers db
                user_supplied_mkr = False
                
                if species.lower() == 'mouse':
                    file_mkr_org = '%s/markers_mm.tsv' % (opt_file_path)
                    file_mkr_new = '%s/markers_mm.tsv' % (data_dir)
                else:
                    file_mkr_org = '%s/markers_hs.tsv' % (opt_file_path)
                    file_mkr_new = '%s/markers_hs.tsv' % (data_dir)

                file_mkr = file_mkr_org                
                df_mkr_db = pd.read_csv(file_mkr, sep = '\t')
                
                ## Rerun HiCAT
                lines = scoda_hicat(adata_t, df_mkr_db, print_prefix = print_prefix, verbose = 1, 
                                    clustering_resolution = hicat_clustering_resolution, 
                                    pct_th_cutoff = hicat_pct_th_cutoff, 
                                    N_cells_max_for_pca = hicat_n_cells_max_for_pca, 
                                    N_cells_max_for_gmm = hicat_n_cells_max_for_gmm,
                                    mkr_selector = hicat_mkr_selector)
                if lines is not None:
                    if lines[-1] == '\n': lines = lines[:-1]
                log_lines = update_log_and_print( lines, flog, log_lines = log_lines, c_prefix = '' )
    
            else:
                s = 'WARNING: celltypes not assigned at all with the user-specified markers db.'
                log_lines = update_log_and_print( s, flog, log_lines = log_lines, c_prefix = '' )
                
                s = 'WARNING: Check the tips for preparing markers db.'
                log_lines = update_log_and_print( s, flog, log_lines = log_lines, c_prefix = '' )
    
        b = adata_t.obs['celltype_major'] != 'unassigned'
        N_cells_valid = np.sum(b)
        s = 'N cells valid/total = %i/%i ' % (N_cells_valid, len(b))
        log_lines = update_log_and_print( s, flog, log_lines = log_lines, c_prefix = c_prefix )
        
        adata_t.write(file_h5ad)

        #################################
        ### tumor cell identification ###
    
        verbose = True
        if tumor_identification:
            
            if species.lower() == 'mouse':
                gtf_file = '%s/mm10_gene_only.gtf' % (opt_file_path)
            else:
                gtf_file = '%s/hg38_gene_only.gtf' % (opt_file_path)

            b = adata_t.obs['celltype_major'].isin(tumor_id_ref_celltypes)
            if (np.sum(b) < min(1000, len(b)*0.1)):
                tumor_id_ref_celltypes = None
    
            # print(tumor_id_ref_condition)
            if tumor_id_ref_condition is not None:
                b = adata_t.obs['condition'].isin(tumor_id_ref_condition)
                if (np.sum(b) < min(1000, len(b)*0.1)):
                    tumor_id_ref_condition = None

            ## Test without Reference 
            ref_types = tumor_id_ref_celltypes
            ref_condition = tumor_id_ref_condition

            '''
            ## clustering_algo = GMM is not suitable for this work
            lines = pp.scoda_icnv_addon_split_run( adata_t, gtf_file, 
                           ref_condition = None, # ['Adj_normal'], 
                           ref_types = ref_types, 
                           ref_key = "celltype_major", 
                           clustering_algo = 'lv', # CLUSTERING_AGO,  
                           clustering_resolution = 6, #cnv_clustering_resolution, 
                           N_tid_runs = 7,
                           N_tid_loops = 5, # cnv_clustering_loop,
                           N_cells_max_for_clustering = 60000, # cnv_clustering_n_cells,
                           n_neighbors = 14,
                           n_pca_comp = 15, 
                           N_cells_max_for_pca = 60000, # cnv_clustering_n_cells,

                           connectivity_threshold = 0.2, # cnv_connectivity_threshold, 
                           ref_pct_min = 0.25,
                           tumor_dec_margin = 0.2, # cnv_tumor_dec_margin, 
                                      
                           net_search_mode = 'sum', 
                           use_cnv_score = False, # True, 
                           print_prefix = '', 
                           n_cores = 4, 
                           verbose = True,
                           group_cell_size = 6000, 
                           cs_ref_quantile = 0,
                           cs_comp_method = 0,
                           cnv_filter_quantile = 0,
                           logreg_correction = False,
                           split_run = False,
                           connectivity_std_scale_factor = 1.5, spf = 0.1,
                           plot_connection_profile = True )
            '''
            ## clustering_algo = GMM is not suitable for this work
            lines = scoda_icnv_addon_split_run( adata_t, gtf_file, 
                               ref_condition = ref_condition, # ['Adj_normal'], 
                               ref_types = ref_types, 
                               ref_key = "celltype_major", 
                               clustering_algo = 'lv', # CLUSTERING_AGO,  
                               clustering_resolution = cnv_clustering_resolution, #cnv_clustering_resolution, 
                               N_tid_runs = 7,
                               N_tid_loops = 5, # cnv_clustering_loop,
                               N_cells_max_for_clustering = cnv_clustering_n_cells, # cnv_clustering_n_cells,
                               n_neighbors = 14,
                               n_pca_comp = 15, 
                               N_cells_max_for_pca = cnv_clustering_n_cells, # cnv_clustering_n_cells,
    
                               connectivity_threshold_min = cnv_connectivity_threshold, # cnv_connectivity_threshold, 
                               connectivity_threshold_max = 0.28, # cnv_connectivity_threshold, 
                               ref_pct_min = cnv_ref_pct_min,
                               tumor_dec_margin = cnv_uc_margin, # cnv_tumor_dec_margin, 
                               gmm_ncomp_n = cnv_gmm_num_components, 
                               gmm_ncomp_t = cnv_gmm_num_components, 
                                          
                               net_search_mode = 'sum', 
                               use_cnv_score = False, # True, 
                               print_prefix = print_prefix, 
                               n_cores = n_cores_to_use, 
                               verbose = verbose,
                               group_cell_size = 6000, 
                               cs_ref_quantile = 0,
                               cs_comp_method = 0,
                               cnv_filter_quantile = 0,
                               logreg_correction = False,
                               split_run = False,
                               connectivity_std_scale_factor = cnv_connectivity_std_sf, 
                               spf = 0.1,
                               plot_connection_profile = False )

            
            if lines is not None:
                if lines[-1] == '\n': lines = lines[:-1]
            log_lines = update_log_and_print( lines, flog, log_lines = log_lines, c_prefix = None )
            
            adata_t.write(file_h5ad)
    
        
            ################################################
            ### Set obs columns for tumor cell CCI & DEG ###
            
            normal_cells = tumor_id_ref_condition # ['Normal', 'Preneoplastic']
            adj_normal_name = 'Adj_normal'
            tumor_ind = 'tumor_origin_ind'
            cond_specific_adj_normal = False
            
            tumor_origin, adata_tmr_only = icnv_set_tumor_info( adata_t, tid_col = 'tumor_dec', 
                                        celltype_col = cci_deg_base,
                                        sample_col = sample_col, 
                                        cond_col = cond_col, 
                                        ref_taxo_level = 'celltype_major',
                                        normal_cells = normal_cells,
                                        adj_normal_name = adj_normal_name, 
                                        tumor_ind_col = tumor_ind,
                                        cond_specific_adj_normal = cond_specific_adj_normal )
    
            cci_deg_base_for_cci = 'celltype_for_cci'
            cci_deg_base_for_deg = 'celltype_for_deg' 
            sample_col_for_deg = sample_col + '_for_deg' 
            cond_col_for_deg = cond_col + '_for_deg' 
        else:
            cci_deg_base_for_cci = cci_deg_base
        
        #################################
        ###   Cell-cell interaction   ###
    
        cpdb_path = file_cpdb        
            
        lines = scoda_cci( adata_t, cpdb_path, 
                   cond_col = cond_col, 
                   sample_col = sample_col_rev, 
                   cci_base = cci_deg_base_for_cci, 
                   unit_of_cci_run = None, # unit_of_cci_run, 
                   n_cores = n_cores_to_use, 
                   min_n_cells_for_cci = min_n_cells_for_cci, 
                   Rth = cci_min_OF_to_count,
                   pval_max = cci_pval_max, 
                   mean_min = cci_mean_min, 
                   data_dir = data_dir,
                   print_prefix = print_prefix, 
                   cpdb_version = 4,
                   verbose = verbose )   
        
        if lines is not None:
            if lines[-1] == '\n': lines = lines[:-1]
        log_lines = update_log_and_print( lines, flog, log_lines = log_lines, c_prefix = None )
        
        adata_t.write(file_h5ad)
        
        #################################
        #####     DEG analysis      #####
    
        lines = scoda_deg_gsea( adata_t, pw_db = file_gmt, 
                        cond_col = cond_col, sample_col = sample_col, 
                        deg_base = cci_deg_base, 
                        ref_group = deg_ref_group, 
                        deg_pval_cutoff = deg_pval_cutoff, 
                        gsea_pval_cutoff = gsea_pval_cutoff,
                        N_cells_min_per_sample = min_n_cells_for_deg, 
                        N_cells_min_per_condition = min_n_cells_for_deg, 
                        n_cores = n_cores_to_use, uns_key_suffix = '',
                        deg_pairwise = True,
                        deg_cmp_mode = 'max',
                        print_prefix = print_prefix, 
                        verbose = verbose)  
        
        ########################################
        ##### DEG analysis for Tumor cells #####

        if tumor_identification & (adata_tmr_only.obs.shape[0] > 100):
            uns_key_suffix = '_tumor'
            lines = scoda_deg_gsea( adata_tmr_only, pw_db = file_gmt, 
                            cond_col = cond_col_for_deg, 
                            sample_col = sample_col_for_deg, 
                            deg_base = cci_deg_base_for_deg, 
                            ref_group = None, ###
                            deg_pval_cutoff = deg_pval_cutoff, 
                            gsea_pval_cutoff = gsea_pval_cutoff,
                            N_cells_min_per_sample = min_n_cells_for_deg, 
                            N_cells_min_per_condition = min_n_cells_for_deg, 
                            n_cores = n_cores_to_use, uns_key_suffix = uns_key_suffix,
                            deg_pairwise = True,
                            deg_cmp_mode = 'max',
                            print_prefix = print_prefix, 
                            verbose = verbose)  

            ## key added: 'DEG_stat_tumor', 'DEG_tumor', 'GSA_up_tumor', 'GSA_down_tumor', 'GSEA_tumor'
            item_lst = ['DEG_stat', 'DEG', 'GSA_up', 'GSA_down', 'GSEA']
            for item in item_lst:
                item_t = item + uns_key_suffix
                if item_t in list(adata_tmr_only.uns.keys()):
                    df_dct = adata_tmr_only.uns[item_t]
                    if item in list(adata_t.uns.keys()):
                        for celltype in df_dct.keys():
                            celltype_t = 'Tumoric ' + celltype
                            adata_t.uns[item][celltype_t] = df_dct[celltype]
                    else:
                        df_dct_new = {}
                        for celltype in df_dct.keys():
                            celltype_t = 'Tumoric ' + celltype
                            df_dct_new[celltype_t] = df_dct[celltype]
                        adata_t.uns[item] = df_dct_new


        #######################################
        ##### Find tumor specific markers #####
        '''
        item = list(adata_tmr_only.uns['DEG' + uns_key_suffix].keys())[0]
        df_deg_dct = adata_tmr_only.uns['DEG' + uns_key_suffix][item] 
        
        n_markers_max = 80
        score_th = 0.25
        taxo_level_for_hicat_mkr = 'major'
        
        mkr_dict, mkrs_all, df_deg_dct_updated = find_condition_specific_markers( df_deg_dct, 
                                                              col_score = 'score',
                                                              n_markers_max = n_markers_max,
                                                              score_th = score_th,
                                                              pval_cutoff = deg_pval_cutoff,
                                                              nz_pct_test_min = 0,
                                                              nz_pct_ref_max = 1,
                                                              n = 1, verbose = verbose )
        
        df_mkr_db = generate_hicat_markers_db( mkr_dict, 
                                               celltype_origin = tumor_origin,
                                               taxo_level = 'major', 
                                               species = 'hs',
                                               tissue = 'Tumor', 
                                               normal_name = 'Normal' )
        '''
        ##### DEG analysis for Tumor cells #####
        ########################################
        
        if lines is not None:
            if lines[-1] == '\n': lines = lines[:-1]
        log_lines = update_log_and_print( lines, flog, log_lines = log_lines, c_prefix = None )
        
        ## generate key #######        
        key2, int_lst2 = get_dataset_key(adata_t, L = kpL, M = kpM)
        keyt = '%i/%i/%s/%s' % (kpL, kpM, key1, key2)
    
        N_cells = adata_t.obs.shape[0]
        N_cells_valid = np.sum(adata_t.obs['celltype_major'] != 'unassinged')
        
        save_info( data_dir + '/%s' % file_hist_expected, 
                   info_dict, session_id, N_cells_org, N_cells, unit_price, keyt )        
    
        lapsed = time.time() - start_time
        result_file = '%s_results.tar.gz' % (session_id)
    
        s1 = 'Result saved to %s' % (result_file)
        log_lines = update_log_and_print( s1, flog, log_lines = log_lines, c_prefix = None )
        
        s2 = 'SUCCESS: %s data successful. (%4.1f mins, %i -> %i/%i cells total)' % \
             (s_prefix, round(lapsed/60, 1), N_cells_org, N_cells, N_cells_valid)
        log_lines = update_log_and_print( s2, flog, log_lines = log_lines, c_prefix = None )
        
        adata_t.uns['log'] = log_lines
        adata_t.uns['usr_param'] = user_param
        
        ## Overwrite 
        adata_t.write(file_h5ad)      
        ###                                                           ###
        #################################################################

        if compress_out:
            pn, fn, ext = get_path_filename_and_ext_of(file_h5ad)
            result_dname = fn            
            res_dir = data_dir + '/%s' % result_dname # '/%s' % session_id
            if not os.path.isdir(res_dir):
                try:
                    os.mkdir(res_dir)
                except ValueError:
                    pass
            
            # cmd = 'mv %s %s' % (file_h5ad, res_dir)
            # run_command(cmd)
            shutil.move(file_h5ad, res_dir)
            
            cur_dir = os.getcwd()
            os.chdir(data_dir)
            
            cmd = 'tar -zcvf %s %s' % (result_file, result_dname) #, session_id)
            run_command(cmd)
            
            os.chdir(cur_dir)
            remove_dir( res_dir )    
        
        ## Remove temporary files & folders
        pycache_dir = data_dir + '/__pycache__'
        remove_dir( pycache_dir )    
        remove_file(file_cfg)
        if user_supplied_mkr: remove_file(file_mkr)
        if user_supplied_pwdb: remove_file( file_gmt )
        if user_supplied_cpdb: remove_file( file_cpdb )
        
        cpdb_dir = data_dir + '/cpdb'
        remove_dir( cpdb_dir ) 
    
        # print('%s data .. completed.' % s_prefix)
        sa = '%s%s' % (c_prefix, s1)
        print(sa, flush = True)
        print(s2)
        return file_h5ad


def scoda_run_all( data_dir, data_type, species, session_id, 
                   tumor_id = 'yes', 
                   home_dir = '/home/comp_serv/sc_comp_serv',
                   c_prefix = 'CLIENT INFO: ',
                   lim_Ncells = True ):

    check_data( data_dir, data_type, species, 
                session_id, tumor_id, home_dir, c_prefix)

    process_data( data_dir, data_type, species, 
                  session_id, tumor_id, home_dir, c_prefix, lim_Ncells)

    return


def scoda_check_data( file_in, data_type = '10x_mtx', 
                      species = 'human', tissue = 'Not specified',
                      tumor_id = 'yes' ):

    session_id = '0'
    home_dir = None
    c_prefix = ''

    pn, fn, ext = get_path_filename_and_ext_of(file_in)

    data_dir = 'scoda_check_data_tmp' 
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    shutil.move( file_in, data_dir )
    
    file_h5ad = check_data( data_dir, data_type, species, 
                    session_id, tumor_id, home_dir, c_prefix)

    if pn == '': pn = './'
    shutil.move( '%s/%s.%s' % ( data_dir, fn, ext ), pn )

    flst = os.listdir(data_dir)
    file_h5ad = None
    for f in flst:
        if f.split('.')[-1] == 'h5ad':
            file_h5ad = f

    if os.path.isfile(file_h5ad):
        os.remove(file_h5ad)
    shutil.move( '%s/%s' % ( data_dir, file_h5ad ), './' )    
    shutil.rmtree(data_dir)

    return file_h5ad



def scoda_process_data( file_h5ad_in, 
                        # data_dir = '.', 
                        species = 'human', 
                        tissue = None,
                        tumor_id = 'no', 
                        file_h5ad_suffix = '_scoda',
                        # data_type = 'h5ad', 
                        # home_dir = '',
                        # c_prefix = '',
                        # lim_Ncells = False, 
                        # compress_out = False, 
                        # session_id = None,
                        out_dir = '.',
                        file_cfg_in = None, 
                        file_mkr_in = None,
                        file_cpdb_in = None, 
                        file_gmt_in = None ):

    # print(os.getcwd())
    file_h5ad = process_data( data_dir = '.', 
                              data_type = 'h5ad', 
                              species = species, 
                              session_id = None, 
                              tumor_id = tumor_id, 
                              home_dir = '',
                              c_prefix = '',
                              lim_Ncells = False, 
                              compress_out = False,
                              tissue = tissue,
                              file_h5ad_in = file_h5ad_in, 
                              file_h5ad_suffix = file_h5ad_suffix, 
                              file_cfg_in = file_cfg_in, 
                              file_mkr_in = file_mkr_in,
                              file_cpdb_in = file_cpdb_in, 
                              file_gmt_in = file_gmt_in )

    # print('file_h5ad = %s' % file_h5ad)
    if file_h5ad is None:
        print('ERROR: error occured while processing data. ')
        return None
    else:
        pn, fn, ext = get_path_filename_and_ext_of(file_h5ad)
        # print('%s, %s, %s' %(pn, fn, ext))
        
        Move = False
        if len(file_h5ad_in.split('/')) == 1: 
            # Move = data_dir != out_dir
            file_log = '%s.%s' % (fn, 'log')
            file_info = '%s.%s' % (fn, 'info')
        else:
            pn2, fn2, ext2 = get_path_filename_and_ext_of(file_h5ad_in)
            Move = pn2 != out_dir   
            file_log = '%s/%s.%s' % (pn2, fn, 'log')
            file_info = '%s/%s.%s' % (pn2, fn, 'info')
    
        if Move:
            file_log_out = '%s/%s.%s' % (out_dir, fn, 'log')
            if os.path.isfile(file_log_out):
                os.remove(file_log_out)
            shutil.move(file_log, file_log_out)
    
            file_info_out = '%s/%s.%s' % (out_dir, fn, 'info')
            if os.path.isfile(file_info_out):
                os.remove(file_info_out)
            shutil.move(file_info, file_info_out)
            
            file_out = '%s/%s.%s' % (out_dir, fn, ext)
            if os.path.isfile(file_out):
                os.remove(file_out)
            shutil.move(file_h5ad, file_out)
        
            return file_out
        else:
            return file_h5ad
            

def scoda_check_and_process( file_in, 
                             data_type, 
                             species = 'human', 
                             tissue = None,
                             tumor_id = 'yes', 
                             file_h5ad_suffix = '_scoda',
                             out_dir = '.',
                             file_cfg_in = None, 
                             file_mkr_in = None,
                             file_cpdb_in = None, 
                             file_gmt_in = None ):

    file_h5ad_in = scoda_check_data( file_in, data_type = data_type,
                                     species = species, tissue = tissue,
                                     tumor_id = tumor_id )

    file_h5ad = scoda_process_data( file_h5ad_in, 
                        species = species, 
                        tissue = tissue,
                        tumor_id = tumor_id, 
                        file_h5ad_suffix = file_h5ad_suffix,
                        out_dir = out_dir,
                        file_cfg_in = file_cfg_in, 
                        file_mkr_in = file_mkr_in,
                        file_cpdb_in = file_cpdb_in, 
                        file_gmt_in = file_gmt_in )

    return file_h5ad


def get_args():

    parser = argparse.ArgumentParser(description='', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser._action_groups.pop()
    required = parser.add_argument_group('REQUIRED PARAMETERS')
    optional = parser.add_argument_group('OPTIONAL PARAMETERS')

    required.add_argument('-data_dir', type=str, metavar='',
                          help='Full Path to the directory where the uploaded file(s) stored.',
                          default=None)
    required.add_argument('-data_type', type=str, metavar='',
                          help='Data type. 10x_mtx or h5ad or csv_mtx',
                          default='10x_mtx')
    required.add_argument('-species', type=str,  metavar='',
                          help='Species. human or mouse', default="human")

    optional.add_argument('-session_id', type=str, metavar='',
                          help='Unique session ID.',
                          default=None)

    optional.add_argument('-sel', type=str, metavar='',
                          help='Procedure selection. 1 for check_data, 2 for process_data, 3 for run_all.',
                          default=3)
    optional.add_argument('-home_dir', type=str,  metavar='',
                          help='Full Path to the SCODA server home directory.', 
                          default="/mnt/HDD2/sc_comp_serv")
    optional.add_argument('-tumor_id', type=str, metavar='',
                          help='Tumor identification. Yes or No',
                          default='yes')
    optional.add_argument('-lim_nc', type=str,  metavar='',
                          help='Limit the number of cells', default="True")
    optional.add_argument('-c_prefix', type=str,  metavar='',
                          help='Client info prefix', default="CLIENT INFO: ")
    # optional.add_argument('-j', type=str,  metavar='',
    #                     help='Jump to step j [0 or 1 or 2]', default="0")
    args = parser.parse_args()
    return args, parser


from datetime import datetime

def main():

    print('+-----------------------------+')
    print('|   MLBI@DKU SCODA pipeline   |')
    print('+-----------------------------+')

    args, parser = get_args()

    lim_Ncells = True
    if (args.lim_nc.upper()[0] == 'F'):
        lim_Ncells = False

    session_id = args.session_id
    if session_id is None:
        # datetime object containing current date and time
        now = datetime.now()
        # dd/mm/YY H:M:S
        dt_string = now.strftime("%Y.%m.%d.%H.%M.%S")
        session_id = dt_string

    print('Current path: ', os.getcwd())

    if (session_id is None) | (args.data_dir is None) | (args.data_type is None): 
        print('ERROR: required arguments not specified in sc_processing.sh.')
        parser.print_help()
        return
    else:
        if (args.sel == 2) | (args.sel == '2'):
            process_data(args.data_dir, args.data_type, args.species, 
                         session_id, args.tumor_id, args.home_dir, args.c_prefix, lim_Ncells)
        elif  (args.sel == 1) | (args.sel == '1'):
            check_data(args.data_dir, args.data_type, args.species, 
                       session_id, args.tumor_id, args.home_dir, args.c_prefix)
        else:
            scoda_run_all(args.data_dir, args.data_type, args.species, 
                          session_id, args.tumor_id, args.home_dir, args.c_prefix, lim_Ncells)


if __name__=="__main__":
    main()

