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
from scipy.sparse import csr_matrix
import anndata
# import infercnvpy as cnv

from scoda.icnv import run_icnv, identify_tumor_cells
from scoda.cpdb import cpdb4_run, cpdb4_get_results, cpdb_plot, cpdb_get_gp_n_cp #, plot_circ
from scoda.gsea import select_samples, run_enrich, run_enrichr, run_prerank
from scoda.deg import deg_multi, get_population, plot_population
from scoda.misc import plot_sankey_e, get_opt_files_path
from scoda.hicat import HiCAT
from scoda.pipeline import detect_tissue, detect_tissue_get_stats
from scoda.viz import get_sample_to_group_map

from scoda.key_gen import get_dataset_key, recover_int_lst

# from pipeline_gsea import scoda_deg_gsea
from scoda.pipeline import scoda_hicat, scoda_icnv_addon, scoda_cci, scoda_deg_gsea
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
        
        if os.path.isdir(ddir):
            # cmd = 'rm -r %s' % ddir
            # run_command(cmd)   
            shutil.rmtree(ddir)

    # print('   Data check passed.')
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
                        if fn != 'features.tsv':
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


def check_data( data_dir, data_type, species, session_id, 
                tumor_id = 'no', 
                home_dir = '/mnt/HDD2/sc_comp_serv',
                c_prefix = 'CLIENT INFO: '  ):

    s_prefix = 'Checking'

    log_lines = ''
    log_file = data_dir + '/%s.log' % session_id
    with open(log_file, 'w+') as flog:
    
        s = 'session ID = %s ' % (session_id)
        flog.writelines(s + '\n')
        log_lines = log_lines + '%s\n' % s
        sa = '%s%s' % (c_prefix, s)
        print(sa, flush = True)
    
        s= '%s data .. Type(%s), Species(%s), TumorId(%s) ' % (s_prefix, data_type, species, tumor_id) 
        flog.writelines(s + '\n')
        log_lines = log_lines + '%s\n' % s
        sa = '%s%s' % (c_prefix, s)
        print(sa, flush = True)
    
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
        flog.writelines(s + '\n')
        log_lines = log_lines + '%s\n' % s
        sa = '%s%s' % (c_prefix, s)
        print(sa, flush = True)
        
        if len(lst) == 0:
            s = 'WARNING: No files uploaded.'
            flog.writelines(s + '\n')
            log_lines = log_lines + '%s\n' % s
            sa = '%s' % (s)
            print(sa, flush = True)
        
            s = 'ERROR: data check failed.'
            flog.writelines(s + '\n')
            log_lines = log_lines + '%s\n' % s
            sa = '%s' % (s)
            print(sa, flush = True)
        
            return None
        else:
            ## Check if uploaded files are compressed or not
            cnt = 0
            for f in lst:
                ext = f.split('.')[-1]
                if (ext == 'zip') | (ext == 'gz'): cnt += 1
            if cnt == 0:
                s = 'WARNING: Uploaded files are not zip or tar.gz. '
                flog.writelines(s + '\n')
                log_lines = log_lines + '%s\n' % s
                sa = '%s' % (s)
                print(sa, flush = True)
                
                s = 'WARNING: Please check the instruction to prepare datasets to upload.'
                flog.writelines(s + '\n')
                log_lines = log_lines + '%s\n' % s
                sa = '%s' % (s)
                print(sa, flush = True)
                
                s = 'ERROR: data check failed.'
                flog.writelines(s + '\n')
                log_lines = log_lines + '%s\n' % s
                sa = '%s' % (s)
                print(sa, flush = True)
                
                return None
    
        ecode, rcode, ddir, fn = decompress(data_dir)
        # print(ecode, rcode, ddir, fn)
        
        file_h5ad = None
        ## Check if any error while decompress the uploaded file(s)
        if (ecode != 0) | (fn is None):
            s = 'WARNING: No valid files found.'
            flog.writelines(s + '\n')
            log_lines = log_lines + '%s\n' % s
            sa = '%s' % (s)
            print(sa, flush = True)
        
            s = 'ERROR: data check failed.'
            flog.writelines(s + '\n')
            log_lines = log_lines + '%s\n' % s
            sa = '%s' % (s)
            print(sa, flush = True)
        
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
            flog.writelines(s + '\n')
            log_lines = log_lines + '%s\n' % s
            sa = '%s' % (s)
            print(sa, flush = True)
        
            s = 'WARNING: Please check the instruction to prepare datasets to upload.'
            flog.writelines(s + '\n')
            log_lines = log_lines + '%s\n' % s
            sa = '%s' % (s)
            print(sa, flush = True)
        
            s = 'ERROR: data check failed.'
            flog.writelines(s + '\n')
            log_lines = log_lines + '%s\n' % s
            sa = '%s' % (s)
            print(sa, flush = True)   
        else: 
            '''
            adata = anndata.read_h5ad(file_h5ad)
            
            if 'condition' in list(adata.obs.columns.values):
                adata.obs['condition'] = adata.obs['condition'].astype(str) 
            else:
                adata.obs['condition'] = 'Not_specified'
    
            if 'sample' in list(adata.obs.columns.values):
                adata.obs['sample'] = adata.obs['sample'].astype(str)  
            else:
                adata.obs['sample'] = 'Not_specified'
    
            adata.write(file_h5ad)   
            '''
            s = 'SUCCESS: data check successful.'
            flog.writelines(s + '\n')
            log_lines = log_lines + '%s\n' % s
            sa = '%s' % (s)
            print(sa, flush = True)
    
            try:
                adata = anndata.read_h5ad(file_h5ad)
            except:
                s = 'WARNING: Cannot open the generated file %s.' % file_h5ad
                flog.writelines(s + '\n')
                log_lines = log_lines + '%s\n' % s
                sa = '%s' % (s)
                print(sa, flush = True)
    
                s = 'ERROR: data processing failed.'
                flog.writelines(s + '\n')
                log_lines = log_lines + '%s\n' % s
                sa = '%s' % (s)
                print(sa, flush = True)
            
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
                  file_h5ad_in = None, file_h5ad_suffix = '_scoda' ):

    #################################################################
    ###                                                           ###
    
    file_h5ad = file_h5ad_in
    if file_h5ad is not None:
        items = file_h5ad.split('.')
        f2 = ''
        if items[-1] == 'h5ad':
            for kx, item in enumerate(items[:-1]):
                f2 = f2 + '%s.' % item
                if kx == (len(items)-1):
                    f2 = f2[:-1] + '%s' % file_h5ad_suffix
            shutil.copyfile( file_h5ad, f2 + '.h5ad' )
        elif data_type == 'h5ad':
            for kx, item in enumerate(items):
                f2 = f2 + '%s.' % item
                if kx == (len(items)-1):
                    f2 = f2[:-1] + '%s' % file_h5ad_suffix    
            shutil.copyfile( file_h5ad + '.h5ad', f2 + '.h5ad' )
            
        file_h5ad = f2 + '.h5ad'
        session_id = f2

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
        # print('Generated file(s): ', flst)
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
        
        # print('%ssession ID = %s ' % (c_prefix, session_id)) #, flush = True)
        s = '%s data .. ' % (s_prefix)
        flog.writelines(s + '\n')
        log_lines = log_lines + '%s\n' % s
        sa = '%s%s' % (c_prefix, s)
        print(sa, flush = True)
        
        # print('Data Dir: %s' % data_dir, flush = True)
        # print('   Data Type: %s' % data_type, flush = True)
        # print('   Species: %s' % species, flush = True)
        # print('Session ID: %s' % session_id, flush = True)
        # print('   Tumor ident.: %s' % tumor_id, flush = True)
        # print('Tumor ident.: %s' % home_dir, flush = True)
            
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
        # print('Default files folder: %s' % default_file_folder)
    
        ########################
        ## Service config params
    
        max_num_cells_allowed = 10000000
        n_cores_to_use = 4
        unit_price = 20
    
        serv_config_file = home_dir + '/service_config.txt'

        if os.path.isdir(serv_config_file):
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
                if (info_dict['Key'] is not None) & (info_dict['N_cells_in'] is not None) & (info_dict['Session_ID'] is not None):
                    if info_dict['Session_ID'] == session_id:
                        s = 'Returning order (%s == %s) ' % (info_dict['Session_ID'], session_id)
                        flog.writelines(s + '\n')
                        sa = '%s%s' % (c_prefix, s)
                    else:
                        s = 'WARNING: It seems returning order. But %s != %s ' % (info_dict['Session_ID'], session_id)
                        flog.writelines(s + '\n')
                        sa = s
                else:
                    s = 'New order '
                    flog.writelines(s + '\n')
                    sa = '%s%s' % (c_prefix, s)
                    
                # log_lines = log_lines + '%s\n' % s
                print(sa)
                    
                # print('KEYs: %i, %i\n%s\n%s' % (kpLo, kpMo, key1o, key2o))
            else:
                if file_h5ad is None:
                    items = f.split('.')
                    if items[-1] == 'h5ad':
                        file_h5ad = data_dir + '/%s' % f
    
        if file_hist is None:
            s = 'WARNING: Order information not provided. Will process assuming new one.'
            flog.writelines(s + '\n')
            print(s)
         
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
            flog.writelines(s + '\n')
            sa = '%s' % (s)
            print(sa, flush = True)
            s = 'ERROR: data processing failed.'
            flog.writelines(s + '\n')
            sa = '%s' % (s)
            print(sa, flush = True)
            return None
    
        glst2 = list(adata.var.index.values)
        glstc = list(set(glst).intersection(glst2))
        if len(glstc) <= len(glst2)*0.25:
            s = 'WARNING: Only %i out of %i genes were overlapped with %s genes. ' % (len(glstc), len(glst2), species)
            flog.writelines(s + '\n')
            sa = '%s' % (s)
            print(sa, flush = True)
            
            s = 'WARNING: Maybe you are using Gene ID, not Gene symbol, or provided wrong species.'
            flog.writelines(s + '\n')
            sa = '%s' % (s)
            print(sa, flush = True)
            
            s = 'ERROR: data processing failed.'
            flog.writelines(s + '\n')
            sa = '%s' % (s)
            print(sa, flush = True)
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
                flog.writelines(s + '\n')
                sa = '%s' % (s)
                print(sa, flush = True)
            
                s = 'WARNING: Please use the data you uploaded before or the SCODA result file you downloaded.'
                flog.writelines(s + '\n')
                sa = '%s' % (s)
                print(sa, flush = True)
            
                s = 'ERROR: data processing failed.'
                flog.writelines(s + '\n')
                sa = '%s' % (s)
                print(sa, flush = True)
            
                return None
            else:
                s = '   CheckSum passed.'
                flog.writelines(s + '\n')
                sa = '%s' % (s)
                print(sa, flush = True)
            
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
            # print('%s   Celltype marker DB not provided. Default DB will be used' % (c_prefix))
            
            if species.lower() == 'mouse':
                file_mkr_org = '%s/markers_mm.tsv' % (opt_file_path)
                file_mkr_new = '%s/markers_mm.tsv' % (data_dir)
            else:
                file_mkr_org = '%s/markers_hs.tsv' % (opt_file_path)
                file_mkr_new = '%s/markers_hs.tsv' % (data_dir)
                
            # cmd = 'cp %s %s' % (file_mkr_org, data_dir)
            # run_command(cmd)
            shutil.copyfile( file_mkr_org, file_mkr_new )
            
            file_mkr = '%s/markers.tsv' % (data_dir)
            # cmd = 'mv %s %s' % (file_mkr_new, file_mkr)
            # run_command(cmd)
            shutil.move(file_mkr_new, file_mkr)
            
            if not os.path.isfile(file_mkr):
                pycache_dir = data_dir + '/__pycache__'
                remove_dir( pycache_dir )    
                remove_file( file_cfg )
                remove_file( file_mkr )
                remove_file( file_gmt )
                remove_file( file_cpdb )
                remove_file( file_h5ad )
                print('ERROR: markers.tsv not found.')
                return
    
        if file_gmt is not None:
            ss2 = ss2 + 'O, '
            user_supplied_pwdb = True
        else:
            ss2 = ss2 + 'X, '
            # print('%s   Pathway DB not provided. Default DB will be used' % (c_prefix))
            # file_gmt_org = '%s/msig.gmt' % (opt_file_path)            
            if species.lower() == 'mouse':
                file_gmt_org = 'wiki_mouse.gmt'
            else:
                file_gmt_org = 'wiki_human.gmt'
                
            # cmd = 'cp %s/%s %s' % (opt_file_path, file_gmt_org, data_dir)
            # run_command(cmd)
            shutil.copyfile( '%s/%s' % (opt_file_path, file_gmt_org), '%s/%s' % (data_dir, file_gmt_org) )
            
            file_gmt = '%s/msig.gmt' % (data_dir)
            # cmd = 'mv %s/%s %s' % (data_dir, file_gmt_org, file_gmt)
            # run_command(cmd)
            shutil.move( '%s/%s' % (data_dir, file_gmt_org), file_gmt )
            
            if not os.path.isfile(file_gmt):
                pycache_dir = data_dir + '/__pycache__'
                remove_dir( pycache_dir )    
                remove_file( file_cfg )
                remove_file( file_mkr )
                remove_file( file_gmt )
                remove_file( file_cpdb )
                remove_file( file_h5ad )
                print('ERROR: msig.gmt not found.')
                return
    
        if file_cpdb is not None:
            ss2 = ss2 + 'O, '
            user_supplied_cpdb = True
        else:
            ss2 = ss2 + 'X, '
            # print('%s   CCI DB not provided. Default DB will be used' % (c_prefix))
            file_cpdb_org = '%s/cpdb.zip' % (opt_file_path)            
            file_cpdb = '%s/cpdb.zip' % (data_dir)
            
            # cmd = 'cp %s %s' % (file_cpdb_org, data_dir)
            # run_command(cmd)
            shutil.copyfile( file_cpdb_org, file_cpdb )
            
            if not os.path.isfile(file_cpdb):
                pycache_dir = data_dir + '/__pycache__'
                remove_dir( pycache_dir )    
                remove_file( file_cfg )
                remove_file( file_mkr )
                remove_file( file_gmt )
                remove_file( file_cpdb )
                remove_file( file_h5ad )
                print('ERROR: cci.db not found.')
                return
    
        if file_cfg is not None:
            ss2 = ss2 + 'O'
            user_supplied_cfg
        else:
            ss2 = ss2 + 'X'
            # print('%s   Client config. not provided. Default config. will be used' % (c_prefix), flush = True)
            file_cfg_org = '%s/analysis_config.py' % (opt_file_path)            
            file_cfg = '%s/analysis_config.py' % (data_dir)
            
            # cmd = 'cp %s %s' % (file_cfg_org, data_dir)
            # run_command(cmd)
            shutil.copyfile( file_cfg_org, file_cfg )
            
            if not os.path.isfile(file_cfg):
                pycache_dir = data_dir + '/__pycache__'
                remove_dir( pycache_dir )    
                remove_file( file_cfg )
                remove_file( file_mkr )
                remove_file( file_gmt )
                remove_file( file_cpdb )
                remove_file( file_h5ad )
                print('ERROR: analysis config file not found.')
                return
    
        
        s = '   User supplied: %s = %s ' % (ss1, ss2)
        flog.writelines(s + '\n')
        log_lines = log_lines + '%s\n' % s
        sa = '%s%s' % (c_prefix, s)
        print(sa)
        '''
        import sys
        sys.path.append(data_dir)
        # anal_cfg = importlib.import_module(data_dir + '.analysis_config') 
    
        import analysis_config as anal_cfg
        param = anal_cfg.get_anal_param()
        for k in param.keys():
            print('%30s: ' % k, param[k])
        '''
        
        ## Main process
        if file_h5ad is None:
            pycache_dir = data_dir + '/__pycache__'
            remove_dir( pycache_dir )    
            remove_file( file_cfg )
            remove_file( file_mkr )
            remove_file( file_gmt )
            remove_file( file_cpdb )
            remove_file( file_h5ad )
            print('ERROR: h5ad file not found.' )
            return None
        '''
        try:
            adata = anndata.read_h5ad(file_h5ad)
        except:
            pycache_dir = data_dir + '/__pycache__'
            remove_dir( pycache_dir )    
            remove_file( file_cfg )
            remove_file( file_mkr )
            remove_file( file_gmt )
            remove_file( file_cpdb )
            remove_file( file_h5ad )
            print('ERROR: Cannot open the generated h5ad file.')
            return None
        '''
        
        #################################################################
        ###                                                           ###
        
        ## Fixed param
        sample_col = 'sample'
        cond_col = 'condition'
    
        ###  common params ###
        # n_cores_to_use = 4
        verbose = True
    
        anal_param = {}
        ###  params for iCNV  ###
        # tumor_id_ref_celltypes = None
        tumor_id_ref_celltypes = ['T cell', 'B cell', 'Myeloid cell', 'Fibroblast', 'Stromal cell']
    
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
        import sys
        sys.path.append(data_dir)
    
        if not os.path.isfile(data_dir + '/analysis_config.py'):
            pycache_dir = data_dir + '/__pycache__'
            remove_dir( pycache_dir )    
            remove_file( file_cfg )
            remove_file( file_mkr )
            remove_file( file_gmt )
            remove_file( file_cpdb )
            remove_file( file_h5ad )
            print('ERROR: Cannot find analysis_config.py ...')
            return
        
        # print('Loading anal_config .. ')
        import analysis_config as anal_cfg
    
        hicat_clustering_resolution = 1
        hicat_pct_th_cutoff = 0.5
        hicat_n_cells_max_for_pca = 60000
        hicat_n_cells_max_for_gmm = 10000
        user_tissue_sel = tissue
        
        # print('Parsing .. ')
        ##########################
        ###  params for HiCAT  ###
        if user_tissue_sel is None:
            var_name = 'TISSUE'
            if hasattr(anal_cfg, var_name):
                user_tissue_sel = anal_cfg.TISSUE
            else:
                user_tissue_sel = 'Generic'
            anal_param[var_name] = user_tissue_sel
        else:
            s = 'User specified tissue: %s' % user_tissue_sel
            flog.writelines(s + '\n')
            log_lines = log_lines + '%s\n' % s
            sa = '%s%s' % (c_prefix, s)
            print(sa)
    
        ##########################
        ###  params for HiCAT  ###
        var_name = 'HICAT_CLUSTERING_RESOLUTION'
        if hasattr(anal_cfg, var_name):
            hicat_clustering_resolution = anal_cfg.HICAT_CLUSTERING_RESOLUTION
        else:
            hicat_clustering_resolution = 1
        anal_param[var_name] = hicat_clustering_resolution
            
        var_name = 'HICAT_PCT_TH_CUTOFF'
        if hasattr(anal_cfg, var_name):
            hicat_pct_th_cutoff = anal_cfg.HICAT_PCT_TH_CUTOFF
        else:
            hicat_pct_th_cutoff = 0.5
        anal_param[var_name] = hicat_pct_th_cutoff
            
        var_name = 'HICAT_N_CELL_MAX_FOR_PCA'
        if hasattr(anal_cfg, var_name):
            hicat_n_cells_max_for_pca = anal_cfg.HICAT_N_CELL_MAX_FOR_PCA
        else:
            hicat_n_cells_max_for_pca = 60000
        anal_param[var_name] = hicat_n_cells_max_for_pca
            
        var_name = 'HICAT_N_CELL_MAX_FOR_GMM'
        if hasattr(anal_cfg, var_name):
            hicat_n_cells_max_for_gmm = anal_cfg.HICAT_N_CELL_MAX_FOR_GMM
        else:
            hicat_n_cells_max_for_gmm = 10000
        anal_param[var_name] = hicat_n_cells_max_for_gmm
            
        ###################################
        ###  params for Cell Filtering  ###
        var_name = 'N_GENES_MIN'
        if hasattr(anal_cfg, var_name):
            N_genes_min = anal_cfg.N_GENES_MIN
        else:
            N_genes_min = 200   
            
        ## N_genes_min이 너무 크면 N_cells가 너무 작음 -> 가격이 줄어듦 -> 나중에 재주문시 N_genes_min을 작게해서 N_cells이 커져도 추가 과금 불가
        ## 이런 문제 해결하려면 임시방편으로 N_genes_min을 제한
        if N_genes_min > 300:
            N_genes_min = 300
            
        anal_param[var_name] = N_genes_min
            
        var_name = 'N_CELLS_MIN'
        if hasattr(anal_cfg, var_name):
            N_cells_min = anal_cfg.N_CELLS_MIN
        else:
            N_cells_min = 10
        anal_param[var_name] = N_cells_min
    
        var_name = 'PCT_COUNT_MT_MAX'
        if hasattr(anal_cfg, var_name):
            Pct_cnt_mt_max = anal_cfg.PCT_COUNT_MT_MAX
        else:
            Pct_cnt_mt_max = 20
        anal_param[var_name] = Pct_cnt_mt_max
            
        #########################
        ###  params for iCNV  ###
        var_name = 'REFERENCE_CELLTYPES_FOR_TUMOR_ID'
        if hasattr(anal_cfg, var_name):
            tumor_id_ref_celltypes = list(anal_cfg.REFERENCE_CELLTYPES_FOR_TUMOR_ID)
        else:
            tumor_id_ref_celltypes = None
        anal_param[var_name] = tumor_id_ref_celltypes
            
        #############################
        ###  params for DEG/GSEA  ###
        var_name = 'MIN_NUM_CELLS_FOR_CCI'
        if hasattr(anal_cfg, var_name):
            min_n_cells_for_cci = anal_cfg.MIN_NUM_CELLS_FOR_CCI
        else:
            min_n_cells_for_cci = 40
        anal_param[var_name] = min_n_cells_for_cci
    
        var_name = 'UNIT_OF_CCI_RUN'
        if hasattr(anal_cfg, var_name):
            unit_of_cci_run = anal_cfg.UNIT_OF_CCI_RUN
        else:
            unit_of_cci_run = sample_col
        anal_param[var_name] = unit_of_cci_run
    
        var_name = 'CCI_MIN_OCC_FREQ_TO_COUNT'
        if hasattr(anal_cfg, var_name):
            cci_min_OF_to_count = anal_cfg.CCI_MIN_OCC_FREQ_TO_COUNT
        else:
            cci_min_OF_to_count = 0.67
        anal_param[var_name] = cci_min_OF_to_count
    
        var_name = 'CCI_PVAL_CUTOFF'
        if hasattr(anal_cfg, var_name):
            cci_pval_max = anal_cfg.CCI_PVAL_CUTOFF
        else:
            cci_pval_max = 0.1
        anal_param[var_name] = cci_pval_max
    
        var_name = 'CCI_MEAN_CUTOFF'
        if hasattr(anal_cfg, var_name):
            cci_mean_min = anal_cfg.CCI_MEAN_CUTOFF
        else:
            cci_mean_min = 0.1
        anal_param[var_name] = cci_mean_min
    
        var_name = 'CCI_DEG_BASE'
        if hasattr(anal_cfg, var_name):
            cci_deg_base = anal_cfg.CCI_DEG_BASE
        else:
            cci_deg_base = 'celltype_minor'
        anal_param[var_name] = cci_deg_base
    
        #############################
        ###  params for DEG/GSEA  ###
        var_name = 'MIN_NUM_CELLS_FOR_DEG'
        if hasattr(anal_cfg, var_name):
            min_n_cells_for_deg = anal_cfg.MIN_NUM_CELLS_FOR_DEG
        else:
            min_n_cells_for_deg = 100
        anal_param[var_name] = min_n_cells_for_deg
    
        var_name = 'DEG_PVAL_CUTOFF_FOR_GSEA'
        if hasattr(anal_cfg, var_name):
            deg_pval_cutoff = anal_cfg.DEG_PVAL_CUTOFF_FOR_GSEA
        else:
            deg_pval_cutoff = 0.1
        anal_param[var_name] = deg_pval_cutoff
    
        var_name = 'DEG_REF_GROUP'
        if hasattr(anal_cfg, var_name):
            deg_ref_group = anal_cfg.DEG_REF_GROUP
        else:
            deg_ref_group = None
        anal_param[var_name] = deg_ref_group
            
        var_name = 'GSA_PVAL_CUTOFF'
        if hasattr(anal_cfg, var_name):
            gsea_pval_cutoff = anal_cfg.GSA_PVAL_CUTOFF
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
    
        bm = adata_t.var_names.str.startswith('MT-')  
        tc = np.array(adata_t.X.sum(axis = 1))[:,0]
        mc = np.array(adata_t[:, bm].X.sum(axis = 1))[:,0]
        bp = (100*mc/tc) <= Pct_cnt_mt_max
        if np.sum(bp) != len(bp):
            s = '   N cells: %i -> %i (%i cells dropped as pct_count_mt > %4.1f) ' % (len(bp), np.sum(bp), len(bp)-np.sum(bp), Pct_cnt_mt_max)
            flog.writelines(s + '\n')
            log_lines = log_lines + '%s\n' % s
            sa = '%s%s' % (c_prefix, s)
            print(sa, flush = True)
            adata_t = adata_t[bp, :]
        
        bc = (np.array((adata_t.X > 0).sum(axis = 1)) >= N_genes_min)[:,0]  
        if np.sum(bc) != len(bc):
            s = '   N cells: %i -> %i (%i cells dropped as gene_count < %i) ' % (len(bc), np.sum(bc), len(bc)-np.sum(bc), N_genes_min)
            flog.writelines(s + '\n')
            log_lines = log_lines + '%s\n' % s
            sa = '%s%s' % (c_prefix, s)
            print(sa, flush = True)
            adata_t = adata_t[bc, :]
            
        bg = (np.array(adata_t.X.sum(axis = 0)) >= N_cells_min)[0,:]
        if np.sum(bg) != len(bg):
            s = '   N genes: %i -> %i (%i genes dropped as cell_count < %i) ' % (len(bg), np.sum(bg), len(bg)-np.sum(bg), N_cells_min)
            flog.writelines(s + '\n')
            log_lines = log_lines + '%s\n' % s
            sa = '%s%s' % (c_prefix, s)
            print(sa, flush = True)
            adata_t = adata_t[:, bg]
    
        #################################
        ### Check the number of cells ###
    
        num_cells = adata_t.obs.shape[0]
        if num_cells > max_num_cells_allowed:
            s = 'WARNING: N cells = %i exceeds the max. number -> down-sampled to %i' % (num_cells, max_num_cells_allowed)
            flog.writelines(s + '\n')
            log_lines = log_lines + '%s\n' % s
            print(s)
            # print('WARNING: down-sampled to %i.' % (max_num_cells_allowed))
    
            cell_barcodes = list(adata_t.obs.index.values)
            cell_barcodes_sel = random.sample(cell_barcodes, max_num_cells_allowed)
            cell_barcodes_all = pd.Series(list(adata_t.obs.index.values), 
                                          index = adata_t.obs.index.values) 
            b = cell_barcodes_all.isin(cell_barcodes_sel)
            adata_t = adata_t[b,:]
    
        #######################################
        ### Tissue detection & load markers ###
    
        tissue_lst = ['Generic', 'Pancreas', 'Lung', 'Liver', 'Blood', 
                      'Bone', 'Brain', 'Embryo', 'Eye', 'Intestine', 'Breast',
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
                flog.writelines(s + '\n')
                log_lines = log_lines + '%s\n' % s
                sa = '%s%s' % (c_prefix, s)
                print(sa, flush = True)
            
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
                s = s + ' Best match: %s (Non-ref fraction: %4.3f/%4.3f) ' % (tissue, df.loc[tissue, 'fnr'], df['fnr'].max())
                flog.writelines(s + '\n')
                log_lines = log_lines + '%s\n' % s
                sa = '%s%s' % (c_prefix, s)
                print(sa, flush = True)
                '''
                s = 'If you want marker DB for specific tissue, set TISSUE in analysis_config.py'
                log_lines = log_lines + '%s\n' % s
                sa = '%s%s' % (c_prefix, s)
                print(sa, flush = True)
                #'''

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

            if tissue != 'Blood':
                b = df_mkr_db['cell_type_minor'] != 'Monocyte'
                df_mkr_db = df_mkr_db.loc[b,:]
                        
        ##################################
        ###  Cell-type identification  ###
    
        '''
        scoda_hicat(adata_t, df_mkr_db, print_prefix = print_prefix, verbose = 1,
                    clustering_resolution = hicat_clustering_resolution, 
                    pct_th_cutoff = hicat_pct_th_cutoff, 
                    N_cells_max_for_pca = hicat_n_cells_max_for_pca, 
                    N_cells_max_for_gmm = hicat_n_cells_max_for_gmm)
        #'''
        lines = scoda_hicat(adata_t, df_mkr_db, print_prefix = print_prefix, verbose = 1, 
                    clustering_resolution = hicat_clustering_resolution, 
                    pct_th_cutoff = hicat_pct_th_cutoff, 
                    N_cells_max_for_pca = hicat_n_cells_max_for_pca, 
                    N_cells_max_for_gmm = hicat_n_cells_max_for_gmm )
        
        flog.writelines(lines)
        log_lines = log_lines + lines
        
        ## Check unassigned
        b = adata_t.obs['celltype_major'] != 'unassigned'
        if np.sum(b) == 0:
            if user_supplied_mkr:
                s = 'WARNING: Celltypes not assigned at all with the user-specified markers db.'
                flog.writelines(s + '\n')
                log_lines = log_lines + '%s\n' % s
                print(s)
                s = 'WARNING: Rerunning HiCAT using the default markers db .. '
                flog.writelines(s + '\n')
                log_lines = log_lines + '%s\n' % s
                print(s)
    
                ## Get default markers db
                if species.lower() == 'mouse':
                    file_mkr_org = '%s/markers_mm.tsv' % (opt_file_path)
                    file_mkr_new = '%s/markers_mm.tsv' % (data_dir)
                else:
                    file_mkr_org = '%s/markers_hs.tsv' % (opt_file_path)
                    file_mkr_new = '%s/markers_hs.tsv' % (data_dir)
                    
                # cmd = 'cp %s %s' % (file_mkr_org, data_dir)
                # run_command(cmd)
                shutil.copyfile( file_mkr_org, file_mkr_new )
                
                file_mkr = '%s/markers.tsv' % (data_dir)
                # cmd = 'mv %s %s' % (file_mkr_new, file_mkr)
                # run_command(cmd)
                shutil.move( file_mkr_new, file_mkr )
                
                df_mkr_db = pd.read_csv(file_mkr, sep = '\t')
                
                ## Rerun HiCAT
                lines = scoda_hicat(adata_t, df_mkr_db, print_prefix = print_prefix, verbose = 1, 
                            clustering_resolution = hicat_clustering_resolution, 
                            pct_th_cutoff = hicat_pct_th_cutoff, 
                            N_cells_max_for_pca = hicat_n_cells_max_for_pca, 
                            N_cells_max_for_gmm = hicat_n_cells_max_for_gmm )
                flog.writelines(lines)
                log_lines = log_lines + lines
    
            else:
                s = 'WARNING: celltypes not assigned at all with the user-specified markers db.'
                flog.writelines(s + '\n')
                log_lines = log_lines + '%s\n' % s
                print(s)
                s = 'WARNING: Check the tips for preparing markers db.'
                flog.writelines(s + '\n')
                log_lines = log_lines + '%s\n' % s
                print(s)
    
        b = adata_t.obs['celltype_major'] != 'unassigned'
        N_cells_valid = np.sum(b)
        s = 'N cells valid/total = %i/%i ' % (N_cells_valid, len(b))
        flog.writelines(s + '\n')
        log_lines = log_lines + '%s\n' % s
        sa = '%s%s' % (c_prefix, s)
        print(sa, flush = True)
        
        adata_t.write(file_h5ad)
        
        #################################
        ### tumor cell identification ###
    
        b = adata_t.obs['celltype_major'].isin(tumor_id_ref_celltypes)
        if (np.sum(b) < min(1000, len(b)*0.1)):
            tumor_id_ref_celltypes = None
    
        verbose = True
        if tumor_identification:
            
            if species.lower() == 'mouse':
                gtf_file = '%s/mm10_gene_only.gtf' % (opt_file_path)
            else:
                gtf_file = '%s/hg38_gene_only.gtf' % (opt_file_path)
                
            ## Test without Reference 
            ref_types = tumor_id_ref_celltypes
    
            ## clustering_algo = GMM is not suitable for this work
            lines = scoda_icnv_addon( adata_t, gtf_file, 
                               ref_types = ref_types, 
                               ref_key = "celltype_major", 
                               use_ref_only = False, 
                               clustering_algo = CLUSTERING_ALGO, # CLUSTERING_AGO,  
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
                               print_prefix = print_prefix, 
                               n_cores = n_cores_to_use, 
                               verbose = verbose )
            flog.writelines(lines)
            log_lines = log_lines + lines
                
            adata_t.write(file_h5ad)
    
        #################################
        ###   Cell-cell interaction   ###
    
        cpdb_path = file_cpdb
        if (sample_col + '_rev') in list(adata_t.obs.columns.values):
            sample_col_rev = sample_col + '_rev'
        else: 
            sample_col_rev = sample_col
            
        lines = scoda_cci( adata_t, cpdb_path, 
                   cond_col = cond_col, 
                   sample_col = sample_col_rev, 
                   cci_base = cci_deg_base, 
                   unit_of_cci_run = unit_of_cci_run, 
                   n_cores = n_cores_to_use, 
                   min_n_cells_for_cci = min_n_cells_for_cci, 
                   Rth = cci_min_OF_to_count,
                   pval_max = cci_pval_max, 
                   mean_min = cci_mean_min, 
                   data_dir = data_dir,
                   print_prefix = print_prefix, 
                   cpdb_version = 4,
                   verbose = verbose )   
        flog.writelines(lines)
        log_lines = log_lines + lines
        
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
                        n_cores = n_cores_to_use, 
                        print_prefix = print_prefix, 
                        verbose = verbose)  
        flog.writelines(lines)
        log_lines = log_lines + lines
    
        ## generate key #######
        sample_group_map = get_sample_to_group_map(adata_t.obs[sample_col_rev], 
                                                   adata_t.obs[cond_col])
        adata_t.uns['lut_sample_to_cond'] = sample_group_map
    
        
        key2, int_lst2 = get_dataset_key(adata_t, L = kpL, M = kpM)
        keyt = '%i/%i/%s/%s' % (kpL, kpM, key1, key2)
    
        N_cells = adata_t.obs.shape[0]
        N_cells_valid = np.sum(adata_t.obs['celltype_major'] != 'unassinged')
        
        save_info( data_dir + '/%s' % file_hist_expected, 
                   info_dict, session_id, N_cells_org, N_cells, unit_price, keyt )        
        '''
        if file_hist is None:
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
    
            with open(data_dir + '/%s' % file_hist_expected, 'wt+') as f:
                f.writelines(info_line_lst)
                f.close()
        '''
    
        lapsed = time.time() - start_time
        result_file = '%s_results.tar.gz' % (session_id)
    
        s1 = 'Result saved to %s' % (result_file)
        flog.writelines(s1 + '\n')
        log_lines = log_lines + '%s\n' % s1
        s2 = 'SUCCESS: %s data successful. (%4.1f mins, %i -> %i/%i cells total)' % (s_prefix, round(lapsed/60, 1), N_cells_org, N_cells, N_cells_valid)
        flog.writelines(s2 + '\n')
        log_lines = log_lines + '%s' % s2
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
        remove_file(file_mkr)
        remove_file(file_gmt)
        remove_file(file_cpdb)
        
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


def scoda_process_data( file_h5ad_in, 
                        data_dir = '.', 
                        species = 'human', 
                        tissue = None,
                        tumor_id = 'no', 
                        file_h5ad_suffix = '_scoda',
                        data_type = 'h5ad', 
                        home_dir = '',
                        c_prefix = '',
                        lim_Ncells = False, 
                        compress_out = False, 
                        session_id = None ):

    return process_data( data_dir = data_dir, 
                         data_type = data_type, 
                         species = species, 
                         session_id = session_id, 
                         tumor_id = tumor_id, 
                         home_dir = '',
                         c_prefix = '',
                         lim_Ncells = lim_Ncells, 
                         compress_out = compress_out,
                         tissue = tissue,
                         file_h5ad_in = file_h5ad_in, 
                         file_h5ad_suffix = file_h5ad_suffix )


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

