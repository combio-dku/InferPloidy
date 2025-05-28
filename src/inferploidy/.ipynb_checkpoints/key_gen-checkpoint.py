import numpy as np

def int_lst_to_bin_str(int_lst, L = 32):
    
    s = ''
    for a in int_lst:
        s = s + bin(a)[2:].zfill(L)
    return s

def bin_str_to_int_lst(bin_ary, L = 32):

    M = int(len(bin_ary)/L)
    int_ary = []
    for i in range(M):
        s = bin_ary[(i*L):((i+1)*L)]
        int_ary.append(int(s,2))
    return int_ary

def bin_str_to_hex_str(bin_ary):
    
    L = 4
    M = int(len(bin_ary)/L)
    hex_str = ''
    for i in range(M):
        s = bin_ary[(i*L):((i+1)*L)]
        hex_str = hex_str + hex(int(s,2))[2:]
        
    return hex_str

def hex_str_to_bin_str(hex_ary):

    M = len(hex_ary)
    bin_ary = ''
    for i in range(M):
        s = hex_ary[i]
        bin_ary = bin_ary + bin(int(s,16))[2:].zfill(4)
        
    return bin_ary

def get_nz_sum_pf( adata_t, M = 50, axis = 1):

    if axis == 1:
        x = np.array((adata_t.X > 0).sum(axis = 1))[:,0]
    else:
        x = np.array((adata_t.X > 0).sum(axis = 0))[0,:]
        
    x[::-1].sort()
    N = int(len(x)/M)
    y = x[np.arange(M)*N]
    
    return list(y)

def get_nc_ng_ns_ncnd( adata_t, sample_key = 'sample', cond_key = 'condition' ):
    nc = adata_t.obs.shape[0]
    ng = adata_t.var.shape[0]
    ns = 0
    ncnd = 0
    if sample_key in list(adata_t.obs.columns.values):
        ns = len(list(adata_t.obs[sample_key].unique()))
    if cond_key in list(adata_t.obs.columns.values):
        ncnd = len(list(adata_t.obs[cond_key].unique()))

    return [nc, ng, ns, ncnd]


def lfsr(seed_bs, taps, L):
    seed_s = [*seed_bs]
    seed = [int(a) for a in seed_s]    
    out = ''
    for i in range(L):
        nxt = sum([ seed[x] for x in taps]) % 2
        out = out + '%i' % nxt
        seed = ([nxt] + seed)[:max(taps)+1]
    return out

def scrample( bseq_str, seed = '101110100', taps = [1,5,6] ):

    scramble_seq_str = lfsr(seed, taps, len(bseq_str))
    bseq = [int(a) for a in [*bseq_str]]
    sseq = [int(a) for a in [*scramble_seq_str]]

    oseq = ''
    for a, b in zip(bseq, sseq):
        oseq = oseq + '%i' % int(a != b)
    return oseq

def get_dataset_key(adata_t, L = 32, M = 20):

    ncgsc = get_nc_ng_ns_ncnd( adata_t, sample_key = 'sample', cond_key = 'condition' )
    int_lst_0 = get_nz_sum_pf( adata_t, M, axis = 0 )
    int_lst_1 = get_nz_sum_pf( adata_t, M, axis = 1 )
    int_lst = ncgsc + int_lst_0 + int_lst_1

    s = int_lst_to_bin_str(int_lst, L)
    s = scrample( s )

    ## bin to hex
    sh = bin_str_to_hex_str(s) # hex(int(s, 2))[2:]
    
    return sh, int_lst

def recover_int_lst( hseq_str, L = 32):
    ## hex to bin
    bseq_str = hex_str_to_bin_str(hseq_str) # bin(int(hseq_str, 16))[2:]
    
    s = scrample( bseq_str )
    int_lst = bin_str_to_int_lst(s, L)
    return int_lst
    