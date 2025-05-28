import gdown, gzip, shutil, os
import scanpy as sc

def download_from_google_drive( file_id, out_path = 'downloaded' ):
    url = f'https://drive.google.com/uc?export=download&id={file_id}'
    gdown.download(url, out_path, quiet = False)
    return out_path

def decompress_gz( file_in, file_out, remove_gz = True ):
    with gzip.open(file_in, 'rb') as f_in:
        with open(file_out, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
            if remove_gz:
                os.remove(file_in)
            print(f'File saved to: {file_out}')
            return file_out

def load_h5ad( tissue ):

    if tissue == 'Lung':
        file_id = '1yMM4eXAdhRDJdyloHACP46TNCpVFnjqD'       
        ## Lung Cancer dataset: selected from GSE131907
    elif (tissue == 'Intestine') | (tissue == 'Colon'):
        tissue = 'Intestine'
        file_id = '1oz1USuvIT7VNSmS2WhJuHDZQhkCH6IPY'  
        ## Colorectal Cancer dataset: selected from GSE132465
    elif tissue == 'Breast':
        file_id = '158LUiHiJNFzYvqY-QzMUm5cvIznBrUAV'     
        ## Breast Cancer dataset: selected from GSE161529
    elif tissue == 'Pancreas':
        file_id = '1OgTsyXczHQoV6PJyo_rfNBDJRRHXRhb-'     
        ## Pancreatic Cancer (PDAC) dataset: selected from GSE161529
    else:
        print('tissue must be one of Breast, Intestine (or Colon), Lung, Pancreas. ')
        return None

    file_down = download_from_google_drive( file_id )
    file_h5ad = decompress_gz( file_down, '%s.h5ad' % tissue, remove_gz = True )

    return file_h5ad


def load_anndata( tissue ):

    file_h5ad = load_h5ad( tissue )
    if file_h5ad is None:
        return None
    else:
        adata = sc.read_h5ad(file_h5ad)
        return adata