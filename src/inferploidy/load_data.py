import gdown, gzip, shutil, os, tarfile
import anndata, zipfile

def download_from_google_drive( file_id, out_path = 'downloaded' ):
    url = f'https://drive.google.com/uc?export=download&id={file_id}'
    gdown.download(url, out_path, quiet = False)
    return out_path

def decompress_gz( file_in, file_out, remove_gz = True ):
    try:
        with gzip.open(file_in, 'rb') as f_in:
            with open(file_out, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
                if remove_gz:
                    os.remove(file_in)
                print(f'File saved to: {file_out}')
                return file_out
    except:
        return None


def decompress_zip(file_in, extract_dir = './', remove_zip = True ):
    try:
        with zipfile.ZipFile(file_in, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
            if remove_zip:
                os.remove(file_in)
            print(f'Files extracted to: {extract_dir}')
            return extract_dir
    except:
        return None


def decompress_zip_and_move(file_in, extract_dir='temp_extract', remove_zip=True):
    try:
        with zipfile.ZipFile(file_in, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
            extracted_files = zip_ref.namelist()

        if len(extracted_files) != 1:
            print("Zip 파일에 하나 이상의 파일이 포함되어 있습니다.")
            return None

        # 압축 해제된 파일 경로
        extracted_path = os.path.join(extract_dir, extracted_files[0])

        # 이동 대상 경로 (현재 작업 디렉토리로 이동)
        file_out = os.path.basename(extracted_files[0])
        shutil.move(extracted_path, file_out)

        # 압축 해제 폴더 및 zip 파일 제거 (옵션)
        if remove_zip:
            os.remove(file_in)
        shutil.rmtree(extract_dir)

        return file_out

    except Exception as e:
        print(f"오류 발생: {e}")
        return None


def decompress_tar_gz( file_in, remove_org = True ):

    try:
        extract_path = 'extract_tmp'
        if os.path.isdir(extract_path):
            shutil.rmtree(extract_path)
    
        with tarfile.open(file_in, "r:gz") as tar:
            tar.extractall(path=extract_path)
    
        file_h5ad = os.listdir(extract_path)[0]
        file = extract_path + '/%s' % file_h5ad
        if os.path.isdir(file):
            file_h5ad = os.listdir(file)[0]
            file = file + '/%s' % (file_h5ad)
    
        if os.path.isfile(file_h5ad):
            os.remove(file_h5ad)
    
        if not os.path.isfile(file_h5ad):
           shutil.move(file, '.')
    
        shutil.rmtree(extract_path)
        if remove_org:
            os.remove(file_in)
    
        print(f'File saved to: {file_h5ad}')
        return file_h5ad
    except:
        return None


sample_data_fid_dict = {
    'Lung':       '1yMM4eXAdhRDJdyloHACP46TNCpVFnjqD', 
    'NSCLC':      '1yMM4eXAdhRDJdyloHACP46TNCpVFnjqD',
    'Intestine':  '1oz1USuvIT7VNSmS2WhJuHDZQhkCH6IPY', 
    'Colon':      '1oz1USuvIT7VNSmS2WhJuHDZQhkCH6IPY', 
    'CRC':        '1oz1USuvIT7VNSmS2WhJuHDZQhkCH6IPY',
    'Breast':     '158LUiHiJNFzYvqY-QzMUm5cvIznBrUAV', 
    'BRCA':       '158LUiHiJNFzYvqY-QzMUm5cvIznBrUAV',
    'Pancreas':   '1OgTsyXczHQoV6PJyo_rfNBDJRRHXRhb-', 
    'PDAC':       '1OgTsyXczHQoV6PJyo_rfNBDJRRHXRhb-', 
    'Colitis-mm': '11cVrrxaeai87pKKiUcSjCd-Lw7hkrzrI', 
    'Colitis-hs': '12641IgY-cidvomm4ZZziAcTplYngaq0C', 
    'Melanoma':   '1hlGXEi9UEIZiHGHZJgxZPClVsPdHpvZS'
}

def load_h5ad( tissue, file_type = 'gz' ):

    tlst = list(sample_data_fid_dict.keys())

    if tissue in tlst:
        file_id = sample_data_fid_dict[tissue]
    else:
        print('tissue must be one of %s. ' % ', '.join(tlst))
        return None

    file_h5ad = None
    file_down = download_from_google_drive( file_id )
    # file_h5ad = decompress_gz( file_down, '%s.h5ad' % tissue, remove_gz = True )
    if file_type == 'tar.gz':
        file_h5ad = decompress_tar_gz( file_down, remove_org = True )    
        adata = anndata.read_h5ad(file_h5ad)
    elif file_type == 'gz':
        file_h5ad = decompress_gz( file_down, '%s.h5ad' % tissue, remove_gz = True )
        adata = anndata.read_h5ad(file_h5ad)
    elif file_type == 'zip':
        file_h5ad = decompress_zip_and_move(file_down, remove_zip=True)
        adata = anndata.read_h5ad(file_h5ad)

    return file_h5ad


def load_anndata( tissue ):

    file_h5ad = load_h5ad( tissue )
    if file_h5ad is None:
        return None
    else:
        adata = anndata.read_h5ad(file_h5ad)
        return adata


def load_sample_data( file_id_or_tissue_name, file_type = 'tar.gz' ):

    tlst = list(sample_data_fid_dict.keys())
    
    if file_id_or_tissue_name in tlst:
        return load_anndata( file_id_or_tissue_name )
    else:
        adata = None
        try:
            file_down = download_from_google_drive( file_id_or_tissue_name )
        except:
            print('ERROR: The file_id you requested does not exist.')
            print('You can try one of %s. ' % ', '.join(tlst))
            return None

        file_h5ad = None
        if file_type == 'tar.gz':
            file_h5ad = decompress_tar_gz( file_down, remove_org = True )   
                
        elif file_type == 'gz':
            file_h5ad = decompress_gz( file_down, 'downloaded.h5ad', remove_gz = True )

        elif file_type == 'zip':
            file_h5ad = decompress_zip_and_move(file_down, remove_zip=True)
        
        if file_h5ad is None:
            print('ERROR: The file_type might be a wrong one.')
            print('You can try with one of tar.gz or gz for file_type argument. ')
            print('Or, You can try one of %s for file_id_or_tissue_name. ' % ', '.join(tlst))
            return None
        else:
            try:
                adata = anndata.read_h5ad(file_h5ad)
            except:
                print('ERROR: Cannot read the downloaded file.')
                print('You can try one of %s for file_id_or_tissue_name. ' % ', '.join(tlst))
                return None
                
        return adata

processed_sample_data_fid_dict = {
    'Lung':       '1Xazyv4JhWlhYkDVk51KXaL3DDlAoxftp', 
    'NSCLC':      '1Xazyv4JhWlhYkDVk51KXaL3DDlAoxftp',
    'Intestine':  '1Xb_dzJDgt_RlkXk5nP0jgRUz_aFdP0G9', 
    'Colon':      '1Xb_dzJDgt_RlkXk5nP0jgRUz_aFdP0G9', 
    'CRC':        '1Xb_dzJDgt_RlkXk5nP0jgRUz_aFdP0G9',
    'Breast':     '1XbX8Q3dH1kOWnM6ppms4BR2ukEAKYisB', 
    'BRCA':       '1XbX8Q3dH1kOWnM6ppms4BR2ukEAKYisB',
    'Pancreas':   '1XbYJQpyo8PaoL_vpjBt4YI5tTi8pgV5o', 
    'PDAC':       '1XbYJQpyo8PaoL_vpjBt4YI5tTi8pgV5o', 
    'Colitis-mm': '1QgdmySeTYQjW0NfNxbpaokU22VpEpcHA', 
    'Colitis-hs': '1qRjo2iPlDVxF88umvpxiNGXk6jFK9dbb' 
}

def load_scoda_processed_anndata( tissue = None, file_type = 'tar.gz' ):

    tlst = list(processed_sample_data_fid_dict.keys())

    if tissue in tlst:
        file_id = processed_sample_data_fid_dict[tissue]
    else:
        print('tissue must be one of %s. ' % ', '.join(tlst))
        return None

    file_h5ad = None
    try:
        file_down = download_from_google_drive( file_id )
        if file_type == 'tar.gz':
            file_h5ad = decompress_tar_gz( file_down, remove_org = True )
        elif file_type == 'gz':
            file_h5ad = decompress_gz( file_down, '%s.h5ad' % tissue, remove_gz = True )
        elif file_type == 'zip':
            file_h5ad = decompress_zip_and_move(file_down, remove_zip=True)
            
    except:
        pass

    if file_h5ad is None:
        return None
    else:
        adata = anndata.read_h5ad(file_h5ad)
        return adata


def load_scoda_processed_sample_data( file_id_or_tissue_name = None, file_type = 'tar.gz' ):

    file_type = 'tar.gz'
    tlst = list(processed_sample_data_fid_dict.keys())

    if file_id_or_tissue_name in tlst:
        return load_scoda_processed_anndata( file_id_or_tissue_name, file_type )
    else:
        adata = None
        try:
            file_down = download_from_google_drive( file_id_or_tissue_name )
        except:
            print('ERROR: The file_id you requested does not exist.')
            print('You can try one of %s. ' % ', '.join(tlst))
            return None

        if file_type == 'tar.gz':
            file_h5ad = decompress_tar_gz( file_down, remove_org = True )    
            adata = anndata.read_h5ad(file_h5ad)
        elif file_type == 'gz':
            file_h5ad = decompress_gz( file_down, 'downloaded.h5ad', remove_gz = True )
            adata = anndata.read_h5ad(file_h5ad)
        elif file_type == 'zip':
            file_h5ad = decompress_zip_and_move(file_down, remove_zip=True)
            adata = anndata.read_h5ad(file_h5ad)

        print('ERROR: The file_type might be a wrong one.')
        print('You can try with one of tar.gz or gz for file_type argument. ')
        print('Or, You can try one of %s for file_id_or_tissue_name. ' % ', '.join(tlst))
        return None
            
        return adata

