import pandas as pd 
import numpy as np
import torch 

def check_model_loading(ckpt_path): 
    ckpt = torch.load(ckpt_path) 
    print(ckpt.keys()) 

def pd_test(): 
    csv_path = '/home/shivac/qml-data/csv_files/org_val_9.csv'
    df = pd.read_csv(csv_path)
    print(df.head())
    patinet_ids = np.unique(df.patient_id)
    n = len(patinet_ids)//2
    first = patinet_ids[:n]
    second = patinet_ids[n:2*n]
    df = df.groupby('patient_id')
    f_lst = list()
    s_lst = list()
    for f, s in zip(first, second): 
        f_lst.append(df.get_group(f)) 
        s_lst.append(df.get_group(s))
    f_df = pd.concat(f_lst) 
    s_df = pd.concat(s_lst)
    n = min(len(f_df), len(s_df))
    f_df = f_df[:n]
    s_df = s_df[:n]
    print(f'{len(f_df) = }')
    print(f'{len(s_df) = }')




if __name__ == '__main__': 
    check_model_loading('./ckpts/test-3/best/encoder.pth')

