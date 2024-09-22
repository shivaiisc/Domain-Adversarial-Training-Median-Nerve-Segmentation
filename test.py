import csv
from curtsies.fmtfuncs import green, red, blue
import pickle
import torch
import numpy as np
from torch.nn import BCEWithLogitsLoss
from model import Decoder, Encoder, Discriminator
from torch.utils.data import DataLoader
from dataset import Dann_dataset, Median_Nerve_dataset
from torch.optim import Adam
from tqdm import tqdm
from utils import save_model, plot
import os
from loss import DiceLoss


@torch.no_grad()
def test(models, loader, dice_criterion, args): 
    encoder = models['encoder'] 
    decoder = models['decoder'] 
    encoder.eval()
    decoder.eval()
    with tqdm(enumerate(loader), total=len(loader)) as pbar: 
        dice_loss_list = list()
        for idx, (img, mask) in pbar: 
            img = img.to(args.dev)
            mask = mask.to(args.dev)
            enc_logits = encoder(img) 
            dec_mask = decoder(*enc_logits) 
            dice_loss = dice_criterion(dec_mask, mask)

            log_dict = {'dice_loss': round(dice_loss.item(), 4)}
            
            dice_loss_list.append(log_dict['dice_loss'])
            pbar.set_postfix(log_dict, refresh=idx%10==0)
        loss_dct = {'dice_loss': round(np.mean(dice_loss_list), 4)}
        pbar.set_postfix(loss_dct)

    return loss_dct


def main(args):
    if args.ckpt_path is None:
        print('Provide ckpt path') 
        exit()
    if args.test is None:
        print('Provide the data to test')
        exit()
    val_data = Median_Nerve_dataset(args.val_csv)
    test_data = Median_Nerve_dataset(args.test_csv)

    loaders = {'val': DataLoader(val_data, args.batch_size),
               'test': DataLoader(test_data, args.batch_size)} 
    print(f'{len(val_data) = }')
    print(f'{len(test_data) = }')
    encoder = Encoder(3).to(args.dev)
    encoder.load_state_dict(torch.load(os.path.join(args.ckpt_path, 'encoder.pth')))
    decoder = Decoder().to(args.dev)
    decoder.load_state_dict(torch.load(os.path.join(args.ckpt_path, 'decoder.pth')))
    dice_criterion = DiceLoss()


    models = {'encoder': encoder, 'decoder': decoder}

    print(vars(args))
    if args.test == 'val': 
        loss = test(models, loaders['val'], dice_criterion, args)
        print(f'val_dice_score = {1-loss["dice_loss"]}')
    elif args.test == 'test': 
        loss = test(models, loaders['test'], dice_criterion, args)
        print(f'test_dice_score = {1-loss["dice_loss"]}')

    print('testing complete')

if __name__ == '__main__': 
    import argparse 
    parser = argparse.ArgumentParser() 
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('-ckpt', '--ckpt_path', type=str) 
    parser.add_argument('-t', '--test', type=str) 
    parser.add_argument('-vc', '--val_csv', type=str, default='../org_data/csv_files/org_val_9.csv') 
    parser.add_argument('-tc', '--test_csv', type=str, default='../org_data/csv_files/org_test_9.csv') 
    parser.add_argument('-ic', '--in_ch', type=int, default=3)
    parser.add_argument('-nf', '--num_filters', type=int, default=3)
    parser.add_argument('--dev', type=str, default='cuda:0')

    args = parser.parse_args()

    print(args)
    args.early_stopping_idx = 0 

    main(args)
