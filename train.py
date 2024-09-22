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
from loss import SSIM_DICE_BCE, DiceLoss

def loop(models, loader, opts, criterion, args, mode='train'): 
    encoder = models['encoder'] 
    decoder = models['decoder'] 
    discriminator = models['discriminator']
    opt_enc = opts['encoder'] 
    opt_dec = opts['decoder']
    opt_disc = opts['discriminator']
    bce_criterion = criterion['bce']
    seg_criterion = criterion['seg']
    with tqdm(enumerate(loader), total=len(loader)) as pbar: 
        pbar.set_description(f'epochs:{args.curr_epoch+1}/{args.epochs}')
        enc_loss_list = list() 
        disc_loss_list = list() 
        dice_loss_list = list()
        bce_loss_list = list()
        ssim_loss_list = list()
        for idx, (train_data, val_data) in pbar: 
            train_img = train_data['img'].to(args.dev)
            val_img = val_data['img'].to(args.dev)
            with torch.no_grad():
                train_enc_logits = encoder(train_img) 
            train_disc = discriminator(*train_enc_logits, detach=True) 
            y_train = torch.ones_like(train_disc, device=args.dev)
            loss_train_disc = bce_criterion(train_disc, y_train) 
            val_img = val_data['img'].to(args.dev)
            val_enc_logits = encoder(val_img)
            val_disc = discriminator(*val_enc_logits, detach=True) 
            y_val = torch.zeros_like(val_disc, device=args.dev)
            loss_val_disc = bce_criterion(val_disc, y_val)
            loss_disc = 0.5 * (loss_val_disc + loss_train_disc)

            if mode == 'train': 
                opt_disc.zero_grad() 
                loss_disc.backward() 
                opt_disc.step()
            val_disc = discriminator(*val_enc_logits) 
            y_val = torch.ones_like(val_disc, device=args.dev)
            loss_enc = bce_criterion(val_disc, y_val) 
            if mode == 'train': 
                opt_enc.zero_grad() 
                loss_enc.backward() 
                opt_enc.step()

            train_enc_logits = encoder(train_img)
            train_logits = decoder(*train_enc_logits) 
            train_mask = train_data['mask'].to(args.dev)
            seg_loss = seg_criterion(train_logits, train_mask)
            ssim_loss = seg_loss['ssim_loss'] 
            dice_loss = seg_loss['dice_loss'] 
            bce_loss = seg_loss['bce_loss'] 
            loss = args.ssim_ratio * ssim_loss + \
                args.dice_ratio * dice_loss + \
                args.bce_ratio * bce_loss 

            if mode == 'train': 
                opt_dec.zero_grad() 
                opt_enc.zero_grad() 
                loss.backward() 
                opt_dec.step() 
                opt_enc.step()

            log_dict = {'loss_enc': round(loss_enc.item(), 4),
                        'loss_disc': round(loss_disc.item(), 4),
                        'train_dice_loss': round(dice_loss.item(), 4),
                        'train_bce_loss': round(bce_loss.item(), 4), 
                        'train_ssim_loss': round(ssim_loss.item(), 4), 
                        'mode': mode,
                        'es': f'{args.early_stopping_idx}/{args.early_stop}'}
            
            enc_loss_list.append(loss_enc.item())
            disc_loss_list.append(loss_disc.item())
            dice_loss_list.append(log_dict['train_dice_loss'])
            bce_loss_list.append(log_dict['train_bce_loss'])
            ssim_loss_list.append(log_dict['train_ssim_loss'])
            pbar.set_postfix(log_dict, refresh=idx%10==0)
        loss_dct = {f'{mode}_loss_enc': round(np.mean(enc_loss_list), 4),
                    f'{mode}_loss_disc': round(np.mean(disc_loss_list), 4),
                    f'{mode}_dice_loss': round(np.mean(dice_loss_list), 4),
                    f'{mode}_ssim_loss': round(np.mean(ssim_loss_list), 4),
                    f'{mode}_bce_loss': round(np.mean(bce_loss_list), 4)} 

        pbar.set_postfix(loss_dct)

    return loss_dct

@torch.no_grad()
def test(models, loader, criterion, args): 
    encoder = models['encoder'] 
    decoder = models['decoder'] 
    dice_criterion = criterion['dice']
    with tqdm(enumerate(loader), total=len(loader)) as pbar: 
        pbar.set_description(f'epochs:{args.curr_epoch+1}/{args.epochs}')
        dice_loss_list = list()
        for idx, (img, mask) in pbar: 
            img = img.to(args.dev)
            mask = mask.to(args.dev)
            enc_logits = encoder(img) 
            dec_mask = decoder(*enc_logits) 
            dice_loss = dice_criterion(dec_mask, mask)

            log_dict = {'dice_loss': round(dice_loss.item(), 4), 
                        'es': f'{args.early_stopping_idx}/{args.early_stop}'}
            
            dice_loss_list.append(log_dict['dice_loss'])
            pbar.set_postfix(log_dict, refresh=idx%10==0)
        loss_dct = {'dice_loss': round(np.mean(dice_loss_list), 4)}
        pbar.set_postfix(loss_dct)

    return loss_dct



def train(models, loaders, optimizers, criterion, args):
    train_loader = loaders['train']
    val_loader = loaders['val']
    test_loader = loaders['test']

    best_loss = float('inf') 
    epochs_range = range(args.epochs) 
    for epoch in epochs_range: 
        args.curr_epoch = epoch

        train_loss = loop(models, train_loader, optimizers, criterion, args, mode='train')
        print(blue(f'Training loss {train_loss}'))

        with torch.no_grad(): 
            val_loss = test(models, val_loader, criterion, args)
        loss = val_loss['dice_loss'] 
        dct = {**train_loss, **val_loss}

        if loss < best_loss: 
            best_loss = loss 
            args.early_stopping_idx = 0 
            save_model(models, args.best_ckpt_path)
            print(green(f'val loss {val_loss}'))
        elif args.early_stopping_idx >= args.early_stop: 
            args.early_stopping_idx -= 1
            print('-'*10, 'Earyly stopping', '-'*10)
            break
        else:
            print(red(f'val loss {val_loss}'))
            args.early_stopping_idx += 1

        save_model(models, args.ckpt_path)

        f = open(args.csv_path, 'a')
        args.log = csv.DictWriter(f, dct.keys()) 
        args.log.writerow(dct)
        f.close()
        plot(args.csv_path, args.plot_path)

    print('==========Training done==============') 


    with torch.no_grad(): 
        test_loss = test(models, test_loader, criterion, args)
        print(blue(f'Test loss: {test_loss}'))  

    with open(args.logs_path + '/test_results.txt', 'w') as f:
        f.write(f'test loss: {test_loss}')


def main(args):
    train_data = Dann_dataset(args.train_csv, args.val_csv)
    val_data = Median_Nerve_dataset(args.val_csv)
    test_data = Median_Nerve_dataset(args.test_csv)

    loaders = {'train': DataLoader(train_data, args.batch_size, shuffle=True),
               'val': DataLoader(val_data, args.batch_size),
               'test': DataLoader(test_data, args.batch_size)} 
    print(f'{len(train_data) = }')
    print(f'{len(val_data) = }')
    print(f'{len(test_data) = }')
    encoder = Encoder(3).to(args.dev)
    disc = Discriminator(args.num_filters).to(args.dev)
    decoder = Decoder().to(args.dev)
    if not args.experiment:
        print('Enter experiment name')
        exit()
    ckpt_dir = f'./ckpts/{args.experiment}/'
    if args.resume: 
        encoder.load_state_dict(torch.load(ckpt_dir + 'encoder.pth'))
        decoder.load_state_dict(torch.load(ckpt_dir + 'decoder.pth'))
        disc.load_state_dict(torch.load(ckpt_dir + 'discriminator.pth'))
        print('loaded encoder, decoder, discriminator') 
    else:
        print('Training from scratch') 
    os.makedirs(ckpt_dir, exist_ok=True)
    args.ckpt_path = ckpt_dir
    args.best_ckpt_path = os.path.join(ckpt_dir, 'best')
    os.makedirs(args.best_ckpt_path, exist_ok=True)
    args.logs_path = os.path.join('logs', f'{args.experiment}/')
    os.makedirs(args.logs_path, exist_ok=True)
    pickle_path = os.path.join(args.logs_path, 'config.pkl')
    with open(pickle_path, 'wb') as f:
        pickle.dump(vars(args), f)
    args.csv_path = os.path.join(args.logs_path, 'log.csv') 
    args.plot_path = os.path.join(args.logs_path, 'plots/')#f'{args.logs_path}/plots/'
    os.makedirs(args.plot_path, exist_ok=True)
    config_txt_path = os.path.join(args.logs_path, 'config.txt') 

    opt_disc = Adam(disc.parameters(), lr=args.lr)
    opt_enc = Adam(encoder.parameters(), lr=args.lr)
    opt_dec = Adam(decoder.parameters(), lr=args.lr)
    dann_criterion = torch.nn.BCEWithLogitsLoss() 
    bce_criterion = BCEWithLogitsLoss()
    seg_criterion = SSIM_DICE_BCE()
    dice_criterion = DiceLoss()

    models = {'encoder': encoder, 'discriminator': disc, 'decoder': decoder}
    optimizers = {'encoder': opt_enc, 'discriminator': opt_disc, 'decoder': opt_dec}
    criterion = {'seg': seg_criterion, 'bce': bce_criterion, 'dice': dice_criterion}

    row = ['train_loss_enc', 'train_loss_disc', 'train_dice_loss', \
           'train_ssim_loss', 'train_bce_loss', 'dice_loss']

    dct = {k:k for k in row}
    f = open(args.csv_path, 'w')
    args.log = csv.DictWriter(f, dct.keys()) 
    args.log.writerow(dct)
    f.close()
    print(vars(args))
    with open(config_txt_path, 'w') as f: 
        f.write(str(vars(args)))
    print(models, file=open(os.path.join(args.logs_path, 'model.txt'), 'w'))
    train(models, loaders, optimizers, criterion, args)

if __name__ == '__main__': 
    import argparse 
    parser = argparse.ArgumentParser() 
    parser.add_argument('-ep', '--epochs', type=int, default=50)
    parser.add_argument('-exp', '--experiment', type=str )
    parser.add_argument('--encoder_model', type=str, default='unet_encoder')
    parser.add_argument('--disc_model', type=str, default='unet_disc')
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--dice_ratio', type=float, default=0.33)
    parser.add_argument('--bce_ratio', type=float, default=0.33)
    parser.add_argument('--ssim_ratio', type=float, default=0.33)
    parser.add_argument('-es', '--early_stop', type=int, default=10) 
    parser.add_argument('-nf', '--num_filters', type=int, default=256) 
    parser.add_argument('-trc', '--train_csv', type=str, default='../qml-data/csv_files/org_train_79.csv') 
    parser.add_argument('-vc', '--val_csv', type=str, default='../qml-data/csv_files/org_val_9.csv') 
    parser.add_argument('-tc', '--test_csv', type=str, default='../qml-data/csv_files/org_test_9.csv') 
    parser.add_argument('-ic', '--in_ch', type=int, default=3)
    parser.add_argument('--dev', type=str, default='cuda:0')
    parser.add_argument('--resume', type=int, default=0)

    args = parser.parse_args()

    print(args)
    args.early_stopping_idx = 0 

    main(args)
