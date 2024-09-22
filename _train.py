import csv
from curtsies.fmtfuncs import green, red, blue
import pickle
import torch
import numpy as np
from model import Encoder, Discriminator
from torch.utils.data import DataLoader
from dataset import Dann_dataset
from torch.optim import Adam
from tqdm import tqdm
from utils import save_model, plot
import os


def loop(encoder, discriminator, loader, opt_enc, opt_disc, criterion, args, mode='train'): 
    with tqdm(enumerate(loader), total=len(loader)) as pbar: 
        pbar.set_description(f'epochs:{args.curr_epoch+1}/{args.epochs}')
        enc_loss_list = list() 
        disc_loss_list = list() 
        total_loss_list = list()
        for idx, (present, _) in pbar: 
            present = present.to(args.dev)
            present = encoder(present) 
            present = discriminator(*present) 
            y_pres = torch.ones_like(present, device=args.dev)
            loss_enc = criterion(present, y_pres) 
            if mode == 'train': 
                opt_enc.zero_grad() 
                loss_enc.backward() 
                opt_enc.step() 
            foreign = torch.randn(args.batch_size, args.latent_dim)
            foreign = foreign.to(args.dev)
            foreign = encoder(foreign)
            foreign = discriminator(*foreign) 
            y_for = torch.zeros_like(foreign, device=args.dev)
            loss_disc = criterion(foreign, y_for) 
            if mode == 'train': 
                opt_disc.zero_grad() 
                loss_disc.backward() 
                opt_disc.step() 
            total_loss = loss_disc.item() + loss_enc.item()
            log_dict = {'loss_enc': round(loss_enc.item(), 4),
                        'loss_disc': round(loss_disc.item(), 4),
                        'total_loss': round(total_loss, 4),
                        'mode': mode,
                        'es': f'{args.early_stopping_idx}/{args.early_stop}'}
            enc_loss_list.append(loss_enc.item())
            disc_loss_list.append(loss_disc.item())
            total_loss_list.append(log_dict['total_loss'])
            pbar.set_postfix(log_dict, refresh=idx%10==0)
        loss_dct = {f'{mode}_loss_enc': round(np.mean(enc_loss_list), 4),
                    f'{mode}_loss_disc': round(np.mean(disc_loss_list), 4),
                    f'{mode}_total_loss': round(np.mean(total_loss_list), 4)}
        pbar.set_postfix(loss_dct)

    return loss_dct

def train(models, loaders, optimizers, criterion, args):
    train_loader = loaders['train']
    val_loader = loaders['val']
    test_loader = loaders['test']

    best_loss = float('inf') 
    epochs_range = range(args.epochs) 
    opt_disc = optimizers['disc']
    opt_enc = optimizers['enc']
    encoder = models['enc']
    discriminator = models['disc']
    for epoch in epochs_range: 
        args.curr_epoch = epoch

        encoder.train() 
        discriminator.train() 
        train_loss = loop(encoder, discriminator, train_loader, 
                opt_enc, opt_disc, criterion, args, mode='train')
        print(blue(f'Training loss {train_loss}'))

        encoder.eval()
        discriminator.eval()
        with torch.no_grad(): 
            val_loss = loop(encoder, discriminator, val_loader, 
                    opt_enc, opt_disc, criterion, args, mode='val')
        loss = val_loss['val_total_loss'] 
        dct = {**train_loss, **val_loss}

        if loss < best_loss: 
            best_loss = loss 
            args.early_stopping_idx = 0 
            save_model(encoder, args.best_enc_path,)
            save_model(discriminator, args.best_disc_path)
            print(green(f'val loss {val_loss}'))
        elif args.early_stopping_idx >= args.early_stop: 
            args.early_stopping_idx -= 1
            print('-'*10, 'Earyly stopping', '-'*10)
            break
        else:
            print(red(f'val loss {val_loss}'))
            args.early_stopping_idx += 1

        save_model(encoder, args.enc_path)
        save_model(discriminator, args.disc_path)

        f = open(args.csv_path, 'a')
        args.log = csv.DictWriter(f, dct.keys()) 
        args.log.writerow(dct)
        f.close()
        plot(args.csv_path, args.plot_path)
        os.system('./g.sh')

    print('==========Training done==============') 


    encoder.eval()
    discriminator.eval()
    with torch.no_grad(): 
        test_loss = loop(encoder, discriminator, test_loader, 
                opt_enc, opt_disc, criterion, args, mode='test')
        print(blue(f'Test loss: {test_loss}'))  

    with open(args.logs_path + '/test_results.txt', 'w') as f:
        f.write(f'test loss: {test_loss}')


def main(args):
    train_data = Dann_dataset(args.train_csv, mode='train')
    val_data = Dann_dataset(args.val_csv, mode='val')
    test_data = Dann_dataset(args.test_csv, mode='test')

    loaders = {'train': DataLoader(train_data, args.batch_size, shuffle=True),
               'val': DataLoader(val_data, args.batch_size),
               'test': DataLoader(test_data, args.batch_size)} 
 
    print(f'{len(train_data) = }')
    print(f'{len(val_data) = }')
    print(f'{len(test_data) = }')
    encoder = Encoder(3).to(args.dev)
    disc = Discriminator().to(args.dev)
    if not args.experiment:
        print('Enter experiment name')
        exit()
    ckpt_dir = f'./ckpts/{args.experiment}/'
    os.makedirs(ckpt_dir, exist_ok=True)
    args.enc_path = os.path.join(ckpt_dir, f'{args.encoder_model}_last.pth')
    args.best_enc_path = os.path.join(ckpt_dir, f'{args.encoder_model}_best.pth')
    args.disc_path = os.path.join(ckpt_dir, f'{args.disc_model}_last.pth')
    args.best_disc_path = os.path.join(ckpt_dir, f'{args.disc_model}_best.pth')
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
    criterion = torch.nn.BCEWithLogitsLoss() 
    models = {'enc': encoder, 'disc': disc}
    optimizers = {'enc': opt_enc, 'disc': opt_disc}

    row = ['train_loss_enc', 'train_loss_disc', 'train_total_loss', \
    'val_loss_enc', 'val_loss_disc', 'val_total_loss']

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
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('-es', '--early_stop', type=int, default=6) 
    parser.add_argument('-trc', '--train_csv', type=str, default='../../qml-data/csv_files/org_train_79.csv') 
    parser.add_argument('-vc', '--val_csv', type=str, default='../../qml-data/csv_files/org_val_9.csv') 
    parser.add_argument('-tc', '--test_csv', type=str, default='../../qml-data/csv_files/org_test_9.csv') 
    parser.add_argument('-ic', '--in_ch', type=int, default=3)
    parser.add_argument('--dev', type=str, default='cuda:0')

    args = parser.parse_args()

    print(args)
    args.early_stopping_idx = 0 

    main(args)
