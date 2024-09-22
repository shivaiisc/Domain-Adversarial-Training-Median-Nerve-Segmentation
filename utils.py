from os.path import join
import torch 
import pandas as pd 
from matplotlib import pyplot as plt 

def save_model(models, path): 
    for k in models.keys(): 
        torch.save(models[k].state_dict(), join(path, k+'.pth'))

def plot(csv_path, save_path):

    df = pd.read_csv(csv_path)

    plt.plot(df['train_loss_enc'], label = 'encoder')
    plt.plot(df['train_loss_disc'], label = 'critic')
    plt.legend() 
    plt.grid()
    plt.show()
    plt.savefig(save_path + 'dann_loss.png')
    plt.clf()

    plt.plot(df['train_dice_loss'], label='train_dice_loss')
    plt.plot(df['train_bce_loss'], label='train_bce_loss')
    plt.plot(df['train_ssim_loss'], label='train_ssim_loss')
    plt.plot(df['dice_loss'], label='val_dice_loss')
    plt.legend() 
    plt.grid()
    plt.show()
    plt.savefig(save_path + 'seg_loss.png')
    plt.clf()



if __name__ == '__main__': 
    plot('./logs/test-dann/log.csv', './')

