import torch 
from tqdm import tqdm 
import numpy as np
from einops import rearrange
from torch import log, nn 
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import make_grid

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__() 

    def forward(self, logits, targets):
        
        intsn = torch.sum(logits * targets)
        union = torch.sum(logits) + torch.sum(targets) 
        dice_loss = 1 - (2.*intsn+1.0)/(union+1.0)
        
        return dice_loss


class DiceScore(nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, inputs, targets, smooth=1):
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return dice
 

class LogCoshDiceLoss(nn.Module):
    def __init__(self):
        super(LogCoshDiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-5):
        
        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)       
        # targets = (targets > 0.5).float()
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return torch.log(torch.cosh(1-dice))

class SSIM(nn.Module):
    def __init__(self, window_size=11):
        super().__init__() 
        self.window = Variable(self._window(window_size))
        self.window_size = window_size

    def _window(self, window_size, sigma=1.5): 
        x = torch.arange(window_size) 
        g_kern = torch.exp(-(x-window_size//2)**2/2/sigma**2)
        g_kern = g_kern.view(1, -1)
        g_kern = g_kern.t() @ g_kern 
        g_kern = g_kern/torch.sum(g_kern)
        return rearrange(g_kern, 'h w -> 1 1 h w') 

    def forward(self, img0, img1):
        device = img0.device
        _, ch, _, _ = img0.shape 
        self.window = self.window.to(device) 
        mu1 = F.conv2d(img0, self.window, padding=self.window_size//2, groups=ch)
        mu2 = F.conv2d(img1, self.window, padding=self.window_size//2, groups=ch) 
        mu1_sq = mu1**2 
        mu2_sq = mu2**2 
        mu1_mu2 = mu1*mu2

        sigma1_sq =  F.conv2d(img0*img0, self.window, padding=self.window_size//2, groups=1) - mu1_sq 
        sigma2_sq =  F.conv2d(img1*img1, self.window, padding=self.window_size//2, groups=1) - mu2_sq 
        sigma_12 =  F.conv2d(img0*img1, self.window, padding=self.window_size//2, groups=1) - mu1_mu2

        c0 = 0.01**2 
        c1 = 0.03**2 

        ssim_map = ((2*mu1_mu2+c0)*(2*sigma_12+c1))/((mu1_sq+mu2_sq+c0)*(sigma1_sq+sigma2_sq+c1))
        return 1-ssim_map.mean()

class SSIM_DICE_BCE(nn.Module):

    def __init__(self):
        super().__init__() 
        self.ssim = SSIM() 
        self.dice= DiceLoss() 
        self.bce = nn.BCELoss()

    def forward(self, logits, targets):
        ssim_loss = self.ssim(logits, targets.to(torch.float32)) 
        bce_loss = self.bce(logits, targets.to(torch.float32))
        dice_loss= self.dice(logits, targets) 
        return {'ssim_loss': ssim_loss, 'bce_loss': bce_loss,\
                'dice_loss': dice_loss}

@torch.no_grad()
def calc_metrics(mask, preds):
    preds = torch.where(preds <= 0.5, 0.0, 1.0)
    
    TP = torch.sum((preds == 1.0) & (mask == 1.0))
    TN = torch.sum((preds == 0.0) & (mask == 0.0))
    FP = torch.sum((preds == 0.0) & (mask == 1.0)) 
    FN = torch.sum((preds == 1.0) & (mask == 0.0))


    acc = (TP+TN)/(TP+TN+FP+FN) 
    if TP:
        presn = TP/(TP+FP)
        recall = TP/(TP+FN)
        f1_score = 2*presn*recall/(presn+recall)
    else:
        presn = torch.tensor(0.0) 
        recall = torch.tensor(0.0) 
        f1_score = torch.tensor(0.0)

    return {'acc': acc.item(), 'precision': presn.item(), 
            'recall': recall.item(), 'f1_score': f1_score.item()}
    

if __name__ == '__main__': 
    from torchvision.transforms import ToPILImage, ToTensor 
    from PIL import Image 
    from models import UNET
    from torch.utils.data import DataLoader

    import argparse 

    parser = argparse.ArgumentParser() 

    parser.add_argument('-w', '--window_size', type=int, default=11)
    parser.add_argument('-imp', '--img_path', type=str, default='./data/train/0/img_111.jpg')
    parser.add_argument('-mkp', '--mask_path', type=str, default='./data/train/0/mask_111.tif')
    parser.add_argument('-sh', '--imshow', type=bool, default=True)

    config = parser.parse_args()

    model = UNET().cuda()
    criterion = SSIM_DICE_BCE()

    mask = Image.open('//home/shivac/qml-data/MEDVID0001_M_20210908_130347_0001_IMAGES/0/mask.png').convert('L') 
    img = Image.open('//home/shivac/qml-data/MEDVID0001_M_20210908_130347_0001_IMAGES/0/img.png').convert('L') 
    img = ToTensor()(img).cuda().unsqueeze(0)
    mask = ToTensor()(mask).cuda().unsqueeze(0)

    model.train()
    logits = model(img)
    loss_metrics = criterion(logits, mask)
    print(loss_metrics)
    # logits = torch.where(logits<=0.5, 0.0, 1.0)

