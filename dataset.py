from os.path import join
from PIL import Image
from torchvision import transforms as T 
from torch.utils.data import Dataset, DataLoader 
from torchvision.transforms import v2 
import pandas as pd 
import torch

class Dann_dataset(Dataset): 
    def __init__(self, train_csv_path, val_csv_path): 
        self.transform = T.Compose([T.ToTensor()])
        self.transform = v2.Compose([v2.RandomHorizontalFlip(0.5),
                                     v2.RandomVerticalFlip(0.5),
                                     v2.ToImage(), 
                                     v2.ToDtype(torch.float32, scale=True)])
        
        self.train_df = pd.read_csv(train_csv_path)
        self.val_df = pd.read_csv(val_csv_path)

    def __len__(self): 
        return len(self.train_df)

    def __getitem__(self, idx): 
        val_idx = idx % len(self.val_df)
        train_img_path = '../qml-data/' + self.train_df.img_path[idx]
        val_img_path = '../qml-data/' + self.val_df.img_path[val_idx]
        train_mask_path = '../qml-data/' + self.train_df.mask_path[idx]
        val_mask_path = '../qml-data/' + self.val_df.mask_path[val_idx]
        train_img = Image.open(train_img_path) 
        val_img = Image.open(val_img_path) 
        val_mask = Image.open(val_mask_path) 
        train_mask = Image.open(train_mask_path) 
        torch.manual_seed(idx)
        train_img = self.transform(train_img) 
        torch.manual_seed(idx)
        val_img = self.transform(val_img) 
        torch.manual_seed(idx)
        train_mask = self.transform(train_mask) 
        torch.manual_seed(idx)
        val_mask = self.transform(val_mask)
        train = {'img': train_img, 'mask': train_mask}
        val = {'img': val_img, 'mask': val_mask}
        return train, val 



class Median_Nerve_dataset(Dataset): 
    def __init__(self, data_csv): 
        super().__init__() 
        self.df = pd.read_csv(data_csv)
        self.transform = T.ToTensor()

    def __len__(self): 
        return len(self.df)

    def __getitem__(self, index): 
        img_path = '../qml-data/'+self.df.img_path[index]
        mask_path = '../qml-data/'+self.df.mask_path[index] 
        img = Image.open(img_path)
        mask = Image.open(mask_path)
        img = self.transform(img) 
        mask = self.transform(mask)
        return img, mask



if __name__ == '__main__': 
    train_csv = '../org_data/csv_files/org_train_79.csv'
    val_csv = '../org_data/csv_files/org_val_9.csv'
    test_csv = '../org_data/csv_files/org_test_9.csv'
    train_data = Dann_dataset(train_csv, val_csv)
    val_data = Median_Nerve_dataset(val_csv)
    test_data = Median_Nerve_dataset(test_csv)
    print(f'{len(train_data) = }')
    print(f'{len(val_data) = }')
    print(f'{len(test_data) = }')
    train_data = DataLoader(train_data, 32)
    val_data = DataLoader(val_data, 32)
    test_data = DataLoader(test_data, 32)
    from tqdm import tqdm 
    for train_img, val_img in tqdm(train_data):
        pass
    for train_img, val_img in tqdm(val_data):
        pass
    for train_img, val_img in tqdm(test_data):
        pass
