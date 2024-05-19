import torch
import os
import rasterio
import pandas as pd
from torchvision.transforms import ToPILImage

class RSDataset(torch.utils.data.Dataset):
    def __init__(self, root, csvfile, transform=None):
        self.root = root
        self.transform = transform
        df = pd.read_csv(os.path.join(root,csvfile))

        self.imgs = df["image_path"].to_list()
        self.cities = df["city"].to_list()

    def __getitem__(self, index):
        file = self.imgs[index]
        city = self.cities[index]
        
        file_path = os.path.join(self.root,city, file)
        

        # 读取图像文件
        
        with rasterio.open(file_path) as src:
            img = src.read()  # 读取图像数据
            img = img.transpose(1, 2, 0)  # 将通道维度调整到最后
            # img = img.astype('uint8')  # 转换数据类型为uint8
            
            img = ToPILImage()(img)  # 将 Numpy 数组转换为 PIL 图像


        if self.transform is not None:
            img = self.transform(img)

        sample = (img, file)
        return sample

    def __len__(self):
        return len(self.imgs)
