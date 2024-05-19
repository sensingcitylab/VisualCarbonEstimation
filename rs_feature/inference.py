import os
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
from dataset import RSDataset

def main(args):
    # 设置预处理转换
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 加载模型
        # 加载模型并移动到指定 GPU
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    model = resnet50().to(device)
    model.fc = torch.nn.Identity()
    # 加载已训练的权重
    model.load_state_dict(torch.load(args.model_path, map_location=device)['model'], strict=False)
    # 去掉最后一层全连接层
    model = torch.nn.Sequential(*list(model.children())[:-1])
    # 设置为评估模式
    model.eval()

    # 创建每个城市保存特征的文件夹
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    for city in args.city_list:
        print(city)
        # 创建数据集实例
        csv_file = 'metadata/' + city + "_rs_paths.csv"
        dataset = RSDataset(args.root_dir, csv_file , transform=preprocess)

        # 用于存储特征和文件路径的列表
        features_data = []
        # 使用 DataLoader 加载数据集
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        # 对每个批次的数据进行迭代
        for batch in tqdm(dataloader):
            imgs, files = batch
            # 将数据移动到指定 GPU
            imgs = imgs.to(device)

            # 使用模型提取特征
            with torch.no_grad():
                features = model(imgs).squeeze().cpu().numpy()
            # 将城市和特征添加到列表
            for feature, file in zip(features, files):
                
                features_data.append({'city': city, 'file':file, 'features': feature})

        # 转换为 DataFrame
        df = pd.DataFrame(features_data)

        # 保存特征数据到 CSV 文件
        csv_file = os.path.join(output_dir, f'{city}_features.csv')
        df.to_csv(csv_file, index=False)

        print(f"City features for {city} saved to {csv_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from images using ResNet-50")
    parser.add_argument("--root_dir", type=str, default="/data_nas/huangyj/Code/RS_patches", help="Root directory containing city images")
    parser.add_argument("--output_dir", type=str, default="/data_nas/huangyj/Code/RS_patches/rs_features", help="Output directory to save feature CSV files")
    parser.add_argument("--model_path", type=str, default="rsp-resnet-50-ckpt.pth", help="Path to the trained ResNet-50 model checkpoint")
    parser.add_argument("--city_list", nargs="+", default=['三亚市'], help="List of cities to process")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for feature extraction")
    parser.add_argument("--gpu_id", type=int, default=1, help="GPU ID to use")
    args = parser.parse_args()

    args.city_list=['三亚市', '三明市', '上海市']

    print(args)

    main(args)
