# For extracting remote feature 
### How to Run the Scripts

1. First, run the `process.ipynb` notebook to generate the image path file.

2. Then, navigate to the `rs_feature` folder and run `inference.py` to extract features. Use the following command:

```bash
python inference.py --root_dir /data_nas/huangyj/Code/RS_patches \
    --output_dir /data_nas/huangyj/Code/RS_patches/rs_features \
    --model_path rsp-resnet-50-ckpt.pth \
    --batch_size 1024 \
    --gpu_id 1 \
    --city_list 三亚市 三明市 上海市
```

checkpoint: https://github.com/ViTAE-Transformer/RSP


Note: It's recommended to modify the city_list parameter in the inference.py file to select the list of cities to process.