# Classification
Implementation of Vision Transformer in PyTorch based on paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale] [1] by Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... & Houlsby.
The github form : [Github - WZMIAOMIAO] [2]

## setting
- 下載權重檔:[路徑]

## Data Prepared
- 訓練集(含驗證集)：一個類別對應一個文件夾。
- 預測集 (Detect)：所有照片全部放在一個資料夾內。

## Training 
1. 使用train.py 進行訓練。
2. 更改參數:
    - --num_classes :更改自定義資料集的類別數。
    - --data-path 路徑:自己的訓練集路徑。
    - --weights 路徑:下載好的權重的存放位置。

## Predict (單張測試)
1. 使用predict.py。
2. 更改參數:
    - img_path :開啟照片檔案路徑。
    - model_weight_path :訓練好的權重檔路徑。

## Detect (多張預測)
1. 使用detect.py 進行照片預測。
2. 更改參數:
    - img_path :預測照片存放的路徑。
    - modelName :訓練好的權重檔路徑。
    - (line14)呼叫模型時，
        - 不用凍結網路層(freeze_layers=False)。
        - num_classes設為自定義資料集的類別數。

## Data Training Detail
- read_split_data：自動將訓練集分割成，訓練資料(8):驗證資料(2)。
- data_transform：自動將輸入圖片resize成224*224。
- MyDataSet：自動生成class_indices(所有類別的dict)。
- num_workers：輸入batch_size，讓電腦自動運算合適的數量。
- create_model：Pretrained Weight 的帶入，
    1. 讀取權重(.pth檔)。
    2. 凍結除分類層外的其他網路層。
    3. 將模型送入GPU (model.to(device))。
- optimizer：只優化需要訓練的網路層部分。

[1]: https://arxiv.org/abs/2010.11929 "Deep Residual Learning for Image Recognition"
[2]: https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/blob/master/pytorch_classification/vision_transformer/README.mdr "Github - WZMIAOMIAO"
[路徑]: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth