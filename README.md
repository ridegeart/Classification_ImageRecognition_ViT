# 3 Layer Classification
Implementation of Vision Transformer in PyTorch based on paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale] [1] by Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... & Houlsby.  
The github form : [Github - WZMIAOMIAO] [2]

## setting
- 資料夾:'./transformer/Deep_3'
- 下載權重檔:[路徑]

## Training Data Prepared
1. dataPickle_Transform.py : 原本資料建立meta資料表 ＆ train/test 資料集資訊。
    - section-1 < meta >: coarse_label_names,fine_label_names.
    - setction-2 train / test > create : filenames , coarse_labels, fine_labels ,third_labels
    - section-3 < read meta dic for data transform index > & < train / test > create : filenames , coarse_labels, fine_labels ,third_labels
    [資料夾說明]
    dataPickle_Transform/preimages/train , dataPickle_Transform/preimage/test 的照片要放在 dataset/images。
    dataPickle_Transform/picklefiles/meta,train,test 要放在 dataset/pickle_files。  
    會產生一個pickleTotal.csv彙整表。
2. process_dataset.py : 建立train.csv / test.csv，將訓練集分割成，訓練資料(9):驗證資料(1)。
3. level_dic.py : 建立 level_1 與 Level_2 與 level_3 階層字典。
4. resize.py : 照片 resize (224*224)。
    - Traintype = True

## Training 
1. train.py：進行訓練。
- 更改參數:
    - --num_classes :更改自定義資料集的類別數(3層)。
    - --data-path 路徑:自己的訓練集路徑。
    - --weights 路徑:下載好的權重的存放位置。
    - --train_csv 路徑
    - --test_csv 路徑
    - --metafile 路徑

### Model
|代號 |python檔 |特徵 |Loss |框架 |
|------|--------|--------|--------|--------|
|A_1layer |參考main |--- |--- |常見分類 |
|B-1_F_H |vit_model_B.py |獨立 |Hierachical loss |階層式分類 |
|B-2_SF_H |vit_model.py |共享 |Hierachical loss |階層式分類 |

## Predict (單張測試)
1. predict.py。
- 更改參數:
    - img_path :開啟照片檔案路徑。
    - model_weight_path :訓練好的權重檔路徑。

## Detect Data Prepared
- 預測集 (Detect)：所有照片全部放在一個資料夾內。

## Detect (多張預測)
1. detect.py：進行照片預測。
- 更改參數:
    - img_path :預測照片存放的路徑。
    - modelName :訓練好的權重檔路徑。
    - (line45)呼叫模型時，
        - 不用凍結網路層(freeze_layers=False)。
        - num_classes設為自定義資料集的類別數。
    
## Other needed .py
1. load_dataset.py：數據集的 data loader。
1. utils.py：訓練模型的步驟/計算模型的準確率。
2. flops.py：計算運算浮點數。
3. hierarchical_loss.py：階層損失函數。

## Data Training Detail
- data_transform：自動將輸入圖片resize成224*224。
- num_workers：輸入batch_size，讓電腦自動運算合適的數量。
- create_model：Pretrained Weight 的帶入，
    1. 讀取權重(.pth檔)。
    2. 凍結除分類層外的其他網路層。
    3. 將模型送入GPU (model.to(device))。
- optimizer：只優化需要訓練的網路層部分。

[1]: https://arxiv.org/abs/2010.11929 "Deep Residual Learning for Image Recognition"
[2]: https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/blob/master/pytorch_classification/vision_transformer/README.mdr "Github - WZMIAOMIAO"
[路徑]: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth

