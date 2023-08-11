import os
import sys
#curPath=os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.getcwd())
import cv2 
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from vit_model import vit_base_patch16_224_in21k as create_model
import json

csv_save_name = 'detect0728_finViT_pre_train91.csv'

def makedirs(path):
    try:
        os.makedirs(path)
    except:
        return

if __name__ == "__main__":
        device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
        data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        # read class_indict
        json_path = './class_indices.json'
        assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

        with open(json_path, "r") as f:
                class_indict = json.load(f)
        
        img_path = './detect_imgs/'

        '''mdoelsave.pth location'''
        modelName = './weights/bestmodel_SGD(0.1).pth'

        ''' Mode read'''
        model = create_model(weights="", freeze_layers=False,num_classes=14, has_logits=False).to(device)
        model.load_state_dict(torch.load(modelName, map_location=device)) #RuntimeError:Error(s) in loading state_dict for DataParallel 訓練與測試環境不同
        model.eval()

        '''data - loader'''
        batch_size=1
        epoch = 1
        datadic={}

        dfsave = pd.DataFrame()

        r=0
        for e in range(epoch):
                for image in os.listdir(img_path):
                        img = Image.open(os.path.join(img_path,image))
                        img = data_transform(img)
                        img = torch.unsqueeze(img, dim=0)
                        flag = image.split('@')
                        print(os.path.join(img_path,image))
                        with torch.no_grad():
                                ''' Tensor balue'''
                                subtwoclass_pred= torch.squeeze(model(img.to(device))).cpu()

                                ''' confidence  & classes'''
                                probs_subtwo = torch.softmax(subtwoclass_pred, dim=0)
                                predict_cla = torch.argmax(probs_subtwo).numpy()
                                imgclasstwo_sub = class_indict[str(predict_cla)]

                                subtwo_value,subtwo_index=torch.topk(probs_subtwo,k=5,largest=True)#torch.topk(取出前幾大) , 2取出幾個        
                                print('subclass',imgclasstwo_sub,',sub Top5 Class:',subtwo_index)
                                ''' Get into datadic '''
                                output_dic = {
                                    'subtwo_conf':[str(index)[:6] for index in subtwo_value.tolist()],
                                    'subtwo_class':[class_indict[str(index)] for index in subtwo_index.tolist()],
                                    'ans':imgclasstwo_sub,
                                    'conf':str(probs_subtwo[predict_cla].numpy()),
                                    'True':flag[0],
                        }
                        ''' dataframe concat'''
                        datadic[os.path.join(img_path,image)] = output_dic
                        df = pd.DataFrame(datadic)
                        df = df.T

                        if  len(dfsave) == 0 :
                                dfsave = df 
                        else :
                                dfsave = pd.concat([df,dfsave],axis=0)

        '''datasave cleaner'''
        index_duplicates = dfsave.index.duplicated()
        dfsave = dfsave.loc[~index_duplicates]
        
        dfsave.to_csv('./result/'+csv_save_name,index=True,index_label='ImagePath')
        print('data_save:'+'./result/'+csv_save_name)

