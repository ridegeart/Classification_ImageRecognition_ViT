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
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from load_dataset import LoadDataset
from helper import read_meta
from urllib.request import urlopen
#from vit_model import vit_base_patch16_224_in21k as create_model
from vit_model_B import vit_base_patch16_224_in21k as create_model #B-1
import json
from my_dataset import MyDataSet

csv_save_name = 'detect0803_finViT_B-1_F_H.csv'

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
        metafile = '/home/agx/AUO_FMA/FMA_transformer/dataset/pickle_files/meta'
        coarse_labels,fine_labels,third_labels = read_meta(metafile)
        
        img_path = './detect_imgs/'

        '''mdoelsave.pth location'''
        modelName = './weights/A_finModel_F_H.pth'
        
        ''' Mode read'''
        model = create_model(weights="", freeze_layers=False,num_classes=[2,4,14], has_logits=False).to(device)
        model.load_state_dict(torch.load(modelName, map_location=device)) #RuntimeError:Error(s) in loading state_dict for DataParallel 訓練與測試環境不同
        model.eval()

        ''' predict csv'''
        #datacsv ='detect.csv' #args.test_csv現在路徑在args.model_save_path

        '''data - loader'''
        batch_size=1
        epoch = 1
        datadic={}

        #detect_dataset = LoadDataset(csv_path=datacsv, transform=data_transform)
        #detect_generator = DataLoader(detect_dataset, batch_size=batch_size, shuffle=False)

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
                                superclass_pred,subclass_pred ,subtwoclass_pred = model(img.to(device))
                                superclass_pred = torch.squeeze(superclass_pred).cpu()
                                subclass_pred = torch.squeeze(subclass_pred).cpu()
                                subtwoclass_pred = torch.squeeze(subtwoclass_pred).cpu()
                                #predicted_super = torch.argmax(superclass_pred, dim=1)#tensor([1])
                                #predicted_sub = torch.argmax(subclass_pred, dim=1)#tensor([9])
                                #predicted_sub tow= torch.argmax(subtwoclass_pred, dim=1)#tensor([9])

                                ''' confidence  & classes'''
                                ''' - superclasses'''
                                probs_supper = torch.softmax(superclass_pred, dim=0)
                                supper_cla = torch.argmax(probs_supper).numpy()
                                imgclass_supper = coarse_labels[supper_cla]
                                supper_value,supper_index=torch.topk(probs_supper,k=2,largest=True)#torch.topk(取出前幾大) , 2取出幾個        
                                print('supperclass',imgclass_supper,',supper Top2 Class:',supper_index)

                                ''' - subclasses'''
                                probs_sub = torch.softmax(subclass_pred, dim=0)
                                sub_cla = torch.argmax(probs_sub).numpy()
                                imgclass_sub = fine_labels[sub_cla]
                                sub_value,sub_index=torch.topk(probs_sub,k=4,largest=True)#torch.topk(取出前幾大) , 2取出幾個        
                                print('subclass',imgclass_sub,',sub Top4 Class:',sub_index)

                                ''' - subtwoclasses'''
                                probs_subtwo = torch.softmax(subtwoclass_pred, dim=0)
                                subtwo_cla = torch.argmax(probs_subtwo).numpy()
                                imgclass_subtwo = third_labels[subtwo_cla]
                                subtwo_value,subtwo_index=torch.topk(probs_subtwo,k=5,largest=True)#torch.topk(取出前幾大) , 2取出幾個        
                                print('subtwoclass',imgclass_subtwo,',subtwo Top5 Class:',subtwo_index)

                                ''' Get into datadic '''
                                output_dic = {
                                    'supper_conf':[str(index)[:6] for index in supper_value.tolist()],
                                    'supper_class':[coarse_labels[index] for index in supper_index.tolist()],
                                    'sub_conf':[str(index)[:6] for index in sub_value.tolist()],
                                    'sub_class':[fine_labels[index] for index in sub_index.tolist()],
                                    'subtwo_conf':[str(index)[:6] for index in subtwo_value.tolist()],
                                    'subtwo_class':[third_labels[index] for index in subtwo_index.tolist()],
                                    'Layer_1_ans':imgclass_sub,
                                    'Layer_1_conf':str(probs_subtwo[subtwo_cla].numpy()),
                                    'Layer_1_True':flag[2],
                                    'Layer_2_ans':imgclass_sub,
                                    'Layer_2_conf':str(probs_subtwo[subtwo_cla].numpy()),
                                    'Layer_2_True':flag[1],
                                    'Layer_3_ans':imgclass_subtwo,
                                    'Layer_3_conf':str(probs_subtwo[subtwo_cla].numpy()),
                                    'Layer_3_True':flag[0]
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
        #dfsave.reset_index(drop=True,inplace=True)
        
        #makedirs(model_save_path+'result/')
        dfsave.to_csv('./result/'+csv_save_name,index=True,index_label='ImagePath')
        print('data_save:'+'./result/'+csv_save_name)

