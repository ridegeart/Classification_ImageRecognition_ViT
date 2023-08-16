import os
import sys
sys.path.append(os.getcwd())
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from helper import read_meta
from vit_model import vit_base_patch16_224_in21k as create_model


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # load image
    img_path = "./TestPicture_2/CF DEFECT@NP@CF@CF/B4VW1YC-1-3.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)
    metafile = '/home/agx/AUO_FMA/FMA_transformer/dataset/pickle_files/meta'
    '''data - loader'''
    coarse_labels,fine_labels,third_labels = read_meta(metafile)

    # create model
    model = create_model(weights="", freeze_layers=False,num_classes=[2,4,14], has_logits=False).to(device)
    # load model weights
    model_weight_path = "./weights/A_finModel_F_H.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    with torch.no_grad():
        # predict class
        superclass_pred,subclass_pred ,subtwoclass_pred = model(img.to(device))
        superclass_pred = torch.squeeze(superclass_pred).cpu()
        subclass_pred = torch.squeeze(subclass_pred).cpu()
        subtwoclass_pred = torch.squeeze(subtwoclass_pred).cpu()
        superclass_pred = torch.softmax(superclass_pred, dim=0)
        superclass_pred_cla = torch.argmax(superclass_pred).numpy()
        subclass_pred = torch.softmax(subclass_pred, dim=0)
        subclass_pred_cla = torch.argmax(subclass_pred).numpy()
        subtwoclass_pred = torch.softmax(subtwoclass_pred, dim=0)
        subtwoclass_pred_cla = torch.argmax(subtwoclass_pred).numpy()
    
    print_super = "superclass: {}  prob: {:.3}".format(coarse_labels[superclass_pred_cla],
                                                 superclass_pred[superclass_pred_cla].numpy())
    
    for i in range(len(superclass_pred)):
        print("superclass: {:10}   prob: {:.3}".format(coarse_labels[i],
                                                  superclass_pred[i].numpy()))
    
    print_sub = "subclass: {}  prob: {:.3}".format(fine_labels[subclass_pred_cla],
                                                 subclass_pred[subclass_pred_cla].numpy())
    
    for i in range(len(subclass_pred)):
        print("subclass: {:10}   prob: {:.3}".format(fine_labels[i],
                                                  subclass_pred[i].numpy()))
    
    print_subtwo = "subTwoclass: {}  prob: {:.3}".format(third_labels[subtwoclass_pred_cla],
                                                 subtwoclass_pred[subtwoclass_pred_cla].numpy())
    
    for i in range(len(subtwoclass_pred)):
        print("subTwoclass: {:10}   prob: {:.3}".format(third_labels[i],
                                                  subtwoclass_pred[i].numpy()))

    plt.title(print_super+', '+print_sub+', '+print_subtwo)
    plt.show()


if __name__ == '__main__':
    main()
