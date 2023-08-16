import os
import sys
sys.path.append(os.getcwd())
import math
import argparse

import torch
print(torch.__version__)
print('Torch Cuda:',torch.cuda.is_available())
print('Torch Cuda Device counts:',torch.cuda.device_count())
from tqdm import tqdm
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from load_dataset import LoadDataset

from level_dict import hierarchy,hierarchy_two
from my_dataset import MyDataSet
#from vit_model import vit_base_patch16_224_in21k as create_model #B-2
from vit_model_B import vit_base_patch16_224_in21k as create_model #B-1
from utils import  train_one_epoch, evaluate, calculate_accuracy
from hierarchical_loss import HierarchicalLossNetwork

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter()

    #train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}

    # 实例化训练数据集
    train_dataset = LoadDataset(image_size=args.img_size, image_depth=args.img_depth, csv_path=args.train_csv,
                                cifar_metafile=args.metafile,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = LoadDataset(image_size=args.img_size, image_depth=args.img_depth, csv_path=args.test_csv,
                                cifar_metafile=args.metafile,
                            transform=data_transform["val"])
    print('train_dataset:'+str(len(train_dataset)))
    print('test_dataset:'+str(len(val_dataset)))

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=LoadDataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=LoadDataset.collate_fn)

    model = create_model(weights=args.weights, freeze_layers=args.freeze_layers, num_classes=args.num_classes, has_logits=False).to(device)
    HLN = HierarchicalLossNetwork(metafile_path=args.metafile, hierarchical_labels_one=hierarchy,hierarchical_labels_two=hierarchy_two, total_level=3,device=device)

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(pg, lr=0.1)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    
    best_acc = 0

    for epoch in range(args.epochs):
        i = 0

        epoch_loss = []
        epoch_superclass_accuracy = []
        epoch_subclass_accuracy = []
        epoch_subtwoclass_accuracy = []
        #------------train-----------------------
        model.train()
        for i, sample in tqdm(enumerate(train_loader)):
            batch_x, batch_y1, batch_y2, batch_y3 = sample['image'].to(device), sample['label_1'].to(device), sample['label_2'].to(device),sample['label_3'].to(device)
            optimizer.zero_grad()
                
            superclass_pred,subclass_pred ,subtwoclass_pred = model(batch_x)
            prediction = [superclass_pred, subclass_pred,subtwoclass_pred] # add subtwoclass - layer3
        
            dloss = HLN.calculate_dloss(prediction, [batch_y1, batch_y2,batch_y3]) #depense loss
            lloss = HLN.calculate_lloss(prediction, [batch_y1, batch_y2,batch_y3]) # layer loss
            
            total_loss = lloss + dloss
            total_loss.backward()
            optimizer.step()
            epoch_loss.append(total_loss.item())
            epoch_superclass_accuracy.append(calculate_accuracy(predictions=prediction[0].detach(), labels=batch_y1))
            epoch_subclass_accuracy.append(calculate_accuracy(predictions=prediction[1].detach(), labels=batch_y2))
            epoch_subtwoclass_accuracy.append(calculate_accuracy(predictions=prediction[2].detach(), labels=batch_y3))


        train_epoch_loss = sum(epoch_loss)/len(epoch_loss)
        train_epoch_superclass_accuracy = sum(epoch_superclass_accuracy)/(i+1)
        train_epoch_subclass_accuracy = sum(epoch_subclass_accuracy)/(i+1)
        train_epoch_subtwoclass_accuracy = sum(epoch_subtwoclass_accuracy)/(i+1)



        print(f'Training Loss at epoch {epoch} : {train_epoch_loss}')
        print(f'Training Superclass accuracy at epoch {epoch} : {train_epoch_superclass_accuracy}')
        print(f'Training Subclass accuracy at epoch {epoch} : {sum(epoch_subclass_accuracy)/(i+1)}')
        print(f'Training Subtwoclass accuracy at epoch {epoch} : {sum(epoch_subtwoclass_accuracy)/(i+1)}')


        j = 0

        epoch_loss = []
        epoch_superclass_accuracy = []
        epoch_subclass_accuracy = []
        epoch_subtwoclass_accuracy = []
        #------------eval--------------------------
        model.eval()
        with torch.set_grad_enabled(False):
            #enumerate(建立批次迴圈計算時間)
            for j, sample in tqdm(enumerate(val_loader)):


                batch_x, batch_y1, batch_y2 ,batch_y3= sample['image'].to(device), sample['label_1'].to(device), sample['label_2'].to(device), sample['label_3'].to(device)
            
                superclass_pred,subclass_pred,subtwoclass_pred= model(batch_x)
                prediction = [superclass_pred,subclass_pred,subtwoclass_pred]

                dloss = HLN.calculate_dloss(prediction, [batch_y1, batch_y2,batch_y3])#depeense loss
                lloss = HLN.calculate_lloss(prediction, [batch_y1, batch_y2,batch_y3])#loss

                total_loss = lloss + dloss

                epoch_loss.append(total_loss.item())
                epoch_superclass_accuracy.append(calculate_accuracy(predictions=prediction[0], labels=batch_y1))
                epoch_subclass_accuracy.append(calculate_accuracy(predictions=prediction[1], labels=batch_y2))
                epoch_subtwoclass_accuracy.append(calculate_accuracy(predictions=prediction[2], labels=batch_y3))


        test_epoch_loss = sum(epoch_loss)/len(epoch_loss)
        test_epoch_superclass_accuracy = sum(epoch_superclass_accuracy)/(j+1)
        test_epoch_subclass_accuracy = sum(epoch_subclass_accuracy)/(j+1)
        test_epoch_subtwoclass_accuracy = sum(epoch_subtwoclass_accuracy)/(j+1)

        print(f'Testing Loss at epoch {epoch} : {test_epoch_loss}')
        print(f'Testing Superclass accuracy at epoch {epoch} : {test_epoch_superclass_accuracy}')
        print(f'Testing Subclass accuracy at epoch {epoch} : {sum(epoch_subclass_accuracy)/(j+1)}')
        print(f'Testing Subtwoclass accuracy at epoch {epoch} : {sum(epoch_subtwoclass_accuracy)/(j+1)}')
        print('-------------------------------------------------------------------------------------------')

        tags = ["train_loss", "train_superclass_acc", "train_subclass_acc", "train_subtwoclass_acc","val_loss", "val_superclass_acc","val_subclass_acc","val_subtwoclass_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_epoch_loss, epoch)
        tb_writer.add_scalar(tags[1], train_epoch_superclass_accuracy, epoch)
        tb_writer.add_scalar(tags[2], train_epoch_subclass_accuracy, epoch)
        tb_writer.add_scalar(tags[3], train_epoch_subtwoclass_accuracy, epoch)
        tb_writer.add_scalar(tags[4], test_epoch_loss, epoch)
        tb_writer.add_scalar(tags[5], test_epoch_superclass_accuracy, epoch)
        tb_writer.add_scalar(tags[6], test_epoch_subclass_accuracy, epoch)
        tb_writer.add_scalar(tags[7], test_epoch_subtwoclass_accuracy, epoch)
        tb_writer.add_scalar(tags[8], optimizer.param_groups[0]["lr"], epoch)

        if best_acc < sum(epoch_subtwoclass_accuracy)/(j+1):
            best_acc = sum(epoch_subtwoclass_accuracy)/(j+1)
            print('Best epoch:',epoch,', Best acc:',best_acc)
            torch.save(model.state_dict(), "./weights/A_bestmodel_F_H.pth")
            print("Best Model saved!")

        torch.save(model.state_dict(), "./weights/A_finModel_F_H.pth")
        print("Final Model saved!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=[2,4,14])
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--img_size', type=int, help='Specify the size of the input image.', default=224)
    parser.add_argument('--img_depth', type=int, help='Specify the depth of the input image.', default=3)
    parser.add_argument('--train_csv', type=str, help='Specify the path to the train csv file.', default='/home/agx/AUO_FMA/FMA_transformer/dataset/train.csv')
    parser.add_argument('--test_csv', type=str, help='Specify the path to the test csv file.', default='/home/agx/AUO_FMA/FMA_transformer/dataset/test.csv')
    parser.add_argument('--metafile', type=str, help='Specify the path to the test csv file.', default='/home/agx/AUO_FMA/FMA_transformer/dataset/pickle_files/meta')

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default="./20_BL")
    parser.add_argument('--model-name', default='', help='create model name')

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='./vit_base_patch16_224_in21k.pth',
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=True)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
