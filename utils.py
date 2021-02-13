import os
import glob
import shutil
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import models
import load_data

def test(model, test_loader, num_classes, device):

    #### initialize
    test_label = [0] * num_classes
    test_acc = 0

    #### test
    for i, test_data in enumerate(test_loader):
        org_image = test_data['input'].to(device)
        gt = test_data['label'].type(torch.FloatTensor).to(device)
        model = model.to(device).eval()

        #### forward path
        output = model(org_image)

        gt_label = torch.argmax(gt, dim=1).cpu().detach().tolist()
        output_label = torch.argmax(torch.sigmoid(output), dim=1).cpu().detach().tolist()

        for idx, label in enumerate(gt_label):
            if label == output_label[idx]:
                test_label[label] += 1

        test_acc += sum(torch.argmax(torch.sigmoid(output), dim=1) == torch.argmax(gt, dim=1)).cpu().detach().item()

    return test_label, test_acc

def board_clear(tensorboard_dir):
    files = glob.glob(tensorboard_dir+"/*")
    for f in files:
        shutil.rmtree(f)

def tensorboard_idx(tensorboard_dir):
    i = 0
    while True:
        if Path(os.path.join(tensorboard_dir, str(i))).exists() == True:
            i += 1
        else:
            Path(os.path.join(tensorboard_dir, str(i))).mkdir(exist_ok=True, parents=True)
            return i

def model_config(net_type, num_classes, OOD_num_classes):
    if net_type == "resnet50":
        model = models.resnet50(num_c=num_classes, num_cc=OOD_num_classes, pretrained=True)
    elif net_type == "resnet34":
        model = models.resnet34(num_c=num_classes, num_cc=OOD_num_classes, pretrained=True)
    elif net_type == "resnet18":
        model = models.resnet18(num_c=num_classes, num_cc=OOD_num_classes, pretrained=True)
    elif net_type == "resnet101":
        model = models.resnet101(num_c=num_classes, num_cc=OOD_num_classes, pretrained=True)
    elif net_type == "vgg19":
        model = models.vgg19(num_c=num_classes, num_cc=OOD_num_classes, pretrained=True)
    elif net_type == "vgg16":
        model = models.vgg16(num_c=num_classes, num_cc=OOD_num_classes, pretrained=True)
    elif net_type == "vgg19_bn":
        model = models.vgg19_bn(num_c=num_classes, num_cc=OOD_num_classes, pretrained=True)
    elif net_type == "vgg16_bn":
        model = models.vgg16_bn(num_c=num_classes, num_cc=OOD_num_classes, pretrained=True)

    return model

def image_dir_config(where, dataset):
    json_dir = 0
    if where == "server2":
        if dataset == "animal":
            image_dir = "/mnt/hdd1/mmi_tr1_hdd1/seonghun20/animal_data"
        elif dataset == "top5":
            image_dir = "/mnt/hdd1/mmi_tr1_hdd1/seonghun20/stanford_dog/top5"
        elif dataset == "group2":
            image_dir = "/mnt/hdd1/mmi_tr1_hdd1/seonghun20/stanford_dog/group2/fitting"
        elif dataset == "caltech":
            image_dir = "/mnt/hdd1/mmi_tr1_hdd1/seonghun20/CALTECH256"
        elif dataset == "dog":
            image_dir = "/mnt/hdd1/mmi_tr1_hdd1/seonghun20/stanford_dog/original"
        elif dataset == "cifar10":
            image_dir = "/mnt/hdd1/mmi_tr1_hdd1/seonghun20/CIFAR10"
        elif dataset == "cifar100":
            image_dir = "/mnt/hdd1/mmi_tr1_hdd1/seonghun20/CIFAR100"
        elif dataset == "mobticon_before":
            image_dir = "/mnt/hdd1/mmi_tr1_hdd1/seonghun20/Mobticon"
        elif dataset == "mobticon":
            # image_dir = "/mnt/nas55/Personal/20seonghun/Mobticon_dataset_revision_LR"
            # json_dir = "/home/seonghun20/code/Mobticon/Mobticon_objective/refer/Annotation_LR.json"
            # image_dir = "/mnt/hdd1/mmi_tr1_hdd1/seonghun20/Mobticon/Mobticon_dataset_ver3_reg_stat"
            # json_dir = "/home/seonghun20/code/Mobticon/Mobticon_objective/refer/Annotation_ver3.json"
            image_dir = "/mnt/hdd1/mmi_tr1_hdd1/seonghun20/Mobticon/Mobticon_dataset_ver4_reg_stat_crop_300x200_overlap_finished"
            json_dir = "/home/seonghun20/code/Mobticon/Mobticon_objective/refer/Annotation_ver4_crop.json"

        elif dataset == "tmp":
            # image_dir = "/mnt/nas55/Personal/20seonghun/dataset/Mobticon_tmp/ICE_ANOMALY_dataset"
            # #### python OOD_classification_transfer.py -d tmp --num_classes 3 --gpu 3 --batch_size 32 --net_type resnet50 --where server2 -l 5e-3

            image_dir = "/mnt/nas55/Personal/20seonghun/dataset/Mobticon_tmp/SNOW_ANOMALY_dataset"
            #### python OOD_classification_transfer.py -d tmp --num_classes 3 --gpu 3 --batch_size 32 --net_type resnet50 --where server2 -l 5e-3

            # image_dir = "/mnt/nas55/Personal/20seonghun/dataset/Mobticon_tmp/ALL_CLASS_dataset"
            # #### python OOD_classification_transfer.py -d tmp --num_classes 4 --gpu 0 --batch_size 32 --net_type resnet50 --where server2 -l 5e-3 --not_test_ODIN

            # image_dir = "/mnt/nas55/Personal/20seonghun/dataset/Mobticon_tmp/ALL_WET_dataset"
            # #### python OOD_classification_transfer.py -d tmp --num_classes 3 --gpu 2 --batch_size 32 --net_type resnet50 --where server2 -l 5e-3 --not_test_ODIN

        OOD_dir = "/mnt/hdd1/mmi_tr1_hdd1/seonghun20/COCO"
        # OOD_dir = "/mnt/hdd1/mmi_tr1_hdd1/seonghun20/CUB_200"
    elif where == "server1":
        image_dir = "/home/seonghun/anomaly/data/MVTec"
    elif where == "local":
        if dataset == "animal":
            image_dir = "/media/seonghun/data1/animal_data"
        elif dataset == "top5":
            image_dir = "/media/seonghun/data/stanford_dog/top5"

    return image_dir, OOD_dir, json_dir



def Membership_Loss(output, gt, num_classes):
    R_wrong = 0
    R_correct = 0
    gt_idx = torch.argmax(gt, dim=1)
    for batch_idx, which in enumerate(gt_idx):
        for idx in range(num_classes):
            output_sigmoid = torch.sigmoid(output)
            if which == idx:
                R_wrong += (1 - output_sigmoid[batch_idx][idx]) ** 2
            else:
                R_correct += output_sigmoid[batch_idx][idx] / (num_classes-1)
    return (R_wrong + R_correct) / output.shape[0]

def Transfer_Loss(model, OOD_data, criterion, normal_class_num, device):


    OOD_image = OOD_data['input'] + 0.01 * torch.randn_like(OOD_data['input'])
    OOD_image = OOD_image.to(device)
    OOD_label = OOD_data['label'].type(torch.FloatTensor)
    OOD_gt = torch.cat((torch.zeros(OOD_image.shape[0], normal_class_num), OOD_label)
                       , dim=1).to(device)


    #### forward path
    OOD_output = model.OOD_forward(OOD_image)
    transfer_loss = criterion(OOD_output, OOD_gt)

    return transfer_loss

def Metric_Loss(output_list, gt, triplet):
    target_layer = output_list[-1]
    negative_list = []
    for batch_idx in range(target_layer.shape[0]):
        gt_arg = gt.argmax(dim=1)
        negative = (gt_arg != gt_arg[batch_idx])
        if batch_idx == 0:
            negative_tensor = target_layer[np.random.choice(np.where(negative.cpu().numpy() == True)[0], 1)[0]]
            positive_tensor = target_layer[np.random.choice(np.delete(
                np.where(~negative.cpu().numpy() == True)[0],np.where(np.where(~negative.cpu().numpy() == True)[0] == batch_idx)),
                1)[0]]
            negative_tensor = torch.unsqueeze(negative_tensor, dim=0)
            positive_tensor = torch.unsqueeze(positive_tensor, dim=0)
        else:
            tmp_negative_tensor = target_layer[np.random.choice(np.where(negative.cpu().numpy() == True)[0], 1)[0]]
            negative_tensor = torch.cat((negative_tensor, torch.unsqueeze(tmp_negative_tensor, dim=0)), dim=0)

            tmp_positive_tensor =  target_layer[np.random.choice(np.delete(
                np.where(~negative.cpu().numpy() == True)[0],np.where(np.where(~negative.cpu().numpy() == True)[0] == batch_idx)),
                1)[0]]
            positive_tensor = torch.cat((positive_tensor, torch.unsqueeze(tmp_positive_tensor, dim=0)), dim=0)

    return 0.5 * triplet(target_layer, positive_tensor, negative_tensor)

def data_config(image_dir, OOD_dir, num_classes, OOD_num_classes, batch_size,
                num_instances, soft_label, custom_sampler, not_test_ODIN, transfer, resize=(160,160)):
    train_dataset = load_data.Dog_metric_dataloader(image_dir = image_dir,
                                                    num_class = num_classes,
                                                    mode = "train",
                                                    resize = resize,
                                                    soft_label = soft_label)
    if custom_sampler:
        MySampler = load_data.customSampler(train_dataset, batch_size, num_instances)
        train_loader = DataLoader(train_dataset,
                                                   batch_sampler= MySampler,
                                                   num_workers=4)
    else:
        train_loader = DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=4)



    test_dataset = load_data.Dog_dataloader(image_dir = image_dir,
                                            num_class = num_classes,
                                            mode = "test",
                                            resize = resize)
    test_loader = DataLoader(test_dataset,
                                              batch_size=8,
                                              shuffle=False,
                                              num_workers=4)

    out_test_dataset, out_test_loader, OOD_dataset, OOD_loader = 0,0,0,0
    ### novelty data
    if not_test_ODIN:
        out_test_dataset = load_data.Dog_dataloader(image_dir = image_dir,
                                                 num_class = num_classes,
                                                 mode = "OOD",
                                                    resize=resize)
        out_test_loader = DataLoader(out_test_dataset,
                                                  batch_size=8,
                                                  shuffle=True,
                                                  num_workers=4)

    ### perfectly OOD data
    if transfer:
        OOD_dataset = load_data.Dog_dataloader(image_dir = OOD_dir,
                                             num_class = OOD_num_classes,
                                             mode = "OOD",
                                                    resize=resize)
        OOD_loader = DataLoader(OOD_dataset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=4)

    return train_dataset, train_loader, test_dataset, test_loader, out_test_dataset, out_test_loader, OOD_dataset, OOD_loader


def mobticon_data_config(image_dir, OOD_dir, json_dir, class_info, batch_size,
                num_instances, soft_label, custom_sampler, not_test_ODIN, transfer, resize=(512,380)):
    train_dataset = load_data.Mobticon_dataloader(image_dir = image_dir,
                                                  json_dir = json_dir,
                                                  mode = "train",
                                                  class_info = class_info,
                                                  resize = resize,
                                                  soft_label = soft_label)
    if custom_sampler:
        MySampler = load_data.customSampler(train_dataset, batch_size, num_instances)
        train_loader = DataLoader(train_dataset,
                                  batch_sampler= MySampler,
                                  num_workers=4)
    else:
        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=4)



    test_dataset = load_data.Mobticon_dataloader(image_dir = image_dir,
                                                  json_dir = json_dir,
                                                  class_info = class_info,
                                                 mode = "test",
                                                 resize = resize)
    test_loader = DataLoader(test_dataset,
                                              batch_size=8,
                                              shuffle=False,
                                              num_workers=4)

    out_test_dataset, out_test_loader, OOD_dataset, OOD_loader = 0,0,0,0
    ### novelty data
    if not_test_ODIN:
        out_test_dataset = load_data.Mobticon_dataloader(image_dir = image_dir,
                                                         json_dir = json_dir,
                                                         mode = "OOD",
                                                         class_info = class_info,
                                                         resize=resize)
        out_test_loader = DataLoader(out_test_dataset,
                                                  batch_size=8,
                                                  shuffle=True,
                                                  num_workers=4)

    ### perfectly OOD data
    if transfer:
        OOD_dataset = load_data.Mobticon_dataloader(image_dir = OOD_dir,
                                             mode = "OOD",
                                                    resize=resize)
        OOD_loader = DataLoader(OOD_dataset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=4)

    return train_dataset, train_loader, test_dataset, test_loader, out_test_dataset, out_test_loader, OOD_dataset, OOD_loader


def mobticon_crop_data_config(image_dir, OOD_dir, json_dir, class_info, batch_size,
                num_instances, soft_label, custom_sampler, not_test_ODIN, transfer, resize=(512,380)):
    train_dataset = load_data.Mobticon_crop_dataloader(image_dir = image_dir,
                                                  json_dir = json_dir,
                                                  mode = "train",
                                                  class_info = class_info,
                                                  resize = resize,
                                                  soft_label = soft_label)
    if custom_sampler:
        MySampler = load_data.customSampler(train_dataset, batch_size, num_instances)
        train_loader = DataLoader(train_dataset,
                                  batch_sampler= MySampler,
                                  num_workers=4)
    else:
        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=4)



    test_dataset = load_data.Mobticon_crop_dataloader(image_dir = image_dir,
                                                  json_dir = json_dir,
                                                  class_info = class_info,
                                                 mode = "test",
                                                 resize = resize)
    test_loader = DataLoader(test_dataset,
                                              batch_size=8,
                                              shuffle=False,
                                              num_workers=4)

    out_test_dataset, out_test_loader, OOD_dataset, OOD_loader = 0,0,0,0
    ### novelty data
    if not_test_ODIN:
        out_test_dataset = load_data.Mobticon_crop_dataloader(image_dir = image_dir,
                                                         json_dir = json_dir,
                                                         mode = "OOD",
                                                         class_info = class_info,
                                                         resize=resize)
        out_test_loader = DataLoader(out_test_dataset,
                                                  batch_size=8,
                                                  shuffle=True,
                                                  num_workers=4)

    ### perfectly OOD data
    if transfer:
        OOD_dataset = load_data.Mobticon_crop_dataloader(image_dir = OOD_dir,
                                             mode = "OOD",
                                                    resize=resize)
        OOD_loader = DataLoader(OOD_dataset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=4)

    return train_dataset, train_loader, test_dataset, test_loader, out_test_dataset, out_test_loader, OOD_dataset, OOD_loader