import os
import random
import copy
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from collections import defaultdict

def extract_time_info(target, check_num=6):
    target = ''.join(x for x in target if x.isdigit())
    return target[:check_num]


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

def str_mapping(path, str=".bmp", str2=".png"):
    file_list = np.array(os.listdir(path))
    file_map = np.array([str in file for file in file_list])
    file_list_final = file_list[file_map]
    if not file_list_final.size:
        file_map = np.array([str2 in file for file in file_list])
        file_list_final = file_list[file_map]

    return list(file_list_final)


def append_value(dict_obj, key, value):
    # Check if key exist in dict or not
    if key in dict_obj:
        # Key exist in dict.
        # Check if type of value of key is list or not
        if not isinstance(dict_obj[key], list):
            # If type is not list then make it list
            dict_obj[key] = [dict_obj[key]]
        # Append the value in list
        if isinstance(value, list):
            dict_obj[key].extend(value)
        else:
            dict_obj[key].append(value)

    else:
        # As key is not in dict,
        # so, add key-value pair
        dict_obj[key] = value


class Mobticon_crop_dataloader(Dataset):
    def __init__(self, image_dir, json_dir, mode, class_info, resize=(512,380), soft_label = False, repeat=1):
        #### json config
        with open(json_dir) as json_file:
            json_data = json.load(json_file)

        for class_idx in class_info[1]:     #### except
            json_data["TRAIN"].pop(class_idx)
            json_data["TEST"].pop(class_idx)

        assert len(class_info[2]) < 2, "OOD_class must be one or zero"         #### OOD class error

        #### same class
        assert len(class_info[0]) > 1 or len(class_info[0]) == 0, "same class args must higher than 2"
        for idx, class_idx in enumerate(class_info[0]):
            if idx == 0:
                first_idx = class_idx
                continue
            tmp_list = json_data["TRAIN"].pop(class_idx)
            append_value(json_data["TRAIN"], first_idx, tmp_list)
            tmp_list = json_data["TEST"].pop(class_idx)
            append_value(json_data["TEST"], first_idx, tmp_list)


        if mode == "train" or mode == "test":
            json_data = json_data["TRAIN"] if mode =="train" else json_data["TEST"]
            for class_idx in class_info[2]:     #### OOD
                json_data.pop(class_idx)

        elif mode == "OOD":
            for class_idx in class_info[2]:
                tmp_list = json_data["TRAIN"][class_idx]
                tmp_list.extend(json_data["TEST"][class_idx])
                json_data=dict()
                json_data[class_idx] = tmp_list


        self.class_list = sorted(list(json_data.keys()))              #### key -> gt

        self.num_per_class = dict()
        for class_index in self.class_list:
            length=0
            for folder_name in json_data[class_index].keys():
                length = length+1 if isinstance(json_data[class_index][folder_name], int) else length + len(json_data[class_index][folder_name])
            self.num_per_class[class_index] = length
        print("number of each class", end="\t")
        print(self.num_per_class)

        #### listing
        self.gt = []
        self.image_list = []
        self.thermal_list = []
        self.ROI_list = []
        self.HT_list = []
        for class_index in self.class_list:
            for folder_name in json_data[class_index].keys():
                thermal_dir = os.path.join(image_dir, folder_name, "Thermal")
                Img_dir = os.path.join(image_dir, folder_name, "Img")

                Img_name = [file for file in os.listdir(Img_dir) if len(file) == 20]
                Thermal_name = [file for file in os.listdir(thermal_dir) if len(file) == 19]

                #### index control
                idx_in_folder_list = json_data[class_index][folder_name]
                if isinstance(idx_in_folder_list, int):
                    full_image_path = os.path.join(Img_dir, os.path.splitext(Img_name[0])[0]+"_{:04d}".format(idx_in_folder_list)+".bmp")
                    full_thermal_path = os.path.join(thermal_dir, os.path.splitext(Thermal_name[0])[0]+"_{:04d}".format(idx_in_folder_list)+".bmp")
                    self.image_list.append(full_image_path)
                    self.thermal_list.append(full_thermal_path)
                    self.gt.extend([class_index])
                else:
                    for index in idx_in_folder_list:
                        full_image_path = os.path.join(Img_dir, os.path.splitext(Img_name[0])[0]+"_{:04d}".format(index)+".bmp")
                        full_thermal_path = os.path.join(thermal_dir, os.path.splitext(Thermal_name[0])[0]+"_{:04d}".format(index)+".bmp")
                        self.image_list.append(full_image_path)
                        self.thermal_list.append(full_thermal_path)
                    self.gt.extend([class_index] * len(idx_in_folder_list))


        self.num_image = len(self.image_list)
        assert self.num_image == len(self.gt) and self.num_image == len(self.thermal_list), "Data length error"



        self.soft_label = soft_label
        self.repeat = repeat

        #### config transform
        self.transform_dict = {
            "crop": transforms.Compose([
                transforms.CenterCrop((1080, 1920)),            #### height, width location
            ]),
            "init": transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(resize),
            ]),
            "vis_norm": transforms.Compose([
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
            "fir_norm": transforms.Compose([
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ])
        }



    def __len__(self):
        return len(self.image_list) * self.repeat

    def __getitem__(self, idx):
        idx = idx % self.num_image
        image = Image.open(self.image_list[idx])
        image = self.transform_dict["vis_norm"](self.transform_dict["init"](image))

        thermal = Image.open(self.thermal_list[idx])
        thermal = self.transform_dict["fir_norm"](self.transform_dict["init"](thermal))

        gt = [0]*len(self.class_list)

        gt[self.class_list.index(self.gt[idx])] = 1
        if self.soft_label:
            gt = [0.1/(len(self.class_list)-1)] * len(self.class_list)
            gt[self.class_list.index(self.gt[idx])] = 0.9
        else:
            gt = [0] * len(self.class_list)
            gt[self.class_list.index(self.gt[idx])] = 1




        # return {'input' : image,
        return {'input' : torch.cat((thermal, image), dim=0),
                'label' : torch.tensor(gt)}


class Mobticon_dataloader(Dataset):
    def __init__(self, image_dir, json_dir, mode, class_info, resize=(512,380), soft_label = False, repeat=1):
        #### json config
        with open(json_dir) as json_file:
            json_data = json.load(json_file)

        for class_idx in class_info[1]:     #### except
            json_data["TRAIN"].pop(class_idx)
            json_data["TEST"].pop(class_idx)

        assert len(class_info[2]) < 2, "OOD_class must be one or zero"         #### OOD class error

        #### same class
        assert len(class_info[0]) > 1 or len(class_info[0]) == 0, "same class args must higher than 2"
        for idx, class_idx in enumerate(class_info[0]):
            if idx == 0:
                first_idx = class_idx
                continue
            tmp_list = json_data["TRAIN"].pop(class_idx)
            append_value(json_data["TRAIN"], first_idx, tmp_list)
            tmp_list = json_data["TEST"].pop(class_idx)
            append_value(json_data["TEST"], first_idx, tmp_list)


        if mode == "train" or mode == "test":
            json_data = json_data["TRAIN"] if mode =="train" else json_data["TEST"]
            for class_idx in class_info[2]:     #### OOD
                json_data.pop(class_idx)

        elif mode == "OOD":
            for class_idx in class_info[2]:
                tmp_list = json_data["TRAIN"][class_idx]
                tmp_list.extend(json_data["TEST"][class_idx])
                json_data=dict()
                json_data[class_idx] = tmp_list


        self.class_list = sorted(list(json_data.keys()))              #### key -> gt

        self.num_per_class = dict()
        for class_index in self.class_list:
            self.num_per_class[class_index] = len(json_data[class_index])
        print("number of each class", end="\t")
        print(self.num_per_class)

        #### listing
        self.gt = []
        self.image_list = []
        self.thermal_list = []
        self.ROI_list = []
        self.HT_list = []
        for class_index in self.class_list:
            thermal_dir = [os.path.join(image_dir, y, "Thermal") for y in json_data[class_index]]
            HT_dir = [os.path.join(image_dir, y, "HT") for y in json_data[class_index]]
            Img_dir = [os.path.join(image_dir, y, "Img") for y in json_data[class_index]]
            for dir in Img_dir:
                self.image_list.extend([os.path.join(dir, y) for y in os.listdir(dir) if ".bmp" in y])
            for dir in thermal_dir:
                self.thermal_list.extend([os.path.join(dir, y) for y in os.listdir(dir) if ".bmp" in y])
                self.ROI_list.extend([os.path.join(dir, y) for y in os.listdir(dir) if ".txt" in y])
            for dir in HT_dir:
                self.HT_list.extend([os.path.join(dir, y) for y in os.listdir(dir) if ".txt" in y])

            self.gt.extend([class_index] * len(Img_dir))

        self.num_image = len(self.image_list)
        assert self.num_image == len(self.gt) and self.num_image == len(self.thermal_list), "Data length error"



        self.soft_label = soft_label
        self.repeat = repeat

        #### config transform
        self.transform_dict = {
            "crop": transforms.Compose([
                transforms.CenterCrop((1080, 1920)),            #### height, width location
            ]),
            "init": transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(resize),
            ]),
            "vis_norm": transforms.Compose([
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
            "fir_norm": transforms.Compose([
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ])
        }



    def __len__(self):
        return len(self.image_list) * self.repeat

    def __getitem__(self, idx):
        idx = idx % self.num_image
        image = Image.open(self.image_list[idx])
        image = self.transform_dict["vis_norm"](self.transform_dict["init"](image))

        thermal = Image.open(self.thermal_list[idx])
        thermal = self.transform_dict["fir_norm"](self.transform_dict["init"](thermal))

        gt = [0]*len(self.class_list)

        gt[self.class_list.index(self.gt[idx])] = 1
        if self.soft_label:
            gt = [0.1/(len(self.class_list)-1)] * len(self.class_list)
            gt[self.class_list.index(self.gt[idx])] = 0.9
        else:
            gt = [0] * len(self.class_list)
            gt[self.class_list.index(self.gt[idx])] = 1




        # return {'input' : image,
        return {'input' : torch.cat((thermal, image), dim=0),
                'label' : torch.tensor(gt)}

class Mobticon_dataloader_before(Dataset):
    def __init__(self, image_dir, num_class, mode, soft_label = False, repeat=1):
        self.soft_label = soft_label
        self.image_dir = os.path.join(image_dir, mode)
        self.class_list = sorted(os.listdir(self.image_dir))
        # assert len(self.class_list) == num_class, "num_class is not equal to dataset's class number"

        self.image_fol_list = []
        self.image_list = []
        self.FIR_image_list = []
        self.gt_list = []
        self.len_list = []
        self.repeat = repeat

        #### config transform
        self.transform_dict = {
            "crop": transforms.Compose([
                transforms.CenterCrop((1080, 1920)),            #### height, width location
            ]),
            "init": transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((512, 320)),
            ]),
            "vis_norm": transforms.Compose([
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
            "fir_norm": transforms.Compose([
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ])
        }

        self.no_thermal_image = (np.ones((512,320)) * 155)


        for gt_idx, fol in enumerate(self.class_list):
            print("label {} : folder {}".format(gt_idx, fol), end="\t")
            locals()[fol] = os.path.join(self.image_dir, fol)
            self.image_fol_list.extend(listdir_fullpath(locals()[fol]))
            # self.gt_list.extend([i]*len(os.listdir(locals()[fol])))

            #### sub_folder(ex BI001) in folder(BI)
            for sub_fol in self.image_fol_list:
                Visual_path = os.path.join(sub_fol, "Img")
                vis_file_list = str_mapping(Visual_path, ".bmp")

                Thermal_path = os.path.join(sub_fol, "Thermal_stat")
                #### check there exists Thermal image
                if Path(Thermal_path).exists():
                    #### exception handling
                    fir_file_list = str_mapping(Thermal_path, ".bmp")
                    fir_timeinfo_list = [extract_time_info(x) for x in fir_file_list]

                    #### check time information between image and thermal camera
                    for file in vis_file_list:
                        vis_timeinfo = extract_time_info(file)
                        try:            #### time informations are equal or not
                            index_val = fir_timeinfo_list.index(vis_timeinfo)
                        except ValueError:
                            index_val = -1
                        self.image_list.append(os.path.join(sub_fol, "Img", file))
                        # np_image = np.asarray(Image.open(os.path.join(sub_fol, "Img", file)).convert("RGB"))
                        # self.image_list.append(self.transform_dict["init"](np_image))
                        self.gt_list.extend([gt_idx])

                        if index_val != -1:
                            self.FIR_image_list.append(os.path.join(sub_fol, "Thermal_stat", fir_file_list[index_val]))
                            # tmp_fir_image = Image.open(os.path.join(sub_fol, "Thermal_stat", fir_file_list[index_val]))
                            # np_thermal = np.asarray(ImageOps.grayscale(tmp_fir_image))
                            # self.FIR_image_list.append(self.transform_dict["init"](np_thermal))
                        else:
                            self.FIR_image_list.append(-1)
                            # self.FIR_image_list.append(self.transform_dict["init"](no_thermal_image))

                else:
                    for file in vis_file_list:
                        # np_image = np.asarray(Image.open(os.path.join(sub_fol, "Img", file)).convert("RGB"))
                        # self.image_list.append(self.transform_dict["init"](np_image))
                        self.image_list.append(os.path.join(sub_fol, "Img", file))
                        self.gt_list.extend([gt_idx])
                        self.FIR_image_list.append(-1)
                        # self.FIR_image_list.append(self.transform_dict["init"](no_thermal_image))

        print()


        self.num_image = len(self.image_list)

    def __len__(self):
        return len(self.image_list) * self.repeat

    def __getitem__(self, idx):
        idx = idx % len(self.image_list)
        image = Image.open(self.image_list[idx])
        if self.FIR_image_list[idx] == -1:
            FIR_image = ImageOps.grayscale(Image.fromarray(self.no_thermal_image))
        else:
            FIR_image = ImageOps.grayscale(Image.open(self.FIR_image_list[idx]))

        # image = copy.deepcopy(self.image_list[idx])
        # FIR_image = copy.deepcopy(self.FIR_image_list[idx])
        # image = self.transform_dict["vis_norm"](image)
        # FIR_image = self.transform_dict["fir_norm"](FIR_image)
        if ".bmp" in self.image_list[idx]:
            image = self.transform_dict["crop"](image)
        image = self.transform_dict["vis_norm"](self.transform_dict["init"](image))
        FIR_image = self.transform_dict["fir_norm"](self.transform_dict["init"](FIR_image))


        if self.soft_label:
            gt = [0.1/(len(self.class_list)-1)] * len(self.class_list)
            gt[self.gt_list[idx]] = 0.9
        else:
            gt = [0] * len(self.class_list)
            gt[self.gt_list[idx]] = 1




        return {'input' : image,
        # return {'input' : torch.cat((FIR_image, image), dim=0),
                'label' : torch.tensor(gt)}



class MNIST_manual(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.folder_list = os.listdir(self.image_dir)

        self.image_list = []
        self.gt_list = []

        for gt in self.folder_list:
            locals()[str(gt)] = os.path.join(image_dir, str(gt))
            self.image_list.extend(listdir_fullpath(locals()[str(gt)]))
            self.gt_list.extend([gt] * len(os.listdir(locals()[str(gt)])))

        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img = self.preprocess(Image.open(self.image_list[idx]))

        return img, torch.tensor(int(self.gt_list[idx]))




class Dog_metric_dataloader(Dataset):
    def __init__(self, image_dir, num_class, mode, soft_label = False, min_instance = 2, repeat=1):
        self.soft_label = soft_label
        self.image_dir = os.path.join(image_dir, mode)
        self.class_list = sorted(os.listdir(self.image_dir))
        assert len(self.class_list) == num_class, "num_class is not equal to dataset's class number"

        self.image_fol_list = []
        self.image_list = []
        self.gt_list = []
        self.len_list = []
        self.repeat = repeat

        for i, fol in enumerate(self.class_list):
            print("label {} : folder {}".format(i, fol), end="\t")
            locals()[fol] = os.path.join(self.image_dir, fol)
            self.image_fol_list.extend(listdir_fullpath(locals()[fol]))
            self.gt_list.extend([i]*len(os.listdir(locals()[fol])))
            self.len_list.append(len(os.listdir(locals()[fol])))


        for file in self.image_fol_list:
            self.image_list.append([np.asarray(Image.open(file).convert("RGB"))])

        print()

        if mode == "train":
            self.preprocess = transforms.Compose([
                transforms.Resize((160,160)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.preprocess = transforms.Compose([
                transforms.Resize((160, 160)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])


        self.num_image = len(self.image_list)

    def __len__(self):
        return len(self.image_list) * self.repeat

    def __getitem__(self, idx):
        idx = idx % len(self.image_list)
        img = self.preprocess(Image.fromarray(self.image_list[idx][0], 'RGB'))

        if self.soft_label:
            gt = [0.1/(len(self.class_list)-1)] * len(self.class_list)
            gt[self.gt_list[idx]] = 0.9
        else:
            gt = [0] * len(self.class_list)
            gt[self.gt_list[idx]] = 1


        return {'input' : img,
                'label' : torch.tensor(gt)}


class Dog_dataloader(Dataset):
    def __init__(self, image_dir, num_class, mode, repeat=1):
        self.image_dir = os.path.join(image_dir, mode)
        self.class_list = sorted(os.listdir(self.image_dir))
        assert len(self.class_list) == num_class, "num_class is not equal to dataset's class number"

        self.image_list = []
        self.gt_list = []
        self.len_list = []
        self.repeat = repeat

        for i, fol in enumerate(self.class_list):
            print("label {} : folder {}".format(i, fol), end="\t")
            locals()[fol] = os.path.join(self.image_dir, fol)
            self.image_list.extend(listdir_fullpath(locals()[fol]))
            self.gt_list.extend([i]*len(os.listdir(locals()[fol])))
            self.len_list.append(len(os.listdir(locals()[fol])))
        print()
        if mode == "train":
            self.preprocess = transforms.Compose([
                transforms.Resize((160,160)),
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomAffine(degrees=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.preprocess = transforms.Compose([
                transforms.Resize((160, 160)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])


        self.num_image = len(self.image_list)

    def __len__(self):
        return len(self.image_list) * self.repeat

    def __getitem__(self, idx):
        idx = idx % len(self.image_list)

        img = Image.open(self.image_list[idx]).convert("RGB")
        img = self.preprocess(img)
        gt = [0] * len(self.class_list)
        gt[self.gt_list[idx]] = 1

        return {'input' : img,
                'label' : torch.tensor(gt)}



class modified_Dog_dataloader(Dataset):
    def __init__(self, image_dir, num_class, mode):
        self.image_dir = os.path.join(image_dir, mode)
        self.class_list = sorted(os.listdir(self.image_dir))
        assert len(self.class_list) == num_class, "num_class is not equal to dataset's class number"

        self.image_list = []
        self.gt_list = []
        self.len_list = []
        for i, fol in enumerate(self.class_list):
            print("label {} : folder {}".format(i, fol), end="\t")
            locals()[fol] = os.path.join(self.image_dir, fol)
            tmp = 0
            for each_fol in os.listdir(locals()[fol]):
                self.image_list.extend(listdir_fullpath(os.path.join(locals()[fol], each_fol)))
                self.gt_list.extend([i]*len(os.listdir(os.path.join(locals()[fol], each_fol))))
                tmp += len(os.listdir(os.path.join(locals()[fol], each_fol)))
            self.len_list.append(tmp)
        print()

        if mode == "train":
            self.preprocess = transforms.Compose([
                transforms.Resize((160,160)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.preprocess = transforms.Compose([
                transforms.Resize((160, 160)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        self.num_image = len(self.image_list)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img = Image.open(self.image_list[idx])
        img = self.preprocess(img)
        gt = [0] * len(self.class_list)
        gt[self.gt_list[idx]] = 1
        return img, torch.tensor(gt)


class customSampler(Sampler):
    def __init__(self, data_source, batch_size, num_instances):
        super().__init__(data_source)
        assert batch_size % num_instances == 0, "batch_size cannot divided by num_instances"

        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_gts_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)

        for idx in range(len(self.data_source)):
            self.index_dic[int(torch.argmax(self.data_source[idx]['label']))].extend([idx])

        self.gts = list(self.index_dic.keys())                      # all labels name

        self.length = 0
        for gt in self.gts:
            idxs = self.index_dic[gt]                               # return every indexes of gt
            num = len(idxs)                                         # # of indexes of gt
            if num < self.num_instances:
                num = self.num_instances                            # if num >= self.num_instances,
            self.length += num - num % self.num_instances           # self.length += overflow number


    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for gt in self.gts:
            idxs = copy.deepcopy(self.index_dic[gt])
            if len(idxs) < self.num_instances:                      # if idxs' # is under the num_instances
                idxs = np.random.choice(idxs, size = self.num_instances, replace = True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:           # if batch_idxs reach num_instances, save idx and gt
                    batch_idxs_dict[gt].append(batch_idxs)
                    batch_idxs = []

        avai_gts = copy.deepcopy(self.gts)
        selected_gts = copy.deepcopy(self.gts)
        final_idxs = []
        batch_idxs = []
        flag = 0                                                    # break flag
        while True:
            count = 0

            for gt in selected_gts:
                tmp_batch_idxs = batch_idxs_dict[gt].pop(0)
                batch_idxs.extend(tmp_batch_idxs)

            if len(batch_idxs) == self.batch_size:
                final_idxs.append(batch_idxs)
                batch_idxs = []

            for key, value in batch_idxs_dict.items():
                if isinstance(value, list):
                    count += len(value)
                    if len(value) == 0:
                        flag += 1
                if count < self.batch_size:
                    flag += 1
            if flag != 0:
                break


            iter_num = (self.batch_size - len(batch_idxs))//self.num_instances
            iter_num = len(self.gts) if iter_num > len(self.gts) else iter_num
            selected_gts = random.sample(avai_gts, iter_num)

        self.length = len(final_idxs)


        return iter(final_idxs)

    def __len__(self):
        return self.length


# class Mobticon_dataloader(Dataset):
#     def __init__(self, image_dir, condition_list, version = "capture", mode = "train"):
#         self.condition_num = len(condition_list)
#
#         self.image_list = []
#         self.gt_list = []
#         for i, condition in enumerate(condition_list):
#             locals()[condition] = os.path.join(image_dir, condition, version, mode)
#             self.image_list.extend(listdir_fullpath(locals()[condition]))
#             self.gt_list.extend([i]*len(os.listdir(locals()[condition])))
#
#         self.preprocess = transforms.Compose([
#             transforms.Resize((512, 320)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ])
#
#         self.num_image = len(self.image_list)
#
#     def __len__(self):
#         return len(self.image_list)
#
#     def __getitem__(self, idx):
#         img = Image.open(self.image_list[idx])
#         img = self.preprocess(img)
#         gt = [0] * self.condition_num
#         gt[self.gt_list[idx]] = 1
#         return img, torch.tensor(gt)



class MNIST(Dataset):
    def __init__(self, imagedir, mode):
        self.imgdir = os.path.join(imagedir, mode)
        self.gtlist = sorted(os.listdir(self.imgdir))
        self.imglist = []
        for fol in self.gtlist:
            self.imglist.extend(sorted(listdir_fullpath(os.path.join(self.imgdir, fol))))
        self.gt = []
        for file in self.imglist:
            self.gt.append(file[file[:file.rfind("/")].rfind("/")+1:file.rfind("/")])
        self.len = len(self.imglist)
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
        ])
    def __len__(self):
        return len(self.imglist)

    def get_gtlist(self):
        return self.gtlist

    def __getitem__(self, idx):
        img = Image.open(self.imglist[idx]).convert("L")
        img = self.preprocess(img)
        return img, self.gt[idx]



class cifar10_dataloader(Dataset):
    def __init__(self, imagedir, mode):
        self.imgdir = os.path.join(imagedir, mode)
        self.imglist = sorted(os.listdir(self.imgdir))
        self.gtlist = []
        if 'train' in mode:
            self.preprocess = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            self.preprocess = transforms.Compose([
                transforms.ToTensor(),
            ])

        f = open(os.path.join(imagedir, "labels.txt"), "r")
        while True:
            line = f.readline()
            if not line: break
            self.gtlist.append(line[:-1])
        f.close()

    def get_gtlist(self):
        return self.gtlist

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, idx):
        gt = []
        for gt_name in self.gtlist:
            if gt_name in self.imglist[idx]:
                gt = self.gtlist.index(gt_name)
                break
        img = Image.open(os.path.join(self.imgdir, self.imglist[idx])).convert("RGB")
        img = self.preprocess(img)
        return img, gt

class MVTec_dataloader(Dataset):
    def __init__(self, image_dir, mode, in_size):
        self.gtlist = sorted(os.listdir(image_dir))
        self.mode = mode
        self.imglist = []
        self.gt = []

        if mode == "train":
            mode += "/good"
            for gt in self.gtlist:
                self.gt.extend(self.gtlist * len(os.path.join(image_dir, gt, mode)))
                self.imglist.extend(sorted(listdir_fullpath(os.path.join(image_dir, gt, mode))))
        elif self.mode == "train_one":
            self.gt = image_dir[image_dir.rfind("/"):]
            self.imglist.extend(sorted(listdir_fullpath(os.path.join(image_dir, "train/good"))))
        elif "test" in self.mode:
            self.test_list = os.listdir(os.path.join(image_dir, self.mode))
            print(self.test_list)
            for list in self.test_list:
                self.imglist.extend(sorted(listdir_fullpath(os.path.join(image_dir, self.mode, list))))
                self.gt.extend(list*len(sorted(listdir_fullpath(os.path.join(image_dir, self.mode, list)))))

        if "train" in mode:
            self.preprocess = transforms.Compose([
                transforms.Resize([in_size,in_size]),
                # transforms.Pad(8,padding_mode="symmetric"),
                # transforms.RandomCrop((in_size,in_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5,0.5,0.5], std = [0.5, 0.5, 0.5]),
            ])
        else:
            self.preprocess = transforms.Compose([
                transforms.Resize([in_size,in_size]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5,0.5,0.5], std = [0.5, 0.5, 0.5]),
            ])

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, idx):
        img = self.preprocess(Image.open(self.imglist[idx]).convert("RGB"))
        gt = self.gt if self.mode == "train_one" else self.gt[idx]
        return img, gt


class cifar_anomaly(Dataset):
    def __init__(self, imagedir, mode):
        self.imgdir = imagedir
        self.imglist = sorted(os.listdir(self.imgdir))
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
        ])
        if 'normal' == mode:
            self.gt = "normal"
        else:
            self.gt = "abnormal"

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.imgdir, self.imglist[idx])).convert("RGB")
        img = self.preprocess(img)
        return img, self.gt



# if __name__ == '__main__':
#     a = cifar100_dataloader(imagedir="/media/seonghun/data1/CIFAR100", mode="fine/train", anomally=None)
#     trainloader = DataLoader(a,
#         batch_size=512, shuffle=True, num_workers=2)
#
#     print(b)

