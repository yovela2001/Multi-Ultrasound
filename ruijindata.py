# -*- coding:utf-8 -*-

from PIL import Image
import torch
import os
import sys
from torch.utils.data import Dataset
from torchvision import transforms
import xml.etree.cElementTree as ET
from tqdm import tqdm
import random

import pandas as pd
from random import randint, sample, shuffle
import av
import numpy as np
# import cv2
# import selectivesearch


class RuijinData(Dataset):
    def __init__(self, root, pre_transform=None, sub_list=[0, 1, 2, 3, 4], task='BM', modality='image', lump=0, train=1):
        self.root = root
        self.task = task
        self.pre_transform = pre_transform
        self.sub_list = sub_list
        self.label_table = pd.read_csv(
            os.path.join(self.root, 'final_label_1.csv'))
        self.bm_patient_info = []
        self.alnm_patient_info = []
        self.modality = modality
        self.lump = lump
        self.train = train
        if self.modality not in ['video', 'image', 'multi', 'multi1', 'multi_importance','deit_attn1']:
            raise NotImplementedError("Modality Error")
        if self.task not in ['BM', 'ALNM']:
            raise NotImplementedError('Task Error')
        for fold in self.sub_list:
            self.scan(fold)

    def scan(self, fold):
        fold_table = self.label_table[self.label_table['{}_{}fold'.format(
            self.task, 'lump_' if self.lump == 1 else '')] == fold].reset_index(drop=True)
        for k in range(len(fold_table)):
            id = fold_table.loc[k, 'ID']
            p_path = os.path.join(self.root, str(id).zfill(9))
            now_patient = {}
            p_label = fold_table.loc[k, self.task]
            now_patient['label'] = p_label
            filesOfPatient = os.listdir(p_path)
            now_patient['id'] = id
            now_patient['video_root'] = [os.path.join(
                p_path, x) for x in filesOfPatient if x.endswith('.mp4')]
            now_patient['img_root'] = [os.path.join(p_path, x) for x in filesOfPatient if x.endswith(
                '.JPG') or x.endswith('.jpg') or x.endswith('.bmp') or x.endswith('.BMP')]  # .jpg should also be read
            if self.task == 'BM':
                self.bm_patient_info.append(now_patient)
            else:
                self.alnm_patient_info.append(now_patient)

    def __getitem__(self, index):
        now_patient = self.bm_patient_info[index] if self.task == 'BM' else self.alnm_patient_info[index]
        label = now_patient['label']
        label = torch.tensor(label)  # tensor type

        video = self.get_tensor(index, modality='video')
        img = self.get_tensor(index, modality = 'image')
        return img, video, label, now_patient

    def get_images(self, index, is_multi_thread_decode=True, frame_num=16, modality='image'):
        now_patient = self.bm_patient_info[index] if self.task == 'BM' else self.alnm_patient_info[index]

        video_path = now_patient['video_root'][0]
        if not os.access(video_path, os.F_OK):
            print('测试文件不存在')
            return

        container = av.open(video_path)
        if is_multi_thread_decode:
            container.streams.video[0].thread_type = "AUTO"

        container.seek(0, any_frame=False, backward=True,
                        stream=container.streams.video[0])

        frames = []
        for frame in container.decode(video=0):
            frames.append(frame)
        container.close()

        # turn frames to image
        result_frames = [frame.to_rgb().to_image() for frame in frames]
        result_frames = random.sample(result_frames, 3)
        result_frames = [self.pre_transform(
            frame) for frame in result_frames]

        return result_frames


    def get_tensor(self, index, is_multi_thread_decode=True, frame_num=16, modality='image'):
        now_patient = self.bm_patient_info[index] if self.task == 'BM' else self.alnm_patient_info[index]

        if modality == 'image':

            image_path = now_patient['img_root']
            imgs = []
            for img_path in image_path:
                img = Image.open(img_path).convert('RGB')
                img = self.pre_transform(img)
                imgs.append(img)

            imgs = torch.stack([x for x in imgs], dim=0)

            return imgs

        elif modality == 'video':

            video_path = now_patient['video_root'][0]
            if not os.access(video_path, os.F_OK):
                print('测试文件不存在')
                return

            container = av.open(video_path)
            if is_multi_thread_decode:
                container.streams.video[0].thread_type = "AUTO"

            container.seek(0, any_frame=False, backward=True,
                           stream=container.streams.video[0])

            frames = []
            for frame in container.decode(video=0):
                frames.append(frame)
            container.close()

            # turn frames to image
            result_frames = [frame.to_rgb().to_image() for frame in frames]
            result_frames = random.sample(result_frames, 16)
            result_frames = [self.pre_transform(
                frame) for frame in result_frames]
            result_frames = torch.stack([x for x in result_frames], dim=0)

            return result_frames

        else:
            print('Modality Error')
            return 0

    def get_shuffled_imgs(self, index, is_multi_thread_decode=True, frame_num=16):
        now_patient = self.bm_patient_info[index] if self.task == 'BM' else self.alnm_patient_info[index]
        image_path = now_patient['img_root']
        imgs = []
        for img_path in image_path:
            img = Image.open(img_path).convert('RGB')
            img = self.pre_transform(img)
            imgs.append(img)

        video_path = now_patient['video_root'][0]
        if not os.access(video_path, os.F_OK):
            print('测试文件不存在')
            return

        container = av.open(video_path)
        if is_multi_thread_decode:
            container.streams.video[0].thread_type = "AUTO"

        container.seek(0, any_frame=False, backward=True,
                       stream=container.streams.video[0])

        frames = []
        for frame in container.decode(video=0):
            frames.append(frame)
        container.close()

        # turn frames to image
        result_frames = [frame.to_rgb().to_image() for frame in frames]
        result_frames = random.sample(result_frames, 16)
        result_frames = [self.pre_transform(frame) for frame in result_frames]
        imgs = imgs + result_frames

        importance_label = [1 for i in range(len(imgs))]
        importance_label[-frame_num:] = [0 for i in range(frame_num)]

        shuffle = random.sample(range(len(imgs)), len(imgs))
        importance_label = [importance_label[i] for i in shuffle]
        imgs = [imgs[i] for i in shuffle]
        result = torch.stack([x for x in imgs], dim=0)

        importance_label = torch.tensor(importance_label)

        return result, importance_label

    def __len__(self):
        if self.task == 'BM':
            return len(self.bm_patient_info)
        else:
            return len(self.alnm_patient_info)

    def visualize_item(self, index):
        now_patient = self.bm_patient_info[index] if self.task=='BM' else self.alnm_patient_info[index]
        label = now_patient['label']
        label = torch.tensor(label)
        imgs = []
        for img_path in now_patient['img_root']:
            img = Image.open(img_path).convert('RGB')
            imgs.append(img)
        video_path = now_patient['video_root'][0]
        container = av.open(video_path)
        frames = []
        for frame in container.decode(video=0):
            frames.append(frame)
        container.close()
        result_frames = [frame.to_rgb().to_image() for frame in frames]
        video_frames = random.sample(result_frames,16)
        return imgs,video_frames



if __name__ == '__main__':
    pre_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Resize((224,224)),
    ])
    # pre_transform = None
    dataset = RuijinData(root='/remote-home/share/RJ_video/RJ_video_crop',
                         pre_transform=pre_transform, task='ALNM', modality='video')
    min_len = 100000
    for i in range(len(dataset)):
        length = dataset[i][0].shape[0]
        if length < min_len:
            min_len = length
        print(i)
    print(min_len)
    # print(dataset[0][0].shape)

    pre_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
    ])
    dataset = RuijinData(root='/remote-home/share/RJ_video/RJ_video_crop',
                         pre_transform=pre_transform, task='ALNM', modality='image')
    print(dataset[1][0].shape)
    print(len(dataset))
