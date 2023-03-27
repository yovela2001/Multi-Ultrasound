from cgi import test
from PIL import Image
from PIL import ImageDraw, ImageFont
import torch
import os
import sys
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import xml.etree.cElementTree as ET
from tqdm import tqdm
import random

import pandas as pd
from random import randint,sample
import av
import numpy as np
from models.DeiT.DeiT import deit_tiny_distilled_patch16_224, deit_tiny_patch16_224, deit_base_distilled_patch16_224, deit_base_patch16_224


class DeiT_Attn2(nn.Module):
    def __init__(self, num_class=2, pretrained = True, deit_size = 'tiny', attention = True, modality = 'video'):
        super(DeiT_Attn2, self).__init__()
        print('DeiT_Attn2 created.')
        self.videobackbone = deit_tiny_distilled_patch16_224(pretrained=pretrained)
        self.imgbackbone = deit_tiny_distilled_patch16_224(pretrained=pretrained)
        print('The size is tiny.')

        self.video_query = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, 64)
        )

        self.classifier = nn.Sequential(
            nn.Linear(2000, 2)
        )

        self.img_key = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, 64)
        )


    # concatenate forward
    def forward(self, img, video, debug=False):
        if debug:
            print('input', video.shape)
            # [1, 16, 3, 224, 224]
        # batch=1 时 
        # img = img.squeeze(0)
        # video = video.squeeze(0)
        
        video_v, video_q = self.videobackbone(video)
        # ([16, 1000],[16,1000])

        img, img_k = self.imgbackbone(img)

        # Q = video_q
        # K = img_k
        Q = self.video_query(video_q)
        K = self.img_key(img_k)

        Q = torch.transpose(Q, 1, 0)
        A = torch.mm(K, Q)
        A = A.mean(axis=0,keepdim=True)
        A = torch.softmax(A,1)
        if debug: print(A, A.size())

        V = torch.mm(A,video_v)
        if debug: print(V.size())
        img = img.mean(axis=0,keepdim=True)
        V = torch.cat([img,V],dim=1)
        prob = self.classifier(V)
        
        return prob, A.mean(axis=0)

class RuijinData(Dataset):
    def __init__(self, root, pre_transform = None, sub_list = [0,1,2,3,4], task = 'BM', modality = 'image', lump = 0, train = 1, video_transform = None, image_transform = None):
        self.root = root
        self.task = task
        self.pre_transform = pre_transform
        self.video_transform = video_transform
        self.image_transform = image_transform
        self.sub_list = sub_list
        self.label_table = pd.read_csv(os.path.join(self.root, 'final_label_1.csv'))
        self.bm_patient_info = []
        self.alnm_patient_info = []
        self.modality = modality
        self.lump = lump
        self.train = train
        if self.modality not in ['video','video1', 'image','multi','multi1','deit_video','deit_multi']:
            raise NotImplementedError("Modality Error")
        if self.task not in ['BM', 'ALNM']:
            raise NotImplementedError('Task Error')
        for fold in self.sub_list:
            self.scan(fold)

    def scan(self, fold):
        # print('lump:',self.lump)
        # get id
        fold_table = self.label_table[self.label_table['{}_{}fold'.format(self.task,'lump_' if self.lump==1 else '')]==fold].reset_index(drop=True)
        for k in range(len(fold_table)):
            id = fold_table.loc[k,'ID']
            p_path = os.path.join(self.root,str(id).zfill(9))
            now_patient = {}
            p_label = fold_table.loc[k,self.task]
            now_patient['label'] = p_label
            filesOfPatient = os.listdir(p_path)
            now_patient['id'] = id
            now_patient['video_root'] = [os.path.join(p_path, x) for x in filesOfPatient if x.endswith('.mp4')]
            now_patient['img_root'] = [os.path.join(p_path, x) for x in filesOfPatient if x.endswith('.JPG') or x.endswith('.jpg') or x.endswith('.bmp') or x.endswith('.BMP')]  # .jpg should also be read
            if self.task =='BM':
                self.bm_patient_info.append(now_patient)
            else:
                self.alnm_patient_info.append(now_patient)
   
    def __getitem__(self, index):
        now_patient = self.bm_patient_info[index] if self.task=='BM' else self.alnm_patient_info[index]
        label = now_patient['label']
        label = torch.tensor(label)  # tensor type

        if self.modality == 'multi1' or self.modality == 'deit_multi':
            # 将视频与图像stack在一起 modality = 'multi1'
            imgs = []
            for img_path in now_patient['img_root']:
                img = Image.open(img_path).convert('RGB')
                if self.image_transform is not None:
                    img = self.image_transform(img)
                imgs.append(img)
            video = self.get_tensor_from_video(now_patient['video_root'][0])
            video = video.permute(1,0,2,3)
            img = torch.stack([x for x in imgs], dim=0)
            multi_result = torch.cat([img, video],dim=0)
            return multi_result, label, now_patient['id']
    
    def get_tensor_from_video(self, video_path,is_multi_thread_decode = True, frame_num = 16):
        """
        :param video_path: 视频文件地址
        :param is_multi_thread_decode: 是否多线程解码文件
        :return: pytorch tensor
        """
        if not os.access(video_path, os.F_OK):
            print('测试文件不存在')
            return

        container = av.open(video_path)
        if is_multi_thread_decode:
            container.streams.video[0].thread_type = "AUTO"

        container.seek(0, any_frame=False, backward=True, stream=container.streams.video[0])

        frames = []
        for frame in container.decode(video=0):
            frames.append(frame)
        container.close()
        # result_frams = None

        # 从视频帧转换为ndarray
        result_frames = [frame.to_rgb().to_ndarray() for frame in frames]
        # 转换成tensor
        result_frames = np.stack(result_frames)
        # 注意：此时result_frames组成的维度为[视频帧数量，高，宽，通道数]
        # result_frames = torch.Tensor(np.transpose(result_frames,(0,3,1,2)))
        sampling_indices = np.random.choice(range(result_frames.shape[0]),frame_num,replace=False)
        sampling_indices = np.sort(sampling_indices)
        result_frames = result_frames[sampling_indices]

        if self.video_transform is not None:
            # out_frames = torch.stack((result_frames[0],))
            # for i in range(1, result_frames.shape[0]):
            #     # frame = self.pre_transform(result_frames[i])
            #     out_frames = torch.cat((out_frames, frame.unsqueeze(0)))
            out_frames = torch.ones_like(torch.Tensor(np.transpose(result_frames,(0,3,1,2))))
            for i in range(1, result_frames.shape[0]):
                out_frames[i] = transforms.ToTensor()(result_frames[i])

            out_frames = self.video_transform(out_frames)
            out_frames = out_frames.permute((1,0,2,3))#CTHW

        return out_frames
    
    def get_multi_tensor(self, index, is_multi_thread_decode = True, frame_num = 16):
        now_patient = self.bm_patient_info[index] if self.task=='BM' else self.alnm_patient_info[index]
        image_path = now_patient['img_root']
        video_path = now_patient['video_root'][0]
        imgs = []
        for img_path in image_path:
            img = Image.open(img_path).convert('RGB')
            if self.image_transform is not None:
                img = self.image_transform(img)
            imgs.append(img)

        imgs = torch.stack([x for x in imgs], dim=0)

        if not os.access(video_path, os.F_OK):
            print('测试文件不存在')
            return

        container = av.open(video_path)
        if is_multi_thread_decode:
            container.streams.video[0].thread_type = "AUTO"

        container.seek(0, any_frame=False, backward=True, stream=container.streams.video[0])

        frames = []
        for frame in container.decode(video=0):
            frames.append(frame)
        container.close()
        # result_frams = None

        result_frames = [frame.to_rgb().to_image() for frame in frames]
        # print('video frame type',type(result_frames[0]))
        result_frames = random.sample(result_frames,16)
        result_frames = [self.image_transform(frame) for frame in result_frames]
        # 转换成tensor
        result_frames = torch.stack([x for x in result_frames], dim=0)
        
        return imgs, result_frames

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

def similarity_calculate(video, imgs):
    pool = nn.AvgPool2d(8)
    imgs = pool(imgs)
    video = pool(video)
    video = video.flatten(1,3)
    imgs = imgs.flatten(1,3)
    results = []
    for frame in video.split(1):
        similarity = []
        for img in imgs.split(1):
            similarity.append(torch.cosine_similarity(frame,img))
        results.append(max(similarity))
#     imgs = imgs.transpose(1,0)
#     similarity = torch.mm(video,imgs)
#     results = similarity.max(dim=1)[1]
    results = torch.tensor(results)
    # results = results*10
    # return nn.Softmax(dim=0)(results)
    return(results)

def display_similarity(testbag, index, model):
    '''with the index of the dataloader , return the consine similarity between video modality and image modality'''
    imgs_v,video_frames_v = testbag.visualize_item(index)
    imgs,video_frames = testbag.get_multi_tensor(index)
    # similarity = similarity_calculate(video_frames,imgs) 
    # similarity = similarity.numpy().tolist()
    # print(similarity)
    _, A = model(imgs,video_frames)
    print(A)
    # pool = nn.AvgPool2d(4)
    # imgs = pool(imgs)
    # video_frames = pool(video_frames)
    PIL_imgs = []
    PIL_frames = []
    for img in imgs_v:
        PIL_imgs.append(np.array(img))
    for i,frame in enumerate(video_frames_v):
        draw = ImageDraw.Draw(frame)
        text_color=(255, 0, 0)
        font = ImageFont.truetype('ARIAL.TTF', 130)
        draw.text((0,0), '%.4f'%(A[i]), text_color, font=font)
        PIL_frames.append(np.array(frame))

    video = np.vstack((np.hstack(PIL_frames[0:4]),np.hstack(PIL_frames[4:8]),np.hstack(PIL_frames[8:12]),np.hstack(PIL_frames[12:16])))
    # video_similarity = np.vstack((np.hstack(PIL_similarity[0:4]),np.hstack(PIL_similarity[4:8]),np.hstack(PIL_similarity[8:12]),np.hstack(PIL_similarity[12:16])))
    # 如经历过to tensor 需乘255
    Image.fromarray((np.hstack(PIL_imgs)).astype(np.uint8)).save('./result/imgs/img_{}.jpg'.format(index))
    Image.fromarray((video).astype(np.uint8)).save('./result/imgs/video_{}.jpg'.format(index))



if __name__ == '__main__':
    test_img_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
    test_video_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                # transforms.ToTensor()
            ])
    testbag = RuijinData('/remote-home/share/RJ_video_crop', image_transform=test_img_transform, video_transform=test_video_transform, 
    sub_list=[0], task='BM', modality='multi', lump=1)

    # Load pretrained model
    model = DeiT_Attn2()
    pretrained = '/remote-home/share/RJ_video/exps/BM_DeiTT_multi1_attn2_0918_0/ckp/net.ckpt29.pth'
    pretrained = torch.load(pretrained)
    pretrained_state = pretrained['net']
    model_state = model.state_dict()
    for key in model_state.keys():
        model_state[key] = pretrained_state[key]
            
    # pretrained_model_state, pretrained_parameters = config_object.trainer.net.get_pretrained_state_dict()
    model.load_state_dict(model_state)

    for i in range(5):
        display_similarity(testbag,i,model)
    


