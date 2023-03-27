# -*- coding: utf-8 -*-
'''
Copyright (c) 2019 gyfastas
'''
from __future__ import absolute_import
import os
from tqdm import tqdm
import torch.nn.functional as F
import utils.utility as utility
from utils.logger import Logger
# from utils import presets
import argparse
from importlib import import_module
from utils.logger import Logger
from torch.optim import Adam
import torch.optim as optim
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import os
from collections import OrderedDict
from ruijindata import RuijinData
from Trainer.MyTrainer import MyMulti1Trainer, MyMultiTrainer, MyMultiImportanceTrainer
from Trainer.DeiTTrainer import DeiTTrainer, DeiTMultiTrainer
from models.MultiModel import ViT_Multi, ViT_Similarity, ViT_Importance
from models.DeitModel import DeiT_Video, DeiT_Tiny, DeiT_Importance, DeiT_Attn, DeiT_Attn_Pure, DeiT_Attn1, DeiT_Attn2, Baseline_test, DeiT_Attn_cls
from models.DeiT.losses import DistillationLoss, noDistillationLoss

import pandas as pd
import pickle


class Multi1_config(object):
    def __init__(self, log_root, args):
        self.net = DeiT_Attn2()
        self.net = self.net.cuda()
        # print('DeiT_Attn_cls,multi')
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.ColorJitter(brightness = 0.25),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            # transforms.ColorJitter(0.25, 0.25, 0.25, 0.25),
            transforms.ToTensor()
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        if args.different_lr:
            print('different_lr')
            # deit_pretrained_params = self.net.videobackbone.named_parameters() + self.net.imgbackbone.named_parameters()
            # pretrained_params = list(map(lambda x: x[1], list(filter(
            #     lambda kv: kv[0] in self.net.backbone.named_parameters(),
            #     self.net.named_parameters()))))
            # params = list(map(lambda x: x[1], list(filter(
            #     lambda kv: kv[0] not in self.net.backbone.named_parameters(),
            #     self.net.named_parameters()))))
            pretrained_params = list(map(lambda x: x[1], list(filter(
                lambda kv: kv[0] in self.net.videobackbone.named_parameters() or kv[0] in self.net.imgbackbone.named_parameters(),
                self.net.named_parameters()))))
            params = list(map(lambda x: x[1], list(filter(
                lambda kv: kv[0] not in self.net.videobackbone.named_parameters() and kv[0] not in self.net.imgbackbone.named_parameters(),
                self.net.named_parameters()))))

            self.optimizer = Adam([{'params': pretrained_params, 'lr': args.lr/100},
                                   {'params': params, 'lr': args.lr}])
        else:
            self.optimizer = Adam(self.net.parameters(), lr=args.lr)
            
        self.lrsch = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[10, 20, 30, 50], gamma=0.5)
        self.logger = Logger(log_root)
        # K-fold Cross-Validation
        self.trainbag = RuijinData(args.data_root, sub_list=[x for x in [0, 1, 2, 3, 4] if x != args.test_fold], pre_transform=self.train_transform,
                                   task=args.task, modality=args.modality, lump=args.lump)
        print(len(self.trainbag))
        self.testbag = RuijinData(args.data_root, sub_list=[args.test_fold], pre_transform=self.test_transform, 
                                  task=args.task, modality=args.modality, lump=args.lump)
        if args.collate:
            self.train_loader = DataLoader(
                self.trainbag, batch_size=args.batchsize, shuffle=True, num_workers=8, collate_fn=self.collate_fn)
            self.val_loader = DataLoader(
                self.testbag, batch_size=args.batchsize, shuffle=False, num_workers=8, collate_fn=self.collate_fn)
        else:
            self.train_loader = DataLoader(
                self.trainbag, batch_size=args.batchsize, shuffle=True, num_workers=8)
            self.val_loader = DataLoader(
                self.testbag, batch_size=args.batchsize, shuffle=False, num_workers=8)
        self.trainer = MyMulti1Trainer(self.net, self.optimizer, self.lrsch,
                                      None, self.train_loader, self.val_loader, self.logger, 0)
        # self.save_config(args)

    def save_config(self, args):
        config_file = './saved_configs/'+args.log_root+'.txt'
        f = open(config_file, 'a+')
        argDict = args.__dict__
        for arg_key, arg_value in argDict.items():
            f.writelines(arg_key+':'+str(arg_value)+'\n')
        f.close()
        self.logger.auto_backup('./')
        self.logger.backup_files([config_file])

    def collate_fn(self, batch):
        img, video, label, patient = list(zip(*batch))
        return img, video, label, patient

class DeiT_Attn1_config(object):
    '''
    Config for modality == 'DeiT_Attn1'
    also conpatible with imgs & video frames
    dataset output: [bs, n, c, h, w]
    '''
    def __init__(self, log_root, args):
        self.net = DeiT_Attn1(attention=True)
        self.net = self.net.cuda()
        print('DeiT_Attn1')
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.ColorJitter(brightness = 0.25),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            # transforms.ColorJitter(0.25, 0.25, 0.25, 0.25),
            transforms.ToTensor()
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        if args.different_lr:
            print('different_lr')
            # deit_pretrained_params = self.net.videobackbone.named_parameters() + self.net.imgbackbone.named_parameters()
            # pretrained_params = list(map(lambda x: x[1], list(filter(
            #     lambda kv: kv[0] in self.net.backbone.named_parameters(),
            #     self.net.named_parameters()))))
            # params = list(map(lambda x: x[1], list(filter(
            #     lambda kv: kv[0] not in self.net.backbone.named_parameters(),
            #     self.net.named_parameters()))))
            pretrained_params = list(map(lambda x: x[1], list(filter(
                lambda kv: kv[0] in self.net.videobackbone.named_parameters() or kv[0] in self.net.imgbackbone.named_parameters(),
                self.net.named_parameters()))))
            params = list(map(lambda x: x[1], list(filter(
                lambda kv: kv[0] not in self.net.videobackbone.named_parameters() and kv[0] not in self.net.imgbackbone.named_parameters(),
                self.net.named_parameters()))))

            self.optimizer = Adam([{'params': pretrained_params, 'lr': args.lr/100},
                                   {'params': params, 'lr': args.lr}])
        else:
            self.optimizer = Adam(self.net.parameters(), lr=args.lr)
            
        self.lrsch = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[10, 20, 30, 50], gamma=0.5)
        self.logger = Logger(log_root)
        # K-fold Cross-Validation
        self.trainbag = RuijinData(args.data_root, sub_list=[x for x in [0, 1, 2, 3, 4] if x != args.test_fold], pre_transform=self.train_transform,
                                   task=args.task, modality=args.modality, lump=args.lump)
        print(len(self.trainbag))
        self.testbag = RuijinData(args.data_root, sub_list=[args.test_fold], pre_transform=self.test_transform, 
                                  task=args.task, modality=args.modality, lump=args.lump)
        if args.collate:
            self.train_loader = DataLoader(
                self.trainbag, batch_size=args.batchsize, shuffle=True, num_workers=8, collate_fn=self.collate_fn)
            self.val_loader = DataLoader(
                self.testbag, batch_size=args.batchsize, shuffle=False, num_workers=8, collate_fn=self.collate_fn)
        else:
            self.train_loader = DataLoader(
                self.trainbag, batch_size=args.batchsize, shuffle=True, num_workers=8)
            self.val_loader = DataLoader(
                self.testbag, batch_size=args.batchsize, shuffle=False, num_workers=8)
        self.trainer = MyMulti1Trainer(self.net, self.optimizer, self.lrsch,
                                      None, self.train_loader, self.val_loader, self.logger, 0)
        # self.save_config(args)

    def save_config(self, args):
        config_file = './saved_configs/'+args.log_root+'.txt'
        f = open(config_file, 'a+')
        argDict = args.__dict__
        for arg_key, arg_value in argDict.items():
            f.writelines(arg_key+':'+str(arg_value)+'\n')
        f.close()
        self.logger.auto_backup('./')
        self.logger.backup_files([config_file])

    def collate_fn(self, batch):
        img, video, label, patient = list(zip(*batch))
        return img, video, label, patient

class Multi_config(object):
    '''
    Config for modality==multi
    dataset output: imgs, frames
    video&img share the same transform
    '''
    def __init__(self, log_root, args):
        #self.net = getattr(import_module('models.graph_attention'),args.net)(t=args.t, task=args.task)
        self.net = DeiT_Attn_Pure()
        print('Deit')
        self.net = self.net.cuda()
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ColorJitter(brightness = 0.25),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            # transforms.ColorJitter(0.25, 0.25, 0.25, 0.25),
            transforms.ToTensor()
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        if args.different_lr:
            print('different_lr')
            pretrained_params = list(map(lambda x: x[1], list(filter(
                lambda kv: kv[0] in self.net.backbone.named_parameters(),
                self.net.named_parameters()))))
            params = list(map(lambda x: x[1], list(filter(
                lambda kv: kv[0] not in self.net.backbone.named_parameters(),
                self.net.named_parameters()))))

            self.optimizer = Adam([{'params': pretrained_params, 'lr': args.lr/100},
                                   {'params': params, 'lr': args.lr}])
        else:
            self.optimizer = Adam(self.net.parameters(), lr=args.lr)
        self.optimizer = Adam(self.net.parameters(), lr=args.lr)
        self.lrsch = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[10, 30, 50, 70], gamma=0.5)
        self.logger = Logger(log_root)
        # K-fold Cross-Validation
        self.trainbag = RuijinData(args.data_root, sub_list=[x for x in [0, 1, 2, 3, 4] if x != args.test_fold], pre_transform=self.train_transform,
                                   task=args.task, modality=args.modality, lump=args.lump)
        print(len(self.trainbag))
        self.testbag = RuijinData(args.data_root, sub_list=[args.test_fold], pre_transform=self.test_transform, 
                                  task=args.task, modality=args.modality, lump=args.lump)
        if args.collate:
            self.train_loader = DataLoader(
                self.trainbag, batch_size=args.batchsize, shuffle=True, num_workers=8, collate_fn=self.collate_fn)
            self.val_loader = DataLoader(
                self.testbag, batch_size=args.batchsize, shuffle=False, num_workers=8, collate_fn=self.collate_fn)
        else:
            self.train_loader = DataLoader(
                self.trainbag, batch_size=args.batchsize, shuffle=True, num_workers=8)
            self.val_loader = DataLoader(
                self.testbag, batch_size=args.batchsize, shuffle=False, num_workers=8)
        self.trainer = MyMultiTrainer(self.net, self.optimizer, self.lrsch,
                                      None, self.train_loader, self.val_loader, self.logger, 0)
        # self.save_config(args)

    def save_config(self, args):
        config_file = './saved_configs/'+args.log_root+'.txt'
        f = open(config_file, 'a+')
        argDict = args.__dict__
        for arg_key, arg_value in argDict.items():
            f.writelines(arg_key+':'+str(arg_value)+'\n')
        f.close()
        self.logger.auto_backup('./')
        self.logger.backup_files([config_file])

    def collate_fn(self, batch):
        img, video, label, patient = list(zip(*batch))
        return img, video, label, patient

class DeiT_config(object):
    def __init__(self, log_root, args):
        self.net = DeiT_Video(pretrained=args.load_pretrain)
        self.teacher_model = ViT_Multi()
        pretrained_model = '/remote-home/share/RJ_video/exps/BM_multiViT8_trans_lump_{}/ckp/net.ckpt50.pth'.format(args.test_fold)
        pretrained_model = torch.load(pretrained_model)
        self.teacher_model.load_state_dict(pretrained_model['net'])
        print('teacher model loaded')

        self.net = self.net.cuda()
        self.teacher_model = self.teacher_model.cuda()
        self.teacher_model.eval()
        
        self.train_img_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            #transforms.ColorJitter(brightness = 0.25),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            # transforms.ColorJitter(0.25, 0.25, 0.25, 0.25),
            transforms.ToTensor()
        ])
        self.test_img_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.train_video_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            #transforms.ColorJitter(brightness = 0.25),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            # transforms.ColorJitter(0.25, 0.25, 0.25, 0.25),
            # transforms.ToTensor()
        ])
        self.test_video_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.ToTensor()
        ])

        self.optimizer = Adam(self.net.parameters(), lr=args.lr)
        self.lrsch = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[10, 30, 50, 70], gamma=0.5)
        self.logger = Logger(log_root)
        
        # K-fold Cross-Validation
        self.trainbag = RuijinData(args.data_root, image_transform=self.train_img_transform, video_transform=self.train_video_transform, sub_list=[x for x in [0, 1, 2, 3, 4] if x != args.test_fold],
                                   task=args.task, modality=args.modality, lump=args.lump)
        self.testbag = RuijinData(args.data_root, image_transform=self.test_img_transform, video_transform=self.test_video_transform, sub_list=[args.test_fold],
                                  task=args.task, modality=args.modality, lump=args.lump)
        self.train_loader = DataLoader(
            self.trainbag, batch_size=args.batchsize, shuffle=True, num_workers=8)
        self.val_loader = DataLoader(
            self.testbag, batch_size=args.batchsize, shuffle=False, num_workers=8)

        if args.modality == 'deit_multi':
            print('multi modality')
            self.trainer = DeiTMultiTrainer(self.net, self.teacher_model, self.optimizer, self.lrsch, DistillationLoss(nn.CrossEntropyLoss(), self.teacher_model),
                                      self.train_loader, self.val_loader, self.logger, 0, task=args.task)
        else:
            print('deit_video modality')
            self.trainer = DeiTTrainer(self.net, self.teacher_model, self.optimizer, self.lrsch, DistillationLoss(nn.CrossEntropyLoss(), self.teacher_model),
                                      self.train_loader, self.val_loader, self.logger, 0, task=args.task)
        self.save_config(args)

    def save_config(self, args):
        config_file = './saved_configs/'+args.log_root+'.txt'
        f = open(config_file, 'a+')
        argDict = args.__dict__
        for arg_key, arg_value in argDict.items():
            f.writelines(arg_key+':'+str(arg_value)+'\n')
        f.close()
        self.logger.auto_backup('./')
        self.logger.backup_files([config_file])

class MultiImportance_config(object):
    '''
    Config for modality==multi_importance
    dataset output: imgs, frames, importance_label
    video&img share the same transform
    '''
    def __init__(self, log_root, args):
        #self.net = getattr(import_module('models.graph_attention'),args.net)(t=args.t, task=args.task)
        self.net = DeiT_Importance(is_train=args.train)
        print('DeiT_Importance')
        self.net = self.net.cuda()
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ColorJitter(brightness = 0.25),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            # transforms.ColorJitter(0.25, 0.25, 0.25, 0.25),
            transforms.ToTensor()
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        if args.different_lr:
            print('different_lr')
            pretrained_params = list(map(lambda x: x[1], list(filter(
                lambda kv: kv[0] in self.net.backbone.named_parameters(),
                self.net.named_parameters()))))
            params = list(map(lambda x: x[1], list(filter(
                lambda kv: kv[0] not in self.net.backbone.named_parameters(),
                self.net.named_parameters()))))

            self.optimizer = Adam([{'params': pretrained_params, 'lr': args.lr/100},
                                   {'params': params, 'lr': args.lr}])
        else:
            self.optimizer = Adam(self.net.parameters(), lr=args.lr)
        self.optimizer = Adam(self.net.parameters(), lr=args.lr)
        self.lrsch = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[10, 30, 50, 70], gamma=0.5)
        self.logger = Logger(log_root)
        # K-fold Cross-Validation
        self.trainbag = RuijinData(args.data_root, sub_list=[x for x in [0, 1, 2, 3, 4] if x != args.test_fold], pre_transform=self.train_transform,
                                   task=args.task, modality=args.modality, lump=args.lump)
        print(len(self.trainbag))
        self.testbag = RuijinData(args.data_root, sub_list=[args.test_fold], pre_transform=self.test_transform, 
                                  task=args.task, modality=args.modality, lump=args.lump, train=args.train)
        if args.collate:
            self.train_loader = DataLoader(
                self.trainbag, batch_size=args.batchsize, shuffle=True, num_workers=8, collate_fn=self.collate_fn)
            self.val_loader = DataLoader(
                self.testbag, batch_size=args.batchsize, shuffle=False, num_workers=8, collate_fn=self.collate_fn)
        else:
            self.train_loader = DataLoader(
                self.trainbag, batch_size=args.batchsize, shuffle=True, num_workers=8)
            self.val_loader = DataLoader(
                self.testbag, batch_size=args.batchsize, shuffle=False, num_workers=8)
        self.trainer = MyMultiImportanceTrainer(self.net, self.optimizer, self.lrsch,
                                      None, self.train_loader, self.val_loader, self.logger, 0)
        # self.save_config(args)

    def save_config(self, args):
        config_file = './saved_configs/'+args.log_root+'.txt'
        f = open(config_file, 'a+')
        argDict = args.__dict__
        for arg_key, arg_value in argDict.items():
            f.writelines(arg_key+':'+str(arg_value)+'\n')
        f.close()
        self.logger.auto_backup('./')
        self.logger.backup_files([config_file])

    def collate_fn(self, batch):
        img, video, label, patient = list(zip(*batch))
        return img, video, label, patient


if __name__ == '__main__':
    #configs = getattr(import_module('configs.'+args.config),'Config')()
    #configs = configs.__dict__
    parser = argparse.ArgumentParser(description='Ruijin Framework')
    # parser.add_argument('--data_root',type=str,default='/remote-home/share/RJ_video/RJ_video_crop')
    parser.add_argument('--data_root', type=str,
                        default='/remote-home/share/RJ_video_crop')
    parser.add_argument('--log_root', type=str, default='./log')
    parser.add_argument('--test_fold', type=int, default=0,
                        help='which fold of data is used for test')
    parser.add_argument('--task', type=str, default='BM', help='BM or ALNM')
    parser.add_argument('--modality', type=str, help='Image or Video or Multi')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--resume', type=int, default=-1)
    parser.add_argument('--batchsize', type=int, default=1)
    parser.add_argument('--net', type=str, default='H_Attention_Graph')
    parser.add_argument('--video_net_type', type=int,
                        default=1, help='1,2,...')
    parser.add_argument('--num_class', type=int, default=2)
    parser.add_argument('--DP', type=bool, default=False, help='DataParallel')

    parser.add_argument('--lump', type=int, default=0,
                        help='only lump data used 0/1')

    # saved models在/remote-home/share/RJ_video/exps/log_ALNM_lump_0/ckp下
    parser.add_argument('--model_path', type=str,
                        default=None, help='load pretrained')
    parser.add_argument('--train', type=int, default=1, help='train or test')
    parser.add_argument('--collate', type=bool,
                        default=False, help='use collate_fn')

    parser.add_argument('--model', type=bool,
                        default=False, help='load pretrain')

    parser.add_argument('--load_pretrain', type=bool,
                        default=False, help='load pretrain')

    parser.add_argument('--different_lr', type=bool,
                        default=False, help='smaller learning rate for pretrained model')
    parser.add_argument('--test', type=bool,
                        default=False, help='smaller learning rate for pretrained model')


    # parse parameters
    args = parser.parse_args()
    log_root = os.path.join('/remote-home/share/RJ_video/exps', args.log_root)
    if not os.path.exists(log_root):
        os.mkdir(log_root)

    print(args.modality)
    if args.train:
        if args.modality == 'video':
            config_object = Multi_config(log_root, args)
        elif args.modality == 'image':
            config_object = Multi_config(log_root, args)
        elif args.modality == 'multi':
            config_object = Multi_config(log_root, args)
        elif args.modality == 'multi1':
            config_object = Multi1_config(log_root, args)
        elif args.modality == 'deit_video':
            config_object = DeiT_config(log_root,args)
        elif args.modality == 'deit_multi':
            config_object = DeiT_config(log_root,args)
        elif args.modality == 'multi_importance':
            config_object = MultiImportance_config(log_root,args)
        elif args.modality == 'deit_attn1':
            config_object = DeiT_Attn1_config(log_root, args)
        else:
            print('MODALITY ERROR,main')

        # config_object.trainer.test()                      


        for epoch in range(config_object.logger.global_step, args.epoch):
            if args.test:
                break
            print('Now epoch {}'.format(epoch))
            config_object.trainer.train()
            config_object.trainer.test()

    else:
        if args.modality == 'video':
            config_object = Multi_config(log_root, args)
        elif args.modality == 'image':
            config_object = Multi_config(log_root, args)
        elif args.modality == 'multi':
            config_object = Multi_config(log_root, args)
        elif args.modality == 'multi1':
            print(args.modality)
            config_object = Multi1_config(log_root, args)
        elif args.modality == 'multi_importance':
            config_object = MultiImportance_config(log_root,args)
        else:
            print('MODALITY ERROR')

        # model_path = os.path.join(
        #     '/remote-home/share/RJ_video/exps', args.model_path)
        # model_CKPT = torch.load(model_path)
        # config_object.trainer.net.load_state_dict(model_CKPT['net'])
        # print(type(config_object.trainer.net))

        '''FOR TEST'''
        pretrained_model = '/remote-home/share/RJ_video/exps/BM_vitt_multi1_1012_0/ckp/net.ckpt29.pth'
        pretrained_model = torch.load(pretrained_model)
        pretrained_state = pretrained_model['net']
        model_state = config_object.trainer.net.state_dict()
        for key in model_state.keys():
            model_state[key] = pretrained_state[key]
        # pretrained_model_state, pretrained_parameters = config_object.trainer.net.get_pretrained_state_dict()
        config_object.trainer.net.load_state_dict(model_state)
        # config_object.trainer.test_video()
        config_object.trainer.test()

        '''FOR PREDICT'''
        # pred, prob, patients = config_object.trainer.predict()
        # results = pd.DataFrame()
        # results['ID'] = 0
        # results['pred'] = 0
        # results['prob'] = 0
        # for i in range(len(pred)):
        #     results.loc[i, 'ID'] = patients[i].tolist()[0]
        #     results.loc[i, 'pred'] = pred[i][0]
        #     results.loc[i, 'prob'] = prob[i][0]
        # results.to_csv(os.path.join('./result', args.log_root+'.csv'))
