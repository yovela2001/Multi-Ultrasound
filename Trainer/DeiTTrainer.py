#!/usr/bin/env
# -*- coding: utf-8 -*-
from __future__ import absolute_import
import os
import sys
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.utility as utility
from utils.logger import Logger
import argparse
from importlib import import_module
from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, accuracy_score
# for debug
import torch.nn.functional as F
import utils.utility as utility
from utils.logger import Logger
import argparse
from importlib import import_module
import math

import json
import pickle


class DeiTTrainer(object):
    def __init__(self, net, teacher_model, optimizer, lrsch, loss, train_loader, val_loader, logger, start_epoch,
                 save_interval=1, task='BM'):

        self.net = net
        self.teacher_model = teacher_model
        self.optimizer = optimizer
        self.lrsch = lrsch
        self.train_loader = train_loader
        self.test_loader = val_loader
        self.logger = logger
        self.logger.global_step = start_epoch
        self.save_interval = save_interval
        self.loss = loss
        if task == 'BM':
            self.threshold = 0.5
        elif task == 'ALNM':
            self.threshold = 0.4

    def train(self):
        self.net.train()
        self.logger.update_step()
        train_loss = 0.
        prob = []
        pred = []
        target = []
        for multi, video, label, _ in (tqdm(self.train_loader, ascii=True, ncols=60)):
            # reset gradients
            self.optimizer.zero_grad()
            multi = multi.cuda()
            video = video.cuda()
            label = label.cuda()

            output = self.net(video)
            prob_label = (output[0]+output[1])/2
            # prob_label = F.softmax(prob_label,dim=1)
            predicted_label = torch.ge(prob_label, 0.5).float()

            loss = self.loss(multi, output, label)
            if not math.isfinite(loss.item()):
                print("Loss is infinite, stopping training")
                sys.exit(1)

            train_loss += loss.item()
            target.append(label.cpu().detach().numpy().ravel())
            pred.append(predicted_label[:,1].cpu().detach().numpy().ravel())
            prob.append(prob_label[:,1].cpu().detach().numpy().ravel())

            # backward pass
            loss.backward()
            # step
            self.optimizer.step()
            self.lrsch.step()

        # calculate loss and error for epoch
        train_loss /= len(self.train_loader)

        
        self.log_metric("Train", target, prob, pred)

        if not (self.logger.global_step % self.save_interval):
            self.logger.save(self.net, self.optimizer, self.lrsch, self.loss)

        print('Loss: {:.4f}'.format(train_loss))

    def test(self):
        self.net.eval()
        test_loss = 0.
        target = []
        pred = []
        prob = []
        for multi, video, label, _ in tqdm(self.test_loader, ascii=True, ncols=60):
            multi = multi.cuda()
            video = video.cuda()
            label = label.cuda()

            output = self.net(video)
            
            prob_label = (output[0]+output[1])/2
            # prob_label = F.softmax(prob_label,dim=1)
            predicted_label = torch.ge(prob_label, 0.5).float()
            
            loss = self.loss(multi, output, label)
            test_loss += loss.item()

            target.append(label.cpu().detach().numpy().ravel())
            pred.append(predicted_label[:,1].cpu().detach().numpy().ravel())
            prob.append(prob_label[:,1].cpu().detach().numpy().ravel())

        self.log_metric("Test", target, prob, pred)


    def log_metric(self, prefix, target, prob, pred):
        pred_list = np.concatenate(pred)
        prob_list = np.concatenate(prob)
        target_list = np.concatenate(target)
        cls_report = classification_report(
            target_list, pred_list, output_dict=True, zero_division=0)
        acc = accuracy_score(target_list, pred_list)
        #print ('acc is {}'.format(acc))
        auc_score = roc_auc_score(target_list, prob_list)
        print('auc is {}'.format(auc_score))
        # print(cls_report)

        self.logger.log_scalar(prefix+'/'+'AUC', auc_score, print=True)
        self.logger.log_scalar(prefix+'/'+'Acc', acc, print=True)
        self.logger.log_scalar(
            prefix+'/'+'Malignant_precision', cls_report['1']['precision'], print=True)
        self.logger.log_scalar(prefix+'/'+'Benign_precision',
                               cls_report['0']['precision'], print=True)
        self.logger.log_scalar(prefix+'/'+'Malignant_recall',
                               cls_report['1']['recall'], print=True)
        self.logger.log_scalar(prefix+'/'+'Benign_recall',
                               cls_report['0']['recall'], print=True)
        self.logger.log_scalar(prefix+'/'+'Malignant_F1',
                               cls_report['1']['f1-score'], print=True)

        '''
        self.logger.log_scalar(prefix+'/'+'Accuracy', acc, print=True)
        self.logger.log_scalar(prefix+'/'+'Precision', cls_report['1.0']['precision'], print=True)
        self.logger.log_scalar(prefix+'/'+'Recall', cls_report['1.0']['recall'], print=True)
        self.logger.log_scalar(prefix+'/'+'F1', cls_report['1.0']['f1-score'], print=True)
        self.logger.log_scalar(prefix+'/'+'Specificity', cls_report['0.0']['recall'], print=True)
        '''


class DeiTMultiTrainer(object):
    def __init__(self, net, teacher_model, optimizer, lrsch, loss, train_loader, val_loader, logger, start_epoch,
                 save_interval=1, task='BM'):

        self.net = net
        self.teacher_model = teacher_model
        self.optimizer = optimizer
        self.lrsch = lrsch
        self.train_loader = train_loader
        self.test_loader = val_loader
        self.logger = logger
        self.logger.global_step = start_epoch
        self.save_interval = save_interval
        self.loss = loss
        if task == 'BM':
            self.threshold = 0.5
        elif task == 'ALNM':
            self.threshold = 0.4

    def train(self):
        self.net.train()
        self.logger.update_step()
        train_loss = 0.
        prob = []
        pred = []
        target = []
        for multi, label, _ in (tqdm(self.train_loader, ascii=True, ncols=60)):
            # reset gradients
            self.optimizer.zero_grad()
            multi = multi.cuda()
            label = label.cuda()

            output = self.net(multi)
            prob_label = (output[0]+output[1])/2
            # prob_label = F.softmax(prob_label,dim=1)
            predicted_label = torch.ge(prob_label, 0.5).float()

            loss = self.loss(multi, output, label)
            if not math.isfinite(loss.item()):
                print("Loss is infinite, stopping training")
                sys.exit(1)

            train_loss += loss.item()
            target.append(label.cpu().detach().numpy().ravel())
            pred.append(predicted_label[:,1].cpu().detach().numpy().ravel())
            prob.append(prob_label[:,1].cpu().detach().numpy().ravel())

            # backward pass
            loss.backward()
            # step
            self.optimizer.step()
            self.lrsch.step()

        # calculate loss and error for epoch
        train_loss /= len(self.train_loader)

        
        self.log_metric("Train", target, prob, pred)

        if not (self.logger.global_step % self.save_interval):
            self.logger.save(self.net, self.optimizer, self.lrsch, self.loss)

        print('Loss: {:.4f}'.format(train_loss))

    def test(self):
        self.net.eval()
        test_loss = 0.
        target = []
        pred = []
        prob = []
        for multi, label, _ in tqdm(self.test_loader, ascii=True, ncols=60):
            multi = multi.cuda()
            label = label.cuda()

            output = self.net(multi)
            
            prob_label = (output[0]+output[1])/2
            # prob_label = F.softmax(prob_label,dim=1)
            predicted_label = torch.ge(prob_label, 0.5).float()
            
            loss = self.loss(multi, output, label)
            test_loss += loss.item()

            target.append(label.cpu().detach().numpy().ravel())
            pred.append(predicted_label[:,1].cpu().detach().numpy().ravel())
            prob.append(prob_label[:,1].cpu().detach().numpy().ravel())

        self.log_metric("Test", target, prob, pred)


    def log_metric(self, prefix, target, prob, pred):
        pred_list = np.concatenate(pred)
        prob_list = np.concatenate(prob)
        target_list = np.concatenate(target)
        cls_report = classification_report(
            target_list, pred_list, output_dict=True, zero_division=0)
        acc = accuracy_score(target_list, pred_list)
        #print ('acc is {}'.format(acc))
        auc_score = roc_auc_score(target_list, prob_list)
        print('auc is {}'.format(auc_score))
        # print(cls_report)

        self.logger.log_scalar(prefix+'/'+'AUC', auc_score, print=True)
        self.logger.log_scalar(prefix+'/'+'Acc', acc, print=True)
        self.logger.log_scalar(
            prefix+'/'+'Malignant_precision', cls_report['1']['precision'], print=True)
        self.logger.log_scalar(prefix+'/'+'Benign_precision',
                               cls_report['0']['precision'], print=True)
        self.logger.log_scalar(prefix+'/'+'Malignant_recall',
                               cls_report['1']['recall'], print=True)
        self.logger.log_scalar(prefix+'/'+'Benign_recall',
                               cls_report['0']['recall'], print=True)
        self.logger.log_scalar(prefix+'/'+'Malignant_F1',
                               cls_report['1']['f1-score'], print=True)

        '''
        self.logger.log_scalar(prefix+'/'+'Accuracy', acc, print=True)
        self.logger.log_scalar(prefix+'/'+'Precision', cls_report['1.0']['precision'], print=True)
        self.logger.log_scalar(prefix+'/'+'Recall', cls_report['1.0']['recall'], print=True)
        self.logger.log_scalar(prefix+'/'+'F1', cls_report['1.0']['f1-score'], print=True)
        self.logger.log_scalar(prefix+'/'+'Specificity', cls_report['0.0']['recall'], print=True)
        '''
