# -*- coding: utf-8 -*-
# from numpy import imag
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import timm
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class ViT_Multi(nn.Module):
    def __init__(self, num_class=2, pretrained_backbone = True):
        super(ViT_Multi, self).__init__()
        self.backbone = timm.create_model(
            'vit_base_patch16_224', pretrained=pretrained_backbone)

        # self.encoder_layer = nn.TransformerEncoderLayer(d_model=1000, nhead=8)
        # self.transformer_encoder = nn.TransformerEncoder(
        #     self.encoder_layer, num_layers=1)

        self.classifier = nn.Sequential(
            nn.Linear(1000, 2),
            nn.Softmax(dim=1)
        )

    # concatenate forward
    def forward(self, images, debug=False):
        if debug:
            print('input_image', images.shape)
            # [1, n, 3, 224, 224]
        # batch=1 时 会去掉第一个维度
        images = images.squeeze(0)

        images = self.backbone(images)
        # [n, 1000]
        if debug:
            print('image_backbone', images.shape)

        # images = torch.stack([images])
        # images = self.transformer_encoder(images)
        # if debug:
        #     print('image_transformer', images.shape)
        # images = images.squeeze(0)

        if debug:
            print('image_squeeze', images.shape)

        pred = self.classifier(images)
        pred = pred.mean(axis=0, keepdim=True)
        if debug:
            print('pred', pred.shape)
            exit()

        return pred


class ViT_Similarity(nn.Module):
    def __init__(self, num_class=2, use_similarity = True, t=1):
        super(ViT_Similarity, self).__init__()
        
        self.use_similarity = use_similarity

        self.backbone = timm.create_model(
            'vit_base_patch16_224', pretrained=True)

        self.t = t

        self.classifier = nn.Sequential(
            nn.Linear(1000, 2),
            nn.Softmax(dim=1)
        )

    # concatenate forward
    def forward(self, images, video, debug=False):

        images = images.squeeze(0)
        # video = video.permute(0,2,1,3,4)
        video = video.squeeze(0)

        video_features = self.backbone(video)
        img_features = self.backbone(images)
        # [n, 1000]

        pred = self.classifier(video_features)
        if self.use_similarity:
            # similarity = self.similarity_calculate(video,images)
            similarity = self.feature_similarity(video_features,img_features)
            similarity = torch.stack([similarity])
            similarity = similarity.cuda()
            pred = torch.mm(similarity,pred)
        else:
            pred = pred.mean(axis=0, keepdim=True)
        if debug:
            print('pred', pred.shape)
            exit()

        return pred

    def similarity_calculate(self, video, imgs):
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
        results = torch.tensor(results)
        results = torch.mul(results,self.t)
        return nn.Softmax(dim=0)(results)

    def feature_similarity(self, video, imgs):
        results = []
        for frame in video.split(1):
            similarity = []
            for img in imgs.split(1):
                similarity.append(torch.cosine_similarity(frame,img))
            results.append(max(similarity))
        results = torch.tensor(results)
        results = torch.mul(results,self.t)
        return nn.Softmax(dim=0)(results)


class ViT_Importance(nn.Module):
    def __init__(self, num_class=2):
        super(ViT_Importance, self).__init__()

        print('ViT_Importance Created.')

        self.backbone = timm.create_model(
            'vit_base_patch16_224', pretrained=True)

        self.class_classifier = nn.Sequential(
            nn.Linear(1000, 2),
            nn.Softmax(dim=1)
        )

        self.importance_classifier = nn.Sequential(
            nn.Linear(1000, 2),
            nn.Softmax(dim=1)
        )

    # concatenate forward
    def forward(self, frames, debug=False):

        frames = frames.squeeze(0)

        frames_features = self.backbone(frames)
        # [n, 1000]
        pred = self.class_classifier(frames_features)
        importance = self.importance_classifier(frames_features)
        weight = importance.split(1,dim=1)[0]
        weight = nn.Softmax(dim=0)(weight)
        weight = weight.transpose(1,0)
        pred = torch.mm(weight,pred)

        if debug:
            print('pred', pred.shape)
            exit()

        return pred, importance


if __name__ == '__main__':
    model = ViT_Importance()
    img = torch.rand(1, 16, 3, 224, 224)
    # video = torch.rand(1,16,3,224,224)
    label = torch.tensor([0])
    prob_label, importance = model(img)
    print(prob_label,importance)
    # loss_func = nn.CrossEntropyLoss()
    # loss = loss_func(prob_label, label)
    # print(loss)
    # print(loss.item())
