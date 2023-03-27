# -*- coding: utf-8 -*-
# from numpy import imag
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import timm
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from models.DeiT.DeiT import deit_tiny_distilled_patch16_224, deit_tiny_patch16_224, deit_base_distilled_patch16_224, deit_base_patch16_224
# from DeiT.DeiT import deit_tiny_distilled_patch16_224



class DeiT_Video(nn.Module):
    def __init__(self, num_class=2, pretrained = False):
        super(DeiT_Video, self).__init__()
        print('DeiT_Video')
        self.backbone = deit_tiny_distilled_patch16_224(pretrained=pretrained)

        self.classifier = nn.Sequential(
            nn.Linear(1000, 2),
            nn.Softmax(dim=1)
        )

    # concatenate forward
    def forward(self, video, debug=False):
        if debug:
            print('input', video.shape)
            # [1, 16, 3, 224, 224]
        # batch=1 时 
        video = video.squeeze(0)
        video = self.backbone(video)
        # ([16, 1000],[16,1000])

        class_token, distill_token = video
        class_pred = self.classifier(class_token)
        distill_pred = self.classifier(distill_token)
        class_pred = class_pred.mean(axis=0, keepdim=True)
        distill_pred = distill_pred.mean(axis=0, keepdim=True)

        return (class_pred,distill_pred)

class DeiT_Tiny(nn.Module):
    def __init__(self, num_class=2, pretrained = False):
        super(DeiT_Tiny, self).__init__()
        print('DeiT_Video')
        self.backbone = deit_tiny_patch16_224(pretrained=pretrained)

        self.classifier = nn.Sequential(
            nn.Linear(1000, 2),
            nn.Softmax(dim=1)
        )

    # concatenate forward
    def forward(self, video, debug=False):
        if debug:
            print('input', video.shape)
            # [1, 16, 3, 224, 224]
        # batch=1 时 
        video = video.squeeze(0)
        video = self.backbone(video)
        # ([16, 1000],[16,1000])

        class_token = video
        class_pred = self.classifier(class_token)
        class_pred = class_pred.mean(axis=0, keepdim=True)

        return class_pred

class DeiT_Importance(nn.Module):
    def __init__(self, num_class=2, pretrained = True, is_train = 1, deit_size = 'tiny'):
        super(DeiT_Importance, self).__init__()
        print('DeiT_Importance created.')
        if deit_size == 'base':
            self.backbone = deit_base_distilled_patch16_224(pretrained=pretrained)
        elif deit_size == 'tiny':
            self.backbone = deit_tiny_distilled_patch16_224(pretrained=pretrained)
            print('The size is tiny.')
        else:
            raise NotImplementedError("Deit Size Error")
        self.is_train = is_train

        self.class_classifier = nn.Sequential(
            nn.Linear(1000, 2),
            # nn.Softmax(dim=1)
        )

        self.importance_classifier = nn.Sequential(
            nn.Linear(1000, 2),
            # nn.Softmax(dim=1)
        )

    # concatenate forward
    def forward(self, video, debug=False):
        if debug:
            print('input', video.shape)
            # [1, 16, 3, 224, 224]
        # batch=1 时 
        video = video.squeeze(0)
        video = self.backbone(video)
        # ([16, 1000],[16,1000])

        frames_features, importance_token = video

        pred = self.class_classifier(frames_features)
        importance = self.importance_classifier(importance_token)
        weight = importance.split(1,dim=1)[1]
        if self.is_train == 0:
            # weight = torch.mul(weight,100)
            weight = torch.mul(weight,0)
        # else:
        #     weight = torch.mul(weight,0.1)
        weight = nn.Softmax(dim=0)(weight)
        weight = weight.transpose(1,0)
        pred = torch.mm(weight,pred)

        if debug:
            print('pred', pred.shape)
            exit()

        return pred, importance

class DeiT_Attn(nn.Module):
    def __init__(self, num_class=2, pretrained = True, deit_size = 'tiny', attention = True, modality = 'video'):
        super(DeiT_Attn, self).__init__()
        print('DeiT_Importance created.')
        self.attention = attention
        if deit_size == 'base':
            self.videobackbone = deit_base_distilled_patch16_224(pretrained=pretrained)
            self.imgbackbone = deit_base_distilled_patch16_224(pretrained=pretrained)
        elif deit_size == 'tiny':
            self.videobackbone = deit_tiny_distilled_patch16_224(pretrained=pretrained)
            self.imgbackbone = deit_tiny_distilled_patch16_224(pretrained=pretrained)
            print('The size is tiny.')
        else:
            raise NotImplementedError("Deit Size Error")

        self.video_query = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, 64)
        )

        self.classifier = nn.Sequential(
            nn.Linear(1000, 2)
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
        img = img.squeeze(0)
        video = video.squeeze(0)
        
        video_v, video_q = self.videobackbone(video)
        # ([16, 1000],[16,1000])

        img, img_k = self.imgbackbone(img)

        Q = self.video_query(video_q)
        K = self.img_key(img_k)

        Q = torch.transpose(Q, 1, 0)
        A = torch.mm(K, Q)
        A = A.mean(axis=0,keepdim=True)
        A = torch.softmax(A,1)
        if debug: print(A, A.size())

        V = torch.mm(A,video_v)
        if debug: print(V.size())
        prob = self.classifier(V)
        
        return prob
   
class DeiT_Attn_Pure(nn.Module):
    def __init__(self, num_class=2, pretrained = True, deit_size = 'tiny', attention = True, modality = 'video'):
        super(DeiT_Attn_Pure, self).__init__()
        print('DeiT_Importance created.')
        self.attention = attention
        if deit_size == 'base':
            self.backbone = deit_base_distilled_patch16_224(pretrained=pretrained)
            print('The size is base.')
        elif deit_size == 'tiny':
            self.backbone = deit_tiny_distilled_patch16_224(pretrained=pretrained)
            print('The size is tiny.')
        else:
            raise NotImplementedError("Deit Size Error")

        self.classifier = nn.Sequential(
            nn.Linear(1000, 2)
        )

    # concatenate forward
    def forward(self, video,  debug=False):
        if debug:
            print('input', video.shape)
            # [1, 16, 3, 224, 224]
        # batch=1 时 

        video = video.squeeze(0)
        
        video_v, video_q = self.backbone(video)
        # ([16, 1000],[16,1000])

        prob = self.classifier(video_v)
        prob = prob.mean(axis=0,keepdim=True)
        
        return prob
   
class DeiT_Attn1(nn.Module):
    def __init__(self, num_class=2, pretrained = True, deit_size = 'tiny', attention = True):
        super(DeiT_Attn1, self).__init__()
        print('DeiT_Attention1 created.')
        self.attention = attention
        if deit_size == 'base':
            self.videobackbone = deit_base_distilled_patch16_224(pretrained=pretrained)
            self.imgbackbone = deit_base_distilled_patch16_224(pretrained=pretrained)
        elif deit_size == 'tiny':
            self.videobackbone = deit_tiny_distilled_patch16_224(pretrained=pretrained)
            self.imgbackbone = deit_tiny_distilled_patch16_224(pretrained=pretrained)
            print('The size is tiny.')
        else:
            raise NotImplementedError("Deit Size Error")

        self.video_query = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, 64)
        )

        self.classifier = nn.Sequential(
            nn.Linear(1000, 2)
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
        img = img.squeeze(0)
        video = video.squeeze(0)
        
        video_v, video_q = self.videobackbone(video)
        # ([16, 1000],[16,1000])

        img, img_k = self.imgbackbone(img)

        Q = self.video_query(video_q)
        K = self.img_key(img_k)

        Q = torch.transpose(Q, 1, 0)
        A = torch.mm(K, Q)
        A = A.mean(axis=0,keepdim=True)
        A = torch.softmax(A,1)
        if debug: print(A, A.size())

        prob = self.classifier(video_v)

        prob = torch.mm(A,prob)
        
        return prob
  

class DeiT_Attn_cls(nn.Module):
    def __init__(self, num_class=2, pretrained = True, deit_size = 'tiny', attention = True):
        super(DeiT_Attn_cls, self).__init__()
        print('DeiT_Attention based on cls created.')
        self.attention = attention
        if deit_size == 'base':
            self.videobackbone = deit_base_distilled_patch16_224(pretrained=pretrained)
            self.imgbackbone = deit_base_distilled_patch16_224(pretrained=pretrained)
        elif deit_size == 'tiny':
            self.videobackbone = deit_tiny_distilled_patch16_224(pretrained=pretrained)
            self.imgbackbone = deit_tiny_distilled_patch16_224(pretrained=pretrained)
            print('The size is tiny.')
        else:
            raise NotImplementedError("Deit Size Error")

        self.video_query = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, 64)
        )

        self.classifier = nn.Sequential(
            nn.Linear(1000, 2)
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
        img = img.squeeze(0)
        video = video.squeeze(0)
        
        video_v, video_q = self.videobackbone(video)
        # ([16, 1000],[16,1000])

        img, img_k = self.imgbackbone(img)

        Q = video_v
        K = img

        Q = torch.transpose(Q, 1, 0)
        A = torch.mm(K, Q)
        A = A.mean(axis=0,keepdim=True)
        A = torch.softmax(A,1)
        if debug: print(A, A.size())

        V = torch.mm(A,video_v)
        if debug: print(V.size())

        # img = img.mean(axis=0,keepdim=True)
        # V = torch.cat([img,V],dim=1)

        prob = self.classifier(V)
        
        return prob

# concatenate the characteristics of video&image
class DeiT_Attn2(nn.Module):
    def __init__(self, num_class=2, pretrained = True, deit_size = 'tiny', attention = True, modality = 'video'):
        super(DeiT_Attn2, self).__init__()
        print('DeiT_Attn2 created.')
        self.attention = attention
        if deit_size == 'base':
            self.videobackbone = deit_base_distilled_patch16_224(pretrained=pretrained)
            self.imgbackbone = deit_base_distilled_patch16_224(pretrained=pretrained)
        elif deit_size == 'tiny':
            self.videobackbone = deit_tiny_distilled_patch16_224(pretrained=pretrained)
            self.imgbackbone = deit_tiny_distilled_patch16_224(pretrained=pretrained)
            print('The size is tiny.')
        else:
            raise NotImplementedError("Deit Size Error")

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
        img = img.squeeze(0)
        video = video.squeeze(0)
        
        video_v, video_q = self.videobackbone(video)
        # ([16, 1000],[16,1000])

        img, img_k = self.imgbackbone(img)

        Q = video_q
        K = img_k
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
        
        return prob

class Baseline_test(nn.Module):
    def __init__(self, num_class=2, pretrained = True):
        super(Baseline_test, self).__init__()
        print('Resnet_test created.')
        self.videobackbone = timm.create_model('resnet18', pretrained=True)
        self.imgbackbone = timm.create_model('resnet18', pretrained=True)

        # print('ViT-Tiny created.')
        # self.videobackbone = deit_tiny_patch16_224(pretrained=True)
        # self.imgbackbone = deit_tiny_patch16_224(pretrained=True)

        self.classifier = nn.Sequential(
            nn.Linear(2000, 2)
        )

    # concatenate forward
    def forward(self, img, video,  debug=False):
        if debug:
            print('input', video.shape)
        img = img.squeeze(0)
        video = video.squeeze(0)
        video = self.videobackbone(video)
        video = video.mean(axis=0, keepdim=True)

        img = self.imgbackbone(img)
        img = img.mean(axis=0, keepdim=True)

        prob = torch.cat([img,video],dim=1)
        prob = self.classifier(prob)
        prob = prob.mean(axis=0,keepdim=True)
        
        return prob
  

if __name__ == '__main__':
    model = DeiT_Attn2(pretrained=False)
    video = torch.randn(1,16,3,224,224)
    img = torch.randn(1,2,3,224,224)
    y = model(img, video, debug = True)
    print(y)
