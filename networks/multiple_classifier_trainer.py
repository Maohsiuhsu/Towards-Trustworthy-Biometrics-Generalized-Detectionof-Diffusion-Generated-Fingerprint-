import functools
import torch
import torch.nn as nn
from networks.classifier import Multi_Binary_Classifier, Classifier
from networks.resnet import resnet50
from networks.discriminator import Discriminator_v1
from networks.base_model import BaseModel, init_weights
from .loss import ContrastiveLoss, TripletLoss_v2
import random
import numpy as np


class mix_Trainer_v1_2(BaseModel):
    def name(self):
        return 'mix_Trainer_v1_2'

    def __init__(self, opt):
        super(mix_Trainer_v1_2, self).__init__(opt)

        if self.isTrain and not opt.continue_train:
            self.model = resnet50(num_classes=1)
            torch.nn.init.normal_(self.model.fc.weight.data, 0.0, opt.init_gain)
            if opt.pretrained_path != None:
                state_dict = torch.load(opt.pretrained_path, map_location='cpu')
                self.model.load_state_dict(state_dict['model'])
            else:
                print(f"your pretrained_path is empty, please set the path.")
            
            # parallel classifier 1
            self.multi_classifier_1 = Classifier(num_classes=3)
            torch.nn.init.normal_(self.multi_classifier_1.fc1.weight.data, 0.0, opt.init_gain)
            torch.nn.init.normal_(self.multi_classifier_1.fc2.weight.data, 0.0, opt.init_gain)
            
            # parallel classifier 2
            self.multi_classifier_2 = Classifier(num_classes=3)
            torch.nn.init.normal_(self.multi_classifier_2.fc1.weight.data, 0.0, opt.init_gain)
            torch.nn.init.normal_(self.multi_classifier_2.fc2.weight.data, 0.0, opt.init_gain)
            
            
            self.classifier = Classifier(num_classes=1)
            torch.nn.init.normal_(self.classifier.fc1.weight.data, 0.0, opt.init_gain)
            torch.nn.init.normal_(self.classifier.fc2.weight.data, 0.0, opt.init_gain)
            
        # Trainable parameters    
        params_F = []
    
        for name, p in self.model.named_parameters():
            if  name=="fc.weight" or name=="fc.bias": 
                p.requires_grad = False
            else:
                params_F.append(p)

        params_Multi_C1 = self.multi_classifier_1.parameters()          
        params_Multi_C2 = self.multi_classifier_2.parameters()         
        params_Binary_C = self.classifier.parameters()            
        
        if self.isTrain:
            self.BCE_loss = nn.BCEWithLogitsLoss()
            self.CE_loss = nn.CrossEntropyLoss()
            # initialize optimizers
            if opt.optim == 'adam':
                self.optimizer_F = torch.optim.Adam(params_F,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_MC1 = torch.optim.Adam(params_Multi_C1,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_MC2 = torch.optim.Adam(params_Multi_C2,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))                                  
                self.optimizer_BC = torch.optim.Adam(params_Binary_C,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))                                  
            elif opt.optim == 'sgd':
                self.optimizer_F = torch.optim.SGD(params_F,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_MC1 = torch.optim.SGD(params_Multi_C1,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_MC2 = torch.optim.Adam(params_Multi_C2,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))                                   
                self.optimizer_BC = torch.optim.SGD(params_Binary_C,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))  
            else:
                raise ValueError("optim should be [adam, sgd]")
            
        self.model.to(opt.gpu_ids[0])
        self.classifier.to(opt.gpu_ids[0])
        self.multi_classifier_1.to(opt.gpu_ids[0])
        self.multi_classifier_2.to(opt.gpu_ids[0])
        

    def adjust_learning_rate(self, min_lr=1e-7):
        for param_group in self.optimizer_F.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        for param_group in self.optimizer_MC1.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        for param_group in self.optimizer_MC2.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False        
        for param_group in self.optimizer_BC.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False        
        return True

    def set_input(self, batch1, batch2):
        # img, multi_label, label, detail_label
        
        self.input1 = batch1[0].to(self.device)
        self.multi_label1 = batch1[1].to(self.device).float()
        self.label1 = batch1[2].to(self.device).float()
        self.detail_label1 = batch1[3].to(self.device).float()
        
        self.input2 = batch2[0].to(self.device)
        self.multi_label2 = batch2[1].to(self.device).float()
        self.label2 = batch2[2].to(self.device).float()
        self.detail_label2 = batch2[3].to(self.device).float()
        
    # all augment feature seen as unseen class
    # pull in center of all, and push out center of domains
    def feature_augment_mix(self, feature, label):
        def interpolate_push(f_x, f_c):
            alpha = random.uniform(0.6, 1.5)
            new_f_x = f_x + alpha * (f_x - f_c)
            return new_f_x
        def interpolate_pull(f_x, f_c):
            # alpha = random.uniform(0.5, 1.5)
            alpha = random.uniform(0.6, 1.5)
            new_f_x = f_x + alpha * (f_c - f_x)
            return new_f_x    
        
        
        # direct manner: directly push out the feature points from the batch centriod same class 
        center = feature.mean(dim=0, keepdim=True)
        # real class
        real_mask = (label  == 0)
        real_features = feature[real_mask]
        real_center = real_features.mean(dim=0, keepdim=True) 
        aug_fncs = [
            lambda f: interpolate_push(f, real_center),
            lambda f: interpolate_pull(f, center)
        ]
        aug_fnc = random.choice(aug_fncs)
        real_aug_feature = torch.stack([aug_fnc(f) for f in real_features])    
        # fake class
        fake_mask = ~real_mask
        fake_features = feature[fake_mask]
        fake_center = fake_features.mean(dim=0, keepdim=True)    
        aug_fncs = [
            lambda f: interpolate_push(f, fake_center),
            lambda f: interpolate_pull(f, center)
        ]
        aug_fnc = random.choice(aug_fncs)
        fake_aug_feature = torch.stack([aug_fnc(f) for f in fake_features])
        #Concate
        aug_feature = torch.cat((real_aug_feature, fake_aug_feature), dim=0)         
        batch_size = aug_feature.size()[0]
        aug_mul_label = torch.tensor([0.0,0.0,1.0]*batch_size).reshape(batch_size,-1)
        aug_label = torch.tensor([1.0]*batch_size).reshape(batch_size)
        return aug_feature, aug_mul_label, aug_label
    
    
    # all augment feature seen as real/fake class
    # real: push out center of reals / seen as real class [1.0, 0.0, 0.0]
    # fake: push out and pull in center of fakes / seen as fake class [0.0, 1.0, 0.0]
    def feature_augment_mix_v2(self, feature, label):
        def interpolate_push(f_x, f_c):
            alpha = random.uniform(0.5, 1.3)
            new_f_x = f_x + alpha * (f_x - f_c)
            return new_f_x
        def interpolate_pull(f_x, f_c):
            # alpha = random.uniform(0.5, 1.5)
            alpha = random.uniform(0.5, 1.5)
            new_f_x = f_x + alpha * (f_c - f_x)
            return new_f_x       
        
        # direct manner: directly push out the feature points from the batch centriod same class 
        # real class
        real_mask = (label  == 0)
        real_features = feature[real_mask]
        real_center = real_features.mean(dim=0, keepdim=True) 
        aug_fncs = [
            lambda f: interpolate_push(f, real_center),
        ]
        aug_fnc = random.choice(aug_fncs)
        real_aug_feature = torch.stack([aug_fnc(f) for f in real_features])
        real_batch_size = real_aug_feature.size()[0]
        real_aug_mul_label = torch.tensor([1.0,0.0,0.0]*real_batch_size).reshape(real_batch_size,-1)
        real_aug_label = torch.tensor([0.0]*real_batch_size).reshape(real_batch_size)
                
        # fake class
        fake_mask = ~real_mask
        fake_features = feature[fake_mask]
        fake_center = fake_features.mean(dim=0, keepdim=True)    
        aug_fncs = [
            lambda f: interpolate_push(f, fake_center),
            lambda f: interpolate_pull(f, fake_center)
        ]
        # aug_fnc = random.choice(aug_fncs)
        fake_aug_feature = torch.stack([random.choice(aug_fncs)(f) for f in fake_features])
        # fake_aug_feature = torch.stack([aug_fnc(f) for f in fake_features])
        fake_batch_size = fake_aug_feature.size()[0]
        fake_aug_mul_label = torch.tensor([0.0,1.0,0.0]*fake_batch_size).reshape(fake_batch_size,-1)
        fake_aug_label = torch.tensor([1.0]*fake_batch_size).reshape(fake_batch_size)
        #Concate
        aug_feature = torch.cat((real_aug_feature, fake_aug_feature), dim=0)             
        aug_mul_label = torch.cat((real_aug_mul_label, fake_aug_mul_label), dim=0)    
        aug_label = torch.cat((real_aug_label, fake_aug_label), dim=0)  
        return aug_feature, aug_mul_label, aug_label    
    

    # all augment feature seen as 
    # real: push out center of reals / seen as half real class [0.7, 0.0, 0.3]
    # fake: push out and pull in center of fakes / seen as half fake class [0.0, 0.7, 0.3]    
    def feature_augment_mix_v3(self, feature, label):
        def interpolate_push(f_x, f_c):
            alpha = random.uniform(0.5, 1.3)
            new_f_x = f_x + alpha * (f_x - f_c)
            return new_f_x
        def interpolate_pull(f_x, f_c):
            # alpha = random.uniform(0.5, 1.5)
            alpha = random.uniform(0.5, 1.3)
            new_f_x = f_x + alpha * (f_c - f_x)
            return new_f_x       
        
        # direct manner: directly push out the feature points from the batch centriod same class 
        # real class
        real_mask = (label  == 0)
        real_features = feature[real_mask]
        real_center = real_features.mean(dim=0, keepdim=True) 
        aug_fncs = [
            lambda f: interpolate_push(f, real_center),
        ]
        aug_fnc = random.choice(aug_fncs)
        real_aug_feature = torch.stack([aug_fnc(f) for f in real_features])
        real_batch_size = real_aug_feature.size()[0]
        real_aug_mul_label = torch.tensor([0.7,0.0,0.3]*real_batch_size).reshape(real_batch_size,-1)
        real_aug_label = torch.tensor([0.3]*real_batch_size).reshape(real_batch_size)
                
        # fake class
        fake_mask = ~real_mask
        fake_features = feature[fake_mask]
        fake_center = fake_features.mean(dim=0, keepdim=True)    
        aug_fncs = [
            lambda f: interpolate_push(f, fake_center),
            lambda f: interpolate_pull(f, fake_center)
        ]
        # aug_fnc = random.choice(aug_fncs)
        fake_aug_feature = torch.stack([random.choice(aug_fncs)(f) for f in fake_features])
        # fake_aug_feature = torch.stack([aug_fnc(f) for f in fake_features])
        fake_batch_size = fake_aug_feature.size()[0]
        fake_aug_mul_label = torch.tensor([0.0,0.7,0.3]*fake_batch_size).reshape(fake_batch_size,-1)
        fake_aug_label = torch.tensor([0.7]*fake_batch_size).reshape(fake_batch_size)
        #Concate
        aug_feature = torch.cat((real_aug_feature, fake_aug_feature), dim=0)             
        aug_mul_label = torch.cat((real_aug_mul_label, fake_aug_mul_label), dim=0)    
        aug_label = torch.cat((real_aug_label, fake_aug_label), dim=0)  
        return aug_feature, aug_mul_label, aug_label    
    
    # all augment feature seen as unseen class
    # pull in center of all, and push out center of domains
    def feature_augment_mix_v4(self, feature, label):
        def interpolate_push(f_x, f_c):
            alpha = random.uniform(0.1, 1.0)
            new_f_x = f_x + alpha * (f_x - f_c)
            return new_f_x
        def interpolate_pull(f_x, f_c):
            # alpha = random.uniform(0.5, 1.5)
            alpha = random.uniform(0.1, 0.6)
            new_f_x = f_x + alpha * (f_c - f_x)
            return new_f_x    
        
        
        # direct manner: directly push out the feature points from the batch centriod same class 
        center = feature.mean(dim=0, keepdim=True)
        # real class
        real_mask = (label  == 0)
        real_features = feature[real_mask]
        real_center = real_features.mean(dim=0, keepdim=True) 
        aug_fncs = [
            lambda f: interpolate_push(f, real_center),
            lambda f: interpolate_pull(f, center)
        ]
        # aug_fnc = random.choice(aug_fncs)
        real_aug_feature = torch.stack([random.choice(aug_fncs)(f) for f in real_features])
        real_batch_size = real_aug_feature.size()[0]
        real_aug_mul_label = torch.tensor([0.7,0.0,0.3]*real_batch_size).reshape(real_batch_size,-1)
        real_aug_label = torch.tensor([0.3]*real_batch_size).reshape(real_batch_size)        
        # fake class
        fake_mask = ~real_mask
        fake_features = feature[fake_mask]
        fake_center = fake_features.mean(dim=0, keepdim=True)    
        aug_fncs = [
            lambda f: interpolate_push(f, fake_center),
            lambda f: interpolate_pull(f, center)
        ]
        # aug_fnc = random.choice(aug_fncs)
        fake_aug_feature = torch.stack([random.choice(aug_fncs)(f) for f in fake_features])
        fake_batch_size = fake_aug_feature.size()[0]
        fake_aug_mul_label = torch.tensor([0.0,0.7,0.3]*fake_batch_size).reshape(fake_batch_size,-1)
        fake_aug_label = torch.tensor([0.7]*fake_batch_size).reshape(fake_batch_size)
        #Concate
        aug_feature = torch.cat((real_aug_feature, fake_aug_feature), dim=0)             
        aug_mul_label = torch.cat((real_aug_mul_label, fake_aug_mul_label), dim=0)    
        aug_label = torch.cat((real_aug_label, fake_aug_label), dim=0)  
        return aug_feature, aug_mul_label, aug_label
    
    
    def get_loss(self):
        return self.BCE_loss(self.output1.squeeze(1), self.label1)

    def optimize_parameters(self, tri_decay=None):
        # Train multi_classifier with feature augment
        _, feature1 = self.model(self.input1, return_feature=True)
        _, feature2 = self.model(self.input2, return_feature=True)
        # for multi_classifier1 (train with detail_label -1 and 0) 
        detail_mask1_1 = ~(self.detail_label1==1)
        aug_feature1, aug_mul_label1, _ = self.feature_augment_mix_v4(feature1[detail_mask1_1], self.label1[detail_mask1_1])
        detail_mask1_2 = ~(self.detail_label2==1)
        aug_feature2, aug_mul_label2, _ = self.feature_augment_mix_v4(feature2[detail_mask1_2], self.label2[detail_mask1_2])
        
        aug_mul_label1 = aug_mul_label1.to(self.device).float()
        aug_mul_label2 = aug_mul_label2.to(self.device).float()
        
        multi_out1 = self.multi_classifier_1(feature1[detail_mask1_1].detach())
        multi_out2 = self.multi_classifier_1(feature2[detail_mask1_2].detach())
        aug_multi_out1 = self.multi_classifier_1(aug_feature1.detach())
        aug_multi_out2 = self.multi_classifier_1(aug_feature2.detach())
        
        # self.loss_MC1 = (self.CE_loss(multi_out1.squeeze(1), self.multi_label1[detail_mask1_1])+\
                               # self.CE_loss(multi_out2.squeeze(1), self.multi_label2[detail_mask1_2])) / 2
        self.loss_MC1 = (1.0) * (((self.CE_loss(multi_out1.squeeze(1), self.multi_label1[detail_mask1_1])+\
                               self.CE_loss(multi_out2.squeeze(1), self.multi_label2[detail_mask1_2])) / 2) + \
                            0.4 * (self.CE_loss(aug_multi_out1.squeeze(1), aug_mul_label1)+\
                                   self.CE_loss(aug_multi_out2.squeeze(1), aug_mul_label2)) / 2)
        self.optimizer_MC1.zero_grad()
        self.loss_MC1.backward()
        self.optimizer_MC1.step()
        
        # for multi_classifier2 (train with detail_label -1 and 1) 
        detail_mask2_1 = ~(self.detail_label1==0)
        aug_feature1, aug_mul_label1, _ = self.feature_augment_mix_v4(feature1[detail_mask2_1], self.label1[detail_mask2_1])
        detail_mask2_2 = ~(self.detail_label2==0)
        aug_feature2, aug_mul_label2, _ = self.feature_augment_mix_v4(feature2[detail_mask2_2], self.label2[detail_mask2_2])
        
        aug_mul_label1 = aug_mul_label1.to(self.device).float()
        aug_mul_label2 = aug_mul_label2.to(self.device).float()
        
        multi_out1 = self.multi_classifier_2(feature1[detail_mask2_1].detach())
        multi_out2 = self.multi_classifier_2(feature2[detail_mask2_2].detach())
        aug_multi_out1 = self.multi_classifier_2(aug_feature1.detach())
        aug_multi_out2 = self.multi_classifier_2(aug_feature2.detach())
        
        # self.loss_MC2 = (self.CE_loss(multi_out1.squeeze(1), self.multi_label1[detail_mask2_1])+\
                               # self.CE_loss(multi_out2.squeeze(1), self.multi_label2[detail_mask2_2])) / 2
        self.loss_MC2 = (1.0) * (((self.CE_loss(multi_out1.squeeze(1), self.multi_label1[detail_mask2_1])+ \
                               self.CE_loss(multi_out2.squeeze(1), self.multi_label2[detail_mask2_2])) / 2) + \
                            0.4 * (self.CE_loss(aug_multi_out1.squeeze(1), aug_mul_label1)+ \
                                   self.CE_loss(aug_multi_out2.squeeze(1), aug_mul_label2)) / 2)
        self.optimizer_MC2.zero_grad()
        self.loss_MC2.backward()
        self.optimizer_MC2.step()
        
        
        # Train Feature extractor
        _, feature1 = self.model(self.input1, return_feature=True)
        _, feature2 = self.model(self.input2, return_feature=True)
        
        multi_out1_1 = self.multi_classifier_1(feature1[detail_mask1_1])
        multi_out1_2 = self.multi_classifier_1(feature2[detail_mask1_2])
        multi_out2_1 = self.multi_classifier_2(feature1[detail_mask2_1])
        multi_out2_2 = self.multi_classifier_2(feature2[detail_mask2_2])
        
        
        self.loss_F = (1.0) * ((self.CE_loss(multi_out1_1.squeeze(1), self.multi_label1[detail_mask1_1])+ \
                               self.CE_loss(multi_out1_2.squeeze(1), self.multi_label2[detail_mask1_2])) / 2) + \
                      (0.4) * ((self.CE_loss(multi_out2_1.squeeze(1), self.multi_label1[detail_mask2_1])+ \
                               self.CE_loss(multi_out2_2.squeeze(1), self.multi_label2[detail_mask2_2])) / 2)         
                               
        self.optimizer_F.zero_grad()
        self.loss_F.backward()
        self.optimizer_F.step()
        
        # Train binary classifier with feature augment       
        _, feature1 = self.model(self.input1, return_feature=True)
        _, feature2 = self.model(self.input2, return_feature=True)
        aug_feature1, _, aug_label1 = self.feature_augment_mix_v4(feature1, self.label1)
        aug_feature2, _, aug_label2 = self.feature_augment_mix_v4(feature2, self.label2)

        aug_label1 = aug_label1.to(self.device).float()
        aug_label2 = aug_label2.to(self.device).float()
        aug_bin_out1 = self.classifier(aug_feature1.detach())
        aug_bin_out2 = self.classifier(aug_feature2.detach())
        
        loss_BC_aug = 0.4 * (self.BCE_loss(aug_bin_out1.squeeze(), aug_label1)+ \
                              self.BCE_loss(aug_bin_out2.squeeze(), aug_label2) / 2)
                              
        bin_out1_1 = self.classifier(feature1[detail_mask1_1].detach())
        bin_out1_2 = self.classifier(feature2[detail_mask1_2].detach())                      
        # self.loss_BC = 1.0 * (self.BCE_loss(bin_out1_1.squeeze(1), self.label1[detail_mask1_1) + \
                              # self.BCE_loss(bin_out1_2.squeeze(1), self.label2[detail_mask1_2)) / 2 
        loss_BC1 = 1.0 * ((self.BCE_loss(bin_out1_1.squeeze(1), self.label1[detail_mask1_1]) + \
                           self.BCE_loss(bin_out1_2.squeeze(1), self.label2[detail_mask1_2])) / 2)
                           
        bin_out2_1 = self.classifier(feature1[detail_mask2_1].detach())
        bin_out2_2 = self.classifier(feature2[detail_mask2_2].detach())                      
        # self.loss_BC = 1.0 * (self.BCE_loss(bin_out2_1.squeeze(1), self.label1[detail_mask2_1) + \
                              # self.BCE_loss(bin_out2_2.squeeze(1), self.label2[detail_mask2_2)) / 2 
        loss_BC2 = 0.4 * ((self.BCE_loss(bin_out2_1.squeeze(1), self.label1[detail_mask2_1]) + \
                           self.BCE_loss(bin_out2_2.squeeze(1), self.label2[detail_mask2_2])) / 2)
                           
        self.loss_BC = loss_BC1 + loss_BC2 + loss_BC_aug                    
        self.optimizer_BC.zero_grad()
        self.loss_BC.backward()
        self.optimizer_BC.step()
        
        self.loss = (self.loss_BC + self.loss_MC1+self.loss_MC2 + self.loss_F) / 4

        
        
class mix_Trainer_v3(BaseModel):
    def name(self):
        return 'mix_Trainer_v3'

    def __init__(self, opt):
        super(mix_Trainer_v3, self).__init__(opt)

        if self.isTrain and not opt.continue_train:
            self.model = resnet50(num_classes=1)
            torch.nn.init.normal_(self.model.fc.weight.data, 0.0, opt.init_gain)
            if opt.pretrained_path != None:
                state_dict = torch.load(opt.pretrained_path, map_location='cpu')
                self.model.load_state_dict(state_dict['model'])
            else:
                print(f"your pretrained_path is empty, please set the path.")
            
            # parallel classifier 1
            self.multi_classifier_1 = Classifier(num_classes=3, input_size=1024)
            torch.nn.init.normal_(self.multi_classifier_1.fc1.weight.data, 0.0, opt.init_gain)
            torch.nn.init.normal_(self.multi_classifier_1.fc2.weight.data, 0.0, opt.init_gain)
            
            # parallel classifier 2
            self.multi_classifier_2 = Classifier(num_classes=3, input_size=1024)
            torch.nn.init.normal_(self.multi_classifier_2.fc1.weight.data, 0.0, opt.init_gain)
            torch.nn.init.normal_(self.multi_classifier_2.fc2.weight.data, 0.0, opt.init_gain)
            
            
            self.classifier = Classifier(num_classes=1)
            torch.nn.init.normal_(self.classifier.fc1.weight.data, 0.0, opt.init_gain)
            torch.nn.init.normal_(self.classifier.fc2.weight.data, 0.0, opt.init_gain)
            
        # Trainable parameters    
        params_F = []
    
        for name, p in self.model.named_parameters():
            if  name=="fc.weight" or name=="fc.bias": 
                p.requires_grad = False
            else:
                params_F.append(p)

        params_Multi_C1 = self.multi_classifier_1.parameters()          
        params_Multi_C2 = self.multi_classifier_2.parameters()         
        params_Binary_C = self.classifier.parameters()            
        
        if self.isTrain:
            self.BCE_loss = nn.BCEWithLogitsLoss()
            self.CE_loss = nn.CrossEntropyLoss()
            # initialize optimizers
            if opt.optim == 'adam':
                self.optimizer_F = torch.optim.Adam(params_F,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_MC1 = torch.optim.Adam(params_Multi_C1,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_MC2 = torch.optim.Adam(params_Multi_C2,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))                                  
                self.optimizer_BC = torch.optim.Adam(params_Binary_C,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))                                  
            elif opt.optim == 'sgd':
                self.optimizer_F = torch.optim.SGD(params_F,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_MC1 = torch.optim.SGD(params_Multi_C1,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_MC2 = torch.optim.Adam(params_Multi_C2,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))                                   
                self.optimizer_BC = torch.optim.SGD(params_Binary_C,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))  
            else:
                raise ValueError("optim should be [adam, sgd]")
            
        self.model.to(opt.gpu_ids[0])
        self.classifier.to(opt.gpu_ids[0])
        self.multi_classifier_1.to(opt.gpu_ids[0])
        self.multi_classifier_2.to(opt.gpu_ids[0])
        

    def adjust_learning_rate(self, min_lr=1e-7):
        for param_group in self.optimizer_F.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        for param_group in self.optimizer_MC1.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        for param_group in self.optimizer_MC2.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False        
        for param_group in self.optimizer_BC.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False        
        return True

    def set_input(self, batch1, batch2):
        # img, multi_label, label, detail_label
        
        self.input1 = batch1[0].to(self.device)
        self.multi_label1 = batch1[1].to(self.device).float()
        self.label1 = batch1[2].to(self.device).float()
        self.detail_label1 = batch1[3].to(self.device).float()
        
        self.input2 = batch2[0].to(self.device)
        self.multi_label2 = batch2[1].to(self.device).float()
        self.label2 = batch2[2].to(self.device).float()
        self.detail_label2 = batch2[3].to(self.device).float()
        
    # all augment feature seen as unseen class
    # pull in center of all, and push out center of domains
    def feature_augment_mix(self, feature, label):
        def interpolate_push(f_x, f_c):
            alpha = random.uniform(0.6, 1.5)
            new_f_x = f_x + alpha * (f_x - f_c)
            return new_f_x
        def interpolate_pull(f_x, f_c):
            # alpha = random.uniform(0.5, 1.5)
            alpha = random.uniform(0.6, 1.5)
            new_f_x = f_x + alpha * (f_c - f_x)
            return new_f_x    
        
        
        # direct manner: directly push out the feature points from the batch centriod same class 
        center = feature.mean(dim=0, keepdim=True)
        # real class
        real_mask = (label  == 0)
        real_features = feature[real_mask]
        real_center = real_features.mean(dim=0, keepdim=True) 
        aug_fncs = [
            lambda f: interpolate_push(f, real_center),
            lambda f: interpolate_pull(f, center)
        ]
        aug_fnc = random.choice(aug_fncs)
        real_aug_feature = torch.stack([aug_fnc(f) for f in real_features])    
        # fake class
        fake_mask = ~real_mask
        fake_features = feature[fake_mask]
        fake_center = fake_features.mean(dim=0, keepdim=True)    
        aug_fncs = [
            lambda f: interpolate_push(f, fake_center),
            lambda f: interpolate_pull(f, center)
        ]
        aug_fnc = random.choice(aug_fncs)
        fake_aug_feature = torch.stack([aug_fnc(f) for f in fake_features])
        #Concate
        aug_feature = torch.cat((real_aug_feature, fake_aug_feature), dim=0)         
        batch_size = aug_feature.size()[0]
        aug_mul_label = torch.tensor([0.0,0.0,1.0]*batch_size).reshape(batch_size,-1)
        aug_label = torch.tensor([1.0]*batch_size).reshape(batch_size)
        return aug_feature, aug_mul_label, aug_label
    
    
    # all augment feature seen as real/fake class
    # real: push out center of reals / seen as real class [1.0, 0.0, 0.0]
    # fake: push out and pull in center of fakes / seen as fake class [0.0, 1.0, 0.0]
    def feature_augment_mix_v2(self, feature, label):
        def interpolate_push(f_x, f_c):
            alpha = random.uniform(0.5, 1.3)
            new_f_x = f_x + alpha * (f_x - f_c)
            return new_f_x
        def interpolate_pull(f_x, f_c):
            # alpha = random.uniform(0.5, 1.5)
            alpha = random.uniform(0.5, 1.5)
            new_f_x = f_x + alpha * (f_c - f_x)
            return new_f_x       
        
        # direct manner: directly push out the feature points from the batch centriod same class 
        # real class
        real_mask = (label  == 0)
        real_features = feature[real_mask]
        real_center = real_features.mean(dim=0, keepdim=True) 
        aug_fncs = [
            lambda f: interpolate_push(f, real_center),
        ]
        aug_fnc = random.choice(aug_fncs)
        real_aug_feature = torch.stack([aug_fnc(f) for f in real_features])
        real_batch_size = real_aug_feature.size()[0]
        real_aug_mul_label = torch.tensor([1.0,0.0,0.0]*real_batch_size).reshape(real_batch_size,-1)
        real_aug_label = torch.tensor([0.0]*real_batch_size).reshape(real_batch_size)
                
        # fake class
        fake_mask = ~real_mask
        fake_features = feature[fake_mask]
        fake_center = fake_features.mean(dim=0, keepdim=True)    
        aug_fncs = [
            lambda f: interpolate_push(f, fake_center),
            lambda f: interpolate_pull(f, fake_center)
        ]
        # aug_fnc = random.choice(aug_fncs)
        fake_aug_feature = torch.stack([random.choice(aug_fncs)(f) for f in fake_features])
        # fake_aug_feature = torch.stack([aug_fnc(f) for f in fake_features])
        fake_batch_size = fake_aug_feature.size()[0]
        fake_aug_mul_label = torch.tensor([0.0,1.0,0.0]*fake_batch_size).reshape(fake_batch_size,-1)
        fake_aug_label = torch.tensor([1.0]*fake_batch_size).reshape(fake_batch_size)
        #Concate
        aug_feature = torch.cat((real_aug_feature, fake_aug_feature), dim=0)             
        aug_mul_label = torch.cat((real_aug_mul_label, fake_aug_mul_label), dim=0)    
        aug_label = torch.cat((real_aug_label, fake_aug_label), dim=0)  
        return aug_feature, aug_mul_label, aug_label    
    

    # all augment feature seen as 
    # real: push out center of reals / seen as half real class [0.7, 0.0, 0.3]
    # fake: push out and pull in center of fakes / seen as half fake class [0.0, 0.7, 0.3]    
    def feature_augment_mix_v3(self, feature, label):
        def interpolate_push(f_x, f_c):
            alpha = random.uniform(0.5, 1.3)
            new_f_x = f_x + alpha * (f_x - f_c)
            return new_f_x
        def interpolate_pull(f_x, f_c):
            # alpha = random.uniform(0.5, 1.5)
            alpha = random.uniform(0.5, 1.3)
            new_f_x = f_x + alpha * (f_c - f_x)
            return new_f_x       
        
        # direct manner: directly push out the feature points from the batch centriod same class 
        # real class
        real_mask = (label  == 0)
        real_features = feature[real_mask]
        real_center = real_features.mean(dim=0, keepdim=True) 
        aug_fncs = [
            lambda f: interpolate_push(f, real_center),
        ]
        aug_fnc = random.choice(aug_fncs)
        real_aug_feature = torch.stack([aug_fnc(f) for f in real_features])
        real_batch_size = real_aug_feature.size()[0]
        real_aug_mul_label = torch.tensor([0.7,0.0,0.3]*real_batch_size).reshape(real_batch_size,-1)
        real_aug_label = torch.tensor([0.3]*real_batch_size).reshape(real_batch_size)
                
        # fake class
        fake_mask = ~real_mask
        fake_features = feature[fake_mask]
        fake_center = fake_features.mean(dim=0, keepdim=True)    
        aug_fncs = [
            lambda f: interpolate_push(f, fake_center),
            lambda f: interpolate_pull(f, fake_center)
        ]
        # aug_fnc = random.choice(aug_fncs)
        fake_aug_feature = torch.stack([random.choice(aug_fncs)(f) for f in fake_features])
        # fake_aug_feature = torch.stack([aug_fnc(f) for f in fake_features])
        fake_batch_size = fake_aug_feature.size()[0]
        fake_aug_mul_label = torch.tensor([0.0,0.7,0.3]*fake_batch_size).reshape(fake_batch_size,-1)
        fake_aug_label = torch.tensor([0.7]*fake_batch_size).reshape(fake_batch_size)
        #Concate
        aug_feature = torch.cat((real_aug_feature, fake_aug_feature), dim=0)             
        aug_mul_label = torch.cat((real_aug_mul_label, fake_aug_mul_label), dim=0)    
        aug_label = torch.cat((real_aug_label, fake_aug_label), dim=0)  
        return aug_feature, aug_mul_label, aug_label    
    
    # all augment feature seen as unseen class
    # pull in center of all, and push out center of domains
    def feature_augment_mix_v4(self, feature, label):
        def interpolate_push(f_x, f_c):
            alpha = random.uniform(0.1, 1.0)
            new_f_x = f_x + alpha * (f_x - f_c)
            return new_f_x
        def interpolate_pull(f_x, f_c):
            # alpha = random.uniform(0.5, 1.5)
            alpha = random.uniform(0.1, 0.6)
            new_f_x = f_x + alpha * (f_c - f_x)
            return new_f_x    
        
        
        # direct manner: directly push out the feature points from the batch centriod same class 
        center = feature.mean(dim=0, keepdim=True)
        # real class
        real_mask = (label  == 0)
        real_features = feature[real_mask]
        real_center = real_features.mean(dim=0, keepdim=True) 
        aug_fncs = [
            lambda f: interpolate_push(f, real_center),
            lambda f: interpolate_pull(f, center)
        ]
        # aug_fnc = random.choice(aug_fncs)
        real_aug_feature = torch.stack([random.choice(aug_fncs)(f) for f in real_features])
        real_batch_size = real_aug_feature.size()[0]
        real_aug_mul_label = torch.tensor([0.7,0.0,0.3]*real_batch_size).reshape(real_batch_size,-1)
        real_aug_label = torch.tensor([0.3]*real_batch_size).reshape(real_batch_size)        
        # fake class
        fake_mask = ~real_mask
        fake_features = feature[fake_mask]
        fake_center = fake_features.mean(dim=0, keepdim=True)    
        aug_fncs = [
            lambda f: interpolate_push(f, fake_center),
            lambda f: interpolate_pull(f, center)
        ]
        # aug_fnc = random.choice(aug_fncs)
        fake_aug_feature = torch.stack([random.choice(aug_fncs)(f) for f in fake_features])
        fake_batch_size = fake_aug_feature.size()[0]
        fake_aug_mul_label = torch.tensor([0.0,0.7,0.3]*fake_batch_size).reshape(fake_batch_size,-1)
        fake_aug_label = torch.tensor([0.7]*fake_batch_size).reshape(fake_batch_size)
        #Concate
        aug_feature = torch.cat((real_aug_feature, fake_aug_feature), dim=0)             
        aug_mul_label = torch.cat((real_aug_mul_label, fake_aug_mul_label), dim=0)    
        aug_label = torch.cat((real_aug_label, fake_aug_label), dim=0)  
        return aug_feature, aug_mul_label, aug_label
    
    
    def get_loss(self):
        return self.BCE_loss(self.output1.squeeze(1), self.label1)

    def optimize_parameters(self, tri_decay=None):
        # Train multi_classifier with feature augment
        _, feature1_1, feature2_1 = self.model(self.input1, return_feature=True, split_feature=True)
        _, feature1_2, feature2_2 = self.model(self.input2, return_feature=True, split_feature=True)

        # for multi_classifier1 (train with detail_label -1 and 0) 
        detail_mask1_1 = ~(self.detail_label1==1)
        aug_feature1, aug_mul_label1, _ = self.feature_augment_mix_v3(feature1_1[detail_mask1_1], self.label1[detail_mask1_1])
        detail_mask1_2 = ~(self.detail_label2==1)
        aug_feature2, aug_mul_label2, _ = self.feature_augment_mix_v3(feature1_2[detail_mask1_2], self.label2[detail_mask1_2])
        
        aug_mul_label1 = aug_mul_label1.to(self.device).float()
        aug_mul_label2 = aug_mul_label2.to(self.device).float()
        
        multi_out1 = self.multi_classifier_1(feature1_1[detail_mask1_1].detach())
        multi_out2 = self.multi_classifier_1(feature1_2[detail_mask1_2].detach())
        aug_multi_out1 = self.multi_classifier_1(aug_feature1.detach())
        aug_multi_out2 = self.multi_classifier_1(aug_feature2.detach())
        
        # self.loss_MC1 = (self.CE_loss(multi_out1.squeeze(1), self.multi_label1[detail_mask1_1])+\
                               # self.CE_loss(multi_out2.squeeze(1), self.multi_label2[detail_mask1_2])) / 2
        self.loss_MC1 = (1.0) * (((self.CE_loss(multi_out1.squeeze(1), self.multi_label1[detail_mask1_1])+\
                               self.CE_loss(multi_out2.squeeze(1), self.multi_label2[detail_mask1_2])) / 2) + \
                            0.4 * (self.CE_loss(aug_multi_out1.squeeze(1), aug_mul_label1)+\
                                   self.CE_loss(aug_multi_out2.squeeze(1), aug_mul_label2)) / 2)
        self.optimizer_MC1.zero_grad()
        self.loss_MC1.backward()
        self.optimizer_MC1.step()
        
        # for multi_classifier2 (train with detail_label -1 and 1) 
        detail_mask2_1 = ~(self.detail_label1==0)
        aug_feature1, aug_mul_label1, _ = self.feature_augment_mix_v3(feature2_1[detail_mask2_1], self.label1[detail_mask2_1])
        detail_mask2_2 = ~(self.detail_label2==0)
        aug_feature2, aug_mul_label2, _ = self.feature_augment_mix_v3(feature2_2[detail_mask2_2], self.label2[detail_mask2_2])
        
        aug_mul_label1 = aug_mul_label1.to(self.device).float()
        aug_mul_label2 = aug_mul_label2.to(self.device).float()
        
        multi_out1 = self.multi_classifier_2(feature2_1[detail_mask2_1].detach())
        multi_out2 = self.multi_classifier_2(feature2_2[detail_mask2_2].detach())
        aug_multi_out1 = self.multi_classifier_2(aug_feature1.detach())
        aug_multi_out2 = self.multi_classifier_2(aug_feature2.detach())
        
        # self.loss_MC2 = (self.CE_loss(multi_out1.squeeze(1), self.multi_label1[detail_mask2_1])+\
                               # self.CE_loss(multi_out2.squeeze(1), self.multi_label2[detail_mask2_2])) / 2
        self.loss_MC2 = (1.0) * (((self.CE_loss(multi_out1.squeeze(1), self.multi_label1[detail_mask2_1])+ \
                               self.CE_loss(multi_out2.squeeze(1), self.multi_label2[detail_mask2_2])) / 2) + \
                            0.4 * (self.CE_loss(aug_multi_out1.squeeze(1), aug_mul_label1)+ \
                                   self.CE_loss(aug_multi_out2.squeeze(1), aug_mul_label2)) / 2)
        self.optimizer_MC2.zero_grad()
        self.loss_MC2.backward()
        self.optimizer_MC2.step()
        
        
        # Train Feature extractor
        _, feature1_1, feature2_1 = self.model(self.input1, return_feature=True, split_feature=True)
        _, feature1_2, feature2_2 = self.model(self.input2, return_feature=True, split_feature=True)
        
        multi_out1_1 = self.multi_classifier_1(feature1_1[detail_mask1_1])
        multi_out1_2 = self.multi_classifier_1(feature1_2[detail_mask1_2])
        multi_out2_1 = self.multi_classifier_2(feature2_1[detail_mask2_1])
        multi_out2_2 = self.multi_classifier_2(feature2_2[detail_mask2_2])
        
        
        self.loss_F = (1.0) * ((self.CE_loss(multi_out1_1.squeeze(1), self.multi_label1[detail_mask1_1])+ \
                               self.CE_loss(multi_out1_2.squeeze(1), self.multi_label2[detail_mask1_2])) / 2) + \
                      (0.4) * ((self.CE_loss(multi_out2_1.squeeze(1), self.multi_label1[detail_mask2_1])+ \
                               self.CE_loss(multi_out2_2.squeeze(1), self.multi_label2[detail_mask2_2])) / 2)         
                               
        self.optimizer_F.zero_grad()
        self.loss_F.backward()
        self.optimizer_F.step()
        
        # Train binary classifier with feature augment
        _, feature1 = self.model(self.input1, return_feature=True)
        _, feature2 = self.model(self.input2, return_feature=True)
        aug_feature1, _, aug_label1 = self.feature_augment_mix_v3(feature1, self.label1)
        aug_feature2, _, aug_label2 = self.feature_augment_mix_v3(feature2, self.label2)
        aug_label1 = aug_label1.to(self.device).float()
        aug_label2 = aug_label2.to(self.device).float()
        aug_bin_out1 = self.classifier(aug_feature1.detach())
        aug_bin_out2 = self.classifier(aug_feature2.detach())
        
        loss_BC_aug = 0.4 * (self.BCE_loss(aug_bin_out1.squeeze(), aug_label1)+ \
                              self.BCE_loss(aug_bin_out2.squeeze(), aug_label2) / 2)
        
        bin_out1_1 = self.classifier(feature1[detail_mask1_1].detach())
        bin_out1_2 = self.classifier(feature2[detail_mask1_2].detach())
        # self.loss_BC = 1.0 * (self.BCE_loss(bin_out1_1.squeeze(1), self.label1[detail_mask1_1]) + \
                              # self.BCE_loss(bin_out1_2.squeeze(1), self.label2[detail_mask1_2])) / 2                             
        loss_BC_1 = 1.0 * ((self.BCE_loss(bin_out1_1.squeeze(1), self.label1[detail_mask1_1]) + \
                              self.BCE_loss(bin_out1_2.squeeze(1), self.label2[detail_mask1_2])) / 2)
        
        bin_out2_1 = self.classifier(feature1[detail_mask2_1].detach())
        bin_out2_2 = self.classifier(feature2[detail_mask2_2].detach())
        # self.loss_BC = 1.0 * (self.BCE_loss(bin_out1_1.squeeze(1), self.label1[detail_mask1_1]) + \
                              # self.BCE_loss(bin_out1_2.squeeze(1), self.label2[detail_mask1_2])) / 2                             
        loss_BC_2 = 0.4 * ((self.BCE_loss(bin_out2_1.squeeze(1), self.label1[detail_mask2_1]) + \
                              self.BCE_loss(bin_out2_2.squeeze(1), self.label2[detail_mask2_2])) / 2)
                              
        self.loss_BC = loss_BC_1 + loss_BC_2 + loss_BC_aug
        self.optimizer_BC.zero_grad()
        self.loss_BC.backward()
        self.optimizer_BC.step()
        
        self.loss = (self.loss_BC + self.loss_MC1+self.loss_MC2 + self.loss_F) / 4
        
class mix_Trainer_v3_2(BaseModel):
    def name(self):
        return 'mix_Trainer_v3_2'

    def __init__(self, opt):
        super(mix_Trainer_v3_2, self).__init__(opt)

        if self.isTrain and not opt.continue_train:
            self.model = resnet50(num_classes=1)
            torch.nn.init.normal_(self.model.fc.weight.data, 0.0, opt.init_gain)
            if opt.pretrained_path != None:
                state_dict = torch.load(opt.pretrained_path, map_location='cpu')
                self.model.load_state_dict(state_dict['model'])
            else:
                print(f"your pretrained_path is empty, please set the path.")
            
            # parallel classifier 1
            self.multi_classifier_1 = Classifier(num_classes=3, input_size=1024)
            torch.nn.init.normal_(self.multi_classifier_1.fc1.weight.data, 0.0, opt.init_gain)
            torch.nn.init.normal_(self.multi_classifier_1.fc2.weight.data, 0.0, opt.init_gain)
            
            # parallel classifier 2
            self.multi_classifier_2 = Classifier(num_classes=3, input_size=1024)
            torch.nn.init.normal_(self.multi_classifier_2.fc1.weight.data, 0.0, opt.init_gain)
            torch.nn.init.normal_(self.multi_classifier_2.fc2.weight.data, 0.0, opt.init_gain)
            
            
            self.classifier = Classifier(num_classes=1)
            torch.nn.init.normal_(self.classifier.fc1.weight.data, 0.0, opt.init_gain)
            torch.nn.init.normal_(self.classifier.fc2.weight.data, 0.0, opt.init_gain)
            
        # Trainable parameters    
        params_F = []
    
        for name, p in self.model.named_parameters():
            if  name=="fc.weight" or name=="fc.bias": 
                p.requires_grad = False
            else:
                params_F.append(p)

        params_Multi_C1 = self.multi_classifier_1.parameters()          
        params_Multi_C2 = self.multi_classifier_2.parameters()         
        params_Binary_C = self.classifier.parameters()            
        
        if self.isTrain:
            self.BCE_loss = nn.BCEWithLogitsLoss()
            self.CE_loss = nn.CrossEntropyLoss()
            # initialize optimizers
            if opt.optim == 'adam':
                self.optimizer_F = torch.optim.Adam(params_F,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_MC1 = torch.optim.Adam(params_Multi_C1,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_MC2 = torch.optim.Adam(params_Multi_C2,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))                                  
                self.optimizer_BC = torch.optim.Adam(params_Binary_C,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))                                  
            elif opt.optim == 'sgd':
                self.optimizer_F = torch.optim.SGD(params_F,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_MC1 = torch.optim.SGD(params_Multi_C1,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_MC2 = torch.optim.Adam(params_Multi_C2,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))                                   
                self.optimizer_BC = torch.optim.SGD(params_Binary_C,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))  
            else:
                raise ValueError("optim should be [adam, sgd]")
            
        self.model.to(opt.gpu_ids[0])
        self.classifier.to(opt.gpu_ids[0])
        self.multi_classifier_1.to(opt.gpu_ids[0])
        self.multi_classifier_2.to(opt.gpu_ids[0])
        

    def adjust_learning_rate(self, min_lr=1e-7):
        for param_group in self.optimizer_F.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        for param_group in self.optimizer_MC1.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        for param_group in self.optimizer_MC2.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False        
        for param_group in self.optimizer_BC.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False        
        return True

    def set_input(self, batch1, batch2):
        # img, multi_label, label, detail_label
        
        self.input1 = batch1[0].to(self.device)
        self.multi_label1 = batch1[1].to(self.device).float()
        self.label1 = batch1[2].to(self.device).float()
        self.detail_label1 = batch1[3].to(self.device).float()
        
        self.input2 = batch2[0].to(self.device)
        self.multi_label2 = batch2[1].to(self.device).float()
        self.label2 = batch2[2].to(self.device).float()
        self.detail_label2 = batch2[3].to(self.device).float()
        
    # all augment feature seen as unseen class
    # pull in center of all, and push out center of domains
    def feature_augment_mix(self, feature, label):
        def interpolate_push(f_x, f_c):
            alpha = random.uniform(0.6, 1.5)
            new_f_x = f_x + alpha * (f_x - f_c)
            return new_f_x
        def interpolate_pull(f_x, f_c):
            # alpha = random.uniform(0.5, 1.5)
            alpha = random.uniform(0.6, 1.5)
            new_f_x = f_x + alpha * (f_c - f_x)
            return new_f_x    
        
        
        # direct manner: directly push out the feature points from the batch centriod same class 
        center = feature.mean(dim=0, keepdim=True)
        # real class
        real_mask = (label  == 0)
        real_features = feature[real_mask]
        real_center = real_features.mean(dim=0, keepdim=True) 
        aug_fncs = [
            lambda f: interpolate_push(f, real_center),
            lambda f: interpolate_pull(f, center)
        ]
        aug_fnc = random.choice(aug_fncs)
        real_aug_feature = torch.stack([aug_fnc(f) for f in real_features])    
        # fake class
        fake_mask = ~real_mask
        fake_features = feature[fake_mask]
        fake_center = fake_features.mean(dim=0, keepdim=True)    
        aug_fncs = [
            lambda f: interpolate_push(f, fake_center),
            lambda f: interpolate_pull(f, center)
        ]
        aug_fnc = random.choice(aug_fncs)
        fake_aug_feature = torch.stack([aug_fnc(f) for f in fake_features])
        #Concate
        aug_feature = torch.cat((real_aug_feature, fake_aug_feature), dim=0)         
        batch_size = aug_feature.size()[0]
        aug_mul_label = torch.tensor([0.0,0.0,1.0]*batch_size).reshape(batch_size,-1)
        aug_label = torch.tensor([1.0]*batch_size).reshape(batch_size)
        return aug_feature, aug_mul_label, aug_label
    
    
    # all augment feature seen as real/fake class
    # real: push out center of reals / seen as real class [1.0, 0.0, 0.0]
    # fake: push out and pull in center of fakes / seen as fake class [0.0, 1.0, 0.0]
    def feature_augment_mix_v2(self, feature, label):
        def interpolate_push(f_x, f_c):
            alpha = random.uniform(0.5, 1.3)
            new_f_x = f_x + alpha * (f_x - f_c)
            return new_f_x
        def interpolate_pull(f_x, f_c):
            # alpha = random.uniform(0.5, 1.5)
            alpha = random.uniform(0.5, 1.5)
            new_f_x = f_x + alpha * (f_c - f_x)
            return new_f_x       
        
        # direct manner: directly push out the feature points from the batch centriod same class 
        # real class
        real_mask = (label  == 0)
        real_features = feature[real_mask]
        real_center = real_features.mean(dim=0, keepdim=True) 
        aug_fncs = [
            lambda f: interpolate_push(f, real_center),
        ]
        aug_fnc = random.choice(aug_fncs)
        real_aug_feature = torch.stack([aug_fnc(f) for f in real_features])
        real_batch_size = real_aug_feature.size()[0]
        real_aug_mul_label = torch.tensor([1.0,0.0,0.0]*real_batch_size).reshape(real_batch_size,-1)
        real_aug_label = torch.tensor([0.0]*real_batch_size).reshape(real_batch_size)
                
        # fake class
        fake_mask = ~real_mask
        fake_features = feature[fake_mask]
        fake_center = fake_features.mean(dim=0, keepdim=True)    
        aug_fncs = [
            lambda f: interpolate_push(f, fake_center),
            lambda f: interpolate_pull(f, fake_center)
        ]
        # aug_fnc = random.choice(aug_fncs)
        fake_aug_feature = torch.stack([random.choice(aug_fncs)(f) for f in fake_features])
        # fake_aug_feature = torch.stack([aug_fnc(f) for f in fake_features])
        fake_batch_size = fake_aug_feature.size()[0]
        fake_aug_mul_label = torch.tensor([0.0,1.0,0.0]*fake_batch_size).reshape(fake_batch_size,-1)
        fake_aug_label = torch.tensor([1.0]*fake_batch_size).reshape(fake_batch_size)
        #Concate
        aug_feature = torch.cat((real_aug_feature, fake_aug_feature), dim=0)             
        aug_mul_label = torch.cat((real_aug_mul_label, fake_aug_mul_label), dim=0)    
        aug_label = torch.cat((real_aug_label, fake_aug_label), dim=0)  
        return aug_feature, aug_mul_label, aug_label    
    

    # all augment feature seen as 
    # real: push out center of reals / seen as half real class [0.7, 0.0, 0.3]
    # fake: push out and pull in center of fakes / seen as half fake class [0.0, 0.7, 0.3]    
    def feature_augment_mix_v3(self, feature, label):
        def interpolate_push(f_x, f_c):
            alpha = random.uniform(0.5, 1.3)
            new_f_x = f_x + alpha * (f_x - f_c)
            return new_f_x
        def interpolate_pull(f_x, f_c):
            # alpha = random.uniform(0.5, 1.5)
            alpha = random.uniform(0.5, 1.3)
            new_f_x = f_x + alpha * (f_c - f_x)
            return new_f_x       
        
        # direct manner: directly push out the feature points from the batch centriod same class 
        # real class
        real_mask = (label  == 0)
        real_features = feature[real_mask]
        real_center = real_features.mean(dim=0, keepdim=True) 
        aug_fncs = [
            lambda f: interpolate_push(f, real_center),
        ]
        aug_fnc = random.choice(aug_fncs)
        real_aug_feature = torch.stack([aug_fnc(f) for f in real_features])
        real_batch_size = real_aug_feature.size()[0]
        real_aug_mul_label = torch.tensor([0.7,0.0,0.3]*real_batch_size).reshape(real_batch_size,-1)
        real_aug_label = torch.tensor([0.3]*real_batch_size).reshape(real_batch_size)
                
        # fake class
        fake_mask = ~real_mask
        fake_features = feature[fake_mask]
        fake_center = fake_features.mean(dim=0, keepdim=True)    
        aug_fncs = [
            lambda f: interpolate_push(f, fake_center),
            lambda f: interpolate_pull(f, fake_center)
        ]
        # aug_fnc = random.choice(aug_fncs)
        fake_aug_feature = torch.stack([random.choice(aug_fncs)(f) for f in fake_features])
        # fake_aug_feature = torch.stack([aug_fnc(f) for f in fake_features])
        fake_batch_size = fake_aug_feature.size()[0]
        fake_aug_mul_label = torch.tensor([0.0,0.7,0.3]*fake_batch_size).reshape(fake_batch_size,-1)
        fake_aug_label = torch.tensor([0.7]*fake_batch_size).reshape(fake_batch_size)
        #Concate
        aug_feature = torch.cat((real_aug_feature, fake_aug_feature), dim=0)             
        aug_mul_label = torch.cat((real_aug_mul_label, fake_aug_mul_label), dim=0)    
        aug_label = torch.cat((real_aug_label, fake_aug_label), dim=0)  
        return aug_feature, aug_mul_label, aug_label    
    
    # all augment feature seen as unseen class
    # pull in center of all, and push out center of domains
    def feature_augment_mix_v4(self, feature, label):
        def interpolate_push(f_x, f_c):
            alpha = random.uniform(0.1, 1.0)
            new_f_x = f_x + alpha * (f_x - f_c)
            return new_f_x
        def interpolate_pull(f_x, f_c):
            # alpha = random.uniform(0.5, 1.5)
            alpha = random.uniform(0.1, 0.6)
            new_f_x = f_x + alpha * (f_c - f_x)
            return new_f_x    
        
        
        # direct manner: directly push out the feature points from the batch centriod same class 
        center = feature.mean(dim=0, keepdim=True)
        # real class
        real_mask = (label  == 0)
        real_features = feature[real_mask]
        real_center = real_features.mean(dim=0, keepdim=True) 
        aug_fncs = [
            lambda f: interpolate_push(f, real_center),
            lambda f: interpolate_pull(f, center)
        ]
        # aug_fnc = random.choice(aug_fncs)
        real_aug_feature = torch.stack([random.choice(aug_fncs)(f) for f in real_features])
        real_batch_size = real_aug_feature.size()[0]
        real_aug_mul_label = torch.tensor([0.7,0.0,0.3]*real_batch_size).reshape(real_batch_size,-1)
        real_aug_label = torch.tensor([0.3]*real_batch_size).reshape(real_batch_size)        
        # fake class
        fake_mask = ~real_mask
        fake_features = feature[fake_mask]
        fake_center = fake_features.mean(dim=0, keepdim=True)    
        aug_fncs = [
            lambda f: interpolate_push(f, fake_center),
            lambda f: interpolate_pull(f, center)
        ]
        # aug_fnc = random.choice(aug_fncs)
        fake_aug_feature = torch.stack([random.choice(aug_fncs)(f) for f in fake_features])
        fake_batch_size = fake_aug_feature.size()[0]
        fake_aug_mul_label = torch.tensor([0.0,0.7,0.3]*fake_batch_size).reshape(fake_batch_size,-1)
        fake_aug_label = torch.tensor([0.7]*fake_batch_size).reshape(fake_batch_size)
        #Concate
        aug_feature = torch.cat((real_aug_feature, fake_aug_feature), dim=0)             
        aug_mul_label = torch.cat((real_aug_mul_label, fake_aug_mul_label), dim=0)    
        aug_label = torch.cat((real_aug_label, fake_aug_label), dim=0)  
        return aug_feature, aug_mul_label, aug_label
    
    
    def get_loss(self):
        return self.BCE_loss(self.output1.squeeze(1), self.label1)

    def optimize_parameters(self, tri_decay=None):
        # Train multi_classifier with feature augment
        _, feature1, feature1_1, feature2_1 = self.model(self.input1, return_feature=True, split_feature=True)
        _, feature2, feature1_2, feature2_2 = self.model(self.input2, return_feature=True, split_feature=True)
        aug_feature1, aug_mul_label1, _ = self.feature_augment_mix_v3(feature1, self.label1)
        aug_feature2, aug_mul_label2, _ = self.feature_augment_mix_v3(feature2, self.label2)
        aug_mul_label1 = aug_mul_label1.to(self.device).float()
        aug_mul_label2 = aug_mul_label2.to(self.device).float()
        
        # for multi_classifier1 (train with detail_label -1 and 0) 
        detail_mask1_1 = ~(self.detail_label1==1)     
        detail_mask1_2 = ~(self.detail_label2==1)
        
        aug_feature1_1 = aug_feature1[:,:,:1024].to(self.device).float()
        aug_feature1_2 = aug_feature2[:,:,:1024].to(self.device).float()
        
        multi_out1_1 = self.multi_classifier_1(feature1_1[detail_mask1_1].detach())
        multi_out1_2 = self.multi_classifier_1(feature1_2[detail_mask1_2].detach())
        aug_multi_out1_1 = self.multi_classifier_1(aug_feature1_1.detach())
        aug_multi_out1_2 = self.multi_classifier_1(aug_feature1_2.detach())
        
        # self.loss_MC1 = (self.CE_loss(multi_out1_1.squeeze(1), self.multi_label1[detail_mask1_1])+\
                               # self.CE_loss(multi_out1_2.squeeze(1), self.multi_label2[detail_mask1_2])) / 2
        self.loss_MC1 = (1.0) * (((self.CE_loss(multi_out1_1.squeeze(1), self.multi_label1[detail_mask1_1])+\
                               self.CE_loss(multi_out1_2.squeeze(1), self.multi_label2[detail_mask1_2])) / 2) + \
                            0.4 * (self.CE_loss(aug_multi_out1_1.squeeze(1), aug_mul_label1)+\
                                   self.CE_loss(aug_multi_out1_2.squeeze(1), aug_mul_label2)) / 2)
        self.optimizer_MC1.zero_grad()
        self.loss_MC1.backward()
        self.optimizer_MC1.step()
        
        # for multi_classifier2 (train with detail_label -1 and 1) 
        detail_mask2_1 = ~(self.detail_label1==0)
        detail_mask2_2 = ~(self.detail_label2==0)
        
        aug_feature2_1 = aug_feature1[:,:,-1024:].to(self.device).float()
        aug_feature2_2 = aug_feature2[:,:,-1024:].to(self.device).float()
                
        multi_out2_1 = self.multi_classifier_2(feature2_1[detail_mask2_1].detach())
        multi_out2_2 = self.multi_classifier_2(feature2_2[detail_mask2_2].detach())
        aug_multi_out2_1 = self.multi_classifier_2(aug_feature2_1.detach())
        aug_multi_out2_2 = self.multi_classifier_2(aug_feature2_2.detach())
        
        # self.loss_MC2 = (self.CE_loss(multi_out2_1.squeeze(1), self.multi_label1[detail_mask2_1])+\
                               # self.CE_loss(multi_out2_2.squeeze(1), self.multi_label2[detail_mask2_2])) / 2
        self.loss_MC2 = (1.0) * (((self.CE_loss(multi_out2_1.squeeze(1), self.multi_label1[detail_mask2_1])+ \
                               self.CE_loss(multi_out2_2.squeeze(1), self.multi_label2[detail_mask2_2])) / 2) + \
                            0.4 * (self.CE_loss(aug_multi_out2_1.squeeze(1), aug_mul_label1)+ \
                                   self.CE_loss(aug_multi_out2_2.squeeze(1), aug_mul_label2)) / 2)
        self.optimizer_MC2.zero_grad()
        self.loss_MC2.backward()
        self.optimizer_MC2.step()
        
        
        # Train Feature extractor
        _, _, feature1_1, feature2_1 = self.model(self.input1, return_feature=True, split_feature=True)
        _, _, feature1_2, feature2_2 = self.model(self.input2, return_feature=True, split_feature=True)
        
        multi_out1_1 = self.multi_classifier_1(feature1_1[detail_mask1_1])
        multi_out1_2 = self.multi_classifier_1(feature1_2[detail_mask1_2])
        multi_out2_1 = self.multi_classifier_2(feature2_1[detail_mask2_1])
        multi_out2_2 = self.multi_classifier_2(feature2_2[detail_mask2_2])
        
        
        self.loss_F = (1.0) * ((self.CE_loss(multi_out1_1.squeeze(1), self.multi_label1[detail_mask1_1])+ \
                               self.CE_loss(multi_out1_2.squeeze(1), self.multi_label2[detail_mask1_2])) / 2) + \
                      (0.4) * ((self.CE_loss(multi_out2_1.squeeze(1), self.multi_label1[detail_mask2_1])+ \
                               self.CE_loss(multi_out2_2.squeeze(1), self.multi_label2[detail_mask2_2])) / 2)         
                               
        self.optimizer_F.zero_grad()
        self.loss_F.backward()
        self.optimizer_F.step()
        
        # Train binary classifier with feature augment
        _, feature1 = self.model(self.input1, return_feature=True)
        _, feature2 = self.model(self.input2, return_feature=True)
        aug_feature1, _, aug_label1 = self.feature_augment_mix_v3(feature1, self.label1)
        aug_feature2, _, aug_label2 = self.feature_augment_mix_v3(feature2, self.label2)
        aug_label1 = aug_label1.to(self.device).float()
        aug_label2 = aug_label2.to(self.device).float()
        aug_bin_out1 = self.classifier(aug_feature1.detach())
        aug_bin_out2 = self.classifier(aug_feature2.detach())
        
        loss_BC_aug = 0.4 * (self.BCE_loss(aug_bin_out1.squeeze(), aug_label1)+ \
                              self.BCE_loss(aug_bin_out2.squeeze(), aug_label2) / 2)
        
        bin_out1_1 = self.classifier(feature1[detail_mask1_1].detach())
        bin_out1_2 = self.classifier(feature2[detail_mask1_2].detach())
        # self.loss_BC = 1.0 * (self.BCE_loss(bin_out1_1.squeeze(1), self.label1[detail_mask1_1]) + \
                              # self.BCE_loss(bin_out1_2.squeeze(1), self.label2[detail_mask1_2])) / 2                             
        loss_BC_1 = 1.0 * ((self.BCE_loss(bin_out1_1.squeeze(1), self.label1[detail_mask1_1]) + \
                              self.BCE_loss(bin_out1_2.squeeze(1), self.label2[detail_mask1_2])) / 2)
        
        bin_out2_1 = self.classifier(feature1[detail_mask2_1].detach())
        bin_out2_2 = self.classifier(feature2[detail_mask2_2].detach())
        # self.loss_BC = 1.0 * (self.BCE_loss(bin_out1_1.squeeze(1), self.label1[detail_mask1_1]) + \
                              # self.BCE_loss(bin_out1_2.squeeze(1), self.label2[detail_mask1_2])) / 2                             
        loss_BC_2 = 0.1 * ((self.BCE_loss(bin_out2_1.squeeze(1), self.label1[detail_mask2_1]) + \
                              self.BCE_loss(bin_out2_2.squeeze(1), self.label2[detail_mask2_2])) / 2)
                              
        self.loss_BC = loss_BC_1 + loss_BC_2 + loss_BC_aug
        self.optimizer_BC.zero_grad()
        self.loss_BC.backward()
        self.optimizer_BC.step()
        
        self.loss = (self.loss_BC + self.loss_MC1+self.loss_MC2 + self.loss_F) / 4



class mix_Trainer_v4_2(BaseModel):
    def name(self):
        return 'mix_Trainer_v4_2'

    def __init__(self, opt):
        super(mix_Trainer_v4_2, self).__init__(opt)

        if self.isTrain and not opt.continue_train:
            self.model = resnet50(num_classes=1)
            # torch.nn.init.normal_(self.model.fc.weight.data, 0.0, opt.init_gain)
            if opt.pretrained_path != None:
                state_dict = torch.load(opt.pretrained_path, map_location='cpu')
                self.model.load_state_dict(state_dict['model'])
            else:
                print(f"your pretrained_path is empty, please set the path.")
            
            # parallel classifier 1
            self.multi_classifier_1 = Classifier(num_classes=3)
            torch.nn.init.normal_(self.multi_classifier_1.fc1.weight.data, 0.0, opt.init_gain)
            torch.nn.init.normal_(self.multi_classifier_1.fc2.weight.data, 0.0, opt.init_gain)
            
            # parallel classifier 2
            self.multi_classifier_2 = Classifier(num_classes=3)
            torch.nn.init.normal_(self.multi_classifier_2.fc1.weight.data, 0.0, opt.init_gain)
            torch.nn.init.normal_(self.multi_classifier_2.fc2.weight.data, 0.0, opt.init_gain)
            
            
            self.classifier = Classifier(num_classes=1)
            torch.nn.init.normal_(self.classifier.fc1.weight.data, 0.0, opt.init_gain)
            torch.nn.init.normal_(self.classifier.fc2.weight.data, 0.0, opt.init_gain)
            
        # Trainable parameters    
        params_F = []
    
        for name, p in self.model.named_parameters():
            if  name=="fc.weight" or name=="fc.bias": 
                p.requires_grad = False
            else:
                params_F.append(p)

        params_Multi_C1 = self.multi_classifier_1.parameters()          
        params_Multi_C2 = self.multi_classifier_2.parameters()         
        params_Binary_C = self.classifier.parameters()            
        
        if self.isTrain:
            self.BCE_loss = nn.BCEWithLogitsLoss()
            self.CE_loss = nn.CrossEntropyLoss()
            # initialize optimizers
            if opt.optim == 'adam':
                self.optimizer_F = torch.optim.Adam(params_F,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_MC1 = torch.optim.Adam(params_Multi_C1,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_MC2 = torch.optim.Adam(params_Multi_C2,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))                                  
                self.optimizer_BC = torch.optim.Adam(params_Binary_C,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))                                  
            elif opt.optim == 'sgd':
                self.optimizer_F = torch.optim.SGD(params_F,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_MC1 = torch.optim.SGD(params_Multi_C1,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_MC2 = torch.optim.Adam(params_Multi_C2,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))                                   
                self.optimizer_BC = torch.optim.SGD(params_Binary_C,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))  
            else:
                raise ValueError("optim should be [adam, sgd]")
            
        self.model.to(opt.gpu_ids[0])
        self.classifier.to(opt.gpu_ids[0])
        self.multi_classifier_1.to(opt.gpu_ids[0])
        self.multi_classifier_2.to(opt.gpu_ids[0])
        

    def adjust_learning_rate(self, min_lr=1e-7):
        for param_group in self.optimizer_F.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        for param_group in self.optimizer_MC1.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        for param_group in self.optimizer_MC2.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False        
        for param_group in self.optimizer_BC.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False        
        return True

    def set_input(self, batch1, batch2):
        # img, multi_label, label, detail_label
        
        self.input1 = batch1[0].to(self.device)
        self.multi_label1 = batch1[1].to(self.device).float()
        self.label1 = batch1[2].to(self.device).float()
        self.detail_label1 = batch1[3].to(self.device).float()
        
        self.input2 = batch2[0].to(self.device)
        self.multi_label2 = batch2[1].to(self.device).float()
        self.label2 = batch2[2].to(self.device).float()
        self.detail_label2 = batch2[3].to(self.device).float()
        
    # all augment feature seen as unseen class
    # pull in center of all, and push out center of domains
    def feature_augment_mix(self, feature, label):
        def interpolate_push(f_x, f_c):
            alpha = random.uniform(0.6, 1.5)
            new_f_x = f_x + alpha * (f_x - f_c)
            return new_f_x
        def interpolate_pull(f_x, f_c):
            # alpha = random.uniform(0.5, 1.5)
            alpha = random.uniform(0.6, 1.5)
            new_f_x = f_x + alpha * (f_c - f_x)
            return new_f_x    
        
        
        # direct manner: directly push out the feature points from the batch centriod same class 
        center = feature.mean(dim=0, keepdim=True)
        # real class
        real_mask = (label  == 0)
        real_features = feature[real_mask]
        real_center = real_features.mean(dim=0, keepdim=True) 
        aug_fncs = [
            lambda f: interpolate_push(f, real_center),
            lambda f: interpolate_pull(f, center)
        ]
        aug_fnc = random.choice(aug_fncs)
        real_aug_feature = torch.stack([aug_fnc(f) for f in real_features])    
        # fake class
        fake_mask = ~real_mask
        fake_features = feature[fake_mask]
        fake_center = fake_features.mean(dim=0, keepdim=True)    
        aug_fncs = [
            lambda f: interpolate_push(f, fake_center),
            lambda f: interpolate_pull(f, center)
        ]
        aug_fnc = random.choice(aug_fncs)
        fake_aug_feature = torch.stack([aug_fnc(f) for f in fake_features])
        #Concate
        aug_feature = torch.cat((real_aug_feature, fake_aug_feature), dim=0)         
        batch_size = aug_feature.size()[0]
        aug_mul_label = torch.tensor([0.0,0.0,1.0]*batch_size).reshape(batch_size,-1)
        aug_label = torch.tensor([1.0]*batch_size).reshape(batch_size)
        return aug_feature, aug_mul_label, aug_label
    
    
    # all augment feature seen as real/fake class
    # real: push out center of reals / seen as real class [1.0, 0.0, 0.0]
    # fake: push out and pull in center of fakes / seen as fake class [0.0, 1.0, 0.0]
    def feature_augment_mix_v2(self, feature, label):
        def interpolate_push(f_x, f_c):
            alpha = random.uniform(0.5, 1.3)
            new_f_x = f_x + alpha * (f_x - f_c)
            return new_f_x
        def interpolate_pull(f_x, f_c):
            # alpha = random.uniform(0.5, 1.5)
            alpha = random.uniform(0.5, 1.5)
            new_f_x = f_x + alpha * (f_c - f_x)
            return new_f_x       
        
        # direct manner: directly push out the feature points from the batch centriod same class 
        # real class
        real_mask = (label  == 0)
        real_features = feature[real_mask]
        real_center = real_features.mean(dim=0, keepdim=True) 
        aug_fncs = [
            lambda f: interpolate_push(f, real_center),
        ]
        aug_fnc = random.choice(aug_fncs)
        real_aug_feature = torch.stack([aug_fnc(f) for f in real_features])
        real_batch_size = real_aug_feature.size()[0]
        real_aug_mul_label = torch.tensor([1.0,0.0,0.0]*real_batch_size).reshape(real_batch_size,-1)
        real_aug_label = torch.tensor([0.0]*real_batch_size).reshape(real_batch_size)
                
        # fake class
        fake_mask = ~real_mask
        fake_features = feature[fake_mask]
        fake_center = fake_features.mean(dim=0, keepdim=True)    
        aug_fncs = [
            lambda f: interpolate_push(f, fake_center),
            lambda f: interpolate_pull(f, fake_center)
        ]
        # aug_fnc = random.choice(aug_fncs)
        fake_aug_feature = torch.stack([random.choice(aug_fncs)(f) for f in fake_features])
        # fake_aug_feature = torch.stack([aug_fnc(f) for f in fake_features])
        fake_batch_size = fake_aug_feature.size()[0]
        fake_aug_mul_label = torch.tensor([0.0,1.0,0.0]*fake_batch_size).reshape(fake_batch_size,-1)
        fake_aug_label = torch.tensor([1.0]*fake_batch_size).reshape(fake_batch_size)
        #Concate
        aug_feature = torch.cat((real_aug_feature, fake_aug_feature), dim=0)             
        aug_mul_label = torch.cat((real_aug_mul_label, fake_aug_mul_label), dim=0)    
        aug_label = torch.cat((real_aug_label, fake_aug_label), dim=0)  
        return aug_feature, aug_mul_label, aug_label    
    

    # all augment feature seen as 
    # real: push out center of reals / seen as half real class [0.7, 0.0, 0.3]
    # fake: push out and pull in center of fakes / seen as half fake class [0.0, 0.7, 0.3]    
    def feature_augment_mix_v3(self, feature, label):
        def interpolate_push(f_x, f_c):
            alpha = random.uniform(0.5, 1.3)
            new_f_x = f_x + alpha * (f_x - f_c)
            return new_f_x
        def interpolate_pull(f_x, f_c):
            # alpha = random.uniform(0.5, 1.5)
            alpha = random.uniform(0.5, 1.3)
            new_f_x = f_x + alpha * (f_c - f_x)
            return new_f_x       
        
        # direct manner: directly push out the feature points from the batch centriod same class 
        # real class
        real_mask = (label  == 0)
        real_features = feature[real_mask]
        real_center = real_features.mean(dim=0, keepdim=True) 
        aug_fncs = [
            lambda f: interpolate_push(f, real_center),
        ]
        aug_fnc = random.choice(aug_fncs)
        real_aug_feature = torch.stack([aug_fnc(f) for f in real_features])
        real_batch_size = real_aug_feature.size()[0]
        real_aug_mul_label = torch.tensor([0.7,0.0,0.3]*real_batch_size).reshape(real_batch_size,-1)
        real_aug_label = torch.tensor([0.3]*real_batch_size).reshape(real_batch_size)
                
        # fake class
        fake_mask = ~real_mask
        fake_features = feature[fake_mask]
        fake_center = fake_features.mean(dim=0, keepdim=True)    
        aug_fncs = [
            lambda f: interpolate_push(f, fake_center),
            lambda f: interpolate_pull(f, fake_center)
        ]
        # aug_fnc = random.choice(aug_fncs)
        fake_aug_feature = torch.stack([random.choice(aug_fncs)(f) for f in fake_features])
        # fake_aug_feature = torch.stack([aug_fnc(f) for f in fake_features])
        fake_batch_size = fake_aug_feature.size()[0]
        fake_aug_mul_label = torch.tensor([0.0,0.7,0.3]*fake_batch_size).reshape(fake_batch_size,-1)
        fake_aug_label = torch.tensor([0.7]*fake_batch_size).reshape(fake_batch_size)
        #Concate
        aug_feature = torch.cat((real_aug_feature, fake_aug_feature), dim=0)             
        aug_mul_label = torch.cat((real_aug_mul_label, fake_aug_mul_label), dim=0)    
        aug_label = torch.cat((real_aug_label, fake_aug_label), dim=0)  
        return aug_feature, aug_mul_label, aug_label  
    
    # all augment feature seen as unseen class
    # pull in center of all, and push out center of domains
    def feature_augment_mix_v4(self, feature, label):
        def interpolate_push(f_x, f_c):
            alpha = random.uniform(0.1, 1.0)
            new_f_x = f_x + alpha * (f_x - f_c)
            return new_f_x
        def interpolate_pull(f_x, f_c):
            # alpha = random.uniform(0.5, 1.5)
            alpha = random.uniform(0.1, 0.5)
            new_f_x = f_x + alpha * (f_c - f_x)
            return new_f_x    
        
        
        # direct manner: directly push out the feature points from the batch centriod same class 
        center = feature.mean(dim=0, keepdim=True)
        # real class
        real_mask = (label  == 0)
        real_features = feature[real_mask]
        real_center = real_features.mean(dim=0, keepdim=True) 
        aug_fncs = [
            lambda f: interpolate_push(f, real_center),
            lambda f: interpolate_pull(f, center)
        ]
        # aug_fnc = random.choice(aug_fncs)
        real_aug_feature = torch.stack([random.choice(aug_fncs)(f) for f in real_features])
        real_batch_size = real_aug_feature.size()[0]
        real_aug_mul_label = torch.tensor([0.7,0.0,0.3]*real_batch_size).reshape(real_batch_size,-1)
        real_aug_label = torch.tensor([0.3]*real_batch_size).reshape(real_batch_size)        
        # fake class
        fake_mask = ~real_mask
        fake_features = feature[fake_mask]
        fake_center = fake_features.mean(dim=0, keepdim=True)    
        aug_fncs = [
            lambda f: interpolate_push(f, fake_center),
            lambda f: interpolate_pull(f, center)
        ]
        # aug_fnc = random.choice(aug_fncs)
        fake_aug_feature = torch.stack([random.choice(aug_fncs)(f) for f in fake_features])
        fake_batch_size = fake_aug_feature.size()[0]
        fake_aug_mul_label = torch.tensor([0.0,0.7,0.3]*fake_batch_size).reshape(fake_batch_size,-1)
        fake_aug_label = torch.tensor([0.7]*fake_batch_size).reshape(fake_batch_size)
        #Concate
        aug_feature = torch.cat((real_aug_feature, fake_aug_feature), dim=0)             
        aug_mul_label = torch.cat((real_aug_mul_label, fake_aug_mul_label), dim=0)    
        aug_label = torch.cat((real_aug_label, fake_aug_label), dim=0)  
        return aug_feature, aug_mul_label, aug_label
    
    
    def get_loss(self):
        return self.BCE_loss(self.output1.squeeze(1), self.label1)

    def optimize_parameters(self, tri_decay=None):
        # Train multi_classifier with feature augment
        # extract features from input images
        _, feature1 = self.model(self.input1, return_feature=True)
        _, feature2 = self.model(self.input2, return_feature=True)
        # combine patch and full together to genearte feature augments
        features = torch.cat((feature1, feature2), dim=0)
        labels = torch.cat((self.label1, self.label2), dim=0)
        aug_feature, aug_mul_label, _  = self.feature_augment_mix_v3(features, labels)
        aug_mul_label = aug_mul_label.to(self.device).float()
        
        # For multi_classifier1 (train with detail_label -1 and 0) 
        detail_mask1_1 = ~(self.detail_label1==1)
        detail_mask1_2 = ~(self.detail_label2==1)    
        
        multi_out1_1 = self.multi_classifier_1(feature1[detail_mask1_1].detach())
        multi_out1_2 = self.multi_classifier_1(feature2[detail_mask1_2].detach())
        aug_multi_out1 = self.multi_classifier_1(aug_feature.detach())
        
        # self.loss_MC1 = (self.CE_loss(multi_out1_1.squeeze(1), self.multi_label1[detail_mask1_1])+\
                               # self.CE_loss(multi_out1_2.squeeze(1), self.multi_label2[detail_mask1_2])) / 2
        self.loss_MC1 = (1.0) * (((self.CE_loss(multi_out1_1.squeeze(1), self.multi_label1[detail_mask1_1])+\
                               self.CE_loss(multi_out1_2.squeeze(1), self.multi_label2[detail_mask1_2])) / 2) + \
                            0.4 * (self.CE_loss(aug_multi_out1.squeeze(1), aug_mul_label)))
        self.optimizer_MC1.zero_grad()
        self.loss_MC1.backward()
        self.optimizer_MC1.step()
        
        # for multi_classifier2 (train with detail_label -1 and 1) 
        detail_mask2_1 = ~(self.detail_label1==0)
        detail_mask2_2 = ~(self.detail_label2==0)
        
        multi_out2_1 = self.multi_classifier_2(feature1[detail_mask2_1].detach())
        multi_out2_2 = self.multi_classifier_2(feature2[detail_mask2_2].detach())
        aug_multi_out2 = self.multi_classifier_2(aug_feature.detach())
        
        # self.loss_MC2 = (self.CE_loss(multi_out2_1.squeeze(1), self.multi_label1[detail_mask2_1])+\
                               # self.CE_loss(multi_out2_2.squeeze(1), self.multi_label2[detail_mask2_2])) / 2
        self.loss_MC2 = (1.0) * (((self.CE_loss(multi_out2_1.squeeze(1), self.multi_label1[detail_mask2_1])+ \
                               self.CE_loss(multi_out2_2.squeeze(1), self.multi_label2[detail_mask2_2])) / 2) + \
                            0.4 * (self.CE_loss(aug_multi_out2.squeeze(1), aug_mul_label)))
        self.optimizer_MC2.zero_grad()
        self.loss_MC2.backward()
        self.optimizer_MC2.step()
        
        
        # Train Feature extractor
        _, feature1 = self.model(self.input1, return_feature=True)
        _, feature2 = self.model(self.input2, return_feature=True)
        
        multi_out1_1 = self.multi_classifier_1(feature1[detail_mask1_1])
        multi_out1_2 = self.multi_classifier_1(feature2[detail_mask1_2])
        # multi_out2_1 = self.multi_classifier_2(feature1[detail_mask2_1])
        # multi_out2_2 = self.multi_classifier_2(feature2[detail_mask2_2])
        
        self.loss_F = (1.0) * ((self.CE_loss(multi_out1_1.squeeze(1), self.multi_label1[detail_mask1_1])+ \
                               self.CE_loss(multi_out1_2.squeeze(1), self.multi_label2[detail_mask1_2])) / 2) 
        # self.loss_F = (1.0) * ((self.CE_loss(multi_out1_1.squeeze(1), self.multi_label1[detail_mask1_1])+ \
                               # self.CE_loss(multi_out1_2.squeeze(1), self.multi_label2[detail_mask1_2])) / 2) + \
                      # (0.4) * ((self.CE_loss(multi_out2_1.squeeze(1), self.multi_label1[detail_mask2_1])+ \
                               # self.CE_loss(multi_out2_2.squeeze(1), self.multi_label2[detail_mask2_2])) / 2)         
                               
        self.optimizer_F.zero_grad()
        self.loss_F.backward()
        self.optimizer_F.step()
        
        # Train binary classifier with feature augment
        _, feature1 = self.model(self.input1, return_feature=True)
        _, feature2 = self.model(self.input2, return_feature=True)
                
        # combine patch and full together to genearte feature augments
        features = torch.cat((feature1, feature2), dim=0)
        labels = torch.cat((self.label1, self.label2), dim=0)
        aug_feature, _ , aug_label  = self.feature_augment_mix_v3(features, labels)
        aug_label = aug_label.to(self.device).float()     
        aug_bin_out = self.classifier(aug_feature.detach())
        
        loss_BC_aug = 0.4 * (self.BCE_loss(aug_bin_out.squeeze(), aug_label)) 
        
        bin_out1_1 = self.classifier(feature1[detail_mask1_1].detach())
        bin_out1_2 = self.classifier(feature2[detail_mask1_2].detach())
        
        loss_BC_1 = 1.0 * ((self.BCE_loss(bin_out1_1.squeeze(1), self.label1[detail_mask1_1]) + \
                              self.BCE_loss(bin_out1_2.squeeze(1), self.label2[detail_mask1_2])) / 2)
                            
        # bin_out2_1 = self.classifier(feature1[detail_mask2_1].detach())
        # bin_out2_2 = self.classifier(feature2[detail_mask2_2].detach())
        
        # loss_BC_2 = 0.4 * ((self.BCE_loss(bin_out2_1.squeeze(1), self.label1[detail_mask2_1]) + \
                            # self.BCE_loss(bin_out2_2.squeeze(1), self.label2[detail_mask2_2])) / 2)            
        
        # self.loss_BC = loss_BC_1 + loss_BC_2 + loss_BC_aug
        # self.loss_BC = loss_BC_1 + loss_BC_2
        self.loss_BC = loss_BC_1 + loss_BC_aug
        # self.loss_BC = loss_BC_1
        self.optimizer_BC.zero_grad()
        self.loss_BC.backward()
        self.optimizer_BC.step()
        
        self.loss = (self.loss_BC + self.loss_MC1+self.loss_MC2 + self.loss_F) / 4



class mix_Trainer_v4_3(BaseModel):
    def name(self):
        return 'mix_Trainer_v4_3'

    def __init__(self, opt):
        super(mix_Trainer_v4_3, self).__init__(opt)

        if self.isTrain and not opt.continue_train:
            self.model = resnet50(num_classes=1)
            # torch.nn.init.normal_(self.model.fc.weight.data, 0.0, opt.init_gain)
            if opt.pretrained_path != None:
                state_dict = torch.load(opt.pretrained_path, map_location='cpu')
                self.model.load_state_dict(state_dict['model'])
            else:
                print(f"your pretrained_path is empty, please set the path.")
            
            # parallel classifier 1
            self.multi_classifier_1 = Classifier(num_classes=3)
            torch.nn.init.normal_(self.multi_classifier_1.fc1.weight.data, 0.0, opt.init_gain)
            torch.nn.init.normal_(self.multi_classifier_1.fc2.weight.data, 0.0, opt.init_gain)
            
            # parallel classifier 2
            self.multi_classifier_2 = Classifier(num_classes=3)
            torch.nn.init.normal_(self.multi_classifier_2.fc1.weight.data, 0.0, opt.init_gain)
            torch.nn.init.normal_(self.multi_classifier_2.fc2.weight.data, 0.0, opt.init_gain)
            
            
            self.classifier = Classifier(num_classes=1)
            torch.nn.init.normal_(self.classifier.fc1.weight.data, 0.0, opt.init_gain)
            torch.nn.init.normal_(self.classifier.fc2.weight.data, 0.0, opt.init_gain)
            
        # Trainable parameters    
        params_F = []
    
        for name, p in self.model.named_parameters():
            if  name=="fc.weight" or name=="fc.bias": 
                p.requires_grad = False
            else:
                params_F.append(p)

        params_Multi_C1 = self.multi_classifier_1.parameters()          
        params_Multi_C2 = self.multi_classifier_2.parameters()         
        params_Binary_C = self.classifier.parameters()            
        
        if self.isTrain:
            self.BCE_loss = nn.BCEWithLogitsLoss()
            self.CE_loss = nn.CrossEntropyLoss()
            self.triplet = nn.TripletMarginLoss(margin=0.2, p=2, eps=1e-6)
            # initialize optimizers
            if opt.optim == 'adam':
                self.optimizer_F = torch.optim.Adam(params_F,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_MC1 = torch.optim.Adam(params_Multi_C1,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_MC2 = torch.optim.Adam(params_Multi_C2,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))                                  
                self.optimizer_BC = torch.optim.Adam(params_Binary_C,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))                                  
            elif opt.optim == 'sgd':
                self.optimizer_F = torch.optim.SGD(params_F,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_MC1 = torch.optim.SGD(params_Multi_C1,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_MC2 = torch.optim.Adam(params_Multi_C2,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))                                   
                self.optimizer_BC = torch.optim.SGD(params_Binary_C,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))  
            else:
                raise ValueError("optim should be [adam, sgd]")
            
        self.model.to(opt.gpu_ids[0])
        self.classifier.to(opt.gpu_ids[0])
        self.multi_classifier_1.to(opt.gpu_ids[0])
        self.multi_classifier_2.to(opt.gpu_ids[0])
        self.triplet = self.triplet.to(opt.gpu_ids[0])

    def adjust_learning_rate(self, min_lr=1e-7):
        for param_group in self.optimizer_F.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        for param_group in self.optimizer_MC1.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        for param_group in self.optimizer_MC2.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False        
        for param_group in self.optimizer_BC.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False        
        return True

    def set_input(self, batch1, batch2):
        # img, multi_label, label, detail_label
        
        self.input1 = batch1[0].to(self.device)
        self.multi_label1 = batch1[1].to(self.device).float()
        self.label1 = batch1[2].to(self.device).float()
        self.detail_label1 = batch1[3].to(self.device).float()
        
        self.input2 = batch2[0].to(self.device)
        self.multi_label2 = batch2[1].to(self.device).float()
        self.label2 = batch2[2].to(self.device).float()
        self.detail_label2 = batch2[3].to(self.device).float()
        
    # all augment feature seen as unseen class
    # pull in center of all, and push out center of domains
    def feature_augment_mix(self, feature, label):
        def interpolate_push(f_x, f_c):
            alpha = random.uniform(0.6, 1.5)
            new_f_x = f_x + alpha * (f_x - f_c)
            return new_f_x
        def interpolate_pull(f_x, f_c):
            # alpha = random.uniform(0.5, 1.5)
            alpha = random.uniform(0.6, 1.5)
            new_f_x = f_x + alpha * (f_c - f_x)
            return new_f_x    
        
        
        # direct manner: directly push out the feature points from the batch centriod same class 
        center = feature.mean(dim=0, keepdim=True)
        # real class
        real_mask = (label  == 0)
        real_features = feature[real_mask]
        real_center = real_features.mean(dim=0, keepdim=True) 
        aug_fncs = [
            lambda f: interpolate_push(f, real_center),
            lambda f: interpolate_pull(f, center)
        ]
        aug_fnc = random.choice(aug_fncs)
        real_aug_feature = torch.stack([aug_fnc(f) for f in real_features])    
        # fake class
        fake_mask = ~real_mask
        fake_features = feature[fake_mask]
        fake_center = fake_features.mean(dim=0, keepdim=True)    
        aug_fncs = [
            lambda f: interpolate_push(f, fake_center),
            lambda f: interpolate_pull(f, center)
        ]
        aug_fnc = random.choice(aug_fncs)
        fake_aug_feature = torch.stack([aug_fnc(f) for f in fake_features])
        #Concate
        aug_feature = torch.cat((real_aug_feature, fake_aug_feature), dim=0)         
        batch_size = aug_feature.size()[0]
        aug_mul_label = torch.tensor([0.0,0.0,1.0]*batch_size).reshape(batch_size,-1)
        aug_label = torch.tensor([1.0]*batch_size).reshape(batch_size)
        return aug_feature, aug_mul_label, aug_label
    
    
    # all augment feature seen as real/fake class
    # real: push out center of reals / seen as real class [1.0, 0.0, 0.0]
    # fake: push out and pull in center of fakes / seen as fake class [0.0, 1.0, 0.0]
    def feature_augment_mix_v2(self, feature, label):
        def interpolate_push(f_x, f_c):
            alpha = random.uniform(0.5, 1.3)
            new_f_x = f_x + alpha * (f_x - f_c)
            return new_f_x
        def interpolate_pull(f_x, f_c):
            # alpha = random.uniform(0.5, 1.5)
            alpha = random.uniform(0.5, 1.5)
            new_f_x = f_x + alpha * (f_c - f_x)
            return new_f_x       
        
        # direct manner: directly push out the feature points from the batch centriod same class 
        # real class
        real_mask = (label  == 0)
        real_features = feature[real_mask]
        real_center = real_features.mean(dim=0, keepdim=True) 
        aug_fncs = [
            lambda f: interpolate_push(f, real_center),
        ]
        aug_fnc = random.choice(aug_fncs)
        real_aug_feature = torch.stack([aug_fnc(f) for f in real_features])
        real_batch_size = real_aug_feature.size()[0]
        real_aug_mul_label = torch.tensor([1.0,0.0,0.0]*real_batch_size).reshape(real_batch_size,-1)
        real_aug_label = torch.tensor([0.0]*real_batch_size).reshape(real_batch_size)
                
        # fake class
        fake_mask = ~real_mask
        fake_features = feature[fake_mask]
        fake_center = fake_features.mean(dim=0, keepdim=True)    
        aug_fncs = [
            lambda f: interpolate_push(f, fake_center),
            lambda f: interpolate_pull(f, fake_center)
        ]
        # aug_fnc = random.choice(aug_fncs)
        fake_aug_feature = torch.stack([random.choice(aug_fncs)(f) for f in fake_features])
        # fake_aug_feature = torch.stack([aug_fnc(f) for f in fake_features])
        fake_batch_size = fake_aug_feature.size()[0]
        fake_aug_mul_label = torch.tensor([0.0,1.0,0.0]*fake_batch_size).reshape(fake_batch_size,-1)
        fake_aug_label = torch.tensor([1.0]*fake_batch_size).reshape(fake_batch_size)
        #Concate
        aug_feature = torch.cat((real_aug_feature, fake_aug_feature), dim=0)             
        aug_mul_label = torch.cat((real_aug_mul_label, fake_aug_mul_label), dim=0)    
        aug_label = torch.cat((real_aug_label, fake_aug_label), dim=0)  
        return aug_feature, aug_mul_label, aug_label    
    

    # all augment feature seen as 
    # real: push out center of reals / seen as half real class [0.7, 0.0, 0.3]
    # fake: push out and pull in center of fakes / seen as half fake class [0.0, 0.7, 0.3]    
    def feature_augment_mix_v3(self, feature, label):
        def interpolate_push(f_x, f_c):
            alpha = random.uniform(0.5, 1.3)
            new_f_x = f_x + alpha * (f_x - f_c)
            return new_f_x
        def interpolate_pull(f_x, f_c):
            # alpha = random.uniform(0.5, 1.5)
            alpha = random.uniform(0.5, 1.3)
            new_f_x = f_x + alpha * (f_c - f_x)
            return new_f_x       
        
        # direct manner: directly push out the feature points from the batch centriod same class 
        # real class
        real_mask = (label  == 0)
        real_features = feature[real_mask]
        real_center = real_features.mean(dim=0, keepdim=True) 
        aug_fncs = [
            lambda f: interpolate_push(f, real_center),
        ]
        aug_fnc = random.choice(aug_fncs)
        real_aug_feature = torch.stack([aug_fnc(f) for f in real_features])
        real_batch_size = real_aug_feature.size()[0]
        real_aug_mul_label = torch.tensor([0.7,0.0,0.3]*real_batch_size).reshape(real_batch_size,-1)
        real_aug_label = torch.tensor([0.3]*real_batch_size).reshape(real_batch_size)
                
        # fake class
        fake_mask = ~real_mask
        fake_features = feature[fake_mask]
        fake_center = fake_features.mean(dim=0, keepdim=True)    
        aug_fncs = [
            lambda f: interpolate_push(f, fake_center),
            lambda f: interpolate_pull(f, fake_center)
        ]
        # aug_fnc = random.choice(aug_fncs)
        fake_aug_feature = torch.stack([random.choice(aug_fncs)(f) for f in fake_features])
        # fake_aug_feature = torch.stack([aug_fnc(f) for f in fake_features])
        fake_batch_size = fake_aug_feature.size()[0]
        fake_aug_mul_label = torch.tensor([0.0,0.7,0.3]*fake_batch_size).reshape(fake_batch_size,-1)
        fake_aug_label = torch.tensor([0.7]*fake_batch_size).reshape(fake_batch_size)
        #Concate
        aug_feature = torch.cat((real_aug_feature, fake_aug_feature), dim=0)             
        aug_mul_label = torch.cat((real_aug_mul_label, fake_aug_mul_label), dim=0)    
        aug_label = torch.cat((real_aug_label, fake_aug_label), dim=0)  
        return aug_feature, aug_mul_label, aug_label  
    
    # all augment feature seen as unseen class
    # pull in center of all, and push out center of domains
    def feature_augment_mix_v4(self, feature, label):
        def interpolate_push(f_x, f_c):
            alpha = random.uniform(0.1, 1.0)
            new_f_x = f_x + alpha * (f_x - f_c)
            return new_f_x
        def interpolate_pull(f_x, f_c):
            # alpha = random.uniform(0.5, 1.5)
            alpha = random.uniform(0.1, 0.5)
            new_f_x = f_x + alpha * (f_c - f_x)
            return new_f_x    
        
        
        # direct manner: directly push out the feature points from the batch centriod same class 
        center = feature.mean(dim=0, keepdim=True)
        # real class
        real_mask = (label  == 0)
        real_features = feature[real_mask]
        real_center = real_features.mean(dim=0, keepdim=True) 
        aug_fncs = [
            lambda f: interpolate_push(f, real_center),
            lambda f: interpolate_pull(f, center)
        ]
        # aug_fnc = random.choice(aug_fncs)
        real_aug_feature = torch.stack([random.choice(aug_fncs)(f) for f in real_features])
        real_batch_size = real_aug_feature.size()[0]
        real_aug_mul_label = torch.tensor([0.7,0.0,0.3]*real_batch_size).reshape(real_batch_size,-1)
        real_aug_label = torch.tensor([0.3]*real_batch_size).reshape(real_batch_size)        
        # fake class
        fake_mask = ~real_mask
        fake_features = feature[fake_mask]
        fake_center = fake_features.mean(dim=0, keepdim=True)    
        aug_fncs = [
            lambda f: interpolate_push(f, fake_center),
            lambda f: interpolate_pull(f, center)
        ]
        # aug_fnc = random.choice(aug_fncs)
        fake_aug_feature = torch.stack([random.choice(aug_fncs)(f) for f in fake_features])
        fake_batch_size = fake_aug_feature.size()[0]
        fake_aug_mul_label = torch.tensor([0.0,0.7,0.3]*fake_batch_size).reshape(fake_batch_size,-1)
        fake_aug_label = torch.tensor([0.7]*fake_batch_size).reshape(fake_batch_size)
        #Concate
        aug_feature = torch.cat((real_aug_feature, fake_aug_feature), dim=0)             
        aug_mul_label = torch.cat((real_aug_mul_label, fake_aug_mul_label), dim=0)    
        aug_label = torch.cat((real_aug_label, fake_aug_label), dim=0)  
        return aug_feature, aug_mul_label, aug_label
    
    def compute_centers_triplet_loss(self, feature1, feature2):
        # compute each batch domain center
        # real center
        real_mask1 = (self.detail_label1 == -1)
        real_center1 = feature1[real_mask1].mean(dim=0, keepdim=True)
        real_mask2 = (self.detail_label2 == -1)
        real_center2 = feature2[real_mask2].mean(dim=0, keepdim=True)
        # detail_label=0 fake' s center    
        fake_mask1_0 = (self.detail_label1 == 0)
        fake_center1_0 = feature1[fake_mask1_0].mean(dim=0, keepdim=True)
        fake_mask2_0 = (self.detail_label2 == 0)
        fake_center2_0 = feature2[fake_mask2_0].mean(dim=0, keepdim=True)     
        
        # detail_label=1 fake' s center 
        fake_mask1_1 = (self.detail_label1 == 1)
        fake_center1_1 = feature1[fake_mask1_1].mean(dim=0, keepdim=True) 
        fake_mask2_1 = (self.detail_label2 == 1)
        fake_center2_1 = feature2[fake_mask2_1].mean(dim=0, keepdim=True) 
        
        #Dis(fake_full, fake_patch) <= Dis(fake, real)
        Anchor = torch.cat((fake_center1_0, fake_center2_0,fake_center1_1,fake_center2_1), dim=0)  
        Positive = torch.cat((fake_center2_0, fake_center1_0,fake_center2_1,fake_center1_1), dim=0)  
        Negative = torch.cat((real_center1, real_center2, real_center1, real_center2), dim=0)  
        loss = self.triplet(Anchor, Positive, Negative)
        return loss    
        
        
    def get_loss(self):
        return self.BCE_loss(self.output1.squeeze(1), self.label1)

    def optimize_parameters(self, tri_decay=None):
        # Train multi_classifier with feature augment
        # extract features from input images
        _, feature1 = self.model(self.input1, return_feature=True)
        _, feature2 = self.model(self.input2, return_feature=True)
        # combine patch and full together to genearte feature augments
        features = torch.cat((feature1, feature2), dim=0)
        labels = torch.cat((self.label1, self.label2), dim=0)
        aug_feature, aug_mul_label, _  = self.feature_augment_mix_v3(features, labels)
        aug_mul_label = aug_mul_label.to(self.device).float()
        
        # For multi_classifier1 (train with detail_label -1 and 0) 
        detail_mask1_1 = ~(self.detail_label1==1)
        detail_mask1_2 = ~(self.detail_label2==1)    
        
        multi_out1_1 = self.multi_classifier_1(feature1[detail_mask1_1].detach())
        multi_out1_2 = self.multi_classifier_1(feature2[detail_mask1_2].detach())
        aug_multi_out1 = self.multi_classifier_1(aug_feature.detach())
        
        # self.loss_MC1 = (self.CE_loss(multi_out1_1.squeeze(1), self.multi_label1[detail_mask1_1])+\
                               # self.CE_loss(multi_out1_2.squeeze(1), self.multi_label2[detail_mask1_2])) / 2
        self.loss_MC1 = (1.0) * (((self.CE_loss(multi_out1_1.squeeze(1), self.multi_label1[detail_mask1_1])+\
                               self.CE_loss(multi_out1_2.squeeze(1), self.multi_label2[detail_mask1_2])) / 2) + \
                            0.4 * (self.CE_loss(aug_multi_out1.squeeze(1), aug_mul_label)))
        self.optimizer_MC1.zero_grad()
        self.loss_MC1.backward()
        self.optimizer_MC1.step()
        
        # for multi_classifier2 (train with detail_label -1 and 1) 
        detail_mask2_1 = ~(self.detail_label1==0)
        detail_mask2_2 = ~(self.detail_label2==0)
        
        multi_out2_1 = self.multi_classifier_2(feature1[detail_mask2_1].detach())
        multi_out2_2 = self.multi_classifier_2(feature2[detail_mask2_2].detach())
        aug_multi_out2 = self.multi_classifier_2(aug_feature.detach())
        
        # self.loss_MC2 = (self.CE_loss(multi_out2_1.squeeze(1), self.multi_label1[detail_mask2_1])+\
                               # self.CE_loss(multi_out2_2.squeeze(1), self.multi_label2[detail_mask2_2])) / 2
        self.loss_MC2 = (1.0) * (((self.CE_loss(multi_out2_1.squeeze(1), self.multi_label1[detail_mask2_1])+ \
                               self.CE_loss(multi_out2_2.squeeze(1), self.multi_label2[detail_mask2_2])) / 2) + \
                            0.4 * (self.CE_loss(aug_multi_out2.squeeze(1), aug_mul_label)))
        self.optimizer_MC2.zero_grad()
        self.loss_MC2.backward()
        self.optimizer_MC2.step()
        
        
        # Train Feature extractor
        _, feature1 = self.model(self.input1, return_feature=True)
        _, feature2 = self.model(self.input2, return_feature=True)
        
        # triplet loss
        tri_loss = self.compute_centers_triplet_loss(feature1, feature2)
        
        multi_out1_1 = self.multi_classifier_1(feature1[detail_mask1_1])
        multi_out1_2 = self.multi_classifier_1(feature2[detail_mask1_2])
        multi_out2_1 = self.multi_classifier_2(feature1[detail_mask2_1])
        multi_out2_2 = self.multi_classifier_2(feature2[detail_mask2_2])
        
        # self.loss_F = (1.0) * ((self.CE_loss(multi_out1_1.squeeze(1), self.multi_label1[detail_mask1_1])+ \
                               # self.CE_loss(multi_out1_2.squeeze(1), self.multi_label2[detail_mask1_2])) / 2) 
        self.loss_F = (1.0) * ((self.CE_loss(multi_out1_1.squeeze(1), self.multi_label1[detail_mask1_1])+ \
                               self.CE_loss(multi_out1_2.squeeze(1), self.multi_label2[detail_mask1_2])) / 2) + \
                      (0.4) * ((self.CE_loss(multi_out2_1.squeeze(1), self.multi_label1[detail_mask2_1])+ \
                               self.CE_loss(multi_out2_2.squeeze(1), self.multi_label2[detail_mask2_2])) / 2) + \
                      (1.0) * tri_loss              
                               
        self.optimizer_F.zero_grad()
        self.loss_F.backward()
        self.optimizer_F.step()
        
        # Train binary classifier with feature augment
        _, feature1 = self.model(self.input1, return_feature=True)
        _, feature2 = self.model(self.input2, return_feature=True)
                
        # combine patch and full together to genearte feature augments
        features = torch.cat((feature1, feature2), dim=0)
        labels = torch.cat((self.label1, self.label2), dim=0)
        aug_feature, _ , aug_label  = self.feature_augment_mix_v3(features, labels)
        aug_label = aug_label.to(self.device).float()     
        aug_bin_out = self.classifier(aug_feature.detach())
        
        loss_BC_aug = 0.4 * (self.BCE_loss(aug_bin_out.squeeze(), aug_label)) 
        
        bin_out1_1 = self.classifier(feature1[detail_mask1_1].detach())
        bin_out1_2 = self.classifier(feature2[detail_mask1_2].detach())
        
        loss_BC_1 = 1.0 * ((self.BCE_loss(bin_out1_1.squeeze(1), self.label1[detail_mask1_1]) + \
                              self.BCE_loss(bin_out1_2.squeeze(1), self.label2[detail_mask1_2])) / 2)
                            
        bin_out2_1 = self.classifier(feature1[detail_mask2_1].detach())
        bin_out2_2 = self.classifier(feature2[detail_mask2_2].detach())
        
        loss_BC_2 = 0.4 * ((self.BCE_loss(bin_out2_1.squeeze(1), self.label1[detail_mask2_1]) + \
                            self.BCE_loss(bin_out2_2.squeeze(1), self.label2[detail_mask2_2])) / 2)            
        
        self.loss_BC = loss_BC_1 + loss_BC_2 + loss_BC_aug
        # self.loss_BC = loss_BC_1 + loss_BC_2
        # self.loss_BC = loss_BC_1 + loss_BC_aug
        # self.loss_BC = loss_BC_1
        self.optimizer_BC.zero_grad()
        self.loss_BC.backward()
        self.optimizer_BC.step()
        
        self.loss = (self.loss_BC + self.loss_MC1+self.loss_MC2 + self.loss_F) / 4

class mix_Trainer_v2_2(BaseModel):
    def name(self):
        return 'mix_Trainer_v2_2'

    def __init__(self, opt):
        super(mix_Trainer_v2_2, self).__init__(opt)

        if self.isTrain and not opt.continue_train:
            self.model = resnet50(num_classes=1)
            # torch.nn.init.normal_(self.model.fc.weight.data, 0.0, opt.init_gain)
            if opt.pretrained_path != None:
                state_dict = torch.load(opt.pretrained_path, map_location='cpu')
                self.model.load_state_dict(state_dict['model'])
            else:
                print(f"your pretrained_path is empty, please set the path.")
            
            # parallel classifier 1
            self.multi_classifier_1 = Classifier(num_classes=3)
            torch.nn.init.normal_(self.multi_classifier_1.fc1.weight.data, 0.0, opt.init_gain)
            torch.nn.init.normal_(self.multi_classifier_1.fc2.weight.data, 0.0, opt.init_gain)
            
            # parallel classifier 2
            self.multi_classifier_2 = Classifier(num_classes=3)
            torch.nn.init.normal_(self.multi_classifier_2.fc1.weight.data, 0.0, opt.init_gain)
            torch.nn.init.normal_(self.multi_classifier_2.fc2.weight.data, 0.0, opt.init_gain)
            
            # parallel classifier 3
            self.multi_classifier_3 = Classifier(num_classes=3)
            torch.nn.init.normal_(self.multi_classifier_3.fc1.weight.data, 0.0, opt.init_gain)
            torch.nn.init.normal_(self.multi_classifier_3.fc2.weight.data, 0.0, opt.init_gain)
            
            
            self.classifier = Classifier(num_classes=1)
            torch.nn.init.normal_(self.classifier.fc1.weight.data, 0.0, opt.init_gain)
            torch.nn.init.normal_(self.classifier.fc2.weight.data, 0.0, opt.init_gain)
            
        # Trainable parameters    
        params_F = []
    
        for name, p in self.model.named_parameters():
            if  name=="fc.weight" or name=="fc.bias": 
                p.requires_grad = False
            else:
                params_F.append(p)

        params_Multi_C1 = self.multi_classifier_1.parameters()          
        params_Multi_C2 = self.multi_classifier_2.parameters()         
        params_Multi_C3 = self.multi_classifier_3.parameters()         
        params_Binary_C = self.classifier.parameters()            
        
        if self.isTrain:
            self.BCE_loss = nn.BCEWithLogitsLoss()
            self.CE_loss = nn.CrossEntropyLoss()
            # initialize optimizers
            if opt.optim == 'adam':
                self.optimizer_F = torch.optim.Adam(params_F,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_MC1 = torch.optim.Adam(params_Multi_C1,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_MC2 = torch.optim.Adam(params_Multi_C2,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999)) 
                self.optimizer_MC3 = torch.optim.Adam(params_Multi_C3,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))                                               
                self.optimizer_BC = torch.optim.Adam(params_Binary_C,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))                                  
            elif opt.optim == 'sgd':
                self.optimizer_F = torch.optim.SGD(params_F,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_MC1 = torch.optim.SGD(params_Multi_C1,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_MC2 = torch.optim.SGD(params_Multi_C2,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_MC3 = torch.optim.SGD(params_Multi_C3,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))                                          
                self.optimizer_BC = torch.optim.SGD(params_Binary_C,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))  
            else:
                raise ValueError("optim should be [adam, sgd]")
            
        self.model.to(opt.gpu_ids[0])
        self.classifier.to(opt.gpu_ids[0])
        self.multi_classifier_1.to(opt.gpu_ids[0])
        self.multi_classifier_2.to(opt.gpu_ids[0])
        self.multi_classifier_3.to(opt.gpu_ids[0])
        

    def adjust_learning_rate(self, min_lr=1e-7):
        for param_group in self.optimizer_F.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        for param_group in self.optimizer_MC1.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        for param_group in self.optimizer_MC2.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False     
        for param_group in self.optimizer_MC3.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False              
        for param_group in self.optimizer_BC.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False        
        return True

    def set_input(self, batch1, batch2):
        # img, multi_label, label, detail_label
        
        self.input1 = batch1[0].to(self.device)
        self.multi_label1 = batch1[1].to(self.device).float()
        self.label1 = batch1[2].to(self.device).float()
        self.detail_label1 = batch1[3].to(self.device).float()
        
        self.input2 = batch2[0].to(self.device)
        self.multi_label2 = batch2[1].to(self.device).float()
        self.label2 = batch2[2].to(self.device).float()
        self.detail_label2 = batch2[3].to(self.device).float()
        
    # all augment feature seen as unseen class
    # pull in center of all, and push out center of domains
    def feature_augment_mix(self, feature, label):
        def interpolate_push(f_x, f_c):
            alpha = random.uniform(0.6, 1.5)
            new_f_x = f_x + alpha * (f_x - f_c)
            return new_f_x
        def interpolate_pull(f_x, f_c):
            # alpha = random.uniform(0.5, 1.5)
            alpha = random.uniform(0.6, 1.5)
            new_f_x = f_x + alpha * (f_c - f_x)
            return new_f_x    
        
        
        # direct manner: directly push out the feature points from the batch centriod same class 
        center = feature.mean(dim=0, keepdim=True)
        # real class
        real_mask = (label  == 0)
        real_features = feature[real_mask]
        real_center = real_features.mean(dim=0, keepdim=True) 
        aug_fncs = [
            lambda f: interpolate_push(f, real_center),
            lambda f: interpolate_pull(f, center)
        ]
        aug_fnc = random.choice(aug_fncs)
        real_aug_feature = torch.stack([aug_fnc(f) for f in real_features])    
        # fake class
        fake_mask = ~real_mask
        fake_features = feature[fake_mask]
        fake_center = fake_features.mean(dim=0, keepdim=True)    
        aug_fncs = [
            lambda f: interpolate_push(f, fake_center),
            lambda f: interpolate_pull(f, center)
        ]
        aug_fnc = random.choice(aug_fncs)
        fake_aug_feature = torch.stack([aug_fnc(f) for f in fake_features])
        #Concate
        aug_feature = torch.cat((real_aug_feature, fake_aug_feature), dim=0)         
        batch_size = aug_feature.size()[0]
        aug_mul_label = torch.tensor([0.0,0.0,1.0]*batch_size).reshape(batch_size,-1)
        aug_label = torch.tensor([1.0]*batch_size).reshape(batch_size)
        return aug_feature, aug_mul_label, aug_label
    
    
    # all augment feature seen as real/fake class
    # real: push out center of reals / seen as real class [1.0, 0.0, 0.0]
    # fake: push out and pull in center of fakes / seen as fake class [0.0, 1.0, 0.0]
    def feature_augment_mix_v2(self, feature, label):
        def interpolate_push(f_x, f_c):
            alpha = random.uniform(0.5, 1.3)
            new_f_x = f_x + alpha * (f_x - f_c)
            return new_f_x
        def interpolate_pull(f_x, f_c):
            # alpha = random.uniform(0.5, 1.5)
            alpha = random.uniform(0.5, 1.5)
            new_f_x = f_x + alpha * (f_c - f_x)
            return new_f_x       
        
        # direct manner: directly push out the feature points from the batch centriod same class 
        # real class
        real_mask = (label  == 0)
        real_features = feature[real_mask]
        real_center = real_features.mean(dim=0, keepdim=True) 
        aug_fncs = [
            lambda f: interpolate_push(f, real_center),
        ]
        aug_fnc = random.choice(aug_fncs)
        real_aug_feature = torch.stack([aug_fnc(f) for f in real_features])
        real_batch_size = real_aug_feature.size()[0]
        real_aug_mul_label = torch.tensor([1.0,0.0,0.0]*real_batch_size).reshape(real_batch_size,-1)
        real_aug_label = torch.tensor([0.0]*real_batch_size).reshape(real_batch_size)
                
        # fake class
        fake_mask = ~real_mask
        fake_features = feature[fake_mask]
        fake_center = fake_features.mean(dim=0, keepdim=True)    
        aug_fncs = [
            lambda f: interpolate_push(f, fake_center),
            lambda f: interpolate_pull(f, fake_center)
        ]
        # aug_fnc = random.choice(aug_fncs)
        fake_aug_feature = torch.stack([random.choice(aug_fncs)(f) for f in fake_features])
        # fake_aug_feature = torch.stack([aug_fnc(f) for f in fake_features])
        fake_batch_size = fake_aug_feature.size()[0]
        fake_aug_mul_label = torch.tensor([0.0,1.0,0.0]*fake_batch_size).reshape(fake_batch_size,-1)
        fake_aug_label = torch.tensor([1.0]*fake_batch_size).reshape(fake_batch_size)
        #Concate
        aug_feature = torch.cat((real_aug_feature, fake_aug_feature), dim=0)             
        aug_mul_label = torch.cat((real_aug_mul_label, fake_aug_mul_label), dim=0)    
        aug_label = torch.cat((real_aug_label, fake_aug_label), dim=0)  
        return aug_feature, aug_mul_label, aug_label    
    

    # all augment feature seen as 
    # real: push out center of reals / seen as half real class [0.7, 0.0, 0.3]
    # fake: push out and pull in center of fakes / seen as half fake class [0.0, 0.7, 0.3]    
    def feature_augment_mix_v3(self, feature, label):
        def interpolate_push(f_x, f_c):
            alpha = random.uniform(0.5, 1.3)
            new_f_x = f_x + alpha * (f_x - f_c)
            return new_f_x
        def interpolate_pull(f_x, f_c):
            # alpha = random.uniform(0.5, 1.5)
            alpha = random.uniform(0.5, 1.3)
            new_f_x = f_x + alpha * (f_c - f_x)
            return new_f_x       
        
        # direct manner: directly push out the feature points from the batch centriod same class 
        # real class
        real_mask = (label  == 0)
        real_features = feature[real_mask]
        real_center = real_features.mean(dim=0, keepdim=True) 
        aug_fncs = [
            lambda f: interpolate_push(f, real_center),
        ]
        aug_fnc = random.choice(aug_fncs)
        real_aug_feature = torch.stack([aug_fnc(f) for f in real_features])
        real_batch_size = real_aug_feature.size()[0]
        real_aug_mul_label = torch.tensor([0.5,0.0,0.5]*real_batch_size).reshape(real_batch_size,-1)
        real_aug_label = torch.tensor([0.7]*real_batch_size).reshape(real_batch_size)
                
        # fake class
        fake_mask = ~real_mask
        fake_features = feature[fake_mask]
        fake_center = fake_features.mean(dim=0, keepdim=True)    
        aug_fncs = [
            lambda f: interpolate_push(f, fake_center),
            lambda f: interpolate_pull(f, fake_center)
        ]
        # aug_fnc = random.choice(aug_fncs)
        fake_aug_feature = torch.stack([random.choice(aug_fncs)(f) for f in fake_features])
        # fake_aug_feature = torch.stack([aug_fnc(f) for f in fake_features])
        fake_batch_size = fake_aug_feature.size()[0]
        fake_aug_mul_label = torch.tensor([0.0,0.5,0.5]*fake_batch_size).reshape(fake_batch_size,-1)
        fake_aug_label = torch.tensor([0.3]*fake_batch_size).reshape(fake_batch_size)
        #Concate
        aug_feature = torch.cat((real_aug_feature, fake_aug_feature), dim=0)             
        aug_mul_label = torch.cat((real_aug_mul_label, fake_aug_mul_label), dim=0)    
        aug_label = torch.cat((real_aug_label, fake_aug_label), dim=0)  
        return aug_feature, aug_mul_label, aug_label  
    
    # all augment feature seen as unseen class
    # pull in center of all, and push out center of domains
    def feature_augment_mix_v4(self, feature, label):
        def interpolate_push(f_x, f_c):
            alpha = random.uniform(0.1, 1.0)
            new_f_x = f_x + alpha * (f_x - f_c)
            return new_f_x
        def interpolate_pull(f_x, f_c):
            # alpha = random.uniform(0.5, 1.5)
            alpha = random.uniform(0.1, 0.5)
            new_f_x = f_x + alpha * (f_c - f_x)
            return new_f_x    
        
        
        # direct manner: directly push out the feature points from the batch centriod same class 
        center = feature.mean(dim=0, keepdim=True)
        # real class
        real_mask = (label  == 0)
        real_features = feature[real_mask]
        real_center = real_features.mean(dim=0, keepdim=True) 
        aug_fncs = [
            lambda f: interpolate_push(f, real_center),
            lambda f: interpolate_pull(f, center)
        ]
        # aug_fnc = random.choice(aug_fncs)
        real_aug_feature = torch.stack([random.choice(aug_fncs)(f) for f in real_features])
        real_batch_size = real_aug_feature.size()[0]
        real_aug_mul_label = torch.tensor([0.7,0.0,0.3]*real_batch_size).reshape(real_batch_size,-1)
        real_aug_label = torch.tensor([0.3]*real_batch_size).reshape(real_batch_size)        
        # fake class
        fake_mask = ~real_mask
        fake_features = feature[fake_mask]
        fake_center = fake_features.mean(dim=0, keepdim=True)    
        aug_fncs = [
            lambda f: interpolate_push(f, fake_center),
            lambda f: interpolate_pull(f, center)
        ]
        # aug_fnc = random.choice(aug_fncs)
        fake_aug_feature = torch.stack([random.choice(aug_fncs)(f) for f in fake_features])
        fake_batch_size = fake_aug_feature.size()[0]
        fake_aug_mul_label = torch.tensor([0.0,0.7,0.3]*fake_batch_size).reshape(fake_batch_size,-1)
        fake_aug_label = torch.tensor([0.7]*fake_batch_size).reshape(fake_batch_size)
        #Concate
        aug_feature = torch.cat((real_aug_feature, fake_aug_feature), dim=0)             
        aug_mul_label = torch.cat((real_aug_mul_label, fake_aug_mul_label), dim=0)    
        aug_label = torch.cat((real_aug_label, fake_aug_label), dim=0)  
        return aug_feature, aug_mul_label, aug_label
    
    # all augment feature seen as 
    # real: push out center of reals / seen as half real class [0.7, 0.0, 0.3]
    # fake: push out and pull in center of fakes / seen as half fake class [0.0, 0.7, 0.3]    
    def feature_augment_mix_v5(self, feature, label):
        def interpolate_push(f_x, f_c, a1=0.5, a2=1.3):
            alpha = random.uniform(a1, a2)
            new_f_x = f_x + alpha * (f_x - f_c)
            return new_f_x
        def interpolate_pull(f_x, f_c, a1=0.5, a2=1.3):
            # alpha = random.uniform(0.5, 1.5)
            alpha = random.uniform(a1, a2)
            new_f_x = f_x + alpha * (f_c - f_x)
            return new_f_x       
        
        # direct manner: directly push out the feature points from the batch centriod same class 
        # real class
        real_mask = (label  == 0)
        real_features = feature[real_mask]
        real_center = real_features.mean(dim=0, keepdim=True) 
        aug_fncs = [
            lambda f: interpolate_push(f, real_center,a1=1.0,a2=1.5),
        ]
        aug_fnc = random.choice(aug_fncs)
        real_aug_feature = torch.stack([aug_fnc(f) for f in real_features])
        real_batch_size = real_aug_feature.size()[0]
        real_aug_mul_label = torch.tensor([0.0,0.3,0.7]*real_batch_size).reshape(real_batch_size,-1)
        real_aug_label = torch.tensor([0.7]*real_batch_size).reshape(real_batch_size)
                
        # fake class
        fake_mask = ~real_mask
        fake_features = feature[fake_mask]
        fake_center = fake_features.mean(dim=0, keepdim=True)    
        aug_fncs = [
            lambda f: interpolate_push(f, fake_center),
            lambda f: interpolate_pull(f, fake_center)
        ]
        # aug_fnc = random.choice(aug_fncs)
        fake_aug_feature = torch.stack([random.choice(aug_fncs)(f) for f in fake_features])
        # fake_aug_feature = torch.stack([aug_fnc(f) for f in fake_features])
        fake_batch_size = fake_aug_feature.size()[0]
        fake_aug_mul_label = torch.tensor([0.0,0.3,0.7]*fake_batch_size).reshape(fake_batch_size,-1)
        fake_aug_label = torch.tensor([0.7]*fake_batch_size).reshape(fake_batch_size)
        #Concate
        aug_feature = torch.cat((real_aug_feature, fake_aug_feature), dim=0)             
        aug_mul_label = torch.cat((real_aug_mul_label, fake_aug_mul_label), dim=0)    
        aug_label = torch.cat((real_aug_label, fake_aug_label), dim=0)  
        return aug_feature, aug_mul_label, aug_label 
    
    def get_loss(self):
        return self.BCE_loss(self.output1.squeeze(1), self.label1)

    def optimize_parameters(self, tri_decay=None):
        # Train multi_classifier with feature augment
        _, feature1 = self.model(self.input1, return_feature=True)
        _, feature2 = self.model(self.input2, return_feature=True)
        aug_feature1, aug_mul_label1, _  = self.feature_augment_mix_v5(feature1, self.label1)
        aug_feature2, aug_mul_label2, _  = self.feature_augment_mix_v5(feature2, self.label2)
        
        # for multi_classifier1 (train with detail_label -1 and 0) 
        detail_mask1_1 = (self.detail_label1==-1) | (self.detail_label1==0)
        detail_mask1_2 = (self.detail_label2==-1) | (self.detail_label1==0)

        aug_mul_label1 = aug_mul_label1.to(self.device).float()
        aug_mul_label2 = aug_mul_label2.to(self.device).float()
        
        multi_out1_1 = self.multi_classifier_1(feature1[detail_mask1_1].detach())
        multi_out1_2 = self.multi_classifier_1(feature2[detail_mask1_2].detach())
        aug_multi_out1_1 = self.multi_classifier_1(aug_feature1.detach())
        aug_multi_out1_2 = self.multi_classifier_1(aug_feature2.detach())
        
        # self.loss_MC1 = (self.CE_loss(multi_out1_1.squeeze(1), self.multi_label1[detail_mask1_1])+\
                               # self.CE_loss(multi_out1_2.squeeze(1), self.multi_label2[detail_mask1_2])) / 2
        self.loss_MC1 = (1.0) * (((self.CE_loss(multi_out1_1.squeeze(1), self.multi_label1[detail_mask1_1])+\
                               self.CE_loss(multi_out1_2.squeeze(1), self.multi_label2[detail_mask1_2])) / 2) + \
                            0.4 * (self.CE_loss(aug_multi_out1_1.squeeze(1), aug_mul_label1)+\
                                   self.CE_loss(aug_multi_out1_2.squeeze(1), aug_mul_label2)) / 2)
        self.optimizer_MC1.zero_grad()
        self.loss_MC1.backward()
        self.optimizer_MC1.step()
        
        # for multi_classifier2 (train with detail_label -1 and 1) 
        detail_mask2_1 = (self.detail_label1==-1) | (self.detail_label1==1)
        detail_mask2_2 = (self.detail_label1==-1) | (self.detail_label1==1)
        
        multi_out2_1 = self.multi_classifier_2(feature1[detail_mask2_1].detach())
        multi_out2_2 = self.multi_classifier_2(feature2[detail_mask2_2].detach())
        aug_multi_out2_1 = self.multi_classifier_2(aug_feature1.detach())
        aug_multi_out2_2 = self.multi_classifier_2(aug_feature2.detach())
        
        # self.loss_MC2 = (self.CE_loss(multi_out2_1.squeeze(1), self.multi_label1[detail_mask2_1])+\
                               # self.CE_loss(multi_out2_2.squeeze(1), self.multi_label2[detail_mask2_2])) / 2
        self.loss_MC2 = (1.0) * (((self.CE_loss(multi_out2_1.squeeze(1), self.multi_label1[detail_mask2_1])+ \
                               self.CE_loss(multi_out2_2.squeeze(1), self.multi_label2[detail_mask2_2])) / 2) + \
                            0.4 * (self.CE_loss(aug_multi_out2_1.squeeze(1), aug_mul_label1)+ \
                                   self.CE_loss(aug_multi_out2_2.squeeze(1), aug_mul_label2)) / 2)
        self.optimizer_MC2.zero_grad()
        self.loss_MC2.backward()
        self.optimizer_MC2.step()
        
        # for multi_classifier3 (train with detail_label -1 and 2) 
        detail_mask3_1 = (self.detail_label1==-1) | (self.detail_label1==2)
        detail_mask3_2 = (self.detail_label1==-1) | (self.detail_label1==2)
        
        multi_out3_1 = self.multi_classifier_3(feature1[detail_mask3_1].detach())
        multi_out3_2 = self.multi_classifier_3(feature2[detail_mask3_2].detach())
        aug_multi_out3_1 = self.multi_classifier_3(aug_feature1.detach())
        aug_multi_out3_2 = self.multi_classifier_3(aug_feature2.detach())
        
        # self.loss_MC2 = (self.CE_loss(multi_out2_1.squeeze(1), self.multi_label1[detail_mask2_1])+\
                               # self.CE_loss(multi_out2_2.squeeze(1), self.multi_label2[detail_mask2_2])) / 2
        self.loss_MC3 = (1.0) * (((self.CE_loss(multi_out3_1.squeeze(1), self.multi_label1[detail_mask3_1])+ \
                               self.CE_loss(multi_out3_2.squeeze(1), self.multi_label2[detail_mask3_2])) / 2) + \
                            0.4 * (self.CE_loss(aug_multi_out3_1.squeeze(1), aug_mul_label1)+ \
                                   self.CE_loss(aug_multi_out3_2.squeeze(1), aug_mul_label2)) / 2)
        self.optimizer_MC3.zero_grad()
        self.loss_MC3.backward()
        self.optimizer_MC3.step()
        
        
        # Train Feature extractor
        _, feature1 = self.model(self.input1, return_feature=True)
        _, feature2 = self.model(self.input2, return_feature=True)
        
        multi_out1_1 = self.multi_classifier_1(feature1[detail_mask1_1])
        multi_out1_2 = self.multi_classifier_1(feature2[detail_mask1_2])
        multi_out2_1 = self.multi_classifier_2(feature1[detail_mask2_1])
        multi_out2_2 = self.multi_classifier_2(feature2[detail_mask2_2])
        multi_out3_1 = self.multi_classifier_3(feature1[detail_mask3_1])
        multi_out3_2 = self.multi_classifier_3(feature2[detail_mask3_2])
        
        # self.loss_F = (1.0) * ((self.CE_loss(multi_out1_1.squeeze(1), self.multi_label1[detail_mask1_1])+ \
                               # self.CE_loss(multi_out1_2.squeeze(1), self.multi_label2[detail_mask1_2])) / 2) 
        self.loss_F = (1.0) * ((self.CE_loss(multi_out1_1.squeeze(1), self.multi_label1[detail_mask1_1])+ \
                               self.CE_loss(multi_out1_2.squeeze(1), self.multi_label2[detail_mask1_2])) / 2) + \
                      (0.4) * ((self.CE_loss(multi_out2_1.squeeze(1), self.multi_label1[detail_mask2_1])+ \
                               self.CE_loss(multi_out2_2.squeeze(1), self.multi_label2[detail_mask2_2])) / 2) + \
                      (0.4) * ((self.CE_loss(multi_out3_1.squeeze(1), self.multi_label1[detail_mask3_1])+ \
                               self.CE_loss(multi_out3_2.squeeze(1), self.multi_label2[detail_mask3_2])) / 2)          
                               
        self.optimizer_F.zero_grad()
        self.loss_F.backward()
        self.optimizer_F.step()
        
        # Train binary classifier with feature augment
        _, feature1 = self.model(self.input1, return_feature=True)
        _, feature2 = self.model(self.input2, return_feature=True)
                
        aug_feature1, _, aug_label1 = self.feature_augment_mix_v5(feature1, self.label1)
        aug_feature2, _, aug_label2 = self.feature_augment_mix_v5(feature2, self.label2)
        
        aug_label1 = aug_label1.to(self.device).float()
        aug_label2 = aug_label2.to(self.device).float()
        aug_bin_out1 = self.classifier(aug_feature1.detach())
        aug_bin_out2 = self.classifier(aug_feature2.detach())
        
        loss_BC_aug = 0.4 * (self.BCE_loss(aug_bin_out1.squeeze(), aug_label1)+ \
                             self.BCE_loss(aug_bin_out2.squeeze(), aug_label2) / 2) 
        
        bin_out1_1 = self.classifier(feature1[detail_mask1_1].detach())
        bin_out1_2 = self.classifier(feature2[detail_mask1_2].detach())
        
        loss_BC_1 = 1.0 * ((self.BCE_loss(bin_out1_1.squeeze(1), self.label1[detail_mask1_1]) + \
                              self.BCE_loss(bin_out1_2.squeeze(1), self.label2[detail_mask1_2])) / 2)
                            
        bin_out2_1 = self.classifier(feature1[detail_mask2_1].detach())
        bin_out2_2 = self.classifier(feature2[detail_mask2_2].detach())
        
        loss_BC_2 = 0.4 * ((self.BCE_loss(bin_out2_1.squeeze(1), self.label1[detail_mask2_1]) + \
                            self.BCE_loss(bin_out2_2.squeeze(1), self.label2[detail_mask2_2])) / 2)
                            
        bin_out3_1 = self.classifier(feature1[detail_mask3_1].detach())
        bin_out3_2 = self.classifier(feature2[detail_mask3_2].detach())
        
        loss_BC_3 = 0.4 * ((self.BCE_loss(bin_out3_1.squeeze(1), self.label1[detail_mask3_1]) + \
                            self.BCE_loss(bin_out3_2.squeeze(1), self.label2[detail_mask3_2])) / 2)                            
        
        self.loss_BC = loss_BC_1 + loss_BC_2 + loss_BC_3 + loss_BC_aug
        # self.loss_BC = loss_BC_1 + loss_BC_2
        # self.loss_BC = loss_BC_1 + loss_BC_aug
        # self.loss_BC = loss_BC_1
        self.optimizer_BC.zero_grad()
        self.loss_BC.backward()
        self.optimizer_BC.step()
        
        self.loss = (self.loss_BC + self.loss_MC1+self.loss_MC2 + self.loss_F) / 4


class mix_Trainer_v2_3(BaseModel):
    def name(self):
        return 'mix_Trainer_v2_3'

    def __init__(self, opt):
        super(mix_Trainer_v2_3, self).__init__(opt)

        if self.isTrain and not opt.continue_train:
            self.model = resnet50(num_classes=1)
            # torch.nn.init.normal_(self.model.fc.weight.data, 0.0, opt.init_gain)
            if opt.pretrained_path != None:
                state_dict = torch.load(opt.pretrained_path, map_location='cpu')
                self.model.load_state_dict(state_dict['model'])
            else:
                print(f"your pretrained_path is empty, please set the path.")
            
            # parallel classifier 1
            self.multi_classifier_1 = Classifier(num_classes=3)
            torch.nn.init.normal_(self.multi_classifier_1.fc1.weight.data, 0.0, opt.init_gain)
            torch.nn.init.normal_(self.multi_classifier_1.fc2.weight.data, 0.0, opt.init_gain)
            
            # parallel classifier 2
            self.multi_classifier_2 = Classifier(num_classes=3)
            torch.nn.init.normal_(self.multi_classifier_2.fc1.weight.data, 0.0, opt.init_gain)
            torch.nn.init.normal_(self.multi_classifier_2.fc2.weight.data, 0.0, opt.init_gain)
            
            
            self.classifier = Classifier(num_classes=1)
            torch.nn.init.normal_(self.classifier.fc1.weight.data, 0.0, opt.init_gain)
            torch.nn.init.normal_(self.classifier.fc2.weight.data, 0.0, opt.init_gain)
            
        # Trainable parameters    
        params_F = []
    
        for name, p in self.model.named_parameters():
            if  name=="fc.weight" or name=="fc.bias": 
                p.requires_grad = False
            else:
                params_F.append(p)

        params_Multi_C1 = self.multi_classifier_1.parameters()          
        params_Multi_C2 = self.multi_classifier_2.parameters()         
        params_Binary_C = self.classifier.parameters()            
        
        if self.isTrain:
            self.BCE_loss = nn.BCEWithLogitsLoss()
            self.CE_loss = nn.CrossEntropyLoss()
            # self.triplet = nn.TripletMarginLoss(margin=0.2, p=2, eps=1e-7)
            self.triplet = TripletLoss_v2(margin=0.2)
            self.triplet.to(opt.gpu_ids[0])
            # initialize optimizers
            if opt.optim == 'adam':
                self.optimizer_F = torch.optim.Adam(params_F,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_MC1 = torch.optim.Adam(params_Multi_C1,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_MC2 = torch.optim.Adam(params_Multi_C2,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))                                  
                self.optimizer_BC = torch.optim.Adam(params_Binary_C,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))                                  
            elif opt.optim == 'sgd':
                self.optimizer_F = torch.optim.SGD(params_F,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_MC1 = torch.optim.SGD(params_Multi_C1,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_MC2 = torch.optim.Adam(params_Multi_C2,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))                                   
                self.optimizer_BC = torch.optim.SGD(params_Binary_C,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))  
            else:
                raise ValueError("optim should be [adam, sgd]")
            
        self.model.to(opt.gpu_ids[0])
        self.classifier.to(opt.gpu_ids[0])
        self.multi_classifier_1.to(opt.gpu_ids[0])
        self.multi_classifier_2.to(opt.gpu_ids[0])
        

    def adjust_learning_rate(self, min_lr=1e-7):
        for param_group in self.optimizer_F.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        for param_group in self.optimizer_MC1.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        for param_group in self.optimizer_MC2.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False        
        for param_group in self.optimizer_BC.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False        
        return True

    def set_input(self, batch1, batch2):
        # img, multi_label, label, detail_label
        
        self.input1 = batch1[0].to(self.device)
        self.multi_label1 = batch1[1].to(self.device).float()
        self.label1 = batch1[2].to(self.device).float()
        self.detail_label1 = batch1[3].to(self.device).float()
        
        self.input2 = batch2[0].to(self.device)
        self.multi_label2 = batch2[1].to(self.device).float()
        self.label2 = batch2[2].to(self.device).float()
        self.detail_label2 = batch2[3].to(self.device).float()
        
    # all augment feature seen as unseen class
    # pull in center of all, and push out center of domains
    def feature_augment_mix(self, feature, label):
        def interpolate_push(f_x, f_c):
            alpha = random.uniform(0.6, 1.5)
            new_f_x = f_x + alpha * (f_x - f_c)
            return new_f_x
        def interpolate_pull(f_x, f_c):
            # alpha = random.uniform(0.5, 1.5)
            alpha = random.uniform(0.6, 1.5)
            new_f_x = f_x + alpha * (f_c - f_x)
            return new_f_x    
        
        
        # direct manner: directly push out the feature points from the batch centriod same class 
        center = feature.mean(dim=0, keepdim=True)
        # real class
        real_mask = (label  == 0)
        real_features = feature[real_mask]
        real_center = real_features.mean(dim=0, keepdim=True) 
        aug_fncs = [
            lambda f: interpolate_push(f, real_center),
            lambda f: interpolate_pull(f, center)
        ]
        aug_fnc = random.choice(aug_fncs)
        real_aug_feature = torch.stack([aug_fnc(f) for f in real_features])    
        # fake class
        fake_mask = ~real_mask
        fake_features = feature[fake_mask]
        fake_center = fake_features.mean(dim=0, keepdim=True)    
        aug_fncs = [
            lambda f: interpolate_push(f, fake_center),
            lambda f: interpolate_pull(f, center)
        ]
        aug_fnc = random.choice(aug_fncs)
        fake_aug_feature = torch.stack([aug_fnc(f) for f in fake_features])
        #Concate
        aug_feature = torch.cat((real_aug_feature, fake_aug_feature), dim=0)         
        batch_size = aug_feature.size()[0]
        aug_mul_label = torch.tensor([0.0,0.0,1.0]*batch_size).reshape(batch_size,-1)
        aug_label = torch.tensor([1.0]*batch_size).reshape(batch_size)
        return aug_feature, aug_mul_label, aug_label
    
    
    # all augment feature seen as real/fake class
    # real: push out center of reals / seen as real class [1.0, 0.0, 0.0]
    # fake: push out and pull in center of fakes / seen as fake class [0.0, 1.0, 0.0]
    def feature_augment_mix_v2(self, feature, label):
        def interpolate_push(f_x, f_c):
            alpha = random.uniform(0.5, 1.3)
            new_f_x = f_x + alpha * (f_x - f_c)
            return new_f_x
        def interpolate_pull(f_x, f_c):
            # alpha = random.uniform(0.5, 1.5)
            alpha = random.uniform(0.5, 1.5)
            new_f_x = f_x + alpha * (f_c - f_x)
            return new_f_x       
        
        # direct manner: directly push out the feature points from the batch centriod same class 
        # real class
        real_mask = (label  == 0)
        real_features = feature[real_mask]
        real_center = real_features.mean(dim=0, keepdim=True) 
        aug_fncs = [
            lambda f: interpolate_push(f, real_center),
        ]
        aug_fnc = random.choice(aug_fncs)
        real_aug_feature = torch.stack([aug_fnc(f) for f in real_features])
        real_batch_size = real_aug_feature.size()[0]
        real_aug_mul_label = torch.tensor([1.0,0.0,0.0]*real_batch_size).reshape(real_batch_size,-1)
        real_aug_label = torch.tensor([0.0]*real_batch_size).reshape(real_batch_size)
                
        # fake class
        fake_mask = ~real_mask
        fake_features = feature[fake_mask]
        fake_center = fake_features.mean(dim=0, keepdim=True)    
        aug_fncs = [
            lambda f: interpolate_push(f, fake_center),
            lambda f: interpolate_pull(f, fake_center)
        ]
        # aug_fnc = random.choice(aug_fncs)
        fake_aug_feature = torch.stack([random.choice(aug_fncs)(f) for f in fake_features])
        # fake_aug_feature = torch.stack([aug_fnc(f) for f in fake_features])
        fake_batch_size = fake_aug_feature.size()[0]
        fake_aug_mul_label = torch.tensor([0.0,1.0,0.0]*fake_batch_size).reshape(fake_batch_size,-1)
        fake_aug_label = torch.tensor([1.0]*fake_batch_size).reshape(fake_batch_size)
        #Concate
        aug_feature = torch.cat((real_aug_feature, fake_aug_feature), dim=0)             
        aug_mul_label = torch.cat((real_aug_mul_label, fake_aug_mul_label), dim=0)    
        aug_label = torch.cat((real_aug_label, fake_aug_label), dim=0)  
        return aug_feature, aug_mul_label, aug_label    
    

    # all augment feature seen as 
    # real: push out center of reals / seen as half real class [0.7, 0.0, 0.3]
    # fake: push out and pull in center of fakes / seen as half fake class [0.0, 0.7, 0.3]    
    def feature_augment_mix_v3(self, feature, label):
        def interpolate_push(f_x, f_c):
            alpha = random.uniform(0.5, 1.3)
            new_f_x = f_x + alpha * (f_x - f_c)
            return new_f_x
        def interpolate_pull(f_x, f_c):
            # alpha = random.uniform(0.5, 1.5)
            alpha = random.uniform(0.5, 1.3)
            new_f_x = f_x + alpha * (f_c - f_x)
            return new_f_x       
        
        # direct manner: directly push out the feature points from the batch centriod same class 
        # real class
        real_mask = (label  == 0)
        real_features = feature[real_mask]
        real_center = real_features.mean(dim=0, keepdim=True) 
        aug_fncs = [
            lambda f: interpolate_push(f, real_center),
        ]
        aug_fnc = random.choice(aug_fncs)
        real_aug_feature = torch.stack([aug_fnc(f) for f in real_features])
        real_batch_size = real_aug_feature.size()[0]
        real_aug_mul_label = torch.tensor([0.7,0.0,0.3]*real_batch_size).reshape(real_batch_size,-1)
        real_aug_label = torch.tensor([0.3]*real_batch_size).reshape(real_batch_size)
                
        # fake class
        fake_mask = ~real_mask
        fake_features = feature[fake_mask]
        fake_center = fake_features.mean(dim=0, keepdim=True)    
        aug_fncs = [
            lambda f: interpolate_push(f, fake_center),
            lambda f: interpolate_pull(f, fake_center)
        ]
        # aug_fnc = random.choice(aug_fncs)
        fake_aug_feature = torch.stack([random.choice(aug_fncs)(f) for f in fake_features])
        # fake_aug_feature = torch.stack([aug_fnc(f) for f in fake_features])
        fake_batch_size = fake_aug_feature.size()[0]
        fake_aug_mul_label = torch.tensor([0.0,0.7,0.3]*fake_batch_size).reshape(fake_batch_size,-1)
        fake_aug_label = torch.tensor([0.7]*fake_batch_size).reshape(fake_batch_size)
        #Concate
        aug_feature = torch.cat((real_aug_feature, fake_aug_feature), dim=0)             
        aug_mul_label = torch.cat((real_aug_mul_label, fake_aug_mul_label), dim=0)    
        aug_label = torch.cat((real_aug_label, fake_aug_label), dim=0)  
        return aug_feature, aug_mul_label, aug_label  
    
    # all augment feature seen as unseen class
    # pull in center of all, and push out center of domains
    def feature_augment_mix_v4(self, feature, label):
        def interpolate_push(f_x, f_c):
            alpha = random.uniform(0.1, 1.0)
            new_f_x = f_x + alpha * (f_x - f_c)
            return new_f_x
        def interpolate_pull(f_x, f_c):
            # alpha = random.uniform(0.5, 1.5)
            alpha = random.uniform(0.1, 0.5)
            new_f_x = f_x + alpha * (f_c - f_x)
            return new_f_x    
        
        
        # direct manner: directly push out the feature points from the batch centriod same class 
        center = feature.mean(dim=0, keepdim=True)
        # real class
        real_mask = (label  == 0)
        real_features = feature[real_mask]
        real_center = real_features.mean(dim=0, keepdim=True) 
        aug_fncs = [
            lambda f: interpolate_push(f, real_center),
            lambda f: interpolate_pull(f, center)
        ]
        # aug_fnc = random.choice(aug_fncs)
        real_aug_feature = torch.stack([random.choice(aug_fncs)(f) for f in real_features])
        real_batch_size = real_aug_feature.size()[0]
        real_aug_mul_label = torch.tensor([0.7,0.0,0.3]*real_batch_size).reshape(real_batch_size,-1)
        real_aug_label = torch.tensor([0.3]*real_batch_size).reshape(real_batch_size)        
        # fake class
        fake_mask = ~real_mask
        fake_features = feature[fake_mask]
        fake_center = fake_features.mean(dim=0, keepdim=True)    
        aug_fncs = [
            lambda f: interpolate_push(f, fake_center),
            lambda f: interpolate_pull(f, center)
        ]
        # aug_fnc = random.choice(aug_fncs)
        fake_aug_feature = torch.stack([random.choice(aug_fncs)(f) for f in fake_features])
        fake_batch_size = fake_aug_feature.size()[0]
        fake_aug_mul_label = torch.tensor([0.0,0.7,0.3]*fake_batch_size).reshape(fake_batch_size,-1)
        fake_aug_label = torch.tensor([0.7]*fake_batch_size).reshape(fake_batch_size)
        #Concate
        aug_feature = torch.cat((real_aug_feature, fake_aug_feature), dim=0)             
        aug_mul_label = torch.cat((real_aug_mul_label, fake_aug_mul_label), dim=0)    
        aug_label = torch.cat((real_aug_label, fake_aug_label), dim=0)  
        return aug_feature, aug_mul_label, aug_label
    
    def compute_triplet_loss(self, feature1, feature2):
        # compute each batch domain center
        real_mask1 = (self.detail_label1 == -1)
        real_mask2 = (self.detail_label2 == -1)
        fake_mask1_0 = (self.detail_label1 == 0)
        fake_mask2_0 = (self.detail_label2 == 0)
        fake_mask1_1 = (self.detail_label1 == 1)
        fake_mask2_1 = (self.detail_label2 == 1)
        mini_num = min(real_mask1.sum().item(), real_mask2.sum().item(), fake_mask1_0.sum().item(),
                       fake_mask2_0.sum().item(), fake_mask1_1.sum().item(), fake_mask2_1.sum().item()) 
        # real feature
        real_feature1 = feature1[real_mask1][:mini_num]
        real_feature2 = feature2[real_mask2][:mini_num]
        # detail_label=0 fake feature
        fake_feature1_0 = feature1[fake_mask1_0][:mini_num]
        fake_feature2_0 = feature2[fake_mask2_0][:mini_num]    
        
        # detail_label=1 fake' s center 
        fake_feature1_1 = feature1[fake_mask1_1][:mini_num]
        fake_feature2_1 = feature2[fake_mask2_1][:mini_num]
        
        #Dis(fake_full, fake_patch) <= Dis(fake, real)
        Anchor = torch.cat((fake_feature1_0, fake_feature2_0,fake_feature1_1,fake_feature2_1), dim=0)  
        Positive = torch.cat((fake_feature2_0, fake_feature1_0,fake_feature2_1,fake_feature1_1), dim=0)  
        Negative = torch.cat((real_feature1, real_feature2, real_feature1, real_feature2), dim=0)  
        loss = self.triplet(Anchor, Positive, Negative)
        # if torch.isnan(loss):
            # print("The loss is NaN.")
        # elif torch.isinf(loss):
            # print("The loss is Inf.")
        # else:
            # print("The loss is:", loss.item())
        return loss    
    
     
    def get_loss(self):
        return self.BCE_loss(self.output1.squeeze(1), self.label1)

    def optimize_parameters(self, tri_decay=None):
        # Train multi_classifier with feature augment
        _, feature1 = self.model(self.input1, return_feature=True)
        _, feature2 = self.model(self.input2, return_feature=True)
        aug_feature1, aug_mul_label1, _  = self.feature_augment_mix_v3(feature1, self.label1)
        aug_feature2, aug_mul_label2, _  = self.feature_augment_mix_v3(feature2, self.label2)
        
        # for multi_classifier1 (train with detail_label -1 and 0) 
        detail_mask1_1 = ~(self.detail_label1==1)
        detail_mask1_2 = ~(self.detail_label2==1)

        aug_mul_label1 = aug_mul_label1.to(self.device).float()
        aug_mul_label2 = aug_mul_label2.to(self.device).float()
        
        multi_out1_1 = self.multi_classifier_1(feature1[detail_mask1_1].detach())
        multi_out1_2 = self.multi_classifier_1(feature2[detail_mask1_2].detach())
        aug_multi_out1_1 = self.multi_classifier_1(aug_feature1.detach())
        aug_multi_out1_2 = self.multi_classifier_1(aug_feature2.detach())
        
        # self.loss_MC1 = (self.CE_loss(multi_out1_1.squeeze(1), self.multi_label1[detail_mask1_1])+\
                               # self.CE_loss(multi_out1_2.squeeze(1), self.multi_label2[detail_mask1_2])) / 2
        self.loss_MC1 = (1.0) * (((self.CE_loss(multi_out1_1.squeeze(1), self.multi_label1[detail_mask1_1])+\
                               self.CE_loss(multi_out1_2.squeeze(1), self.multi_label2[detail_mask1_2])) / 2) + \
                            0.4 * (self.CE_loss(aug_multi_out1_1.squeeze(1), aug_mul_label1)+\
                                   self.CE_loss(aug_multi_out1_2.squeeze(1), aug_mul_label2)) / 2)
        self.optimizer_MC1.zero_grad()
        self.loss_MC1.backward()
        self.optimizer_MC1.step()
        
        # for multi_classifier2 (train with detail_label -1 and 1) 
        detail_mask2_1 = ~(self.detail_label1==0)
        detail_mask2_2 = ~(self.detail_label2==0)
        
        multi_out2_1 = self.multi_classifier_2(feature1[detail_mask2_1].detach())
        multi_out2_2 = self.multi_classifier_2(feature2[detail_mask2_2].detach())
        aug_multi_out2_1 = self.multi_classifier_2(aug_feature1.detach())
        aug_multi_out2_2 = self.multi_classifier_2(aug_feature2.detach())
        
        # self.loss_MC2 = (self.CE_loss(multi_out2_1.squeeze(1), self.multi_label1[detail_mask2_1])+\
                               # self.CE_loss(multi_out2_2.squeeze(1), self.multi_label2[detail_mask2_2])) / 2
        self.loss_MC2 = (1.0) * (((self.CE_loss(multi_out2_1.squeeze(1), self.multi_label1[detail_mask2_1])+ \
                               self.CE_loss(multi_out2_2.squeeze(1), self.multi_label2[detail_mask2_2])) / 2) + \
                            0.4 * (self.CE_loss(aug_multi_out2_1.squeeze(1), aug_mul_label1)+ \
                                   self.CE_loss(aug_multi_out2_2.squeeze(1), aug_mul_label2)) / 2)
        self.optimizer_MC2.zero_grad()
        self.loss_MC2.backward()
        self.optimizer_MC2.step()
        
        
        # Train Feature extractor
        _, feature1 = self.model(self.input1, return_feature=True)
        _, feature2 = self.model(self.input2, return_feature=True)
        
        # triplet loss
        tri_loss = self.compute_triplet_loss(feature1, feature2)
        
        multi_out1_1 = self.multi_classifier_1(feature1[detail_mask1_1])
        multi_out1_2 = self.multi_classifier_1(feature2[detail_mask1_2])
        multi_out2_1 = self.multi_classifier_2(feature1[detail_mask2_1])
        multi_out2_2 = self.multi_classifier_2(feature2[detail_mask2_2])
        
        # self.loss_F = (1.0) * ((self.CE_loss(multi_out1_1.squeeze(1), self.multi_label1[detail_mask1_1])+ \
                               # self.CE_loss(multi_out1_2.squeeze(1), self.multi_label2[detail_mask1_2])) / 2) 
        self.loss_F = (1.0) * ((self.CE_loss(multi_out1_1.squeeze(1), self.multi_label1[detail_mask1_1])+ \
                               self.CE_loss(multi_out1_2.squeeze(1), self.multi_label2[detail_mask1_2])) / 2) + \
                      (0.4) * ((self.CE_loss(multi_out2_1.squeeze(1), self.multi_label1[detail_mask2_1])+ \
                               self.CE_loss(multi_out2_2.squeeze(1), self.multi_label2[detail_mask2_2])) / 2) + \
                      (0.4) * tri_loss
                               
        self.optimizer_F.zero_grad()
        self.loss_F.backward()
        self.optimizer_F.step()
        
        # Train binary classifier with feature augment
        _, feature1 = self.model(self.input1, return_feature=True)
        _, feature2 = self.model(self.input2, return_feature=True)
                
        aug_feature1, _, aug_label1 = self.feature_augment_mix_v3(feature1, self.label1)
        aug_feature2, _, aug_label2 = self.feature_augment_mix_v3(feature2, self.label2)
        
        aug_label1 = aug_label1.to(self.device).float()
        aug_label2 = aug_label2.to(self.device).float()
        aug_bin_out1 = self.classifier(aug_feature1.detach())
        aug_bin_out2 = self.classifier(aug_feature2.detach())
        
        loss_BC_aug = 0.4 * (self.BCE_loss(aug_bin_out1.squeeze(), aug_label1)+ \
                             self.BCE_loss(aug_bin_out2.squeeze(), aug_label2) / 2) 
        
        bin_out1_1 = self.classifier(feature1[detail_mask1_1].detach())
        bin_out1_2 = self.classifier(feature2[detail_mask1_2].detach())
        
        loss_BC_1 = 1.0 * ((self.BCE_loss(bin_out1_1.squeeze(1), self.label1[detail_mask1_1]) + \
                              self.BCE_loss(bin_out1_2.squeeze(1), self.label2[detail_mask1_2])) / 2)
                            
        bin_out2_1 = self.classifier(feature1[detail_mask2_1].detach())
        bin_out2_2 = self.classifier(feature2[detail_mask2_2].detach())
        
        loss_BC_2 = 0.4 * ((self.BCE_loss(bin_out2_1.squeeze(1), self.label1[detail_mask2_1]) + \
                            self.BCE_loss(bin_out2_2.squeeze(1), self.label2[detail_mask2_2])) / 2)            
        
        self.loss_BC = loss_BC_1 + loss_BC_2 + loss_BC_aug
        # self.loss_BC = loss_BC_1 + loss_BC_2
        # self.loss_BC = loss_BC_1 + loss_BC_aug
        # self.loss_BC = loss_BC_1
        self.optimizer_BC.zero_grad()
        self.loss_BC.backward()
        self.optimizer_BC.step()
        
        self.loss = (self.loss_BC + self.loss_MC1+self.loss_MC2 + self.loss_F) / 4


class triple_Trainer_v2_2(BaseModel):
    def name(self):
        return 'triple_Trainer_v2_2'

    def __init__(self, opt):
        super(triple_Trainer_v2_2, self).__init__(opt)

        if self.isTrain and not opt.continue_train:
            self.model = resnet50(num_classes=1)
            # torch.nn.init.normal_(self.model.fc.weight.data, 0.0, opt.init_gain)
            if opt.pretrained_path != None:
                state_dict = torch.load(opt.pretrained_path, map_location='cpu')
                self.model.load_state_dict(state_dict['model'])
            else:
                print(f"your pretrained_path is empty, please set the path.")
            
            # parallel classifier 1
            self.multi_classifier_1 = Classifier(num_classes=3)
            torch.nn.init.normal_(self.multi_classifier_1.fc1.weight.data, 0.0, opt.init_gain)
            torch.nn.init.normal_(self.multi_classifier_1.fc2.weight.data, 0.0, opt.init_gain)
            
            # parallel classifier 2
            self.multi_classifier_2 = Classifier(num_classes=3)
            torch.nn.init.normal_(self.multi_classifier_2.fc1.weight.data, 0.0, opt.init_gain)
            torch.nn.init.normal_(self.multi_classifier_2.fc2.weight.data, 0.0, opt.init_gain)
            
            
            self.classifier = Classifier(num_classes=1)
            torch.nn.init.normal_(self.classifier.fc1.weight.data, 0.0, opt.init_gain)
            torch.nn.init.normal_(self.classifier.fc2.weight.data, 0.0, opt.init_gain)
            
        # Trainable parameters    
        params_F = []
    
        for name, p in self.model.named_parameters():
            if  name=="fc.weight" or name=="fc.bias": 
                p.requires_grad = False
            else:
                params_F.append(p)

        params_Multi_C1 = self.multi_classifier_1.parameters()          
        params_Multi_C2 = self.multi_classifier_2.parameters()         
        params_Binary_C = self.classifier.parameters()            
        
        if self.isTrain:
            self.BCE_loss = nn.BCEWithLogitsLoss()
            self.CE_loss = nn.CrossEntropyLoss()
            # self.triplet = nn.TripletMarginLoss(margin=0.2, p=2, eps=1e-7)
            self.triplet = TripletLoss_v2(margin=0.2)
            self.triplet.to(opt.gpu_ids[0])
            # initialize optimizers
            if opt.optim == 'adam':
                self.optimizer_F = torch.optim.Adam(params_F,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_MC1 = torch.optim.Adam(params_Multi_C1,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_MC2 = torch.optim.Adam(params_Multi_C2,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))                                  
                self.optimizer_BC = torch.optim.Adam(params_Binary_C,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))                                  
            elif opt.optim == 'sgd':
                self.optimizer_F = torch.optim.SGD(params_F,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_MC1 = torch.optim.SGD(params_Multi_C1,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_MC2 = torch.optim.Adam(params_Multi_C2,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))                                   
                self.optimizer_BC = torch.optim.SGD(params_Binary_C,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))  
            else:
                raise ValueError("optim should be [adam, sgd]")
            
        self.model.to(opt.gpu_ids[0])
        self.classifier.to(opt.gpu_ids[0])
        self.multi_classifier_1.to(opt.gpu_ids[0])
        self.multi_classifier_2.to(opt.gpu_ids[0])
        

    def adjust_learning_rate(self, min_lr=1e-7):
        for param_group in self.optimizer_F.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        for param_group in self.optimizer_MC1.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        for param_group in self.optimizer_MC2.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False        
        for param_group in self.optimizer_BC.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False        
        return True

    def set_input(self, batch1, batch2):
        # img, multi_label, label, detail_label, real_img
        
        self.input1 = batch1[0].to(self.device)
        self.multi_label1 = batch1[1].to(self.device).float()
        self.label1 = batch1[2].to(self.device).float()
        self.detail_label1 = batch1[3].to(self.device).float()
        self.real_img1 = batch1[4].to(self.device)
        
        self.input2 = batch2[0].to(self.device)
        self.multi_label2 = batch2[1].to(self.device).float()
        self.label2 = batch2[2].to(self.device).float()
        self.detail_label2 = batch2[3].to(self.device).float()
        self.real_img2 = batch2[4].to(self.device)
        
    # all augment feature seen as unseen class
    # pull in center of all, and push out center of domains
    def feature_augment_mix(self, feature, label):
        def interpolate_push(f_x, f_c):
            alpha = random.uniform(0.6, 1.5)
            new_f_x = f_x + alpha * (f_x - f_c)
            return new_f_x
        def interpolate_pull(f_x, f_c):
            # alpha = random.uniform(0.5, 1.5)
            alpha = random.uniform(0.6, 1.5)
            new_f_x = f_x + alpha * (f_c - f_x)
            return new_f_x    
        
        
        # direct manner: directly push out the feature points from the batch centriod same class 
        center = feature.mean(dim=0, keepdim=True)
        # real class
        real_mask = (label  == 0)
        real_features = feature[real_mask]
        real_center = real_features.mean(dim=0, keepdim=True) 
        aug_fncs = [
            lambda f: interpolate_push(f, real_center),
            lambda f: interpolate_pull(f, center)
        ]
        aug_fnc = random.choice(aug_fncs)
        real_aug_feature = torch.stack([aug_fnc(f) for f in real_features])    
        # fake class
        fake_mask = ~real_mask
        fake_features = feature[fake_mask]
        fake_center = fake_features.mean(dim=0, keepdim=True)    
        aug_fncs = [
            lambda f: interpolate_push(f, fake_center),
            lambda f: interpolate_pull(f, center)
        ]
        aug_fnc = random.choice(aug_fncs)
        fake_aug_feature = torch.stack([aug_fnc(f) for f in fake_features])
        #Concate
        aug_feature = torch.cat((real_aug_feature, fake_aug_feature), dim=0)         
        batch_size = aug_feature.size()[0]
        aug_mul_label = torch.tensor([0.0,0.0,1.0]*batch_size).reshape(batch_size,-1)
        aug_label = torch.tensor([1.0]*batch_size).reshape(batch_size)
        return aug_feature, aug_mul_label, aug_label
    
    
    # all augment feature seen as real/fake class
    # real: push out center of reals / seen as real class [1.0, 0.0, 0.0]
    # fake: push out and pull in center of fakes / seen as fake class [0.0, 1.0, 0.0]
    def feature_augment_mix_v2(self, feature, label):
        def interpolate_push(f_x, f_c):
            alpha = random.uniform(0.5, 1.3)
            new_f_x = f_x + alpha * (f_x - f_c)
            return new_f_x
        def interpolate_pull(f_x, f_c):
            # alpha = random.uniform(0.5, 1.5)
            alpha = random.uniform(0.5, 1.5)
            new_f_x = f_x + alpha * (f_c - f_x)
            return new_f_x       
        
        # direct manner: directly push out the feature points from the batch centriod same class 
        # real class
        real_mask = (label  == 0)
        real_features = feature[real_mask]
        real_center = real_features.mean(dim=0, keepdim=True) 
        aug_fncs = [
            lambda f: interpolate_push(f, real_center),
        ]
        aug_fnc = random.choice(aug_fncs)
        real_aug_feature = torch.stack([aug_fnc(f) for f in real_features])
        real_batch_size = real_aug_feature.size()[0]
        real_aug_mul_label = torch.tensor([1.0,0.0,0.0]*real_batch_size).reshape(real_batch_size,-1)
        real_aug_label = torch.tensor([0.0]*real_batch_size).reshape(real_batch_size)
                
        # fake class
        fake_mask = ~real_mask
        fake_features = feature[fake_mask]
        fake_center = fake_features.mean(dim=0, keepdim=True)    
        aug_fncs = [
            lambda f: interpolate_push(f, fake_center),
            lambda f: interpolate_pull(f, fake_center)
        ]
        # aug_fnc = random.choice(aug_fncs)
        fake_aug_feature = torch.stack([random.choice(aug_fncs)(f) for f in fake_features])
        # fake_aug_feature = torch.stack([aug_fnc(f) for f in fake_features])
        fake_batch_size = fake_aug_feature.size()[0]
        fake_aug_mul_label = torch.tensor([0.0,1.0,0.0]*fake_batch_size).reshape(fake_batch_size,-1)
        fake_aug_label = torch.tensor([1.0]*fake_batch_size).reshape(fake_batch_size)
        #Concate
        aug_feature = torch.cat((real_aug_feature, fake_aug_feature), dim=0)             
        aug_mul_label = torch.cat((real_aug_mul_label, fake_aug_mul_label), dim=0)    
        aug_label = torch.cat((real_aug_label, fake_aug_label), dim=0)  
        return aug_feature, aug_mul_label, aug_label    
    

    # all augment feature seen as 
    # real: push out center of reals / seen as half real class [0.7, 0.0, 0.3]
    # fake: push out and pull in center of fakes / seen as half fake class [0.0, 0.7, 0.3]    
    def feature_augment_mix_v3(self, feature, label):
        def interpolate_push(f_x, f_c):
            alpha = random.uniform(0.5, 1.3)
            new_f_x = f_x + alpha * (f_x - f_c)
            return new_f_x
        def interpolate_pull(f_x, f_c):
            # alpha = random.uniform(0.5, 1.5)
            alpha = random.uniform(0.5, 1.3)
            new_f_x = f_x + alpha * (f_c - f_x)
            return new_f_x       
        
        # direct manner: directly push out the feature points from the batch centriod same class 
        # real class
        real_mask = (label  == 0)
        real_features = feature[real_mask]
        real_center = real_features.mean(dim=0, keepdim=True) 
        aug_fncs = [
            lambda f: interpolate_push(f, real_center),
        ]
        aug_fnc = random.choice(aug_fncs)
        real_aug_feature = torch.stack([aug_fnc(f) for f in real_features])
        real_batch_size = real_aug_feature.size()[0]
        real_aug_mul_label = torch.tensor([0.7,0.0,0.3]*real_batch_size).reshape(real_batch_size,-1)
        real_aug_label = torch.tensor([0.3]*real_batch_size).reshape(real_batch_size)
                
        # fake class
        fake_mask = ~real_mask
        fake_features = feature[fake_mask]
        fake_center = fake_features.mean(dim=0, keepdim=True)    
        aug_fncs = [
            lambda f: interpolate_push(f, fake_center),
            lambda f: interpolate_pull(f, fake_center)
        ]
        # aug_fnc = random.choice(aug_fncs)
        fake_aug_feature = torch.stack([random.choice(aug_fncs)(f) for f in fake_features])
        # fake_aug_feature = torch.stack([aug_fnc(f) for f in fake_features])
        fake_batch_size = fake_aug_feature.size()[0]
        fake_aug_mul_label = torch.tensor([0.0,0.7,0.3]*fake_batch_size).reshape(fake_batch_size,-1)
        fake_aug_label = torch.tensor([0.7]*fake_batch_size).reshape(fake_batch_size)
        #Concate
        aug_feature = torch.cat((real_aug_feature, fake_aug_feature), dim=0)             
        aug_mul_label = torch.cat((real_aug_mul_label, fake_aug_mul_label), dim=0)    
        aug_label = torch.cat((real_aug_label, fake_aug_label), dim=0)  
        return aug_feature, aug_mul_label, aug_label  
    
    # all augment feature seen as unseen class
    # pull in center of all, and push out center of domains
    def feature_augment_mix_v4(self, feature, label):
        def interpolate_push(f_x, f_c):
            alpha = random.uniform(0.1, 1.0)
            new_f_x = f_x + alpha * (f_x - f_c)
            return new_f_x
        def interpolate_pull(f_x, f_c):
            # alpha = random.uniform(0.5, 1.5)
            alpha = random.uniform(0.1, 0.5)
            new_f_x = f_x + alpha * (f_c - f_x)
            return new_f_x    
        
        
        # direct manner: directly push out the feature points from the batch centriod same class 
        center = feature.mean(dim=0, keepdim=True)
        # real class
        real_mask = (label  == 0)
        real_features = feature[real_mask]
        real_center = real_features.mean(dim=0, keepdim=True) 
        aug_fncs = [
            lambda f: interpolate_push(f, real_center),
            lambda f: interpolate_pull(f, center)
        ]
        # aug_fnc = random.choice(aug_fncs)
        real_aug_feature = torch.stack([random.choice(aug_fncs)(f) for f in real_features])
        real_batch_size = real_aug_feature.size()[0]
        real_aug_mul_label = torch.tensor([0.7,0.0,0.3]*real_batch_size).reshape(real_batch_size,-1)
        real_aug_label = torch.tensor([0.3]*real_batch_size).reshape(real_batch_size)        
        # fake class
        fake_mask = ~real_mask
        fake_features = feature[fake_mask]
        fake_center = fake_features.mean(dim=0, keepdim=True)    
        aug_fncs = [
            lambda f: interpolate_push(f, fake_center),
            lambda f: interpolate_pull(f, center)
        ]
        # aug_fnc = random.choice(aug_fncs)
        fake_aug_feature = torch.stack([random.choice(aug_fncs)(f) for f in fake_features])
        fake_batch_size = fake_aug_feature.size()[0]
        fake_aug_mul_label = torch.tensor([0.0,0.7,0.3]*fake_batch_size).reshape(fake_batch_size,-1)
        fake_aug_label = torch.tensor([0.7]*fake_batch_size).reshape(fake_batch_size)
        #Concate
        aug_feature = torch.cat((real_aug_feature, fake_aug_feature), dim=0)             
        aug_mul_label = torch.cat((real_aug_mul_label, fake_aug_mul_label), dim=0)    
        aug_label = torch.cat((real_aug_label, fake_aug_label), dim=0)  
        return aug_feature, aug_mul_label, aug_label
    
    def compute_triplet_loss(self, feature1, feature2, base_feature1, base_feature2):
        # compute each batch domain center
        real_mask1 = (self.detail_label1 == -1)
        real_mask2 = (self.detail_label2 == -1)
        fake_mask1_0 = (self.detail_label1 == 0)
        fake_mask2_0 = (self.detail_label2 == 0)
        fake_mask1_1 = (self.detail_label1 == 1)
        fake_mask2_1 = (self.detail_label2 == 1)
        mini_num = min(real_mask1.sum().item(), real_mask2.sum().item(), fake_mask1_0.sum().item(),
                       fake_mask2_0.sum().item(), fake_mask1_1.sum().item(), fake_mask2_1.sum().item()) 
        # anchor real feature
        real_feature1 = feature1[real_mask1][:mini_num]
        real_feature2 = feature2[real_mask2][:mini_num]
        
        # positive real feature
        pos_feature1 = base_feature1[:mini_num]
        pos_feature2 = base_feature2[:mini_num]
        
        # negative fake feature
        # detail_label=0 fake feature
        if np.random.rand() < 0.5:
            ngt_feature1 = feature1[fake_mask1_0][:mini_num]
        else:
            ngt_feature1 = feature1[fake_mask1_1][:mini_num]
            
        if np.random.rand() < 0.5:
            ngt_feature2 = feature2[fake_mask2_0][:mini_num]
        else:
            ngt_feature2 = feature2[fake_mask2_1][:mini_num]    
        
        
        #Dis(fake_full, fake_patch) <= Dis(fake, real)
        Anchor = torch.cat((real_feature1, real_feature2), dim=0)  
        Positive = torch.cat((pos_feature1, pos_feature2), dim=0)  
        Negative = torch.cat((ngt_feature1, ngt_feature2), dim=0)  
        loss = self.triplet(Anchor, Positive, Negative)

        return loss    
    
     
    def get_loss(self):
        return self.BCE_loss(self.output1.squeeze(1), self.label1)

    def optimize_parameters(self, tri_decay=None):
        # Train multi_classifier with feature augment
        _, feature1 = self.model(self.input1, return_feature=True)
        _, feature2 = self.model(self.input2, return_feature=True)
        aug_feature1, aug_mul_label1, _  = self.feature_augment_mix_v3(feature1, self.label1)
        aug_feature2, aug_mul_label2, _  = self.feature_augment_mix_v3(feature2, self.label2)
        
        # for multi_classifier1 (train with detail_label -1 and 0) 
        detail_mask1_1 = ~(self.detail_label1==1)
        detail_mask1_2 = ~(self.detail_label2==1)

        aug_mul_label1 = aug_mul_label1.to(self.device).float()
        aug_mul_label2 = aug_mul_label2.to(self.device).float()
        
        multi_out1_1 = self.multi_classifier_1(feature1[detail_mask1_1].detach())
        multi_out1_2 = self.multi_classifier_1(feature2[detail_mask1_2].detach())
        aug_multi_out1_1 = self.multi_classifier_1(aug_feature1.detach())
        aug_multi_out1_2 = self.multi_classifier_1(aug_feature2.detach())
        
        # self.loss_MC1 = (self.CE_loss(multi_out1_1.squeeze(1), self.multi_label1[detail_mask1_1])+\
                               # self.CE_loss(multi_out1_2.squeeze(1), self.multi_label2[detail_mask1_2])) / 2
        self.loss_MC1 = (1.0) * (((self.CE_loss(multi_out1_1.squeeze(1), self.multi_label1[detail_mask1_1])+\
                               self.CE_loss(multi_out1_2.squeeze(1), self.multi_label2[detail_mask1_2])) / 2) + \
                            0.4 * (self.CE_loss(aug_multi_out1_1.squeeze(1), aug_mul_label1)+\
                                   self.CE_loss(aug_multi_out1_2.squeeze(1), aug_mul_label2)) / 2)
        self.optimizer_MC1.zero_grad()
        self.loss_MC1.backward()
        self.optimizer_MC1.step()
        
        # for multi_classifier2 (train with detail_label -1 and 1) 
        detail_mask2_1 = ~(self.detail_label1==0)
        detail_mask2_2 = ~(self.detail_label2==0)
        
        multi_out2_1 = self.multi_classifier_2(feature1[detail_mask2_1].detach())
        multi_out2_2 = self.multi_classifier_2(feature2[detail_mask2_2].detach())
        aug_multi_out2_1 = self.multi_classifier_2(aug_feature1.detach())
        aug_multi_out2_2 = self.multi_classifier_2(aug_feature2.detach())
        
        # self.loss_MC2 = (self.CE_loss(multi_out2_1.squeeze(1), self.multi_label1[detail_mask2_1])+\
                               # self.CE_loss(multi_out2_2.squeeze(1), self.multi_label2[detail_mask2_2])) / 2
        self.loss_MC2 = (1.0) * (((self.CE_loss(multi_out2_1.squeeze(1), self.multi_label1[detail_mask2_1])+ \
                               self.CE_loss(multi_out2_2.squeeze(1), self.multi_label2[detail_mask2_2])) / 2) + \
                            0.4 * (self.CE_loss(aug_multi_out2_1.squeeze(1), aug_mul_label1)+ \
                                   self.CE_loss(aug_multi_out2_2.squeeze(1), aug_mul_label2)) / 2)
        self.optimizer_MC2.zero_grad()
        self.loss_MC2.backward()
        self.optimizer_MC2.step()
        
        
        # Train Feature extractor
        _, feature1 = self.model(self.input1, return_feature=True)
        _, feature2 = self.model(self.input2, return_feature=True)
        _, base_feature1 = self.model(self.real_img1, return_feature=True)
        _, base_feature2 = self.model(self.real_img2, return_feature=True)
        
        # triplet loss
        tri_loss = self.compute_triplet_loss(feature1, feature2, base_feature1, base_feature2)
        
        multi_out1_1 = self.multi_classifier_1(feature1[detail_mask1_1])
        multi_out1_2 = self.multi_classifier_1(feature2[detail_mask1_2])
        multi_out2_1 = self.multi_classifier_2(feature1[detail_mask2_1])
        multi_out2_2 = self.multi_classifier_2(feature2[detail_mask2_2])
        
        # self.loss_F = (1.0) * ((self.CE_loss(multi_out1_1.squeeze(1), self.multi_label1[detail_mask1_1])+ \
                               # self.CE_loss(multi_out1_2.squeeze(1), self.multi_label2[detail_mask1_2])) / 2) 
        self.loss_F = (1.0) * ((self.CE_loss(multi_out1_1.squeeze(1), self.multi_label1[detail_mask1_1])+ \
                               self.CE_loss(multi_out1_2.squeeze(1), self.multi_label2[detail_mask1_2])) / 2) + \
                      (0.4) * ((self.CE_loss(multi_out2_1.squeeze(1), self.multi_label1[detail_mask2_1])+ \
                               self.CE_loss(multi_out2_2.squeeze(1), self.multi_label2[detail_mask2_2])) / 2) + \
                      (0.8) * tri_loss
                               
        self.optimizer_F.zero_grad()
        self.loss_F.backward()
        self.optimizer_F.step()
        
        # Train binary classifier with feature augment
        _, feature1 = self.model(self.input1, return_feature=True)
        _, feature2 = self.model(self.input2, return_feature=True)
                
        aug_feature1, _, aug_label1 = self.feature_augment_mix_v3(feature1, self.label1)
        aug_feature2, _, aug_label2 = self.feature_augment_mix_v3(feature2, self.label2)
        
        aug_label1 = aug_label1.to(self.device).float()
        aug_label2 = aug_label2.to(self.device).float()
        aug_bin_out1 = self.classifier(aug_feature1.detach())
        aug_bin_out2 = self.classifier(aug_feature2.detach())
        
        loss_BC_aug = 0.4 * (self.BCE_loss(aug_bin_out1.squeeze(), aug_label1)+ \
                             self.BCE_loss(aug_bin_out2.squeeze(), aug_label2) / 2) 
        
        bin_out1_1 = self.classifier(feature1[detail_mask1_1].detach())
        bin_out1_2 = self.classifier(feature2[detail_mask1_2].detach())
        
        loss_BC_1 = 1.0 * ((self.BCE_loss(bin_out1_1.squeeze(1), self.label1[detail_mask1_1]) + \
                              self.BCE_loss(bin_out1_2.squeeze(1), self.label2[detail_mask1_2])) / 2)
                            
        bin_out2_1 = self.classifier(feature1[detail_mask2_1].detach())
        bin_out2_2 = self.classifier(feature2[detail_mask2_2].detach())
        
        loss_BC_2 = 0.4 * ((self.BCE_loss(bin_out2_1.squeeze(1), self.label1[detail_mask2_1]) + \
                            self.BCE_loss(bin_out2_2.squeeze(1), self.label2[detail_mask2_2])) / 2)            
        
        self.loss_BC = loss_BC_1 + loss_BC_2 + loss_BC_aug
        # self.loss_BC = loss_BC_1 + loss_BC_2
        # self.loss_BC = loss_BC_1 + loss_BC_aug
        # self.loss_BC = loss_BC_1
        self.optimizer_BC.zero_grad()
        self.loss_BC.backward()
        self.optimizer_BC.step()
        
        self.loss = (self.loss_BC + self.loss_MC1+self.loss_MC2 + self.loss_F) / 4
        
class mix_Distrciminator_Trainer(BaseModel):
    def name(self):
        return 'mix_Distrciminator_Trainer'

    def __init__(self, opt):
        super(mix_Distrciminator_Trainer, self).__init__(opt)

        if self.isTrain and not opt.continue_train:
            self.model = resnet50(num_classes=1)
            # torch.nn.init.normal_(self.model.fc.weight.data, 0.0, opt.init_gain)
            if opt.pretrained_path != None:
                state_dict = torch.load(opt.pretrained_path, map_location='cpu')
                self.model.load_state_dict(state_dict['model'])
            else:
                print(f"your pretrained_path is empty, please set the path.")
            
            # parallel classifier 1
            self.multi_classifier_1 = Classifier(num_classes=3)
            torch.nn.init.normal_(self.multi_classifier_1.fc1.weight.data, 0.0, opt.init_gain)
            torch.nn.init.normal_(self.multi_classifier_1.fc2.weight.data, 0.0, opt.init_gain)
            
            # parallel classifier 2
            self.multi_classifier_2 = Classifier(num_classes=3)
            torch.nn.init.normal_(self.multi_classifier_2.fc1.weight.data, 0.0, opt.init_gain)
            torch.nn.init.normal_(self.multi_classifier_2.fc2.weight.data, 0.0, opt.init_gain)
            
            # classifier 
            self.classifier = Classifier(num_classes=1)
            torch.nn.init.normal_(self.classifier.fc1.weight.data, 0.0, opt.init_gain)
            torch.nn.init.normal_(self.classifier.fc2.weight.data, 0.0, opt.init_gain)
            
            # discriminator
            self.discriminator = Discriminator_v1(2048)
                
        # Trainable parameters    
        params_F = []
    
        for name, p in self.model.named_parameters():
            if  name=="fc.weight" or name=="fc.bias": 
                p.requires_grad = False
            else:
                params_F.append(p)

        params_Multi_C1 = self.multi_classifier_1.parameters()          
        params_Multi_C2 = self.multi_classifier_2.parameters()         
        params_Binary_C = self.classifier.parameters()            
        params_D = self.discriminator.parameters()            
        
        if self.isTrain:
            self.BCE_loss = nn.BCEWithLogitsLoss()
            self.CE_loss = nn.CrossEntropyLoss()
            # initialize optimizers
            if opt.optim == 'adam':
                self.optimizer_F = torch.optim.Adam(params_F,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_MC1 = torch.optim.Adam(params_Multi_C1,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_MC2 = torch.optim.Adam(params_Multi_C2,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))                                  
                self.optimizer_BC = torch.optim.Adam(params_Binary_C,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))   
                self.optimizer_D = torch.optim.Adam(params_D,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))                                                    
            elif opt.optim == 'sgd':
                self.optimizer_F = torch.optim.SGD(params_F,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_MC1 = torch.optim.SGD(params_Multi_C1,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_MC2 = torch.optim.Adam(params_Multi_C2,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))                                   
                self.optimizer_BC = torch.optim.SGD(params_Binary_C,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_D = torch.optim.SGD(params_D,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))                                                   
            else:
                raise ValueError("optim should be [adam, sgd]")
            
        self.model.to(opt.gpu_ids[0])
        self.classifier.to(opt.gpu_ids[0])
        self.multi_classifier_1.to(opt.gpu_ids[0])
        self.multi_classifier_2.to(opt.gpu_ids[0])
        self.discriminator.to(opt.gpu_ids[0])

    def adjust_learning_rate(self, min_lr=1e-7):
        for param_group in self.optimizer_F.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        for param_group in self.optimizer_MC1.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        for param_group in self.optimizer_MC2.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False        
        for param_group in self.optimizer_BC.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False    
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False                   
        return True


    def set_input(self, batch1, batch2):
        # batch = [img, multi_label, label, detail_label, real_img]
        self.input1 = batch1[0].to(self.device)
        self.multi_label1 = batch1[1].to(self.device).float()
        self.label1 = batch1[2].to(self.device).float()
        self.detail_label1 = batch1[3].to(self.device).float()
        self.real_img1 = batch1[4].to(self.device)
        
        self.input2 = batch2[0].to(self.device)
        self.multi_label2 = batch2[1].to(self.device).float()
        self.label2 = batch2[2].to(self.device).float()
        self.detail_label2 = batch2[3].to(self.device).float()
        self.real_img2 = batch2[4].to(self.device)
        
    # all augment feature seen as unseen class
    # pull in center of all, and push out center of domains
    def feature_augment_mix(self, feature, label):
        def interpolate_push(f_x, f_c):
            alpha = random.uniform(0.6, 1.5)
            new_f_x = f_x + alpha * (f_x - f_c)
            return new_f_x
        def interpolate_pull(f_x, f_c):
            # alpha = random.uniform(0.5, 1.5)
            alpha = random.uniform(0.6, 1.5)
            new_f_x = f_x + alpha * (f_c - f_x)
            return new_f_x    
        
        
        # direct manner: directly push out the feature points from the batch centriod same class 
        center = feature.mean(dim=0, keepdim=True)
        # real class
        real_mask = (label  == 0)
        real_features = feature[real_mask]
        real_center = real_features.mean(dim=0, keepdim=True) 
        aug_fncs = [
            lambda f: interpolate_push(f, real_center),
            lambda f: interpolate_pull(f, center)
        ]
        aug_fnc = random.choice(aug_fncs)
        real_aug_feature = torch.stack([aug_fnc(f) for f in real_features])    
        # fake class
        fake_mask = ~real_mask
        fake_features = feature[fake_mask]
        fake_center = fake_features.mean(dim=0, keepdim=True)    
        aug_fncs = [
            lambda f: interpolate_push(f, fake_center),
            lambda f: interpolate_pull(f, center)
        ]
        aug_fnc = random.choice(aug_fncs)
        fake_aug_feature = torch.stack([aug_fnc(f) for f in fake_features])
        #Concate
        aug_feature = torch.cat((real_aug_feature, fake_aug_feature), dim=0)         
        batch_size = aug_feature.size()[0]
        aug_mul_label = torch.tensor([0.0,0.0,1.0]*batch_size).reshape(batch_size,-1)
        aug_label = torch.tensor([1.0]*batch_size).reshape(batch_size)
        return aug_feature, aug_mul_label, aug_label
    
    
    # all augment feature seen as real/fake class
    # real: push out center of reals / seen as real class [1.0, 0.0, 0.0]
    # fake: push out and pull in center of fakes / seen as fake class [0.0, 1.0, 0.0]
    def feature_augment_mix_v2(self, feature, label):
        def interpolate_push(f_x, f_c):
            alpha = random.uniform(0.5, 1.3)
            new_f_x = f_x + alpha * (f_x - f_c)
            return new_f_x
        def interpolate_pull(f_x, f_c):
            # alpha = random.uniform(0.5, 1.5)
            alpha = random.uniform(0.5, 1.5)
            new_f_x = f_x + alpha * (f_c - f_x)
            return new_f_x       
        
        # direct manner: directly push out the feature points from the batch centriod same class 
        # real class
        real_mask = (label  == 0)
        real_features = feature[real_mask]
        real_center = real_features.mean(dim=0, keepdim=True) 
        aug_fncs = [
            lambda f: interpolate_push(f, real_center),
        ]
        aug_fnc = random.choice(aug_fncs)
        real_aug_feature = torch.stack([aug_fnc(f) for f in real_features])
        real_batch_size = real_aug_feature.size()[0]
        real_aug_mul_label = torch.tensor([1.0,0.0,0.0]*real_batch_size).reshape(real_batch_size,-1)
        real_aug_label = torch.tensor([0.0]*real_batch_size).reshape(real_batch_size)
                
        # fake class
        fake_mask = ~real_mask
        fake_features = feature[fake_mask]
        fake_center = fake_features.mean(dim=0, keepdim=True)    
        aug_fncs = [
            lambda f: interpolate_push(f, fake_center),
            lambda f: interpolate_pull(f, fake_center)
        ]
        # aug_fnc = random.choice(aug_fncs)
        fake_aug_feature = torch.stack([random.choice(aug_fncs)(f) for f in fake_features])
        # fake_aug_feature = torch.stack([aug_fnc(f) for f in fake_features])
        fake_batch_size = fake_aug_feature.size()[0]
        fake_aug_mul_label = torch.tensor([0.0,1.0,0.0]*fake_batch_size).reshape(fake_batch_size,-1)
        fake_aug_label = torch.tensor([1.0]*fake_batch_size).reshape(fake_batch_size)
        #Concate
        aug_feature = torch.cat((real_aug_feature, fake_aug_feature), dim=0)             
        aug_mul_label = torch.cat((real_aug_mul_label, fake_aug_mul_label), dim=0)    
        aug_label = torch.cat((real_aug_label, fake_aug_label), dim=0)  
        return aug_feature, aug_mul_label, aug_label    
    

    # all augment feature seen as 
    # real: push out center of reals / seen as half real class [0.7, 0.0, 0.3]
    # fake: push out and pull in center of fakes / seen as half fake class [0.0, 0.7, 0.3]    
    def feature_augment_mix_v3(self, feature, label):
        def interpolate_push(f_x, f_c):
            alpha = random.uniform(0.5, 1.3)
            new_f_x = f_x + alpha * (f_x - f_c)
            return new_f_x
        def interpolate_pull(f_x, f_c):
            # alpha = random.uniform(0.5, 1.5)
            alpha = random.uniform(0.5, 1.3)
            new_f_x = f_x + alpha * (f_c - f_x)
            return new_f_x       
        
        # direct manner: directly push out the feature points from the batch centriod same class 
        # real class
        real_mask = (label  == 0)
        real_features = feature[real_mask]
        real_center = real_features.mean(dim=0, keepdim=True) 
        aug_fncs = [
            lambda f: interpolate_push(f, real_center),
        ]
        aug_fnc = random.choice(aug_fncs)
        real_aug_feature = torch.stack([aug_fnc(f) for f in real_features])
        real_batch_size = real_aug_feature.size()[0]
        real_aug_mul_label = torch.tensor([0.7,0.0,0.3]*real_batch_size).reshape(real_batch_size,-1)
        real_aug_label = torch.tensor([0.3]*real_batch_size).reshape(real_batch_size)
                
        # fake class
        fake_mask = ~real_mask
        fake_features = feature[fake_mask]
        fake_center = fake_features.mean(dim=0, keepdim=True)    
        aug_fncs = [
            lambda f: interpolate_push(f, fake_center),
            lambda f: interpolate_pull(f, fake_center)
        ]
        # aug_fnc = random.choice(aug_fncs)
        fake_aug_feature = torch.stack([random.choice(aug_fncs)(f) for f in fake_features])
        # fake_aug_feature = torch.stack([aug_fnc(f) for f in fake_features])
        fake_batch_size = fake_aug_feature.size()[0]
        fake_aug_mul_label = torch.tensor([0.0,0.7,0.3]*fake_batch_size).reshape(fake_batch_size,-1)
        fake_aug_label = torch.tensor([0.7]*fake_batch_size).reshape(fake_batch_size)
        #Concate
        aug_feature = torch.cat((real_aug_feature, fake_aug_feature), dim=0)             
        aug_mul_label = torch.cat((real_aug_mul_label, fake_aug_mul_label), dim=0)    
        aug_label = torch.cat((real_aug_label, fake_aug_label), dim=0)  
        return aug_feature, aug_mul_label, aug_label  
    
    # all augment feature seen as unseen class
    # pull in center of all, and push out center of domains
    def feature_augment_mix_v4(self, feature, label):
        def interpolate_push(f_x, f_c):
            alpha = random.uniform(0.1, 1.0)
            new_f_x = f_x + alpha * (f_x - f_c)
            return new_f_x
        def interpolate_pull(f_x, f_c):
            # alpha = random.uniform(0.5, 1.5)
            alpha = random.uniform(0.1, 0.5)
            new_f_x = f_x + alpha * (f_c - f_x)
            return new_f_x    
        
        
        # direct manner: directly push out the feature points from the batch centriod same class 
        center = feature.mean(dim=0, keepdim=True)
        # real class
        real_mask = (label  == 0)
        real_features = feature[real_mask]
        real_center = real_features.mean(dim=0, keepdim=True) 
        aug_fncs = [
            lambda f: interpolate_push(f, real_center),
            lambda f: interpolate_pull(f, center)
        ]
        # aug_fnc = random.choice(aug_fncs)
        real_aug_feature = torch.stack([random.choice(aug_fncs)(f) for f in real_features])
        real_batch_size = real_aug_feature.size()[0]
        real_aug_mul_label = torch.tensor([0.7,0.0,0.3]*real_batch_size).reshape(real_batch_size,-1)
        real_aug_label = torch.tensor([0.3]*real_batch_size).reshape(real_batch_size)        
        # fake class
        fake_mask = ~real_mask
        fake_features = feature[fake_mask]
        fake_center = fake_features.mean(dim=0, keepdim=True)    
        aug_fncs = [
            lambda f: interpolate_push(f, fake_center),
            lambda f: interpolate_pull(f, center)
        ]
        # aug_fnc = random.choice(aug_fncs)
        fake_aug_feature = torch.stack([random.choice(aug_fncs)(f) for f in fake_features])
        fake_batch_size = fake_aug_feature.size()[0]
        fake_aug_mul_label = torch.tensor([0.0,0.7,0.3]*fake_batch_size).reshape(fake_batch_size,-1)
        fake_aug_label = torch.tensor([0.7]*fake_batch_size).reshape(fake_batch_size)
        #Concate
        aug_feature = torch.cat((real_aug_feature, fake_aug_feature), dim=0)             
        aug_mul_label = torch.cat((real_aug_mul_label, fake_aug_mul_label), dim=0)    
        aug_label = torch.cat((real_aug_label, fake_aug_label), dim=0)  
        return aug_feature, aug_mul_label, aug_label
    
    
    def get_loss(self):
        return self.BCE_loss(self.output1.squeeze(1), self.label1)

    def optimize_parameters(self, tri_decay=None):
        # Train multi_classifier with feature augment
        _, feature1 = self.model(self.input1, return_feature=True)
        _, feature2 = self.model(self.input2, return_feature=True)
        aug_feature1, aug_mul_label1, _  = self.feature_augment_mix_v3(feature1, self.label1)
        aug_feature2, aug_mul_label2, _  = self.feature_augment_mix_v3(feature2, self.label2)
        
        # for multi_classifier1 (train with detail_label -1 and 0) 
        detail_mask1_1 = ~(self.detail_label1==1)
        detail_mask1_2 = ~(self.detail_label2==1)

        aug_mul_label1 = aug_mul_label1.to(self.device).float()
        aug_mul_label2 = aug_mul_label2.to(self.device).float()
        
        multi_out1_1 = self.multi_classifier_1(feature1[detail_mask1_1].detach())
        multi_out1_2 = self.multi_classifier_1(feature2[detail_mask1_2].detach())
        aug_multi_out1_1 = self.multi_classifier_1(aug_feature1.detach())
        aug_multi_out1_2 = self.multi_classifier_1(aug_feature2.detach())
        
        # self.loss_MC1 = (self.CE_loss(multi_out1_1.squeeze(1), self.multi_label1[detail_mask1_1])+\
                               # self.CE_loss(multi_out1_2.squeeze(1), self.multi_label2[detail_mask1_2])) / 2
        self.loss_MC1 = (1.0) * (((self.CE_loss(multi_out1_1.squeeze(1), self.multi_label1[detail_mask1_1])+\
                               self.CE_loss(multi_out1_2.squeeze(1), self.multi_label2[detail_mask1_2])) / 2) + \
                            0.4 * (self.CE_loss(aug_multi_out1_1.squeeze(1), aug_mul_label1)+\
                                   self.CE_loss(aug_multi_out1_2.squeeze(1), aug_mul_label2)) / 2)
        self.optimizer_MC1.zero_grad()
        self.loss_MC1.backward()
        self.optimizer_MC1.step()
        
        # for multi_classifier2 (train with detail_label -1 and 1) 
        detail_mask2_1 = ~(self.detail_label1==0)
        detail_mask2_2 = ~(self.detail_label2==0)
        
        multi_out2_1 = self.multi_classifier_2(feature1[detail_mask2_1].detach())
        multi_out2_2 = self.multi_classifier_2(feature2[detail_mask2_2].detach())
        aug_multi_out2_1 = self.multi_classifier_2(aug_feature1.detach())
        aug_multi_out2_2 = self.multi_classifier_2(aug_feature2.detach())
        
        # self.loss_MC2 = (self.CE_loss(multi_out2_1.squeeze(1), self.multi_label1[detail_mask2_1])+\
                               # self.CE_loss(multi_out2_2.squeeze(1), self.multi_label2[detail_mask2_2])) / 2
        self.loss_MC2 = (1.0) * (((self.CE_loss(multi_out2_1.squeeze(1), self.multi_label1[detail_mask2_1])+ \
                               self.CE_loss(multi_out2_2.squeeze(1), self.multi_label2[detail_mask2_2])) / 2) + \
                            0.4 * (self.CE_loss(aug_multi_out2_1.squeeze(1), aug_mul_label1)+ \
                                   self.CE_loss(aug_multi_out2_2.squeeze(1), aug_mul_label2)) / 2)
        self.optimizer_MC2.zero_grad()
        self.loss_MC2.backward()
        self.optimizer_MC2.step()
        
        # Train Discriminator
        # try to distinguish the extracted feature of real full and real patch
        _, real_feature1 = self.model(self.real_img1, return_feature=True) # real full
        _, real_feature2 = self.model(self.real_img2, return_feature=True) # real patch
        _, feature1 = self.model(self.input1, return_feature=True) 
        _, feature2 = self.model(self.input2, return_feature=True)
        
        # real mask for feature1 and feature2
        real_mask1 = self.label1 == 0
        real_mask2 = self.label2 == 0
        
        dis_diff = self.discriminator(real_feature1.detach(), real_feature2.detach())
        dis_same1 = self.discriminator(feature1[real_mask1].detach(), real_feature1[real_mask1].detach())
        dis_same2 = self.discriminator(feature2[real_mask2].detach(), real_feature2[real_mask2].detach())

        ones1 = torch.ones(dis_same1.size()[0], 1, requires_grad=False).to(self.device) 
        ones2 = torch.ones(dis_same2.size()[0], 1, requires_grad=False).to(self.device) 
        zeros = torch.zeros(dis_diff.size()[0], 1, requires_grad=False).to(self.device)

        self.loss_D = ((self.BCE_loss(dis_same1, ones1)+self.BCE_loss(dis_same2, ones2)) * \
                                (dis_diff.size()[0]/(dis_same1.size()[0]+dis_same2.size()[0])) + \
                        self.BCE_loss(dis_diff, zeros)) / 2
        
        self.optimizer_D.zero_grad()
        self.loss_D.backward()
        self.optimizer_D.step()
        
        # Train Feature extractor
        
        multi_out1_1 = self.multi_classifier_1(feature1[detail_mask1_1])
        multi_out1_2 = self.multi_classifier_1(feature2[detail_mask1_2])
        multi_out2_1 = self.multi_classifier_2(feature1[detail_mask2_1])
        multi_out2_2 = self.multi_classifier_2(feature2[detail_mask2_2])
        
        dis_same = self.discriminator(real_feature1, real_feature2)
        # dis_diff = self.discriminator(feature1[real_mask1], real_feature1[real_mask1]) + \
                   # self.discriminator(feature2[real_mask2], real_feature2[real_mask2])
        
        ones = torch.ones(dis_same.size()[0], 1, requires_grad=False).to(self.device) 
        zeros = torch.zeros(dis_diff.size()[0], 1, requires_grad=False).to(self.device)

        # self.loss_D = (self.BCE_loss(dis_same, ones) + self.BCE_loss(dis_diff, zeros)) / 2
        discriminator_loss = self.BCE_loss(dis_same, ones)
        
        # self.loss_F = (1.0) * ((self.CE_loss(multi_out1_1.squeeze(1), self.multi_label1[detail_mask1_1])+ \
                               # self.CE_loss(multi_out1_2.squeeze(1), self.multi_label2[detail_mask1_2])) / 2) 
        self.loss_F = (1.0) * ((self.CE_loss(multi_out1_1.squeeze(1), self.multi_label1[detail_mask1_1])+ \
                               self.CE_loss(multi_out1_2.squeeze(1), self.multi_label2[detail_mask1_2])) / 2) + \
                      (0.4) * ((self.CE_loss(multi_out2_1.squeeze(1), self.multi_label1[detail_mask2_1])+ \
                               self.CE_loss(multi_out2_2.squeeze(1), self.multi_label2[detail_mask2_2])) / 2) + \
                      (0.8) * discriminator_loss             
                               
        self.optimizer_F.zero_grad()
        self.loss_F.backward()
        self.optimizer_F.step()
        
        # Train binary classifier with feature augment
        _, feature1 = self.model(self.input1, return_feature=True)
        _, feature2 = self.model(self.input2, return_feature=True)
                
        aug_feature1, _, aug_label1 = self.feature_augment_mix_v3(feature1, self.label1)
        aug_feature2, _, aug_label2 = self.feature_augment_mix_v3(feature2, self.label2)
        
        aug_label1 = aug_label1.to(self.device).float()
        aug_label2 = aug_label2.to(self.device).float()
        aug_bin_out1 = self.classifier(aug_feature1.detach())
        aug_bin_out2 = self.classifier(aug_feature2.detach())
        
        loss_BC_aug = 0.4 * (self.BCE_loss(aug_bin_out1.squeeze(), aug_label1)+ \
                             self.BCE_loss(aug_bin_out2.squeeze(), aug_label2) / 2) 
        
        bin_out1_1 = self.classifier(feature1[detail_mask1_1].detach())
        bin_out1_2 = self.classifier(feature2[detail_mask1_2].detach())
        
        loss_BC_1 = 1.0 * ((self.BCE_loss(bin_out1_1.squeeze(1), self.label1[detail_mask1_1]) + \
                              self.BCE_loss(bin_out1_2.squeeze(1), self.label2[detail_mask1_2])) / 2)
                            
        bin_out2_1 = self.classifier(feature1[detail_mask2_1].detach())
        bin_out2_2 = self.classifier(feature2[detail_mask2_2].detach())
        
        loss_BC_2 = 0.4 * ((self.BCE_loss(bin_out2_1.squeeze(1), self.label1[detail_mask2_1]) + \
                            self.BCE_loss(bin_out2_2.squeeze(1), self.label2[detail_mask2_2])) / 2)            
        
        self.loss_BC = loss_BC_1 + loss_BC_2 + loss_BC_aug
        # self.loss_BC = loss_BC_1 + loss_BC_2
        # self.loss_BC = loss_BC_1 + loss_BC_aug
        # self.loss_BC = loss_BC_1
        self.optimizer_BC.zero_grad()
        self.loss_BC.backward()
        self.optimizer_BC.step()
        
        self.loss = (self.loss_BC + self.loss_MC1+self.loss_MC2 + self.loss_F) / 4

