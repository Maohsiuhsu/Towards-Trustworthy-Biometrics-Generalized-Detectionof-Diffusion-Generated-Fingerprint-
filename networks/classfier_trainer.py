import functools
import torch
import torch.nn as nn
from networks.classifier import Multi_Binary_Classifier, Classifier, FuseClassifier
from networks.resnet import resnet50
from networks.discriminator import Discriminator_v1
from networks.base_model import BaseModel, init_weights
from .loss import ContrastiveLoss, TripletLoss_v2
import random
import numpy as np

class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)

        if self.isTrain and not opt.continue_train:
            self.model = resnet50(num_classes=1)
            if opt.pretrained_path != None:
                state_dict = torch.load(opt.pretrained_path, map_location='cpu')
                self.model.load_state_dict(state_dict['model'])
            else:
                raise(f"yout pretrained_path is empty, please set the path.")
            
            # label 0:[1,0,0] is real, label 1:[0,1,0] is fake, label 2:[0,0,1] is unseen
            self.classifier = Multi_Binary_Classifier(num_classes=3)
            torch.nn.init.normal_(self.classifier.fc1.weight.data, 0.0, opt.init_gain)
            torch.nn.init.normal_(self.classifier.fc2.weight.data, 0.0, opt.init_gain)
        
        # freeze the feature extractor and just train classifier
        # print('Freeze the feature extractor and just train classifier')
        params = []
        # for name, p in self.model.named_parameters():
            # if  name=="fc.weight" or name=="fc.bias": 
                # p.requires_grad = False 
            # else:
                # params.append(p)
        
        for name, p in self.model.named_parameters():
            p.requires_grad = False 

        
        for name, p in self.classifier.named_parameters():
            params.append(p) 

        # params = self.classifier.parameters()           

        if self.isTrain:
            self.BCE_loss = nn.BCEWithLogitsLoss()
            self.CE_loss = nn.CrossEntropyLoss()
            # initialize optimizers
            if opt.optim == 'adam':
                self.optimizer = torch.optim.Adam(params,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
            elif opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD(params,
                                                 lr=opt.lr, momentum=0.0, weight_decay=0)
            else:
                raise ValueError("optim should be [adam, sgd]")

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.epoch)
            
        self.model.to(opt.gpu_ids[0])
        self.classifier.to(opt.gpu_ids[0])

    def compute_centriod(self,):
        pass
    def adjust_learning_rate(self, min_lr=5e-7):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 8.
            if param_group['lr'] < min_lr:
                return False
        return True

    def set_input(self, input):
        # img, multi_label, label
        self.input = input[0].to(self.device)
        self.multi_label = input[1].to(self.device).float()
        self.label = input[2].to(self.device).float()
    
    def forward(self):
        _, self.feature = self.model(self.input, True)
        # print(type(self.feature))
        self.multi_out, self.bin_out = self.classifier(self.feature.detach())
        # self.multi_out, self.bin_out = self.classifier(self.feature)
        
        # feature augment
        # aug_feature, aug_multi_label, aug_label = self.feature_augment_mix()
        # print(self.device)
        # self.aug_feature = aug_feature.to(self.device)
        # self.aug_multi_label = aug_multi_label.to(self.device).float()
        # self.aug_label = aug_label.to(self.device).float()
        # self.aug_multi_out, self.aug_bin_out = self.classifier(self.aug_feature)

   
    
    def feature_augment_center(self,):
        def interpolate_pull(f_x, f_c):
            alpha = random.uniform(0.5, 1.5)
            new_f_x = f_x + alpha * (f_c - f_x)
            return new_f_x
        
        # direct manner: directly push out the feature points from the batch centriod same class 
        center = self.feature.mean(dim=0, keepdim=True)
        aug_fnc = lambda f: interpolate_pull(f, center)
        aug_feature = torch.stack([aug_fnc(f) for f in self.feature])
        # Concate
        # aug_feature = torch.cat((real_aug, fake_aug), dim=0)
        # aug_feature = real_aug      
        batch_size = aug_feature.size()[0]
        aug_mul_label = torch.tensor([0.0,0.0,1.0]*batch_size).reshape(batch_size,-1)
        aug_label = torch.tensor([1.0]*batch_size).reshape(batch_size)
        return aug_feature, aug_mul_label, aug_label
    

    def feature_augment(self):
        def interpolate_push(f_x, f_c):
            alpha = random.uniform(0.5, 1.5)
            new_f_x = f_x + alpha * (f_x - f_c)
            return new_f_x
        
        # direct manner: directly push out the feature points from the batch centriod same class 
      
        # real class
        real_mask = (self.label  == 1)
        real_features = self.feature[real_mask]
        real_center = real_features.mean(dim=0, keepdim=True)
        aug_fnc = lambda f: interpolate_push(f, real_center)
        real_aug = torch.stack([aug_fnc(f) for f in real_features])
        # fake class  
        fake_mask = ~real_mask
        fake_features = self.feature[fake_mask]
        fake_center = fake_features.mean(dim=0, keepdim=True)
        aug_fnc = lambda f: interpolate_push(f, fake_center)
        fake_aug = torch.stack([aug_fnc(f) for f in fake_features])
        # Concate
        aug_feature = torch.cat((real_aug, fake_aug), dim=0)   
        batch_size = aug_feature.size()[0]
        aug_mul_label = torch.tensor([0.0,0.0,1.0]*batch_size).reshape(batch_size,-1)
        aug_label = torch.tensor([1.0]*batch_size).reshape(batch_size)
        return aug_feature, aug_mul_label, aug_label
    
    def feature_augment_mix(self):
        def interpolate_push(f_x, f_c):
            alpha = random.uniform(0.5, 1.5)
            new_f_x = f_x + alpha * (f_x - f_c)
            return new_f_x
        def interpolate_pull(f_x, f_c):
            alpha = random.uniform(0.5, 1.5)
            new_f_x = f_x + alpha * (f_c - f_x)
            return new_f_x    
        
        
        # direct manner: directly push out the feature points from the batch centriod same class 
        center = self.feature.mean(dim=0, keepdim=True)
        # real class
        real_mask = (self.label  == 1)
        real_features = self.feature[real_mask]
        real_center = real_features.mean(dim=0, keepdim=True) 
        aug_fncs = [
            lambda f: interpolate_push(f, real_center),
            lambda f: interpolate_pull(f, center)
        ]
        aug_fnc = random.choice(aug_fncs)
        real_aug_feature = torch.stack([aug_fnc(f) for f in real_features])    
        # fake class
        fake_mask = ~real_mask
        fake_features = self.feature[fake_mask]
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
    
        
    def get_loss(self):
        CE_loss = 0.3 * self.CE_loss(self.multi_out.squeeze(1), self.multi_label)
        BCE_loss = self.BCE_loss(self.bin_out.squeeze(1), self.label)
        return  CE_loss + BCE_loss

    def optimize_parameters(self):
        self.forward()        
        # CE_loss = 1.0 * (self.CE_loss(self.multi_out.squeeze(1), self.multi_label)+ \
                         # 0.3 * self.CE_loss(self.aug_multi_out.squeeze(1), self.aug_multi_label))
        CE_loss = 1.0 * (self.CE_loss(self.multi_out.squeeze(1), self.multi_label))                
        # BCE_loss = (self.BCE_loss(self.bin_out.squeeze(1), self.label) + \
                    # 0.8 * self.BCE_loss(self.aug_bin_out.squeeze(1), self.aug_label.unsqueeze(1)))
        # self.loss = CE_loss + BCE_loss
        self.loss = CE_loss
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()


class Contrastive_Trainer(BaseModel):
    def name(self):
        return 'Contrastive_Trainer'

    def __init__(self, opt):
        super(Contrastive_Trainer, self).__init__(opt)

        if self.isTrain and not opt.continue_train:
            self.model = resnet50(pretrained=True)
            self.model.fc = nn.Linear(2048, 1)
            torch.nn.init.normal_(self.model.fc.weight.data, 0.0, opt.init_gain)
        
        if opt.fix_backbone:
            params = []
            for name, p in self.model.named_parameters():
                if  name=="fc.weight" or name=="fc.bias": 
                    params.append(p) 
                else:
                    p.requires_grad = False
        else:
            print("Your backbone is not fixed.")
            import time 
            time.sleep(3)
            params = self.model.parameters()            
        
        
        if not self.isTrain or opt.continue_train:
            self.model = resnet50(num_classes=1)

        if self.isTrain:
            self.BCE_loss = nn.BCEWithLogitsLoss()
            self.Contrastive_loss = ContrastiveLoss(margin=2.0)
            # initialize optimizers
            if opt.optim == 'adam':
                self.optimizer = torch.optim.Adam(params,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
            elif opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD(params,
                                                 lr=opt.lr, momentum=0.0, weight_decay=0)
            else:
                raise ValueError("optim should be [adam, sgd]")

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.epoch)
        self.model.to(opt.gpu_ids[0])


    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        return True

    def set_input(self, input):
        self.input = input[0].to(self.device)
        self.label = input[1].to(self.device).float()
        self.input1 = input[2].to(self.device)
        self.pair_label = input[3].to(self.device).float()
    
    def forward(self):
        self.output, self.feature = self.model(self.input, True)
        self.output1, self.feature1 = self.model(self.input1, True)

    def get_loss(self):
        return self.BCE_loss(self.output.squeeze(1), self.label)

    def optimize_parameters(self):
        self.forward()
        BCE = self.BCE_loss(self.output.squeeze(1), self.label)
        CTT = self.Contrastive_loss(self.feature.squeeze(1), self.feature1.squeeze(1), self.pair_label) 
        self.loss = BCE + 0.4*CTT
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()


class Triplet_Trainer(BaseModel):
    def name(self):
        return 'Triplet_Trainer'

    def __init__(self, opt):
        super(Triplet_Trainer, self).__init__(opt)

        if self.isTrain and not opt.continue_train:
            self.model = resnet50(pretrained=True)
            self.model.fc = nn.Linear(2048, 1)
            torch.nn.init.normal_(self.model.fc.weight.data, 0.0, opt.init_gain)
            if opt.pretrained_path != None:
                state_dict = torch.load(opt.pretrained_path, map_location='cpu')
                self.model.load_state_dict(state_dict['model'])
        if opt.fix_backbone:
            params = []
            for name, p in self.model.named_parameters():
                if  name=="fc.weight" or name=="fc.bias": 
                    params.append(p) 
                else:
                    p.requires_grad = False
        else:
            print("Your backbone is not fixed.")
            import time 
            time.sleep(3)
            params = self.model.parameters()            
        
        
        if not self.isTrain or opt.continue_train:
            self.model = resnet50(num_classes=1)

        if self.isTrain:
            self.BCE_loss = nn.BCEWithLogitsLoss()
            # self.triplet = nn.TripletMarginLoss(margin=2.0, p=2, eps=1e-7)
            self.triplet = TripletLoss_v2(margin=2.0)
            self.triplet.to('cuda')

            # initialize optimizers
            if opt.optim == 'adam':
                self.optimizer = torch.optim.Adam(params,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
            elif opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD(params,
                                                 lr=opt.lr, momentum=0.0, weight_decay=0)
            else:
                raise ValueError("optim should be [adam, sgd]")

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.epoch)
        self.model.to(opt.gpu_ids[0])


    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        return True

    def set_input(self, input):
        # anchor, label, positive, negative
        self.anchor = input[0].to(self.device)
        self.label = input[1].to(self.device).float()
        self.positive = input[2].to(self.device)
        self.negative = input[3].to(self.device)
    
    def forward(self):
        self.out_a, self.feature_a = self.model(self.anchor, True)
        self.out_p, self.feature_p = self.model(self.positive, True)
        self.out_n, self.feature_n = self.model(self.negative, True)


    def get_loss(self):
        return self.BCE_loss(self.feature_a.squeeze(1), self.label)

    def optimize_parameters(self):
        self.forward()
        BCE = self.BCE_loss(self.out_a.squeeze(1), self.label)
        Tri = self.triplet(self.feature_a.squeeze(1), self.feature_p.squeeze(1), self.feature_n.squeeze(1))
        self.loss = BCE + 0.4*Tri
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()



class Discriminator_Trainer(BaseModel):
    def name(self):
        return 'Discriminator_Trainer'

    def __init__(self, opt):
        super(Discriminator_Trainer, self).__init__(opt)
        if self.isTrain and not opt.continue_train:
            self.model = resnet50(pretrained=True)
            self.model.fc = nn.Linear(2048, 1)
            torch.nn.init.normal_(self.model.fc.weight.data, 0.0, opt.init_gain)
            if opt.pretrained_path != None:
                state_dict = torch.load(opt.pretrained_path, map_location='cpu')
                self.model.load_state_dict(state_dict['model'])
            self.discriminator = Discriminator_v1(2048)  
            
        if opt.fix_backbone:
            params = []
            for name, p in self.model.named_parameters():
                if  name=="fc.weight" or name=="fc.bias": 
                    params.append(p) 
                else:
                    p.requires_grad = False
        else:
            print("Your backbone is not fixed.")
            import time 
            time.sleep(3)
            params = self.model.parameters()            
        
        
        if not self.isTrain or opt.continue_train:
            self.model = resnet50(num_classes=1)

        if self.isTrain:
            self.BCE_Logit = nn.BCEWithLogitsLoss()
            self.BCE_loss = nn.BCELoss()
            # self.triplet = nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)

            # initialize optimizers
            if opt.optim == 'adam':
                self.optimizer = torch.optim.Adam(params,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(),
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
            elif opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD(params,
                                                 lr=opt.lr, momentum=0.0, weight_decay=0)
                self.optimizer_D = torch.optim.SGD(self.discriminator.parameters(),
                                                 lr=opt.lr, momentum=0.0, weight_decay=0)                                 
            else:
                raise ValueError("optim should be [adam, sgd]")

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.epoch)
        self.model.to(opt.gpu_ids[0])
        self.discriminator.to(opt.gpu_ids[0])


    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        return True

    def set_input(self, input):
        # anchor, label, positive, negative
        self.anchor = input[0].to(self.device)
        self.label = input[1].to(self.device).float()
        self.positive = input[2].to(self.device)
        self.negative = input[3].to(self.device)

    def optimize_parameters(self):
        # Adversarial ground truths
        simi = torch.ones(self.anchor.size()[0], 1, requires_grad=False) ## postive (similiar)
        disimi = torch.zeros(self.anchor.size()[0], 1, requires_grad=False) ##negative (dissimiliar)
        
        # -----------------
        #  Train ResNet-50 classifier
        # -----------------
        out_a, feature_a = self.model(self.anchor, True)
        out_p, feature_p = self.model(self.positive, True)
        out_n, feature_n = self.model(self.negative, True)
        # Move disimi tensor to the same device as out_a tensor
        simi = simi.to(out_a.device)
        disimi = disimi.to(out_a.device)
        
        self.optimizer.zero_grad()
        out_loss = self.BCE_Logit(out_a.squeeze(1), self.label)
        # print(self.discriminator(feature_a, feature_p).size())
        AP = self.discriminator(feature_a, feature_p)
        dis_p = self.BCE_loss(AP, disimi)
        AN = self.discriminator(feature_a, feature_n)
        dis_n = self.BCE_loss(AN,simi)
        dis_loss = (dis_p+dis_n)/2
        self.loss = out_loss + 0.15*dis_loss
        self.loss.backward()
        self.optimizer.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        self.optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        postive_loss = self.BCE_loss(self.discriminator(feature_a.detach(), feature_p.detach()), simi)
        negative_loss = self.BCE_loss(self.discriminator(feature_a.detach(), feature_n.detach()), disimi)
        self.d_loss = (postive_loss + negative_loss) / 2

        self.d_loss.backward()
        self.optimizer_D.step()


class mix_Trainer(BaseModel):
    def name(self):
        return 'mix_Trainer'

    def __init__(self, opt):
        super(mix_Trainer, self).__init__(opt)

        if self.isTrain and not opt.continue_train:
            self.model = resnet50(pretrained=True)
            # self.model = resnet50(pretrained=False, num_classes=1)
            self.model.fc = nn.Linear(2048, 1)
            torch.nn.init.normal_(self.model.fc.weight.data, 0.0, opt.init_gain)
            if opt.pretrained_path != None:
                state_dict = torch.load(opt.pretrained_path, map_location='cpu')
                # print(state_dict.keys())
                self.model.load_state_dict(state_dict)
        if opt.fix_backbone:
            params = []
            for name, p in self.model.named_parameters():
                if  name=="fc.weight" or name=="fc.bias": 
                    params.append(p) 
                else:
                    p.requires_grad = False
        else:
            print("Your backbone is not fixed.")
            import time 
            time.sleep(3)
            params = self.model.parameters()            
        
        
        if not self.isTrain or opt.continue_train:
            self.model = resnet50(num_classes=1)

        if self.isTrain:
            self.BCE_loss = nn.BCEWithLogitsLoss()
            # initialize optimizers
            if opt.optim == 'adam':
                self.optimizer = torch.optim.Adam(params,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
            elif opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD(params,
                                                 lr=opt.lr, momentum=0.0, weight_decay=0)
            else:
                raise ValueError("optim should be [adam, sgd]")

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.epoch)
        self.model.to(opt.gpu_ids[0])


    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        return True

    def set_input(self, batch1, batch2):
        self.input1 = batch1[0].to(self.device)
        self.label1 = batch1[1].to(self.device).float()
        self.input2 = batch2[0].to(self.device)
        self.label2 = batch2[1].to(self.device).float()
    
    def forward(self):
        self.output1 = self.model(self.input1)
        self.output2 = self.model(self.input2)

    def get_loss(self):
        return self.BCE_loss(self.output1.squeeze(1), self.label1)

    def optimize_parameters(self):
        self.forward()
        self.loss = (self.BCE_loss(self.output1.squeeze(1), self.label1) + self.BCE_loss(self.output2.squeeze(1), self.label2)) / 2
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
 


class mix_Contrastive_Trainer(BaseModel):
    def name(self):
        return 'mix_Contrastive_Trainer'

    def __init__(self, opt):
        super(mix_Contrastive_Trainer, self).__init__(opt)

        if self.isTrain and not opt.continue_train:
            self.model = resnet50(pretrained=True)
            self.model.fc = nn.Linear(2048, 1)
            torch.nn.init.normal_(self.model.fc.weight.data, 0.0, opt.init_gain)
            if opt.pretrained_path != None:
                state_dict = torch.load(opt.pretrained_path, map_location='cpu')
                self.model.load_state_dict(state_dict['model'])
                print(f'loading the weight {opt.pretrained_path} to fintinuing')
        if opt.fix_backbone:
            params = []
            for name, p in self.model.named_parameters():
                if  name=="fc.weight" or name=="fc.bias": 
                    params.append(p) 
                else:
                    p.requires_grad = False
        else:
            print("Your backbone is not fixed.")
            import time 
            time.sleep(3)
            params = self.model.parameters()            
        
        
        if not self.isTrain or opt.continue_train:
            self.model = resnet50(num_classes=1)

        if self.isTrain:
            self.BCE_loss = nn.BCEWithLogitsLoss()
            self.Contrastive_loss = ContrastiveLoss(margin=2.0)
            # initialize optimizers
            if opt.optim == 'adam':
                self.optimizer = torch.optim.Adam(params,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
            elif opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD(params,
                                                 lr=opt.lr, momentum=0.0, weight_decay=0)
            else:
                raise ValueError("optim should be [adam, sgd]")

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.epoch)
        self.model.to(opt.gpu_ids[0])


    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        return True

    def set_input(self, batch1, batch2):
        # batch = [anchor, label, positive, negative]
        self.input1_1 = batch1[0].to(self.device)
        self.label1 = batch1[1].to(self.device).float()
        self.input1_2 = batch1[2].to(self.device)
        self.pair_label1 = batch1[3].to(self.device).float()
        
        self.input2_1 = batch2[0].to(self.device)
        self.label2 = batch2[1].to(self.device).float()
        self.input2_2 = batch2[2].to(self.device)
        self.pair_label2 = batch2[3].to(self.device).float()
        
            
    
    def forward(self):
        self.out1_1, self.feature1_1 = self.model(self.input1_1, True)
        self.out1_2, self.feature1_2 = self.model(self.input1_2, True)
        self.out2_1, self.feature2_1 = self.model(self.input2_1, True)
        self.out2_2, self.feature2_2 = self.model(self.input2_2, True)


    def optimize_parameters(self):
        self.forward()
        BCE = self.BCE_loss(self.out1_1.squeeze(1), self.label1) + self.BCE_loss(self.out2_1.squeeze(1), self.label2)
        CTT1 = self.Contrastive_loss(self.feature1_1.squeeze(1), self.feature1_2.squeeze(1), self.pair_label1)
        CTT2 = self.Contrastive_loss(self.feature2_1.squeeze(1), self.feature2_2.squeeze(1), self.pair_label2)         
        CTT = (CTT1 + CTT2) / 2
        self.loss = BCE + 0.4*CTT
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()




class mix_Triplet_Trainer(BaseModel):
    def name(self):
        return 'mix_Triplet_Trainer'

    def __init__(self, opt):
        super(mix_Triplet_Trainer, self).__init__(opt)

        if self.isTrain and not opt.continue_train:
            self.model = resnet50(pretrained=True)
            self.model.fc = nn.Linear(2048, 1)
            torch.nn.init.normal_(self.model.fc.weight.data, 0.0, opt.init_gain)
            if opt.pretrained_path != None:
                state_dict = torch.load(opt.pretrained_path, map_location='cpu')
                self.model.load_state_dict(state_dict['model'])
                print(f'loading the weight {opt.pretrained_path} to fintinuing')
        if opt.fix_backbone:
            params = []
            for name, p in self.model.named_parameters():
                if  name=="fc.weight" or name=="fc.bias": 
                    params.append(p) 
                else:
                    p.requires_grad = False
        else:
            print("Your backbone is not fixed.")
            import time 
            time.sleep(3)
            params = self.model.parameters()            
        
        
        if not self.isTrain or opt.continue_train:
            self.model = resnet50(num_classes=1)

        if self.isTrain:
            self.BCE_loss = nn.BCEWithLogitsLoss()
            self.triplet = nn.TripletMarginLoss(margin=2.0, p=2, eps=1e-7)
            # self.triplet = TripletLoss_v2(margin=1.0)
            # self.triplet.to(opt.gpu_ids[0])
            # initialize optimizers
            if opt.optim == 'adam':
                self.optimizer = torch.optim.Adam(params,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
            elif opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD(params,
                                                 lr=opt.lr, momentum=0.0, weight_decay=0)
            else:
                raise ValueError("optim should be [adam, sgd]")

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.epoch)
        self.model.to(opt.gpu_ids[0])


    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        return True

    def set_input(self, batch1, batch2):
        # batch = [anchor, label, positive, negative]
        self.input1 = batch1[0].to(self.device)
        self.label1 = batch1[1].to(self.device).float()
        self.positive1 = batch1[2].to(self.device)
        self.negative1 = batch1[3].to(self.device)
        self.input2 = batch2[0].to(self.device)
        self.label2 = batch2[1].to(self.device).float()
        self.positive2 = batch2[2].to(self.device)
        self.negative2 = batch2[3].to(self.device)
        
        if self.label1[0] == self.label2[0]:
            self.same_label = True
        else:
            self.same_label = False  
            
    
    def forward(self):
        self.a_out1, self.a_feat1 = self.model(self.input1, True)
        self.a_out2, self.a_feat2 = self.model(self.input2, True)
        if self.same_label:
            self.p_out1, self.p_feat1 = self.model(self.positive2, True)
            self.p_out2, self.p_feat2 = self.model(self.positive1, True)
            self.n_out1, self.n_feat1 = self.model(self.negative2, True)
            self.n_out2, self.n_feat2 = self.model(self.negative1, True)
        else:
            self.p_out1, self.p_feat1 = self.model(self.positive1, True)
            self.p_out2, self.p_feat2 = self.model(self.positive2, True)
            self.n_out1, self.n_feat1 = self.model(self.negative1, True)
            self.n_out2, self.n_feat2 = self.model(self.negative2, True)
            

    def get_loss(self):
        return self.BCE_loss(self.a_out1.squeeze(1), self.label1)

    def optimize_parameters(self):
        self.forward()
        BCE_loss = self.BCE_loss(self.a_out1.squeeze(1), self.label1) + self.BCE_loss(self.a_out2.squeeze(1), self.label2)
        Tri_loss1 = self.triplet(self.a_feat1.squeeze(1), self.p_feat1.squeeze(1), self.n_feat1.squeeze(1))
        Tri_loss2 = self.triplet(self.a_feat2.squeeze(1), self.p_feat2.squeeze(1), self.n_feat2.squeeze(1))
        Tri_loss = (Tri_loss1 + Tri_loss2) / 2
        self.loss = BCE_loss + 0.4*Tri_loss
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        
        

class mix_Distrciminator_Trainer(BaseModel):
    def name(self):
        return 'mix_Distrciminator_Trainer'

    def __init__(self, opt):
        super(mix_Distrciminator_Trainer, self).__init__(opt)

        if self.isTrain and not opt.continue_train:
            self.model = resnet50(pretrained=True)
            self.model.fc = nn.Linear(2048, 1)
            torch.nn.init.normal_(self.model.fc.weight.data, 0.0, opt.init_gain)
            if opt.pretrained_path != None:
                state_dict = torch.load(opt.pretrained_path, map_location='cpu')
                self.model.load_state_dict(state_dict['model'])
                print(f'loading the weight {opt.pretrained_path} to fintinuing')
            self.discriminator = Discriminator_v1(2048)  
            
        if opt.fix_backbone:
            params = []
            for name, p in self.model.named_parameters():
                if  name=="fc.weight" or name=="fc.bias": 
                    params.append(p) 
                else:
                    p.requires_grad = False
        else:
            print("Your backbone is not fixed.")
            import time 
            time.sleep(3)
            params = self.model.parameters()            
        
        
        if not self.isTrain or opt.continue_train:
            self.model = resnet50(num_classes=1)

        if self.isTrain:
            self.BCE_Logit = nn.BCEWithLogitsLoss()
            self.BCE_loss = nn.BCELoss()

            # initialize optimizers
            if opt.optim == 'adam':
                self.optimizer = torch.optim.Adam(params,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(),
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
            elif opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD(params,
                                                 lr=opt.lr, momentum=0.0, weight_decay=0)
                self.optimizer_D = torch.optim.SGD(self.discriminator.parameters(),
                                                 lr=opt.lr, momentum=0.0, weight_decay=0)                                 
            else:
                raise ValueError("optim should be [adam, sgd]")

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.epoch)
        self.model.to(opt.gpu_ids[0])
        self.discriminator.to(opt.gpu_ids[0])


    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        return True

    def set_input(self, batch1, batch2):
        # batch = [anchor, label, positive, negative]
        self.input1 = batch1[0].to(self.device)
        self.label1 = batch1[1].to(self.device).float()
        self.positive1 = batch1[2].to(self.device)
        self.negative1 = batch1[3].to(self.device)
        self.input2 = batch2[0].to(self.device)
        self.label2 = batch2[1].to(self.device).float()
        self.positive2 = batch2[2].to(self.device)
        self.negative2 = batch2[3].to(self.device)
        
        if self.label1[0] == self.label2[0]:
            self.same_label = True
        else:
            self.same_label = False  

    def optimize_parameters(self):
        # Adversarial ground truths
        simi = torch.ones(self.input1.size()[0], 1, requires_grad=False) ## postive (similiar)
        disimi = torch.zeros(self.input1.size()[0], 1, requires_grad=False) ##negative (dissimiliar)
        
        # -----------------
        #  Train ResNet-50 classifier
        # -----------------
        a_out1, a_feat1 = self.model(self.input1, True)
        a_out2, a_feat2 = self.model(self.input2, True)
        if self.same_label:
            p_out1, p_feat1 = self.model(self.positive2, True)
            p_out2, p_feat2 = self.model(self.positive1, True)
            n_out1, n_feat1 = self.model(self.negative2, True)
            n_out2, n_feat2 = self.model(self.negative1, True)
        else:
            p_out1, p_feat1 = self.model(self.positive1, True)
            p_out2, p_feat2 = self.model(self.positive2, True)
            n_out1, n_feat1 = self.model(self.negative1, True)
            n_out2, n_feat2 = self.model(self.negative2, True)
        # Move disimi tensor to the same device as out_a tensor
        simi = simi.to(a_out1.device)
        disimi = disimi.to(a_out1.device)
        
        self.optimizer.zero_grad()
        out_loss = self.BCE_Logit(a_out1.squeeze(1), self.label1) + self.BCE_Logit(a_out2.squeeze(1), self.label2)
        # print(self.discriminator(feature_a, feature_p).size())
        AP1 = self.discriminator(a_feat1, p_feat1)
        AP2 = self.discriminator(a_feat2, p_feat2)
        dis_p = (self.BCE_loss(AP1, disimi) + self.BCE_loss(AP2, disimi)) / 2
        AN1 = self.discriminator(a_feat1, n_feat1)
        AN2 = self.discriminator(a_feat2, n_feat2)
        dis_n = (self.BCE_loss(AN1,simi) + self.BCE_loss(AN2,simi)) / 2
        
        self.loss = out_loss + 0.15*dis_p + 0.15*dis_n
        self.loss.backward()
        self.optimizer.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        self.optimizer_D.zero_grad()
        # Measure discriminator's ability to classify real from generated samples
        postive_loss1 = self.BCE_loss(self.discriminator(a_feat1.detach(), p_feat1.detach()), simi)
        postive_loss2 = self.BCE_loss(self.discriminator(a_feat2.detach(), p_feat2.detach()), simi)
        postive_loss = (postive_loss1 + postive_loss2) / 2
        negative_loss1 = self.BCE_loss(self.discriminator(a_feat1.detach(), n_feat1.detach()), disimi)
        negative_loss2 = self.BCE_loss(self.discriminator(a_feat2.detach(), n_feat2.detach()), disimi)
        negative_loss = (negative_loss1 + negative_loss2) / 2
        self.d_loss = postive_loss + negative_loss

        self.d_loss.backward()
        self.optimizer_D.step()      
        


class mix_Patch_Trainer(BaseModel):
    def name(self):
        return 'mix_Patch_Trainer'

    def __init__(self, opt):
        super(mix_Patch_Trainer, self).__init__(opt)
        self.in_f = opt.fuse_layer
        if self.isTrain and not opt.continue_train:
            self.model = resnet50(num_classes=1)
            if opt.pretrained_path != None:
                state_dict = torch.load(opt.pretrained_path, map_location='cpu')
                self.model.load_state_dict(state_dict['model'])
            else:
                raise(f"yout pretrained_path is empty, please set the path.")
            
            # label 0:[1,0,0] is real, label 1:[0,1,0] is fake, label 2:[0,0,1] is unseen
            self.classifier = FuseClassifier(num_classes=1, in_f=self.in_f)
            # self.classifier = Classifier(num_classes=1)
            torch.nn.init.normal_(self.classifier.fc1.weight.data, 0.0, opt.init_gain)
            torch.nn.init.normal_(self.classifier.fc2.weight.data, 0.0, opt.init_gain)
            torch.nn.init.normal_(self.classifier.fc3.weight.data, 0.0, opt.init_gain)
        
        # freeze the feature extractor and just train classifier
        # print('Freeze the feature extractor and just train classifier')
        params = []        
        for name, p in self.model.named_parameters():
            p.requires_grad = False 

        
        for name, p in self.classifier.named_parameters():
            params.append(p) 

        # params = self.classifier.parameters()           

        if self.isTrain:
            self.BCE_loss = nn.BCEWithLogitsLoss()
            self.CE_loss = nn.CrossEntropyLoss()
            # initialize optimizers
            if opt.optim == 'adam':
                self.optimizer = torch.optim.Adam(params,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
            elif opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD(params,
                                                 lr=opt.lr, momentum=0.0, weight_decay=0)
            else:
                raise ValueError("optim should be [adam, sgd]")

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.epoch)
            
        self.model.to(opt.gpu_ids[0])
        self.classifier.to(opt.gpu_ids[0])


    def adjust_learning_rate(self, min_lr=5e-7):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 8.
            if param_group['lr'] < min_lr:
                return False
        return True

    def set_input(self, input):
        # img, patch1, patch2, patch3, label, multi_label
        self.input = input[0].to(self.device)
        self.patch1 = input[1].to(self.device)
        self.patch2 = input[2].to(self.device)
        self.patch3 = input[3].to(self.device)
        self.label = input[4].to(self.device).float()
        self.multi_label = input[5].to(self.device).float()
        self.patchs = [self.patch1, self.patch2, self.patch3]
    
    def forward(self):
        # the whole image
        _, f_img, x1_img, x2_img, x3_img = self.model(self.input, return_features=True)
        features_pool = {'layer1':x1_img, 'layer2':x2_img, 'layer3':x3_img, 'feature':f_img}
        img_features = [features_pool[f] for f in self.in_f]
        self.bin_out_img = self.classifier(img_features)
        # patch
        # patch_select = random.randint(0,2)
        # _, f_patch, x1_patch, x2_patch, x3_patch = self.model(self.patchs[patch_select], return_features=True)
        # features_patch_pool = {'layer1':x1_patch, 'layer2':x2_patch, 'layer3':x3_patch, 'feature':f_patch}
        # patch_features = [features_patch_pool[f] for f in self.in_f]
        # self.bin_out_patch = self.classifier(patch_features)     
        
        
        # Feature augment
        # the whole image
        # aug_feature, _, aug_label = self.feature_augment_mix(f_img, self.label)
        # aug_feature_x1, _, aug_label = self.feature_augment_mix(x1_img, self.label)
        # aug_feature_x2, _, aug_label = self.feature_augment_mix(x2_img, self.label)
        # aug_feature_x3, _, aug_label = self.feature_augment_mix(x3_img, self.label)
        # aug_feature = aug_feature.to(self.device)
        # aug_feature_x1 = aug_feature_x1.to(self.device)
        # aug_feature_x2 = aug_feature_x2.to(self.device)
        # aug_feature_x3 = aug_feature_x3.to(self.device)
        
        # self.aug_label = aug_label.to(self.device).float()
        # aug_features_pool = {'layer1':x1_img, 'layer2':x2_img, 'layer3':x3_img, 'feature':aug_feature}
        # aug_img_features = [aug_features_pool[f] for f in self.in_f]
        # self.aug_bin_out = self.classifier(aug_img_features)
        # patch
        # patch_aug_feature, _, _ = self.feature_augment_mix(f_patch, self.label)
        # patch_aug_feature = patch_aug_feature.to(self.device)
        # patch_aug_features_pool = {'layer1':x1_patch, 'layer2':x2_patch, 'layer3':x3_patch, 'feature':patch_aug_feature}
        # patch_aug_img_features = [patch_aug_features_pool[f] for f in self.in_f]
        # self.patch_aug_bin_out = self.classifier(patch_aug_img_features)
 
    
    def feature_augment_mix(self, feature, label):
        def interpolate_push(f_x, f_c):
            alpha = random.uniform(0.75, 1.5)
            new_f_x = f_x + alpha * (f_x - f_c)
            return new_f_x
        def interpolate_pull(f_x, f_c):
            # alpha = random.uniform(0.5, 1.5)
            alpha = random.uniform(0.5, 1.5)
            new_f_x = f_x + alpha * (f_c - f_x)
            return new_f_x    
        
        
        # direct manner: directly push out the feature points from the batch centriod same class 
        center = feature.mean(dim=0, keepdim=True)
        # print(feature.size())
        # real class
        real_mask = (label  == 1)
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
        aug_feature = torch.cat((real_aug_feature, fake_aug_feature), dim=0).squeeze(1) 
        # print(aug_feature.size())
        batch_size = aug_feature.size()[0]
        aug_mul_label = torch.tensor([0.0,0.0,1.0]*batch_size).reshape(batch_size,-1)
        aug_label = torch.tensor([1.0]*batch_size).reshape(batch_size)
        return aug_feature, aug_mul_label, aug_label
    
        
    def get_loss(self):
        # CE_loss = 0.3 * self.CE_loss(self.multi_out.squeeze(1), self.multi_label)
        BCE_loss = self.BCE_loss(self.bin_out.squeeze(1), self.label)
        return  BCE_loss

    def optimize_parameters(self):
        self.forward()        
        BCE_loss = self.BCE_loss(self.bin_out_img.squeeze(1), self.label)         
        # BCE_loss = (self.BCE_loss(self.bin_out_img.squeeze(1), self.label) + self.BCE_loss(self.bin_out_patch.squeeze(1), self.label))/2         
        # aug_BCE_loss = 0.2*(self.BCE_loss(self.aug_bin_out.squeeze(1), self.label))
        # aug_BCE_loss = 0.2*((self.BCE_loss(self.aug_bin_out.squeeze(1), self.label)+self.BCE_loss(self.patch_aug_bin_out.squeeze(1), self.label))/2)
        
        self.loss = BCE_loss
        # self.loss = BCE_loss + aug_BCE_loss
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()    