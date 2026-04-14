import os
import sys
import time
import torch
import torch.nn
import argparse
from PIL import Image
from tensorboardX import SummaryWriter
from random import random
from validate import validate, validate_multiple, validate_v2
from data import create_dataloader
from earlystop import EarlyStopping
from options.train_options import TrainOptions
import importlib
import itertools

"""Currently assumes jpg_prob, blur_prob 0 or 1"""
def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    # val_opt.dataroot = '{}/{}/'.format(val_opt.dataroot, val_opt.val_split)
    val_opt.isTrain = False
    val_opt.flip = False
    val_opt.serial_batches = True
    val_opt.jpg_method = ['pil']
    if len(val_opt.blur_sig) == 2:
        b_sig = val_opt.blur_sig
        val_opt.blur_sig = [(b_sig[0] + b_sig[1]) / 2]
    if len(val_opt.jpg_qual) != 1:
        j_qual = val_opt.jpg_qual
        val_opt.jpg_qual = [int((j_qual[0] + j_qual[-1]) / 2)]
    val_opt.dataset_mode = "normal"
    
    return val_opt


if __name__ == '__main__':
    opt = TrainOptions().parse()
    val_opt = get_val_opt()
    paths1 = {"real":[os.path.join(opt.data_root, 'train', opt.real_data_name)], 
             "fake":[os.path.join(opt.data_root, 'train', fake_name) for fake_name in opt.fake_data_name]}
    data_loader1 = create_dataloader(paths1, opt)    
    dataset_size1 = len(data_loader1)
    paths2 = {"real":[os.path.join(opt.data_root, 'patch_48', 'train', f'{opt.real_data_name}_patch48')], 
             "fake":[os.path.join(opt.data_root, 'patch_48', 'train', f'{fake_name}_patch48') for fake_name in opt.fake_data_name]}
    data_loader2 = create_dataloader(paths2, opt)    
    dataset_size2 = len(data_loader2)
    print('#training batches = %d + %d' % (dataset_size1, dataset_size2))

    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))
    
    trainer_module = importlib.import_module('networks.multiple_classifier_trainer_v2', __package__)
    if opt.dataset_mode == "normal":
        TrainerClass = getattr(trainer_module, 'mix_commen_Trainer')
    
    else:
        raise ValueError("Invalid dataset mode:", opt.dataset_mode)
    
    # Now you can use DatasetClass as the dataset class you need
    model = TrainerClass(opt)
    
    
    best_acc = 0
    early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.001, verbose=True)
    for epoch in range(opt.niter):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for (data1, data2) in itertools.zip_longest(data_loader1, data_loader2, fillvalue=None):
                
            if data1 is not None:
                if data1[0].size()[0] == data2[0].size()[0]:
                    model.set_input(data1, data2)   
                else:
                    continue
            # elif random() < 0.3:
                # model.set_input(data2, data2)
            else:
                break
                
            model.optimize_parameters(decay=max(0.1*epoch,0.4))
            model.total_steps += 1
            epoch_iter += opt.batch_size    

            if model.total_steps % opt.loss_freq == 0:
                print("Train loss: {} at step: {}".format(model.loss, model.total_steps))
                train_writer.add_scalar('loss', model.loss, model.total_steps)

            # if model.total_steps % opt.save_latest_freq == 0:
                # print('saving the latest model %s (epoch %d, model.total_steps %d)' %
                      # (opt.name, epoch, model.total_steps))
                # model.save_networks('latest')

            # print("Iter time: %d sec" % (time.time()-iter_data_time))
            # iter_data_time = time.time()

        # if epoch % opt.save_epoch_freq == 0:
            # print('saving the model at the end of epoch %d, iters %d' %
                  # (epoch, model.total_steps))
            # model.save_networks('latest')
            # model.save_networks(epoch)

        # Validation
        model.eval()
        # acc, ap = validate_multiple(model.model, model.classifier, val_opt)[4:6]
        acc, ap = validate_v2(model.model, model.classifier, val_opt)[:2]
        # val_writer.add_scalar('accuracy', acc, model.total_steps)
        # val_writer.add_scalar('ap', ap, model.total_steps)
        print("(Val @ epoch {}) acc: {}; ap: {}".format(epoch, acc, ap))

        early_stopping(acc, model, epoch, val_opt)
        if early_stopping.early_stop:
            cont_train = model.adjust_learning_rate()
            if cont_train:
                print("Learning rate dropped by 10, continue training...")
                early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.002, verbose=True, eval_best=early_stopping.eval_best)
            else:
                print("Early stopping.")
                break
        model.train()

