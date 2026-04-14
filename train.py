import os
import sys
import time
import torch
import torch.nn
import argparse
from PIL import Image
from tensorboardX import SummaryWriter
from random import random
from validate import validate, validate_multiple
from data import create_dataloader
from earlystop import EarlyStopping
from options.train_options import TrainOptions
import importlib

"""Currently assumes jpg_prob, blur_prob 0 or 1"""
def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    # val_opt.dataroot = '{}/{}/'.format(val_opt.dataroot, val_opt.val_split)
    val_opt.isTrain = False
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
    paths = {"real":os.path.join(opt.data_root, 'train', opt.real_data_name), 
             "fake":os.path.join(opt.data_root, 'train', opt.fake_data_name)}
    data_loader = create_dataloader(paths, opt)
    dataset_size = len(data_loader)
    print('#training batches = %d' % dataset_size)

    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))
    
    trainer_module = importlib.import_module('networks.trainer', __package__)
    if opt.dataset_mode == "normal":
        TrainerClass = getattr(trainer_module, 'Trainer')
    elif opt.dataset_mode == "pair":
        TrainerClass = getattr(trainer_module, 'Contrastive_Trainer')
    elif opt.dataset_mode == "triple":
        TrainerClass = getattr(trainer_module, 'Triplet_Trainer')
    elif opt.dataset_mode == "two_triple":
        pass
    elif opt.dataset_mode == "discriminator":
        TrainerClass = getattr(trainer_module, 'Discriminator_Trainer')
    else:
        raise ValueError("Invalid dataset mode:", opt.dataset_mode)
    
    # Now you can use DatasetClass as the dataset class you need
    model = TrainerClass(opt)
    
    
    
    early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.001, verbose=True)
    for epoch in range(opt.niter):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(data_loader):
            model.total_steps += 1
            epoch_iter += opt.batch_size
            
            model.set_input(data)      
            model.optimize_parameters()

            if model.total_steps % opt.loss_freq == 0:
                print("Train loss: {} at step: {}".format(model.loss, model.total_steps))
                train_writer.add_scalar('loss', model.loss, model.total_steps)

            if model.total_steps % opt.save_latest_freq == 0:
                print('saving the latest model %s (epoch %d, model.total_steps %d)' %
                      (opt.name, epoch, model.total_steps))
                model.save_networks('latest')

            # print("Iter time: %d sec" % (time.time()-iter_data_time))
            # iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, model.total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)

        # Validation
        model.eval()
        # acc, ap = validate(model.model, val_opt)[:2]
        # acc, ap = validate_multiple(model.model, model.classifier, val_opt)[4:6]
        acc, ap = validate_multiple(model.model, model.classifier, val_opt)[:2]
        val_writer.add_scalar('accuracy', acc, model.total_steps)
        val_writer.add_scalar('ap', ap, model.total_steps)
        print("(Val @ epoch {}) acc: {}; ap: {}".format(epoch, acc, ap))

        early_stopping(acc, model)
        if early_stopping.early_stop:
            cont_train = model.adjust_learning_rate()
            if cont_train:
                print("Learning rate dropped by 10, continue training...")
                early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.002, verbose=True)
            else:
                print("Early stopping.")
                break
        model.train()

