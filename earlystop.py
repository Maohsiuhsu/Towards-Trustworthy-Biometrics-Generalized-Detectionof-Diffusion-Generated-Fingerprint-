import numpy as np
import torch
from eval_config import *
from options.test_options import TestOptions
from validate import validate, validate_patch, validate_v2

def evaluate_all(model, classifier, opt):
    acc_cum = 0
    for v_id, val in enumerate(vals):
        model.eval()
        classifier.eval()
        opt.no_resize = True    # testing without resizing by default

        # acc = validate_v2(model, classifier, opt, no_eval=False, eval_fake_name=val)[0]
        acc = validate(model, classifier, opt, no_eval=False, eval_fake_name=val)[0]
        # acc = validate_patch(model, classifier, opt, no_eval=False, eval_fake_name=val)[0]

        acc_cum = acc_cum + acc
        
    return acc_cum / len(vals)
    
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=1, verbose=False, delta=0, eval_best=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.score_max = -np.Inf
        self.delta = delta
        self.eval_best = eval_best

    def __call__(self, score, model, epoch, opt):
        if self.best_score is None:
            self.best_score = score  
            avg_acc = evaluate_all(model.model, model.classifier, opt)
            if self.eval_best < avg_acc:
                self.save_checkpoint(score, model, epoch)
                self.eval_best = avg_acc
                print(f'\nthe best test avg acc is {self.eval_best}\n') 
                
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
            avg_acc = evaluate_all(model.model, model.classifier, opt)
            if self.eval_best < avg_acc:
                self.save_checkpoint(score, model, epoch)
                self.eval_best = avg_acc
                print(f'\nThe best test avg acc is {self.eval_best}\n')     
        else:
            self.best_score = score
            avg_acc = evaluate_all(model.model, model.classifier, opt)
            if self.eval_best < avg_acc:
                self.save_checkpoint(score, model, epoch)
                self.eval_best = avg_acc
                print(f'\nThe best test avg acc is {self.eval_best}\n') 
            # self.save_checkpoint(score, model, epoch)
            self.counter = 0    

    def save_checkpoint(self, score, model, epoch):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation accuracy increased ({self.score_max:.6f} --> {score:.6f}\n).  Saving model ...')
        model.save_networks(f'best_{epoch}')
        self.score_max = score
