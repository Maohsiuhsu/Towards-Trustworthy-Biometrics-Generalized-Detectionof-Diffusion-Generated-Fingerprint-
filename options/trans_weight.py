import os
import csv
import torch
from networks.classifier import Classifier
from networks.resnet import resnet50
from options.test_options import TestOptions
import numpy as np
import argparse


def save_networks(save_dir, model, classifier):
    save_filename = 'best_weights.pth'
    save_path = os.path.join(save_dir, save_filename)

    # serialize model and optimizer to dict
    state_dict = {
        'model': model.state_dict(),
        'classifier': classifier.state_dict(),
        # 'optimizer' : self.optimizer.state_dict(),
        # 'total_steps' : self.total_steps,
    }

    torch.save(state_dict, save_path)


parser = argparse.ArgumentParser()
parser.add_argument('--dir_path', type=str)
parser.add_argument('--model_name', type=str)
# Parse arguments
args = parser.parse_args()



model = resnet50(num_classes=1)

classifier = Classifier(num_classes=1)


# load weight
model_path = os.path.join(args.dir_path, args.model_name)
state_dict = torch.load(model_path, map_location='cpu')
model.load_state_dict(state_dict['model'])
classifier.load_state_dict(state_dict['classifier'])
model.del(model.fc)

save_networks(args.dir_path, model,classifier)