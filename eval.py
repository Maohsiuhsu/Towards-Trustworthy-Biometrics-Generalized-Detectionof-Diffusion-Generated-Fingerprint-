import os
import csv
import torch
from networks.classifier import Multi_Binary_Classifier, Classifier, FuseClassifier
from validate import validate, validate_multiple, validate_patch, validate_v2
from networks.resnet import resnet50
from options.test_options import TestOptions
from eval_config import *
import numpy as np


# Running tests
opt = TestOptions().parse(print_options=False)
# model_name = os.path.basename(opt.model_path).replace('.pth', '')
rows = [["{} model testing on...".format(opt.model_path)],
        ['testset', 'accuracy', 'avg precision', 'real acc', 'fake acc']]

model = resnet50(num_classes=1)
# classifier = Multi_Binary_Classifier(num_classes=3)
classifier = Classifier(num_classes=1)
# classifier = FuseClassifier(num_classes=1, in_f=opt.fuse_layer)

# load weight
state_dict = torch.load(opt.model_path, map_location='cpu')
model.load_state_dict(state_dict['model'])
classifier.load_state_dict(state_dict['classifier'])

model.cuda()
classifier.cuda()
model.eval()
classifier.eval()

print("{} model testing on...".format(opt.model_path))
for v_id, val in enumerate(vals):
    opt.no_resize = True    # testing without resizing by default
    opt.fake_data_name = val
    

    acc, ap, r_acc, f_acc, _, _ = validate(model, classifier, opt, no_eval=False, eval_fake_name=val)
    # acc, ap, r_acc, f_acc, _, _ = validate_patch(model, classifier, opt, no_eval=False, eval_fake_name=val)
    # acc, ap, r_acc, f_acc = validate_multiple(model, classifier, opt, no_eval=False)[:4]
    rows.append([val, acc, ap, r_acc, f_acc])
    print("({}) acc: {}; ap: {}; r_acc: {}; f_acc: {};".format(val, acc, ap, r_acc, f_acc))

csv_name = results_dir + '/{}.csv'.format(opt.name)
with open(csv_name, 'w') as f:
    csv_writer = csv.writer(f, delimiter=',')
    csv_writer.writerows(rows)
"""    
    
# Running tests
opt = TestOptions().parse(print_options=False)
# model_name = os.path.basename(opt.model_path).replace('.pth', '')

print("{} model testing on...".format(opt.model_path))
best_acc = 0
n = len(vals)

model = resnet50(num_classes=1)
# classifier = Multi_Binary_Classifier(num_classes=3)
# classifier = Classifier(num_classes=1)
classifier = FuseClassifier(num_classes=1, in_f=['layer1','layer2','layer3','feature'])

# load weight
state_dict = torch.load(opt.model_path, map_location='cpu')
model.load_state_dict(state_dict['model'])
classifier.load_state_dict(state_dict['classifier'])

model.cuda()
classifier.cuda()
model.eval()
classifier.eval()

for threshold in np.arange(0.11, 0.20, 0.005):
    print(f"offset: {threshold}")
    rows = [["{} model testing on...".format(opt.model_path)],
            ['testset', 'accuracy', 'avg precision', 'real acc', 'fake acc']]
    cur_acc = 0        
    
    for v_id, val in enumerate(vals):
        print(val)
        opt.no_resize = True    # testing without resizing by default
        opt.fake_data_name = val
        
        # acc, ap, r_acc, f_acc, _, _ = validate(model, classifier, opt, no_eval=False)
        acc, ap, r_acc, f_acc, _, _ = validate_patch(model, classifier, opt, no_eval=False)
        # acc, ap, r_acc, f_acc, _, _ = validate_softmax_diff(model, classifier, opt, no_eval=False, offset=threshold)
        # acc, ap, r_acc, f_acc = validate_multiple(model, classifier, opt, no_eval=False)[:4]
        rows.append([val, acc, ap, r_acc, f_acc])
        # print("({}) acc: {}; ap: {}; r_acc: {}; f_acc: {};".format(val, acc, ap, r_acc, f_acc))
        cur_acc = cur_acc + acc
      
      
    avg_acc = cur_acc/n
    if avg_acc > best_acc:
        best_acc = avg_acc
        best_threshold = threshold
        best_rows = rows
        print(f"In threshold {threshold}, find best avg acc {best_acc}\n")
        
best_rows.append(["avg acc", best_acc, "threshold", best_threshold])        
csv_name = results_dir + '/{}.csv'.format(opt.name)
with open(csv_name, 'w') as f:
    csv_writer = csv.writer(f, delimiter=',')
    csv_writer.writerows(best_rows)    
"""