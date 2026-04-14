import torch
import numpy as np
from networks.resnet import resnet50
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from options.test_options import TestOptions
from data import create_dataloader
import os



def validate(model, classifier, opt, no_eval=True, eval_fake_name=None):
    if not no_eval:
        print(f'start eval')
        paths = {"real":[os.path.join(opt.data_root, 'test', opt.real_data_name)], 
                 "fake":[os.path.join(opt.data_root, 'test', eval_fake_name)]}
    elif no_eval:
        paths = {"real":[os.path.join(opt.data_root, 'val', opt.real_data_name)], 
                 "fake":[os.path.join(opt.data_root, 'val', fake_name) for fake_name in opt.fake_data_name]}
             
    data_loader = create_dataloader(paths, opt)
    softmax = torch.nn.Softmax(dim=1)
    
    with torch.no_grad():
        y_true, y_pred = [], []
        for img, multi_label, label, _ in data_loader:
            in_tens = img.cuda()
            _, feature = model(in_tens, True)
            y_out = classifier(feature)
            # if label[0]==0:
                # print(softmax(multi_pred))
            y_pred.extend(y_out.sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > 0.5)
    f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > 0.5)
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)
    return acc, ap, r_acc, f_acc, y_true, y_pred


def validate_v2(model, classifier, opt, no_eval=True, eval_fake_name=None):
    if not no_eval:
        print(f'start eval')
        paths = {"real":[os.path.join(opt.data_root, 'test', opt.real_data_name)], 
                 "fake":[os.path.join(opt.data_root, 'test', eval_fake_name)]}
    elif no_eval:
        paths = {"real":[os.path.join(opt.data_root, 'val', opt.real_data_name)], 
                 "fake":[os.path.join(opt.data_root, 'val', fake_name) for fake_name in opt.fake_data_name]}
             
    data_loader = create_dataloader(paths, opt)
    softmax = torch.nn.Softmax(dim=1)
    
    with torch.no_grad():
        y_true, y_pred = [], []
        for img, multi_label, label, _ in data_loader:
            in_tens = img.cuda()
            feature = model(in_tens, True)
            y_out = classifier(feature)
            # if label[0]==0:
                # print(softmax(multi_pred))
            y_pred.extend(y_out.sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > 0.5)
    f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > 0.5)
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)
    return acc, ap, r_acc, f_acc, y_true, y_pred

def validate_multiple(model, classifier, opt, no_eval=True, eval_fake_name=None):
    if not no_eval:
        print(f'start eval')
        paths = {"real":[os.path.join(opt.data_root, 'test', opt.real_data_name)], 
                 "fake":[os.path.join(opt.data_root, 'test', eval_fake_name)]}
    elif no_eval:
        paths = {"real":[os.path.join(opt.data_root, 'val', opt.real_data_name)], 
                 "fake":[os.path.join(opt.data_root, 'val', fake_name) for fake_name in opt.fake_data_name]}
             
    data_loader = create_dataloader(paths, opt)
    softmax = torch.nn.Softmax(dim=1)
    
    with torch.no_grad():
        y_true, y_pred, y_multi_pred = [], [], []
        for img, multi_label, label in data_loader:
            in_tens = img.cuda()
            _, feature = model(in_tens, True)
            multi_pred, y_out = classifier(feature)

            multi_out = softmax(multi_pred)
            # convert the multiple pred to class predictions
            class_pred = torch.argmax(multi_out, dim=1)
            y_multi_pred.extend(class_pred.cpu().tolist())
            y_pred.extend(y_out.sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_multi_pred = np.array(y_multi_pred)
    # multiple predict result
    y_multi_pred[y_multi_pred==2] = 1
    r_multi_acc = accuracy_score(y_true[y_true==0], y_multi_pred[y_true==0])
    f_multi_acc = accuracy_score(y_true[y_true==1], y_multi_pred[y_true==1])
    acc_multi = accuracy_score(y_true, y_multi_pred)
    ap_multi = average_precision_score(y_true, y_multi_pred)
    
    # binary predict result
    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > 0.5)
    f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > 0.5)
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)
    return acc, ap, r_acc, f_acc, acc_multi, ap_multi, r_multi_acc, f_multi_acc



def validate_patch(model, classifier, opt, no_eval=True,eval_fake_name=None):
    if not no_eval:
        # print(f'start eval')
        paths = {"real":[os.path.join(opt.data_root, 'test', opt.real_data_name)], 
                 "fake":[os.path.join(opt.data_root, 'test', eval_fake_name)]}
    elif no_eval:
        paths = {"real":[os.path.join(opt.data_root, 'val', opt.real_data_name)], 
                 "fake":[os.path.join(opt.data_root, 'val', fake_name) for fake_name in opt.fake_data_name]}
             
    data_loader = create_dataloader(paths, opt)
    softmax = torch.nn.Softmax(dim=1)
    
    with torch.no_grad():
        y_true, y_pred = [], []
        # y_img, y_patch1_pred, y_patch2_pred, y_patch3_pred = [], [], [], []
        for data in data_loader:

            img, patch1, patch2, patch3, label, _ = data
            # predict the whole image
            in_tens = img.cuda()
            _, f_img, x1_img, x2_img, x3_img = model(in_tens, return_features=True)
            features_pool = {'layer1':x1_img, 'layer2':x2_img, 'layer3':x3_img, 'feature':f_img}
            #["layer1", "layer2", "layer3", "feature"]
            in_features = [features_pool[f] for f in opt.fuse_layer]
            y_img_out = classifier(in_features)
            
            # predict the patch1
            # patch1_tens = patch1.cuda()
            # _, feature1 = model(patch1_tens, True)
            
            # predict the patch2
            # patch2_tens = patch2.cuda()
            # _, feature2 = model(patch2_tens, True)
            
            # predict the patch3
            # patch3_tens = patch3.cuda()
            # _, feature3 = model(patch3_tens, True)
            # y_patch3_out = classifier(in_feature)    
            
            
            y_true.extend(label.flatten().tolist())
            y_pred.extend(y_img_out.sigmoid().flatten().tolist())
            # y_patch1_pred.extend(y_patch1_out.sigmoid().flatten().tolist())
            # y_patch2_pred.extend(y_patch2_out.sigmoid().flatten().tolist())
            # y_patch3_pred.extend(y_patch3_out.sigmoid().flatten().tolist())
            
    y_true, y_pred = np.array(y_true), np.array(y_pred) 
    # y_true = np.array(y_true)
    # y_patch3_pred = np.array(y_patch3_pred)
    # y_patch1_pred, y_patch2_pred, y_patch3_pred = np.array(y_patch1_pred), np.array(y_patch2_pred), np.array(y_patch3_pred)


    # print(y_true)
    # print(len(y_true))
    # print(y_pred)
    # print(len(y_pred))
    
    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > 0.5)
    f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > 0.5)
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)
    return acc, ap, r_acc, f_acc, y_true, y_pred








if __name__ == '__main__':
    opt = TestOptions().parse(print_options=False)

    model = resnet50(num_classes=1)
    state_dict = torch.load(opt.model_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    model.cuda()
    model.eval()

    acc, avg_precision, r_acc, f_acc, y_true, y_pred = validate(model, opt)

    print("accuracy:", acc)
    print("average precision:", avg_precision)

    print("accuracy of real images:", r_acc)
    print("accuracy of fake images:", f_acc)
