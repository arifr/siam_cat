import os
import sys
import argparse
import datetime
import time
import os.path as osp
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np


import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision
#from torchvision import models
import datasets
import models
from utils import AverageMeter, Logger
#from center_loss import CenterLoss
#from center_loss2 import CenterLoss
import copy
from tqdm import tqdm
from sklearn import preprocessing




parser = argparse.ArgumentParser("Test Center Loss Example")
# dataset
parser.add_argument('-d', '--dataset', type=str, default='mnist', choices=['mnist','fmnist','SOP','SOPTest','inshop','inshopTest','custShopTest'])
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
# optimization
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--lr-model', type=float, default=0.001, help="learning rate for model")
parser.add_argument('--lr-cent', type=float, default=0.5, help="learning rate for center loss")
parser.add_argument('--weight-cent', type=float, default=1, help="weight for center loss")
parser.add_argument('--max-epoch', type=int, default=100)
parser.add_argument('--stepsize', type=int, default=20)
parser.add_argument('--gamma', type=float, default=0.5, help="learning rate decay")
# model
parser.add_argument('--model', type=str, default='cnn')
parser.add_argument('--model-fname', type=str, default='', help="model file path")
parser.add_argument('--feat-dim', type=int, default=2, help="feature dimension")
parser.add_argument('--sample', type=bool, default=False)


# misc
parser.add_argument('--eval-freq', type=int, default=10)
parser.add_argument('--print-freq', type=int, default=50)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--use-cpu', action='store_true')
parser.add_argument('--save-dir', type=str, default='log')
parser.add_argument('--plot', action='store_true', help="whether to plot features for every epoch")

args = parser.parse_args()



FEATURES=()


def hook_fn(self,input,output):
    FEATURES = input    






def main():
    prefix = 'features'
    
    
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False

    #sys.stdout = Logger(osp.join(args.save_dir, 'log_' + args.dataset + '.txt'))
       

    if use_gpu:
        print("Currently using GPU: {}".format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
        device = 'cuda'
    else:
        print("Currently using CPU")
        device = 'cpu'

    print("Creating dataset: {}".format(args.dataset))
    
    dataset = datasets.create(
        name=args.dataset, batch_size=args.batch_size, use_gpu=use_gpu,
        num_workers=args.workers, sample = args.sample) 
    
    testloader = dataset.testloader
    #testloader = dataset.trainloader
    

    print("Loading model: {}".format(args.model))
    model = torch.load(args.model_fname,map_location=device)

    #model.base_model.classifier = nn.Identity()
    print(model)
    #model.base_model.fc[2] = nn.Identity()
    #model.module.base_model.classifier = nn.Identity()
    model.base_model.classifier[3] = nn.Identity()
          
    
    #print(model.module.base_model)
    
    fname = os.path.basename(args.model_fname)
    feature_dim = args.feat_dim
    print(feature_dim)
    
    if use_gpu:
        #model = nn.DataParallel(model).cuda()
        model = model.cuda()
            
    
    start_time = time.time()
    
    log_dir = os.path.join(args.save_dir,prefix)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    #log = open('{}/{}.log'.format(log_dir,fname),'w')
    #log.write('acc;\n')
    #fname = 'reg_norm_' + fname
    fname = os.path.join(log_dir,fname)
    save_features(model, testloader, feature_dim, fname, use_gpu)
    #log.write('{};\n'.format(acc) )
    #log.close()

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))



def save_features(model, testloader, feature_dim, fname, use_gpu):  #, epoch):
    
    model.eval()
    feature_dim +=1
    embeddings= np.ndarray((len(testloader.dataset),feature_dim),dtype='float32')
    with torch.no_grad():
        i = 0
        for data, labels in tqdm(testloader):
            if use_gpu:
                #data, labels = data.cuda().half(), labels.cuda()
                data, labels = data.cuda(), labels.cuda()
                
            #features, outputs = model(data)
            outputs = model(data)
            outputs = preprocessing.normalize(outputs.detach().cpu().numpy(),norm='l2')            
            for output, label  in zip(outputs,labels):
                #print('feat:',feature)
                #embeddings[i,:feature_dim-1] = output.detach().cpu().numpy()
                #output = preprocessing.normalize(output.detach().cpu().numpy(),norm='l2')            
                embeddings[i,:feature_dim-1] = output
                                
                embeddings[i,-1] = label
                i+=1

    np.save('{}.npy'.format(fname),embeddings)



if __name__ == '__main__':
    main()

