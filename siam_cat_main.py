import os
import sys
import argparse
import datetime
import time
#import numpy as np
from pytorch_metric_learning import losses, miners,testers

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
import datasets
import models
import copy
from tqdm import tqdm
from utils import   AverageMeter

parser = argparse.ArgumentParser("Train CNN Multiloss Class")
# dataset
parser.add_argument('-d', '--dataset', type=str, default='SOPClass', choices=['mnist','fmnist','SOPClass','inshop','custShopClass'])
parser.add_argument('-j', '--workers', default=2, type=int,
                    help="number of data loading workers (default: 2)")
# optimization
parser.add_argument('--batch-size', type=int, default=80)
parser.add_argument('--lr-model', type=float, default=0.0001, help="learning rate for model")
parser.add_argument('--max-epoch', type=int, default=1)
parser.add_argument('--stepsize', type=int, default=20)
parser.add_argument('--gamma', type=float, default=0.5, help="learning rate decay")

# model & loss
parser.add_argument('--model', type=str, default='mobilenet')
parser.add_argument('--code-size', type=int, default=512)
parser.add_argument('--miner', type=str, default='pairmargin')
parser.add_argument('--loss-fn', type=str, default='pair')


#parser.add_argument('--sample', type=bool, default=False)

# misc
parser.add_argument('--eval-freq', type=int, default=10)
parser.add_argument('--print-freq', type=int, default=50)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--use-cpu', action='store_true')
parser.add_argument('--save-dir', type=str, default='log')
#parser.add_argument('--plot', action='store_true', help="whether to plot features for every epoch")

args = parser.parse_args()
torch.autograd.set_detect_anomaly(True)


# config
coder_loss_fn = { 'triplet': losses.TripletMarginLoss(),  'pair':losses.ContrastiveLoss()
}

coder_miner = {
    'pairmargin': miners.PairMarginMiner(pos_margin=0.2, neg_margin=0.8,use_similarity=False), 
    'triplet_all': miners.TripletMarginMiner(margin=0.2),
    'triplet_hard':  miners.TripletMarginMiner(margin=0.2,type_of_triplets="hard"),
    'triplet_semi':  miners.TripletMarginMiner(margin=0.2,type_of_triplets="semihard"),
    'triplet_easy':  miners.TripletMarginMiner(margin=0.2,type_of_triplets="easy")
     
}


#hook of layer
FEATURES=()
def hook_fn(self,input,output):
    global FEATURES
    FEATURES = input    


def main():
    prefix = 'train'
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False

    
    if use_gpu:
        print("Currently using GPU: {}".format(args.gpu))
        torch.cuda.empty_cache()
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
        device = 'cuda'

    else:
        print("Currently using CPU")
        device = 'cpu'

    
    print("Creating dataset: {}".format(args.dataset))
    dataset = datasets.create(
        name=args.dataset, batch_size=args.batch_size, use_gpu=use_gpu,
        num_workers=args.workers, sample = False, 
    )

    trainloader = dataset.trainloader
    
    print("Creating model: {}".format(args.model))
    model = models.create(name=args.model, num_classes=dataset.num_classes).to(device)
    coder = nn.Linear(model.feature_dim, args.code_size).to(device)
    #model.hook_layer.register_forward_hook(hook_fn)
    model.set_hook(hook_fn)
    
    fname = 'mg_catsiam_{}_{}_{}_{}_{}_{}_{}_{}'.format(model.name, args.dataset,args.batch_size, args.max_epoch, args.miner,args.loss_fn, args.lr_model,args.code_size)
    coder_fname = 'mg_catsiam_emb_{}_{}_{}_{}_{}_{}_{}_{}'.format(model.name, args.dataset,args.batch_size, args.max_epoch, args.miner,args.loss_fn, args.lr_model,args.code_size)
        
    print(model)
    print(coder)
        
    print('file name:',fname)
        
    if use_gpu:
        model = model.cuda()
        '''
        model.cuda().half()
        for layer in model.modules():
            if isinstance(layer,nn.BatchNorm2d):
                layer.float()

        coder.cuda().half()
        '''
        #model = nn.DataParallel(model).cuda()

    #class_loss = nn.CrossEntropyLoss()

    class_loss = nn.MultiMarginLoss(p=2,margin=2.0)
    
    
    # coder loss fn
    coder_loss = coder_loss_fn[args.loss_fn]
    #coder miner 
    miner = coder_miner[args.miner]
    

    opt_model = torch.optim.Adam(model.parameters(), lr=args.lr_model, weight_decay=0.0001) #, momentum=0.9)
    opt_coder = torch.optim.Adam(coder.parameters(), lr=args.lr_model , weight_decay=0.0001) #, momentum=0.9)
        
    
    
    if args.stepsize > 0:
        scheduler = lr_scheduler.StepLR(opt_model, step_size=args.stepsize, gamma=args.gamma)

    start_time = time.time()
    best_model= copy.deepcopy(model.state_dict())
    best_acc = 0.0

    best_coder= copy.deepcopy(coder.state_dict())
    best_loss = sys.maxsize
    

    log_dir = os.path.join(args.save_dir,prefix)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log = open('{}/{}.log'.format(log_dir,fname),'w')
    log.write('epoch;acc;loss;coder\n')

    model.train()
    for epoch in range(args.max_epoch):
        print(">> Epoch {}/{}".format(epoch+1, args.max_epoch))

        acc, loss_model, loss_coder = train(model, coder, class_loss , coder_loss, miner, opt_model, opt_coder, trainloader, use_gpu) #, dataset.num_classes, epoch)
        log.write('{};{};{};{}\n'.format(epoch,acc,loss_model,loss_coder) )
        
        if args.stepsize > 0: scheduler.step()
        
        if acc > best_acc :
            best_acc = acc
            #best_model = copy.deepcopy(model.state_dict())
            #best_coder = copy.deepcopy(coder.state_dict())
            
        
        if loss_coder < best_loss :
            best_loss = loss_coder
            best_coder = copy.deepcopy(coder.state_dict())
            #best_model = copy.deepcopy(model.state_dict())
            best_model = copy.deepcopy(model.state_dict())
                                                

        print("acc: {:.3f}/{:.3f} loss: {:.3f}".format(acc, best_acc,loss_model))
     
    log.close()

    model.load_state_dict(best_model)
    torch.save(model,'{}/{}.pth'.format(log_dir,fname) )
    print(fname,' saved.')

    coder.load_state_dict(best_coder)
    torch.save(coder,'{}/{}.pth'.format(log_dir,coder_fname) )
    print(coder_fname,' saved.')


    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("== Validation ==\n")
    print("Accuracy (%): {}\t Error rate (%): {}".format(best_acc, 100.0-best_acc))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))


def train(model,coder,class_loss, coder_loss, miner,
          opt_model, opt_coder, trainloader, use_gpu):

    losses_model = AverageMeter()
    losses_coder = AverageMeter()
    
    correct = 0.0
    total = 0.0
    running_loss_model = 0.0
    running_loss_coder = 0.0
    
    #if use_gpu:
    #    torch.cuda.empty_cache()

    #pair_selector =  AllPositivePairSelector()
    
   
    for data, labels, superlabels in tqdm(trainloader):
        if use_gpu:
        
            #data, labels = data.cuda().half() , labels.cuda()
            data, labels, superlabels = data.cuda() , labels.cuda(), superlabels.cuda()
            #data = data.cuda().half()   
                    
        outputs = model(data)
        #outputs,aux = model(data)
                
        predictions = outputs.data.max(1)[1]
        total += superlabels.size(0)
        correct += (predictions == superlabels.data).sum()

        #loss = loss_func(embeddings, labels, indices_tuple)
        loss_model = class_loss(outputs, superlabels)
        opt_model.zero_grad()

        if  FEATURES[0].size(0) > 0:
            embeddings = copy.copy(FEATURES[0])
            out_feat = coder(embeddings)
                        
            indices_tuple = miner(out_feat, labels)
        
            loss_coder = coder_loss(out_feat,labels,indices_tuple)
            #print("loss:", loss_coder)
            opt_coder.zero_grad()
            
        loss_model.backward()
        loss_coder.backward()
        
        opt_model.step()
        opt_coder.step()
        
        running_loss_model += loss_model.item() * data.size(0)
        losses_model.update(loss_model.item(), labels.size(0))
        
        running_loss_coder += loss_coder.item() * data.size(0)
        losses_coder.update(loss_coder.item(), labels.size(0))
        
        
    acc = correct * 100. / total
    epoch_loss_model = running_loss_model / total
    epoch_loss_coder = running_loss_coder / total
    
    return acc,epoch_loss_model, epoch_loss_coder

if __name__ == '__main__':
    main()
