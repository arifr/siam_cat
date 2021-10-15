import torch
#import torchvision
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from torchvision import transforms #, datasets
from PIL import Image
#from sampler import BalancedBatchSampler
from pytorch_metric_learning import samplers
import itertools
import numpy as np
from math import ceil



class MNIST(object):
    def __init__(self, batch_size, use_gpu, num_workers):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        pin_memory = True if use_gpu else False

        trainset = torchvision.datasets.MNIST(root='./data/mnist', train=True, download=True, transform=transform)
        
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        
        testset = torchvision.datasets.MNIST(root='./data/mnist', train=False, download=True, transform=transform)
        
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        self.trainloader = trainloader
        self.testloader = testloader
        self.num_classes = 10



class FMNIST(object):
    def __init__(self, batch_size, use_gpu, num_workers,sample):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        pin_memory = True if use_gpu else False

        trainset = torchvision.datasets.FashionMNIST(root='./data/fmnist', train=True, download=True, transform=transform)
        
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        
        testset = torchvision.datasets.FashionMNIST(root='./data/fmnist', train=False, download=True, transform=transform)
        
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        self.trainloader = trainloader
        self.testloader = testloader
        self.num_classes = 10





'''
__factory = {
    'mnist': MNIST,
}
'''


class StanfordProductsTrain(Dataset) :
    def __init__(self,root,transform,train=True):
        if train:
            info_path = "/Ebay_train.txt"
        else:
            info_path = "/Ebay_test.txt"
        files = pd.read_csv(root+info_path, header=0, delimiter=' ',usecols=['path','super_class_id'])[['path','super_class_id']]
        self.data = files.to_dict(orient='record')
        self.transform = transform
        self.root = root
    def __getitem__(self,index):
        image = Image.open(self.root +'/'+ self.data[index]['path'])
        if (image.mode != 'RGB'):
            image = image.convert('RGB')

        trans = self.transform(image)
        return  trans, self.data[index]['super_class_id']-1
        
    def __len__(self):
        return len(self.data)





class SOProducts(object):
    def __init__(self, batch_size, use_gpu, num_workers):
        
        pin_memory = True if use_gpu else False

        '''
        img_transform_train = transforms.Compose([transforms.RandomResizedCrop(size=227),
                                        transforms.RandomHorizontalFlip(0.5),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        '''

        img_transform_train = transforms.Compose([transforms.RandomResizedCrop(size=299),
                                                transforms.RandomHorizontalFlip(0.5),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        
            
        #img_transform_test = transforms.Compose([transforms.Resize(256),
        #                                transforms.CenterCrop(227),
        img_transform_test = transforms.Compose([transforms.Resize(299),
                                                transforms.CenterCrop(229),   
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
            
        #img_transform_test = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(227),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
        #root = '/media/arif/data/datasets/Stanford_Online_Products'
        root = '/home/m405305/dataset/Stanford_Online_Products'
        #root = '/work/uah001/datasets/Stanford_Online_Products'
    
        #image_path = 'images'
        train_dataset = StanfordProductsTrain(root,transform=img_transform_train,train=True)
        test_dataset = StanfordProductsTrain(root,transform=img_transform_test,train=False)
    
        #if sample :
        #    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, num_workers=num_workers, sampler = BalancedBatchSampler(train_dataset),pin_memory=pin_memory)
        #    testloader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, num_workers=num_workers,sampler = BalancedBatchSampler(test_dataset) ,pin_memory=pin_memory)
        #else :        
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, num_workers=num_workers,shuffle=True,pin_memory=pin_memory)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, num_workers=num_workers,shuffle=True,pin_memory=pin_memory)
        
        
        self.trainloader = trainloader
        self.testloader = testloader
        self.num_classes = 12




class StanfordProductsClass(Dataset) :
    def __init__(self,root,transform,train=True):
        if train:
            info_path = "/Ebay_train.txt"
        else:
            info_path = "/Ebay_test.txt"
        files = pd.read_csv(root+info_path, header=0, delimiter=' ',usecols=['path','class_id','super_class_id'])[['path','class_id','super_class_id']]
        self.data = files.to_dict(orient='record')
        self.transform = transform
        self.root = root
    def __getitem__(self,index):
        #print("index type:", type(index) )
        #print("index col:", len(index[0]))
                
        image = Image.open(self.root +'/'+ self.data[index]['path'])
        if (image.mode != 'RGB'):
            image = image.convert('RGB')

        trans = self.transform(image)
        return  trans, self.data[index]['class_id']-1, self.data[index]['super_class_id']-1
        
    def __len__(self):
        return len(self.data)




class SOPClass(object):
    def __init__(self, batch_size, use_gpu, sample, num_workers):
        
        pin_memory = True if use_gpu else False

        '''
        img_transform_train = transforms.Compose([transforms.RandomResizedCrop(size=227),
                                        transforms.RandomHorizontalFlip(0.5),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        '''

        img_transform_train = transforms.Compose([transforms.RandomResizedCrop(size=299),
                                                transforms.RandomHorizontalFlip(0.5),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        
            
        #img_transform_test = transforms.Compose([transforms.Resize(256),
        #                                transforms.CenterCrop(227),
        img_transform_test = transforms.Compose([transforms.Resize(299),
                                                transforms.CenterCrop(229),   
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
            
        #img_transform_test = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(227),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
        #root = '/media/arif/data/datasets/Stanford_Online_Products'
        root = '/home/m405305/dataset/Stanford_Online_Products'
        #root = '/work/uah001/datasets/Stanford_Online_Products'
    
        #image_path = 'images'
        train_dataset = StanfordProductsClass(root,transform=img_transform_train,train=True)
        test_dataset = StanfordProductsClass(root,transform=img_transform_test,train=False)

        print("create labels..")

        '''
        train_labels = torch.Tensor(len(train_dataset),2)
        for idx,(_,label,superlabel) in enumerate(train_dataset) :
            train_labels[idx,0] =  label
            train_labels[idx,1] =  superlabel
        '''
        #train_labels = [ (label,superlabel) for img,label,superlabel in train_dataset]
        #train_labels = np.load('train_labels.npy').tolist()
        #train_labels = np.load('train_labels.npy')
        #train_labels = np.expand_dims(train_labels,axis=0) 
        #print(train_labels.shape)
        #print(len(train_labels))
        #print(train_labels[1000][0])
        #print(train_labels[1000][1])
        #print(type(train_labels[1000][0]))
        #print(type(train_labels[1000][1]))
                                 
                                
        
        #np.save('train_labels.npy',np.array(train_labels) )
        #train_labels = [label,superlabel for label,superlabel in train_dataset] 
        '''
        test_labels = torch.Tensor(len(test_dataset),2)
        for idx,(_,label,superlabel) in enumerate(test_dataset) :
            test_labels[idx,0] =  label
            test_labels[idx,1] =  superlabel
        '''
        #test_labels = np.load('test_labels.npy').tolist()
        #test_labels = np.load('test_labels.npy')
        #test_labels = np.expand_dims(test_labels,axis=0) 
                                                            
        #test_labels = [(label,superlabel) for img,label,superlabel in test_dataset]
        #np.save('test_labels.npy',np.array(test_labels) )
                 


        
        if sample :
            total_train_batch = ceil(len(train_dataset)/batch_size)
            print("sampling..")
                            
            train_sampler= samplers.HierarchicalSampler(
            labels= train_labels, 
            batch_size=batch_size,
            samples_per_class='all',
            batches_per_super_tuple=total_train_batch,
            super_classes_per_batch=12,
            inner_label=0,
            outer_label=1,
            )

            total_test_batch = ceil(len(test_dataset)/batch_size)
            
            test_sampler= samplers.HierarchicalSampler(
            labels= test_labels, 
            batch_size=batch_size,
            samples_per_class='all',
            batches_per_super_tuple=total_test_batch,
            super_classes_per_batch=12,
            inner_label=0,
            outer_label=1,
            )

            print("done")
                    
            
            trainloader = DataLoader(train_dataset, num_workers=num_workers, batch_sampler = train_sampler,pin_memory=pin_memory)
            testloader = DataLoader(test_dataset, num_workers=num_workers,batch_sampler = test_sampler,pin_memory=pin_memory)
        else :    
        
        #else :        
            trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, num_workers=num_workers,shuffle=True,pin_memory=pin_memory)
            testloader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, num_workers=num_workers,shuffle=True,pin_memory=pin_memory)
        
        
        self.trainloader = trainloader
        self.testloader = testloader
        self.num_classes = 12



class StanfordProductsTest(Dataset) :
    def __init__(self,root,transform,train=True):
        if train:
            info_path = "/Ebay_train.txt"
        else:
            info_path = "/Ebay_test.txt"
        files = pd.read_csv(root+info_path, header=0, delimiter=' ',usecols=['path','class_id'])[['path','class_id']]
        self.data = files.to_dict(orient='record')
        self.transform = transform
        self.root = root
    def __getitem__(self,index):
        image = Image.open(self.root +'/'+ self.data[index]['path'])
        if (image.mode != 'RGB'):
            image = image.convert('RGB')

        trans = self.transform(image)
        return  trans, self.data[index]['class_id']-1
        
    def __len__(self):
        return len(self.data)



class SOProductsTest(object):
    def __init__(self, batch_size, use_gpu, num_workers, sample):
        
        pin_memory = True if use_gpu else False

        img_transform_train = transforms.Compose([transforms.RandomResizedCrop(size=227),
                                        transforms.RandomHorizontalFlip(0.5),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    
        img_transform_test = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(227),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    
        #root = '/media/arif/data/datasets/Stanford_Online_Products'
        root = '/home/m405305/dataset/Stanford_Online_Products'
        #root = '/work/uah001/datasets/Stanford_Online_Products'
    
        #image_path = 'images'
        train_dataset = StanfordProductsTest(root,transform=img_transform_train,train=True)
        test_dataset = StanfordProductsTest(root,transform=img_transform_test,train=False)
    
        trainloader = DataLoader(train_dataset, batch_size = batch_size, num_workers=num_workers,shuffle=True,pin_memory=pin_memory)
        testloader = DataLoader(test_dataset, batch_size = batch_size, num_workers=num_workers,shuffle=True,pin_memory=pin_memory)
        
        self.trainloader = trainloader
        self.testloader = testloader
        self.num_classes = 12


class InShopTrain(Dataset) :

    def __init__(self,root,transform,train=True):

        self.root = root
        files = pd.read_csv(root+'/Eval/list_eval_partition.csv', header=0, delimiter=';')[['image_name','class_id','evaluation_status']]

    
        if train:
            str_query = "evaluation_status == 'train'"
        else:
            str_query = "evaluation_status == 'query'"


        #print(files.to_dict(orient='records'))
        

        self.data = files.query(str_query).to_dict(orient='record')
        #for dt in self.data :
        #    dt['item_id'] = int(dt['item_id'][3:].strip('0'))
        self.transform = transform

        #def
    def __getitem__(self,index):
        image = Image.open(self.root + '/'+ self.data[index]['image_name'])
        #print(self.root + '/'+ self.data[index]['image_name'])
        #image.show()
        #print (self.data[index])
        #trans = transforms.ToTensor()
        #image = trans(image)
        #return  self.transform(image), self.data[index]['item_id']
        return  self.transform(image), self.data[index]['class_id']-1
        
        #return  self.transform(image), self.data[index]['class_id']
        #{'image':im, 'target':self.data[index]['class_id']}


    def __len__(self):
        return len(self.data)


class InShop(object):
    def __init__(self, batch_size, use_gpu, num_workers):
        
        pin_memory = True if use_gpu else False

        #img_transform_train = transforms.Compose([transforms.RandomResizedCrop(size=227),
        img_transform_train = transforms.Compose([transforms.RandomResizedCrop(size=299), 
                                        transforms.RandomHorizontalFlip(0.5),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    
        img_transform_test = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(227),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    
        #root = '/media/arif/data/datasets/Stanford_Online_Products'
        root = '/home/m405305/dataset/inshop'
        #root = '/work/uah001/datasets/Stanford_Online_Products'
    
        #image_path = 'images'
        train_dataset = InShopTrain(root,transform=img_transform_train,train=True)
        test_dataset = InShopTrain(root,transform=img_transform_test,train=False)
    
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, num_workers=num_workers,shuffle=True,pin_memory=pin_memory)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, num_workers=num_workers,shuffle=True,pin_memory=pin_memory)
        #trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, num_workers=num_workers, sampler = BalancedBatchSampler(train_dataset),pin_memory=pin_memory)
        #testloader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, num_workers=num_workers,sampler = BalancedBatchSampler(test_dataset) ,pin_memory=pin_memory)
    
        
        self.trainloader = trainloader
        self.testloader = testloader
        self.num_classes = 23



class InShopDataTest(Dataset) :

    def __init__(self,root,transform,train=True):

        self.root = root
        files = pd.read_csv(root+'/Eval/list_eval_partition.csv', header=0, delimiter=';')[['image_name','item_id','evaluation_status']]

    
        if train:
            str_query = "evaluation_status == 'train'"
        else:
            str_query = "evaluation_status == 'query'"


        #print(files.to_dict(orient='records'))
        

        self.data = files.query(str_query).to_dict(orient='record')
        for dt in self.data :
            dt['item_id'] = int(dt['item_id'][3:].strip('0'))
        self.transform = transform

        #def
    def __getitem__(self,index):
        image = Image.open(self.root + '/'+ self.data[index]['image_name'])
        #print(self.root + '/'+ self.data[index]['image_name'])
        #image.show()
        #print (self.data[index])
        #trans = transforms.ToTensor()
        #image = trans(image)
        return  self.transform(image), self.data[index]['item_id']
        #return  self.transform(image), self.data[index]['class_id']-1
        
        #return  self.transform(image), self.data[index]['class_id']
        #{'image':im, 'target':self.data[index]['class_id']}


    def __len__(self):
        return len(self.data)


class InShopTest(object):
    def __init__(self, batch_size, use_gpu, num_workers): #, sample):
        
        pin_memory = True if use_gpu else False

        img_transform_train = transforms.Compose([transforms.RandomResizedCrop(size=227),
                                        transforms.RandomHorizontalFlip(0.5),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    
        img_transform_test = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(227),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    
        #root = '/media/arif/data/datasets/Stanford_Online_Products'
        root = '/home/m405305/dataset/inshop'
        #root = '/work/uah001/datasets/Stanford_Online_Products'
    
        #image_path = 'images'
        train_dataset = InShopDataTest(root,transform=img_transform_train,train=True)
        test_dataset = InShopDataTest(root,transform=img_transform_test,train=False)
    
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, num_workers=num_workers,shuffle=True,pin_memory=pin_memory)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, num_workers=num_workers,shuffle=True,pin_memory=pin_memory)
        #trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, num_workers=num_workers, sampler = BalancedBatchSampler(train_dataset),pin_memory=pin_memory)
        #testloader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, num_workers=num_workers,sampler = BalancedBatchSampler(test_dataset) ,pin_memory=pin_memory)
    
        
        self.trainloader = trainloader
        self.testloader = testloader
        self.num_classes = 23



class CustomerToShop(object):
    def __init__(self, batch_size, use_gpu, num_workers):
        
        pin_memory = True if use_gpu else False

        #img_transform_train = transforms.Compose([transforms.RandomResizedCrop(size=227),
        img_transform_train = transforms.Compose([transforms.RandomResizedCrop(size=299), 
                                        transforms.RandomHorizontalFlip(0.5),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    
        img_transform_test = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(227),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    
        #root = '/media/arif/data/datasets/Stanford_Online_Products'
        root = '/home/m405305/dataset/cust-shop'
        #root = '/work/uah001/datasets/Stanford_Online_Products'
    
        #image_path = 'images'
        train_dataset = CustomerToShopTrain(root,transform=img_transform_train,train=True)
        test_dataset = CustomerToShopTrain(root,transform=img_transform_test,train=False)
    
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, num_workers=num_workers,shuffle=True,pin_memory=pin_memory)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, num_workers=num_workers,shuffle=True,pin_memory=pin_memory)
        #trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, num_workers=num_workers, sampler = BalancedBatchSampler(train_dataset),pin_memory=pin_memory)
        #testloader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, num_workers=num_workers,sampler = BalancedBatchSampler(test_dataset) ,pin_memory=pin_memory)
    
        
        self.trainloader = trainloader
        self.testloader = testloader
        self.num_classes = 23



class CustomerToShopTrain(Dataset) :

    def __init__(self,root,transform,train=True):
        self.root = root
        
        files = pd.read_csv(self.root+'/Eval/list_eval_partition.csv', header=0, delimiter=';')[['image_path','class_id','evaluation_status']]

        ##image_name	item_id	evaluation_status

        if train:
            str_query = "evaluation_status == 'train'"
        else:
            str_query = "evaluation_status == 'test' " #or evaluation_status == 'val' "


        #print(files.to_dict(orient='records'))
        #print (files.to_dict(orient='record'))

        self.data = files.query(str_query).to_dict(orient='record')
        #self.image_path = image_path
        #for dt in self.data :
        #    dt['item_id'] = int(dt['item_id'][3:].strip('0'))

        self.transform = transform
        #print(type(self.data['item_id']))
        #print(len(self.data))

        
        #def
    def __getitem__(self,index):
        image = Image.open(self.root + '/'+ self.data[index]['image_path'])
        #image.show()
        #print (self.data[index])
        if (image.mode != 'RGB'):
            image = image.convert('RGB')
        trans = self.transform(image)
        #image = trans(image)
        
        #print('from get: \n') 
        #print(type(itemid))
        
                
        return  trans, self.data[index]['class_id']-1

        #return  self.transform(image), self.data[index]['class_id']
        #{'image':im, 'target':self.data[index]['class_id']}

    def __len__(self):
        return len(self.data)



class CustomerToShopTest(object):
    def __init__(self, batch_size, use_gpu, num_workers):
        
        pin_memory = True if use_gpu else False

        #img_transform_train = transforms.Compose([transforms.RandomResizedCrop(size=227),
        img_transform_train = transforms.Compose([transforms.RandomResizedCrop(size=299), 
                                        transforms.RandomHorizontalFlip(0.5),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    
        img_transform_test = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(227),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    
        #root = '/media/arif/data/datasets/Stanford_Online_Products'
        root = '/home/m405305/dataset/cust-shop'
        #root = '/work/uah001/datasets/Stanford_Online_Products'
    
        #image_path = 'images'
        train_dataset = CustomerToShopDataTest(root,transform=img_transform_train,train=True)
        test_dataset = CustomerToShopDataTest(root,transform=img_transform_test,train=False)
    
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, num_workers=num_workers,shuffle=True,pin_memory=pin_memory)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, num_workers=num_workers,shuffle=True,pin_memory=pin_memory)
        #trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, num_workers=num_workers, sampler = BalancedBatchSampler(train_dataset),pin_memory=pin_memory)
        #testloader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, num_workers=num_workers,sampler = BalancedBatchSampler(test_dataset) ,pin_memory=pin_memory)
    
        
        self.trainloader = trainloader
        self.testloader = testloader
        self.num_classes = 23



class CustomerToShopDataTest(Dataset) :

    def __init__(self,root,transform,train=True):
        self.root = root
        
        files = pd.read_csv(self.root+'/Eval/list_eval_partition.csv', header=0, delimiter=';')[['image_path','item_id','evaluation_status']]

        ##image_name	item_id	evaluation_status

        if train:
            str_query = "evaluation_status == 'train'"
        else:
            str_query = "evaluation_status == 'test' " #or evaluation_status == 'val' "


        #print(files.to_dict(orient='records'))
        #print (files.to_dict(orient='record'))

        self.data = files.query(str_query).to_dict(orient='record')
        #self.image_path = image_path
        for dt in self.data :
            dt['item_id'] = int(dt['item_id'][3:].strip('0'))

        self.transform = transform
        #print(type(self.data['item_id']))
        #print(len(self.data))

        
        #def
    def __getitem__(self,index):
        image = Image.open(self.root + '/'+ self.data[index]['image_path'])
        #image.show()
        #print (self.data[index])
        if (image.mode != 'RGB'):
            image = image.convert('RGB')
        trans = self.transform(image)
        #image = trans(image)
        
        #print('from get: \n') 
        #print(type(itemid))
        
                
        return  trans, self.data[index]['item_id']

        #return  self.transform(image), self.data[index]['class_id']
        #{'image':im, 'target':self.data[index]['class_id']}

    def __len__(self):
        return len(self.data)







class CustomerToShopClass(object):
    def __init__(self, batch_size, use_gpu, sample, num_workers):
        
        pin_memory = True if use_gpu else False

        #img_transform_train = transforms.Compose([transforms.RandomResizedCrop(size=227),
        img_transform_train = transforms.Compose([transforms.RandomResizedCrop(size=299), 
                                        transforms.RandomHorizontalFlip(0.5),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    
        img_transform_test = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(227),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    
        #root = '/media/arif/data/datasets/Stanford_Online_Products'
        root = '/home/m405305/dataset/cust-shop'
        #root = '/work/uah001/datasets/Stanford_Online_Products'
    
        #image_path = 'images'
        train_dataset = CustomerToShopDataClass(root,transform=img_transform_train,train=True)
        test_dataset = CustomerToShopDataClass(root,transform=img_transform_test,train=False)
    
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, num_workers=num_workers,shuffle=True,pin_memory=pin_memory)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, num_workers=num_workers,shuffle=True,pin_memory=pin_memory)
        #trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, num_workers=num_workers, sampler = BalancedBatchSampler(train_dataset),pin_memory=pin_memory)
        #testloader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, num_workers=num_workers,sampler = BalancedBatchSampler(test_dataset) ,pin_memory=pin_memory)
    
        
        self.trainloader = trainloader
        self.testloader = testloader
        self.num_classes = 23



class CustomerToShopDataClass(Dataset) :

    def __init__(self,root,transform,train=True):
        self.root = root
        
        files = pd.read_csv(self.root+'/Eval/list_eval_partition.csv', header=0, delimiter=';')[['image_path','item_id','class_id','evaluation_status']]

        ##image_name	item_id	evaluation_status

        if train:
            str_query = "evaluation_status == 'train'"
        else:
            str_query = "evaluation_status == 'test' " #or evaluation_status == 'val' "


        #print(files.to_dict(orient='records'))
        #print (files.to_dict(orient='record'))

        self.data = files.query(str_query).to_dict(orient='record')
        #self.image_path = image_path
        for dt in self.data :
            dt['item_id'] = int(dt['item_id'][3:].strip('0'))

        self.transform = transform
        #print(type(self.data['item_id']))
        #print(len(self.data))

        
        #def
    def __getitem__(self,index):
        image = Image.open(self.root + '/'+ self.data[index]['image_path'])
        #image.show()
        #print (self.data[index])
        if (image.mode != 'RGB'):
            image = image.convert('RGB')
        trans = self.transform(image)
        #image = trans(image)
        
        #print('from get: \n') 
        #print(type(itemid))
        
                
        return  trans, self.data[index]['item_id']-1, self.data[index]['class_id']-1

        #return  self.transform(image), self.data[index]['class_id']
        #{'image':im, 'target':self.data[index]['class_id']}

    def __len__(self):
        return len(self.data)



__factory = {
    'mnist': MNIST,
    'fmnist': FMNIST,
    'SOP': SOProducts,
    'SOPTest': SOProductsTest,
    'SOPClass': SOPClass,
    'inshop':InShop, 
    'inshopTest' : InShopTest,
    'custShop' : CustomerToShop,
    'custShopTest' : CustomerToShopTest,
    'custShopClass' : CustomerToShopClass,
}



def create(name, batch_size, use_gpu, sample, num_workers):
    if name not in __factory.keys():
        raise KeyError("Unknown dataset: {}".format(name))
    return __factory[name](batch_size, use_gpu,sample, num_workers)
