#import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models

class ResNet18(nn.Module):
    
    def __init__(self, num_classes):
        
        super(ResNet18, self).__init__()
        self.num_classes = num_classes
        self.base_model = models.resnet18(pretrained=True)
        self.feature_dim = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(self.feature_dim,self.num_classes)
        self.hook_layer = self.base_model.fc         
        self.name = 'ResNet18' 

    def forward(self,x):
        x = self.base_model(x)
        return x


class MobileNet(nn.Module):
    def __init__(self, num_classes):
        super(MobileNet, self).__init__()
        self.num_classes = num_classes
        self.base_model = models.mobilenet_v3_large(pretrained=True)
        #self.base_model = models.mobilenet_v2(pretrained=True)
               
        self.feature_dim = self.base_model.classifier[-1].in_features
        self.base_model.classifier[-1] = nn.Linear(self.feature_dim,self.num_classes)    
        #self.hook_layer = self.base_model.classifier[3]       
        self.name = 'MobileNetV3' 

    def forward(self,x):
        x = self.base_model(x)
        return x
    def set_identity(self):
        self.base_model.classifier[-1] = nn.Identity()
    def set_hook(self,hook_fn):
        self.base_model.classifier[-1].register_forward_hook(hook_fn)

__factory = {
    'resnet18': ResNet18,
    'mobilenet': MobileNet,
}

def create(name, num_classes):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))
    return __factory[name](num_classes)

if __name__ == '__main__':
    pass
