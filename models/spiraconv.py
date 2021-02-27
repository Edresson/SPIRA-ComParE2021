"""
SpiraConv models
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.generic_utils import Mish
from vit_pytorch import ViT, SpT


class SpiraConvV2(nn.Module):
    ''' Is the same than V1 but we change batchnorm to Group Norm'''
    def __init__(self, config, conv_num=4):
        super(SpiraConvV2, self).__init__()
        self.config = config
        self.audio = self.config['audio']
        self.temporal_control = self.config.dataset['temporal_control']
        self.max_seq_len = self.config.dataset['max_seq_len']
        print("Model Inicialized With Feature %s "%self.config.audio['feature'])
        if self.config.audio['feature'] == 'spectrogram':
            self.num_feature = self.config.audio['num_freq']
        elif self.config.audio['feature'] == 'melspectrogram':
            self.num_feature = self.config.audio['num_mels']
        elif self.config.audio['feature'] == 'mfcc':
            self.num_feature = self.config.audio['num_mfcc']
        else:
            self.num_feature = None
            raise ValueError('Feature %s Dont is supported'%self.config.audio['feature'])
        conv_num 
        convs = [
            # cnn1
            nn.Conv2d(1, 32, kernel_size=(7,1), dilation=(2, 1)),
            nn.GroupNorm(16, 32), Mish(), nn.MaxPool2d(kernel_size=(2,1)), nn.Dropout(p=0.7), 
            
            # cnn2
            nn.Conv2d(32, 16, kernel_size=(5, 1), dilation=(2, 1)),
            nn.GroupNorm(8, 16), Mish(),  nn.MaxPool2d(kernel_size=(2,1)), nn.Dropout(p=0.7), 

            # cnn3
            nn.Conv2d(16, 8, kernel_size=(3, 1), dilation=(2, 1)), 
            nn.GroupNorm(4, 8), Mish(),  nn.MaxPool2d(kernel_size=(2,1)), nn.Dropout(p=0.7), 
            # cnn4
            nn.Conv2d(8, 4, kernel_size=(2, 1), dilation=(1, 1)), 
            nn.GroupNorm(2, 4), Mish(), nn.Dropout(p=0.7)]

        self.conv = nn.Sequential(*convs)

        if self.temporal_control == 'padding':
            # its very useful because if you change the convlutional arquiture the model calculate its, and you dont need change this :)
            # I prefer activate the network in toy example because is more easy than calculate the conv output
            # get zeros input
            inp = torch.zeros(1, 1, self.max_seq_len, self.num_feature)
            # get out shape
            toy_activation_shape = self.conv(inp).shape
            # set fully connected input dim 
            fc1_input_dim = toy_activation_shape[1]*toy_activation_shape[2]*toy_activation_shape[3]
            self.fc1 = nn.Linear(fc1_input_dim, self.config.model['fc1_dim'])
        elif self.temporal_control == 'overlapping':
            # dinamic calculation num_feature, its useful if you use maxpooling or other pooling in feature dim, and this model dont break
            inp = torch.zeros(1, 1, 500 ,self.num_feature)
            # get out shape 
            self.fc1 = nn.Linear(4*self.conv(inp).shape[-1], self.config.model['fc1_dim'])
        elif self.temporal_control == 'avgpool': # avgpool
            pool_size = int(self.config.model['fc1_dim']/2)
            self.avg_pool = nn.AdaptiveAvgPool2d((pool_size, self.num_feature))
            # dinamic calculation num_feature, its useful if you use maxpooling or other pooling in feature dim, and this model dont break
            '''inp = torch.zeros(2, 1, 900 ,self.num_feature)
            conv = self.conv(inp)
            print(conv.shape)
            print(self.avg_pool(conv).shape)
            exit()'''
            self.fc1 = nn.Linear(4*pool_size*self.num_feature, self.config.model['fc1_dim'])

        self.mish = Mish()
        self.fc2 = nn.Linear(self.config.model['fc1_dim'], self.config.model['num_class'])
        self.dropout = nn.Dropout(p=0.7)
    def forward(self, x):
        # x: [B, T, num_feature]
        x = x.unsqueeze(1)
        # print(x.shape)
        # x: [B, 1, T, num_feature]
        x = self.conv(x)
        #print(x.shape)
        # print(x.shape)
        if self.temporal_control == 'avgpool':
            x = self.avg_pool(x)
        else:
            # x: [B, n_filters, T, num_feature]
            x = x.transpose(1, 2).contiguous()
        # x: [B, T, n_filters, num_feature]
        # print(x.shape)    
        # x: [B, T*n_filters*num_feature]
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x) # x: [B, T, num_class]
        x = self.mish(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        #print(x.shape)
        return x

class SpiraSpTv1(nn.Module):
    ''' Is the same than V1 but we change batchnorm to Group Norm'''
    def __init__(self, config):
        super(SpiraSpTv1, self).__init__()
        self.config = config
        self.audio = self.config['audio']
        self.temporal_control = self.config.dataset['temporal_control']
        self.max_seq_len = self.config.dataset['max_seq_len']
        print("Model Inicialized With Feature %s "%self.config.audio['feature'])
        if self.config.audio['feature'] == 'spectrogram':
            self.num_feature = self.config.audio['num_freq']
        elif self.config.audio['feature'] == 'melspectrogram':
            self.num_feature = self.config.audio['num_mels']
        elif self.config.audio['feature'] == 'mfcc':
            self.num_feature = self.config.audio['num_mfcc']
        else:
            self.num_feature = None
            raise ValueError('Feature %s Dont is supported'%self.config.audio['feature'])
        seq_len = int(((self.config.dataset['window_len']*self.config.audio['sample_rate'])/self.config.audio['hop_length'])+1)

        pool_type = 'mean' # mean or cls
        
        dropout_rate = 0.8
        # 61: dp 0.8, "learning_rate": 1e-3, pos_embedding_dim = 1000, transformer_mlp_dim = 50,heads=16, num_layers = 1
        # 60/62 =  "learning_rate": 1e-3,warmup_steps= 1 dropout_rate = 0.8, pos_embedding_dim = 1000, transformer_mlp_dim = 50, heads = 8
        # embedding 
        pos_embedding_dim = 1000
        emb_dropout = dropout_rate
        # transformer
        transformer_dropout = dropout_rate
        transformer_mlp_dim = 50
        heads = 8
        num_layers = 1
        num_classes = 1


        self.spt = SpT(input_dim = seq_len * self.num_feature, num_max_patches = self.config.dataset['num_max_patches'], num_classes = num_classes, dim =pos_embedding_dim, depth = num_layers, heads = heads, mlp_dim = transformer_mlp_dim, dropout = transformer_dropout, emb_dropout = emb_dropout, pool=pool_type)
        self.mish = Mish()
        # self.inp_dropout = nn.Dropout(p=0.9)
    def forward(self, x, mask=None):
        # x: [B, T, num_feature]
        # x: [B, T, n_filters, num_feature]
        # print(x.shape)
        # x = self.inp_dropout(x)
        x = self.spt(x)
        # print(x.shape)
        x = torch.sigmoid(x)
        #print(x.shape)
        return x

class SpiraVITv1(nn.Module):
    ''' Is the same than V1 but we change batchnorm to Group Norm'''
    def __init__(self, config):
        super(SpiraVITv1, self).__init__()
        self.config = config
        self.audio = self.config['audio']
        self.temporal_control = self.config.dataset['temporal_control']
        self.max_seq_len = self.config.dataset['max_seq_len']
        print("Model Inicialized With Feature %s "%self.config.audio['feature'])
        if self.config.audio['feature'] == 'spectrogram':
            self.num_feature = self.config.audio['num_freq']
        elif self.config.audio['feature'] == 'melspectrogram':
            self.num_feature = self.config.audio['num_mels']
        elif self.config.audio['feature'] == 'mfcc':
            self.num_feature = self.config.audio['num_mfcc']
        else:
            self.num_feature = None
            raise ValueError('Feature %s Dont is supported'%self.config.audio['feature'])
        
        conv_channels = self.num_feature*2
        num_channels = 3
        convs = [
            # cnn1
            nn.Conv2d(1, conv_channels, kernel_size=(7,1), dilation=(2, 1)),  
            Mish(),
            nn.Conv2d(conv_channels, num_channels, kernel_size=(7,1), dilation=(2, 1)),
            Mish()
            ]

        self.conv = nn.Sequential(*convs)
        pool_size = 128
        self.avg_pool = nn.AdaptiveAvgPool2d(pool_size)
        self.vit = v = ViT(image_size = pool_size, patch_size = int(pool_size/8), num_classes = 1, dim = int(pool_size*4), depth = 2, heads = 16, mlp_dim = int(pool_size*8), dropout = 0.1, emb_dropout = 0.1, channels = num_channels)
        self.mish = Mish()
    def forward(self, x):
        # x: [B, T, num_feature]
        x = x.unsqueeze(1)
        # print(x.shape)
        # x: [B, 1, T, num_feature]
        x = self.conv(x)
        # print(x.shape)
        # print(x.shape)
        x = self.avg_pool(x)
        # x: [B, T, n_filters, num_feature]
        # print(x.shape)    
        # x: [B, T*n_filters*num_feature]
        x = self.vit(x)
        # print(x.shape)
        x = torch.sigmoid(x)
        #print(x.shape)
        return x

class SpiraVITv2(nn.Module):
    ''' Is the same than V1 but we change batchnorm to Group Norm'''
    def __init__(self, config):
        super(SpiraVITv2, self).__init__()
        self.config = config
        self.audio = self.config['audio']
        self.temporal_control = self.config.dataset['temporal_control']
        self.max_seq_len = self.config.dataset['max_seq_len']
        print("Model Inicialized With Feature %s "%self.config.audio['feature'])
        if self.config.audio['feature'] == 'spectrogram':
            self.num_feature = self.config.audio['num_freq']
        elif self.config.audio['feature'] == 'melspectrogram':
            self.num_feature = self.config.audio['num_mels']
        elif self.config.audio['feature'] == 'mfcc':
            self.num_feature = self.config.audio['num_mfcc']
        else:
            self.num_feature = None
            raise ValueError('Feature %s Dont is supported'%self.config.audio['feature'])

        pool_size = 128
        self.avg_pool = nn.AdaptiveAvgPool2d(pool_size)
        self.vit = v = ViT(image_size = pool_size, patch_size = int(pool_size/8), num_classes = self.config.model['num_class'], dim = int(pool_size*4), depth = 2, heads = 16, mlp_dim = int(pool_size*8), dropout = 0.1, emb_dropout = 0.1, channels = self.num_feature)
        self.mish = Mish()
    def forward(self, x):
        # x: [B, T, num_feature]
        x = x.unsqueeze(1)
        x = x.transpose(1, 3)
        # print(x.shape)
        # x: [B, 1, T, num_feature]
        # x = self.conv(x)
        # print(x.shape)
        # print(x.shape)
        x = self.avg_pool(x)
        # x: [B, T, n_filters, num_feature]
        # print(x.shape)    
        # x: [B, T*n_filters*num_feature]
        x = self.vit(x)
        # print(x.shape)
        x = torch.sigmoid(x)
        #print(x.shape)
        return x

class SpiraConvV1(nn.Module):
    def __init__(self, config):
        super(SpiraConvV1, self).__init__()
        self.config = config
        self.audio = self.config['audio']
        self.padding_with_max_lenght = self.config.dataset['padding_with_max_lenght']
        self.max_seq_len = self.config.dataset['max_seq_len']
        print("Model Inicialized With Feature %s "%self.config.audio['feature'])
        if self.config.audio['feature'] == 'spectrogram':
            self.num_feature = self.config.audio['num_freq']
        elif self.config.audio['feature'] == 'melspectrogram':
            self.num_feature = self.config.audio['num_mels']
        elif self.config.audio['feature'] == 'mfcc':
            self.num_feature = self.config.audio['num_mfcc']
        else:
            self.num_feature = None
            raise ValueError('Feature %s Dont is supported'%self.config.audio['feature'])
        convs = [
            # cnn1
            nn.Conv2d(1, 32, kernel_size=(7,1), dilation=(2, 1)),
            nn.BatchNorm2d(32), Mish(), nn.MaxPool2d(kernel_size=(2,1)), nn.Dropout(p=0.7), 
            
            # cnn2
            nn.Conv2d(32, 16, kernel_size=(5, 1), dilation=(2, 1)),
            nn.BatchNorm2d(16), Mish(),  nn.MaxPool2d(kernel_size=(2,1)), nn.Dropout(p=0.7), 

            # cnn3
            nn.Conv2d(16, 8, kernel_size=(3, 1), dilation=(2, 1)), 
            nn.BatchNorm2d(8), Mish(),  nn.MaxPool2d(kernel_size=(2,1)), nn.Dropout(p=0.7), 
            # cnn4
            nn.Conv2d(8, 4, kernel_size=(2, 1), dilation=(1, 1)), 
            nn.BatchNorm2d(4), Mish(), nn.Dropout(p=0.7)]

        self.conv = nn.Sequential(*convs)

        if self.padding_with_max_lenght:
            # its very useful because if you change the convlutional arquiture the model calculate its, and you dont need change this :)
            # I prefer aactivate the network in toy example because is more easy than calculate the conv output
            # get zeros input
            inp = torch.zeros(1, 1, self.max_seq_len, self.num_feature)
            # get out shape
            toy_activation_shape = self.conv(inp).shape
            # set fully connected input dim 
            fc1_input_dim = toy_activation_shape[1]*toy_activation_shape[2]*toy_activation_shape[3]
            self.fc1 = nn.Linear(fc1_input_dim, self.config.model['fc1_dim'])
        else:
            # dinamic calculation num_feature, its useful if you use maxpooling or other pooling in feature dim, and this model dont break
            inp = torch.zeros(1, 1, 500 ,self.num_feature)
            # get out shape 
            self.fc1 = nn.Linear(4*self.conv(inp).shape[-1], self.config.model['fc1_dim'])
        self.mish = Mish()
        self.fc2 = nn.Linear(self.config.model['fc1_dim'], self.config.model['num_class'])
        self.dropout = nn.Dropout(p=0.7)
    def forward(self, x):
        # x: [B, T, num_feature]
        x = x.unsqueeze(1)
        #print(x.shape)
        # x: [B, 1, T, num_feature]
        x = self.conv(x)
        #print(x.shape)
        # x: [B, n_filters, T, num_feature]
        x = x.transpose(1, 2).contiguous()
        # x: [B, T, n_filters, num_feature]
        #print(x.shape)
        if self.padding_with_max_lenght:
             # x: [B, T*n_filters*num_feature]
            x = x.view(x.size(0), -1)
        else:
             # x: [B, T, n_filters*num_feature]
            x = x.view(x.size(0), x.size(1), -1)
    
       
        #print(x.shape)
        x = self.fc1(x) # x: [B, T, num_class]
        #print(x.shape)
        x = self.mish(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        #print(x.shape)
        return x