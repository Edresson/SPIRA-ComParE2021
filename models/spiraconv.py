"""
SpiraConv models
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.generic_utils import Mish
from vit_pytorch import ViT, SpT

from utils.generic_utils import SpecAugmentation, do_mixup, set_init_dict


class SpiraConvV4(nn.Module):
    ''' Is the same than V2 but is parametrized'''
    def __init__(self, config, kernel_sizes = [(7,1), (5, 1), (3, 1), (2, 1)],  dilation = (2, 1), dropout_rate = 0.7):
        super(SpiraConvV4, self).__init__()
        self.config = config
        self.audio = self.config['audio']
        self.temporal_control = self.config.dataset['temporal_control']
        self.max_seq_len = self.config.dataset['max_seq_len']

        if 'kernel_sizes' in self.config.model:
            kernel_sizes = self.config.model['kernel_sizes']
        if 'dilation' in self.config.model:
            dilation = self.config.model['dilation']
        if 'dropout_rate' in self.config.model:
            dropout_rate = self.config.model['dropout_rate']

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
            nn.Conv2d(1, 32, kernel_size=kernel_sizes[0], dilation=dilation),
            nn.GroupNorm(16, 32), Mish(), nn.MaxPool2d(kernel_size=(2,1)), nn.Dropout(p=dropout_rate), 
            
            # cnn2
            nn.Conv2d(32, 16, kernel_size=kernel_sizes[1], dilation=dilation),
            nn.GroupNorm(8, 16), Mish(), nn.MaxPool2d(kernel_size=(2,1)), nn.Dropout(p=dropout_rate), 

            # cnn3
            nn.Conv2d(16, 8, kernel_size=kernel_sizes[2], dilation=dilation), 
            nn.GroupNorm(4, 8), Mish(),  nn.MaxPool2d(kernel_size=(2,1)), nn.Dropout(p=dropout_rate), 
            # cnn4
            nn.Conv2d(8, 4, kernel_size=kernel_sizes[3], dilation=(1, 1)), 
            nn.GroupNorm(2, 4), Mish(), nn.Dropout(p=dropout_rate),
            ]

        self.conv = nn.Sequential(*convs)

        if self.temporal_control == 'padding' or self.temporal_control == 'overlapping' or self.temporal_control == 'one_window':
            # its very useful because if you change the convlutional arquiture the model calculate its, and you dont need change this :)
            # I prefer activate the network in toy example because is more easy than calculate the conv output
            # get zeros input
            print(self.max_seq_len)
            inp = torch.zeros(1, 1, self.max_seq_len, self.num_feature)
            # get out shape
            toy_activation_shape = self.conv(inp).shape
            # set fully connected input dim 
            fc1_input_dim = toy_activation_shape[1]*toy_activation_shape[2]*toy_activation_shape[3]
            self.fc1 = nn.Linear(fc1_input_dim, self.config.model['fc1_dim'])
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
        self.dropout = nn.Dropout(p=dropout_rate)
    def forward(self, x, mixup_lambda=None):
        # x: [B, T, num_feature]
        x = x.unsqueeze(1)


        # Mixup on feature
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)


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


class SpiraConvAttnV1(nn.Module):
    ''' Is the same than V1 but we change batchnorm to Group Norm'''
    def __init__(self, config, conv_num=4):
        super(SpiraConvAttnV1, self).__init__()
        self.config = config

        num_heads = 1
        dp_rate = 0.8

     
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

        convs = [
            # cnn1
            nn.Conv2d(1, 32, kernel_size=(7,1), dilation=(1, 1), padding=(7 - 1)//2),
            nn.GroupNorm(16, 32), Mish(), nn.Dropout(p=dp_rate), 
            
            # cnn2
            nn.Conv2d(32, 16, kernel_size=(5, 1), dilation=(1, 1), padding=(5 - 1)//2),
            nn.GroupNorm(8, 16), Mish(), nn.Dropout(p=dp_rate), 

            # cnn3
            nn.Conv2d(16, 8, kernel_size=(3, 1), dilation=(1, 1), padding=(3 - 1)//2), 
            nn.GroupNorm(4, 8), Mish(), nn.Dropout(p=dp_rate), 
            # cnn4
            nn.Conv2d(8, 4, kernel_size=(1, 1), dilation=(1, 1), padding=(1 - 1)//2), 
            nn.GroupNorm(2, 4), Mish(), nn.Dropout(p=dp_rate)]
        '''convs = [
            # cnn1
            nn.Conv2d(1, 32, kernel_size=(7,1), dilation=(2, 1)),
            nn.GroupNorm(16, 32), Mish(), nn.MaxPool2d(kernel_size=(2,1)), nn.Dropout(p=dp_rate), 
            
            # cnn2
            nn.Conv2d(32, 16, kernel_size=(5, 1), dilation=(2, 1)),
            nn.GroupNorm(8, 16), Mish(),  nn.MaxPool2d(kernel_size=(2,1)), nn.Dropout(p=dp_rate), 

            # cnn3
            nn.Conv2d(16, 8, kernel_size=(3, 1), dilation=(2, 1)), 
            nn.GroupNorm(4, 8), Mish(),  nn.MaxPool2d(kernel_size=(2,1)), nn.Dropout(p=dp_rate), 
            # cnn4
            nn.Conv2d(8, 4, kernel_size=(2, 1), dilation=(1, 1)), 
            nn.GroupNorm(2, 4), Mish(), nn.Dropout(p=dp_rate)]'''
        self.conv = nn.Sequential(*convs)
        
        if self.temporal_control == 'padding' or self.temporal_control == 'overlapping':
            # its very useful because if you change the convlutional arquiture the model calculate its, and you dont need change this :)
            # I prefer activate the network in toy example because is more easy than calculate the conv output
            # get zeros input
            inp = torch.zeros(1, 1, self.max_seq_len, self.num_feature)
            # get out shape
            # print(inp.shape)
            conv_out = self.conv(inp)
            conv_out = conv_out.transpose(1, 2)
            
            toy_activation_shape = conv_out.shape

            attn_dim = toy_activation_shape[2]*toy_activation_shape[3]

            self.attn = torch.nn.MultiheadAttention(attn_dim, num_heads, dropout=dp_rate)

            self.fc1 = nn.Linear(toy_activation_shape[1]*attn_dim, self.config.model['fc1_dim'])          
            
        
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
        self.dropout = nn.Dropout(p=dp_rate)
    def forward(self, x):
        '''with torch.no_grad():
            pad_token_id = torch.Tensor([0]*x.size(2)).to(x.device) 
            m =  ~(x == pad_token_id)
            mask = m[:,:,0]
        # print(mask.shape)
        print(m)
        print(x)
        print(mask)
        print(m.shape, x.shape, mask.shape)'''
        # x: [B, T, num_feature]
        x = x.unsqueeze(1)
        # print(x.shape)
        # x: [B, 1, T, num_feature]
        x = self.conv(x)
        # print(x.shape)
        # print(x.shape)
        if self.temporal_control == 'avgpool':
            x = self.avg_pool(x)
        else:
            # x: [B, n_filters, T, num_feature]
            x = x.transpose(1, 2).contiguous()
        # x: [B, T, n_filters, num_feature]
        # print(x.shape)    
        
        # x: [B, T*n_filters*num_feature]
        x = x.reshape(x.size(0), x.size(1), -1)

        x = x.transpose(0, 1)
        # print(x.shape)
        x, _ = self.attn(x, x, x)
        x = x.transpose(0, 1)
        
        x = x.reshape(x.size(0), -1)
        x = self.dropout(x)

        x = self.fc1(x) # x: [B, T, num_class]
        x = self.mish(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

class SpiraConvLSTMV1(nn.Module):
    ''' Is the same than V1 but we change batchnorm to Group Norm'''
    def __init__(self, config, conv_num=4):
        super(SpiraConvLSTMV1, self).__init__()
        self.config = config
        # 68 %
        # lstm_dim = 40
        # dp_rate = 0.8

        # 64
        # lstm_dim = 60
        # dp_rate = 0.8

        # 66
        # lstm_dim = 50
        # dp_rate = 0.8

        # 61
        # lstm_dim = 35
        # dp_rate = 0.8

        lstm_dim = 40
        dp_rate = 0.8

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

        convs = [
            # cnn1
            nn.Conv2d(1, 32, kernel_size=(7,1), dilation=(2, 1)),
            nn.GroupNorm(16, 32), Mish(), nn.MaxPool2d(kernel_size=(2,1)), nn.Dropout(p=dp_rate), 
            
            # cnn2
            nn.Conv2d(32, 16, kernel_size=(5, 1), dilation=(2, 1)),
            nn.GroupNorm(8, 16), Mish(),  nn.MaxPool2d(kernel_size=(2,1)), nn.Dropout(p=dp_rate), 

            # cnn3
            nn.Conv2d(16, 8, kernel_size=(3, 1), dilation=(2, 1)), 
            nn.GroupNorm(4, 8), Mish(),  nn.MaxPool2d(kernel_size=(2,1)), nn.Dropout(p=dp_rate), 
            # cnn4
            nn.Conv2d(8, 4, kernel_size=(2, 1), dilation=(1, 1)), 
            nn.GroupNorm(2, 4), Mish(), nn.Dropout(p=dp_rate)]

        self.conv = nn.Sequential(*convs)
        
        if self.temporal_control == 'padding' or self.temporal_control == 'overlapping':
            # its very useful because if you change the convlutional arquiture the model calculate its, and you dont need change this :)
            # I prefer activate the network in toy example because is more easy than calculate the conv output
            # get zeros input
            inp = torch.zeros(1, 1, self.max_seq_len, self.num_feature)
            # get out shape
            # print(inp.shape)
            conv_out = self.conv(inp)
            conv_out = conv_out.transpose(1, 2)
            
            toy_activation_shape = conv_out.shape
            # set fully connected input dim
            self.lstm1 = torch.nn.LSTM(
                        input_size=toy_activation_shape[2]*toy_activation_shape[3],
                        hidden_size=lstm_dim,
                        num_layers=1,
                        batch_first=True)
            self.fc1 = nn.Linear(toy_activation_shape[1]*lstm_dim, self.config.model['fc1_dim'])          
            
        
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
        self.dropout = nn.Dropout(p=dp_rate)
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
        x = x.reshape(x.size(0), x.size(1), -1)
        x, _ = self.lstm1(x)

        x = x.reshape(x.size(0), -1)
        x = self.dropout(x)

        x = self.fc1(x) # x: [B, T, num_class]
        x = self.mish(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        #print(x.shape)
        return x


class SpiraConvV3(nn.Module):
    ''' Is the same than V1 but we change batchnorm to Group Norm'''
    def __init__(self, config, conv_num=3, kernels=(3,1), dropout_rate=0.2, dilatation=(1, 1)):
        super(SpiraConvV3, self).__init__()
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


        convs = []
        num_filter = 2**(conv_num+1)
        f_inp = 1
        f_l = num_filter
        for i in range(conv_num-1):
            if i != 0:
                f_inp = f_l
                f_l = int(num_filter/2)
            convs += [ nn.Conv2d(f_inp, f_l, kernel_size=kernels, dilation=dilatation),nn.GroupNorm(int(f_l/2), f_l), Mish(), nn.MaxPool2d(kernel_size=(2,1)), nn.Dropout(dropout_rate)]
        
        convs += [nn.Conv2d(f_l, 1, kernel_size=kernels, dilation=dilatation)]
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




class SpiraConvV2(nn.Module):
    ''' Is the same than V1 but we change batchnorm to Group Norm'''
    def __init__(self, config):
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



class SpiraSpTv2(nn.Module):
    ''' Is the same than V1 but we change batchnorm to Group Norm'''
    def __init__(self, config, dropout_rate = 0.4, pool_type = 'cls', pos_embedding_dim = 1000, transformer_mlp_dim = 50, num_layers = 1, heads = 8):
        # 'dropout_rate': 0.4, 'pool_type': 'cls', 'pos_embedding_dim': 1000, 'transformer_mlp_dim': 50, 'num_layers': 1, 'heads': 8
        super(SpiraSpTv2, self).__init__()
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
        
        '''convs = []
        num_filter = self.num_feature
        if prenet_layers:
            self.prenet = True
            f_inp = 10
            f_l = num_filter
            for i in range(prenet_layers-1):
                if i != 0:
                    f_inp = f_l
                    f_l = int(num_filter/i)
                convs += [ nn.Conv2d(f_inp, f_l, kernel_size=(3,3), dilation=(1, 1)), Mish(), nn.Dropout(dropout_rate)]
            convs += [nn.Conv2d(f_l, 1, kernel_size=(2, 1), dilation=(1, 1))]
            self.conv = nn.Sequential(*convs)
        else:
            self.prenet = False

        pool_type = 'mean' # mean or cls

        if not self.prenet:
            transformer_input_dim = seq_len * self.num_feature
        else: 
            inp = torch.zeros(1, 1, 500 ,self.num_feature)
            # get out shape 
            print(self.conv(inp).shape)
            exit()'''

        # 61: dp 0.8, "learning_rate": 1e-3, pos_embedding_dim = 1000, transformer_mlp_dim = 50,heads=16, num_layers = 1
        # 60/62 =  "learning_rate": 1e-3,warmup_steps= 1 dropout_rate = 0.8, pos_embedding_dim = 1000, transformer_mlp_dim = 50, heads = 8
        # embedding 
        
        emb_dropout = dropout_rate
        # transformer
        transformer_dropout = dropout_rate
        
        
        
        num_classes = 1

        transformer_input_dim = seq_len * self.num_feature

        self.spt = SpT(input_dim = transformer_input_dim, num_max_patches = self.config.dataset['num_max_patches'], num_classes = num_classes, dim =pos_embedding_dim, depth = num_layers, heads = heads, mlp_dim = transformer_mlp_dim, dropout = transformer_dropout, emb_dropout = emb_dropout, pool=pool_type)
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
    def forward(self, x, mixup_lambda=None, mask=None):
        # Mixup on feature
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

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
