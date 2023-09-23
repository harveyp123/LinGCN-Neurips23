from audioop import bias
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from torch.autograd import Variable
import numpy as np
import pandas as pd


from net.utils.tgcn import ConvTemporalGraphical
from net.utils.graph import Graph


def xact(input,weight,bias):
    '''
    Applies the x^2 Unit (x2act) function element-wise:
        x2act(x) = x*x
    '''
#   return 0.005 * torch.mul(input, input) + relu(input) # use torch.sigmoid to make sure that we created the most efficient implemetation based on builtin PyTorch functions
    return weight[0]*torch.mul(input, input)+weight[1]*input+bias

class xact_module(nn.Module):
    def __init__(self):
        super(x2act_module,self).__init__() # init the base class
        self.weight = torch.nn.Parameter(torch.FloatTensor([0, 1]), requires_grad=False)
        self.bias= torch.nn.Parameter(torch.FloatTensor([0]), requires_grad=False)

    def forward(self, input):
        return xact(input,self.weight,self.bias)

def x2act(input,weight,bias):
    '''
    Applies the x^2 Unit (x2act) function element-wise:
        x2act(x) = x*x
    '''
#   return 0.005 * torch.mul(input, input) + relu(input) # use torch.sigmoid to make sure that we created the most efficient implemetation based on builtin PyTorch functions
    return weight[0]*torch.mul(input, input)+weight[1]*input+bias

class x2act_module(nn.Module):
    def __init__(self):
        super(x2act_module,self).__init__() # init the base class
        self.weight = torch.nn.Parameter(torch.FloatTensor([0.005, 1]), requires_grad=True)
        self.bias= torch.nn.Parameter(torch.FloatTensor([0]), requires_grad=False)

    def forward(self, input):
        return x2act(input,self.weight,self.bias)


class Model(nn.Module):
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, in_channels, num_class, graph_args,
                 edge_importance_weighting, **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((

            st_gcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            #st_gcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs0 , relu_reduce=True),

            # st_gcn(64, 64, kernel_size, 1, **kwargs),
            # st_gcn(64, 64, kernel_size, 1, **kwargs),
            #st_gcn(64, 64, kernel_size, 1, **kwargs),

            st_gcn(64, 128, kernel_size, 2, **kwargs, residual=False),
            #st_gcn(64, 128, kernel_size, 2, **kwargs, residual=False, relu_reduce=True),

            st_gcn(128, 128, kernel_size, 1, **kwargs),
            #st_gcn(128, 128, kernel_size, 1, **kwargs, relu_reduce=True),

            # st_gcn(128, 128, kernel_size, 1, **kwargs),
            #st_gcn(128, 256, kernel_size, 2, **kwargs),
            #st_gcn(256, 256, kernel_size, 1, **kwargs),

            # st_gcn(256, 256, kernel_size, 1, **kwargs),
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # fcn for prediction

        self.fcn = nn.Conv2d(128, num_class, kernel_size=1)
        #self.fcn = nn.Conv2d(256, num_class, kernel_size=1)

    def forward(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)


        # forward
        #print('------------------------Input data---------------------------',x)
        #print('-------------------------Input feature map size--------------',x.size())

        #x.view(-1)
        #temp1=x.cpu().detach().numpy()
        # with open("data_1d.txt","w") as output:
        #     output.write(str(temp1))

        # fixed_scale=pow(2,10)
        # x= ((x*fixed_scale).floor())*(1/fixed_scale)

        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)
        
        #print('-------------------------Output feature map size--------------',x.size())

        # temp1=x.cpu().detach().numpy()
        # np.save("data_conv.csv",temp1)

        #print("check the size-------------------------------",x.size()[2:])
        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])

        #//print("check the size-------------------------------",x.size())
        x = x.view(N, M, -1, 1, 1).mean(dim=1)

        # temp2=x.view(-1).tolist()
        # with open("predictedresult_2^10.txt","w") as output:
        #     output.write(str(temp2))
        
        
        #print( "Before the fcn layer -----------",x.size())

        # temp2=x.view(-1).tolist()
        # with open("predictedresult_1d.txt","w") as output:
        #     output.write(str(temp2))

        # prediction
        x = self.fcn(x)
        x = x.view(x.size(0), -1)

        # temp2=x.view(-1).tolist()
        # with open("predictedresult_1d.txt","w") as output:
        #     output.write(str(temp2))

        # print("--------------------------predicted results--------------------",x)
        # print("--------------------Predicted result size---------------------",x.size())

        return x

    def extract_feature(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        _, c, t, v = x.size()
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)

        # prediction
        x = self.fcn(x)
        output = x.view(N, M, -1, t, v).permute(0, 2, 3, 4, 1)

        return output, feature

class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True,
                 relu_reduce=False):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        if not relu_reduce:

            self.tcn = nn.Sequential(
                nn.BatchNorm2d(out_channels),
                x2act_module(),
                #nn.ReLU(inplace=True),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    (kernel_size[0], 1),
                    (stride, 1),
                    padding,
                ),
                nn.BatchNorm2d(out_channels),
                nn.Dropout(dropout, inplace=True),
            )

        else:

            self.tcn = nn.Sequential(
                nn.BatchNorm2d(out_channels),
                #xact_module(),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    (kernel_size[0], 1),
                    (stride, 1),
                    padding,
                ),
                nn.BatchNorm2d(out_channels),
                nn.Dropout(dropout, inplace=True),
            )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1),
                    ),
                nn.BatchNorm2d(out_channels),
            )
        
        #self.relu = nn.ReLU(inplace=True)
        #self.relu = x2act_module()
        
        if not relu_reduce:
            self.relu = x2act_module()
            #self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = lambda x: x

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res
        #print("st-gcn output feature map", x.size())

        return self.relu(x), A