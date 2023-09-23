import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from net.utils.tgcn import ConvTemporalGraphical
from net.utils.graph import Graph

class STEFunction_structured(torch.autograd.Function):
    """ define straight through estimator with overrided gradient (gate) """
    @staticmethod
    def forward(ctx, aux1, aux2):
        ctx.save_for_backward(aux1, aux2)
        aux1_stack = torch.stack([aux1, aux2])
        ##### Sort the parameter according to the 0th dimension
        aux1_stack_sorted, indices = torch.sort(aux1_stack, dim=0)
        ##### sum up the the parameter of each row in two rows
        aux1_sorted_sum = torch.sum(aux1_stack_sorted, dim=1)
        ##### Get the gated mask output for each row
        gate_shuffled = (aux1_sorted_sum>0).float()
        ##### Extend the matrix back to (2, 25)
        gate_shuffled = gate_shuffled.repeat(25, 1).transpose(0, 1)
        ##### Get unsorted gate output (sort the selection back)
        gate_origin = torch.gather(gate_shuffled, 0, indices.argsort(dim=0))
        return gate_origin[0], gate_origin[1]

    @staticmethod
    def backward(ctx, grad_output1, grad_output2):
        aux1,aux2 = ctx.saved_tensors
        return torch.mul(F.softplus(aux1), grad_output1), torch.mul(F.softplus(aux2), grad_output2)


class STEFunction_layer_wise(torch.autograd.Function):
    """ define straight through estimator with overrided gradient (gate) """
    @staticmethod
    def forward(ctx, aux1, aux2):
        ctx.save_for_backward(aux1, aux2)
        return (torch.sum(aux1) > 0).float().expand(25), (torch.sum(aux2) > 0).float().expand(25)

    @staticmethod
    def backward(ctx, grad_output1, grad_output2):
        aux1,aux2 = ctx.saved_tensors
        return torch.mul(F.softplus(aux1), grad_output1), torch.mul(F.softplus(aux2), grad_output2)

class STE_square(torch.autograd.Function):
    """ define straight through estimator with overrided gradient (gate) """
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.mul(input, input)

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return 2*torch.mul(input, grad_output)

def x2act(input,weight,bias, scale_x2 = 0.001):
    '''
    Applies the x^2 Unit (x2act) function element-wise:
        x2act(x) = x*x
    '''
#   return 0.005 * torch.mul(input, input) + relu(input) # use torch.sigmoid to make sure that we created the most efficient implemetation based on builtin PyTorch functions
    # return 0.001 * weight[0] * STE_square.apply(input) + weight[1] * input + bias #torch.mul(input, input)
    # return torch.mul(input, (0.005 * torch.mul(weight[0], input) + weight[1])) + bias #torch.mul(input, input)
    # print("input shape:", input.shape)
    # print(weight[0].shape, weight[1].shape)
    return torch.mul(input, (scale_x2 * weight[0] * input + weight[1])) + bias #torch.mul(input, input)
    
# def x2act(input,weight,bias):
#     '''
#     Applies the x^2 Unit (x2act) function element-wise:
#         x2act(x) = x*x
#     '''
#     return torch.mul(input, (0.1 * input + 1))



class x2act_node_module(nn.Module):
    def __init__(self, reduce = False, keep_relu = False, scale_x2 = 0.01):
        super(x2act_node_module,self).__init__() # init the base class
        self.weight_poly_x2 = torch.nn.Parameter(torch.full((25,), 0.005),  requires_grad=True)
        self.weight_poly_x = torch.nn.Parameter(torch.full((25,), 1.0),  requires_grad=True)
        self.weight_poly = [self.weight_poly_x2, self.weight_poly_x]
        self.bias_poly = torch.nn.Parameter(torch.FloatTensor([0]), requires_grad=False)
        param_shape = (25)
        # Create a nn.Parameter object with shape param_shape and initialize it to 1
        self.poly_aux = nn.Parameter(torch.ones(param_shape), requires_grad=False)
        self.reduce = reduce
        if self.reduce:
            self.poly_aux.requires_grad = True
            # nn.init.uniform_(self.poly_aux, 0.0005, 0.001)
            # nn.init.uniform_(self.poly_aux, -0.001, 0)
            # print("Polynomial reduction: {}".format(self.reduce))
        self.keep_relu = keep_relu
        self.scale_x2 = scale_x2
    def forward(self, input):
        #### Train all polynomial

        # ##### Shortcut to boost training speed performance for ReLU replacement
        # neuron_poly_mask = self.neuron_poly_mask
        # neuron_pass_mask = 1 - neuron_poly_mask
        # return torch.mul(F.relu(input), neuron_poly_mask) + torch.mul(input, neuron_pass_mask)
    
        if self.reduce:
            neuron_poly_mask = self.neuron_poly_mask
            neuron_pass_mask = 1 - neuron_poly_mask
            if self.keep_relu:
                return torch.mul(F.relu(input), neuron_poly_mask) + torch.mul(input, neuron_pass_mask)
            else:
                return torch.mul(x2act(input, self.weight_poly, self.bias_poly, scale_x2 = self.scale_x2), neuron_poly_mask) + torch.mul(input, neuron_pass_mask)
        else: 
            return x2act(input, self.weight_poly, self.bias_poly, scale_x2 = self.scale_x2)
    
        #### Train polynomial replacement
        # neuron_poly_mask = self.poly_aux
        # neuron_pass_mask = 1 - neuron_poly_mask
        # return torch.mul(x2act(input, self.weight_poly, self.bias_poly), neuron_poly_mask) + torch.mul(input, neuron_pass_mask)


class x2act_module(nn.Module):
    def __init__(self):
        super(x2act_module,self).__init__() # init the base class
        self.weight_poly = torch.nn.Parameter(torch.FloatTensor([0.005, 1]), requires_grad=True)
        self.bias_poly = torch.nn.Parameter(torch.FloatTensor([0]), requires_grad=False)
        param_shape = (25)
        # Create a nn.Parameter object with shape param_shape and initialize it to 1
        self.poly_aux = nn.Parameter(torch.ones(param_shape), requires_grad=False)

    def forward(self, input):
        #### Train all polynomial
        return x2act(input, self.weight_poly, self.bias_poly)
    
        #### Train polynomial replacement
        # neuron_poly_mask = self.poly_aux
        # neuron_pass_mask = 1 - neuron_poly_mask
        # return torch.mul(x2act(input, self.weight_poly, self.bias_poly), neuron_poly_mask) + torch.mul(input, neuron_pass_mask)

def replace_relu_w_poly(model, replacement_class, x2act_list, poly_reduce = False, keep_relu = False, scale_x2 = 0.01):
    for name, module in model.named_children():
        if isinstance(module, nn.ReLU):
            
            model._modules[name] = replacement_class(reduce = poly_reduce, keep_relu = keep_relu, scale_x2 = scale_x2)
            # print("Parameter: {}", model._modules[name].poly_aux)
            x2act_list.append(model._modules[name])
        else:
            replace_relu_w_poly(module, replacement_class, x2act_list, poly_reduce, keep_relu, scale_x2 = scale_x2)

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
        # self.st_gcn_networks = nn.ModuleList((
        #     st_gcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
        #     st_gcn(64, 64, kernel_size, 1, **kwargs),
        #     st_gcn(64, 64, kernel_size, 1, **kwargs),
        #     st_gcn(64, 64, kernel_size, 1, **kwargs),
        #     st_gcn(64, 128, kernel_size, 2, **kwargs),
        #     st_gcn(128, 128, kernel_size, 1, **kwargs),
        #     st_gcn(128, 128, kernel_size, 1, **kwargs),
        #     st_gcn(128, 256, kernel_size, 2, **kwargs),
        #     st_gcn(256, 256, kernel_size, 1, **kwargs),
        #     st_gcn(256, 256, kernel_size, 1, **kwargs),
        # ))
    
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 128, kernel_size, 2, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 256, kernel_size, 2, **kwargs),
            st_gcn(256, 256, kernel_size, 1, **kwargs),
        ))


        ###### Record the feature map output for each st-gcn
        self.x_feat = []

        # self.st_gcn_networks = nn.ModuleList((
        #     st_gcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
        #     st_gcn(64, 128, kernel_size, 2, **kwargs),
        #     st_gcn(128, 256, kernel_size, 2, **kwargs),
        #     st_gcn(256, 256, kernel_size, 1, **kwargs),
        # ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # fcn for prediction
        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)

        # if replace_x2:
        #     replace_relu_w_poly(self, x2act_module)
    def forward(self, x):
        
        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forwad
        # if self.return_feat: 
        self.x_feat.clear()
        for i, (gcn, importance) in enumerate(zip(self.st_gcn_networks, self.edge_importance)):
            x, _ = gcn(x, self.A * importance)
            exec('x_{} = x'.format(i))
            self.x_feat.append(eval('x_{}'.format(i)))
        # else:
        #     for i, (gcn, importance) in enumerate(zip(self.st_gcn_networks, self.edge_importance)):
        #         x, _ = gcn(x, self.A * importance)
        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)

        # prediction
        x = self.fcn(x)
        x = x.view(x.size(0), -1)
   
        # if self.return_feat:
        #     return x, x_feat
        # else:
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

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        _, c, t, v = x.size()
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)

        # prediction
        x = self.fcn(x)
        output = x.view(N, M, -1, t, v).permute(0, 2, 3, 4, 1)

        return output, feature

class Model_3layers_1(nn.Module):
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
        # self.st_gcn_networks = nn.ModuleList((
        #     st_gcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
        #     st_gcn(64, 64, kernel_size, 1, **kwargs),
        #     st_gcn(64, 64, kernel_size, 1, **kwargs),
        #     st_gcn(64, 64, kernel_size, 1, **kwargs),
        #     st_gcn(64, 128, kernel_size, 2, **kwargs),
        #     st_gcn(128, 128, kernel_size, 1, **kwargs),
        #     st_gcn(128, 128, kernel_size, 1, **kwargs),
        #     st_gcn(128, 256, kernel_size, 2, **kwargs),
        #     st_gcn(256, 256, kernel_size, 1, **kwargs),
        #     st_gcn(256, 256, kernel_size, 1, **kwargs),
        # ))
    
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(64, 128, kernel_size, 2, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs)
        ))
        ###### Record the feature map output for each st-gcn
        self.x_feat = []

        # self.st_gcn_networks = nn.ModuleList((
        #     st_gcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
        #     st_gcn(64, 128, kernel_size, 2, **kwargs),
        #     st_gcn(128, 256, kernel_size, 2, **kwargs),
        #     st_gcn(256, 256, kernel_size, 1, **kwargs),
        # ))

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

        # if replace_x2:
        #     replace_relu_w_poly(self, x2act_module)
    def forward(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forwad
        # if self.return_feat: 
        self.x_feat.clear()
        for i, (gcn, importance) in enumerate(zip(self.st_gcn_networks, self.edge_importance)):
            x, _ = gcn(x, self.A * importance)
            exec('x_{} = x'.format(i))
            self.x_feat.append(eval('x_{}'.format(i)))
        # else:
        #     for i, (gcn, importance) in enumerate(zip(self.st_gcn_networks, self.edge_importance)):
        #         x, _ = gcn(x, self.A * importance)
        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)

        # prediction
        x = self.fcn(x)
        x = x.view(x.size(0), -1)
   
        # if self.return_feat:
        #     return x, x_feat
        # else:
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

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        _, c, t, v = x.size()
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)

        # prediction
        x = self.fcn(x)
        output = x.view(N, M, -1, t, v).permute(0, 2, 3, 4, 1)

        return output, feature

class Model_3layers_2(nn.Module):
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
        # self.st_gcn_networks = nn.ModuleList((
        #     st_gcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
        #     st_gcn(64, 64, kernel_size, 1, **kwargs),
        #     st_gcn(64, 64, kernel_size, 1, **kwargs),
        #     st_gcn(64, 64, kernel_size, 1, **kwargs),
        #     st_gcn(64, 128, kernel_size, 2, **kwargs),
        #     st_gcn(128, 128, kernel_size, 1, **kwargs),
        #     st_gcn(128, 128, kernel_size, 1, **kwargs),
        #     st_gcn(128, 256, kernel_size, 2, **kwargs),
        #     st_gcn(256, 256, kernel_size, 1, **kwargs),
        #     st_gcn(256, 256, kernel_size, 1, **kwargs),
        # ))
    
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, 128, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(128, 256, kernel_size, 2, **kwargs),
            st_gcn(256, 256, kernel_size, 1, **kwargs)
        ))
        ###### Record the feature map output for each st-gcn
        self.x_feat = []

        # self.st_gcn_networks = nn.ModuleList((
        #     st_gcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
        #     st_gcn(64, 128, kernel_size, 2, **kwargs),
        #     st_gcn(128, 256, kernel_size, 2, **kwargs),
        #     st_gcn(256, 256, kernel_size, 1, **kwargs),
        # ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # fcn for prediction
        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)

        # if replace_x2:
        #     replace_relu_w_poly(self, x2act_module)
    def forward(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forwad
        # if self.return_feat: 
        self.x_feat.clear()
        for i, (gcn, importance) in enumerate(zip(self.st_gcn_networks, self.edge_importance)):
            x, _ = gcn(x, self.A * importance)
            exec('x_{} = x'.format(i))
            self.x_feat.append(eval('x_{}'.format(i)))
        # else:
        #     for i, (gcn, importance) in enumerate(zip(self.st_gcn_networks, self.edge_importance)):
        #         x, _ = gcn(x, self.A * importance)
        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)

        # prediction
        x = self.fcn(x)
        x = x.view(x.size(0), -1)
   
        # if self.return_feat:
        #     return x, x_feat
        # else:
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

        # forwad
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
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
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
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x), A

    # , replace_x2 = False


### Model with ReLU Replacement(RP)
class Model_replace(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        # self.model = model
        #### Initialize model architecture
        self.kwargs = kwargs
        self.model_args = ['in_channels', 'num_class', 'dropout', \
                           'edge_importance_weighting', 'graph_args']
        self.model_kwargs = {k: v for k, v in self.kwargs.items() if k in self.model_args}
        self.util_kwargs = {k: v for k, v in self.kwargs.items() if k not in self.model_args}
        self.replace_poly = self.util_kwargs['replace_poly']
        self.node_wise = None
        if 'node_wise' in self.util_kwargs.keys():
            self.node_wise = self.util_kwargs['node_wise']

        self.clip_poly_mark = None
        if 'clip_poly' in self.util_kwargs.keys():
            self.clip_poly_mark = self.util_kwargs['clip_poly']

        self.poly_reduce = None
        if 'poly_reduce' in self.util_kwargs.keys():
            self.poly_reduce = self.util_kwargs['poly_reduce']  

        self.keep_relu = False
        if 'keep_relu' in self.util_kwargs.keys():
            self.keep_relu = self.util_kwargs['keep_relu']  

        self.lambda_penalty = 0
        if 'lambda_penalty' in self.util_kwargs.keys():
            self.lambda_penalty = self.util_kwargs['lambda_penalty']

        self.backbone = 'Model'
        if 'backbone' in self.util_kwargs.keys():
            self.backbone = self.util_kwargs['backbone']


        self.model = eval("{}(**self.model_kwargs)".format(self.backbone))



        self.scale_x2 = 0.01
        if 'scale_x2' in self.util_kwargs.keys():
            self.scale_x2 = self.util_kwargs['scale_x2']

        self.ste_func = STEFunction_structured
        if 'layer_wise' in self.util_kwargs.keys():
            if self.util_kwargs['layer_wise']:
                self.ste_func = STEFunction_layer_wise

        if self.replace_poly:
            self.x2act_list = []
            if self.node_wise: 
                replace_relu_w_poly(self, x2act_node_module, self.x2act_list, poly_reduce = self.poly_reduce, keep_relu = self.keep_relu, scale_x2 = self.scale_x2)
                pass
            else:
                replace_relu_w_poly(self, x2act_module, self.x2act_list)
        
            self.num_x2act = len(self.x2act_list)
        
        self._weights_poly = []
        self._weights_aux = []
        for name, parameter in self.named_parameters():
            if ('poly' in name) and ('aux' not in name):
                self._weights_poly.append((name, parameter)) 
            elif ('poly_aux' in name):
                self._weights_aux.append((name, parameter)) 

        self.freeze_gate_ = False
        if 'freeze_gate' in self.util_kwargs.keys():
            self.freeze_gate_ = self.util_kwargs['freeze_gate']  
        if self.freeze_gate_:
            self.freeze_gate()

    def weights_poly(self):
        for n, p in self._weights_poly:
            yield p
    def named_weights_poly(self):
        for n, p in self._weights_poly:
            yield n, p
    def freeze_poly(self):
        for n, p in self._weights_poly:
            p.requires_grad = False
    def freeze_gate(self):
        for n, p in self._weights_aux:
            p.requires_grad = False
    def clip_poly(self):
        # Set the minimum and maximum values
        with torch.no_grad():
            min_val = -2
            max_val = 2
            for name, param in self.named_weights_poly():
            # Clip the range of the parameter to be between min_val and max_val
                param.data = torch.clamp(param.data, min=min_val, max=max_val)
    def init_poly_aux(self): 
        for name, parameter in self._weights_aux:
            nn.init.uniform_(parameter, 0.0005, 0.001)
        # print(self._weights_aux)
    def gate_forward(self):
        num_gate = self.num_x2act//2
        if hasattr(self, 'gate_mask'):
            del self.gate_mask
        self.gate_mask = []
        for i in range(num_gate):
            x2act_module_1 = self.x2act_list[2*i]
            x2act_module_2 = self.x2act_list[2*i + 1]
            # print("Parameter: {}, {}".format(x2act_module_1.poly_aux, x2act_module_2.poly_aux))
            x2act_module_1.neuron_poly_mask, x2act_module_2.neuron_poly_mask = \
                self.ste_func.apply(x2act_module_1.poly_aux, x2act_module_2.poly_aux)
            self.gate_mask.append(x2act_module_1.neuron_poly_mask)
            self.gate_mask.append(x2act_module_2.neuron_poly_mask)
    def gate_density_forward(self):
        total_num_gate = 0
        remain_gate = 0
        for gate_mask in self.gate_mask:
            num_gate = gate_mask.shape[0]
            remain_gate_origin = gate_mask.sum()
            total_num_gate += num_gate
            remain_gate += remain_gate_origin
        global_density = remain_gate/total_num_gate
        # print("Added density forward plus penalty: {}".format(self.lambda_penalty))
        
        return global_density * self.lambda_penalty
    def print_gate(self, print_func):  #self.io.print_log
        with torch.no_grad():
            total_num_gate = 0
            remain_gate = 0
            if hasattr(self, 'remain_gate_list'):
                del self.remain_gate_list
            self.remain_gate_list = []

            for gate_mask in self.gate_mask:
                num_gate = gate_mask.shape[0]
                remain_gate_origin = gate_mask.sum().data.item()
                total_num_gate += num_gate
                remain_gate += remain_gate_origin
                self.remain_gate_list.append(remain_gate_origin)
            print_func("Total original gate: {}. Total remaining gate: {}. Percentage: {}. Layers: {}".format(\
                total_num_gate, remain_gate, remain_gate/total_num_gate, remain_gate//25))
            print_func("Remaining gate: " + str(self.remain_gate_list))
            

    def forward(self, x):
        
        if self.clip_poly_mark:
            self.clip_poly()
        if self.poly_reduce:
            self.gate_forward()
        out = self.model(x)
        return out
        # if config.dataset == "cifar100":
        #     self.model = eval(config.arch + '(num_classes = 100)')
        # elif config.dataset == "cifar10":
        #     self.model = eval(config.arch + '(num_classes = 10)')
        # else:
        #     print("dataset not supported yet")

        # # Initialize the weight mask
        # self.layer_pruned = []
        # self.eval()
        # with torch.no_grad():
        #     if config.weight_prune:
        #         if config.train_mask:
        #             for layer in self.model.modules():
        #                 if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
        #                     self.layer_pruned.append(layer)
        #                     layer.weight_aux = nn.Parameter(torch.ones_like(layer.weight))

        #                     layer.weight.requires_grad = True
        #                     layer.weight_aux.requires_grad = True
        #                     nn.init.uniform_(layer.weight_aux)
        #                 # This is the monkey-patch overriding layer.forward to custom function.
        #                 # layer.forward will pass nn.Linear with weights: 'w' and 'm' elementwised
        #                 if isinstance(layer, nn.Linear):
        #                     layer.forward = types.MethodType(mask_train_forward_linear, layer)

        #                 if isinstance(layer, nn.Conv2d):
        #                     layer.forward = types.MethodType(mask_train_forward_conv2d, layer)
        #         else:
        #             for layer in self.model.modules():
        #                 if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
        #                     self.layer_pruned.append(layer)
        #                     layer.weight_aux = nn.Parameter(torch.ones_like(layer.weight))
        #                     layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
        #                     layer.weight.requires_grad = True
        #                     layer.weight_aux.requires_grad = False
        #                     layer.weight_mask.requires_grad = False
        #                     nn.init.uniform_(layer.weight_aux)
        #                     layer.weight_mask.data.copy_(STEFunction.apply(layer.weight_aux))
        #                 # This is the monkey-patch overriding layer.forward to custom function.
        #                 # layer.forward will pass nn.Linear with weights: 'w' and 'm' elementwised
        #                 if isinstance(layer, nn.Linear):
        #                     layer.forward = types.MethodType(mask_fixed_forward_linear, layer)

        #                 if isinstance(layer, nn.Conv2d):
        #                     layer.forward = types.MethodType(mask_fixed_forward_conv2d, layer)
        # if config.relu_prune:
        #     # Initialize the activation mask
        #     self.x_size = config.x_size #### Input image size, for example in cifar 10, it's [1, 3, 32, 32]
        #     ReLU_masked_model = eval(config.act_type + '(config)') #ReLU_masked()
        #     # ReLU_masked_model = ReLU_masked(config) #ReLU_masked()
        #     if "efficientnet" in config.arch:
        #         replace_siLU(self, ReLU_masked_model)
        #     else: 
        #         replace_relu(self, ReLU_masked_model)
        #     #### Get the name and model_stat of sparse ReLU model ####
        #     self._ReLU_sp_models = []
        #     for name, model_stat in self.named_modules(): 
        #         if 'ReLU_masked' in type(model_stat).__name__:
        #             self._ReLU_sp_models.append(model_stat)

        #     #### Initialize alpha_aux pameters in ReLU_sp model ####
        #     #### through single step inference ####
        #     self.eval()
        #     with torch.no_grad():
        #         in_mock_tensor = torch.Tensor(*self.x_size)
        #         self.forward(in_mock_tensor)
        #         del in_mock_tensor
        #         for model in self._ReLU_sp_models:
        #             model.init = 0
            
        # ### Initialize _alpha_aux, _weights lists
        # ### self._alpha_aux[i] is the ith _alpha_aux parameter
        # self._aux = []
        # self._weights = []
        # self._weights_aux = []
        # self._relu_aux = []
        # self._weights_and_aux = []
        # for name, parameter in self.named_parameters():
        #     if 'aux' in name:
        #         self._aux.append((name, parameter)) 
        #         if 'weight' in name: 
        #             self._weights_aux.append((name, parameter))
        #         if 'relu' in name: 
        #             self._relu_aux.append((name, parameter))
        #         self._weights_and_aux.append((name, parameter))             
        #     else: 
        #         self._weights.append((name, parameter))
        #         self._weights_and_aux.append((name, parameter))
            
        # # self._alpha_mask = []
        # # for name, parameter in self.named_parameters():
        # #     if 'alpha_mask' in name:
        # #         self._alpha_mask.append((name, parameter))                 
    # def weights(self):
    #     for n, p in self._weights:
    #         yield p
    # def named_weights(self):
    #     for n, p in self._weights:
    #         yield n, p
    # def weights_and_aux(self):
    #     for n, p in self._weights_and_aux:
    #         yield p
    # def named_weights_and_aux(self):
    #     for n, p in self._weights_and_aux:
    #         yield n, p
    # def aux(self):
    #     for n, p in self._aux:
    #         yield p
    # def named_aux(self):
    #     for n, p in self._aux:
    #         yield n, p
    # ### Get Total number of gate parameter
    # def _get_num_gates(self):
    #     with torch.no_grad():
    #         num_gates = torch.tensor(0.)
    #         for name, aux in self._aux:
    #             num_gates += aux.numel()
    #     return num_gates

    # def train_fz_bn(self, freeze_bn=True, freeze_bn_affine=True, mode=True):
    #     """
    #         Override the default train() to freeze the BN parameters
    #     """
    #     # super(VGG, self).train(mode)
    #     self.train(mode)
    #     for m in self.modules():
    #         if isinstance(m, nn.BatchNorm2d):
    #             m.eval()
    #             if (freeze_bn_affine and m.affine == True):
    #                 m.weight.requires_grad = not freeze_bn
    #                 m.bias.requires_grad = not freeze_bn

    # def load_pretrained(self, pretrained_path = False):
    #     if pretrained_path:
    #         if os.path.isfile(pretrained_path):
    #             print("=> loading checkpoint '{}'".format(pretrained_path))
    #             checkpoint = torch.load(pretrained_path, map_location = "cpu")   
    #             # print('state_dict' in checkpoint.keys())  
    #             if 'state_dict' in checkpoint.keys():
    #                 pretrained_dict = checkpoint['state_dict']
    #             else:
    #                 pretrained_dict = checkpoint

    #             # pretrained_dict = checkpoint
    #             model_dict = self.state_dict()

    #             # print("pretrained_dict", [k for k, v in pretrained_dict.items()])
    #             # print("model_dict", [k for k, v in model_dict.items()])
    #             # exit()

    #             pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                
    #             # print(pretrained_dict)
    #             # print({k for k, v in pretrained_dict.items()})
    #             # print({k for k, v in model_dict.items()})
    #             # exit(0)
    #             model_dict.update(pretrained_dict) 
    #             self.load_state_dict(model_dict)
    #         else:
    #             print("=> no checkpoint found at '{}'".format(pretrained_path))

    # def load_check_point(self, check_point_path = False):
    #     if os.path.isfile(check_point_path):
    #         print("=> loading checkpoint from '{}'".format(check_point_path))
    #         checkpoint = torch.load(check_point_path, map_location = "cpu")
    #         start_epoch = checkpoint['epoch']
    #         best_prec1 = checkpoint['best_prec1']
    #         self.load_state_dict(checkpoint['state_dict'])
    #         print("=> loaded checkpoint at epoch {}"
    #               .format(checkpoint['epoch']))
    #         print("Is best result?: ", checkpoint['is_best'])
    #         return start_epoch, best_prec1
    #     else:
    #         print("=> no checkpoint found at '{}'".format(check_point_path))
               
    # def save_checkpoint(self, epoch, best_prec1, is_best, filename='checkpoint.pth.tar'):
    #     """
    #     Save the training model
    #     """
    #     state = {
    #             'epoch': epoch + 1,
    #             'state_dict': self.state_dict(),
    #             'best_prec1': best_prec1,
    #             'is_best': is_best
    #         }
    #     torch.save(state, filename)
    # def update_wgt_sparse_list(self):
    #     # Update the sparse_list correspond to alpha_aux
    #     # Format: [layer name, Total original ReLU count, Pruned count, Pruned percentage]
    #     # Update the global_sparsity
    #     # Format: [Total original ReLU count, Pruned count, Pruned percentage]
    #     if(hasattr(self, "sparse_list_wgt")):
    #         del self.sparse_list_wgt
    #     if(hasattr(self, "global_sparsity_wgt")):
    #         del self.global_sparsity_wgt
    #     self.sparse_list_wgt = []
    #     total_count_global = 0
    #     sparsity_count_global = 0
    #     with torch.no_grad():
    #         for name, param in self._weights_aux:
    #             weight_mask = STEFunction.apply(param)
    #             total_count = weight_mask.numel()
    #             sparsity_count = torch.sum(weight_mask).item()
    #             sparsity_pert = sparsity_count/total_count
    #             self.sparse_list_wgt.append([name, total_count, sparsity_count, sparsity_pert])
    #             total_count_global += total_count
    #             sparsity_count_global += sparsity_count
    #     sparsity_pert_global = sparsity_count_global/total_count_global
    #     self.global_sparsity_wgt = [total_count_global, sparsity_count_global, sparsity_pert_global]
    # def print_wgt_sparse_list(self, logger):
    #     # remove formats
    #     org_formatters = []
    #     for handler in logger.handlers:
    #         org_formatters.append(handler.formatter)
    #         handler.setFormatter(logging.Formatter("%(message)s"))

    #     # Logging alpha data: 
    #     logger.info("####### Weight Sparsity #######")
    #     logger.info("# Layer wise weight sparsity for the model")
    #     logger.info("# Format: [layer name, Total original weight count, remained count, remained percentage]")
    #     for sparse_list in self.sparse_list_wgt:
    #         logger.info(sparse_list)
    #     logger.info("#\n Global weight sparsity for the model")
    #     logger.info("# Format: [Total original weight count, remained count, remained percentage]")
    #     logger.info(self.global_sparsity_wgt)
    #     logger.info("########## End ###########")
                    
    #     # restore formats
    #     for handler, formatter in zip(logger.handlers, org_formatters):
    #         handler.setFormatter(formatter)

    # def update_relu_sparse_list(self):
    #     # Update the sparse_list correspond to alpha_aux
    #     # Format: [layer name, Total original ReLU count, Pruned count, Pruned percentage]
    #     # Update the global_sparsity
    #     # Format: [Total original ReLU count, Pruned count, Pruned percentage]
    #     if(hasattr(self, "sparse_list_relu")):
    #         del self.sparse_list_relu
    #     if(hasattr(self, "global_sparsity_relu")):
    #         del self.global_sparsity_relu
    #     self.sparse_list_relu = []
    #     total_count_global = 0
    #     sparsity_count_global = 0
    #     with torch.no_grad():
    #         for name, param in self._relu_aux:
    #             weight_mask = STEFunction.apply(param)
    #             total_count = weight_mask.numel()
    #             sparsity_count = torch.sum(weight_mask).item()
    #             sparsity_pert = sparsity_count/total_count
    #             self.sparse_list_relu.append([name, total_count, sparsity_count, sparsity_pert])
    #             total_count_global += total_count
    #             sparsity_count_global += sparsity_count
    #     sparsity_pert_global = sparsity_count_global/total_count_global
    #     self.global_sparsity_relu = [total_count_global, sparsity_count_global, sparsity_pert_global]
    # def print_relu_sparse_list(self, logger):
    #     # remove formats
    #     org_formatters = []
    #     for handler in logger.handlers:
    #         org_formatters.append(handler.formatter)
    #         handler.setFormatter(logging.Formatter("%(message)s"))

    #     # Logging alpha data: 
    #     logger.info("####### ReLU Sparsity #######")
    #     logger.info("# Layer wise relu sparsity for the model")
    #     logger.info("# Format: [layer name, Total original ReLU count, remained count, remained percentage]")
    #     for sparse_list in self.sparse_list_relu:
    #         logger.info(sparse_list)
    #     logger.info("#\n Global relu sparsity for the model")
    #     logger.info("# Format: [Total original ReLU count, remained count, remained percentage]")
    #     logger.info(self.global_sparsity_relu)
    #     logger.info("########## End ###########")
                    
    #     # restore formats
    #     for handler, formatter in zip(logger.handlers, org_formatters):
    #         handler.setFormatter(formatter)


    # def fix_mask(self):
    #     self.eval()
    #     with torch.no_grad():
    #         if self.config.weight_prune:
    #             for layer in self.layer_pruned:
    #                 if not hasattr(self, "weight_mask"):
    #                     layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
    #                 layer.weight_aux.requires_grad = False
    #                 layer.weight_mask.requires_grad = False
    #                 layer.weight_mask.data.copy_(STEFunction.apply(layer.weight_aux))
    #                 if isinstance(layer, nn.Linear):
    #                     layer.forward = types.MethodType(mask_fixed_forward_linear, layer)

    #                 if isinstance(layer, nn.Conv2d):
    #                     layer.forward = types.MethodType(mask_fixed_forward_conv2d, layer)
    #             for name, aux in self._aux:
    #                 aux.requires_grad = False
    #         if self.config.relu_prune:
    #             for ReLU_sp_models in self._ReLU_sp_models:
    #                 ReLU_sp_models.fix_mask()
    # def mask_fixed_update(self):
    #     self.eval()
    #     with torch.no_grad():
    #         if self.config.weight_prune:
    #             for layer in self.layer_pruned:
    #                 layer.weight_mask.data.copy_(STEFunction.apply(layer.weight_aux))
    #         if self.config.relu_prune:
    #             for ReLU_sp_models in self._ReLU_sp_models:
    #                 ReLU_sp_models.mask_fixed_update()
    # def mask_density_forward_wgt(self):
    #     total_count_global = 0
    #     sparse_list = []
    #     sparse_pert_list = []
    #     sparsity_count_global = 0
    #     # for name, param in self._aux:
    #         # weight_mask = STEFunction.apply(param)
    #     for layer in self.layer_pruned:
    #         weight_mask = STEFunction.apply(layer.weight_aux)
    #         total_count = weight_mask.numel()
    #         sparsity_count = torch.sum(weight_mask)
    #         sparsity_pert = sparsity_count/total_count
    #         sparse_list.append(sparsity_count.item())
    #         sparse_pert_list.append(sparse_list[-1]/total_count)
    #         total_count_global += total_count
    #         sparsity_count_global += sparsity_count
    #     global_density = sparsity_count_global/total_count_global 
    #     return global_density, sparse_list, sparse_pert_list, total_count_global
    # def mask_density_forward_relu(self):
    #     return self._ReLU_sp_models[0].mask_density_forward()
