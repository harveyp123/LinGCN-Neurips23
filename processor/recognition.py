#!/usr/bin/env python
# pylint: disable=W0201
import sys
import argparse
import yaml
import numpy as np
import math
# torch
import torch
import torch.nn as nn
import torch.optim as optim

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from .processor import Processor

from .optimizer import Lookahead
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import functional as F

# # Define the custom communication hook
# def fp16_compress_hook(state: object, bucket: dist.GradBucket) -> torch.futures.Future:
#     compressed_tensors = [tensor.to(dtype=torch.float16) for tensor in bucket.get_tensors()]
#     return state.allreduce(compressed_tensors).then(lambda fut: fut.wait().to(dtype=torch.float32))



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
class SoftTarget(nn.Module):
	'''
	Distilling the Knowledge in a Neural Network
	https://arxiv.org/pdf/1503.02531.pdf
	'''
	def __init__(self, T):
		super(SoftTarget, self).__init__()
		self.T = T

	def forward(self, out_s, out_t):
		loss = F.kl_div(F.log_softmax(out_s/self.T, dim=1),
						F.softmax(out_t/self.T, dim=1),
						reduction='batchmean') * self.T * self.T

		return loss

##### Borrowed from https://github.com/szagoruyko/attention-transfer/blob/master/utils.py
def at(x):
    return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))

def at_loss(x, y):
    return (at(x) - at(y)).pow(2).mean()

def at_loss_model(model_s, model_t):
    loss = 0
    for act_s, act_t in zip(model_s.model.x_feat, model_t.model.x_feat):
        loss += at_loss(act_s, act_t)
    return loss
class REC_Processor(Processor):
    """
        Processor for Skeleton-based Action Recgnition
    """

    def load_model(self):
        
        self.model = self.io.load_model(self.arg.model,
                                        **(self.arg.model_args))
        self.model.apply(weights_init)
        self.loss = nn.CrossEntropyLoss()

    def load_model_teacher(self):
        
        self.model_teacher = self.io.load_model(self.arg.model_teacher,
                                        **(self.arg.model_teacher_args))
        self.model_teacher.apply(weights_init)
        self.loss_kd_div = SoftTarget(4.0)
        self.loss_kd_at = at_loss_model
    
    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()
        # self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #         self.optimizer, self.arg.num_epoch, eta_min=1e-5)
        if self.arg.lookahead:
            self.optimizer = Lookahead(self.optimizer)

    def adjust_lr(self):
        if self.arg.optimizer == 'SGD' and self.arg.step:
            lr = self.arg.base_lr * (
                0.1**np.sum(self.meta_info['epoch']>= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr
        else:
            self.lr = self.arg.base_lr
        # self.lr_scheduler.step()

    def show_topk(self, k):
        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
        self.io.print_log('\tTop{}: {:.2f}%'.format(k, 100 * accuracy))
        self.writer.add_scalar('val/top{}'.format(k), accuracy, self.meta_info['iter'])
        return accuracy

    def train(self):

        if self.meta_info['epoch'] > self.arg.freeze_gate_epoch:
            self.io.print_log('Freeze the gate parameter')
            self.model.freeze_gate()

        if self.arg.freeze_poly:
            self.io.print_log('Freeze the polynomial parameter')
            self.model.freeze_poly()
        self.model.train()
        if self.arg.distil: 
            self.model_teacher.eval()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []
        if self.arg.mix_precision: 
            if self.arg.format == 'fp16':
                use_amp = True
                scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
                dtype = torch.float16
            elif self.arg.format == 'bf16':
                use_amp = True
                scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
                dtype = torch.bfloat16
            elif self.arg.format == 'tf32':
                torch.backends.cuda.matmul.allow_tf32=True
                torch.backends.cudnn.allow_tf32=True
            else:
                self.io.print_log("Format {} is not supported yet".format(self.arg.format))
                exit()
        # if self.arg.grad_compression:
        #     self.model.register_comm_hook(state=None, hook=fp16_compress_hook)
        #     # use_amp = False
        #     # scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        for data, label in loader:

            # get data
            if self.arg.frame_reduce:
                data = data[:, :, 22:278, :, :]
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)
            if (self.arg.mix_precision) and (self.arg.format != 'tf32'): 
                
                with torch.autocast(device_type='cuda', dtype=dtype, enabled=use_amp):
                    # forward
                    output = self.model(data)
                    loss = self.loss(output, label)
                    
                    if self.arg.distil: 
                        output_teacher = self.model_teacher(data)
                        loss_kd_div = self.loss_kd_div(output, output_teacher)
                        loss_kd_at = self.loss_kd_at(self.model, self.model_teacher)
                        #self.arg.eta self.arg.varphi
                        loss = (1-self.arg.eta) * loss + self.arg.eta * loss_kd_div + self.arg.varphi / 2 * loss_kd_at
                    if hasattr(self.model, 'lambda_penalty'):
                        loss += (self.model.gate_density_forward() if self.model.lambda_penalty else 0)
                # backward
                self.optimizer.zero_grad()

                #### Add loss NaN skipping. 
                if torch.isnan(loss).any():
                    self.io.print_log("Skipping update step due to NaN in loss")
                    # print(f"Skipping update step due to NaN in loss")
                    continue

                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)
                scaler.step(self.optimizer)
                scaler.update()
                loss = loss.float()
                # loss.backward()
                # self.optimizer.step()
            else: 
                # forward
                output = self.model(data)
                loss = self.loss(output, label)
                if self.arg.distil: 
                    output_teacher = self.model_teacher(data)
                    loss_kd_div = self.loss_kd_div(output, output_teacher)
                    loss_kd_at = self.loss_kd_at(self.model, self.model_teacher)
                    loss = 0.1 * loss + 0.9 * loss_kd_div + 500 * loss_kd_at
                # backward
                if hasattr(self.model, 'lambda_penalty'):
                    loss += (self.model.gate_density_forward() if self.model.lambda_penalty else 0)
                self.optimizer.zero_grad()

                #### Add loss NaN skipping. 
                if torch.isnan(loss).any():
                    self.io.print_log("Skipping update step due to NaN in loss")
                    # print(f"Skipping update step due to NaN in loss")
                    continue

                loss.backward()
                self.optimizer.step()
                # ###### Add gradient scaler and gradient compression
                # if self.arg.grad_compression:
                #     for param in self.model.parameters():
                #         if param.grad is not None:
                #             param.grad.data = scaler.scale(param.grad.data).to(dtype=torch.float16)
                #             # param.grad.data = scaler.scale(param.grad.data).to(dtype=torch.float16)
                #     # scaler.unscale_(self.optimizer)
                #     # scaler.step(self.optimizer)
                #     self.optimizer.step()
                #     # scaler.update()
                # else:
                #     self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            # self.iter_info['lr'] = '{:.6f}'.format(self.lr_scheduler.get_last_lr()[0])
            loss_value.append(self.iter_info['loss'])
            self.show_iter_info()
            ##### Write the loss to tensorboard #####
            self.writer.add_scalar('train/loss', loss.data.item(), self.meta_info['iter'])

            # print('********** iteration {} **********'.format(self.meta_info['iter']))
            # for name, para in self.model.named_weights_poly():
                
            #     print('name: ', name, 'parameter: ', para)
            self.meta_info['iter'] += 1
        torch.cuda.empty_cache()
        self.epoch_info['mean_loss']= np.mean(loss_value)
        self.show_epoch_info()
        self.io.print_timer()

    def test(self, evaluation=True):

        self.model.eval()
        loader = self.data_loader['test']
        loss_value = []
        result_frag = []
        label_frag = []
        if self.arg.mix_precision: 
            if self.arg.format == 'fp16':
                use_amp = True
                dtype = torch.float16
            elif self.arg.format == 'bf16':
                use_amp = True
                dtype = torch.bfloat16
            elif self.arg.format == 'tf32':
                torch.backends.cuda.matmul.allow_tf32=True
                torch.backends.cudnn.allow_tf32=True
            else:
                self.io.print_log("Format {} is not supported yet".format(self.arg.format))
                exit()



        for data, label in loader:
            # get data
            if self.arg.frame_reduce:
                data = data[:, :, 22:278, :, :]
            # get data
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)

            # inference
            if (self.arg.mix_precision) and (self.arg.format != 'tf32'): 
                with torch.autocast(device_type='cuda', dtype=dtype, enabled=use_amp):
                    with torch.no_grad():
                        output = self.model(data)
                output = output.float()
            else:
                with torch.no_grad():
                    output = self.model(data)

            result_frag.append(output.data.cpu().numpy())

            # get loss
            if evaluation:
                loss = self.loss(output, label)
                # print(loss.item(), type(loss.item()))
                if not math.isnan(loss):
                    loss_value.append(loss.item())
                label_frag.append(label.data.cpu().numpy())
                
            torch.cuda.empty_cache()

        self.result = np.concatenate(result_frag)

        if evaluation:
            self.label = np.concatenate(label_frag)
            self.epoch_info['mean_loss']= np.mean(loss_value)
            self.show_epoch_info()

            # print("Starting evaluation")
            if hasattr(self.model, 'poly_reduce'): 
                if self.model.poly_reduce:
                    self.model.print_gate(self.io.print_log)

            #### Only conduct tb logging and best model seeking at training phase
            if self.arg.phase != 'test':
                self.writer.add_scalar('val/loss', self.epoch_info['mean_loss'], self.meta_info['iter'])

            # show top-k accuracy
            for k in self.arg.show_topk:
                acc_k = self.show_topk(k)
                if self.arg.phase != 'test':
                    if k == 1: 
                        if acc_k > self.best_top1:
                            filename = 'best_model.pt'
                            self.io.save_model(self.model, filename)
                            self.best_top1 = acc_k
                            self.io.print_log('Save model with best performance, accuracy: {}'.format(\
                                self.best_top1))
                
    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Spatial Temporal Graph Convolution Network')

        # region arguments yapf: disable
        # evaluation
        parser.add_argument('--show_topk', type=int, default=[1, 5], nargs='+', help='which Top K accuracy will be shown')
        # optim 
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--eta', type=float, default=0.9, help='KL divergence penalty')
        parser.add_argument('--varphi', type=float, default=1000, help='Feature map similarity')
        parser.add_argument('--step', type=int, default=[], nargs='+', help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--lookahead', type=str2bool, default=False, help='use lookahead optimizer or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        # endregion yapf: enable

        # Training config, enable mix precision training or not
        parser.add_argument('--mix_precision', type=str2bool, default=False, help='use GPUs or not')
        parser.add_argument('--grad_compression', type=str2bool, default=False, help='use gradient compression or not')
        parser.add_argument('--format', type=str, default='fp32', choices = ['fp32', 'fp16', 'bf16', 'tf32'], \
                            help='use which float point format for computation')
        parser.add_argument('--load_poly', type=str2bool, default=False, \
                            help='load the polynomial weight or not? ')
        parser.add_argument('--freeze_poly', type=str2bool, default=False, \
                            help='Freeze the polynomial weight or not? ')
        parser.add_argument('--freeze_gate_epoch', type=int, default=9999, \
                            help='Freeze the gate parameter at which epoch? ')
        parser.add_argument('--load_poly_teacher', type=str2bool, default=False, \
                            help='load the polynomial weight for teacher or not? ')

        parser.add_argument('--train_gate_from_scratch', type=str2bool, default=False, \
                            help='Train gate from scratch or not? ')
        parser.add_argument('--frame_reduce', type=str2bool, default=False, \
                            help='Reduce the number of frames to 256 or not')
        return parser
