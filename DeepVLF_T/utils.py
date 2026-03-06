import math
import copy
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def curriculum_learning(args, step):
    if step <= args.steps_snr1:
        snr1 = args.CL_snr1 * (1 - step / args.steps_snr1) + (step / args.steps_snr1) * args.snr1
        snr2 = 100
        belief_threshold_rx = 1000 * args.belief_threshold_rx - 999
    elif args.steps_snr1 < step <= 2 * args.steps_snr1:
        snr1 = args.snr1
        snr2 = 100 * (1 - step / args.steps_snr1) + (step / args.steps_snr1) * args.snr2
        belief_threshold_rx = 100 * args.belief_threshold_rx - 99
    elif 2 * args.steps_snr1 < step <= 3 * args.steps_snr1:
        snr1 = args.snr1
        snr2 = args.snr2
        belief_threshold_rx = 10 * args.belief_threshold_rx - 9
    else:
        snr1 = args.snr1
        snr2 = args.snr2
        belief_threshold_rx = args.belief_threshold_rx

    return snr1, snr2, belief_threshold_rx


def initialize(args):
    origin_bits = torch.randint(0, 2 ** args.m, (args.batchsize, args.l))
    bits = F.one_hot(origin_bits, num_classes=8)

    belief_tx = torch.full((args.batchsize, args.l, 2 ** args.m), fill_value=1.0, requires_grad=False).to(
        args.device)
    belief_rx = torch.full((args.batchsize, args.l, 2 ** args.m), fill_value=1.0, requires_grad=False).to(
        args.device)
    infor_tx = torch.zeros((args.batchsize, args.l, args.infor_size)).to(args.device)
    infor_rx = torch.zeros((args.batchsize, args.l, args.infor_size)).to(args.device)
    mask = torch.zeros(args.batchsize, args.l, dtype=torch.bool).to(args.device)
    losses = torch.tensor(0.).to(args.device)
    return bits.float().to(args.device), origin_bits.to(args.device), ([], [], [], []), (
    belief_tx, belief_rx, infor_tx, infor_rx), losses, mask


def Rayleigh(sigma, shape):
    x1 = torch.normal(0, sigma, size=shape)
    x2 = torch.normal(0, sigma, size=shape)
    return (x1 ** 2 + x2 ** 2) ** 0.5


def generate_noise(args, snr1, snr2):
    std1 = 10 ** (-snr1 * 1.0 / 10 / 2)  # forward snr
    std2 = 10 ** (-snr2 * 1.0 / 10 / 2)
    fwd_noise = torch.normal(0, std=std1, size=(args.batchsize, args.l, 1), requires_grad=False).to(args.device)
    fb_noise = torch.normal(0, std=std2, size=(args.batchsize, args.l, 1), requires_grad=False).to(args.device)
    # get this std from a Dir(1,1,...,1) with 2^m ones.
    belief_noise = torch.normal(0, std=std2, size=(args.batchsize, args.l, 2 ** args.m), requires_grad=False).to(
        args.device) * 0.109375 ** 0.5

    if snr2 == 100:
        fb_noise = torch.zeros_like(fb_noise).to(args.device)
        belief_noise = torch.zeros_like(belief_noise).to(args.device)

    return fwd_noise, fb_noise, belief_noise


def calculate_equ_noise(noise, h, h_l2):
    # real(y) = real(x) + [real(n)real(h)+image(n)image(h)]/h^2
    # image(y) = image(y) + [image(n)real(h)-real(n)image(h)]/h^2
    equ_noise = torch.zeros_like(noise)
    l = h.shape[1]
    n = int(l/2+0.5)
    for i in range(0,n):
        equ_noise[:,i] = (noise[:,i]*h[:,i]+noise[:,i+n]*h[:,i+n])/h_l2[:,i]
        equ_noise[:,i+n] = (noise[:,i+n]*h[:,i]-noise[:,i]*h[:,i+n])/h_l2[:,i]
    return equ_noise

def fading_process(noise, fading):
    fwd_noise, fb_noise, belief_noise = noise
    h, h_l2, h_b, h_b_l2 = fading

    fwd_noise = calculate_equ_noise(fwd_noise, h, h_l2)
    fb_noise = calculate_equ_noise(fb_noise, h, h_l2)
    belief_noise = belief_noise.reshape(-1, 16*8, 1)
    belief_noise = calculate_equ_noise(belief_noise, h_b, h_b_l2)
    belief_noise = belief_noise.reshape(-1, 16, 8)

    return fwd_noise, fb_noise, belief_noise


def generate_fb_channel_state(args, missing_connection):
    e1 = args.NFRA_epsilon
    e2 = args.NFRA_tilde_epsilon
    if e1 * e2 == 0:
        return torch.zeros((args.batchsize, 1), dtype=torch.bool).to(args.device)
    elif 0 < e1 < 1 and 0 < e2 < 1:
        if missing_connection is None:
            missing_connection = torch.rand((args.batchsize, 1)) > (1 - e2) / (1 + e1 - e2)
        else:
            missing_connection = torch.rand((args.batchsize, 1)) > 1 - e1 + missing_connection * (e1 - e2)
        return missing_connection.to(args.device)
    else:
        raise ValueError("The values of NFRA_epsilon and NFRA_tilde_epsilon should be in [0,1)")


def get_loss_parameters(args, time, bits, belief):
    if args.loss_coefficient:
        loss_coefficient = 10 ** (time / args.max_tau * 5 - 4)
    else:
        loss_coefficient = 1

    if args.loss_level:
        index = torch.argmax(belief, dim=-1)
        right = (F.one_hot(index, num_classes=8) + bits) == 2
        # 正确的belief使其增加至所需量级，错误的belief使其回归正确
        label = bits * 0.9
        label[right] = belief[right] * 0.1 + 0.9
        labels = bits.float().reshape(-1, 8)
    else:
        labels = bits.float().reshape(-1, 8)
    return loss_coefficient, labels


def get_layers(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class PositionalEncoder(nn.Module):
    def __init__(self, lenWord=32, max_seq_len=200, dropout=0.0):
        super().__init__()
        self.lenWord = lenWord
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, lenWord)
        for pos in range(max_seq_len):
            for i in range(0, lenWord, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / lenWord)))
                if lenWord != 1:
                    pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / lenWord)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.lenWord)
        # add constant to embedding
        seq_len = x.size(1)
        pe = Variable(self.pe[:, :seq_len], requires_grad=False)
        x = x + pe
        return self.dropout(x)


class Power_reallocate(nn.Module):
    def __init__(self, args):
        super(Power_reallocate, self).__init__()
        self.args = args
        self.weight1 = torch.nn.Parameter(torch.Tensor(args.l, 1), requires_grad=True)
        self.weight1.data.uniform_(1.0, 1.0)

    def forward(self, inputs):
        # phase-level power allocation
        self.wt1 = torch.sqrt(self.weight1 ** 2 * (self.args.l / torch.sum(self.weight1 ** 2)))
        inputs = inputs * self.wt1  # block_wise scaling

        return inputs


class DataLoader:
    def __init__(self, train, real_fading, sigma1=0, sigma2=0):
        if train == 1:
            path = 'hmatrix_train.mat'
        elif train == 0:
            path = 'hmatrix_test.mat'
        file = h5py.File(path, 'r')
        self.shape = file['hmatrix'].shape
        real = torch.tensor(file['hmatrix'][:]['real']).unsqueeze(2)  # data length * 8 * 1
        imag = torch.tensor(file['hmatrix'][:]['imag']).unsqueeze(2)  # data length * 8 * 1
        self.h = torch.cat((real, imag),dim=1) # # data length * 16 * 1,  h[:,:8,:] is real, h[:, 8:, :] is image
        self.h_l2 = real**2+imag**2 # data length * 8 * 1
        self.real_fading = real_fading
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.i = None
        self.j = None

    def generate_fading(self, time, length=2048, l=16, m=3, device=torch.device('cpu'), max_tau=20):
        if self.real_fading == 1:
            if time == 0:
                self.i = torch.randint(0, self.shape[0]-max_tau, (length,))
                self.j = torch.randint(0, self.shape[0]-max_tau, (length * 8,))
            else:
                # 对每个batch，要取max_tau个连续的h
                self.i += 1
                self.j += 1

            h = self.h[self.i]   # length * 16 * 1
            h_l2 = self.h_l2[self.i]   # length * 8 * 1
            h_b = self.h[self.j] * 0.125 ** 0.5
            h_b_l2 = self.h_l2[self.j] * 0.125 ** 0.5

            return h.to(device), h_l2.to(device), h_b.reshape(length,8*16,1).to(device), h_b_l2.reshape(length,8*8,1).to(device)

        # generated fading
        elif self.real_fading == 0:
            if self.sigma1 == 0:
                fw_h = 1
            else:
                fw_h = Rayleigh(self.sigma1, (length, l, 1)).to(device)

            if self.sigma2 == 0:
                fb_h = 1
                belief_h = 1
            else:
                fb_h = Rayleigh(self.sigma1, (length, l, 1)).to(device)
                belief_h = Rayleigh(self.sigma1, (length, l, 2 ** m)).to(device)

        return fw_h, fb_h, belief_h

    def get_shape(self):
        return self.shape


if __name__ == '__main__':
    path = 'hmatrix_test.mat'
    file = h5py.File(path, 'r')
    shape = file['hmatrix'].shape
    # real = torch.tensor(file['hmatrix'][2:10]['real'])
    imag = torch.tensor(file['hmatrix'][:10, :2]['imag'])

    print(imag)
    print(imag.reshape(2, 10))
    print(imag.unsqueeze(2).shape)
    print(imag.shape)
