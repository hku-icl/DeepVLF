import os
import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *


class AttentionLayer(nn.Module):
    def __init__(self, input_size, num_head):
        super(AttentionLayer, self).__init__()
        self.hidden_layer = 4 * input_size

        self.norm1 = nn.LayerNorm(input_size, eps=1e-5)
        self.norm2 = nn.LayerNorm(input_size, eps=1e-5)

        # self attention
        self.self_attn = MultiHeadAttention(num_head, input_size)
        # dimensions bs * sequenceLen * input_size
        # feedforward net
        self.linear1 = nn.Linear(input_size, self.hidden_layer)
        self.linear2 = nn.Linear(self.hidden_layer, input_size)
        self.activation = F.relu

    def forward(self, x, mask):
        x_att = self.norm1(x)
        x = x + self.self_attn(x_att, x_att, x_att, attn_mask=mask)

        x_ff = self.norm2(x)
        x_ff = self.activation(self.linear1(x_ff))
        x_ff = self.linear2(x_ff)

        x = x + x_ff
        return x


def attention(q, k, v, d_k, attn_mask=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    # bs * heads * sequenceLen * sequenceLen

    if attn_mask is not None:
        mask = attn_mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)

    output = torch.matmul(scores, v)
    # pdb.set_trace()
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, input_size):
        super(MultiHeadAttention, self).__init__()

        self.input_size = input_size
        self.key_dim = input_size // heads
        self.h = heads

        self.q_linear = nn.Linear(input_size, input_size, bias=False)
        self.v_linear = nn.Linear(input_size, input_size, bias=False)
        self.k_linear = nn.Linear(input_size, input_size, bias=False)

        self.FC = nn.Linear(input_size, input_size)

    def forward(self, q, k, v, attn_mask=None, decoding=0):
        bs = q.size(0)
        # perform linear operation and split into N heads
        q = self.q_linear(q).view(bs, -1, self.h, self.key_dim)
        k = self.k_linear(k).view(bs, -1, self.h, self.key_dim)
        v = self.v_linear(v).view(bs, -1, self.h, self.key_dim)

        # transpose to get dimensions bs * heads * sequenceLen * input_size/heads
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next

        scores = attention(q, k, v, self.key_dim, attn_mask)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.input_size)
        # dimensions bs * sequenceLen * input_size
        output = self.FC(concat)

        return output


class Encoder(nn.Module):
    def __init__(self, input_size, m, attention_size, num_layers, infor_size, output_size=1):
        super(Encoder, self).__init__()
        self.Num_layers = num_layers
        self.infor_size = infor_size
        self.m = m
        self.output_size = output_size
        # feature exacter
        self.linear_input1 = nn.Linear(input_size, attention_size * 3)
        self.linear_input2 = nn.Linear(attention_size * 3, attention_size * 3)
        self.linear_input3 = nn.Linear(attention_size * 3, attention_size)
        # Self-attention coefficients
        self.attention_layers = get_layers(AttentionLayer(attention_size, num_head=1), num_layers)
        self.norm = nn.LayerNorm(attention_size, eps=1e-5)
        # feed forward networks
        self.linear_output1 = nn.Linear(attention_size, self.infor_size)
        self.linear_output2 = nn.Linear(self.infor_size, self.output_size)

    def forward(self, x, mask, position_embedding):
        x = x.float()
        x = self.linear_input1(x)
        x = F.relu(x, inplace=False)
        x = self.linear_input2(x)
        x = F.relu(x, inplace=False)
        x = self.linear_input3(x)

        x = position_embedding(x)
        for layer in self.attention_layers:
            x = layer(x, mask)
        x = self.norm(x)

        x = self.linear_output1(x)
        infor = x
        x = F.relu(x, inplace=False)
        parity = self.linear_output2(x)
        return parity, infor


class Decoder(nn.Module):
    def __init__(self, input_size, m, attention_size, num_layers, dropout, infor_size):
        super(Decoder, self).__init__()
        self.Num_layers = num_layers
        self.m = m
        # feature exacter
        self.linear_input1 = nn.Linear(input_size, attention_size * 3)
        self.linear_input2 = nn.Linear(attention_size * 3, attention_size * 3)
        self.linear_input3 = nn.Linear(attention_size * 3, attention_size)
        # Self-attention coefficients
        self.attention_layers = get_layers(AttentionLayer(attention_size, num_head=1), num_layers)
        self.norm = nn.LayerNorm(attention_size, eps=1e-5)
        # feed forward networks
        self.dropout = nn.Dropout(dropout)
        self.linear_output1 = nn.Linear(attention_size, attention_size)
        self.linear_output2 = nn.Linear(attention_size, infor_size)
        self.linear_output3 = nn.Linear(infor_size, 2 ** self.m)

    def forward(self, x, mask, position_embedding):
        x = x.float()

        x = self.linear_input1(x)
        x = F.relu(x, inplace=False)
        x = self.linear_input2(x)
        x = F.relu(x, inplace=False)
        x = self.linear_input3(x)

        x = position_embedding(x)
        for layer in self.attention_layers:
            x = layer(x, mask)
        x = self.norm(x)

        x = self.linear_output1(x)
        x = F.relu(x, inplace=False)
        infor = self.linear_output2(x)
        output = F.relu(infor, inplace=False)
        output = self.dropout(output)
        output = self.linear_output3(output)
        return output, infor


class Decoder_i(nn.Module):
    def __init__(self, input_size, m, attention_size, num_layers, dropout, infor_size, parity_length=1, order='first'):
        super(Decoder_i, self).__init__()
        self.Num_layers = num_layers
        self.m = m
        self.parity_length = parity_length
        self.order = order
        # feature exacter
        self.linear_input0 = nn.Linear(parity_length, 2 ** m)
        self.linear_input1 = nn.Linear(input_size, attention_size * 3)
        self.linear_input2 = nn.Linear(attention_size * 3, attention_size)
        # Self-attention coefficients
        self.attention_layers = get_layers(AttentionLayer(attention_size, num_head=1), num_layers)
        self.norm = nn.LayerNorm(attention_size, eps=1e-5)
        # feed forward networks
        self.dropout = nn.Dropout(dropout)
        self.linear_output1 = nn.Linear(attention_size, attention_size)
        self.linear_output2 = nn.Linear(attention_size, infor_size)
        self.linear_output3 = nn.Linear(infor_size, 2 ** self.m)

    def forward(self, x, mask, position_embedding):
        x = x.float()
        x_1 = self.linear_input0(x[:, :, -self.parity_length:])
        x = torch.cat([x_1, x[:, :, :-self.parity_length]], dim=2)

        x = self.linear_input1(x)
        x = F.relu(x, inplace=False)
        x = self.linear_input2(x)

        x = position_embedding(x)
        for layer in self.attention_layers:
            x = layer(x, mask)
        x = self.norm(x)

        x = self.linear_output1(x)
        x = F.relu(x, inplace=False)
        x = self.dropout(x)
        infor = self.linear_output2(x)
        output = F.relu(infor, inplace=False)
        output = self.linear_output3(output)
        return output, infor


class DeepVLFT(nn.Module):
    def __init__(self, args):
        super(DeepVLFT, self).__init__()
        self.args = args
        self.pe = PositionalEncoder()

        # Transmitter
        self.Tmodel = Encoder(2 ** args.m + 2 * (args.max_tau - 1), args.m, args.attention_size,
                              args.num_layers_encoder, args.infor_size)
        # Receiver
        self.Rmodel = Decoder(args.max_tau, args.m, args.attention_size, args.num_layers_decoder, 0, args.infor_size)

        # Power Reallocation as in deepcode work
        self.total_power_reloc = Power_reallocate(args)

    def generate_power_constraint(self, inputs, step, tau=0, direction='fw'):

        path = 'statistics/ModelName{}'.format(self.args.model_name)

        path_mean = path + '/round{}_mean_{}'.format(tau, direction)
        path_std = path + '/round{}_std_{}'.format(tau, direction)

        # assume the batch size is fixed
        if step == 0:
            this_mean = torch.mean(inputs, 0)
            this_std = torch.std(inputs, 0)

            torch.save(this_mean, path_mean)
            torch.save(this_std, path_std)
        elif step <= 100:
            this_mean = torch.load(path_mean) * (step - 1) / step + torch.mean(inputs, 0) / step
            this_std = torch.load(path_std) * (step - 1) / step + torch.std(inputs, 0) / step

            torch.save(this_mean, path_mean)
            torch.save(this_std, path_std)
        else:
            this_mean = torch.load(path_mean)
            this_std = torch.load(path_std)

        outputs = (inputs - this_mean) * 1.0 / (this_std + 1e-8)
        return outputs

    def generate_power_constraint_for_test(self, fadingLoader, steps=100):
        path = 'statistics/ModelName{}'.format(self.args.model_name)

        if os.path.exists(path):
            warnings.warn(
                "Please make sure you have run function 'generate_power_constraint_for_test' for this model. If so, ignore it. Otherwise remove {} and retry.".format(
                    path), ResourceWarning)

        if not os.path.exists('statistics'):
            os.mkdir('statistics')
        if not os.path.exists(path):
            os.mkdir(path)
        with torch.no_grad():
            for step in range(steps):
                snr1, snr2, belief_threshold_rx = self.args.snr1, self.args.snr2, self.args.belief_threshold_rx
                # Initializing
                bits, oribits, symbolsAndFeedbacks, beliefAndInfor, losses, mask = initialize(self.args)
                missing_connection = generate_fb_channel_state(self.args, None)
                self.args.optimizer.zero_grad()
                # one train step
                for time in range(1, self.args.max_tau + 1):
                    noise = generate_noise(self.args, snr1, snr2)
                    Fading = fadingLoader.generate_fading(time=time-1,length=self.args.batchsize, l=self.args.l, m=self.args.m, device=self.args.device)
                    noise = fading_process(noise, Fading)

                    missing_connection = generate_fb_channel_state(self.args, missing_connection)

                    symbols_tx, symbols_rx, feedbacks, outputs_rx = symbolsAndFeedbacks
                    belief_tx, belief_rx, infor_tx, infor_rx = beliefAndInfor
                    fwd_noise, fb_noise, belief_noise = noise

                    # Encoder
                    input_tx = torch.cat([bits, *symbols_tx, torch.zeros(
                        (self.args.batchsize, self.args.l, self.args.max_tau - 1 - len(symbols_tx))).to(self.args.device),
                                          *feedbacks, torch.zeros(
                            (self.args.batchsize, self.args.l, self.args.max_tau - 1 - len(symbols_tx))).to(self.args.device)], dim=2)
                    parity, _ = self.Tmodel(input_tx, None, self.pe)
                    parity = self.generate_power_constraint(parity, step, tau=time, direction='fw')
                    parity = self.total_power_reloc(parity)
                    parity = torch.where(mask.unsqueeze(2), torch.zeros_like(parity).to(self.args.device), parity)

                    # Channel and Memory
                    received = torch.where(mask.unsqueeze(2), torch.zeros_like(parity).to(self.args.device), parity + fwd_noise)
                    feedback = self.generate_power_constraint(parity + fwd_noise, step, tau=time, direction='fb')
                    feedback = torch.where(mask.unsqueeze(2), torch.zeros_like(parity).to(self.args.device), feedback + fb_noise)

                    symbols_tx.append(parity)
                    symbols_rx.append(received)
                    feedbacks.append(feedback)

                    # Decoder
                    input_rx = torch.cat([*symbols_rx, torch.zeros(
                        (self.args.batchsize, self.args.l, self.args.max_tau - len(symbols_rx))).to(self.args.device)], dim=2)
                    output_rx, _ = self.Rmodel(input_rx, None, self.pe)
                    outputs_rx.append(output_rx)
                    # generate mask
                    belief_rx_new = F.softmax(output_rx, dim=2)
                    belief_rx_new = torch.where(mask.unsqueeze(2), belief_rx, belief_rx_new)
                    # mask will share to Tx immediately
                    belief_tx_new = belief_rx_new + belief_noise
                    belief_tx_new = torch.where(mask.unsqueeze(2), belief_tx, belief_tx_new)

                    symbolsAndFeedbacks, beliefAndInfor, mask = (symbols_tx, symbols_rx, feedbacks, outputs_rx), (
                    belief_tx_new, belief_rx_new, infor_tx, infor_rx), mask

    def power_constraint(self, inputs, isTraining, tau=0, direction='fw'):
        if isTraining == 1:
            # train
            this_mean = torch.mean(inputs, 0)
            this_std = torch.std(inputs, 0)
        elif isTraining == 0:
            # test
            tau+=1
            path = 'statistics/ModelName{}'.format(self.args.model_name)

            if not os.path.exists(path):
                raise Exception("Please run generate_power_constraint_for_test first.")

            path_mean = path + '/round{}_mean_{}'.format(tau, direction)
            path_std = path + '/round{}_std_{}'.format(tau, direction)
            this_mean = torch.load(path_mean)
            this_std = torch.load(path_std)

        outputs = (inputs - this_mean) * 1.0 / (this_std + 1e-8)
        return outputs

    def termination(self, mask=None, belief_tx=None, belief_rx=None, origin_bits=None, belief_threshold_tx=None,
                    isTraining=1):
        if isTraining == 1:
            return False
        elif isTraining == 'test_final':
            stopped = torch.ones((self.args.batchsize,), dtype=torch.bool)
            stopped_num = sum(stopped)
            _, block_index = torch.max(belief_rx[stopped, :, :], dim=2)
            tmp = block_index == origin_bits[stopped, :]
            package_error_num = sum(torch.sum(tmp, dim=1) != tmp.shape[1])
            return stopped, stopped_num, package_error_num
        else:
            # calculate stopped
            masked_belief_tx = torch.where(mask.unsqueeze(2), torch.ones_like(belief_tx).to(self.args.device), belief_tx)
            masked_oribits = torch.where(mask, torch.zeros_like(origin_bits).to(self.args.device), origin_bits)

            threshold_index = torch.min(torch.max(masked_belief_tx, 2).values, 1).values > belief_threshold_tx
            _, block_index = torch.max(masked_belief_tx, dim=2)
            tmp = block_index == masked_oribits
            real_index = torch.min(tmp, dim=1).values
            real_index[~threshold_index] = False
            stopped = real_index
            stopped_num = sum(stopped)
            # calculate package_error_num

            _, block_index = torch.max(belief_rx[stopped, :, :], dim=2)
            tmp = block_index == origin_bits[stopped, :]
            package_error_num = sum(torch.sum(tmp, dim=1) != tmp.shape[1])
            return stopped, stopped_num, package_error_num

    def forward(self, tau, bits, symbolsAndFeedbacks, inforAndBelief, mask, missing_connection, Noise,
                belief_threshold_rx, isTraining=1):
        symbols_tx, symbols_rx, feedbacks, outputs_rx = symbolsAndFeedbacks
        belief_tx, belief_rx, infor_tx, infor_rx = inforAndBelief
        fwd_noise, fb_noise, belief_noise = Noise
        # Encoder
        input_tx = torch.cat([bits, *symbols_tx,
                              torch.zeros((self.args.batchsize, self.args.l, self.args.max_tau - 1 - len(symbols_tx))).to(self.args.device),
                              *feedbacks,
                              torch.zeros((self.args.batchsize, self.args.l, self.args.max_tau - 1 - len(symbols_tx))).to(self.args.device)],
                             dim=2)
        parity, _ = self.Tmodel(input_tx, None, self.pe)
        parity = self.power_constraint(parity, isTraining, tau=tau, direction='fw')
        parity = self.total_power_reloc(parity)
        parity = torch.where(mask.unsqueeze(2), torch.zeros_like(parity).to(self.args.device), parity)


        # Channel and Memory
        received = torch.where(mask.unsqueeze(2), torch.zeros_like(parity).to(self.args.device),  parity + fwd_noise)
        feedback = self.power_constraint(parity + fwd_noise, isTraining, tau=tau, direction='fb')
        feedback = torch.where(mask.unsqueeze(2), torch.zeros_like(parity).to(self.args.device), feedback + fb_noise)
        symbols_tx.append(parity)
        symbols_rx.append(received)
        feedbacks.append(feedback)

        # Decoder
        input_rx = torch.cat(
            [*symbols_rx, torch.zeros((self.args.batchsize, self.args.l, self.args.max_tau - len(symbols_rx))).to(self.args.device)], dim=2)
        output_rx, _ = self.Rmodel(input_rx, None, self.pe)
        outputs_rx.append(output_rx)
        belief_rx_new = F.softmax(output_rx, dim=2)
        belief_rx_new = torch.where(mask.unsqueeze(2), belief_rx, belief_rx_new)

        # mask will share to Tx immediately
        belief_tx_new = belief_rx_new + belief_noise
        belief_tx_new = torch.where(mask.unsqueeze(2), belief_tx, belief_tx_new)

        return (symbols_tx, symbols_rx, feedbacks, outputs_rx), (
            belief_tx_new, belief_rx_new, infor_tx, infor_rx), mask



def get_model(args):
    if args.model_type == 1:
        if args.NFRA_epsilon != 0: raise ValueError(
            'this model is not designed for NFRA channel, but "NFRA_epsilon" is not 0')
        return DeepVLFT(args)
    elif args.model_type == 2:
        if args.NFRA_epsilon != 0: raise ValueError(
            'this model is not designed for NFRA channel, but "NFRA_epsilon" is not 0')
        raise ValueError('wrong model_type')
    elif args.model_type == 3:
        raise ValueError('wrong model_type')
    else:
        raise ValueError('wrong model_type')
