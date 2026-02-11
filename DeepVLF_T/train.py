import os
import torch
import numpy
import torch.nn.functional as F
from parameters import *
from model import *
from utils import *
from tqdm import tqdm


def train(model, args):
    fadingLoader = DataLoader(args.train, args.real_fading, args.sigma1, args.sigma2)
    max_tau = args.max_tau
    if not os.path.exists('weights'):
        os.mkdir('weights')
    # train loop
    for step in range(args.start_step, args.train_steps):

        # curriculum learning
        snr1, snr2, belief_threshold_rx = curriculum_learning(args, step)
        # Initializing
        bits, oribits, symbolsAndFeedbacks, beliefAndInfor, losses, mask = initialize(args)
        missing_connection = generate_fb_channel_state(args, None)
        args.optimizer.zero_grad()
        # train step
        for time in range(max_tau):
            noise = generate_noise(args, snr1, snr2)
            Fading = fadingLoader.generate_fading(time=time, length=args.batchsize, l=args.l, m=args.m, device=args.device)
            noise = fading_process(noise, Fading)
            missing_connection = generate_fb_channel_state(args, missing_connection)
            symbolsAndFeedbacks, beliefAndInfor, mask = model(time, bits, symbolsAndFeedbacks, beliefAndInfor, mask, missing_connection, noise, belief_threshold_rx)
            belief_tx, belief_rx, _, _ = beliefAndInfor
            if time > args.loss_start_time:
                loss_coefficient, labels = get_loss_parameters(args, time, bits, belief_rx)
                loss = F.cross_entropy(belief_rx.reshape(-1, 8), labels) * loss_coefficient
                # loss = F.nll_loss((F.softmax(belief,dim=-1).reshape(-1,8)+1E-10).log(), oribits.reshape(-1,))*loss_coefficient
                losses += loss
            if model.termination(mask=mask):
                break

        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_th)
        args.optimizer.step()
        args.scheduler.step()

        if step % 100 == 99:
            print('step:{:d}\tloss:{:.5f}'.format(step+1, losses.item()))

        if step % 20000 == 19999:
            if not os.path.exists('weights/{}'.format(args.model_name)):
                os.mkdir('weights/{}'.format(args.model_name))
            saveDir = 'weights/{}/model_weights_{}_{}_'.format(args.model_name,args.snr1,args.snr2) + str(step+1)
            torch.save(model.state_dict(), saveDir)
    else:
        saveDir = 'weights/{}/latest'.format(args.model_name)
        torch.save(model.state_dict(), saveDir)

