import torch
import numpy
import torch.nn.functional as F
from parameters import *
from model import *
from utils import *
from tqdm import tqdm
import warnings


def test(model, args):
    fadingLoader = DataLoader(args.train, args.real_fading, args.sigma1, args.sigma2)
    model.load_state_dict(torch.load(args.model_weights,map_location=args.device))
    max_tau = args.max_tau
    sum_PE_num = 0
    sum_average_time = 0
    sum_transmitted_symbols = 0
    # initializing power constraint
    model.generate_power_constraint_for_test(fadingLoader)
    # test loop
    for batch in range(1, args.test_steps+1):
        snr1, snr2, belief_threshold_rx = args.snr1, args.snr2, args.belief_threshold_rx
        bits, oribits, symbolsAndFeedbacks, beliefAndInfor, _, mask = initialize(args)
        missing_connection = generate_fb_channel_state(args, None)
        stopped_num = 0

        for time in range(max_tau):
            noise = generate_noise(args, snr1, snr2)
            Fading = fadingLoader.generate_fading(time=time, length=args.batchsize, l=args.l, m=args.m, device=args.device)
            noise = fading_process(noise, Fading)

            missing_connection = generate_fb_channel_state(args, missing_connection)
            sum_average_time += args.batchsize - stopped_num
            sum_transmitted_symbols += args.batchsize * args.l - torch.sum(mask)
            with torch.no_grad():
                symbolsAndFeedbacks, beliefAndInfor, mask = model(time, bits, symbolsAndFeedbacks, beliefAndInfor, mask, missing_connection, noise, belief_threshold_rx, isTraining=0)
                belief_tx, belief_rx, _, _ = beliefAndInfor
                if time != max_tau-1:
                    stopped, stopped_num, package_error_num = model.termination(mask=mask, belief_tx=belief_tx, belief_rx=belief_rx, origin_bits=oribits, belief_threshold_tx=args.belief_threshold_tx, isTraining=0)
                else:
                    stopped, stopped_num, package_error_num = model.termination(mask=mask, belief_tx=belief_rx, belief_rx=belief_rx, origin_bits=oribits, belief_threshold_tx=0, isTraining='test_final')
                mask[stopped]=True
                sum_PE_num+=package_error_num
            if stopped_num == args.batchsize:
                break
        max_time = time
        if batch % 100 == 0:
            print('Batch:{}\tPER:{}\tNormalized average time:{}\tAverage Time:{}\tMax Time:{}'.format(batch, sum_PE_num / batch / args.batchsize,
                                                                      sum_transmitted_symbols / batch / args.batchsize / args.l,
                                                                      sum_average_time / batch / args.batchsize, max_time+1))
