import torch

from utils import *
from model import get_model
from parameters import args_parser
from test import test
from train import train
import os

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICE'] = '0'
    ##################### Initialize the parameters ################
    print('------------ Initialize the model ------------')
    args = args_parser()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('train=', args.train, 'snr1=', args.snr1, 'snr2=', args.snr2, '\nK=', args.k, 'm=', args.m,'max_t=', args.max_tau,
          '\nbatch size=', args.batchsize, 'belief_threshold_tx=', args.belief_threshold_tx, 'belief_threshold_rx=', args.belief_threshold_rx, sep='\t')
    ##################### Initialize the model ################
    print('------------ Initialize the model ------------')
    model = get_model(args)
    model = model.to(args.device)

    ##################### Initialize the optimizer ################
    args.optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08,
                                       weight_decay=args.wd, amsgrad=False)

    lr_lambda = lambda epoch: (1 - epoch / args.train_steps)
    args.scheduler = torch.optim.lr_scheduler.LambdaLR(args.optimizer, lr_lambda=lr_lambda)




    if args.start_step !=0:
        model.load_state_dict(torch.load(args.start_model))
    if args.train == 1:
        train(model, args)
    else:
        test(model, args)

