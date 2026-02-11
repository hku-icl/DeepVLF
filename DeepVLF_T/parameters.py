import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=int, default=1, help="1:DeepVLFT, 2:"None", 3:"None")

    # Sequence arguments
    parser.add_argument('--snr1', type=float, default=2, help="Transmission SNR")
    parser.add_argument('--snr2', type=float, default=100, help="Feedback SNR")
    parser.add_argument('--real_fading', type=int, default=1, help="1:from .mat or 0:from simulation")
    parser.add_argument('--sigma1', type=float, default=0, help="Transmission fading, 0 for AWGN channel: mean=\sqrt(\pi/2) * \sigma")
    parser.add_argument('--sigma2', type=float, default=0, help="Feedback fading")

    parser.add_argument('--CL_snr1',type=float, default=3, help="T SNR for curriculum learning")
    parser.add_argument('--steps_snr1',type=int, default=40000, help="steps for curriculum learning")
    parser.add_argument('--k', type=int, default=48, help="Sequence length")
    parser.add_argument('--l', type=int, default=16, help='num of little blocks')
    parser.add_argument('--m', type=int, default=3, help='bits of one little block')
    parser.add_argument('--max_tau', type=int, default=20, help='maximum number of transmit, 0=infinite')

    # training parameters
    parser.add_argument('--train', type=int, default=1, help='train:1, test:0')
    parser.add_argument('--model_name', type=str, default='tmp')
    # parser.add_argument('--model_weights', type=str, default='C:\\Users\\Administrator\\Desktop\\latest', help='Only when testing')
    # parser.add_argument('--model_weights', type=str, default='D:\\Desktop\\temp\\VLFT-i\\latest', help='Only when testing')
    parser.add_argument('--model_weights', type=str, default='weights/VLFT/vt_9/latest',help='Only when testing')
    parser.add_argument('--start_step', type=int, default=0, help='the start step for retrained model; if not 0, start model is needed')
    parser.add_argument('--start_model', type=str, default='None', help='the path of retrained model')
    parser.add_argument('--train_steps', type=int, default=160000, help="number of total steps to train")
    parser.add_argument('--test_steps', type=int, default=10000, help="number of total steps to test")

    parser.add_argument('--loss_start_time', type=int, default=2, help='from when the loss is calculated')
    parser.add_argument('--batchsize', type=int, default=100, help="batch size of one step")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--wd', type=float, default=0.01, help="weight decay")
    parser.add_argument('--clip_th', type=float, default=0.5, help="clipping threshold")
    parser.add_argument('--belief_threshold_tx', type=float, default=0.1, help="decode at transmitter when all bit groups over the threshold of belief")
    parser.add_argument('--belief_threshold_rx', type=float, default=0.9999, help="decode at receiver when one bit group over the threshold of belief")

    # model parameters
    parser.add_argument('--attention_size', type=int, default=32)
    parser.add_argument('--infor_size', type=int, default=8)
    parser.add_argument('--num_layers_encoder', type=int, default=2)
    parser.add_argument('--num_layers_decoder', type=int, default=3)
    parser.add_argument('--loss_coefficient', type=int, default=1, help='0:close, 1:open')
    parser.add_argument('--loss_level', type=int, default=1, help='0:close, 1:open')
    parser.add_argument('--delta_belief', type=int, default=1, help='0:close, 1:open')
    parser.add_argument('--replace_type', type=float, default=99, help='0:hard, (0,1):soft, 99:gate')

    parser.add_argument('--NFRA_epsilon', type=float, default=0, help='probability of missing connection when successfully connection last in previous round')
    parser.add_argument('--NFRA_tilde_epsilon', type=float, default=0, help='probability of missing connection when missing connection last in previous round')


    args = parser.parse_args()

    return args
