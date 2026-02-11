import torch,os
from torch.autograd import Variable
from fadingloader import *
from nn_layers import *
import math

class PositionalEncoder_fixed(nn.Module):
    def __init__(self, lenWord=32, max_seq_len=200, dropout=0.0):
        super().__init__()
        self.lenWord = lenWord
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_seq_len, lenWord)
        for pos in range(max_seq_len):
            for i in range(0, lenWord, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / lenWord)))
                if lenWord != 1:
                    pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / lenWord)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * math.sqrt(self.lenWord)
        seq_len = x.size(1)
        pe = Variable(self.pe[:, :seq_len], requires_grad=False)
        x = x + pe
        return self.dropout(x)


def termination(belief, belief_threshold, bVec, snr, bool_vector):

    T = 0
    B, N, C = belief.shape
    if T == 0:
        mask = (torch.max(belief, dim=2)[0] > belief_threshold) & bool_vector
    else:
        map_vec = torch.tensor([1, 2, 4], device='cuda')
        mask = (torch.max(belief, dim=2)[0] > belief_threshold) & bool_vector

        std1 = 10 ** (-snr * 1.0 / 10 / 2)
        noise = torch.normal(0, std=std1, size=(B, N, C), requires_grad=False).to('cuda') * 0.109375 ** 0.5
        belief_noisy = belief + noise
        belief_noisy = belief_noisy.clamp(min=1e-6)
        belief = belief_noisy / belief_noisy.sum(dim=2, keepdim=True)

        max_vals, pred_class = belief.max(dim=2)  # (B,16), (B,16)
        bVec_mc = torch.matmul(bVec.float(), map_vec.float())
        true_class = bVec_mc.long().contiguous()
        all_correct = (pred_class == true_class).all(dim=1)  # (B,)

        min_of_max = max_vals.min(dim=1).values  # (B,)
        confidence_ok = min_of_max > 0.9

        override = all_correct & confidence_ok  # (B,)
        mask[override] = True

    return mask

def compute_mask(belief, belief_threshold, bVec):
    device = belief.device
    B, N, C = belief.shape  # N=16, C=8

    max_vals, pred_class = belief.max(dim=2)  # (B,16), (B,16)
    mask = max_vals > belief_threshold        # (B,16)

    # --------------------------------------------------
    # 2. bVec -> true class (1~8 -> 0~7)
    # --------------------------------------------------
    weights = torch.tensor([4, 2, 1], device=device)
    true_class = (bVec * weights).sum(dim=2) - 1  # (B,16)

    # --------------------------------------------------
    # 3. 条件一：16 个 block 预测类别全部正确
    # --------------------------------------------------
    all_class_match = (pred_class == true_class).all(dim=1)  # (B,)

    # --------------------------------------------------
    # 4. 条件二：16 个 block 的最大 belief 中的最小值 > 0.1
    # --------------------------------------------------
    min_of_max = max_vals.min(dim=1).values  # (B,)
    confidence_ok = min_of_max > 0.1

    # --------------------------------------------------
    # 5. 同时满足两个条件的 batch，整体 override
    # --------------------------------------------------
    override = all_class_match & confidence_ok  # (B,)

    mask[override] = True

    return mask



class DeepVLF(nn.Module):
    def __init__(self, args):
        super(DeepVLF, self).__init__()
        self.args = args
        self.truncated = args.truncated
        self.pe = PositionalEncoder_fixed()
        ######### Initialize encoder and decoder ###############
        self.Tmodel = Transformer(mod="trx",
                           input_size=args.block_size+2*(args.truncated-1), 
                           block_size=args.block_size, 
                           d_model=args.d_model_trx, 
                           N=args.N_trx, 
                           heads=args.heads_trx, 
                           dropout=args.dropout, 
                           custom_attn=args.custom_attn,
                           multclass=args.multclass,
                           )
        
        self.Rmodel = Transformer(mod="rec",
                           input_size=args.block_class+args.truncated,
                           block_size=args.block_size, 
                           d_model=args.d_model_trx, 
                           N=args.N_trx+1, 
                           heads=args.heads_trx, 
                           dropout=args.dropout, 
                           custom_attn=args.custom_attn,
                           multclass=args.multclass,
                           )


    def power_constraint(self, inputs, isTraining, eachbatch, idx=0, direction='fw'):
        # direction = 'fw' or 'fb'
        if isTraining == 1:
            # training
            this_mean = torch.mean(inputs, 0)
            this_std = torch.std(inputs, 0)
        elif isTraining == 0:
            # test
            if eachbatch == 0:
                this_mean = torch.mean(inputs, 0)
                this_std = torch.std(inputs, 0)
                if not os.path.exists('statistics'):
                    os.mkdir('statistics')
                if not os.path.exists('statistics/{}'.format(self.args.model_name)):
                    os.mkdir('statistics/{}'.format(self.args.model_name))
                torch.save(this_mean, 'statistics/'+self.args.model_name+'/this_mean' + str(idx) + direction)
                torch.save(this_std, 'statistics/'+self.args.model_name+'/this_std' + str(idx) + direction)
            elif eachbatch <= 100:
                this_mean = torch.load('statistics/'+self.args.model_name+'/this_mean' + str(idx) + direction) * eachbatch / (
                            eachbatch + 1) + torch.mean(inputs, 0) / (eachbatch + 1)
                this_std = torch.load('statistics/'+self.args.model_name+'/this_std' + str(idx) + direction) * eachbatch / (
                            eachbatch + 1) + torch.std(inputs, 0) / (eachbatch + 1)
                torch.save(this_mean, 'statistics/'+self.args.model_name+'/this_mean' + str(idx) + direction)
                torch.save(this_std, 'statistics/'+self.args.model_name+'/this_std' + str(idx) + direction)
            else:
                this_mean = torch.load('statistics/'+self.args.model_name+'/this_mean' + str(idx) + direction)
                this_std = torch.load('statistics/'+self.args.model_name+'/this_std' + str(idx) + direction)

        outputs = (inputs - this_mean) * 1.0 / (this_std + 1e-8)
        return outputs

    ########### IMPORTANT ##################
    # We use unmodulated bits at encoder
    #######################################
    def forward_train(self, belief_threshold, eachbatch, bVec, fwd_noise_par,fb_noise_par, ys, optimizer):
        combined_noise_par = fwd_noise_par + fb_noise_par
        bVec_md = 2*bVec-1
        belief = torch.full((self.args.batchSize, 
                             self.args.numb_block, 
                             self.args.block_class), 
                             fill_value=1 /self.args.block_class,
                             requires_grad=False).to(self.args.device)
        mask = torch.zeros(self.args.batchSize, 
                           self.args.numb_block,dtype=torch.bool).to(self.args.device)
        train_log = []
        es=[]
        losses = torch.tensor(0.).to(self.args.device)
        ##############Define lower bound of communication rounds######################
        if belief_threshold<=0.99999:
            mu = 5
        elif 0.99999<belief_threshold<=0.999999:
            mu = 6
        else:
            mu = 7
        eta = 10**(self.args.snr1/10)
        tau_plus = max(mu,round(2*self.args.block_size /math.log((1 + eta),2)))

        for idx in range(self.truncated):
            optimizer.zero_grad()
            ############# Generate the input features ###################################################
            if idx == 0: # phase 0
                src = torch.cat([bVec_md,torch.zeros(self.args.batchSize, 
                                                     self.args.numb_block,
                                                     2*(self.truncated-1)).to(self.args.device)],dim=2)
            else:
                src_new = torch.cat([bVec_md, 
                                     parity_all,
                                     torch.zeros(self.args.batchSize, self.args.numb_block, self.truncated-(idx+1)).to(self.args.device),
                                     combined_noise_par[:, :, :idx],
                                     torch.zeros(self.args.batchSize,self.args.numb_block, self.truncated - (idx + 1)).to(
                                         self.args.device)],dim=2)
                src = torch.where(mask.unsqueeze(2),src,src_new)

            ############# Generate the parity ###################################################
            output = self.Tmodel(src, None,self.pe,idx,self.args.tau_vd)
            parity = self.power_constraint(output,
                                           eachbatch=eachbatch,
                                           idx=idx,
                                           isTraining=1)

            ############# Generate the received symbols ###################################################
            if idx == 0:
                parity_all = parity
                received = torch.cat([parity + fwd_noise_par[:,:,0].unsqueeze(-1),
                                      torch.zeros(self.args.batchSize, self.args.numb_block,self.truncated-1).
                                to(self.args.device),belief], dim= 2)
            else:
                parity_all = torch.cat([parity_all, parity], dim=2)
                received_new = torch.cat([parity_all+ fwd_noise_par[:,:,:idx+1],
                                          torch.zeros(self.args.batchSize,self.args.numb_block,self.truncated-(1+idx)).to(self.args.device),
                                          belief], dim = 2)
                received = torch.where(mask.unsqueeze(2),received,received_new)

            ############# Update the received beliefs ###################################################
            belief_new = self.Rmodel(received, None,self.pe,idx,self.args.tau_vd)
            belief = torch.where(mask.unsqueeze(2), belief, belief_new)

            if idx+1>=tau_plus:
                ############# Backwarding and update gradient ###################################################
                preds = torch.log(belief.contiguous().view(-1, belief.size(-1)))
                mask_flatten = mask.view(-1).to(self.args.device)
                loss = F.nll_loss(preds[~mask_flatten], ys.to(self.args.device)[~mask_flatten])
                if not mask.all():
                    loss_cof = 10**(idx+1-self.args.offset)
                    losses += loss_cof*loss
                    ############# Update the decoding decision ###################################################
                    mask = (torch.max(belief, dim=2)[0] > belief_threshold) & torch.ones(self.args.batchSize,
                                                                                         self.args.numb_block,
                                                                                         dtype=torch.bool).to(self.args.device)
                # if self.args.break_trained and mask.all():
                #     break
                ############# logging early_stop ###################################################
                early_stop = torch.sum(mask) - sum(es[:idx])
                es.append(early_stop.item())
                train_log.append({"round":idx,"loss":loss.item(),"early_stop":early_stop.item()})
            else:
                train_log.append({"round": idx, "loss": None, "early_stop": 0})
        losses.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.args.clip_th)
        optimizer.step()
        return train_log,preds,losses.item()
    

    def forward_evaluate(self, belief_threshold, eachbatch, bVec, fwd_noise_par,fb_noise_par):

        combined_noise_par = fwd_noise_par + fb_noise_par
        bVec_md = 2*bVec-1
        belief = torch.full((self.args.batchSize, 
                             self.args.numb_block, 
                             self.args.block_class), 
                             fill_value=1 /self.args.block_class,
                             requires_grad=False).to(self.args.device)
        mask = torch.zeros(self.args.batchSize, self.args.numb_block,dtype=torch.bool).to(self.args.device)
        test_log = []
        comm_rounds = 0
        es = []
        ##############Define lower bound of communication rounds######################
        if belief_threshold <= 0.99999:
            mu = 5
        elif 0.99999 < belief_threshold <= 0.999999:
            mu = 6
        else:
            mu = 7
        eta = 10 ** (self.args.snr1 / 10)
        tau_plus = max(mu,round(2*self.args.block_size/math.log((1 + eta),2)))


        for idx in range(self.truncated):
            ############# Generate the input features ###################################################
            if idx == 0: # phase 0
                src = torch.cat([bVec_md,torch.zeros(self.args.batchSize, 
                                                     self.args.numb_block,
                                                     2*(self.truncated-1)).to(self.args.device)],dim=2)
            else:
                src_new = torch.cat([bVec_md, 
                                     parity_all,
                                     torch.zeros(self.args.batchSize, self.args.numb_block, self.truncated-(idx+1)).to(self.args.device),
                                     combined_noise_par[:, :, :idx],
                                     torch.zeros(self.args.batchSize,self.args.numb_block, self.truncated - (idx + 1)).to(
                                         self.args.device)],dim=2)
                src = torch.where(mask.unsqueeze(2),src,src_new)

            ############# Generate the parity ###################################################
            output = self.Tmodel(src, None,self.pe,idx,self.args.tau_vd)
            parity = self.power_constraint(output, 
                                           eachbatch=eachbatch, 
                                           idx=idx, 
                                           isTraining=0)

            ############# Generate the received symbols ###################################################
            if idx == 0:
                parity_all = parity
                received = torch.cat([parity + fwd_noise_par[:,:,0].unsqueeze(-1),
                                      torch.zeros(self.args.batchSize, self.args.numb_block,self.truncated-1).
                                to(self.args.device),belief], dim= 2)
            else:
                parity_all = torch.cat([parity_all, parity], dim=2)
                received_new = torch.cat([parity_all+ fwd_noise_par[:,:,:idx+1],
                                          torch.zeros(self.args.batchSize,self.args.numb_block,self.truncated-(1+idx)).to(self.args.device),
                                          belief], dim = 2)
                received = torch.where(mask.unsqueeze(2),received,received_new)

            ############# Update the received beliefs ###################################################
            belief_new = self.Rmodel(received, None,self.pe,idx,self.args.tau_vd)
            belief = torch.where(mask.unsqueeze(2), belief, belief_new)

            if idx+1>=tau_plus:
                ############# Update the decoding decision ###################################################
                num_not_all_decoded = (~mask.all(dim=1)).sum().item()
                comm_rounds += num_not_all_decoded
                # 要在这里加一个termination部分，要读取的是belief，输出mask应该就行了吧
                mask = termination(belief, belief_threshold, bVec, self.args.snr2, bool_vector=torch.ones(self.args.batchSize, self.args.numb_block, dtype=torch.bool).to(self.args.device))

                ############# logging early_stop ###################################################
                early_stop = torch.sum(mask) - sum(es[:idx])
                es.append(early_stop.item())
                test_log.append({"round": idx, "early_stop": early_stop.item()})

                if eachbatch > 100 and mask.all():
                    break
            else: 
                test_log.append({"round": idx,  "early_stop": 0})
                num_not_all_decoded = self.args.batchSize
                comm_rounds += num_not_all_decoded
        return test_log,belief,comm_rounds

    def forward(self,belief_threshold, eachbatch, bVec, fwd_noise_par,fb_noise_par,ys,isTraining=1, fadingLoader=None):
        if self.args.fading_process==1:
            for time in range(self.args.truncated):
                Fading = fadingLoader.generate_fading(time=time, length=self.args.batchSize, l=self.args.numb_block, m=self.args.block_size,device=self.args.device, max_tau=self.args.truncated)
                fwd_noise,fb_noise = fading_process(fwd_noise_par[:,:,(time,)],fb_noise_par[:,:,(time,)], Fading)
                fwd_noise_par[:, :, (time,)] = fwd_noise
                fb_noise_par[:, :, (time,)] = fb_noise
        if isTraining:
            optimizer=self.args.optimizer
            return self.forward_train(belief_threshold, eachbatch, bVec, fwd_noise_par,fb_noise_par, ys,optimizer)
        else:
            return self.forward_evaluate(belief_threshold, eachbatch, bVec, fwd_noise_par,fb_noise_par)



