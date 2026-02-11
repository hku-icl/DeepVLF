import h5py
import torch


def calculate_equ_noise(noise, h, h_l2):
    # real(y) = real(x) + [real(n)real(h)+image(n)image(h)]/h^2
    # image(y) = image(y) + [image(n)real(h)-real(n)image(h)]/h^2
    equ_noise = torch.zeros_like(noise)
    l = h.shape[1]
    n = int(l/2+0.5)
    for i in range(0,n):
        equ_noise[:,i] = (noise[:,i]*h[:,i]+noise[:,i+n]*h[:,i+n])/h_l2[:,i]
        equ_noise[:,i+n] = (noise[:,i+n]*h[:,i]+noise[:,i]*h[:,i+n])/h_l2[:,i]
    return equ_noise


def fading_process(fwd_noise,fb_noise, fading):
    # noise : length * 16 * 10
    h, h_l2 = fading # bs * 16 * 1

    fwd_noise = calculate_equ_noise(fwd_noise, h, h_l2)
    fb_noise = calculate_equ_noise(fb_noise, h, h_l2)

    return fwd_noise, fb_noise


class DataLoader:
    def __init__(self, train, real_fading, sigma1=0, sigma2=0):
        if train == 1:
            path = '../DeepVLFT/hmatrix_train.mat'
        elif train == 0:
            path = '../DeepVLFT/hmatrix_test.mat'
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

    def generate_fading(self,time, length=2048, l=16, m=3, device=torch.device('cpu'), max_tau=20):
        if self.real_fading == 1:
            if time == 0:
                self.i = torch.randint(0, self.shape[0]-max_tau, (length,))
            else:
                # 对每个batch，要取max_tau个连续的h
                self.i += 1

            h = self.h[self.i]   # length * 16 * 1
            h_l2 = self.h_l2[self.i]   # length * 8 * 1

            return h.to(device), h_l2.to(device)


        # generated fading
        elif self.real_fading == 0:
            raise 'wrong version'

    def get_shape(self):
        return self.shape
