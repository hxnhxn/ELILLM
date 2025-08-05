import numpy as np
import torch
from torch.distributions.normal import Normal
class AcquisitionFunction:
    def __init__(self, surrogate, device, maxy=-1000.0):
        self.surrogate = surrogate
        self.maxy = maxy
        self.device = device

    def update(self, maxy):
        self.maxy = maxy

    # def EI(self, x):
    #     mu, std = self.surrogate.batch_predict(x)
    #     maxy = self.maxy
    #     gamma = (mu - maxy) / std
    #
    #     m = Normal(torch.tensor([0.0]).to(dtype=mu.dtype), torch.tensor([1.0]).to(dtype=mu.dtype))
    #     # print("m in EI",m.log_prob(gamma).exp(),"gamma=",gamma,"m.cdf(gamma)=",m.cdf(gamma),"(mu-maxy)/std",mu,maxy,std)
    #     pdfgamma = m.log_prob(gamma).exp()
    #     cdfgamma = m.cdf(gamma)
    #     result = std * (pdfgamma + gamma * cdfgamma)
    #     return result

    # Minimize optimization
    # Follow GP-UCB
    def LCB(self, x, t, delta = 0.1):
        mu, std = self.surrogate.batch_predict(x)
        beta_t = 2 * np.log(t ** 2 * np.pi ** 2 / (6 * delta))
        kappa_t = np.sqrt(beta_t)
        result = mu - kappa_t * std
        return result, mu, std
