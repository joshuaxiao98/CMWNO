import torch
import torch.nn as nn

import numpy as np

from scipy.special import eval_legendre
from sympy import Poly, legendre, Symbol

def legendreDer(k, x):
    def _legendre(k, x):
        return (2*k+1) * eval_legendre(k, x)
    out = 0
    for i in np.arange(k-1,-1,-2):
        out += _legendre(i, x)
    return out


def get_phi_psi(k, base):
    
    x = Symbol('x')
    if base == 'legendre':
        phi_coeff = np.zeros((k,k))
        phi_2x_coeff = np.zeros((k,k))
        for ki in range(k):
            coeff_ = Poly(legendre(ki, 2*x-1), x).all_coeffs()
            phi_coeff[ki,:ki+1] = np.flip(np.sqrt(2*ki+1) * np.array(coeff_).astype(np.float64))
            coeff_ = Poly(legendre(ki, 4*x-1), x).all_coeffs()
            phi_2x_coeff[ki,:ki+1] = np.flip(np.sqrt(2) * np.sqrt(2*ki+1) * np.array(coeff_).astype(np.float64))
        
    psi1_coeff = np.zeros((k, k))
    psi2_coeff = np.zeros((k, k))
    for ki in range(k):
        psi1_coeff[ki,:] = phi_2x_coeff[ki,:]
        for i in range(k):
            a = phi_2x_coeff[ki,:ki+1]
            b = phi_coeff[i, :i+1]
            prod_ = np.convolve(a, b)
            prod_[np.abs(prod_)<1e-8] = 0
            proj_ = (prod_ * 1/(np.arange(len(prod_))+1) * np.power(0.5, 1+np.arange(len(prod_)))).sum()
            psi1_coeff[ki,:] -= proj_ * phi_coeff[i,:]
            psi2_coeff[ki,:] -= proj_ * phi_coeff[i,:]
        for j in range(ki):
            a = phi_2x_coeff[ki,:ki+1]
            b = psi1_coeff[j, :]
            prod_ = np.convolve(a, b)
            prod_[np.abs(prod_)<1e-8] = 0
            proj_ = (prod_ * 1/(np.arange(len(prod_))+1) * np.power(0.5, 1+np.arange(len(prod_)))).sum()
            psi1_coeff[ki,:] -= proj_ * psi1_coeff[j,:]
            psi2_coeff[ki,:] -= proj_ * psi2_coeff[j,:]

        a = psi1_coeff[ki,:]
        prod_ = np.convolve(a, a)
        prod_[np.abs(prod_)<1e-8] = 0
        norm1 = (prod_ * 1/(np.arange(len(prod_))+1) * np.power(0.5, 1+np.arange(len(prod_)))).sum()

        a = psi2_coeff[ki,:]
        prod_ = np.convolve(a, a)
        prod_[np.abs(prod_)<1e-8] = 0
        norm2 = (prod_ * 1/(np.arange(len(prod_))+1) * (1-np.power(0.5, 1+np.arange(len(prod_))))).sum()
        norm_ = np.sqrt(norm1 + norm2)
        psi1_coeff[ki,:] /= norm_
        psi2_coeff[ki,:] /= norm_
        psi1_coeff[np.abs(psi1_coeff)<1e-8] = 0
        psi2_coeff[np.abs(psi2_coeff)<1e-8] = 0

    phi = [np.poly1d(np.flip(phi_coeff[i,:])) for i in range(k)]
    psi1 = [np.poly1d(np.flip(psi1_coeff[i,:])) for i in range(k)]
    psi2 = [np.poly1d(np.flip(psi2_coeff[i,:])) for i in range(k)]
        
    return phi, psi1, psi2


def get_filter(base, k):
    
    def psi(psi1, psi2, i, inp):
        out = []
        for x in inp:
            if x<=0.5:
                out.append(psi1[i](x))
            else:
                out.append(psi2[i](x))
        return out
    
    if base not in ['legendre']:
        error('Base not supported')
    
    x = Symbol('x')
    H0 = np.zeros((k,k))
    H1 = np.zeros((k,k))
    G0 = np.zeros((k,k))
    G1 = np.zeros((k,k))
    phi, psi1, psi2 = get_phi_psi(k, base)
    if base == 'legendre':
        roots = Poly(legendre(k, 2*x-1)).all_roots()
        x_m = np.array([rt.evalf(20) for rt in roots]).astype(np.float64)
        wm = 1/k/legendreDer(k,2*x_m-1)/eval_legendre(k-1,2*x_m-1)
        
        for ki in range(k):
            for kpi in range(k):
                H0[ki, kpi] = 1/np.sqrt(2) * (wm * phi[ki](x_m/2) * phi[kpi](x_m)).sum()
                G0[ki, kpi] = 1/np.sqrt(2) * (wm * psi(psi1, psi2, ki, x_m/2) * phi[kpi](x_m)).sum()
                H1[ki, kpi] = 1/np.sqrt(2) * (wm * phi[ki]((x_m+1)/2) * phi[kpi](x_m)).sum()
                G1[ki, kpi] = 1/np.sqrt(2) * (wm * psi(psi1, psi2, ki, (x_m+1)/2) * phi[kpi](x_m)).sum()

        H0[np.abs(H0)<1e-8] = 0
        H1[np.abs(H1)<1e-8] = 0
        G0[np.abs(G0)<1e-8] = 0
        G1[np.abs(G1)<1e-8] = 0
        
    return H0, H1, G0, G1

def train(model, train_loader, optimizer, epoch, device, verbose = 0,
    lossFn = None, lr_schedule=None):
        
    if lossFn is None:
        lossFn = nn.MSELoss()

    model.train()
    
    total_loss = 0.

    for batch_idx, (data, target) in enumerate(train_loader):
        
        bs = len(data)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        output = model(data)
        loss = lossFn(output.view(bs, -1), target.view(bs, -1))
        
        loss.backward()
        optimizer.step()
        total_loss += loss.sum().item()
    if lr_schedule is not None: lr_schedule.step()
    
    if verbose>0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
        
    return total_loss/len(train_loader.dataset)


def test(model, test_loader, device, verbose=0, lossFn=None):
    
    model.eval()
    if lossFn is None:
        lossFn = nn.MSELoss()
    
    
    total_loss = 0.
    predictions = []
    
    with torch.no_grad():
        for data, target in test_loader:
            bs = len(data)

            data, target = data.to(device), target.to(device)
            output = model(data)
            
            loss = lossFn(output.view(bs, -1), target.view(bs, -1))
            predictions.extend(output.cpu().data.numpy())
            total_loss += loss.sum().item()
    
    return total_loss/len(test_loader.dataset), predictions


# taken from FNO paper:

# normalization, pointwise gaussian
class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:,sample_idx]+ self.eps # T*batch*n
                mean = self.mean[:,sample_idx]

        # x is in shape of batch*n or T*batch*n
        x = (x * std) + mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

# normalization, Gaussian
class GaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(GaussianNormalizer, self).__init__()

        self.mean = torch.mean(x)
        self.std = torch.std(x)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        x = (x * (self.std + self.eps)) + self.mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


# normalization, scaling by range
class RangeNormalizer(object):
    def __init__(self, x, low=0.0, high=1.0):
        super(RangeNormalizer, self).__init__()
        mymin = torch.min(x, 0)[0].view(-1)
        mymax = torch.max(x, 0)[0].view(-1)

        self.a = (high - low)/(mymax - mymin)
        self.b = -self.a*mymax + high

    def encode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = self.a*x + self.b
        x = x.view(s)
        return x

    def decode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = (x - self.b)/self.a
        x = x.view(s)
        return x
    
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)
