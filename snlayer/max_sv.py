import torch
import torch.nn.functional as F


def _l2normalize(v,eps=1e-12):
    return v/(torch.norm(v)+eps)


def max_singular_value(W,u=None,iteration=1):
    if not iteration>=1:
        raise ValueError('Power iteration should be a positive integer')
    if u is None:
        u=torch.FloatTensor(1,W.size(0)).normal_(0,1).cuda()
    _u=u
    for _ in range(iteration):
        _v=_l2normalize(torch.matmul(_u,W.data),eps=1e-12)
        _u=_l2normalize(torch.matmul(_v,torch.transpose(W.data,0,1)),eps=1e-12)
    sigma=torch.sum(F.linear(_u,torch.transpose(W.data,0,1))*_v)
    return sigma,_u
