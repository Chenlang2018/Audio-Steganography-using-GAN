import torch
import torch.nn.functional as F
from torch.nn.modules import Linear
from snlayer.max_sv import max_singular_value


class SNLinear(Linear):
    def __init__(self,in_features,out_features,bias=True):
        super(SNLinear,self).__init__(in_features,out_features,bias)
        self.register_buffer('u',torch.Tensor(1,out_features).normal_())

    @property
    def W_(self):
        w_mat=self.weight.view(self.weight.size(0),-1)
        sigma,_u=max_singular_value(w_mat,self.u)
        self.u.copy_(_u)
        return self.weight/sigma

    def forward(self,input):
        return F.linear(input,self.W_,self.bias)
