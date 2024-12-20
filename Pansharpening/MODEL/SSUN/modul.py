import torch
import torch.nn as nn
import torch.nn.functional as F

def linear_attn(q, k, x, EPS=1e-7):
    l1, d1  = q.shape[-2:]
    l2, d2 = x.shape[-2:]
    k = k.transpose(-2, -1)
    if l1*d1*l2+l1*l2*d2<= d2*l2*d1+d2*d1*l1:
        q = q@k
        q = q/(q.sum(dim=-1, keepdim=True)+EPS)
        x = q@x
    else:
        x = q@(k@x)
        q = q@k.sum(dim=-1, keepdim=True) + EPS
        x = x/q
    return x
class convMultiheadAttetionV2(nn.Module):

    def __init__(self, convDim, numHeads, patchSize, qkScale=None, qkvBias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.convDim = convDim
        self.numHeads = numHeads
        self.patchSize = patchSize
        self.relu = nn.ReLU()
        self.register_buffer('one', None)
        self.qkv = nn.Conv2d(self.convDim, self.convDim * 3, 1, 1, 0)
        self.proj = nn.Conv2d(self.convDim, self.convDim, 3, 1, 1)

    def forward(self, x, mask=None, padsize=0):
        B, C, H, W = x.shape
        qkv = self.qkv(x).reshape(B, 3, self.convDim, H, W).transpose(0, 1)
        q, k, x = qkv[0], qkv[1], qkv[2]
        del qkv
        q, k = self.relu(q), self.relu(k)
        q = F.unfold(q, self.patchSize, padding=(self.patchSize - 1) // 2, stride=1)  # [B, numHead, headDim, H, W]
        k = F.unfold(k, self.patchSize, padding=(self.patchSize - 1) // 2, stride=1)
        x = x.view(B, C, H * W)
        if q.shape[1] % self.numHeads != 0:
            padsize = (q.shape[1] // self.numHeads + 1) * self.numHeads - q.shape[1]
            q = F.pad(q, [0, 0, 0, padsize], mode='replicate')
            k = F.pad(k, [0, 0, 0, padsize], mode='replicate')
            x = F.pad(x, [0, 0, 0, padsize], mode='replicate')
        q = q.view(B, self.numHeads, -1, H * W).transpose(-1, -2)
        k = k.view(B, self.numHeads, -1, H * W).transpose(-1, -2)
        x = x.view(B, self.numHeads, -1, H * W).transpose(-1, -2)
        x = linear_attn(q, k, x)
        x = x.transpose(-1, -2).contiguous().view(B, -1, H * W)
        if padsize != 0:
            x = x[:, :-padsize, :]
        x = x.view(B, C, H, W)
        x = self.proj(x)
        return x
    def flops(self, x):
        B, C, H, W = x.shape
        d1 = C * self.patchSize ** 2  # for q and k
        l1 = H * W
        d2 = C
        flops = 0
        flops += l1 * C * 3 * C
        flops += min(l1 * d1 * l1 + l1 * l1 * d2, l1 * d1 * d2 + d1 * l1 * d2)
        flops += l1 * C * 9 * C  #
        return flops
class FeedForward(nn.Module):
    def __init__(self, dim):
        super(FeedForward, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.gelu1 = nn.GELU()
        self.depthConv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.gelu2 = nn.GELU()
        self.conv2 = nn.Conv2d(dim, dim, 1, 1, 0)
    def forward(self, x):
        x = self.conv1(x)
        x = self.gelu1(x)
        x = self.depthConv(x)
        x = self.gelu2(x)
        x = self.conv2(x)
        return x
class speMultiAttn(nn.Module):
    def __init__(self, convDim, numHeads=8, poolSize=4, ksize=9) -> None:
        super().__init__()
        self.numHeads = numHeads
        self.convDim = convDim
        self.poolSize = poolSize
        self.avepool = nn.AdaptiveAvgPool2d(self.poolSize)
        self.relu = nn.ReLU()
        self.q = nn.Linear(self.poolSize ** 2, self.poolSize ** 2 * self.numHeads)
        self.k = nn.Linear(self.poolSize ** 2, self.poolSize ** 2 * self.numHeads)
        self.v = nn.Conv2d(self.convDim, self.convDim, 1, 1, 0)
        self.proj = nn.Conv2d(self.convDim, self.convDim, ksize, 1, (ksize - 1) // 2, groups=self.convDim)

    def forward(self, x, padsize=0):
        B, C, H, W = x.shape
        q = self.avepool(x)
        q = q.view(B, C, -1)
        q = self.q(q)
        k = self.avepool(x)
        k = k.view(B, C, -1)
        k = self.k(k)
        x = self.v(x)
        q = self.relu(q).view(B, C, self.numHeads, -1)
        k = self.relu(k).view(B, C, self.numHeads, -1)

        x = x.view(B, C, -1)
        if x.shape[2] % self.numHeads != 0:
            padsize = (x.shape[2] // self.numHeads + 1) * self.numHeads - x.shape[2]
            x = F.pad(x, [0, padsize, 0, 0], mode='replicate')
        x = x.view(B, C, self.numHeads, -1)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        x = x.transpose(1, 2)
        x = linear_attn(q, k, x)
        x = x.transpose(1, 2).contiguous().view(B, C, -1)
        if padsize:
            x = x[:, :, :-padsize]
        x = x.view(B, C, H, W)
        x = self.proj(x)
        return x
        # pass
class spaTransBlock(nn.Module):
    def __init__(self, convDim):
        super().__init__()
        self.multiAttn = convMultiheadAttetionV2(convDim, 8, 1)
        self.ffn = FeedForward(convDim)
        self.reweight = nn.Parameter(torch.zeros(1)) if False else 1
    def forward(self, x):
        x = x + self.multiAttn(x) * self.reweight
        x = x + self.ffn(x) * self.reweight
        return x
class speTransBlock(nn.Module):
    def __init__(self, convDim):
        super().__init__()
        self.multiattn = speMultiAttn(convDim)
        self.ffn = FeedForward(convDim)
        self.reweight = nn.Parameter(torch.zeros(1)) if False else 1
    def forward(self, x):
        x = x + self.multiattn(x) * self.reweight
        x = x + self.ffn(x) * self.reweight
        return x


