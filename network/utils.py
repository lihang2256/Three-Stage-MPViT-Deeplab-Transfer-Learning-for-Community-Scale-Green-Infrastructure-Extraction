import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict


class _SimpleSegmentationModel(nn.Module):
    def __init__(self, backbone, classifier):
        super(_SimpleSegmentationModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        x = self.classifier(features)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x


class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Arguments:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).

    Examples::

        >>> m = torchvision.models.resnet18(pretrained=True)
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> new_m = torchvision.models._utils.IntermediateLayerGetter(m,
        >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = new_m(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
    """

    def __init__(self, model, return_layers):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.named_children():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out

# PreNorm
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


# FFN
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# MultiHeadAttention
class MultiHeadAttention(nn.Module):
    def __init__(self,
                 dim,
                 heads=8,
                 dim_head=64,
                 dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5  # 1 / sqrt(d_k)

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        # [B, N, h*d] -> [B, N, h, d] -> [B, h, N, d]
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        B, NQ = q.shape[:2]
        B, NK = k.shape[:2]
        B, NV = v.shape[:2]
        q = q.view(B, NQ, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()
        k = k.view(B, NK, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()
        v = v.view(B, NV, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()

        # Q*K^T / sqrt(d_k) : [B, h, N, d] X [B, h, d, N] = [B, h, N, N]
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # softmax
        attn = self.attend(dots)
        # softmax(Q*K^T / sqrt(d_k)) * V ：[B, h, N, N] X [B, h, N, d] = [B, h, N, d]
        out = torch.matmul(attn, v)
        # [B, h, N, d] -> [B, N, h*d]=[B, N, C_out], C_out = h*d
        out = out.permute(0, 2, 1, 3).contiguous().view(B, NQ, -1)

        return self.to_out(out)


# Transformer Encoder Layer
class TransformerEncoder(nn.Module):
    def __init__(self,
                 dim,  # hidden_dim
                 depth,
                 heads,
                 dim_head,
                 mlp_dim=2048,
                 dropout=0.):
        super().__init__()
        self.encoder_layers = nn.ModuleList([])
        for _ in range(depth):
            self.encoder_layers.append(nn.ModuleList([
                PreNorm(dim, MultiHeadAttention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))

    def forward(self, x):
        # x -> [B, N, d_in]
        for attn, ffn in self.encoder_layers:
            x = attn(x) + x
            x = ffn(x) + x
        return x
