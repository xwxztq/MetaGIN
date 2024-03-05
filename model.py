import numpy as np
import torch as pt
import torch.nn as nn
import torch.nn.parameter as nnp
import torch.nn.functional as nnf

from torch_scatter import segment_coo, segment_csr
from torch.distributions.log_normal import LogNormal


DIM_WARP   = 32
EMBED_POS  = 16           # random walk positional embedding
EMBED_ATOM = 90           # zero for masked
EMBED_BOND = [21, 3, 4]   # zero for masked
RESCALE_GRAPH, RESCALE_NODE, RESCALE_EDGE = (1, 3), (1, 2), (1, 1)


class DenseLayer(nn.Module):
    def __init__(self, width_in, width_out, bias=False, nhead=1):
        super().__init__()
        self.conv = nn.Conv1d(width_in, width_out, 1, bias=bias, groups=nhead)

    def forward(self, x):
        return self.conv(x.unsqueeze(-1)).squeeze(-1)

# ReZero: https://arxiv.org/abs/2003.04887
# LayerScale: https://arxiv.org/abs/2103.17239
class ScaleLayer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.scale = nnp.Parameter(pt.ones(width))
        self.bias = nnp.Parameter(pt.zeros(width))

    def forward(self, x):
        return self.scale * x + self.bias


# GLU: https://arxiv.org/abs/1612.08083
# GLU-variants: https://arxiv.org/abs/2002.05202
class GatedLinearBlock(nn.Module):
    def __init__(self, width, nhead, resca_norm=1, resca_act=1, width_in=None, skip_pre=False, width_out=None):
        super().__init__()
        self.width = width
        if width_in is None: width_in = width
        width_norm = DIM_WARP * nhead * resca_norm
        width_act  = DIM_WARP * nhead * resca_act
        if width_out is None: width_out = width

        if skip_pre:
            self.pre = nn.GroupNorm(nhead, width_norm, affine=False)
        else:
            self.pre = nn.Sequential(DenseLayer(width_in, width_norm),
                           nn.GroupNorm(nhead, width_norm, affine=False))
        self.gate  = nn.Sequential(DenseLayer(width_norm, width_act, nhead=nhead),
                         nn.ReLU(), nn.Dropout(0.1))
        self.value = DenseLayer(width_norm, width_act, nhead=nhead)
        self.post  = DenseLayer(width_act, width_out)

    def forward(self, x, gate_bias=None, out_norm=False):
        xn = self.pre(x)
        if gate_bias is None:
            xx = self.gate(xn) * self.value(xn)
        else:
            xx = self.gate(xn + gate_bias) * self.value(xn)
        xx = self.post(xx)
        if out_norm:
            return xx, xn
        else:
            return xx

class ConvBlock(nn.Module):
    def __init__(self, width, nhead, bond_size):
        super().__init__()
        self.width = width
        width_norm = DIM_WARP * nhead * RESCALE_EDGE[0]

        self.src = DenseLayer(width, width_norm)
        self.tgt = DenseLayer(width, width_norm)
        self.conv_encoder = nn.EmbeddingBag(bond_size, width_norm, scale_grad_by_freq=True, padding_idx=0)
        self.msg = GatedLinearBlock(width, nhead, *RESCALE_EDGE, skip_pre=True)
        print('##params[conv]:', np.sum([np.prod(p.shape) for p in self.parameters()]))

    def forward(self, x, edge_idx, edge_attr):
        xx = self.src(x)[edge_idx[0]] + self.tgt(x)[edge_idx[1]]
        xx = self.msg(xx, self.conv_encoder(edge_attr))
        xx = segment_coo(xx, edge_idx[1], dim_size=len(x), reduce='sum')
        return xx

# GIN-virtual: https://arxiv.org/abs/2103.09430
class VirtBlock(nn.Module):
    def __init__(self, width, nhead):
        super().__init__()
        self.width = width
        width_norm = DIM_WARP * nhead * RESCALE_NODE[0]

        self.msg  = GatedLinearBlock(width, nhead, *RESCALE_NODE)
        self.gate = nn.Sequential(DenseLayer(width, width_norm),  # to eval
                        nn.GroupNorm(nhead, width_norm, affine=False),
                        ScaleLayer(width_norm))
        print('##params[virt]:', np.sum([np.prod(p.shape) for p in self.parameters()]))

    def forward(self, x, virt_res, batch, idxptr):
        if virt_res is None:
            virt_res = segment_csr(x, idxptr, reduce='sum')
        else:
            virt_res = segment_csr(x, idxptr, reduce='sum') + virt_res
        xx = self.msg(virt_res[batch], self.gate(x))
        return xx, virt_res

# MetaFormer: https://arxiv.org/abs/2210.13452
class MetaBlock(nn.Module):
    def __init__(self, width, nhead, resca_norm=1, resca_act=1, use_residual=True):
        super().__init__()
        self.width = width

        self.mix0 = ScaleLayer(width)
        self.mix1 = ScaleLayer(width) if use_residual else None
        self.ffn  = GatedLinearBlock(width, nhead, resca_norm, resca_act)
        print('##params[meta]:', np.sum([np.prod(p.shape) for p in self.parameters()]), use_residual)

    def forward(self, x, res):
        xx = self.mix0(x) + res
        if self.mix1 is None:
            xx = self.ffn(xx)
        else:
            xx = self.mix1(xx) + self.ffn(xx)
        return xx

class HeadBlock(nn.Module):
    def __init__(self, width, nhead):
        super().__init__()
        self.width = width

        self.head = GatedLinearBlock(width, nhead, 1, 1, width_out=1)
        self.bias = nnp.Parameter(pt.ones(1))
        print('##params[head]:', np.sum([np.prod(p.shape) for p in self.parameters()]))

    def forward(self, x, idxptr):
        xx = segment_csr(x, idxptr, reduce='sum')
        xx, xn = self.head(xx, out_norm=True)
        xx = (xx + self.bias) * 5.6  # 98% between 3.18 and 8.77
        return xx, xn


# VoVNet: https://arxiv.org/abs/1904.09730
# GNN-AK: https://openreview.net/forum?id=Mspk_WYKoEH
class MetaKernel(nn.Module):
    def __init__(self, width, nhead, hop, kernel, skip_virt=False):
        super().__init__()
        self.width = width
        self.hop = hop
        self.kernel = kernel

        self.virt = nn.ModuleList()
        self.conv = nn.ModuleList()
        self.meta = nn.ModuleList()
        for k in range(kernel):
            if not skip_virt:
                self.virt.append(VirtBlock(width, nhead))
            else:
                self.virt.append(None)
            for h in range(hop):
                self.conv.append(ConvBlock(width, nhead, EMBED_BOND[h]))
            self.meta.append(MetaBlock(width, nhead, *RESCALE_NODE, k<kernel-1))
        size = np.sum([np.prod(p.shape) for n, p in self.named_parameters()])
        print('##params:', size, '%.2fM' % (size/1024/1024))

    def forward(self, x, virt, edge, batch, idxptr):
        x_kernel = x_hop = x
        for k in range(self.kernel):
            if self.virt[k] is not None:
                x_out, virt = self.virt[k](x_kernel, virt, batch, idxptr)
            else:
                x_out = 0
            for h in range(self.hop):
                edge_index, edge_attr = edge[h]
                x_hop = self.conv[k*self.hop+h](x_hop, edge_index, edge_attr)
                x_out = x_out + x_hop
            x_kernel = x_hop = self.meta[k](x_kernel, x_out)
        return x_kernel, virt

# GIN: https://openreview.net/forum?id=ryGs6iA5Km
# Graph PE: https://arxiv.org/abs/2110.07875
class MetaGIN(nn.Module):
    def __init__(self, width, nhead, depth, kernel, hop):
        super().__init__()
        self.width = width
        self.nhead = nhead
        self.depth = depth
        self.kernel = kernel
        self.hop = hop
        print('#model:', width, nhead, depth, kernel, hop)

        self.atom_encoder = nn.EmbeddingBag(EMBED_ATOM, width, scale_grad_by_freq=True, padding_idx=0)
        self.atom_pos  = GatedLinearBlock(width, nhead, *RESCALE_NODE, width_in=EMBED_POS)

        self.kernel = nn.ModuleList()
        for layer in range(depth):
            self.kernel.append(MetaKernel(width, nhead, hop, kernel[layer], layer==0))

        self.head = HeadBlock(width, nhead)
        size = np.sum([np.prod(p.shape) for n, p in self.named_parameters()])
        print('#params:', size, '%.2fM' % (size/1024/1024))

    def getEdge(self, graph):  # to eval
        idx1, attr1 = graph['bond'].edge_index, graph['bond'].edge_attr
        edge = [[idx1, attr1]]

        idx2, attr2 = graph['angle'].edge_index, graph['angle'].edge_attr
        attr2.clamp_(None, EMBED_BOND[1]-1)
        edge += [[idx2, attr2]]

        idx3, attr3 = graph['torsion'].edge_index, graph['torsion'].edge_attr
        attr3.clamp_(None, EMBED_BOND[2]-1)
        edge += [[idx3, attr3]]

        return edge

    def forward(self, graph):
        x, z = graph['atom'].x, graph['atom'].pos_rw
        batch, idxptr = graph['atom'].batch, graph['atom'].ptr
        edge = self.getEdge(graph)

        h_node, h_virt = self.atom_encoder(x) + self.atom_pos(z), None
        for kernel in self.kernel:
            h_node, h_virt = kernel(h_node, h_virt, edge, batch, idxptr)
        h_out, _ = self.head(h_node, idxptr)

        return h_out


if __name__=="__main__":
    for dhead, nhead in ((16, 16), (32, 8), (32, 12), (32, 16)):
        print('#width:', dhead, nhead)
        for i in (1, 2):
            for j in (i, i+1, i+2):
                layer = GatedLinearBlock(dhead*nhead, nhead, i, j)
                print('##params[%d,%d]:' % (i, j), np.sum([np.prod(p.shape) for p in layer.parameters()]))
        print()

