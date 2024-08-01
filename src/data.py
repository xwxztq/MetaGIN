import os
import os.path as osp
import sys
import shutil
from ogb.utils.torch_util import replace_numpy_with_torchtensor
from ogb.utils.url import decide_download, download_url, extract_zip

import h5py
import torch as pt
import numpy as np
import pandas as pd

from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem, BRICS
from rdkit.Chem.Lipinski import RotatableBondSmarts

from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector
from ogb.lsc import PCQM4Mv2Evaluator

from torch_sparse import coalesce, spspmm
from torch_geometric.utils import degree
from torch_geometric.data import Data, HeteroData, InMemoryDataset
# from torch_geometric.transforms import AddRandomWalkPE
from torch_geometric.loader import DataLoader


from typing import Any, Optional

import numpy as np
import torch
from torch_sparse import SparseTensor

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import (
    get_laplacian,
    get_self_loop_attr,
    to_scipy_sparse_matrix,
)

def add_node_attr(data: Data, value: Any,
                  attr_name: Optional[str] = None) -> Data:
    # TODO Move to `BaseTransform`.
    if attr_name is None:
        if 'x' in data:
            x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat([x, value.to(x.device, x.dtype)], dim=-1)
        else:
            data.x = value
    else:
        data[attr_name] = value

    return data

# @functional_transform('add_random_walk_pe')
class AddRandomWalkPE(BaseTransform):
    r"""Adds the random walk positional encoding from the `"Graph Neural
    Networks with Learnable Structural and Positional Representations"
    <https://arxiv.org/abs/2110.07875>`_ paper to the given graph
    (functional name: :obj:`add_random_walk_pe`).
    Args:
        walk_length (int): The number of random walk steps.
        attr_name (str, optional): The attribute name of the data object to add
            positional encodings to. If set to :obj:`None`, will be
            concatenated to :obj:`data.x`.
            (default: :obj:`"laplacian_eigenvector_pe"`)
    """
    def __init__(
        self,
        walk_length: int,
        attr_name: Optional[str] = 'random_walk_pe',
    ):
        self.walk_length = walk_length
        self.attr_name = attr_name

    def __call__(self, data: Data) -> Data:
        num_nodes = data.num_nodes
        edge_index, edge_weight = data.edge_index, data.edge_weight

        adj = SparseTensor.from_edge_index(edge_index, edge_weight,
                                           sparse_sizes=(num_nodes, num_nodes))

        # Compute D^{-1} A:
        deg_inv = 1.0 / adj.sum(dim=1)
        deg_inv[deg_inv == float('inf')] = 0
        adj = adj * deg_inv.view(-1, 1)

        out = adj
        row, col, value = out.coo()
        pe_list = [get_self_loop_attr((row, col), value, num_nodes)]
        for _ in range(self.walk_length - 1):
            out = out @ adj
            row, col, value = out.coo()
            pe_list.append(get_self_loop_attr((row, col), value, num_nodes))
        pe = torch.stack(pe_list, dim=-1)

        data = add_node_attr(data, pe, attr_name=self.attr_name)
        return data


ATOM_SIZE = [36, 4, 12, 12, 10, 6, 6, 3]
ATOM_CUMSIZE = pt.from_numpy(np.array([1] + ATOM_SIZE).cumsum())
#0 atomic_num: class36 *
#1 global_chirality: class4 *
#3 degree: value12
#4 formal_charge: value12[-5]
#5 numH: value10
#6 number_radical_e: value6
#7 hybridization: class6 *
#8 aromatic+ring: class3 *

BOND_SIZE = [5, 5, 6, 2, 2]
BOND_CUMSIZE = pt.from_numpy(np.array([1] + BOND_SIZE).cumsum())
#0 bond_type: class5 *
#1 local_chirality: class5 *
#2 bond_stereo: class6 *
#3 is_conjugated: bool
#4 is_rotatable: bool


def sort_transform(idx, attr):
    _, i = pt.sort(idx[0], stable=True)
    idx, attr = idx[:, i], attr[i]
    _, i = pt.sort(idx[1], stable=True)
    idx, attr = idx[:, i], attr[i]
    return idx, attr

def hetero_transform(graph):
    # input
    size = graph.num_nodes
    head = [graph.edge_index[0] == i for i in range(size)]
    head = [[graph.edge_index[1, i], graph.edge_attr[i]] for i in head]
    tail = [graph.edge_index[1] == i for i in range(size)]
    tail = [[graph.edge_index[0, i], graph.edge_attr[i]] for i in tail]
    pair_idx, pair_attr = pt.cat([pt.arange(size) for _ in range(2)]).reshape(2, -1).long(), pt.zeros(size).int()

    # one-hop neighbors
    hop1_attr, hop1_idx = [], []
    for (i0, i1), a0 in zip(graph.edge_index.T.tolist(), graph.edge_attr.tolist()):
        hop1_idx += [[i0, i1]]
        hop1_attr += [a0]
    if len(hop1_attr) > 0:
        hop1_idx = pt.tensor(hop1_idx).T
        hop1_attr = pt.tensor(hop1_attr)
        pair_idx, pair_attr = pt.cat([pair_idx, hop1_idx], 1), pt.cat([pair_attr, pt.ones(len(hop1_attr))], 0)
        pair_idx, pair_attr = coalesce(pair_idx, pair_attr, size, size, op='min')
    else:
        hop1_idx = pt.zeros([0, 2]).T
        hop1_attr = pt.zeros([0, 5])

    # two-hop neighbors
    hop2_attr, hop2_idx = [], []
    for i1 in range(size):
        ei0, ea0 = tail[i1]
        ei1, ea1 = head[i1]
        for i0, a0 in zip(ei0.tolist(), ea0.tolist()):
            for i2, a1 in zip(ei1.tolist(), ea1.tolist()):
                if i0 == i2: continue  # loop
                hop2_idx += [[i0, i2]]
                hop2_attr += [[1]]
    if len(hop2_attr) > 0:
        hop2_idx = pt.tensor(hop2_idx).long().T
        hop2_attr = pt.tensor(hop2_attr).int()
        hop2_idx, hop2_attr = coalesce(hop2_idx, hop2_attr, size, size, op='sum')
        pair_idx, pair_attr = pt.cat([pair_idx, hop2_idx], 1), pt.cat([pair_attr, pt.ones(len(hop2_attr))*2], 0)
        pair_idx, pair_attr = coalesce(pair_idx, pair_attr, size, size, op='min')
    else:
        hop2_idx = pt.zeros([0, 2]).T
        hop2_attr = pt.zeros([0, 1])

    # three-hop neighbors
    hop3_attr, hop3_idx = [], []
    for hop1, (i1, i2), a1 in zip(range(len(graph.edge_attr)), graph.edge_index.T.tolist(), graph.edge_attr.tolist()):
        ei0, ea0 = tail[i1]
        ei2, ea2 = head[i2]
        for i0, a0 in zip(ei0.tolist(), ea0.tolist()):
            for i3, a2 in zip(ei2.tolist(), ea2.tolist()):
                if i0 == i2 or i0 == i3 or i1 == i3: continue  # loop
                hop3_idx += [[i0, i3]]
                hop3_attr += [[1]]
    if len(hop3_attr) > 0:
        hop3_idx = pt.tensor(hop3_idx).long().T
        hop3_attr = pt.tensor(hop3_attr).int()
        hop3_idx, hop3_attr = coalesce(hop3_idx, hop3_attr, size, size, op='sum')
        pair_idx, pair_attr = pt.cat([pair_idx, hop3_idx], 1), pt.cat([pair_attr, pt.ones(len(hop3_attr))*3], 0)
        pair_idx, pair_attr = coalesce(pair_idx, pair_attr, size, size, op='min')
    else:
        hop3_idx = pt.zeros([0, 2]).T
        hop3_attr = pt.zeros([0, 1])

    # remote neighbors
    for i in range(4, size):
        chk_shape = pair_idx.shape
        chk_idx, chk_attr = spspmm(pair_idx, pair_attr, pair_idx, pair_attr, size, size, size)
        pair_idx, pair_attr = pt.cat([pair_idx, chk_idx], 1), pt.cat([pair_attr, pt.ones_like(chk_attr)*i], 0)
        pair_idx, pair_attr = coalesce(pair_idx, pair_attr, size, size, op='min')
        if pair_idx.shape == chk_shape: break

    # disconnected neighbors
    chk_idx = pt.arange(size)
    chk_idx = pt.nonzero(chk_idx[:, None] == chk_idx[None, :]).T
    if pair_idx.shape != chk_idx.shape:
        pair_idx, pair_attr = pt.cat([pair_idx, chk_idx], 1), pt.cat([pair_attr, -pt.ones(chk_idx.shape[1])], 0)
        pair_idx, pair_attr = coalesce(pair_idx, pair_attr, size, size, op='max')
    chk_idx = pair_idx[0] < pair_idx[1]
    pair_idx, pair_attr = pair_idx[:, chk_idx], pair_attr[chk_idx].unsqueeze(-1)

    # output
    g = HeteroData()
    g['atom'].x = graph.x.short()
    #g['atom'].pos_3d = graph.pos_3d.half()
    g['atom'].pos_rw = graph.pos_rw.half()
    hop1_idx, hop1_attr = sort_transform(hop1_idx, hop1_attr)
    g['atom', 'bond', 'atom'].edge_index = hop1_idx.short()
    g['atom', 'bond', 'atom'].edge_attr = hop1_attr.short()
    g['atom'].deg_bond = degree(hop1_idx[1], graph.num_nodes).short()
    hop2_idx, hop2_attr = sort_transform(hop2_idx, hop2_attr)
    g['atom', 'angle', 'atom'].edge_index = hop2_idx.short()
    g['atom', 'angle', 'atom'].edge_attr = hop2_attr.short()
    g['atom'].deg_angle = degree(hop2_idx[1], graph.num_nodes).short()
    hop3_idx, hop3_attr = sort_transform(hop3_idx, hop3_attr)
    g['atom', 'torsion', 'atom'].edge_index = hop3_idx.short()
    g['atom', 'torsion', 'atom'].edge_attr = hop3_attr.short()
    g['atom'].deg_torsion = degree(hop3_idx[1], graph.num_nodes).short()
    pair_idx, pair_attr = sort_transform(pair_idx, pair_attr)
    g['atom', 'pair', 'atom'].edge_index = pair_idx.short()
    g['atom', 'pair', 'atom'].edge_attr = pair_attr.short()
    g.y = graph.y.half()

    return g


def rotate_transform(graph):
    rotate = pt.tensor([[0, 1, 2, 3, 4], [0, 1, 3, 4, 2], [0, 1, 4, 2, 3],
                        [0, 2, 1, 4, 3], [0, 2, 4, 3, 1], [0, 2, 3, 1, 4],
                        [0, 3, 4, 1, 2], [0, 3, 1, 2, 4], [0, 3, 2, 4, 1],
                        [0, 4, 3, 2, 1], [0, 4, 2, 1, 3], [0, 4, 1, 3, 2]]).long()

    index = pt.argwhere(graph['atom'].x[:, 1] > ATOM_CUMSIZE[1]).view(-1)
    rotate = rotate[pt.randint(len(rotate), index.shape)]
    for idx, rot in zip(index, rotate):
        mask = graph['bond'].edge_index[1] == idx
        graph['bond'].edge_attr[mask, 1] = rot[graph['bond'].edge_attr[mask, 1] - BOND_CUMSIZE[1]] + BOND_CUMSIZE[1]
    return graph

def cast_transform(graph, nopair=True):
    g = HeteroData()
    g['atom'].x = graph['atom'].x.long()
    g['atom'].x = g['atom'].x + ATOM_CUMSIZE[:-1]
    #g['atom'].pos_3d = graph['atom'].pos_3d.float()
    g['atom'].pos_rw = graph['atom'].pos_rw.float()
    #g['atom'].deg_bond = graph['atom'].deg_bond.long()
    #g['atom'].deg_angle = graph['atom'].deg_angle.long()
    #g['atom'].deg_torsion = graph['atom'].deg_torsion.long()
    g['atom', 'bond', 'atom'].edge_index = graph['bond'].edge_index.long()
    g['atom', 'bond', 'atom'].edge_attr = graph['bond'].edge_attr.long() + BOND_CUMSIZE[:-1]
    g['atom', 'angle', 'atom'].edge_index = graph['angle'].edge_index.long()
    g['atom', 'angle', 'atom'].edge_attr = graph['angle'].edge_attr.long()
    g['atom', 'torsion', 'atom'].edge_index = graph['torsion'].edge_index.long()
    g['atom', 'torsion', 'atom'].edge_attr = graph['torsion'].edge_attr.long()
    if not nopair:
        g['atom', 'pair', 'atom'].edge_index = graph['pair'].edge_index.long()
        g['atom', 'pair', 'atom'].edge_attr = graph['pair'].edge_attr.long()
    g.y = graph.y.float()
    g = rotate_transform(g)

    return g


class PygPCQM4Mv2Dataset(InMemoryDataset):
    def __init__(self, root='data', transform=None, pre_transform=None):
        '''
            Pytorch Geometric PCQM4Mv2 dataset object
                - root (str): the dataset folder will be located at root/pcqm4m_kddcup2021
        '''

        self.original_root = root
        self.folder = osp.join(root, 'pcqm4m-metagin')
        self.version = 1
        
        self.url = 'https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/pcqm4m-v2.zip'

        # check version and update if necessary
        if osp.isdir(self.folder) and (not osp.exists(osp.join(self.folder, f'RELEASE_v{self.version}.txt'))):
            print('PCQM4Mv2 dataset has been updated.')
            if input('Will you update the dataset now? (y/N)\n').lower() == 'y':
                shutil.rmtree(self.folder)

        super(PygPCQM4Mv2Dataset, self).__init__(self.folder, transform, pre_transform)

        self.data, self.slices = pt.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'data.csv.gz'

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def download(self):
        if decide_download(self.url):
            path = download_url(self.url, self.original_root)
            extract_zip(path, self.original_root)
            os.unlink(path)
        else:
            print('Stop download.')
            exit(-1)

    def molecule2graph(self, mol):
        # atoms
        atom_features_list = []
        for atom in mol.GetAtoms():
            v = atom_to_feature_vector(atom)
            v = v[:1] + [0] + v[2:7] + [v[7]+v[8]]
            atom_features_list.append(v)
        for idx, chi in Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False):
            if chi == 'R':
                atom_features_list[idx][1] = 1
            elif chi == 'S':
                atom_features_list[idx][1] = 2
            else:
                atom_features_list[idx][1] = 3
        x = np.array(atom_features_list, dtype = np.int16)
        num_nodes = len(x)

        # bonds
        num_bond_features = 5  # bond type, bond stereo, local chirality, is_conjugated, rotatable
        rotate = np.zeros([num_nodes, num_nodes], dtype = np.int16)
        for i, j in mol.GetSubstructMatches(RotatableBondSmarts):
            rotate[i, j] = rotate[j, i] = 1
        if len(mol.GetBonds()) > 0:  # mol has bonds
            edges_list = []
            edge_features_list = []
            chirality_rank = np.zeros(num_nodes, dtype=np.int16)
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()

                edge_feature = bond_to_feature_vector(bond)
                chi, chj = 0, 0
                if x[i, 1] > 0:
                    chirality_rank[i] += 1
                    chi += chirality_rank[i]
                if x[j, 1] > 0:
                    chirality_rank[j] += 1
                    chj += chirality_rank[j]

                # add edges in both directions
                edges_list.append((i, j))
                edge_features_list.append(edge_feature[:1] + [chj] + edge_feature[1:] + [rotate[i, j]])
                edges_list.append((j, i))
                edge_features_list.append(edge_feature[:1] + [chi] + edge_feature[1:] + [rotate[j, i]])

            # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
            edge_index = np.array(edges_list, dtype = np.int32).T

            # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
            edge_attr = np.array(edge_features_list, dtype = np.int16)

            inverse = np.array([0, 2, 1, 3, 4], dtype=np.int16)
            for idx, chi in Chem.FindMolChiralCenters(mol, includeCIP=False, includeUnassigned=True, useLegacyImplementation=False):
                if chi == 'Tet_CW':
                    edge_attr[edge_index[1] == idx, 1] = inverse[edge_attr[edge_index[1] == idx, 1]]
        else:  # mol has no bonds
            edge_index = np.empty((2, 0), dtype = np.int32)
            edge_attr = np.empty((0, num_bond_features), dtype = np.int16)

        graph = dict()
        graph['num_nodes'] = num_nodes
        graph['node_feat'] = x
        try:
            graph['node_pos3d'] = mol.GetConformer().GetPositions().astype(np.float16)
        except:
            graph['node_pos3d'] = np.zeros([0, 3], dtype=np.float16)
        graph['edge_index'] = edge_index
        graph['edge_feat'] = edge_attr
        return graph 

    def process(self):
        split_dict = self.get_idx_split()
        size_dict = sum([len(v) for v in split_dict.values()])
        print('#dict:', size_dict, [[min(v).item(), max(v).item()] for v in split_dict.values()])

        print('#loading CSV file ...')
        data_df = pd.read_csv(osp.join(self.raw_dir, 'data.csv.gz'))
        smiles_list = data_df['smiles']
        homolumogap_list = data_df['homolumogap']
        print('#loaded:', len(smiles_list), len(homolumogap_list), size_dict)
        assert len(smiles_list) == size_dict
        assert len(homolumogap_list) == size_dict

        try:
            print('#loading SDF file ...')
            data_sdf = Chem.SDMolSupplier(osp.join(self.raw_dir, 'embed3d-train.sdf'), removeHs=True)  # hydrogen
            mol3d_list = []
            for d in tqdm(data_sdf, total=len(split_dict['train'])):
                mol3d_list.append(d)
            print('#loaded:', len(mol3d_list), len(split_dict['train']))
            assert len(mol3d_list) == len(split_dict['train'])
            mol3d_list += [None] * (size_dict - len(mol3d_list))
        except:
            mol3d_list = [None] * size_dict
        print('#padded:', len(mol3d_list), size_dict)
        assert len(mol3d_list) == size_dict

        print('#converting molecules into graphs...')
        data_list = []
        for smiles, mol3d, homolumogap in tqdm(zip(smiles_list, mol3d_list, homolumogap_list), total=size_dict):
            try:
                graph = self.molecule2graph(mol3d)
            except Exception as e:
                mol2d = Chem.MolFromSmiles(smiles, params)
                graph = self.molecule2graph(mol2d)
            assert(len(graph['edge_feat']) == graph['edge_index'].shape[1])
            assert(len(graph['node_feat']) == graph['num_nodes'])

            data = Data()
            data.edge_index = pt.from_numpy(graph['edge_index']).long()
            data.edge_attr = pt.from_numpy(graph['edge_feat']).int()
            i2w = np.array([1, 2, 3, 1.5, 0], dtype=np.float32)
            data.edge_weight = pt.from_numpy(i2w[graph['edge_feat'][:, 0]]).float()
            data.x = pt.from_numpy(graph['node_feat']).int()
            data.pos_3d = pt.from_numpy(graph['node_pos3d']).float()
            data = addPosRW(data)
            data.y = pt.Tensor([homolumogap]).float()
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)
        print('#converted:', len(data_list), size_dict)
        assert len(data_list) == size_dict

        # double-check prediction target
        assert(all([not pt.isnan(data_list[i].y)[0] for i in split_dict['train']]))
        #assert(all([len(data_list[i]['atom'].pos_3d) > 0 for i in split_dict['train']]))
        assert(all([not pt.isnan(data_list[i].y)[0] for i in split_dict['valid']]))
        #assert(all([len(data_list[i]['atom'].pos_3d) == 0 for i in split_dict['valid']]))
        assert(all([pt.isnan(data_list[i].y)[0] for i in split_dict['test-dev']]))
        #assert(all([len(data_list[i]['atom'].pos_3d) == 0 for i in split_dict['test-dev']]))
        assert(all([pt.isnan(data_list[i].y)[0] for i in split_dict['test-challenge']]))
        #assert(all([len(data_list[i]['atom'].pos_3d) == 0 for i in split_dict['test-challenge']]))

        data, slices = self.collate(data_list)

        print('Saving...')
        pt.save((data, slices), self.processed_paths[0])

    def get_idx_split(self):
        split_dict = replace_numpy_with_torchtensor(pt.load(osp.join(self.root, 'split_dict.pt')))
        return split_dict

addPosRW = AddRandomWalkPE(16, 'pos_rw')
params   = Chem.SmilesParserParams(); params.removeHs = True  # hydrogen

# dataset  = PygPCQM4Mv2Dataset(root='data', pre_transform=hetero_transform, transform=cast_transform)
# dataidx  = dataset.get_idx_split()
# dataeval = PCQM4Mv2Evaluator()

class SmilesPCQM4Mv2Dataset(InMemoryDataset):
    def __init__(self, smiles_list, homo_list=None, root='data', transform=None, pre_transform=None, pre_filter=None):
        '''
            Pytorch Geometric PCQM4Mv2 dataset object
                - root (str): the dataset folder will be located at root/pcqm4m_kddcup2021
        '''

        self.original_root = root
        self.folder = osp.join(root, 'pcqm4m-metagin-temp')
        self.version = 1
        self.smiles_list = smiles_list
        self.homo_list = homo_list


        super(InMemoryDataset, self).__init__(self.folder, transform, pre_transform)

        self.data, self.slices = pt.load(self.processed_paths[0])

        os.remove(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'data.csv.gz'

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def download(self):
        pass

    def molecule2graph(self, mol):
        # atoms
        atom_features_list = []
        for atom in mol.GetAtoms():
            v = atom_to_feature_vector(atom)
            v = v[:1] + [0] + v[2:7] + [v[7]+v[8]]
            atom_features_list.append(v)
        for idx, chi in Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False):
            if chi == 'R':
                atom_features_list[idx][1] = 1
            elif chi == 'S':
                atom_features_list[idx][1] = 2
            else:
                atom_features_list[idx][1] = 3
        x = np.array(atom_features_list, dtype = np.int16)
        num_nodes = len(x)

        # bonds
        num_bond_features = 5  # bond type, bond stereo, local chirality, is_conjugated, rotatable
        rotate = np.zeros([num_nodes, num_nodes], dtype = np.int16)
        for i, j in mol.GetSubstructMatches(RotatableBondSmarts):
            rotate[i, j] = rotate[j, i] = 1
        if len(mol.GetBonds()) > 0:  # mol has bonds
            edges_list = []
            edge_features_list = []
            chirality_rank = np.zeros(num_nodes, dtype=np.int16)
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()

                edge_feature = bond_to_feature_vector(bond)
                chi, chj = 0, 0
                if x[i, 1] > 0:
                    chirality_rank[i] += 1
                    chi += chirality_rank[i]
                if x[j, 1] > 0:
                    chirality_rank[j] += 1
                    chj += chirality_rank[j]

                # add edges in both directions
                edges_list.append((i, j))
                edge_features_list.append(edge_feature[:1] + [chj] + edge_feature[1:] + [rotate[i, j]])
                edges_list.append((j, i))
                edge_features_list.append(edge_feature[:1] + [chi] + edge_feature[1:] + [rotate[j, i]])

            # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
            edge_index = np.array(edges_list, dtype = np.int32).T

            # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
            edge_attr = np.array(edge_features_list, dtype = np.int16)

            inverse = np.array([0, 2, 1, 3, 4], dtype=np.int16)
            for idx, chi in Chem.FindMolChiralCenters(mol, includeCIP=False, includeUnassigned=True, useLegacyImplementation=False):
                if chi == 'Tet_CW':
                    edge_attr[edge_index[1] == idx, 1] = inverse[edge_attr[edge_index[1] == idx, 1]]
        else:  # mol has no bonds
            edge_index = np.empty((2, 0), dtype = np.int32)
            edge_attr = np.empty((0, num_bond_features), dtype = np.int16)

        graph = dict()
        graph['num_nodes'] = num_nodes
        graph['node_feat'] = x
        try:
            graph['node_pos3d'] = mol.GetConformer().GetPositions().astype(np.float16)
        except:
            graph['node_pos3d'] = np.zeros([0, 3], dtype=np.float16)
        graph['edge_index'] = edge_index
        graph['edge_feat'] = edge_attr
        return graph 

    def process(self):
        smiles_list = self.smiles_list
        homolumogap_list = self.homo_list if (self.homo_list is not None) else ([np.nan] * len(smiles_list))
        size_dict = len(smiles_list)
        # homolumogap_list = self.homo_list 

        print('#loaded:', len(smiles_list), len(homolumogap_list))
        assert len(smiles_list) == len(homolumogap_list)

        print('#converting molecules into graphs...')
        data_list = []
        for smiles, homolumogap in tqdm(zip(smiles_list,homolumogap_list), total=size_dict):
            mol2d = Chem.MolFromSmiles(smiles, params)
            graph = self.molecule2graph(mol2d)
            assert(len(graph['edge_feat']) == graph['edge_index'].shape[1])
            assert(len(graph['node_feat']) == graph['num_nodes'])

            data = Data()
            data.edge_index = pt.from_numpy(graph['edge_index']).long()
            data.edge_attr = pt.from_numpy(graph['edge_feat']).int()
            i2w = np.array([1, 2, 3, 1.5, 0], dtype=np.float32)
            data.edge_weight = pt.from_numpy(i2w[graph['edge_feat'][:, 0]]).float()
            data.x = pt.from_numpy(graph['node_feat']).int()
            data.pos_3d = pt.from_numpy(graph['node_pos3d']).float()
            addPosRW(data)
            data.y = pt.Tensor([homolumogap]).float()
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)
        print('#converted:', len(data_list), size_dict)
        assert len(data_list) == size_dict

        # double-check prediction target
        data, slices = self.collate(data_list)

        print('Saving...')
        pt.save((data, slices), self.processed_paths[0])

    def get_idx_split(self):
        split_dict = replace_numpy_with_torchtensor(pt.load(osp.join(self.root, 'split_dict.pt')))
        return split_dict

if __name__=="__main__":
    print('#atom_cumsize:', *ATOM_CUMSIZE.tolist())
    print('#bond_cumsize:', *BOND_CUMSIZE.tolist())
    print()

    g0 = dataset[0]
    print('#data:', g0)
    print('#atom:', g0['atom'].x[:12])
    for k in ['atom']:
        print('#dtype:', k, g0[k].x.dtype)
    print('#bond:', g0['bond'].edge_attr[:12])
    for k in ['bond', 'angle', 'torsion']:
        print('#dtype:', k, g0[k].edge_index.dtype, g0[k].edge_attr.dtype)
    print('#dtype:', 'y', g0.y.dtype)
    print()

    loader = DataLoader(dataset, batch_size=12, shuffle=True, drop_last=True)
    for b in loader:
        print('#batch:')
        print(b)
        print()
        break

    print('#done!!!')

