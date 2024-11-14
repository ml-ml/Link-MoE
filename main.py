import sys
sys.path.append("../..") 

import torch
import argparse
import scipy.sparse as ssp
from gnn_model import *
from utils import *
import time
from torch.utils.data import DataLoader

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from evalutors import evaluate_hits, evaluate_mrr, evaluate_auc
from torch_geometric_signed_directed.data import load_directed_real_data
from torch_geometric.utils import negative_sampling, to_undirected, to_scipy_sparse_matrix
from torch_geometric_signed_directed.utils import in_out_degree, directed_features_in_out

import numpy as np
import networkx as nx
from networkx.algorithms import tree
from torch_geometric.utils import negative_sampling, to_undirected, to_scipy_sparse_matrix

from torch_geometric_signed_directed.utils import in_out_degree
from torch_geometric_signed_directed.data import load_directed_real_data
import argparse

import torch.nn.functional as F

import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, hamming_loss, multilabel_confusion_matrix


dir_path = get_root_dir()


def concat_scores(former_scores, new_scores):
    pos_test_score = torch.concat((former_scores[0], new_scores[0]), dim=0)
    neg_test_score = torch.concat((former_scores[1], new_scores[1]), dim=0)
    pos_valid_score = torch.concat((former_scores[2], new_scores[2]), dim=0)
    neg_valid_score = torch.concat((former_scores[3], new_scores[3]), dim=0)
    scores = [pos_test_score, neg_test_score, pos_valid_score, neg_valid_score]
    return scores


def get_scores(file_name, data_name, model_name, seed):

    file_path = os.path.join(file_name, data_name, model_name)
    document_name = sorted(os.listdir(file_path))[seed]
    data = torch.load(os.path.join(file_path, document_name))

    print(os.path.join(file_path, document_name))

    if 'pos_test_score' in data.keys() and 'neg_test_score' in data.keys():
        pos_test_score = data['pos_test_score']
        neg_test_score = data['neg_test_score']
        if neg_test_score.dim() == 2 and data_name == 'citation2':
            neg_test_score = neg_test_score.view(-1)
    
    if 'pos_valid_score' in data.keys() and 'neg_valid_score' in data.keys():
        pos_valid_score = data['pos_valid_score']
        neg_valid_score = data['neg_valid_score']
        if neg_valid_score.dim() == 2 and data_name == 'citation2':
            neg_valid_score = neg_valid_score.view(-1)

    scores = [pos_test_score.unsqueeze(0), neg_test_score.unsqueeze(0), pos_valid_score.unsqueeze(0), neg_valid_score.unsqueeze(0)]
    return scores


class mlp_score(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(mlp_score, self).__init__()

        self.lins = torch.nn.ModuleList()
        if num_layers == 1: 
            self.lins.append(torch.nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x):   
        
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.lins[-1](x)
        return torch.softmax(x, dim=1)


log_print = get_logger('testrun', 'log', get_config_dir())


def get_metric_score_citation2(evaluator_mrr, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):
    
    k_list = [20, 50, 100]
    result = {}

    # result_mrr_train = evaluate_mrr( evaluator_mrr,  pos_train_pred, neg_val_pred)
    result_mrr_val = evaluate_mrr( evaluator_mrr, pos_val_pred, neg_val_pred )
    result_mrr_test = evaluate_mrr( evaluator_mrr, pos_test_pred, neg_test_pred )
    
   
    result['MRR'] = (0, result_mrr_val['MRR'], result_mrr_test['MRR'])
    for K in k_list:
        result[f'mrr_hit{K}'] = (0, result_mrr_val[f'mrr_hit{K}'], result_mrr_test[f'mrr_hit{K}'])

    return result


def get_metric_score(evaluator_hit, pos_train_pred, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):

    result = {}
    k_list = [20, 50, 100]
    result_hit_train = evaluate_hits(evaluator_hit, pos_train_pred, neg_val_pred, k_list)
    result_hit_val = evaluate_hits(evaluator_hit, pos_val_pred, neg_val_pred, k_list)
    result_hit_test = evaluate_hits(evaluator_hit, pos_test_pred, neg_test_pred, k_list)

    for K in k_list:
        result[f'Hits@{K}'] = (result_hit_train[f'Hits@{K}'], result_hit_val[f'Hits@{K}'], result_hit_test[f'Hits@{K}'])
    
    return result
 

def train(model_feature, model_structure, score_func, optimizer, train_pos_feature, train_neg_feature, pos_train_score, neg_train_score,feature_input_channel, batch_size, device=0):
    model_feature.train()
    model_structure.train()
    score_func.train()

    feat_pos = train_pos_feature[:, :feature_input_channel]
    feat_neg = train_neg_feature[:, :feature_input_channel]

    struct_pos = train_pos_feature[:, feature_input_channel:]
    struct_neg = train_neg_feature[:, feature_input_channel:]

    total_loss = total_examples = 0

    batch_size1 = train_pos_feature.size(0) // batch_size + 1
    batch_size2 = train_neg_feature.size(0) // batch_size + 1

    pos_dataloader = DataLoader(range(train_pos_feature.size(0)), batch_size1, shuffle=True)
    neg_dataloader = DataLoader(range(train_neg_feature.size(0)), batch_size2, shuffle=True)
    
    for perm1, perm2 in zip(pos_dataloader, neg_dataloader) :
        optimizer.zero_grad()

        feat_pos_h = model_feature(feat_pos[perm1].to(device))
        struct_pos_h = model_structure(struct_pos[perm1].to(device))
        train_pos_h = torch.concat((feat_pos_h, struct_pos_h), dim=1)

        pos_weights = score_func(train_pos_h) 
        pos_scores = pos_train_score[:, perm1]
        pos_out = torch.sum(pos_weights * pos_scores.t(), dim=1, keepdim=True)
        pos_out = F.sigmoid(pos_out)
        
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        feat_neg_h = model_feature(feat_neg[perm2].to(device))
        struct_neg_h = model_structure(struct_neg[perm2].to(device))
        train_neg_h = torch.concat((feat_neg_h, struct_neg_h), dim=1)
        
        neg_weights = score_func(train_neg_h)
        neg_scores = neg_train_score[:, perm2]
        neg_out = torch.sum(neg_weights * neg_scores.t(), dim=1, keepdim=True)    
        neg_out = F.sigmoid(neg_out)
       
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        loss = pos_loss + 10 * neg_loss

        loss.backward()

        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()
def test_edge(model_feature, model_structure, score_func, test_pos_feature, pos_test_score, feature_input_channel, batch_size, device=0):

    feat_pos = test_pos_feature[:, :feature_input_channel]
    struct_pos = test_pos_feature[:, feature_input_channel:]
    preds = []

    for perm  in DataLoader(range(test_pos_feature.size(0)), batch_size):
        feat_pos_h = model_feature(feat_pos[perm].to(device))
        struct_pos_h = model_structure(struct_pos[perm].to(device))
        test_pos_h = torch.concat((feat_pos_h, struct_pos_h), dim=1)

        pos_weights = score_func(test_pos_h)
        pos_scores = pos_test_score[:, perm]
        pos_out = torch.sum(pos_weights * pos_scores.t(), dim=1, keepdim=True)
        pos_out = F.sigmoid(pos_out)
        preds += [pos_out]
    pos_test_pred = torch.cat(preds, dim=0)

    return pos_test_pred


@torch.no_grad()
def test(data_name, model_feature, model_structure, score_func, evaluator_hit, evaluator_mrr, val_set, test_set, feature_input_channel, batch_size, device=0):
    model_feature.eval()
    model_structure.eval()
    score_func.eval()
    # import ipdb; ipdb.set_trace()
    valid_pos_feature_val, valid_neg_feature_val, pos_valid_score_val, neg_valid_score_val = val_set
    test_pos_feature, test_neg_feature, pos_test_score, neg_test_score = test_set

    pos_test_pred = test_edge(model_feature, model_structure, score_func, test_pos_feature, pos_test_score, feature_input_channel, batch_size, device=device)
    neg_test_pred = test_edge(model_feature, model_structure, score_func, test_neg_feature, neg_test_score, feature_input_channel, batch_size, device=device)

    pos_valid_pred = test_edge(model_feature, model_structure, score_func, valid_pos_feature_val, pos_valid_score_val, feature_input_channel, batch_size, device=device)
    neg_valid_pred= test_edge(model_feature, model_structure, score_func, valid_neg_feature_val, neg_valid_score_val, feature_input_channel, batch_size, device=device)

    pos_test_pred, neg_test_pred = torch.flatten(pos_test_pred), torch.flatten(neg_test_pred)
    pos_valid_pred, neg_valid_pred = torch.flatten(pos_valid_pred), torch.flatten(neg_valid_pred)

    if data_name != 'ogbl-citation2':
        result = get_metric_score(evaluator_hit, pos_test_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred)
    else:
        neg_test_pred = neg_test_pred.view(-1, 1000)
        neg_valid_pred = neg_valid_pred.view(-1, 1000)
        result = get_metric_score_citation2(evaluator_mrr, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred)
    
    score_emb = [pos_valid_pred.cpu(), neg_valid_pred.cpu(), neg_test_pred.cpu(), pos_test_pred.cpu()]

    return result, score_emb


def get_feature(edge_list, feat_embedding, norm, device):
    if norm:
        feat_embedding = F.normalize(feat_embedding, 1)
    feat_embedding = feat_embedding.to(device)
    feat = torch.empty((edge_list.size(1), feat_embedding.size(1)))
    link_loader = DataLoader(range(edge_list.size(1)), 50000)
    import tqdm
    for ind in tqdm.tqdm(link_loader):
        src, dst = edge_list[0, ind], edge_list[1, ind]
        src_embedding, dst_embedding = feat_embedding[src], feat_embedding[dst]
        scores = src_embedding * dst_embedding
        feat[ind] = scores.cpu()
    return torch.FloatTensor(feat)


def get_degree(degree_tensor, edge_index, batch_size=1000000000):
    scores = []
    link_loader = DataLoader(range(edge_index.size(1)), batch_size)
    for ind in link_loader:
        src, dst = edge_index[0, ind], edge_index[1, ind]
        src_degree, dst_degree = degree_tensor[src], degree_tensor[dst]
        scores.append(torch.reshape(torch.stack([src_degree, dst_degree]), (edge_index.size(1), 2)))
    return scores[0]


def concat_feat(feat_path, valid_pos_feature, valid_neg_feature, test_pos_feature, test_neg_feature):

    structure_feat = torch.load(feat_path)
    
    valid_pos_ppr = structure_feat['pos_valid_score']
    valid_neg_ppr = structure_feat['neg_valid_score']
    test_pos_ppr = structure_feat['pos_test_score']
    test_neg_ppr = structure_feat['neg_test_score']
    if 'shortest_path' in feat_path:
         valid_pos_ppr = 1 / valid_pos_ppr
         valid_neg_ppr = 1 / valid_neg_ppr
         test_pos_ppr = 1 / test_pos_ppr
         test_neg_ppr = 1 / test_neg_ppr

    valid_pos_feature = torch.concat((valid_pos_feature, valid_pos_ppr.unsqueeze(1)), dim=1)
    valid_neg_feature = torch.concat((valid_neg_feature, valid_neg_ppr.unsqueeze(1)), dim=1)
    test_pos_feature = torch.concat((test_pos_feature, test_pos_ppr.unsqueeze(1)), dim=1)
    test_neg_feature = torch.concat((test_neg_feature, test_neg_ppr.unsqueeze(1)), dim=1)    

    return valid_pos_feature, valid_neg_feature, test_pos_feature, test_neg_feature


def get_all_features(args, valid_pos, valid_neg, test_pos, test_neg, feature_embeddings, degree_tensor, aa_path, cn_path, katz_path, ppr_path, ra_path, sp_path):

    if args.use_feature:
        valid_pos_feature = get_feature(valid_pos, feature_embeddings, args.norm, args.device)
        valid_neg_feature = get_feature(valid_neg, feature_embeddings, args.norm, args.device)
        test_pos_feature = get_feature(test_pos, feature_embeddings, args.norm, args.device)
        test_neg_feature = get_feature(test_neg, feature_embeddings, args.norm, args.device)
    else:
        valid_pos_feature = torch.FloatTensor([])
        valid_neg_feature = torch.FloatTensor([])
        test_pos_feature = torch.FloatTensor([])
        test_neg_feature = torch.FloatTensor([])

    if args.use_aa:
        valid_pos_feature, valid_neg_feature, test_pos_feature, test_neg_feature \
            = concat_feat(aa_path, valid_pos_feature, valid_neg_feature, test_pos_feature, test_neg_feature)

    if args.use_cn:
        valid_pos_feature, valid_neg_feature, test_pos_feature, test_neg_feature \
            = concat_feat(cn_path, valid_pos_feature, valid_neg_feature, test_pos_feature, test_neg_feature)

    if args.use_katz:
        valid_pos_feature, valid_neg_feature, test_pos_feature, test_neg_feature \
            = concat_feat(katz_path, valid_pos_feature, valid_neg_feature, test_pos_feature, test_neg_feature)

    if args.use_ppr:
        valid_pos_feature, valid_neg_feature, test_pos_feature, test_neg_feature \
            = concat_feat(ppr_path, valid_pos_feature, valid_neg_feature, test_pos_feature, test_neg_feature)

    if args.use_ra:
        valid_pos_feature, valid_neg_feature, test_pos_feature, test_neg_feature \
            = concat_feat(ra_path, valid_pos_feature, valid_neg_feature, test_pos_feature, test_neg_feature)
        
    if args.use_sp:       
        valid_pos_feature, valid_neg_feature, test_pos_feature, test_neg_feature \
            = concat_feat(sp_path, valid_pos_feature, valid_neg_feature, test_pos_feature, test_neg_feature)

    if args.use_degree:

        valid_pos_degree = get_degree(degree_tensor, valid_pos)
        valid_neg_degree = get_degree(degree_tensor, valid_neg)
        test_pos_degree = get_degree(degree_tensor, test_pos)
        test_neg_degree = get_degree(degree_tensor, test_neg)

        valid_pos_feature = torch.concat((valid_pos_feature, valid_pos_degree), dim=1)
        valid_neg_feature = torch.concat((valid_neg_feature, valid_neg_degree), dim=1)
        test_pos_feature = torch.concat((test_pos_feature, test_pos_degree), dim=1)
        test_neg_feature = torch.concat((test_neg_feature, test_neg_degree), dim=1)

    return valid_pos_feature, valid_neg_feature, test_pos_feature, test_neg_feature


def get_all_scores(args, file_name):

    former_scores = [torch.FloatTensor([]), torch.FloatTensor([]), torch.FloatTensor([]), torch.FloatTensor([])]
    
    if args.gcn:
        scores = get_scores(file_name, args.name, 'gcn', args.score_number)
        former_scores = concat_scores(former_scores, scores)

    if args.mlp:
        scores = get_scores(file_name, args.name, 'mlp', args.score_number)
        former_scores = concat_scores(former_scores, scores)

    if args.seal:
        scores = get_scores(file_name, args.name, 'seal', args.score_number)
        former_scores = concat_scores(former_scores, scores)

    if args.ncnc:
        scores = get_scores(file_name, args.name, 'ncnc', args.score_number)
        former_scores = concat_scores(former_scores, scores)

    if args.buddy:
        scores = get_scores(file_name, args.name, 'buddy', args.score_number)
        former_scores = concat_scores(former_scores, scores)

    if args.ncn:
        scores = get_scores(file_name, args.name, 'ncn', args.score_number)
        former_scores = concat_scores(former_scores, scores)

    if args.neognn:
        scores = get_scores(file_name, args.name, 'neognn', args.score_number)
        former_scores = concat_scores(former_scores, scores)

    if args.n2v:
        scores = get_scores(file_name, args.name, 'n2v', args.score_number)
        former_scores = concat_scores(former_scores, scores)

    if args.peg:
        scores = get_scores(file_name, args.name, 'peg', args.score_number)
        former_scores = concat_scores(former_scores, scores)

    if args.nbfnet:
        scores = get_scores(file_name, args.name, 'nbfnet', args.score_number)
        former_scores = concat_scores(former_scores, scores)

    return former_scores


def split_feature(tensor, ratio, negative=False):
    if not negative:
        len_tensor = tensor.shape[0]
        train = tensor[:int(len_tensor*ratio)]
        val = tensor[int(len_tensor*ratio):]
    else:
        feature_dim = tensor.shape[-1]
        tensor = tensor.view(-1, 1000, feature_dim)
        len_tensor = tensor.shape[0]
        train = tensor[:int(len_tensor*ratio)].view(-1, feature_dim)
        val = tensor[int(len_tensor*ratio):].view(-1, feature_dim)
        tensor = tensor.view(-1, feature_dim)
    return train, val


def split_score(tensor, ratio, negative=False):
    train, val = split_feature(tensor.t(), ratio, negative=negative)
    return train.t(), val.t()


def main():
    parser = argparse.ArgumentParser(description='homo')
    parser.add_argument('--data_name', type=str, default='ogbl-collab')
    parser.add_argument('--neg_mode', type=str, default='equal')
    parser.add_argument('--gnn_model', type=str, default='mlp_model')
    parser.add_argument('--name', type=str, default='collab')
    parser.add_argument('--score_number', type=int, default=0) # which scores from 10 runs of three models
    parser.add_argument('--end_epochs', type=int, default=200)
    parser.add_argument('--ratio', type=float, default=0.5)
    parser.add_argument('--norm', action='store_true', default=False)

    # model
    parser.add_argument('--mlp', action='store_true', default=False)
    parser.add_argument('--seal', action='store_true', default=False)
    parser.add_argument('--gcn', action='store_true', default=False)
    parser.add_argument('--n2v', action='store_true', default=False)
    parser.add_argument('--neognn', action='store_true', default=False)
    parser.add_argument('--ncn', action='store_true', default=False)
    parser.add_argument('--ncnc', action='store_true', default=False)
    parser.add_argument('--buddy', action='store_true', default=False)
    parser.add_argument('--peg', action='store_true', default=False)
    parser.add_argument('--nbfnet', action='store_true', default=False)

    # structure setteing
    parser.add_argument('--use_feature', action='store_true', default=False)
    parser.add_argument('--use_aa', action='store_true', default=False)
    parser.add_argument('--use_cn', action='store_true', default=False) 
    parser.add_argument('--use_katz', action='store_true', default=False)
    parser.add_argument('--use_ppr', action='store_true', default=False)
    parser.add_argument('--use_ra', action='store_true', default=False)
    parser.add_argument('--use_sp', action='store_true', default=False)
    parser.add_argument('--use_degree', action='store_true', default=False)

    parser.add_argument('--use_valedges_as_input', action='store_true', default=False)

    ##gnn setting
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_layers_predictor', type=int, default=1)
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.5)

    ### train setting
    parser.add_argument('--train_batch_size', type=int, default=100000)
    parser.add_argument('--test_batch_size', type=int, default=2000000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=9999)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--kill_cnt', dest='kill_cnt', default=10, type=int, help='early stopping')
    parser.add_argument('--output_dir', type=str, default='output_test')
    parser.add_argument('--l2', type=float, default=0.0, help='L2 Regularization for Optimizer')
    parser.add_argument('--seed', type=int, default=999)
    
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--use_saved_model', action='store_true', default=False)
    parser.add_argument('--metric', type=str, default='Hits@20')
    parser.add_argument('--device', type=int, default=4)
    parser.add_argument('--log_steps', type=int, default=1)
    
    parser.add_argument('--gin_mlp_layer', type=int, default=2)
    parser.add_argument('--gat_head', type=int, default=1)
    parser.add_argument('--cat_node_feat_mf', default=False, action='store_true')
    parser.add_argument('--cat_n2v_feat', default=False, action='store_true')

    args = parser.parse_args()
    ogb_path = '~/ogb_data'
    dataset = PygLinkPropPredDataset(name=args.data_name, root=os.path.join(ogb_path, "dataset", args.data_name))

    data = dataset[0]
    
    split_edge = dataset.get_edge_split()
    node_num = data.num_nodes
    edge_index = data.edge_index

    if hasattr(data, 'x'):
        if data.x != None:
            feature_embeddings = data.x.to(torch.float)
        else:
            feature_embeddings = torch.nn.Embedding(node_num, args.hidden_channels).to(device)
    else:
        feature_embeddings = torch.nn.Embedding(node_num, args.hidden_channels).to(device)

    if hasattr(data, 'edge_weight'):
        if data.edge_weight != None:
            edge_weight = data.edge_weight.view(-1).to(torch.float)
           
        else:
            edge_weight = torch.ones(data.edge_index.size(1), dtype=int)

    else:
        
        edge_weight = torch.ones(data.edge_index.size(1), dtype=int)

    if args.data_name != 'ogbl-citation2':
        pos_train_edge = split_edge['train']['edge'].t()
        pos_valid_edge = split_edge['valid']['edge'].t()
        neg_valid_edge = split_edge['valid']['edge_neg'].t()
        pos_test_edge = split_edge['test']['edge'].t()
        neg_test_edge = split_edge['test']['edge_neg'].t()
    
    else:
        source_edge, target_edge = split_edge['train']['source_node'], split_edge['train']['target_node']
        pos_train_edge = torch.cat([source_edge.unsqueeze(0), target_edge.unsqueeze(0)], dim=0)

        source, target = split_edge['valid']['source_node'],  split_edge['valid']['target_node']
        pos_valid_edge = torch.cat([source.unsqueeze(0), target.unsqueeze(0)], dim=0)
        val_neg_edge = split_edge['valid']['target_node_neg'] 

        neg_valid_edge = torch.stack([source.repeat_interleave(val_neg_edge.size(1)), 
                                val_neg_edge.view(-1)])

        source, target = split_edge['test']['source_node'],  split_edge['test']['target_node']
        pos_test_edge = torch.cat([source.unsqueeze(0), target.unsqueeze(0)], dim=0)
        test_neg_edge = split_edge['test']['target_node_neg']

        neg_test_edge = torch.stack([source.repeat_interleave(test_neg_edge.size(1)), 
                                test_neg_edge.view(-1)])
        idx = torch.tensor([1,0])
        edge_index = torch.cat([edge_index, edge_index[idx]], dim=1)
        edge_weight = torch.ones(edge_index.size(1), dtype=int)

    A = ssp.csr_matrix((edge_weight, (edge_index[0], edge_index[1])), 
                       shape=(node_num, node_num))
    
    if args.use_valedges_as_input:
        print('use validation!!!')
        val_edge_index = pos_valid_edge
        val_edge_index = to_undirected(val_edge_index)

        edge_index = torch.cat([edge_index, val_edge_index], dim=-1)
        val_edge_weight = torch.ones([val_edge_index.size(1)], dtype=int)

        edge_weight = torch.cat([edge_weight, val_edge_weight], 0)
        
        full_A = ssp.csr_matrix((edge_weight, (edge_index[0], edge_index[1])), 
                        shape=(node_num, node_num)) 
    else:
        print('no validation!!!')

        full_A = A
    
    print('A: ', A.nnz)
    print('full_A: ', full_A.nnz)

    print('edge size', pos_valid_edge.size(), neg_valid_edge.size(), pos_test_edge.size(), neg_test_edge.size())
    # import ipdb; ipdb.set_trace()
    print(args)

    init_seed(args.seed)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    aa_path = '~/heuristics/'+args.name+'/AA'
    cn_path = '~/heuristics/'+args.name+'/CN'
    katz_path = '~/heuristics/'+args.name+'/katz_apro'
    ppr_path = '~/heuristics/'+args.name+'/ppr'
    ra_path = '~/heuristics/'+args.name+'/RA'
    sp_path = '~/heuristics/'+args.name+'/shortest_path'

    degree = full_A.sum(axis=1)
    degree_dense = degree.A.flatten()
    degree_tensor = torch.tensor(degree_dense, dtype=torch.float)    


    valid_pos_feature, valid_neg_feature, test_pos_feature, test_neg_feature = \
        get_all_features(args, pos_valid_edge, neg_valid_edge, pos_test_edge, neg_test_edge, feature_embeddings, degree_tensor, aa_path, cn_path, katz_path, ppr_path, ra_path, sp_path)


    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    file_name = '~/prediction_socres'
    pos_test_score, neg_test_score, pos_valid_score, neg_valid_score = get_all_scores(args, file_name)
    pos_valid_score = pos_valid_score.to(device)
    neg_valid_score = neg_valid_score.to(device)
    pos_test_score = pos_test_score.to(device)
    neg_test_score = neg_test_score.to(device)

    # split val into val_train & val_val
    if args.data_name != 'ogbl-citation2':
        valid_pos_feature_train, valid_pos_feature_val = split_feature(valid_pos_feature, args.ratio)
        valid_neg_feature_train, valid_neg_feature_val = split_feature(valid_neg_feature, args.ratio)

        pos_valid_score_train, pos_valid_score_val = split_score(pos_valid_score, args.ratio)
        neg_valid_score_train, neg_valid_score_val = split_score(neg_valid_score, args.ratio)
    else:
        valid_pos_feature_train, valid_pos_feature_val = split_feature(valid_pos_feature, args.ratio)
        valid_neg_feature_train, valid_neg_feature_val = split_feature(valid_neg_feature, args.ratio, negative=True)

        pos_valid_score_train, pos_valid_score_val = split_score(pos_valid_score, args.ratio)
        neg_valid_score_train, neg_valid_score_val = split_score(neg_valid_score, args.ratio, negative=True)

    val_set = [valid_pos_feature_val, valid_neg_feature_val, pos_valid_score_val, neg_valid_score_val]
    test_set = [test_pos_feature, test_neg_feature, pos_test_score, neg_test_score]

    output_channel = pos_valid_score.size(0)
    feature_input_channel = feature_embeddings.size(1)
    structure_input_channel = valid_pos_feature.size(1) - feature_input_channel
    
    model_feature = eval(args.gnn_model)(feature_input_channel, args.hidden_channels,
                    args.hidden_channels, args.num_layers, args.dropout, args.gin_mlp_layer, args.gat_head, node_num, args.cat_node_feat_mf).to(device)
    
    model_structure = eval(args.gnn_model)(structure_input_channel, args.hidden_channels,
                args.hidden_channels, args.num_layers, args.dropout, args.gin_mlp_layer, args.gat_head, node_num, args.cat_node_feat_mf).to(device)
    
    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')

    if args.data_name != 'ogbl-citation2':
        loggers = {
            'Hits@20': Logger(args.runs),
            'Hits@50': Logger(args.runs),
            'Hits@100': Logger(args.runs)  
        }
    else:
        loggers = {
            'MRR': Logger(args.runs),
            'mrr_hit20':  Logger(args.runs),
            'mrr_hit50':  Logger(args.runs),
            'mrr_hit100':  Logger(args.runs)
        }

    if args.data_name =='ogbl-collab':
        eval_metric = 'Hits@50'
    elif args.data_name =='ogbl-ppa':
        eval_metric = 'Hits@100'
    elif args.data_name =='ogbl-citation2':
        eval_metric = 'MRR'

    
    print('**************** mlp model *****************:')
    print(model_feature)

    score_func = mlp_score(args.hidden_channels*2, args.hidden_channels,
                    output_channel, args.num_layers_predictor, args.dropout).to(device)
   
    print('**************** score model *****************:')
    print(score_func)

    for run in range(args.runs):

        print('#################################          ', run, '          #################################')
        
        if args.runs == 1:
            seed = args.seed
        else:
            seed = run
        print('seed: ', seed)

        init_seed(seed)

        save_path = args.output_dir+'/lr'+str(args.lr) + '_drop' + str(args.dropout) + '_l2'+ str(args.l2) + '_numlayer' + str(args.num_layers)+ '_numPredlay' + str(args.num_layers_predictor) + '_numGinMlplayer' + str(args.gin_mlp_layer)+'_dim'+str(args.hidden_channels) + '_'+ 'best_run_'+str(seed)

        model_feature.reset_parameters()
        model_structure.reset_parameters()
        score_func.reset_parameters()

        optimizer = torch.optim.Adam(
                list(model_feature.parameters()) + list(model_structure.parameters()) + list(score_func.parameters()),lr=args.lr, weight_decay=args.l2)

        best_valid = 0
        kill_cnt = 0
        for epoch in range(1, 1 + args.end_epochs):
            # print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            loss = train(model_feature, model_structure, score_func, optimizer, valid_pos_feature_train, valid_neg_feature_train, pos_valid_score_train, neg_valid_score_train, feature_input_channel, args.train_batch_size, device=args.device)
            if epoch % args.eval_steps == 0:

                results_rank, score_emb = test(args.data_name, model_feature, model_structure, score_func, evaluator_hit, evaluator_mrr, val_set, test_set, feature_input_channel, args.test_batch_size, device=args.device)
                
                for key, result in results_rank.items():
                    loggers[key].add_result(run, result)

                if epoch % args.log_steps == 0:
                    for key, result in results_rank.items():
                        
                        print(key)
                        
                        train_hits, valid_hits, test_hits = result

                        log_print.info(
                            f'Run: {run + 1:02d}, '
                              f'Epoch: {epoch:02d}, '
                              f'Loss: {loss:.4f}, '
                              f'Train: {100 * train_hits:.2f}%, '
                              f'Valid: {100 * valid_hits:.2f}%, '
                              f'Test: {100 * test_hits:.2f}%')

             
                r = torch.tensor(loggers[eval_metric].results[run])
                best_valid_current = round(r[:, 1].max().item(),4)
                best_test = round(r[r[:, 1].argmax(), 2].item(), 4)

                print(eval_metric)
                log_print.info(f'best valid: {100*best_valid_current:.2f}%, '
                                f'best test: {100*best_test:.2f}%')   
                print('---')    

                if best_valid_current > best_valid:
                    best_valid = best_valid_current
                    kill_cnt = 0

                    if args.save:

                        save_emb(score_emb, save_path)
                
                else:
                    kill_cnt += 1
                    
                    if kill_cnt > args.kill_cnt: 
                        print("Early Stopping!!")
                        break
        
        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run)


    result_all_run = {}
    for key in loggers.keys():
        print(key)
        
        best_metric,  best_valid_mean, mean_list, var_list = loggers[key].print_statistics()

        if key == eval_metric:
            best_metric_valid_str = best_metric
            best_valid_mean_metric = best_valid_mean

        # if key == 'AUC':
        #     best_auc_valid_str = best_metric
        #     best_auc_metric = best_valid_mean

        result_all_run[key] = [mean_list, var_list]
        
    
    # print(best_metric_valid_str +' ' +best_auc_valid_str)

    # return best_valid_mean_metric, best_auc_metric, result_all_run
    return best_valid_mean_metric, result_all_run



if __name__ == "__main__":
    main()

   
