import math
import random 
from asyncore import read
from platform import node
from dgllife.model.gnn import GAT, AttentiveFPGNN
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import edge_softmax
import dgl
from dgllife.model.readout.weighted_sum_and_max import WeightedSumAndMax
from dgl.nn import GATConv

#part I: AA embedding

def bls(sequences):
    BLS_dic = {
"A":[4,0,-2,-1,-2,0,-2,-1,-1,-1,-1,-2,-1,-1,-1,1,0,0,-3,-2],
"C":[0,9,-3,-4,-2,-3,-3,-1,-3,-1,-1,-3,-3,-3,-3,-1,-1,-1,-2,-2],
"D":[-2,-3,6,2,-3,-1,-1,-3,-1,-4,-3,1,-1,0,-2,0,-1,-3,-4,-3],
"E":[-1,-4,2,5,-3,-2,0,-3,1,-3,-2,0,-1,2,0,0,-1,-2,-3,-2],
"F":[-2,-2,-3,-3,6,-3,-1,0,-3,0,0,-3,-4,-3,-3,-2,-2,-1,1,3],
"G":[0,-3,-1,-2,-3,6,-2,-4,-2,-4,-3,0,-2,-2,-2,0,-2,-3,-2,-3],
"H":[-2,-3,-1,0,-1,-2,8,-3,-1,-3,-2,1,-2,0,0,-1,-2,-3,-2,2],
"I":[-1,-1,-3,-3,0,-4,-3,4,-3,2,1,-3,-3,-3,-3,-2,-1,3,-3,-1],
"K":[-1,-3,-1,1,-3,-2,-1,-3,5,-2,-1,0,-1,1,2,0,-1,-2,-3,-2],
"L":[-1,-1,-4,-3,0,-4,-3,2,-2,4,2,-3,-3,-2,-2,-2,-1,1,-2,-1],
"M":[-1,-1,-3,-2,0,-3,-2,1,-1,2,5,-2,-2,0,-1,-1,-1,1,-1,-1],
"N":[-2,-3,1,0,-3,0,1,-3,0,-3,-2,6,-2,0,0,1,0,-3,-4,-2],
"P":[-1,-3,-1,-1,-4,-2,-2,-3,-1,-3,-2,-2,7,-1,-2,-1,-1,-2,-4,-3],
"Q":[-1,-3,0,2,-3,-2,0,-3,1,-2,0,0,-1,5,1,0,-1,-2,-2,-1],
"R":[-1,-3,-2,0,-3,-2,0,-3,2,-2,-1,0,-2,1,5,-1,-1,-3,-3,-2],
"S":[1,-1,0,0,-2,0,-1,-2,0,-2,-1,1,-1,0,-1,4,1,-2,-3,-2],
"T":[0,-1,-1,-1,-2,-2,-2,-1,-1,-1,-1,0,-1,-1,-1,1,5,0,-2,-2],
"V":[0,-1,-3,-2,-1,-3,-3,3,-2,1,1,-3,-2,-2,-3,-2,0,4,-3,-1],
"W":[-3,-2,-4,-3,1,-2,-2,-3,-3,-2,-1,-4,-4,-2,-3,-3,-2,-3,11,2],
"Y":[-2,-2,-3,-2,3,-3,2,-1,-2,-1,-1,-2,-3,-1,-2,-2,-2,-1,2,7]}
    encode_data0 = []
    for i, peptides in enumerate(sequences):
        encode_data1 =[]
        for j in peptides:
            encode_data1.append(BLS_dic[j])
        encode_data0.append(encode_data1)

    return encode_data0



#part II: transformer model

# 1. position encoding
def position_encoding(input):

    x = torch.zeros(9,20)
    position = torch.arange(0, 9, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, 20, 2).float() * (-math.log(10000.0) / 20))
    x[:, 0::2] = torch.sin(position * div_term)
    x[:, 1::2] = torch.cos(position * div_term)
    x = x.unsqueeze(0).transpose(0, 1)

    return x.reshape(9,20)+input  



#part III model
class Encoder(nn.Module):
    def __init__(self, n_heads, dropout):
        super(Encoder, self).__init__()
        self.selfattention = SelfAttention(n_heads, dropout)
        self.feedforward = FeedForward()

    def forward(self, inputs):
        outputs = self.selfattention(inputs)
        outputs = self.feedforward(outputs)
        return outputs
    
class Decoder(nn.Module):  
    def __init__(self, n_heads, dropout):
        super(Decoder, self).__init__()
        self.selfattention = SelfAttention(n_heads, dropout)
        self.feedforward = FeedForward()

    def forward(self, inputs):
        outputs = self.selfattention(inputs)
        outputs = self.feedforward(outputs)
        return outputs

class SelfAttention(nn.Module):
    def __init__(self, n_heads, dropout, hid_dim = 20):
        super().__init__()

        self.hidden_dim = hid_dim  #这里取决于embedding的维度
        self.n_heads = n_heads 

        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)

        self.linear = nn.Linear(hid_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(hid_dim)

    def forward(self, input):
    
        batch_size = input.shape[0] #这里的input是batch,9,20

        Q = self.w_q(input)  #batch,9,20
        K = self.w_k(input)
        V = self.w_v(input)

        Q = Q.view(batch_size, -1, self.n_heads, self.hidden_dim // self.n_heads).permute(0,2,1,3)  #batch, n_heads, 9, 20/n_heads
        K = K.view(batch_size, -1, self.n_heads, self.hidden_dim // self.n_heads).permute(0,2,1,3)
        V = V.view(batch_size, -1, self.n_heads, self.hidden_dim // self.n_heads).permute(0,2,1,3)

        a = torch.matmul(Q,K.permute(0,1,3,2)) / 3   # 3 = sqrt(9)
        attention = self.dropout(F.softmax(a, dim = -1))  # batch, n_heads, 9 ,9
        b = torch.matmul(attention, V) # batch, n_heads, 9 ,5
        b = b.transpose(2,3).reshape(batch_size,9,20)

        output = self.linear(b) #batch,9,20
        return self.layernorm(output+input)  # add and norm
    
class FeedForward(nn.Module):
    def __init__(self):
        super(FeedForward, self).__init__()
        d_ff = 512
        d_hid = 20
        self.linear = nn.Sequential(
            nn.Linear(d_hid, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_hid, bias=False) )
        self.layernorm = nn.LayerNorm(d_hid)  #batch,9,20
        
    def forward(self, inputs):
        input0 = inputs
        output = self.linear(inputs)

        return(self.layernorm(input0+output))
    
class Transformer_model(nn.Module):
    def __init__(self, n_heads, dropout):
        super(Transformer_model, self).__init__()
        self.encoder = Encoder(n_heads, dropout)
        self.decoder = Decoder(n_heads, dropout)
        self.predictor = nn.Sequential(
                         nn.Linear(180,256),
                         nn.Dropout(dropout),
                         nn.LeakyReLU(),
                         nn.BatchNorm1d(256),
                         nn.Linear(256,256),
                         nn.Dropout(dropout),
                         nn.LeakyReLU(),
                         nn.BatchNorm1d(256),
                         nn.Linear(256,2))
        
    def forward(self, input):
        output = self.encoder(input)
        output2 = self.decoder(output)
        output2 = output2.reshape(-1, 180)
        pred = self.predictor(output2)

        return(pred)

class trans_IGN(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, num_layers, graph_feat_size, outdim_g3,
                 d_FC_layer, n_FC_layer, dropout, n_tasks, n_heads):
        super(trans_IGN, self).__init__()

        # graph layers for ligand and protein
        self.cov_graph = ModifiedAttentiveFPPredictorV2(node_feat_size, edge_feat_size, num_layers, graph_feat_size, dropout)

        # graph layers for ligand and protein interaction
        self.noncov_graph = DTIConvGraph3Layer(graph_feat_size+1, outdim_g3, dropout)

        # read out
        self.readout = EdgeWeightedSumAndMax(outdim_g3)

        # MLP predictor
        self.FC_graph = nn.Sequential(
            nn.Linear(400,200),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.BatchNorm1d(200)
        )
        self.FC_trans = nn.Sequential(
            nn.Linear(180,200),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.BatchNorm1d(200)
        )
        self.FC_all = nn.Sequential(
            nn.Linear(400,200),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.BatchNorm1d(200),
            nn.Linear(200,2)
        )

        # transformer
        self.encoder = Encoder(n_heads, dropout)
        self.decoder = Decoder(n_heads, dropout)

    def forward(self, bg, bg3, keys):
        atom_feats = bg.ndata.pop('h')
        bond_feats = bg.edata.pop('e')
        atom_feats = self.cov_graph(bg, atom_feats, bond_feats)
        bond_feats3 = bg3.edata['e']
        bond_feats3 = self.noncov_graph(bg3, atom_feats, bond_feats3)
        readouts, weights = self.readout(bg3, bond_feats3)  # readouts: batch, 400
        output_IGN = self.FC_graph(readouts)

        trans_out1 = self.encoder(keys)
        trans_out2 = self.decoder(trans_out1)
        trans_out2 = trans_out2.reshape(-1,180)  # 180 = 9 * 20 transout2: batch,180
        output_trans = self.FC_trans(trans_out2)

        fc_input = torch.cat([output_IGN,output_trans], dim=1)
        output = self.FC_all(fc_input)
        
        return output
    
class ModifiedAttentiveFPPredictorV2(nn.Module):
    def __init__(self,
                 node_feat_size,
                 edge_feat_size,
                 num_layers=2,
                 graph_feat_size=200,
                 dropout=0.):
        super(ModifiedAttentiveFPPredictorV2, self).__init__()

        self.gnn = ModifiedAttentiveFPGNNV2(node_feat_size=node_feat_size,
                                            edge_feat_size=edge_feat_size,
                                            num_layers=num_layers,
                                            graph_feat_size=graph_feat_size,
                                            dropout=dropout)
        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(graph_feat_size, 1)
        )

    def forward(self, g, node_feats, edge_feats):
        sum_node_feats = self.gnn(g, node_feats, edge_feats)
        return sum_node_feats

class ModifiedAttentiveFPGNNV2(nn.Module):
    def __init__(self,
                 node_feat_size,
                 edge_feat_size,
                 num_layers=2,
                 graph_feat_size=200,
                 dropout=0.):
        super(ModifiedAttentiveFPGNNV2, self).__init__()

        self.init_context = GetContext(node_feat_size, edge_feat_size, graph_feat_size, dropout)
        self.gnn_layers = nn.ModuleList()
        self.sum_node_feats = 0
        for _ in range(num_layers - 1):
            self.gnn_layers.append(GNNLayer(graph_feat_size, graph_feat_size, dropout))


    def forward(self, g, node_feats, edge_feats):
        node_feats = self.init_context(g, node_feats, edge_feats)
        self.sum_node_feats = node_feats  # atom update function. from the fusion of ai(new) and mi
        for gnn in self.gnn_layers:
            node_feats = gnn(g, node_feats)
            self.sum_node_feats = self.sum_node_feats + node_feats
        return self.sum_node_feats
    

class GetContext(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, graph_feat_size, dropout):
        super(GetContext, self).__init__()

        self.project_node = nn.Sequential(
            nn.Linear(node_feat_size, graph_feat_size),
            nn.LeakyReLU()
        )
        self.project_edge1 = nn.Sequential(
            nn.Linear(node_feat_size + edge_feat_size, graph_feat_size),
            nn.LeakyReLU()
        )
        self.project_edge2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * graph_feat_size, 1),
            nn.LeakyReLU()
        )
        self.attentive_gru = AttentiveGRU1(graph_feat_size, graph_feat_size,
                                           graph_feat_size, dropout)

    def apply_edges1(self, edges):
        return {'he1': torch.cat([edges.src['hv'], edges.data['he']], dim=1)}

    def apply_edges2(self, edges):
        return {'he2': torch.cat([edges.dst['hv_new'], edges.data['he1']], dim=1)}

    def forward(self, g, node_feats, edge_feats):
        g = g.local_var()
        g.ndata['hv'] = node_feats

        g.ndata['hv_new'] = self.project_node(node_feats)  #ai(new)
        g.edata['he'] = edge_feats


        g.apply_edges(self.apply_edges1) #[aij||bij]
        g.edata['he1'] = self.project_edge1(g.edata['he1'])   #bijnew
        g.apply_edges(self.apply_edges2)  #[aijnew || bijnew]
        logits = self.project_edge2(g.edata['he2'])  #sij

        return self.attentive_gru(g, logits, g.edata['he1'], g.ndata['hv_new']) #hi0
    
class AttentiveGRU1(nn.Module):

    def __init__(self, node_feat_size, edge_feat_size, edge_hidden_size, dropout):
        super(AttentiveGRU1, self).__init__()

        self.edge_transform = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(edge_feat_size, edge_hidden_size)
        )
        self.gru = nn.GRUCell(edge_hidden_size, node_feat_size)

    def forward(self, g, edge_logits, edge_feats, node_feats):
        g = g.local_var()
        g.edata['e'] = edge_softmax(g, edge_logits) * self.edge_transform(edge_feats)
        g.update_all(fn.copy_edge('e', 'm'), fn.sum('m', 'c'))
        context = F.elu(g.ndata['c'])
        return F.relu(self.gru(context, node_feats))
    
class GNNLayer(nn.Module):
    def __init__(self, node_feat_size, graph_feat_size, dropout):
        super(GNNLayer, self).__init__()

        self.project_edge = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * node_feat_size, 1),
            nn.LeakyReLU()
        )
        self.attentive_gru = AttentiveGRU2(node_feat_size, graph_feat_size, dropout)
        self.bn_layer = nn.BatchNorm1d(graph_feat_size)

    def apply_edges(self, edges):
        return {'he': torch.cat([edges.dst['hv'], edges.src['hv']], dim=1)}

    def forward(self, g, node_feats):
        g = g.local_var()
        g.ndata['hv'] = node_feats  
        g.apply_edges(self.apply_edges)
        logits = self.project_edge(g.edata['he'])

        return self.bn_layer(self.attentive_gru(g, logits, node_feats))

class AttentiveGRU2(nn.Module):
    def __init__(self, node_feat_size, edge_hidden_size, dropout):
        super(AttentiveGRU2, self).__init__()

        self.project_node = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(node_feat_size, edge_hidden_size)
        )
        self.gru = nn.GRUCell(edge_hidden_size, node_feat_size)

    def forward(self, g, edge_logits, node_feats):
        g = g.local_var()
        g.edata['a'] = edge_softmax(g, edge_logits)
        g.edata['e'] = g.edata['a'] + 1
        g.ndata['hv'] = self.project_node(node_feats)

        g.update_all(fn.src_mul_edge('hv', 'e', 'm'), fn.sum('m', 'c'))
        context = F.elu(g.ndata['c'])
        return F.relu(self.gru(context, node_feats))
    
class DTIConvGraph3Layer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):  # in_dim = graph module1 output dim + 1
        super(DTIConvGraph3Layer, self).__init__()
        # the MPL for update the edge state
        self.grah_conv = DTIConvGraph3(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.bn_layer = nn.BatchNorm1d(out_dim)

    def forward(self, bg, atom_feats, bond_feats):
        new_feats = self.grah_conv(bg, atom_feats, bond_feats)
        return self.bn_layer(self.dropout(new_feats))
    

class DTIConvGraph3(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DTIConvGraph3, self).__init__()
    # the MPL for update the edge state
        self.mpl = nn.Sequential(nn.Linear(in_dim, out_dim),
                                 nn.LeakyReLU(),
                                 nn.Linear(out_dim, out_dim),
                                 nn.LeakyReLU(),
                                 nn.Linear(out_dim, out_dim),
                                 nn.LeakyReLU())

    def EdgeUpdate(self, edges):
        return {'e': self.mpl(torch.cat([edges.data['e'], edges.data['m']], dim=1))}

    def forward(self, bg, atom_feats, bond_feats):
        bg.ndata['h'] = atom_feats
        bg.edata['e'] = bond_feats
        with bg.local_scope():
            bg.apply_edges(dgl.function.u_add_v('h', 'h', 'm')) 
            bg.apply_edges(self.EdgeUpdate)
            return bg.edata['e']


class EdgeWeightedSumAndMax(nn.Module):
    """
    for normal use, please delete the 'temporary version' line and meanwhile recover the 'normal version'
    """
    def __init__(self, in_feats):
        super(EdgeWeightedSumAndMax, self).__init__()
        self.weight_and_sum = EdgeWeightAndSum(in_feats)

    def forward(self, bg, edge_feats):
        # h_g_sum = self.weight_and_sum(bg, edge_feats)  # normal version
        h_g_sum, weights = self.weight_and_sum(bg, edge_feats)  # temporary version

        # print('bond_feats3:', edge_feats, edge_feats.shape)

        with bg.local_scope():
            bg.edata['e'] = edge_feats
            h_g_max = dgl.max_edges(bg, 'e')

        # print('hg_sum:',h_g_sum,h_g_sum.shape)
        # print('hg_max:',h_g_max,h_g_max.shape)

        h_g = torch.cat([h_g_sum, h_g_max], dim=1)


        # return h_g  # normal version
        return h_g, weights  # temporary version

class EdgeWeightAndSum(nn.Module):
    """
    for normal use, please delete the 'temporary version' line and meanwhile recover the 'normal version'
    """
    def __init__(self, in_feats):
        super(EdgeWeightAndSum, self).__init__()
        self.in_feats = in_feats
        self.atom_weighting = nn.Sequential(
            nn.Linear(in_feats, 1),
            nn.Tanh()
        )

    def forward(self, g, edge_feats):
        with g.local_scope():
            g.edata['e'] = edge_feats
            g.edata['w'] = self.atom_weighting(g.edata['e'])
            weights = g.edata['w']  # temporary version

            h_g_sum = dgl.sum_edges(g, 'e', 'w')  


        # return h_g_sum  # normal version
        return h_g_sum, weights  # temporary version


class FC(nn.Module):
    def __init__(self, d_graph_layer, d_FC_layer, n_FC_layer, dropout, n_tasks):
        super(FC, self).__init__()
        self.d_graph_layer = d_graph_layer
        self.d_FC_layer = d_FC_layer
        self.n_FC_layer = n_FC_layer
        self.dropout = dropout
        self.predict = nn.ModuleList()

        for j in range(self.n_FC_layer):
            if j == 0:
                self.predict.append(nn.Linear(self.d_graph_layer, self.d_FC_layer))
                self.predict.append(nn.Dropout(self.dropout))
                self.predict.append(nn.LeakyReLU())
                self.predict.append(nn.BatchNorm1d(d_FC_layer))
            if j == self.n_FC_layer - 1:
                self.predict.append(nn.Linear(self.d_FC_layer, n_tasks))
            else:
                self.predict.append(nn.Linear(self.d_FC_layer, self.d_FC_layer))
                self.predict.append(nn.Dropout(self.dropout))
                self.predict.append(nn.LeakyReLU())
                self.predict.append(nn.BatchNorm1d(d_FC_layer))

    def forward(self, h):
        for layer in self.predict:
            h = layer(h)
        return h







