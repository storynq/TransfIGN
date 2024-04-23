import rdkit
from graph_constructor import *
from utils import *
from model_trans import *
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
import time
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr
import warnings
import torch
import pandas as pd
import os
from dgl.data.utils import split_dataset
from sklearn.metrics import mean_squared_error
import argparse
import math

torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def run_a_eval_epoch(model, validation_dataloader, device):
    true = []
    pred_BA = []
    pred_EL = []
    key = []
    model.eval()
    with torch.no_grad():
        for i_batch, batch in enumerate(validation_dataloader):
            model.zero_grad()
            bg, bg3, Ys, keys = batch
            keys0 = bls(keys)
            transformer_input = []
            for i in keys0:
                z = torch.tensor(i)
                input = position_encoding(z)
                transformer_input.append(input)
            transformer_input = torch.stack(transformer_input)
            bg, bg3, Ys, key1 = bg.to(device), bg3.to(device), Ys.to(device), transformer_input.to(device)
            outputs = model(bg, bg3, key1)
            output_BA, output_EL = torch.chunk(outputs,2,dim=1)
            output_BA = torch.sigmoid(output_BA)
            output_EL = torch.sigmoid(output_EL)
            true.append(Ys.data.cpu().numpy())
            pred_BA.append(output_BA.data.cpu().numpy())
            pred_EL.append(output_EL.data.cpu().numpy())
            key.append(keys)
    return true, pred_BA, pred_EL, key


lr = 10 ** -3.5
epochs = 5000
batch_size = 48
num_workers = 0
tolerance = 0.0
patience = 70
l2 = 10 ** -6
repetitions = 3
# paras for model
node_feat_size = 40
edge_feat_size_2d = 12
edge_feat_size_3d = 21
graph_feat_size = 256
num_layers = 3
outdim_g3 = 200
d_FC_layer, n_FC_layer = 200, 2
dropout = 0.1
n_tasks = 2
mark = '3d'
path_marker = '/'



if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--graph_ls_path', type=str, default='./prediction_graph/graph_ls_path',
                           help="absolute path for storing graph list objects")
    argparser.add_argument('--graph_dic_path', type=str, default='./prediction_graph/graph_dic_path',
                           help="absolute path for storing graph dictionary objects (temporary files)")
    argparser.add_argument('--model_path', type=str, default='/home/nanqi/InteractionGraphNet-master/model_save/1.pth',
                           help="absolute path for storing pretrained model")
    argparser.add_argument('--cpu', type=bool, default=True,
                           help="using cpu for the prediction (default:True)")
    argparser.add_argument('--gpuid', type=int, default=0,
                           help="the gpu id for the prediction")
    argparser.add_argument('--num_process', type=int, default=12,
                           help="the number of process for generating graph objects")
    argparser.add_argument('--input_path', type=str, default='./examples/ign_input',
                           help="the absoute path for storing ign input files")
    args = argparser.parse_args()
    graph_ls_path, graph_dic_path, model_path, cpu, gpuid, num_process, input_path = args.graph_ls_path, \
                                                                                     args.graph_dic_path, \
                                                                                     args.model_path, \
                                                                                     args.cpu, \
                                                                                     args.gpuid, \
                                                                                     args.num_process, \
                                                                                     args.input_path
    if not os.path.exists('./stats'):
        os.makedirs('./stats')

    if not os.path.exists('./stats'):
        os.makedirs('./stats')

    keys = os.listdir(input_path)
    labels_BA = []
    labels_EL = []
    data_dirs = []
    for key in keys:
        data_dirs.append(input_path + path_marker + key)
        labels_BA.append(0)
        labels_EL.append(0)

    labels_BA = [-1 if math.isnan(x) else x for x in labels_BA]
    labels_EL = [-1 if math.isnan(x) else x for x in labels_EL]
    labels = list(zip(labels_BA, labels_EL))
    limit = None
    dis_threshold = 8.0

    # generating the graph objective using multi process
    test_dataset = GraphDatasetV2MulPro(keys=keys[:limit], labels=labels[:limit], data_dirs=data_dirs[:limit],
                                        graph_ls_path=graph_ls_path,
                                        graph_dic_path=graph_dic_path,
                                        num_process=num_process, dis_threshold=dis_threshold, path_marker=path_marker)
    test_dataloader = DataLoaderX(test_dataset, batch_size, shuffle=False, num_workers=num_workers,
                                  collate_fn=collate_fn_v2_MulPro)


    DTIModel = trans_IGN(node_feat_size=node_feat_size, edge_feat_size=edge_feat_size_3d, num_layers=num_layers,
                                graph_feat_size=graph_feat_size, outdim_g3=outdim_g3,
                                d_FC_layer=d_FC_layer, n_FC_layer=n_FC_layer, dropout=dropout, n_tasks=n_tasks, n_heads=4)
    if cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:%s" % gpuid)
    DTIModel.load_state_dict(torch.load(model_path, map_location='cpu')['model_state_dict'])
    DTIModel.to(device)


    test_true, test_pred_BA, test_pred_EL, key = run_a_eval_epoch(DTIModel, test_dataloader, device)

    test_pred_BA = np.concatenate(np.array(test_pred_BA), 0).flatten()
    test_pred_EL = np.concatenate(np.array(test_pred_EL), 0).flatten()

    key = np.concatenate(np.array(key), 0).flatten()

    res = pd.DataFrame({'key': key, 'pred_BA': test_pred_BA, 'pred_EL': test_pred_EL})
    res.to_csv('./stats/prediction_results.csv', index=False)

