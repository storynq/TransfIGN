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
torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')
import argparse
path_marker = '/'
limit = None
num_process = 12



class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def run_a_train_epoch(model, loss_fn, train_dataloader, optimizer, device):
    # training model for one epoch
    model.train()
    for i_batch, batch in enumerate(train_dataloader):
        model.zero_grad()
        bg, bg3, Ys, keys = batch
        keys0 = bls(keys)
        transformer_input = []
        for i in keys0:
            z = torch.tensor(i)
            input = position_encoding(z)
            transformer_input.append(input)
        transformer_input = torch.stack(transformer_input)
        bg, bg3, Ys, keys = bg.to(device), bg3.to(device), Ys.to(device), transformer_input.to(device)
        outputs = model(bg, bg3, keys)

        Ys = torch.reshape(Ys, (len(Ys),2))

        # mask
        mask = torch.ones_like(Ys)
        mask[Ys == -1] = 0
        Ys_2 = Ys * mask
        output2 = outputs * mask

        #loss calculation
        Ys_BA, Ys_EL = torch.chunk(Ys_2,2,dim=1)
        output_BA, output_EL = torch.chunk(output2,2,dim=1)
        output_EL = torch.sigmoid(output_EL)
        output_BA = torch.sigmoid(output_BA)
        mask_BA, mask_EL = torch.chunk(mask,2,dim=1)
        
        loss1 = torch.sum(loss_fn(output_BA, Ys_BA)*mask_BA)/(torch.sum(mask_BA)+1e-8)
        loss2 = torch.sum(loss_fn(output_EL, Ys_EL)*mask_EL)/(torch.sum(mask_EL)+1e-8)

        loss = loss1*4 + loss2  

        loss.backward()
        optimizer.step()        



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
            bg, bg3, Ys, keys = bg.to(device), bg3.to(device), Ys.to(device), transformer_input.to(device)
            outputs = model(bg, bg3, keys)
            output_BA, output_EL = torch.chunk(outputs,2,dim=1)
            output_BA = torch.sigmoid(output_BA)
            output_EL = torch.sigmoid(output_EL)
            true.append(Ys.data.cpu().numpy())
            pred_BA.append(output_BA.data.cpu().numpy())
            pred_EL.append(output_EL.data.cpu().numpy())
            key.append(keys)
    return true, pred_BA, pred_EL, key

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--gpuid', type=str, default='0', help="gpu id for training model")
    argparser.add_argument('--lr', type=float, default=10 ** -4.0, help="Learning rate")
    argparser.add_argument('--epochs', type=int, default=5000, help="Number of epochs in total")
    argparser.add_argument('--batch_size', type=int, default=200, help="Batch size")
    argparser.add_argument('--tolerance', type=float, default=0.0, help="early stopping tolerance")
    argparser.add_argument('--patience', type=int, default=70, help="early stopping patience")
    argparser.add_argument('--l2', type=float, default=10 ** -6, help="L2 regularization")
    argparser.add_argument('--repetitions', type=int, default=3, help="the number of independent runs")
    argparser.add_argument('--node_feat_size', type=int, default=40)
    argparser.add_argument('--edge_feat_size_2d', type=int, default=12)
    argparser.add_argument('--edge_feat_size_3d', type=int, default=21)
    argparser.add_argument('--graph_feat_size', type=int, default=256)
    argparser.add_argument('--num_layers', type=int, default=3, help='the number of intra-molecular layers')
    argparser.add_argument('--outdim_g3', type=int, default=200, help='the output dim of inter-molecular layers')
    argparser.add_argument('--d_FC_layer', type=int, default=200, help='the hidden layer size of task networks')
    argparser.add_argument('--n_FC_layer', type=int, default=2, help='the number of hidden layers of task networks')
    argparser.add_argument('--dropout', type=float, default=0.1, help='dropout ratio')
    argparser.add_argument('--n_tasks', type=int, default=1)
    argparser.add_argument('--num_workers', type=int, default=0,
                           help='number of workers for loading data in Dataloader')
    argparser.add_argument('--model_save_dir', type=str, default='./model_save', help='path for saving model')
    argparser.add_argument('--mark', type=str, default='3d')

    args = argparser.parse_args()
    gpuid, lr, epochs, batch_size, num_workers, model_save_dir = args.gpuid, args.lr, args.epochs, args.batch_size, args.num_workers, args.model_save_dir
    tolerance, patience, l2, repetitions = args.tolerance, args.patience, args.l2, args.repetitions
    # paras for model
    node_feat_size, edge_feat_size_2d, edge_feat_size_3d = args.node_feat_size, args.edge_feat_size_2d, args.edge_feat_size_3d
    graph_feat_size, num_layers = args.graph_feat_size, args.num_layers
    outdim_g3, d_FC_layer, n_FC_layer, dropout, n_tasks, mark = args.outdim_g3, args.d_FC_layer, args.n_FC_layer, args.dropout, args.n_tasks, args.mark

    HOME_PATH = os.getcwd()
    all_data = pd.read_excel('./data/All_Data.xlsx')

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    if not os.path.exists('./stats'):
        os.makedirs('./stats')

    # data
    train_dir = './virtual_mutation/ign_input_examples/train'
    valid_dir = './virtual_mutation/ign_input_examples/valid'
    

    # training data
    train_keys = os.listdir(train_dir)
    train_labels_BA = []
    train_labels_EL = []
    train_data_dirs = []
    for key in train_keys:
        train_labels_BA.append(all_data[all_data['PDB'] == key]['BA'].values[0]) #label ba
        train_labels_EL.append(all_data[all_data['PDB'] == key]['EL'].values[0]) #label el
        train_data_dirs.append(train_dir + path_marker + key) #key in fact key is 'pdb id'

    train_labels_BA = [-1 if math.isnan(x) else x for x in train_labels_BA]
    train_labels_EL = [-1 if math.isnan(x) else x for x in train_labels_EL]
    train_labels = list(zip(train_labels_BA, train_labels_EL))



    # validtion data
    valid_keys = os.listdir(valid_dir)
    valid_labels_BA = []
    valid_labels_EL = []
    valid_data_dirs = []
    for key in valid_keys:
        valid_labels_BA.append(all_data[all_data['PDB'] == key]['BA'].values[0])
        valid_labels_EL.append(all_data[all_data['PDB'] == key]['EL'].values[0])
        valid_data_dirs.append(valid_dir + path_marker + key)

    valid_labels_BA = [-1 if math.isnan(x) else x for x in valid_labels_BA]
    valid_labels_EL = [-1 if math.isnan(x) else x for x in valid_labels_EL]
    valid_labels = list(zip(valid_labels_BA, valid_labels_EL))



    # generating the graph objective using multi process
    train_dataset = GraphDatasetV2MulPro(keys=train_keys[:limit], labels=train_labels[:limit], data_dirs=train_data_dirs[:limit],
                                        graph_ls_path= './codes_transfIGN/train_graph/graph_ls_path',
                                        graph_dic_path= './codes_transfIGN/train_graph/graph_ls_path',
                                        num_process=num_process, path_marker=path_marker)

    valid_dataset = GraphDatasetV2MulPro(keys=valid_keys[:limit], labels=valid_labels[:limit], data_dirs=valid_data_dirs[:limit],
                                         graph_ls_path= './codes_transfIGN/valid_graph/graph_dic_path',
                                         graph_dic_path= './codes_transfIGN/valid_graph/graph_dic_path',
                                         num_process=num_process, path_marker=path_marker)



    stat_res = []


    for repetition_th in range(repetitions):
        set_random_seed(repetition_th)
        # print('the number of train data:', len(train_dataset))
        # print('the number of valid data:', len(valid_dataset))
        train_dataloader = DataLoaderX(train_dataset, batch_size, shuffle=True, num_workers=num_workers,
                                       collate_fn=collate_fn_v2_MulPro)
        valid_dataloader = DataLoaderX(valid_dataset, batch_size, shuffle=False, num_workers=num_workers,
                                       collate_fn=collate_fn_v2_MulPro)

        # model
        Trans_model = trans_IGN(node_feat_size=node_feat_size, edge_feat_size=edge_feat_size_3d, num_layers=num_layers,
                                graph_feat_size=graph_feat_size, outdim_g3=outdim_g3,
                                d_FC_layer=d_FC_layer, n_FC_layer=n_FC_layer, dropout=dropout, n_tasks=n_tasks, n_heads=4)
        
        print('number of parameters : ', sum(p.numel() for p in Trans_model.parameters() if p.requires_grad))
        if repetition_th == 0:
            print(Trans_model)
        device = torch.device("cuda:%s" % gpuid if torch.cuda.is_available() else "cpu")
        # device = torch.device('cpu')
        Trans_model.to(device)
        optimizer = torch.optim.Adam(Trans_model.parameters(), lr=lr, weight_decay=l2)
        dt = datetime.datetime.now()

        filename = './model_save/Trans{}_{:02d}_{:02d}_{:02d}_{:d}.pth'.format(dt.date(), dt.hour, dt.minute, dt.second,
                                                                          dt.microsecond)


        stopper = EarlyStopping(mode='lower', patience=patience, tolerance=tolerance,
                                filename=filename)
        loss_fn = nn.BCELoss()

        for epoch in range(epochs):
            st = time.time()
            # train
            run_a_train_epoch(Trans_model, loss_fn, train_dataloader, optimizer, device)
            # validation
            train_true, train_pred_BA, train_pred_EL, _ = run_a_eval_epoch(Trans_model, train_dataloader, device)
            valid_true, valid_pred_BA, valid_pred_EL, _ = run_a_eval_epoch(Trans_model, valid_dataloader, device)
            
            train_true = np.concatenate(np.array(train_true), 0)
            train_pred_BA = np.concatenate(np.array(train_pred_BA), 0)
            train_pred_EL = np.concatenate(np.array(train_pred_EL), 0)

            valid_true = np.concatenate(np.array(valid_true), 0)
            valid_pred_BA = np.concatenate(np.array(valid_pred_BA), 0)
            valid_pred_EL = np.concatenate(np.array(valid_pred_EL), 0)

            train_true_BA, train_true_EL = np.hsplit(train_true.squeeze(-1),2)
            valid_true_BA, valid_true_EL = np.hsplit(valid_true.squeeze(-1),2)

            train_true_BA = train_true_BA.reshape(len(train_pred_BA),1)
            valid_true_BA = valid_true_BA.reshape(len(valid_pred_BA),1)

            mask_BA = torch.ones_like(torch.tensor(train_true_BA))
            mask_BA[torch.tensor(train_true_BA) == -1] = 0
            train_true_BA2 = torch.tensor(train_true_BA) * mask_BA
            train_pred_BA2 = torch.tensor(train_pred_BA) * mask_BA

            mask_EL = torch.ones_like(torch.tensor(train_true_EL))
            mask_EL[torch.tensor(train_true_EL) == -1] = 0
            train_true_EL2 = torch.tensor(train_true_EL) * mask_EL
            train_pred_EL2 = torch.tensor(train_pred_EL) * mask_EL   

            mask_BA2 = torch.ones_like(torch.tensor(valid_true_BA))      
            mask_BA2[torch.tensor(valid_true_BA) == -1] = 0
            valid_true_BA2 = torch.tensor(valid_true_BA) * mask_BA2
            valid_pred_BA2 = torch.tensor(valid_pred_BA) * mask_BA2

            mask_EL2 = torch.ones_like(torch.tensor(valid_true_EL))      
            mask_EL2[torch.tensor(valid_true_EL) == -1] = 0
            valid_true_EL2 = torch.tensor(valid_true_EL) * mask_EL2
            valid_pred_EL2 = torch.tensor(valid_pred_EL) * mask_EL2        

            train_rmse = np.sqrt(mean_squared_error(train_true_BA2, train_pred_BA2)) + np.sqrt(mean_squared_error(train_true_EL2, train_pred_EL2))
            valid_rmse = np.sqrt(mean_squared_error(valid_true_BA2, valid_pred_BA2)) + np.sqrt(mean_squared_error(valid_true_EL2, valid_pred_EL2))
            #
            early_stop = stopper.step(valid_rmse, Trans_model)
            end = time.time()
            if early_stop:
                break
            print(
                "epoch:%s \t train_rmse:%.4f \t valid_rmse:%.4f \t time:%.3f s" % (epoch, train_rmse, valid_rmse, end - st))

        # load the best model
        stopper.load_checkpoint(Trans_model)
        train_dataloader = DataLoaderX(train_dataset, batch_size, shuffle=False, num_workers=num_workers,
                                       collate_fn=collate_fn_v2_MulPro)
        valid_dataloader = DataLoaderX(valid_dataset, batch_size, shuffle=False, num_workers=num_workers,
                                       collate_fn=collate_fn_v2_MulPro)



        train_true, train_pred_BA, train_pred_EL, _ = run_a_eval_epoch(Trans_model, train_dataloader, device)
        valid_true, valid_pred_BA, valid_pred_EL, _ = run_a_eval_epoch(Trans_model, valid_dataloader, device)


        # metrics
        train_true = np.concatenate(np.array(train_true), 0)
        train_true_BA, train_true_EL = np.hsplit(train_true,2)  

        train_true_BA = np.concatenate(np.array(train_true_BA), 0).flatten()
        train_true_EL = np.concatenate(np.array(train_true_EL), 0).flatten()
        train_pred_BA = np.concatenate(np.array(train_pred_BA), 0).flatten()
        train_pred_EL = np.concatenate(np.array(train_pred_EL), 0).flatten()

        valid_true = np.concatenate(np.array(valid_true), 0)
        valid_true_BA, valid_true_EL = np.hsplit(valid_true,2)  

        valid_true_BA = np.concatenate(np.array(valid_true_BA), 0).flatten()
        valid_true_EL = np.concatenate(np.array(valid_true_EL), 0).flatten()
        valid_pred_BA = np.concatenate(np.array(valid_pred_BA), 0).flatten()
        valid_pred_EL = np.concatenate(np.array(valid_pred_EL), 0).flatten()


    
        pd_tr = pd.DataFrame({'key': train_keys, 'train_true_BA': train_true_BA, 'train_pred_BA': train_pred_BA, 'train_true_EL': train_true_EL, 'train_pred_EL': train_pred_EL})
        pd_va = pd.DataFrame({'key': valid_keys, 'valid_true_BA': valid_true_BA, 'valid_pred_BA': valid_pred_BA, 'valid_true_EL': valid_true_EL, 'valid_pred_EL': valid_pred_EL})

        pd_tr.to_csv('/home/nanqi/BA_EL2/model2/stats/trans{}_{:02d}_{:02d}_{:02d}_{:d}_tr.csv'.format(dt.date(), dt.hour, dt.minute, dt.second,
                                                                          dt.microsecond), index=False)
        pd_va.to_csv('/home/nanqi/BA_EL2/model2/stats/trans{}_{:02d}_{:02d}_{:02d}_{:d}_va.csv'.format(dt.date(), dt.hour, dt.minute, dt.second,
                                                                          dt.microsecond), index=False)

        mask_BAt = torch.ones_like(torch.tensor(train_true_BA))
        mask_BAt[torch.tensor(train_true_BA) == -1] = 0
        train_true_BA2 = torch.tensor(train_true_BA) * mask_BAt
        train_pred_BA2 = torch.tensor(train_pred_BA) * mask_BAt           
        train_rmse, train_r2, train_mae, train_rp = np.sqrt(mean_squared_error(train_true_BA2, train_pred_BA2)), \
                                                    r2_score(train_true_BA2, train_pred_BA2), \
                                                    mean_absolute_error(train_true_BA2, train_pred_BA2), \
                                                    pearsonr(train_true_BA2, train_pred_BA2)
                                                    
        mask_BAv = torch.ones_like(torch.tensor(valid_true_BA))
        mask_BAv[torch.tensor(valid_true_BA) == -1] = 0
        valid_true_BA2 = torch.tensor(valid_true_BA) * mask_BAv
        valid_pred_BA2 = torch.tensor(valid_pred_BA) * mask_BAv                                                    
        valid_rmse, valid_r2, valid_mae, valid_rp = np.sqrt(mean_squared_error(valid_true_BA2, valid_pred_BA2)), \
                                                    r2_score(valid_true_BA2, valid_pred_BA2), \
                                                    mean_absolute_error(valid_true_BA2, valid_pred_BA2), \
                                                    pearsonr(valid_true_BA2, valid_pred_BA2)
                                                    

        mask_ELt = torch.ones_like(torch.tensor(train_true_EL))
        mask_ELt[torch.tensor(train_true_EL) == -1] = 0
        train_true_EL2 = torch.tensor(train_true_EL) * mask_ELt
        train_pred_EL2 = torch.tensor(train_pred_EL) * mask_ELt           
        train_rmse2, train_r22, train_mae2, train_rp2 = np.sqrt(mean_squared_error(train_true_EL2, train_pred_EL2)), \
                                                    r2_score(train_true_EL2, train_pred_EL2), \
                                                    mean_absolute_error(train_true_EL2, train_pred_EL2), \
                                                    pearsonr(train_true_EL2, train_pred_EL2)
                                                    
        mask_ELv = torch.ones_like(torch.tensor(valid_true_EL))
        mask_ELv[torch.tensor(valid_true_EL) == -1] = 0
        valid_true_EL2 = torch.tensor(valid_true_EL) * mask_ELv
        valid_pred_EL2 = torch.tensor(valid_pred_EL) * mask_ELv                                                    
        valid_rmse2, valid_r22, valid_mae2, valid_rp2 = np.sqrt(mean_squared_error(valid_true_EL2, valid_pred_EL2)), \
                                                    r2_score(valid_true_EL2, valid_pred_EL2), \
                                                    mean_absolute_error(valid_true_EL2, valid_pred_EL2), \
                                                    pearsonr(valid_true_EL2, valid_pred_EL2)
                                                    

        print('***best %s model***' % repetition_th)
        print("train_rmse:%.4f \t train_r2:%.4f \t train_mae:%.4f \t train_rp:%.4f" % (
            train_rmse, train_r2, train_mae, train_rp[0]))
        print("valid_rmse:%.4f \t valid_r2:%.4f \t valid_mae:%.4f \t valid_rp:%.4f" % (
            valid_rmse, valid_r2, valid_mae, valid_rp[0]))
        print("train_rmseEL:%.4f \t train_r2EL:%.4f \t train_maeEL:%.4f \t train_rpEL:%.4f" % (
            train_rmse2, train_r22, train_mae2, train_rp2[0]))
        print("valid_rmse:%.4f \t valid_r2:%.4f \t valid_mae:%.4f \t valid_rp:%.4f" % (
            valid_rmse2, valid_r22, valid_mae2, valid_rp2[0]))
        
        stat_res.append([repetition_th, 'trainBA', train_rmse, train_r2, train_mae, train_rp[0]])
        stat_res.append([repetition_th, 'validBA', valid_rmse, valid_r2, valid_mae, valid_rp[0]])
        stat_res.append([repetition_th, 'trainEL', train_rmse2, train_r22, train_mae2, train_rp2[0]])
        stat_res.append([repetition_th, 'validEL', valid_rmse2, valid_r22, valid_mae2, valid_rp2[0]])

    stat_res_pd = pd.DataFrame(stat_res, columns=['repetition', 'group', 'rmse', 'r2', 'mae', 'rp'])
    stat_res_pd.to_csv(
        './stats/trans{}_{:02d}_{:02d}_{:02d}_{:d}.csv'.format(dt.date(), dt.hour, dt.minute, dt.second, dt.microsecond),
        index=False)
    print(stat_res_pd[stat_res_pd.group == 'train'].mean().values[-4:],
          stat_res_pd[stat_res_pd.group == 'train'].std().values[-4:])
    print(stat_res_pd[stat_res_pd.group == 'valid'].mean().values[-4:],
          stat_res_pd[stat_res_pd.group == 'valid'].std().values[-4:])
