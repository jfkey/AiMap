import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset


# # inference datasets
# class CutInfData(Dataset):
#     def __init__(self, node_file, cut_file, cell_file):
#         # node_id, node_features,
#         self.node_data = pd.read_csv(node_file)
#         self.node_data.set_index('node_id', inplace=True)
#         for col in ['fanout', 'level', 'inv', 'fanout_p1', 'level_p1', 'inv_p1', 'fanout_p2', 'level_p2', 'inv_p2', 're_level']:
#             mean_val, std_val = self.node_data[col].mean(), self.node_data[col].std()
#             self.node_data[col] = self.node_data[col].apply(lambda x: (x - mean_val) / std_val if std_val != 0 else x)
#         self.node_data.fillna(0, inplace=True)
#
#         # node_id, c1, c2, c3, c4, c5, .... , cell_name, ..., phase
#         self.cut_data = pd.read_csv(cut_file)
#         for col in ['nleaves', 'nvolume', 'p1_nleaves', 'p2_nleaves', 'p1_nvolume', 'p2_nvolume', 'max_level', 'min_level', 'level_gap', 'max_fanout', 'min_fanout', 'fanout_gap',
#                     's_delay_r', 's_delay_f', 's_area', 's_f1_delay', 's_f2_delay', 's_f3_delay', 's_f4_delay', 's_f5_delay', 's_f6_delay']:
#             mean_val, std_val = self.cut_data[col].mean(), self.cut_data[col].std()
#             self.cut_data[col] = self.cut_data[col].apply(lambda x: (x - mean_val) / std_val if std_val != 0 else x)
#         self.cut_data.fillna(0, inplace=True)
#
#         # cell_name, cell features
#         self.cell_data = pd.read_csv(cell_file)
#         self.cell_data.set_index('cell_name', inplace=True)
#         for col in ['area', 'leakage_power', 'n_inputs', 'n_outputs', 'cap1', 'cap2', 'cap3', 'cap4', 'cap5', 'delay_p1',
#                     'ld_s_p1', 'delay_s_p1', 'ld_m_p1', 'delay_m_p1', 'ld_l_p1', 'delay_l_p1', 'delay_p2', 'ld_s_p2', 'delay_s_p2',
#                     'ld_m_p2', 'delay_m_p2', 'ld_l_p2', 'delay_l_p2', 'delay_p3', 'ld_s_p3', 'delay_s_p3', 'ld_m_p3', 'delay_m_p3',
#                     'ld_l_p3', 'delay_l_p3', 'delay_p4', 'ld_s_p4', 'delay_s_p4', 'ld_m_p4', 'delay_m_p4', 'ld_l_p4', 'delay_l_p4',
#                     'delay_p5', 'ld_s_p5', 'delay_s_p5', 'ld_m_p5', 'delay_m_p5', 'ld_l_p5', 'delay_l_p5', 'delay_all', 'delay_avg']:
#             mean_val, std_val = self.cell_data[col].mean(), self.cell_data[col].std()
#             self.cell_data[col] = self.cell_data[col].apply(lambda x: (x - mean_val) / std_val if std_val != 0 else x)
#         self.cell_data.fillna(0, inplace=True)
#
#
#     def __len__(self):
#         return len(self.cut_data)
#
#     def __getitem__(self, idx):
#         cutCell_fea = self.cut_data.iloc[idx]
#         # get node features; dim= 10
#         node_id = cutCell_fea['node_id']
#         node_fea = self.node_data.loc[node_id]
#         node_fea = torch.tensor(node_fea, dtype=torch.float)
#
#         # get cut features   dim= 61
#         cut_ids = []
#         for i in range(5):
#             if (cutCell_fea['c'+str(i+1)] != -1):
#                 cut_ids.append(cutCell_fea['c'+str(i+1)])
#         cut_fea = self.node_data.loc[cut_ids[0]]
#         cut_fea_0 = self.node_data.loc[cut_ids[0]].fillna(0)
#
#         for i, cid in enumerate(cut_ids):
#             if i == 0: continue
#             cut_fea = pd.concat([cut_fea, self.node_data.loc[cut_ids[i]]], axis =0)
#
#         for i in range(len(cut_ids), 5):
#             cut_fea = pd.concat([cut_fea, cut_fea_0], axis=0)
#
#         cut_fea = pd.concat([cut_fea, cutCell_fea[6:17]], axis=0)
#         cut_fea = torch.tensor(cut_fea, dtype=torch.float)
#
#         # get cell features dim=56
#         cell_name = cutCell_fea['cell_name']
#         cell_fea = self.cell_data.loc[cell_name]
#         cell_fea = pd.concat([cell_fea, cutCell_fea[19:29]], axis=0)
#         cell_fea = torch.tensor(cell_fea, dtype=torch.float)
#
#         return node_fea, cut_fea, cell_fea


class CutInfData(Dataset):
    def __init__(self, node_files, cut_files, cell_files, levelEmb, fanoutEmb, rootFanoutEmb, min_arr, max_arr):
        self.levelEmb, self.fanoutEmb, self.rootFanoutEmb = levelEmb, fanoutEmb, rootFanoutEmb
        self.min_arr, self.max_arr = min_arr, max_arr
        # node_id, node_features,
        dfArr = []
        nodeCapacity= []
        bef = 0
        for node_file in node_files:
            tmpdf = pd.read_csv(node_file)
            tmpdf['node_id'] = tmpdf['node_id'].apply(lambda x: x+bef )
            nodeCapacity.append(bef)
            bef = bef + len(tmpdf)
            dfArr.append(tmpdf)
        self.node_data = pd.concat(dfArr, ignore_index=True)

        # self.node_data.set_index('node_id', inplace=True)
        for i,  col in enumerate(['inv',  'inv_p1',   'inv_p2', 're_level']):
            # minv, maxv = self.node_data[col].min(), self.node_data[col].max()
            min_val, max_val = self.min_arr[i], self.max_arr[i]
            self.node_data[col] = self.node_data[col].apply(lambda x: (x - min_val) /(max_val-min_val)* 2-1)
        self.node_data.fillna(0, inplace=True)

        # node_id, c1, c2, c3, c4, c5, .... , cell_name, ..., phase

        #self.cut_data = pd.read_csv(cut_file)
        dfArr = []
        for idx, cut_file in enumerate(cut_files):
            tmpdf =  pd.read_csv(cut_file)
            tmpdf['node_id'] = tmpdf['node_id'].apply(lambda x: x + nodeCapacity[idx])
            tmpdf['c1'] = tmpdf['c1'].apply(lambda x: x + nodeCapacity[idx] if x != -1 else x)
            tmpdf['c2'] = tmpdf['c2'].apply(lambda x: x + nodeCapacity[idx] if x != -1 else x)
            tmpdf['c3'] = tmpdf['c3'].apply(lambda x: x + nodeCapacity[idx] if x != -1 else x)
            tmpdf['c4'] = tmpdf['c4'].apply(lambda x: x + nodeCapacity[idx] if x != -1 else x)
            tmpdf['c5'] = tmpdf['c5'].apply(lambda x: x + nodeCapacity[idx] if x != -1 else x)
            dfArr.append(tmpdf)
        self.cut_data = pd.concat(dfArr, ignore_index=True)
        # root_fanout, phase
        for i, col in enumerate(['nleaves', 'nvolume', 'p1_nleaves', 'p2_nleaves', 'p1_nvolume', 'p2_nvolume', 'max_level', 'min_level', 'level_gap', 'max_fanout', 'min_fanout', 'fanout_gap',
                    's_delay_r', 's_delay_f', 's_area', 'est_delay1', 'est_delay2', 'est_delay3', 'est_delay4', 's_f1_delay', 's_f2_delay', 's_f3_delay', 's_f4_delay', 's_f5_delay', 's_f6_delay']):
            # min_val, max_val = self.cut_data[col].min(), self.cut_data[col].max()
            min_val, max_val = self.min_arr[i+4], self.max_arr[i+4]
            self.cut_data[col] = self.cut_data[col].apply(lambda x: (x - min_val) / (max_val - min_val)* 2 -1)


        self.cut_data.fillna(0, inplace=True)

        # cell_name, cell features, only read one cell file
        dfArr = []
        for idx, cell_file in enumerate(cell_files):
            tmpdf = pd.read_csv(cell_file)
            dfArr.append(tmpdf)
            break
        self.cell_data = pd.concat(dfArr, ignore_index=True)
        # self.cell_data = pd.read_csv(cell_file)
        self.cell_data.set_index('cell_name', inplace=True)
        for i, col in enumerate(['area', 'leakage_power', 'n_inputs', 'n_outputs', 'cap1', 'cap2', 'cap3', 'cap4', 'cap5', 'delay_p1',
                    'ld_s_p1', 'delay_s_p1', 'ld_m_p1', 'delay_m_p1', 'ld_l_p1', 'delay_l_p1', 'delay_p2', 'ld_s_p2', 'delay_s_p2',
                    'ld_m_p2', 'delay_m_p2', 'ld_l_p2', 'delay_l_p2', 'delay_p3', 'ld_s_p3', 'delay_s_p3', 'ld_m_p3', 'delay_m_p3',
                    'ld_l_p3', 'delay_l_p3', 'delay_p4', 'ld_s_p4', 'delay_s_p4', 'ld_m_p4', 'delay_m_p4', 'ld_l_p4', 'delay_l_p4',
                    'delay_p5', 'ld_s_p5', 'delay_s_p5', 'ld_m_p5', 'delay_m_p5', 'ld_l_p5', 'delay_l_p5', 'delay_all', 'delay_avg']):
            min_val, max_val = self.min_arr[i + 29], self.max_arr[i + 29]
            # min_val, max_val = self.cell_data[col].min(), self.cell_data[col].max()
            self.cell_data[col] = self.cell_data[col].apply(lambda x:  (x - min_val) / (max_val - min_val)* 2 -1)

        self.cell_data.fillna(0, inplace=True)

        # node_id, phase, cell_name, delay, max_delay

        # self.labels_data = pd.concat(dfArr, ignore_index=True)
        # self.cutCell = pd.merge(self.cut_data, self.labels_data, how='inner', on=['node_id', 'phase', 'cell_name', 'cut_hashID'])
        # self.mean, self.std = self.cutCell['delay'].mean(), self.cutCell['delay'].std()
        # self.cutCell['delay'] = self.cutCell['delay'].apply(lambda x: (x - self.mean) / self.std if self.std != 0 else x)


    def __len__(self):
        return len(self.cut_data)

    def __getitem__(self, idx):
        cutCell_fea = self.cut_data.iloc[idx]
        # 1. get root node features; 1x28
        node_id = cutCell_fea['node_id']
        # node_fea = torch.tensor(node_fea, dtype=torch.float)
        other_fea = self.node_data.loc[node_id, ['inv', 'inv_p1', 'inv_p2', 're_level']]
        node_fea = self.node_data.loc[node_id]
        nF = self.fanoutEmb(torch.LongTensor([node_fea['fanout']]))
        nL = self.levelEmb(torch.LongTensor([node_fea['level']]))
        nFp1 = self.fanoutEmb(torch.LongTensor([node_fea['fanout_p1']]))
        nLp1 = self.levelEmb(torch.LongTensor([node_fea['level_p1']]))
        nFp2 = self.fanoutEmb(torch.LongTensor([node_fea['fanout_p2']]))
        nLp2 = self.levelEmb(torch.LongTensor([node_fea['level_p2']]))
        other_fea = torch.FloatTensor(other_fea.values).unsqueeze(0)
        node_fea = torch.cat([nF, nL, nFp1, nLp1, nFp2, nLp2, other_fea], dim=1)

        # 1. get leaves of the cut features: 6x28
        cut_ids = []
        for i in range(5):
            if (cutCell_fea['c' + str(i + 1)] != -1):
                cut_ids.append(cutCell_fea['c' + str(i + 1)])
        for i, node_id in enumerate(cut_ids):
            # tmp_node_fea = self.node_data.loc[node_id]
            other_fea = self.node_data.loc[node_id, ['inv', 'inv_p1', 'inv_p2', 're_level']]
            tmp_node_fea = self.node_data.loc[node_id]
            nF = self.fanoutEmb(torch.LongTensor([tmp_node_fea['fanout']]))
            nL = self.levelEmb(torch.LongTensor([tmp_node_fea['level']]))
            nFp1 = self.fanoutEmb(torch.LongTensor([tmp_node_fea['fanout_p1']]))
            nLp1 = self.levelEmb(torch.LongTensor([tmp_node_fea['level_p1']]))
            nFp2 = self.fanoutEmb(torch.LongTensor([tmp_node_fea['fanout_p2']]))
            nLp2 = self.levelEmb(torch.LongTensor([tmp_node_fea['level_p2']]))
            other_fea = torch.FloatTensor(other_fea.values).unsqueeze(0)
            tmp_node_fea = torch.cat([nF, nL, nFp1, nLp1, nFp2, nLp2, other_fea], dim=1)
            node_fea = torch.cat([node_fea, tmp_node_fea], dim=0)

        for i in range(len(cut_ids), 5):
            tmp_node_fea_zero = torch.zeros([1,node_fea.shape[1]], dtype=torch.float32)
            node_fea = torch.cat([node_fea, tmp_node_fea_zero], dim=0)

        # 3. get cut structure fea:
        cut_fea = torch.tensor(cutCell_fea[6:18], dtype =torch.float32).unsqueeze(0)
        nF = self.rootFanoutEmb(torch.LongTensor([cutCell_fea['root_fanout']]))
        cut_fea = torch.cat([cut_fea, nF], dim=1)
        # 4. get cell feature
        cell_name = cutCell_fea['cell_name']
        cell_fea = self.cell_data.loc[cell_name]
        cell_fea = pd.concat([cell_fea, cutCell_fea[20:23],  cutCell_fea[24:35]], axis=0)
        cell_fea = torch.tensor(cell_fea, dtype=torch.float32).unsqueeze(0)

        return node_fea, cut_fea, cell_fea

class CutCellData(Dataset):
    def __init__(self, node_files, cut_files, cell_files, labels_files, levelEmb, fanoutEmb, rootFanoutEmb):
        self.levelEmb, self.fanoutEmb, self.rootFanoutEmb = levelEmb, fanoutEmb, rootFanoutEmb
        self.min_arr, self.max_arr = [], []
        # node_id, node_features,
        dfArr = []
        nodeCapacity= []
        bef = 0
        for node_file in node_files:
            tmpdf = pd.read_csv(node_file)
            tmpdf['node_id'] = tmpdf['node_id'].apply(lambda x: x+bef )
            nodeCapacity.append(bef)
            bef = bef + len(tmpdf)
            dfArr.append(tmpdf)
        self.node_data = pd.concat(dfArr, ignore_index=True)

        # self.node_data.set_index('node_id', inplace=True)
        for col in ['inv',  'inv_p1',   'inv_p2', 're_level']:
            min_val, max_val = self.node_data[col].min(), self.node_data[col].max()
            self.node_data[col] = self.node_data[col].apply(lambda x: (x - min_val) /(max_val-min_val)* 2-1)
            self.min_arr.append(min_val)
            self.max_arr.append(max_val)
        self.node_data.fillna(0, inplace=True)

        # node_id, c1, c2, c3, c4, c5, .... , cell_name, ..., phase

        #self.cut_data = pd.read_csv(cut_file)
        dfArr = []
        for idx, cut_file in enumerate(cut_files):
            tmpdf =  pd.read_csv(cut_file)
            tmpdf['node_id'] = tmpdf['node_id'].apply(lambda x: x + nodeCapacity[idx])
            tmpdf['c1'] = tmpdf['c1'].apply(lambda x: x + nodeCapacity[idx] if x != -1 else x)
            tmpdf['c2'] = tmpdf['c2'].apply(lambda x: x + nodeCapacity[idx] if x != -1 else x)
            tmpdf['c3'] = tmpdf['c3'].apply(lambda x: x + nodeCapacity[idx] if x != -1 else x)
            tmpdf['c4'] = tmpdf['c4'].apply(lambda x: x + nodeCapacity[idx] if x != -1 else x)
            tmpdf['c5'] = tmpdf['c5'].apply(lambda x: x + nodeCapacity[idx] if x != -1 else x)
            dfArr.append(tmpdf)
        self.cut_data = pd.concat(dfArr, ignore_index=True)
        # root_fanout, phase
        for col in ['nleaves', 'nvolume', 'p1_nleaves', 'p2_nleaves', 'p1_nvolume', 'p2_nvolume', 'max_level', 'min_level', 'level_gap', 'max_fanout', 'min_fanout', 'fanout_gap',
                    's_delay_r', 's_delay_f', 's_area', 'est_delay1', 'est_delay2', 'est_delay3', 'est_delay4', 's_f1_delay', 's_f2_delay', 's_f3_delay', 's_f4_delay', 's_f5_delay', 's_f6_delay']:
            min_val, max_val = self.cut_data[col].min(), self.cut_data[col].max()
            self.cut_data[col] = self.cut_data[col].apply(lambda x: (x - min_val) / (max_val - min_val)* 2 -1)
            self.min_arr.append(min_val)
            self.max_arr.append(max_val)

        self.cut_data.fillna(0, inplace=True)

        # cell_name, cell features, only read one cell file
        dfArr = []
        for idx, cell_file in enumerate(cell_files):
            tmpdf = pd.read_csv(cell_file)
            dfArr.append(tmpdf)
            break
        self.cell_data = pd.concat(dfArr, ignore_index=True)
        # self.cell_data = pd.read_csv(cell_file)
        self.cell_data.set_index('cell_name', inplace=True)
        for col in ['area', 'leakage_power', 'n_inputs', 'n_outputs', 'cap1', 'cap2', 'cap3', 'cap4', 'cap5', 'delay_p1',
                    'ld_s_p1', 'delay_s_p1', 'ld_m_p1', 'delay_m_p1', 'ld_l_p1', 'delay_l_p1', 'delay_p2', 'ld_s_p2', 'delay_s_p2',
                    'ld_m_p2', 'delay_m_p2', 'ld_l_p2', 'delay_l_p2', 'delay_p3', 'ld_s_p3', 'delay_s_p3', 'ld_m_p3', 'delay_m_p3',
                    'ld_l_p3', 'delay_l_p3', 'delay_p4', 'ld_s_p4', 'delay_s_p4', 'ld_m_p4', 'delay_m_p4', 'ld_l_p4', 'delay_l_p4',
                    'delay_p5', 'ld_s_p5', 'delay_s_p5', 'ld_m_p5', 'delay_m_p5', 'ld_l_p5', 'delay_l_p5', 'delay_all', 'delay_avg']:
            min_val, max_val = self.cell_data[col].min(), self.cell_data[col].max()
            self.min_arr.append(min_val)
            self.max_arr.append(max_val)
            self.cell_data[col] = self.cell_data[col].apply(lambda x:  (x - min_val) / (max_val - min_val)* 2 -1)

        self.cell_data.fillna(0, inplace=True)

        # node_id, phase, cell_name, delay, max_delay
        dfArr = []
        for idx, label_file in enumerate(labels_files):
            tmpdf = pd.read_csv(label_file)
            tmpdf['node_id'] = tmpdf['node_id'].apply(lambda x: x + nodeCapacity[idx])
            dfArr.append(tmpdf)
        self.labels_data = pd.concat(dfArr, ignore_index=True)
        self.cutCell = pd.merge(self.cut_data, self.labels_data, how='inner', on=['node_id', 'phase', 'cell_name', 'cut_hashID'])
        # self.mean, self.std = self.cutCell['delay'].mean(), self.cutCell['delay'].std()
        # self.cutCell['delay'] = self.cutCell['delay'].apply(lambda x: (x - self.mean) / self.std if self.std != 0 else x)


    def __len__(self):
        return len(self.cutCell)

    def __getitem__(self, idx):
        cutCell_fea = self.cutCell.iloc[idx]
        # 1. get root node features; 1x28
        node_id = cutCell_fea['node_id']
        # node_fea = torch.tensor(node_fea, dtype=torch.float)
        other_fea = self.node_data.loc[node_id, ['inv', 'inv_p1', 'inv_p2', 're_level']]
        node_fea = self.node_data.loc[node_id]
        nF = self.fanoutEmb(torch.LongTensor([node_fea['fanout']]))
        nL = self.levelEmb(torch.LongTensor([node_fea['level']]))
        nFp1 = self.fanoutEmb(torch.LongTensor([node_fea['fanout_p1']]))
        nLp1 = self.levelEmb(torch.LongTensor([node_fea['level_p1']]))
        nFp2 = self.fanoutEmb(torch.LongTensor([node_fea['fanout_p2']]))
        nLp2 = self.levelEmb(torch.LongTensor([node_fea['level_p2']]))
        other_fea = torch.FloatTensor(other_fea.values).unsqueeze(0)
        node_fea = torch.cat([nF, nL, nFp1, nLp1, nFp2, nLp2, other_fea], dim=1)

        # 1. get leaves of the cut features: 6x28
        cut_ids = []
        for i in range(5):
            if (cutCell_fea['c' + str(i + 1)] != -1):
                cut_ids.append(cutCell_fea['c' + str(i + 1)])
        for i, node_id in enumerate(cut_ids):
            # tmp_node_fea = self.node_data.loc[node_id]
            other_fea = self.node_data.loc[node_id, ['inv', 'inv_p1', 'inv_p2', 're_level']]
            tmp_node_fea = self.node_data.loc[node_id]
            nF = self.fanoutEmb(torch.LongTensor([tmp_node_fea['fanout']]))
            nL = self.levelEmb(torch.LongTensor([tmp_node_fea['level']]))
            nFp1 = self.fanoutEmb(torch.LongTensor([tmp_node_fea['fanout_p1']]))
            nLp1 = self.levelEmb(torch.LongTensor([tmp_node_fea['level_p1']]))
            nFp2 = self.fanoutEmb(torch.LongTensor([tmp_node_fea['fanout_p2']]))
            nLp2 = self.levelEmb(torch.LongTensor([tmp_node_fea['level_p2']]))
            other_fea = torch.FloatTensor(other_fea.values).unsqueeze(0)
            tmp_node_fea = torch.cat([nF, nL, nFp1, nLp1, nFp2, nLp2, other_fea], dim=1)
            node_fea = torch.cat([node_fea, tmp_node_fea], dim=0)

        for i in range(len(cut_ids), 5):
            tmp_node_fea_zero = torch.zeros([1,node_fea.shape[1]], dtype=torch.float32)
            node_fea = torch.cat([node_fea, tmp_node_fea_zero], dim=0)

        # 3. get cut structure fea:
        cut_fea = torch.tensor(cutCell_fea[6:18], dtype =torch.float32).unsqueeze(0)
        nF = self.rootFanoutEmb(torch.LongTensor([cutCell_fea['root_fanout']]))
        cut_fea = torch.cat([cut_fea, nF], dim=1)
        # 4. get cell feature
        cell_name = cutCell_fea['cell_name']
        cell_fea = self.cell_data.loc[cell_name]
        cell_fea = pd.concat([cell_fea, cutCell_fea[20:23],  cutCell_fea[24:35]], axis=0)
        cell_fea = torch.tensor(cell_fea, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor([cutCell_fea['delay']],  dtype=torch.float32)

        return node_fea, cut_fea, cell_fea, label

# node_file =  '../data/train/adder_node_emb.csv'
# cut_file =  '../data/train/adder_cut_emb.csv'
# cell_file =  '../data/train/adder_cell_emb.csv'
# label_file  =  '../data/train/adder_lables_recovery.csv'
#
#
# cd = CutCellData(node_file,cut_file,cell_file,label_file)
# node_fea, cut_fea, cell_fea, label = cd.__getitem__(3)
# print(node_fea)
# print(cut_fea)
# print(cell_fea)
# print(label)
#