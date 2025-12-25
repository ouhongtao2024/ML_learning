# # helpers/StagewiseGCNReader.py
# from typing import Dict, List, Tuple, Any, Optional
# import numpy as np
# import scipy.sparse as sp
# import pandas as pd
# from helpers.BaseReader import BaseReader
# import argparse

# class StagewiseGCNReader(BaseReader):
#     """支持多阶段图卷积网络的专用数据读取器"""
    
#     @staticmethod
#     def parse_data_args(parser) -> argparse.ArgumentParser:
#         # TODO: 需要调用BaseReader.parse_data_args，添加阶段相关参数
#         pass
    
#     def __init__(self, args):
#         super().__init__(args)
#         # TODO: 初始化阶段参数，加载预计算的邻接矩阵
#         # 需要：args.stage_ratios, args.alpha_list, args.adj_type
#         # 需要：邻接矩阵存储路径 (path_cold)
#         pass
    
#     def _prepare_stagewise_data(self) -> None:
#         """准备分阶段数据"""
#         # TODO: 实现阶段数据划分和邻接矩阵预计算
#         # 需要：load_data.py中的create_adj_mat方法
#         # 需要：存储adj_45, adj_54, adj_55到self.stage_adj_matrices
#         pass
    
#     def get_stage_adjacency(self, stage: int) -> sp.csr_matrix:
#         """获取指定阶段的归一化邻接矩阵"""
#         # TODO: 返回对应阶段的预计算邻接矩阵
#         # 需要：根据stage参数返回adj_53, adj_54或adj_55
#         pass
    
#     def get_stage_data(self, stage: int, phase: str = 'train') -> pd.DataFrame:
#         """获取指定阶段和阶段的数据"""
#         # TODO: 按阶段划分数据集（基于时间戳或交互比例）
#         # 需要：corpus.data_df中的时间信息
#         pass


# helpers/StagewiseGCNReader.py
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import scipy.sparse as sp
import pandas as pd
from helpers.BaseReader import BaseReader
import argparse
import os
import pickle
import logging

class StagewiseGCNReader(BaseReader):
    """支持多阶段图卷积网络的专用数据读取器"""
    
    @staticmethod
    def parse_data_args(parser) -> argparse.ArgumentParser:
        parser = BaseReader.parse_data_args(parser)
        parser.add_argument('--stage_ratios', type=str, default='0.5,0.75,1.0',
                           help='各阶段数据划分比例（累计值）')
        parser.add_argument('--adj_type', type=str, default='symmetric',
                           help='邻接矩阵类型：symmetric, row_norm, column_norm')
        parser.add_argument('--alpha_list', type=str, default='0.5,0.4,0.5',
                           help='各阶段的alpha参数（列幂次）')
        parser.add_argument('--beta_list', type=str, default='0.5,0.3,0.5',
                           help='各阶段的beta参数（行幂次）')
        return parser
    
    def __init__(self, args):
        super().__init__(args)
        # 解析阶段参数
        self.stage_ratios = [float(x) for x in args.stage_ratios.split(',')]
        self.alpha_list = [float(x) for x in args.alpha_list.split(',')]
        self.beta_list = [float(x) for x in args.beta_list.split(',')]
        self.adj_type = args.adj_type
        self.n_stages = len(self.stage_ratios)
        
        # 确保路径存在
        self.stage_dir = os.path.join(self.prefix, self.dataset, 'stagewise_adj')
        os.makedirs(self.stage_dir, exist_ok=True)
        
        # 准备阶段数据
        self._prepare_stagewise_data()
        self._prepare_stage_adjacency()
    
    def _prepare_stagewise_data(self) -> None:
        """准备分阶段数据"""
        self.stage_data = {'train': [], 'dev': [], 'test': []}
        
        # 按时间戳划分阶段（如果数据有时间信息）
        all_data = self.all_df.sort_values('time')
        total_interactions = len(all_data)
        
        for stage, ratio in enumerate(self.stage_ratios):
            cutoff = int(total_interactions * ratio)
            stage_df = all_data.iloc[:cutoff]
            
            # 划分训练/验证/测试集（保持时间顺序）
            train_size = int(len(stage_df) * 0.7)
            dev_size = int(len(stage_df) * 0.85)
            
            train_df = stage_df.iloc[:train_size]
            dev_df = stage_df.iloc[train_size:dev_size]
            test_df = stage_df.iloc[dev_size:]
            
            # 转换为dict格式
            self.stage_data['train'].append(self._df_to_dict(train_df))
            self.stage_data['dev'].append(self._df_to_dict(dev_df))
            self.stage_data['test'].append(self._df_to_dict(test_df))
    
    def _df_to_dict(self, df: pd.DataFrame) -> Dict[str, list]:
        """DataFrame转换为字典格式"""
        data_dict = {}
        for col in df.columns:
            data_dict[col] = df[col].tolist()
        return data_dict
    
    def _prepare_stage_adjacency(self) -> None:
        """准备各阶段的归一化邻接矩阵"""
        self.stage_adj_matrices = []
        cache_file = os.path.join(self.stage_dir, f'stage_adj_{self.adj_type}.pkl')
        
        if os.path.exists(cache_file):
            logging.info(f'从缓存加载邻接矩阵: {cache_file}')
            with open(cache_file, 'rb') as f:
                self.stage_adj_matrices = pickle.load(f)
        else:
            logging.info(f'预计算各阶段邻接矩阵...')
            
            # 构建基础二部图
            R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
            for u in self.train_clicked_set:
                for i in self.train_clicked_set[u]:
                    R[u, i] = 1.0
            
            # 构建增广邻接矩阵
            adj_mat = sp.dok_matrix((self.n_users + self.n_items, 
                                    self.n_users + self.n_items), dtype=np.float32)
            adj_mat = adj_mat.tolil()
            adj_mat[:self.n_users, self.n_users:] = R
            adj_mat[self.n_users:, :self.n_users] = R.T
            adj_mat = adj_mat.todok()
            
            # 为每个阶段创建不同的归一化矩阵
            for stage in range(self.n_stages):
                alpha = self.alpha_list[stage]
                beta = self.beta_list[stage]
                norm_adj = self._normalized_adj_symetric(adj_mat, alpha, beta)
                self.stage_adj_matrices.append(norm_adj.tocsr())
            
            # 缓存邻接矩阵
            with open(cache_file, 'wb') as f:
                pickle.dump(self.stage_adj_matrices, f)
            logging.info(f'邻接矩阵已保存到: {cache_file}')
    
    def _normalized_adj_symetric(self, adj: sp.dok_matrix, 
                                alpha: float, beta: float) -> sp.coo_matrix:
        """对称归一化邻接矩阵：D_row^alpha * A * D_column^beta"""
        adj = sp.coo_matrix(adj + sp.eye(adj.shape[0]))  # 添加自连接
        
        rowsum = np.array(adj.sum(axis=1)).flatten()
        colsum = np.array(adj.sum(axis=0)).flatten()
        
        # 计算行和列的幂次
        d_row_inv_sqrt = np.power(rowsum, alpha)
        d_row_inv_sqrt[np.isinf(d_row_inv_sqrt)] = 0.
        d_row_mat_inv_sqrt = sp.diags(d_row_inv_sqrt)
        
        d_col_inv_sqrt = np.power(colsum, beta)
        d_col_inv_sqrt[np.isinf(d_col_inv_sqrt)] = 0.
        d_col_mat_inv_sqrt = sp.diags(d_col_inv_sqrt)
        
        # 对称归一化
        return d_row_mat_inv_sqrt.dot(adj).dot(d_col_mat_inv_sqrt).tocoo()
    
    def get_stage_adjacency(self, stage: int) -> sp.csr_matrix:
        """获取指定阶段的归一化邻接矩阵"""
        if stage < 0 or stage >= len(self.stage_adj_matrices):
            raise ValueError(f"阶段索引超出范围: {stage}，最大阶段数: {len(self.stage_adj_matrices)}")
        return self.stage_adj_matrices[stage]
    
    def get_stage_data(self, stage: int, phase: str = 'train') -> Dict[str, list]:
        """获取指定阶段和阶段的数据"""
        if stage < 0 or stage >= self.n_stages:
            raise ValueError(f"阶段索引超出范围: {stage}，最大阶段数: {self.n_stages}")
        if phase not in ['train', 'dev', 'test']:
            raise ValueError(f"阶段类型错误: {phase}，必须是train/dev/test")
        
        return self.stage_data[phase][stage]