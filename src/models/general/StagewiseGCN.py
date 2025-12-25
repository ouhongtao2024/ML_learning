# # models/StagewiseGCN.py
# from typing import Dict, Any, Tuple, Optional, List
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from models.BaseModel import GeneralModel
# from helpers import StagewiseGCNReader
# import argparse

# class StagewiseGCN(GeneralModel):
#     """多阶段图卷积网络模型"""
    
#     reader = 'StagewiseGCNReader'
#     runner = 'StagewiseGCNRunner'
#     extra_log_args = ['emb_size', 'n_stages', 'n_layers', 'stage_fusion']
    
#     @staticmethod
#     def parse_model_args(parser:argparse.ArgumentParser) -> argparse.ArgumentParser:
#         # TODO: 需要调用GeneralModel.parse_model_args，添加阶段GCN特有参数
#         pass
    
#     def __init__(self, args, corpus: StagewiseGCNReader):
#         super().__init__(args, corpus)
#         # TODO: 初始化模型参数和预计算邻接矩阵
#         # 需要：从corpus获取预计算的邻接矩阵并转换为稀疏张量
#         # 需要：参考LightGCN的_convert_sp_mat_to_sp_tensor方法
#         pass
    
#     def _define_params(self) -> None:
#         """定义模型参数"""
#         # TODO: 定义用户/物品嵌入层（参考LightGCN的embedding_dict）
#         # TODO: 定义阶段特定的图卷积层（参考LGCNEncoder）
#         # TODO: 可选：阶段融合模块（attention/weighted/concat）
#         pass
    
#     def _prepare_adjacency_matrices(self) -> List[torch.Tensor]:
#         """准备邻接矩阵张量"""
#         # TODO: 将预计算的稀疏邻接矩阵转换为PyTorch稀疏张量
#         # 需要：corpus.get_stage_adjacency(stage)获取sp.csr_matrix
#         # 需要：LGCNEncoder._convert_sp_mat_to_sp_tensor转换
#         pass
    
#     def _stagewise_propagation(self, 
#                               embeddings: torch.Tensor,
#                               stage: int) -> torch.Tensor:
#         """执行指定阶段的图传播"""
#         # TODO: 实现LightGCN风格的图传播
#         # 需要：使用torch.sparse.mm进行稀疏矩阵乘法
#         # 需要：参考LGCNEncoder.forward中的传播逻辑
#         pass
    
#     def forward(self, feed_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
#         """前向传播"""
#         # TODO: 实现多阶段图卷积和特征融合
#         # 需要：根据self.stage 选择邻接矩阵
#         # 需要：多阶段传播 -> 阶段融合 -> 计算预测分数
#         pass
    
    
#     class Dataset(GeneralModel.Dataset):
#         """StagewiseGCN专用数据集类"""
        
#         def __init__(self, model: 'StagewiseGCN', corpus: StagewiseGCNReader, phase: str):
#             super().__init__(model, corpus, phase)
#             # TODO: 初始化阶段数据
#             pass
        
#         def _get_feed_dict(self, index: int) -> Dict[str, Any]:
#             # TODO: 扩展基类feed_dict，添加阶段信息
#             # 需要：参考GeneralModel.Dataset._get_feed_dict
#             pass


# models/StagewiseGCF.py
from typing import Dict, Any, Tuple, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.BaseModel import GeneralModel
from helpers import StagewiseGCNReader
import argparse
import scipy.sparse as sp
import numpy as np
import logging

class LightGCNBase(object):
	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--emb_size', type=int, default=64,
							help='Size of embedding vectors.')
		parser.add_argument('--n_layers', type=int, default=3,
							help='Number of LightGCN layers.')
		return parser
	
	@staticmethod
	def build_adjmat(user_count, item_count, train_mat, selfloop_flag=False):
		R = sp.dok_matrix((user_count, item_count), dtype=np.float32)
		for user in train_mat:
			for item in train_mat[user]:
				R[user, item] = 1
		R = R.tolil()

		adj_mat = sp.dok_matrix((user_count + item_count, user_count + item_count), dtype=np.float32)
		adj_mat = adj_mat.tolil()

		adj_mat[:user_count, user_count:] = R
		adj_mat[user_count:, :user_count] = R.T
		adj_mat = adj_mat.todok()

		def normalized_adj_single(adj):
			# D^-1/2 * A * D^-1/2
			rowsum = np.array(adj.sum(1)) + 1e-10

			d_inv_sqrt = np.power(rowsum, -0.5).flatten()
			d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
			d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

			bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
			return bi_lap.tocoo()

		if selfloop_flag:
			norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
		else:
			norm_adj_mat = normalized_adj_single(adj_mat)

		return norm_adj_mat.tocsr()

	def _base_init(self, args : argparse.Namespace, corpus ):
		self.emb_size = args.emb_size
		self.n_layers = args.n_layers
		self.norm_adj = self.build_adjmat(corpus.n_users, corpus.n_items, corpus.train_clicked_set)
		self._base_define_params()
		self.apply(self.init_weights)
	
	def _base_define_params(self):	
		self.encoder = LGCNEncoder(self.user_num, self.item_num, self.emb_size, self.norm_adj, self.n_layers)

	def forward(self, feed_dict : dict[str,any]):
		self.check_list = []
		user, items = feed_dict['user_id'], feed_dict['item_id']
		u_embed, i_embed = self.encoder(user, items)

		prediction = (u_embed[:, None, :] * i_embed).sum(dim=-1)  # [batch_size, -1]
		u_v = u_embed.repeat(1,items.shape[1]).view(items.shape[0],items.shape[1],-1)
		i_v = i_embed
		return {'prediction': prediction.view(feed_dict['batch_size'], -1), 'u_v': u_v, 'i_v':i_v}

class StagewiseBase(LightGCNBase):
    def _base_init(self, args: argparse.Namespace, corpus):
        """
        重载父类初始化，支持多阶段训练。
        """
        self.emb_size = args.emb_size
        self.n_layers = args.n_layers
        self.current_stage = 1  # 默认阶段1
        self.stage_norm_adj = {}  # 存储各阶段归一化矩阵
        self.corpus = corpus
        self.stages = args.n_stages
        self.current_norm_adj = None
        self.c = 0.1
        
        logging.info(f"[StagewiseBase] Initializing model with emb_size={self.emb_size}, n_layers={self.n_layers}")
        self._prepare_stage_norm_adjs()
        self._base_define_params()
        self.apply(self.init_weights)

    def _prepare_stage_norm_adjs(self):
        """
        动态生成多阶段归一化矩阵
        可以根据业务逻辑生成多个相似矩阵
        """
        logging.info("[StagewiseBase] Preparing stage-wise normalized adjacency matrices")

        # 示例：三阶段，每阶段可以加入不同的归一化策略
        for stage in range(self.stages):
            # 动态生成归一化矩阵，可以调整 selfloop_flag 或其他参数
            norm_adj = self.build_adjmat(
                self.corpus.n_users,
                self.corpus.n_items,
                self.corpus.train_clicked_set,
                d1=-0.5,
                d2=-0.5 + (self.stages - stage - 1) * self.c,
                selfloop_flag=True  # 可以根据 stage 动态调整
            )
            self.stage_norm_adj[stage] = norm_adj
            logging.info(f"[StagewiseBase] Stage {stage} normalized adjacency matrix shape: {norm_adj.shape}, nnz: {norm_adj.nnz}")

    def build_adjmat(self, user_count: int, item_count: int,
                     train_mat: dict, d1: float = -0.5, d2: float = -0.5, 
                     selfloop_flag: bool = True) -> sp.csr_matrix:
        """
        构建归一化邻接矩阵（适用于多阶段可控）
        
        :param user_count: 用户数量
        :param item_count: 物品数量
        :param train_mat: 用户->物品字典
        :param d1: 左侧归一化指数（可阶段化）
        :param d2: 右侧归一化指数（可阶段化）
        :param selfloop_flag: 是否加自环
        :return: 归一化后的 CSR 矩阵
        """
        # 构造原始评分矩阵 R
        R = sp.dok_matrix((user_count, item_count), dtype=np.float32)
        for u, items in train_mat.items():
            for i in items:
                R[u, i] = 1.
        R = R.tolil()

        # 构造完整邻接矩阵
        adj_mat = sp.dok_matrix((user_count + item_count, user_count + item_count), dtype=np.float32)
        adj_mat[:user_count, user_count:] = R
        adj_mat[user_count:, :user_count] = R.T
        adj_mat = adj_mat.tolil()

        if selfloop_flag:
            adj_mat = adj_mat + sp.eye(user_count + item_count)

        # 对称归一化
        adj_mat = adj_mat.tocoo()
        rowsum = np.array(adj_mat.sum(1)).flatten()
        d_inv_sqrt = np.power(rowsum, d1)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        d_inv_sqrt_last = np.power(rowsum, d2)
        d_inv_sqrt_last[np.isinf(d_inv_sqrt_last)] = 0.
        d_mat_inv_sqrt_last = sp.diags(d_inv_sqrt_last)

        norm_adj = adj_mat.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt_last).tocoo()
        return norm_adj.tocsr()

    def set_stage(self, stage: int):
        """
        切换训练阶段
        """
        if stage not in self.stage_norm_adj:
            raise ValueError(f"Invalid stage {stage}, available stages: {list(self.stage_norm_adj.keys())}")
        
        self.current_stage = stage
        self.encoder._stage_change(self.stage_norm_adj[stage])
        # 按照阶段修改当前的编码器所使用的归一化矩阵
        logging.info(f"[StagewiseBase] Switched to stage {stage}")

    def _base_define_params(self):
        """
        定义编码器，按当前阶段使用对应邻接矩阵
        """
        norm_adj = self.stage_norm_adj.get(self.current_stage)
        logging.info(f"[StagewiseBase] Defining encoder with stage {self.current_stage} norm_adj")
        self.encoder = LGCNEncoder(
            self.corpus.n_users,
            self.corpus.n_items,
            self.emb_size,
            norm_adj,
            self.n_layers
        )

    def forward(self, feed_dict: Dict[str, Any]):
        """
        重载 forward 支持多阶段训练，可记录每阶段 embedding
        """
        # logging.info(f"[StagewiseBase] Forward pass at stage {self.current_stage}")
        user, items = feed_dict['user_id'], feed_dict['item_id']
        u_embed, i_embed = self.encoder(user, items)

        prediction = (u_embed[:, None, :] * i_embed).sum(dim=-1)  # [batch_size, n_items]
        u_v = u_embed.repeat(1, items.shape[1]).view(items.shape[0], items.shape[1], -1)
        i_v = i_embed

        # logging.debug(f"[StagewiseBase] Prediction shape: {prediction.shape}, u_v shape: {u_v.shape}, i_v shape: {i_v.shape}")
        return {
            'prediction': prediction.view(feed_dict['batch_size'], -1),
            'u_v': u_v,
            'i_v': i_v
        }


class StagewiseGCN(StagewiseBase,GeneralModel):
    """多阶段图卷积网络模型"""
    
    reader = 'BaseReader'
    runner = 'BaseRunner'
    extra_log_args = ['n_stages']
    
    @staticmethod
    def parse_model_args(parser) -> argparse.ArgumentParser:
        parser = StagewiseBase.parse_model_args(parser)
        parser.add_argument('--n_stages', type=int, default=3,
                           help='阶段数量')
        
        print("Parsing StagewiseGCN model args")
        return GeneralModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        GeneralModel.__init__(self, args, corpus)
        self._base_init(args, corpus)

    def forward(self, feed_dict):
        out_dict = StagewiseBase.forward(self, feed_dict)
        return {'prediction': out_dict['prediction']}
            
class LGCNEncoder(nn.Module):
	def __init__(self, user_count, item_count, emb_size, norm_adj, n_layers = 3):
		super(LGCNEncoder, self).__init__()
		self.user_count = user_count
		self.item_count = item_count
		self.emb_size = emb_size
		self.layers = [emb_size] * n_layers
		self.norm_adj = norm_adj

		self.embedding_dict = self._init_model()
		self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj).cuda()

	def _init_model(self):
		initializer = nn.init.xavier_uniform_
		embedding_dict = nn.ParameterDict({
			'user_emb': nn.Parameter(initializer(torch.empty(self.user_count, self.emb_size))),
			'item_emb': nn.Parameter(initializer(torch.empty(self.item_count, self.emb_size))),
		})
		return embedding_dict
    
	def _stage_change(self,norm_adj):
		self.norm_adj = norm_adj
		self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj).cuda()

	@staticmethod
	def _convert_sp_mat_to_sp_tensor(X):
		coo = X.tocoo()
		i = torch.LongTensor([coo.row, coo.col])
		v = torch.from_numpy(coo.data).float()
		return torch.sparse.FloatTensor(i, v, coo.shape)

	def forward(self, users, items):
		ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
		all_embeddings = [ego_embeddings]

		for k in range(len(self.layers)):
			ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
			all_embeddings += [ego_embeddings]

		all_embeddings = torch.stack(all_embeddings, dim=1)
		all_embeddings = torch.mean(all_embeddings, dim=1)

		user_all_embeddings = all_embeddings[:self.user_count, :]
		item_all_embeddings = all_embeddings[self.user_count:, :]

		user_embeddings = user_all_embeddings[users, :]
		item_embeddings = item_all_embeddings[items, :]

		return user_embeddings, item_embeddings