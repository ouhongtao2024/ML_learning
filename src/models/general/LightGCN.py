# -*- coding: UTF-8 -*-
# @Author  : Chenyang Wang
# @Email   : THUwangcy@gmail.com

import torch
import numpy as np
import torch.nn as nn
import scipy.sparse as sp
import argparse
import helpers.BaseReader
from typing import Dict
from models.BaseModel import GeneralModel
from models.BaseImpressionModel import ImpressionModel

class LightGCNBase(object):
	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--emb_size', type=int, default=64,
							help='Size of embedding vectors.')
		parser.add_argument('--n_layers', type=int, default=3,
							help='Number of LightGCN layers.')
		return parser
	
	@staticmethod
	def build_adjmat(user_count: int, item_count: int,
					train_mat: Dict[int, list],
					d1: float = -0.5, d2: float = -0.5,
					selfloop_flag: bool = True) -> sp.csr_matrix:
		"""
		内存友好型邻接矩阵构建

		:param user_count: 用户数量
		:param item_count: 物品数量
		:param train_mat: {user: [item,...]}
		:param d1: 左侧归一化指数
		:param d2: 右侧归一化指数
		:param selfloop_flag: 是否加自环
		:return: 对称归一化 csr_matrix
		"""
		n_nodes = user_count + item_count

		# 收集三元组
		row = []
		col = []
		data = []

		for u, items in train_mat.items():
			row.extend([u] * len(items))
			col.extend([user_count + i for i in items])
			data.extend([1.0] * len(items))

		# 双向边
		row_b = col.copy()
		col_b = row.copy()
		data_b = data.copy()

		row.extend(row_b)
		col.extend(col_b)
		data.extend(data_b)

		if selfloop_flag:
			row.extend(range(n_nodes))
			col.extend(range(n_nodes))
			data.extend([1.0] * n_nodes)

		# 转 numpy
		row = np.array(row, dtype=np.int32)
		col = np.array(col, dtype=np.int32)
		data = np.array(data, dtype=np.float32)

		# 构建 COO
		adj = sp.coo_matrix((data, (row, col)), shape=(n_nodes, n_nodes), dtype=np.float32)

		# 对称归一化
		rowsum = np.array(adj.sum(axis=1)).flatten()
		d_inv_sqrt = np.power(rowsum, d1)
		d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
		D1 = sp.diags(d_inv_sqrt)

		d_inv_sqrt_last = np.power(rowsum, d2)
		d_inv_sqrt_last[np.isinf(d_inv_sqrt_last)] = 0.
		D2 = sp.diags(d_inv_sqrt_last)

		norm_adj = D1.dot(adj).dot(D2).tocsr()
		return norm_adj

	def _base_init(self, args : argparse.Namespace, corpus : helpers.BaseReader.BaseReader):
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

class LightGCN(GeneralModel, LightGCNBase):
	reader = 'BaseReader'
	runner = 'BaseRunner'
	extra_log_args = ['emb_size', 'n_layers', 'batch_size']

	@staticmethod
	def parse_model_args(parser):
		parser = LightGCNBase.parse_model_args(parser)
		return GeneralModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		GeneralModel.__init__(self, args, corpus)
		self._base_init(args, corpus)

	def forward(self, feed_dict):
		out_dict = LightGCNBase.forward(self, feed_dict)
		return {'prediction': out_dict['prediction']}
	
class LightGCNImpression(ImpressionModel, LightGCNBase):
	reader = 'ImpressionReader'
	runner = 'ImpressionRunner'
	extra_log_args = ['emb_size', 'n_layers', 'batch_size']

	@staticmethod
	def parse_model_args(parser):

		parser = LightGCNBase.parse_model_args(parser)
		return ImpressionModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		ImpressionModel.__init__(self, args, corpus)
		self._base_init(args, corpus)

	def forward(self, feed_dict):
		return LightGCNBase.forward(self, feed_dict)

class LGCNEncoder(nn.Module):
	def __init__(self, user_count, item_count, emb_size, norm_adj, n_layers=3):
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
	
	@staticmethod
	def _convert_sp_mat_to_sp_tensor(X: sp.csr_matrix, device: torch.device = torch.device('cuda')) -> torch.sparse.FloatTensor:
		"""
		内存友好型 CSR/COO -> Torch Sparse Tensor
		"""
		if not sp.isspmatrix_coo(X):
			X = X.tocoo()

		# 直接用 numpy 堆叠，避免 Python list
		indices = np.vstack((X.row, X.col)).astype(np.int64)  # shape [2, nnz]
		values = X.data.astype(np.float32)                     # shape [nnz]

		# 转为 Torch Tensor
		i = torch.from_numpy(indices)
		v = torch.from_numpy(values)
		return torch.sparse_coo_tensor(i, v, X.shape, device=device)

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
