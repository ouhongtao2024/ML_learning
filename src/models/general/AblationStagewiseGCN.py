# models/StagewiseGCF.py
# from helpers import StagewiseGCNReader
import argparse
import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.BaseModel import GeneralModel


class _LightGCNBase(object):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument(
            "--emb_size", type=int, default=64, help="Size of embedding vectors."
        )
        parser.add_argument(
            "--n_layers", type=int, default=3, help="Number of LightGCN layers."
        )
        parser.add_argument(
            "--odd_layer", type=int, default=3, help="selected odd layer."
        )
        parser.add_argument(
            "--even_layer", type=int, default=2, help="selected even layer."
        )
        parser.add_argument("--c", type=float, default=0.1, help="c value")
        parser.add_argument(
            "--enabled_stage", type=int, default=1, help="enabled stage"
        )

        return parser

    def _base_init(self, args: argparse.Namespace, corpus):
        self.emb_size = args.emb_size
        self.n_layers = args.n_layers
        self.norm_adj = self.build_adjmat(
            corpus.n_users, corpus.n_items, corpus.train_clicked_set
        )
        self._base_define_params()
        self.apply(self.init_weights)

    def _base_define_params(self):
        self.encoder = _LGCNEncoder(
            self.user_num,
            self.item_num,
            self.emb_size,
            self.norm_adj,
            self.n_layers,
            self,
        )

    def forward(self, feed_dict: dict[str, any]):
        self.check_list = []
        user, items = feed_dict["user_id"], feed_dict["item_id"]
        u_embed, i_embed = self.encoder(user, items)
        prediction = (u_embed[:, None, :] * i_embed).sum(dim=-1)  # [batch_size, -1]
        u_v = u_embed.repeat(1, items.shape[1]).view(items.shape[0], items.shape[1], -1)
        i_v = i_embed
        return {
            "prediction": prediction.view(feed_dict["batch_size"], -1),
            "u_v": u_v,
            "i_v": i_v,
        }


class AblationStagewiseBase(_LightGCNBase):
    def _base_init(self, args: argparse.Namespace, corpus):
        """
        重载父类初始化，支持多阶段训练。
        """
        self.emb_size = args.emb_size
        self.n_layers = args.n_layers
        self.current_stage = 1  # 默认阶段1
        self.stage_norm_adj: dict[int, sp.csr_matrix] = {}  # 存储各阶段归一化矩阵
        self.corpus = corpus
        self.stages = args.n_stages
        self.current_norm_adj = None
        self.c = args.c
        self.odd_layer = args.odd_layer
        self.even_layer = args.even_layer
        self.l2 = args.l2
        self.enabled_stage = args.enabled_stage

        logging.info(
            f"[StagewiseBase] Initializing model with emb_size={self.emb_size}, n_layers={self.n_layers}"
        )
        self._prepare_stage_norm_adjs()
        self._base_define_params()
        self.apply(self.init_weights)

    def _prepare_stage_norm_adjs(self):
        """
        动态生成多阶段归一化矩阵
        可以根据业务逻辑生成多个相似矩阵
        """
        logging.info(
            "[StagewiseBase] Preparing stage-wise normalized adjacency matrices"
        )
        print(self.corpus)

        for stage in range(1, self.stages + 1):

            norm_adj = self.build_adjmat(
                self.corpus.n_users,
                self.corpus.n_items,
                self.corpus.train_clicked_set,
                d1=-0.5,
                d2=-0.5 + (self.stages - stage) * self.c,
                selfloop_flag=True,
            )
            self.stage_norm_adj[stage] = norm_adj
            logging.info(
                f"[StagewiseBase] Stage {stage} normalized adjacency matrix shape: {norm_adj.shape}, nnz: {norm_adj.nnz}"
            )

    @staticmethod
    def build_adjmat(
        user_count: int,
        item_count: int,
        train_mat: Dict[int, list],
        d1: float = -0.5,
        d2: float = -0.5,
        selfloop_flag: bool = True,
    ) -> sp.csr_matrix:
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
        adj = sp.coo_matrix(
            (data, (row, col)), shape=(n_nodes, n_nodes), dtype=np.float32
        )

        # 对称归一化
        rowsum = np.array(adj.sum(axis=1)).flatten()
        d_inv_sqrt = np.power(rowsum, d1)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        D1 = sp.diags(d_inv_sqrt)

        d_inv_sqrt_last = np.power(rowsum, d2)
        d_inv_sqrt_last[np.isinf(d_inv_sqrt_last)] = 0.0
        D2 = sp.diags(d_inv_sqrt_last)

        norm_adj = D1.dot(adj).dot(D2).tocsr()
        return norm_adj

    def set_stage(self, stage: int):
        """
        切换训练阶段
        """
        if stage not in self.stage_norm_adj:
            raise ValueError(
                f"Invalid stage {stage}, available stages: {list(self.stage_norm_adj.keys())}"
            )
        self.current_stage = stage
        # 按照阶段修改当前的编码器所使用的归一化矩阵
        logging.info(f"[StagewiseBase] Switched to stage {stage}")

    def _base_define_params(self):
        """
        定义编码器，传入所有阶段的邻接矩阵
        """
        logging.info(
            f"[StagewiseBase] Defining encoder with stage {self.current_stage} norm_adj"
        )
        self.encoder = _LGCNEncoder(
            self.corpus.n_users,
            self.corpus.n_items,
            self.emb_size,
            self.stage_norm_adj,
            self.n_layers,
        )

    def forward(self, feed_dict):
        return self._forward_stagewise(feed_dict, self.current_stage)

    def _forward_stagewise(
        self, feed_dict: Dict[str, Any], num_stages: int
    ) -> Dict[str, Any]:
        """
        通用 stage-wise forward
        支持任意阶段数的训练与 embedding 记录
        """
        user, items = feed_dict["user_id"], feed_dict["item_id"]

        odd_user_embs: List[torch.Tensor] = []
        odd_item_embs: List[torch.Tensor] = []
        even_user_embs: List[torch.Tensor] = []
        even_item_embs: List[torch.Tensor] = []
        odd_score_list: List[torch.Tensor] = []
        even_score_list: List[torch.Tensor] = []

        for stage in range(1, num_stages + 1):
            odd_u, odd_i = self.encoder(user, items, self.odd_layer, stage)
            even_u, even_i = self.encoder(user, items, self.even_layer, stage)

            odd_user_embs.append(odd_u)
            odd_item_embs.append(odd_i)
            even_user_embs.append(even_u)
            even_item_embs.append(even_i)

            stage_odd_score = (odd_u[:, None, :] * odd_i).sum(dim=-1)
            stage_even_score = (even_u[:, None, :] * even_i).sum(dim=-1)

            if odd_score_list is None:
                odd_score_list = [stage_odd_score]
                even_score_list = [stage_even_score]
            else:
                odd_score_list.append(stage_odd_score)
                even_score_list.append(stage_even_score)

        prediction = sum(odd_score_list) + sum(even_score_list)

        return {
            "prediction": prediction.view(feed_dict["batch_size"], -1),
            "odd_score_list": odd_score_list,
            "even_score_list": even_score_list,
            "odd_user_emb_list": odd_user_embs,
            "odd_item_emb_list": odd_item_embs,
            "even_user_emb_list": even_user_embs,
            "even_item_emb_list": even_item_embs,
        }


class AblationStagewiseGCN(AblationStagewiseBase, GeneralModel):
    """多阶段图卷积网络模型"""

    reader = "BaseReader"
    runner = "AblationStagewiseRunner"
    extra_log_args = ["n_stages", "c", "odd_layer", "even_layer", "enabled_stage"]

    @staticmethod
    def parse_model_args(parser) -> argparse.ArgumentParser:
        parser = AblationStagewiseBase.parse_model_args(parser)
        parser.add_argument("--n_stages", type=int, default=3, help="阶段数量")

        print("Parsing StagewiseGCN model args")
        return GeneralModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        GeneralModel.__init__(self, args, corpus)
        self._base_init(args, corpus)

    def forward(self, feed_dict):
        # 按道理来说，前向传播应该仅仅回传预测结果，但是这里为了适配架构和后面的损失函数计算，因此整个字典返回了
        return AblationStagewiseBase.forward(self, feed_dict)

    def loss(self, out_dict: Dict) -> torch.Tensor:
        """
        Stage-wise loss:
        按颗粒度累加分离开的偶数层和奇数层的误差，并加上每个阶段对应的正则化项
        - 每个 stage 单独计算 odd / even 排序损失
        - 最终 loss 为所有 stage 的聚合
        """

        odd_score_list = out_dict["odd_score_list"]  # List[[B, 1 + neg]]
        even_score_list = out_dict["even_score_list"]

        stage_losses: torch.Tensor = 0

        for odd_score, even_score in zip(odd_score_list, even_score_list):
            # odd
            pos_odd = odd_score[:, 0]
            neg_odd = odd_score[:, 1:]
            odd_loss = F.softplus(neg_odd - pos_odd).mean()

            # even
            pos_even = even_score[:, 0]
            neg_even = even_score[:, 1:]
            even_loss = F.softplus(neg_even - pos_even).mean()

            stage_losses += odd_loss + even_loss

        # 默认：所有 stage 等权

        # -------- Stage-wise L2 regularization --------
        reg_loss = 0.0
        # reg_cnt = 0

        # for key in [
        #     'odd_user_emb_list',
        #     'odd_item_emb_list',
        #     'even_user_emb_list',
        #     'even_item_emb_list'
        # ]:
        #     for emb in out_dict[key]:
        #         reg_loss += emb.pow(2).sum(dim=1).mean()
        #         reg_cnt += 1

        # if reg_cnt > 0:
        #     reg_loss = reg_loss / reg_cnt

        return stage_losses + self.l2 * reg_loss


class _LGCNEncoder(nn.Module):
    def __init__(
        self,
        user_count,
        item_count,
        emb_size,
        stage_orm_adjs: Dict[int, sp.csr_matrix],
        n_layers=3,
        n_stages=3,
    ):
        super(_LGCNEncoder, self).__init__()
        self.user_count = user_count
        self.item_count = item_count
        self.emb_size = emb_size
        self.layers = [emb_size] * n_layers
        self.stage_norm_adj = stage_orm_adjs
        self.n_stages = n_stages

        self.embedding_dict = self._init_model()
        self.sparse_norm_adjs: Dict[int, torch.sparse.FloatTensor] = {}
        for stage in range(1, self.n_stages + 1):
            self.sparse_norm_adjs[stage] = self._convert_sp_mat_to_sp_tensor(
                self.stage_norm_adj[stage]
            ).cuda()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict(
            {
                "user_emb": nn.Parameter(
                    initializer(torch.empty(self.user_count, self.emb_size))
                ),
                "item_emb": nn.Parameter(
                    initializer(torch.empty(self.item_count, self.emb_size))
                ),
            }
        )
        return embedding_dict

    @staticmethod
    def _convert_sp_mat_to_sp_tensor(
        X: sp.csr_matrix, device: torch.device = torch.device("cuda")
    ) -> torch.sparse.FloatTensor:
        """
        内存友好型 CSR/COO -> Torch Sparse Tensor
        """
        if not sp.isspmatrix_coo(X):
            X = X.tocoo()

        # 直接用 numpy 堆叠，避免 Python list
        indices = np.vstack((X.row, X.col)).astype(np.int64)  # shape [2, nnz]
        values = X.data.astype(np.float32)  # shape [nnz]

        # 转为 Torch Tensor
        i = torch.from_numpy(indices)
        v = torch.from_numpy(values)
        return torch.sparse_coo_tensor(i, v, X.shape, device=device)

    def forward(
        self, users, items, n_layers: int = 3, stage: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        按照层数和阶段数进行高阶连接信息整合，并返回对应的用户和商品的编码向量
        """
        ego_embeddings = torch.cat(
            [self.embedding_dict["user_emb"], self.embedding_dict["item_emb"]], dim=0
        )

        for _ in range(n_layers):
            ego_embeddings = torch.sparse.mm(
                self.sparse_norm_adjs[stage], ego_embeddings
            )

        user_all_embeddings = ego_embeddings[: self.user_count, :]
        item_all_embeddings = ego_embeddings[self.user_count :, :]

        user_embeddings = user_all_embeddings[users, :]
        item_embeddings = item_all_embeddings[items, :]

        return user_embeddings, item_embeddings
