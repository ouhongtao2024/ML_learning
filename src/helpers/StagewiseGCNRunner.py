# # helpers/StagewiseGCNRunner.py
# from typing import Dict, List, Optional, Tuple
# import os
# import logging
# import numpy as np
# import torch
# from helpers.BaseRunner import BaseRunner
# from models.general.StagewiseGCF import StagewiseGCN
# from helpers.StagewiseGCNReader import StagewiseGCNReader
# import argparse

# class StagewiseGCNRunner(BaseRunner):
#     """多阶段GCN训练器"""
    
#     @staticmethod
#     def parse_runner_args(parser) -> argparse.ArgumentParser:
#         # TODO: 需要调用BaseRunner.parse_runner_args，添加阶段训练参数
#         pass
    
#     def __init__(self, args):
#         super().__init__(args)
#         # TODO: 初始化阶段训练参数
#         # 需要：args.stage_epochs, args.stage_lrs, args.stage_transfer等
#         pass
    
#     def train(self, data_dict: Dict[str, StagewiseGCN.Dataset]) -> None:
#         """多阶段训练主循环"""
#         # TODO: 实现分阶段训练（参考JMP-GCF.py中的训练循环）
#         # 需要：阶段1: adj_53, 阶段2: adj_54, 阶段3: adj_55
#         # 需要：每个阶段独立的epoch数和学习率
#         pass
    
#     def _train_single_stage(self, 
#                            stage_idx: int,
#                            max_epochs: int,
#                            data_dict: Dict[str, StagewiseGCN.Dataset]) -> Dict[str, List]:
#         """训练单个阶段"""
#         # TODO: 单个阶段的训练循环
#         # 需要：设置模型当前阶段(model.current_stage = stage_idx)
#         # 需要：调整学习率，执行fit和evaluate
#         pass
    
#     def _before_stage_training(self, 
#                               stage_idx: int,
#                               model: StagewiseGCN,
#                               data_dict: Dict[str, StagewiseGCN.Dataset]) -> None:
#         """阶段训练前的准备工作"""
#         # TODO: 阶段切换的准备工作
#         # 需要：参数迁移（如果stage_transfer启用）
#         # 需要：优化器重置和学习率调整
#         pass
    
#     def _after_stage_training(self,
#                              stage_idx: int,
#                              model: StagewiseGCN,
#                              data_dict: Dict[str, StagewiseGCN.Dataset],
#                              stage_metrics: Dict[str, List]) -> None:
#         """阶段训练后的处理"""
#         # TODO: 阶段训练后的清理和分析
#         # 需要：保存阶段最佳模型，更新性能历史
#         pass
    
#     def _adjust_stage_learning_rate(self,
#                                    stage_idx: int,
#                                    model: StagewiseGCN) -> None:
#         """调整阶段学习率"""
#         # TODO: 根据stage_idx设置对应的学习率
#         pass

# helpers/StagewiseGCNRunner.py
from typing import Dict, List, Optional, Tuple
import os
import logging
import numpy as np
import torch
from helpers.BaseRunner import BaseRunner
from models.general.StagewiseGCN import StagewiseGCN
from helpers.StagewiseGCNReader import StagewiseGCNReader
import argparse
from tqdm import tqdm

class StagewiseGCNRunner(BaseRunner):
    """多阶段GCN训练器"""
    
    @staticmethod
    def parse_runner_args(parser) -> argparse.ArgumentParser:
        parser = BaseRunner.parse_runner_args(parser)
        parser.add_argument('--stage_epochs', type=str, default='300,300,400',
                           help='各阶段训练轮数')
        parser.add_argument('--stage_lrs', type=str, default='0.001,0.0005,0.0001',
                           help='各阶段学习率')
        parser.add_argument('--stage_transfer', type=int, default=1,
                           help='是否进行阶段间参数迁移')
        parser.add_argument('--stage_eval_freq', type=int, default=5,
                           help='阶段内评估频率')
        return parser
    
    def __init__(self, args):
        super().__init__(args)
        # 解析阶段训练参数
        self.stage_epochs = [int(x) for x in args.stage_epochs.split(',')]
        self.stage_lrs = [float(x) for x in args.stage_lrs.split(',')]
        self.stage_transfer = args.stage_transfer
        self.stage_eval_freq = args.stage_eval_freq
        
        if len(self.stage_epochs) != len(self.stage_lrs):
            raise ValueError("stage_epochs和stage_lrs长度必须相同")
        self.n_stages = len(self.stage_epochs)
    
    def train(self, data_dict: Dict[str, StagewiseGCN.Dataset]) -> None:
        """多阶段训练主循环"""
        model = data_dict['train'].model
        all_stage_metrics = []
        
        for stage_idx in range(self.n_stages):
            logging.info(f"=== 开始第 {stage_idx + 1}/{self.n_stages} 阶段训练 ===")
            
            # 阶段训练前的准备
            self._before_stage_training(stage_idx, model, data_dict)
            
            # 单个阶段训练
            stage_metrics = self._train_single_stage(
                stage_idx, self.stage_epochs[stage_idx], data_dict)
            all_stage_metrics.append(stage_metrics)
            
            # 阶段训练后的处理
            self._after_stage_training(stage_idx, model, data_dict, stage_metrics)
        
        # 最终评估
        logging.info(f"=== 多阶段训练完成 ===")
        self._summarize_stage_metrics(all_stage_metrics)
    
    def _train_single_stage(self, 
                           stage_idx: int,
                           max_epochs: int,
                           data_dict: Dict[str, StagewiseGCN.Dataset]) -> Dict[str, List]:
        """训练单个阶段"""
        model = data_dict['train'].model
        model.stage = stage_idx  # 设置当前阶段
        
        stage_metrics = {
            'train_loss': [],
            'dev_main_metric': [],
            'dev_results': []
        }
        
        for epoch in range(max_epochs):
            # 训练
            loss = self.fit(data_dict['train'], epoch=epoch + 1)
            stage_metrics['train_loss'].append(loss)
            
            # 定期评估
            if (epoch + 1) % self.stage_eval_freq == 0:
                dev_result = self.evaluate(data_dict['dev'], [self.main_topk], self.metrics)
                main_metric_value = dev_result[self.main_metric]
                stage_metrics['dev_main_metric'].append(main_metric_value)
                stage_metrics['dev_results'].append(dev_result)
                
                logging.info(f'阶段 {stage_idx + 1}, 轮次 {epoch + 1}/{max_epochs}, '
                            f'损失: {loss:.4f}, {self.main_metric}: {main_metric_value:.4f}')
                
                # 提前停止检查
                if self._check_stage_early_stop(stage_metrics['dev_main_metric']):
                    logging.info(f"阶段 {stage_idx + 1} 提前停止于轮次 {epoch + 1}")
                    break
        
        return stage_metrics
    
    def _check_stage_early_stop(self, metric_history: List[float]) -> bool:
        """检查阶段内提前停止条件"""
        if self.early_stop <= 0 or len(metric_history) < self.early_stop * 2:
            return False
        
        # 检查最近early_stop次结果是否持续下降
        recent_metrics = metric_history[-self.early_stop:]
        return all(recent_metrics[i] >= recent_metrics[i + 1] 
                  for i in range(len(recent_metrics) - 1))
    
    def _before_stage_training(self, 
                              stage_idx: int,
                              model: StagewiseGCN,
                              data_dict: Dict[str, StagewiseGCN.Dataset]) -> None:
        """阶段训练前的准备工作"""
        logging.info(f"阶段 {stage_idx + 1} 准备中...")
        
        # 设置模型阶段
        model.current_stage = stage_idx
        
        # 调整学习率
        self._adjust_stage_learning_rate(stage_idx, model)
        
        # 阶段间参数迁移
        if self.stage_transfer and stage_idx > 0:
            logging.info(f"从阶段 {stage_idx} 迁移参数到阶段 {stage_idx + 1}")
            model.transfer_stage_parameters(stage_idx - 1, stage_idx)
        
        # 重置优化器（使用新阶段的学习率）
        model.optimizer = self._build_optimizer(model)
        
        # 清除GPU缓存
        torch.cuda.empty_cache()
    
    def _after_stage_training(self,
                             stage_idx: int,
                             model: StagewiseGCN,
                             data_dict: Dict[str, StagewiseGCN.Dataset],
                             stage_metrics: Dict[str, List]) -> None:
        """阶段训练后的处理"""
        # 保存阶段最佳模型
        stage_model_path = model.model_path.replace('.pt', f'_stage{stage_idx}.pt')
        model.save_model(stage_model_path)
        logging.info(f"阶段 {stage_idx + 1} 模型已保存: {stage_model_path}")
        
        # 分析阶段性能
        if stage_metrics['dev_main_metric']:
            best_metric = max(stage_metrics['dev_main_metric'])
            best_epoch = stage_metrics['dev_main_metric'].index(best_metric) * self.stage_eval_freq
            logging.info(f"阶段 {stage_idx + 1} 最佳 {self.main_metric}: {best_metric:.4f} (轮次 {best_epoch})")
    
    def _adjust_stage_learning_rate(self,
                                   stage_idx: int,
                                   model: StagewiseGCN) -> None:
        """调整阶段学习率"""
        if stage_idx < len(self.stage_lrs):
            self.learning_rate = self.stage_lrs[stage_idx]
            logging.info(f"阶段 {stage_idx + 1} 学习率设置为: {self.learning_rate}")
    
    def _summarize_stage_metrics(self, all_stage_metrics: List[Dict]) -> None:
        """总结各阶段性能指标"""
        logging.info("=== 各阶段训练总结 ===")
        for stage_idx, metrics in enumerate(all_stage_metrics):
            if metrics['dev_main_metric']:
                best_metric = max(metrics['dev_main_metric'])
                logging.info(f"阶段 {stage_idx + 1}: 最佳 {self.main_metric} = {best_metric:.4f}, "
                            f"最终损失 = {metrics['train_loss'][-1]:.4f}")