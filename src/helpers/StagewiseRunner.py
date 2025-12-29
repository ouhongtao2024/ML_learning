import logging
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from helpers.BaseRunner import BaseRunner
from models.BaseModel import BaseModel
from models.general.StagewiseGCN import StagewiseGCN
from utils import utils


class StagewiseRunner(BaseRunner):
    def train(self, data_dict: Dict[str, BaseModel.Dataset]) -> None:
        model: StagewiseGCN = data_dict["train"].model
        main_metric_results, dev_results = list(), list()
        self._check_time(start=True)

        try:
            for stage in range(1, model.stages + 1):
                model.optimizer = None

                logging.info(f"开始第 {stage}/{model.stages} 阶段训练")
                model.set_stage(stage=stage)

                stage_main_metric_results = []

                for epoch in range(1, self.epoch + 1):
                    self._check_time()
                    torch.cuda.empty_cache()

                    loss = self.fit(data_dict["train"], epoch=epoch)

                    if torch.isnan(torch.tensor(loss)):
                        logging.warning(
                            f"在阶段 {stage} 训练的 第 {epoch} 轮中，损失函数为 NaN，停止"
                        )
                        break

                    training_time = self._check_time()

                    dev_result = self.evaluate(
                        data_dict["dev"], [self.main_topk], self.metrics
                    )
                    dev_results.append(dev_result)
                    main_metric_results.append(dev_result[self.main_metric])
                    stage_main_metric_results.append(dev_result[self.main_metric])

                    logging_str = f"[Stage {stage}][Epoch {epoch}] loss={loss:.4f} [{training_time:.1f}s] dev=({dev_result})"  # noqa
                    if dev_result[self.main_metric] == max(stage_main_metric_results):
                        model.save_model()
                        logging_str += " *"

                    logging.info(logging_str)

                    if self.early_stop > 0 and self.eval_termination(
                        stage_main_metric_results
                    ):
                        logging.info(f"在第 {stage} 阶段第 {epoch} 轮发生早停")
                        break

        except KeyboardInterrupt:
            logging.info("训练被手动中断")
            exit(1)

        model.load_model()

    def fit(self, dataset: BaseModel.Dataset, epoch=-1) -> float:
        model: StagewiseGCN = dataset.model
        if model.optimizer is None:
            model.optimizer = self._build_optimizer(model)

        dataset.actions_before_epoch()
        model.train()
        loss_lst = []

        dl = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=dataset.collate_batch,
            pin_memory=self.pin_memory,
        )

        for batch in tqdm(
            dl, leave=False, desc=f"Epoch {epoch}", ncols=100, mininterval=1
        ):
            batch = utils.batch_to_gpu(batch, model.device)
            model.optimizer.zero_grad()
            out_dict = model(batch)
            loss = model.loss(out_dict)
            loss.backward()
            model.optimizer.step()
            loss_lst.append(loss.detach().cpu().item())

        return float(sum(loss_lst) / len(loss_lst))

    def predict(self, dataset: BaseModel.Dataset, save_prediction: bool = False):
        model: StagewiseGCN = dataset.model
        model.eval()
        predictions = []

        dl = DataLoader(
            dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=dataset.collate_batch,
            pin_memory=self.pin_memory,
        )

        for batch in tqdm(dl, leave=False, desc="Predict", ncols=100, mininterval=1):
            batch = utils.batch_to_gpu(batch, model.device)
            out_dict = model(batch)
            predictions.extend(out_dict["prediction"].detach().cpu().numpy())

        return np.array(predictions)
