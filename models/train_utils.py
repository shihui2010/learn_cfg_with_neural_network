import os
import numpy as np
import json
import torch
import glob
from torch.utils.tensorboard import SummaryWriter
from models.universal_model import (
    UniversalModel,
    LSTMModel,
    TransformerModel)
from models.metrics import (
    ClassificationMetric,
    LMMetric,
    PDAMetric,
    FullMetric)
from dataset.dataset import Batcher


def save_model(model: UniversalModel, log_dir: str, tag: str) -> None:
    """
    :param model: universal model
    :param log_dir: logging directory
    :param tag: post-fix for model file
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    torch.save(model.state_dict(), f"{log_dir}/model_{tag}.pt")
    with open(f"{log_dir}/hps.json", "w") as fp:
        json.dump(model.hps, fp, indent=2)


def restore_model(log_dir: str, tag: str) -> UniversalModel:
    with open(f"{log_dir}/hps.json", "r") as fp:
        hps = json.load(fp)
        if "stack_vocab_size" not in hps:
            hps["stack_vocab_size"] = hps["vocab_size"]
    if "LSTMModel" in hps["seq_model"]:
        seq_model = LSTMModel(**hps["seq_model_hps"])
    else:
        seq_model = TransformerModel(**hps["seq_model_hps"])
    model = UniversalModel(seq_model, hps)
    if tag is None:
        return model
    try:
        model.load_state_dict(torch.load(f"{log_dir}/model_{tag}.pt"))
    except RuntimeError:
        model.load_state_dict(torch.load(f"{log_dir}/model_{tag}.pt",
                                         map_location=torch.device('cpu')))
    return model


def _get_state_size(hps):
    base = hps["state_size"] + hps["max_recur"] * hps["stack_vocab_size"]
    return base * hps["multiplier"]


def get_model(batcher, args):
    hps = {
        "vocab_size": batcher.vocab_size,
        "state_size": batcher.state_size,
        "stack_vocab_size": batcher.stack_vocab_size,
        "max_recur": batcher.max_rec,
        "multiplier": args.scale_factor,
        "train_sigma": args.train_sigma,
        "latent_factorization": args.latent_factorization
    }
    hidden_size = _get_state_size(hps)
    if args.model == "lstm":
        hps["emb_dim"] = batcher.vocab_size * 2
        seq_model = LSTMModel(emb_dim=hps["emb_dim"],
                              hidden_size=hidden_size,
                              n_layers=args.n_layers)
    else:
        use_mask = args.causal
        hps["emb_dim"] = hidden_size
        seq_model = TransformerModel(emb_dim=hps["emb_dim"],
                                     n_layers=args.n_layers,
                                     n_heads=args.n_heads,
                                     causal=use_mask)
    return UniversalModel(seq_model, hps)


class Trainer:
    def __init__(self, model: UniversalModel,
                 batcher: Batcher,
                 log_dir: str):
        # remove old tensorboard logs
        fs = glob.glob(f"{log_dir}/events.out.tfevents*")
        for f in fs:
            try:
                os.remove(f)
            except Exception as e:
                print("Error while deleting file : ", f, e)

        self._writer = SummaryWriter(log_dir)
        self.cls_metric = [ClassificationMetric(self._writer, "Train"),
                           ClassificationMetric(self._writer, "Test")]
        self.lm_metric = [LMMetric(self._writer, "Train"),
                          LMMetric(self._writer, "Test")]
        self.pda_metric = [PDAMetric(self._writer, "Train", batcher.max_rec),
                           PDAMetric(self._writer, "Test", batcher.max_rec)]
        self.full_metric = [FullMetric(self._writer, "Train", batcher.max_rec),
                            FullMetric(self._writer, "Test", batcher.max_rec)]

        self.model = model
        self.batcher = batcher
        self.cls_epoch = 0
        self.lm_epoch = 0
        self.pda_epoch = 0
        self.full_epoch = 0
        self.cls_phase0_best = 0
        self.cls_phase1_best = 0
        self.lm_phase0_best = 0
        self.perplexity0_best = 10000
        self.perplexity1_best = 10000
        self.lm_phase1_best = 0
        self.lm_best = 0   # for full supervision training
        self.state_best = 0
        self.stack_best = 0
        self.stack_best_list = None

    def reset_cls_metric(self):
        self.cls_metric[0].reset_early_stop(self.cls_epoch)
        self.cls_metric[1].reset_early_stop(self.cls_epoch)

    def reset_lm_metric(self):
        self.lm_metric[0].reset_early_stop(self.lm_epoch)
        self.lm_metric[1].reset_early_stop(self.lm_epoch)

    def run_cls_epoch(self, batch_size) -> bool:
        for step, (seq, label) in enumerate(
                self.batcher.cls_train_samples(batch_size)):
            pred, loss = self.model.classifier(
                seq, label, train=True,
                pad_token=self.batcher.pad_token)
            self.cls_metric[0].update_state(loss, pred, label, step)
            if step % 100 == 0:
                self.cls_metric[0].print(self.cls_epoch, step)
        self.cls_metric[0].reset_state(self.cls_epoch)

        for step, (seq, label) in enumerate(
                self.batcher.cls_test_samples(batch_size)):
            pred, loss = self.model.classifier(
                seq, label, train=False,
                pad_token=self.batcher.pad_token)
            self.cls_metric[1].update_state(loss, pred, label, step)
            if step % 100 == 0:
                self.cls_metric[1].print(self.cls_epoch, step)
        early_stop = self.cls_metric[1].early_stop(self.cls_epoch)
        self.cls_metric[1].reset_state(self.cls_epoch)
        if self.full_epoch > 0 or self.pda_epoch > 0:
            self.cls_phase1_best = max(self.cls_phase1_best,
                                       self.cls_metric[1].best_acc)
        else:
            self.cls_phase0_best = max(self.cls_phase0_best,
                                       self.cls_metric[1].best_acc)
        self.cls_epoch += 1
        return early_stop

    def run_lm_epoch(self, batch_size) -> bool:
        for step, (seq, mask) in enumerate(
                self.batcher.lm_train_samples(batch_size)):
            pred, loss = self.model.language_model(seq, train=True)
            self.lm_metric[0].update_state(loss, pred, seq, mask, step)
            if step % 100 == 0:
                self.lm_metric[0].print(self.lm_epoch, step)
        self.lm_metric[0].reset_state(self.lm_epoch)

        for step, (seq, mask) in enumerate(
                self.batcher.lm_test_samples(batch_size)):
            pred, loss = self.model.language_model(seq, train=False)
            self.lm_metric[1].update_state(loss, pred, seq, mask, step)
            if step % 100 == 0:
                self.lm_metric[1].print(self.lm_epoch, step)
        early_stop = self.lm_metric[1].early_stop(self.lm_epoch)
        self.lm_metric[1].reset_state(self.lm_epoch)
        if self.full_epoch > 0 or self.pda_epoch > 0:
            self.lm_phase1_best = max(self.lm_phase1_best,
                                      self.lm_metric[1].best_acc)
            self.perplexity1_best = min(self.perplexity1_best,
                                        np.exp(self.lm_metric[1].best_loss))

        else:
            self.lm_phase0_best = max(self.lm_phase0_best,
                                      self.lm_metric[1].best_acc)
            self.perplexity0_best = min(self.perplexity0_best,
                                        np.exp(self.lm_metric[1].best_loss))
        self.lm_epoch += 1
        return early_stop

    def run_pda_epoch(self, batch_size) -> bool:
        for step, (seq, state, stack, _) in enumerate(
                self.batcher.fs_train_samples(batch_size)):
            state_pred, stack_pred, loss = self.model.pda_model(
                seq, state, stack, train=True)
            self.pda_metric[0].update_state(
                loss, state, state_pred, stack, stack_pred, step)
            if step % 100 == 0:
                self.pda_metric[0].print(self.pda_epoch, step)
        self.pda_metric[0].reset_state(self.pda_epoch)

        for step, (seq, state, stack, _) in enumerate(
                self.batcher.fs_test_samples(batch_size)):
            state_pred, stack_pred, loss = self.model.pda_model(
                seq, state, stack, train=False)
            self.pda_metric[1].update_state(
                loss, state, state_pred, stack, stack_pred, step)
            if step % 100 == 0:
                self.pda_metric[1].print(self.pda_epoch, step)
        early_stop = self.pda_metric[1].early_stop(self.pda_epoch)
        self.pda_metric[1].reset_state(self.pda_epoch)
        self.pda_epoch += 1
        self.state_best = max(self.state_best, self.pda_metric[1].best_state)
        if self.pda_metric[1].best_stack > self.stack_best:
            self.stack_best = self.pda_metric[1].best_stack
            self.stack_best_list = self.pda_metric[1].best_stack_list
        return early_stop

    def run_full_epoch(self, batch_size) -> bool:
        for step, (seq, state, stack, mask) in enumerate(
                self.batcher.fs_train_samples(batch_size)):
            res = self.model.full_supervision(seq, state, stack, train=True)
            symbol_pred, state_pred, stack_pred, loss = res
            self.full_metric[0].update_state(
                loss, seq, symbol_pred, mask,
                state, state_pred, stack, stack_pred,
                step)
            if step % 100 == 0:
                self.full_metric[0].print(self.full_epoch, step)
        self.full_metric[0].reset_state(self.full_epoch)

        for step, (seq, state, stack, mask) in enumerate(
                self.batcher.fs_test_samples(batch_size)):
            res = self.model.full_supervision(seq, state, stack, train=False)
            symbol_pred, state_pred, stack_pred, loss = res
            self.full_metric[1].update_state(
                loss, seq, symbol_pred, mask,
                state, state_pred, stack, stack_pred,
                step)
            if step % 100 == 0:
                self.full_metric[1].print(self.full_epoch, step)
        early_stop = self.full_metric[1].early_stop(self.full_epoch)
        self.full_metric[1].reset_state(self.full_epoch)
        self.full_epoch += 1
        self.lm_best = max(self.lm_best, self.full_metric[1].best_symbol)
        self.state_best = max(self.state_best, self.full_metric[1].best_state)
        if self.pda_metric[1].best_stack > self.stack_best:
            self.stack_best = self.pda_metric[1].best_stack
            self.stack_best_list = self.pda_metric[1].best_stack_list
        return early_stop
