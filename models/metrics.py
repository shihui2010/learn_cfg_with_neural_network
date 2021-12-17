import numpy as np


LOG_LEVEL = {"EPOCH"}


class _Average:
    def __init__(self):
        self._val = 0
        self._count = 0

    def update_scalar(self, val, weight=None):
        try:
            self._val += val.detach().cpu().item()
        except Exception:
            self._val += val
        self._count += 1 if weight is None else weight

    def update_tensor(self, target, pred, mask=None):
        pred = pred.detach().cpu().numpy()
        if len(pred.shape) == len(target.shape):
            pred = np.greater(pred, 0).astype(int)
        else:
            pred = np.argmax(pred, axis=-1)
        if mask is None:
            self._val += np.equal(pred, target).astype(int).sum()
        else:
            idx1 = np.arange(pred.shape[0]).repeat(pred.shape[1])
            idx1 = idx1.reshape([mask.shape[0], mask.shape[1]])
            idx2 = np.arange(pred.shape[1]).repeat(pred.shape[0])
            idx2 = idx2.reshape([mask.shape[1], mask.shape[0]]).T
            self._val += mask[idx1, idx2, pred].sum()
        self._count += pred.size

    def reset(self):
        self._val = 0
        self._count = 0

    def result(self):
        if self._count == 0:
            return 0
        return self._val / self._count

    def __repr__(self):
        return f"{self.result(): .5f}"


class _StackAcc:
    def __init__(self, max_recur):
        self._accs = [_Average() for _ in range(max_recur)]

    def update_tensor(self, target, pred, mask=None):
        """
        :param target: [batch_size, seq_len - 1, max_recur]
        :param pred: [batch_size, seq_len - 1, max_recur, vocab_size]
        :param mask: None or [batch_size, seq_len - 1, max_recur]
        :return:
        """
        if mask is not None:
            print(target.shape, pred.shape, mask.shape)
        for pos, acc in enumerate(self._accs):
            this_mask = None if mask is None else mask[:, :, pos]
            acc.update_tensor(target[:, :, pos],
                              pred[:, :, pos], this_mask)

    def reset(self):
        for acc in self._accs:
            acc.reset()

    def result(self):
        return [a.result() for a in self._accs]

    def avg_result(self):
        return sum(self.result()) / max(1, len(self._accs))

    def __repr__(self):
        return "|".join([f"Pos{i}: {a.result(): .5f}"
                         for i, a in enumerate(self._accs)])


class ClassificationMetric:
    def __init__(self, writer, tag: str):
        self._loss = _Average()
        self._accuracy = _Average()
        self._writer = writer
        self._tag = tag
        self.best_acc = 0
        self._best_loss = 1000
        self._best_epoch = 0

    def update_state(self, loss, pred, target, step: int):
        self._loss.update_scalar(loss)
        self._accuracy.update_tensor(target=target, pred=pred)
        if "STEP" in LOG_LEVEL:
            self._writer.add_scalar(
                f"[CLS]{self._tag}LossStep", self._loss.result(), step)
            self._writer.add_scalar(
                f"[CLS]{self._tag}AccuracyStep", self._accuracy.result(), step)

    def reset_state(self, epoch: int):
        if "EPOCH" in LOG_LEVEL:
            self._writer.add_scalar(
                f"[CLS]{self._tag}LossEpoch", self._loss.result(), epoch)
            self._writer.add_scalar(
                f"[CLS]{self._tag}AccuracyEpoch", self._accuracy.result(),
                epoch)
        print(f"[CLS SUM] {self._tag} | Epoch {epoch} | "
              f"Acc. {str(self._accuracy)} | "
              f"Loss {str(self._loss)}")
        if (self._accuracy.result() > self.best_acc or
                self._loss.result() < self._best_loss):
            self.best_acc = self._accuracy.result()
            self._best_loss = self._loss.result()
            self._best_epoch = epoch
        self._loss.reset()
        self._accuracy.reset()

    def print(self, epoch, step):
        print(f"[CLS] {self._tag} Epoch {epoch} | Step {step} | "
              f"Acc. {str(self._accuracy)} | Loss {str(self._loss)}")

    def early_stop(self, epoch):
        if epoch - self._best_epoch > 5:
            return True
        return abs(self._accuracy.result() - 1) < 1e-7

    def reset_early_stop(self, cur_epoch):
        self.best_acc = 0
        self._best_loss = 1000
        self._best_epoch = cur_epoch


class LMMetric:
    def __init__(self, writer, tag: str):
        self._loss = _Average()
        self._accuracy = _Average()
        self._writer = writer
        self._tag = tag
        self.best_acc = 0
        self.best_loss = 0  # make public for compute perplexity
        self._best_epoch = 0

    def update_state(self, loss, pred, target, mask, step: int):
        self._loss.update_scalar(loss)
        if mask is not None:
            self._accuracy.update_tensor(
                target=target[:, 1:],
                pred=pred[:, :-1],
                mask=mask[:, 1:])
        else:
            self._accuracy.update_tensor(
                target=target[:, 1:],
                pred=pred[:, :-1],
                mask=None)
        if "STEP" in LOG_LEVEL:
            self._writer.add_scalar(
                f"[LM]{self._tag}LossStep", self._loss.result(), step)
            self._writer.add_scalar(
                f"[LM]{self._tag}AccuracyStep", self._accuracy.result(), step)

    @property
    def accuracy(self):
        return self._accuracy.result()

    def reset_state(self, epoch: int):
        if "EPOCH" in LOG_LEVEL:
            self._writer.add_scalar(
                f"[LM]{self._tag}LossEpoch", self._loss.result(), epoch)
            self._writer.add_scalar(
                f"[LM]{self._tag}AccuracyEpoch", self._accuracy.result(),
                epoch)
        print(f"[LM SUM] {self._tag} | Epoch {epoch} | "
              f"Acc. {str(self._accuracy)} | "
              f"Loss {str(self._loss)}")
        if (self._accuracy.result() > self.best_acc or
                self._loss.result() < self.best_loss):
            self.best_acc = self._accuracy.result()
            self.best_loss = self._loss.result()
            self._best_epoch = epoch
        self._loss.reset()
        self._accuracy.reset()

    def print(self, epoch, step):
        print(f"[LM] {self._tag} Epoch {epoch} | Step {step} | "
              f"Acc. {str(self._accuracy)} | Loss {str(self._loss)}")

    def early_stop(self, epoch):
        if epoch - self._best_epoch > 5:
            return True
        return abs(self._accuracy.result() - 1) < 1e-7

    def reset_early_stop(self, cur_epoch):
        self.best_acc = 0
        self.best_loss = 1000
        self._best_epoch = cur_epoch


class PDAMetric:
    def __init__(self, writer, tag: str, max_recur: int):
        self._loss = _Average()
        self._stack_acc = _StackAcc(max_recur)
        self._state_acc = _Average()
        self._writer = writer
        self._tag = tag
        self.best_state = 0
        self.best_stack = 0  # stack average
        self.best_stack_list = None  # per-position accuracy
        self._best_loss = 100
        self._best_epoch = 0

    def update_state(self, loss, state, state_pred, stack, stack_pred, step):
        self._loss.update_scalar(loss)
        self._state_acc.update_tensor(
            target=state[:, 1:], pred=state_pred[:, :-1])
        self._stack_acc.update_tensor(
            target=stack[:, 1:], pred=stack_pred[:, :-1])
        if "STEP" in LOG_LEVEL:
            self._writer.add_scalar(
                f"[PDA]{self._tag}LossStep", self._loss.result(), step)
            self._writer.add_scalar(
                f"[PDA]{self._tag}StateAccuracyStep",
                self._state_acc.result(), step)
            self._writer.add_scalar(
                f"[PDA]{self._tag}StackAccuracySetp", 
                self._stack_acc.avg_result(), step)
            for i, r in enumerate(self._stack_acc.result()):
                self._writer.add_scalar(
                    f"[PDA]{self._tag}StackPos{i}AccuracyStep", r, step)

    def reset_state(self, epoch):
        if "EPOCH" in LOG_LEVEL:
            self._writer.add_scalar(
                f"[PDA]{self._tag}LossEpoch", self._loss.result(), epoch)
            self._writer.add_scalar(
                f"[PDA]{self._tag}StateAccuracyEpoch",
                self._state_acc.result(), epoch)
            self._writer.add_scalar(
                f"[PDA]{self._tag}StackAccuracyEpoch",
                self._stack_acc.avg_result(), epoch)
            for i, r in enumerate(self._stack_acc.result()):
                self._writer.add_scalar(
                    f"[PDA]{self._tag}StackPos{i}AccuracyEpoch", r, epoch)
        print(f"[PDA SUM] {self._tag} | Epoch {epoch} | "
              f"StateAcc. {str(self._state_acc)}"
              f" | StackAcc. {str(self._stack_acc)} | "
              f"Loss {str(self._loss)}")
        if (self._state_acc.result() > self.best_state or
                np.mean(self._stack_acc.result()) > self.best_stack or
                self._loss.result() < self._best_loss):
            self.best_state = self._state_acc.result()
            self.best_stack = np.mean(self._stack_acc.result())
            self.best_stack_list = self._stack_acc.result()
            self._best_loss = self._loss.result()
            self._best_epoch = epoch
        self._loss.reset()
        self._state_acc.reset()
        self._stack_acc.reset()

    def print(self, epoch, step):
        print(f"[PDA] {self._tag} Epoch {epoch} | Step {step} | "
              f"StateAcc. {str(self._state_acc)} | "
              f"StackAcc. {str(self._stack_acc)} | "
              f"Loss {str(self._loss)}")

    def early_stop(self, epoch):
        if epoch - self._best_epoch > 10:
            return True
        if abs(self._state_acc.result() - 1) > 1e-7:
            return False
        for r in self._stack_acc.result():
            if abs(r - 1) > 1e-7:
                return False
        return True


class FullMetric:
    def __init__(self, writer: str, tag: str, max_recur: int):
        self._loss = _Average()
        self._symbol_acc = _Average()
        self._stack_acc = _StackAcc(max_recur)
        self._state_acc = _Average()
        self._writer = writer
        self._tag = tag
        self.best_state = 0
        self.best_stack = 0
        self.best_stack_list = None
        self.best_symbol = 0
        self._best_loss = 100
        self._best_epoch = 0

    def update_state(self, loss, seq, seq_pred, mask,
                     state, state_pred, stack, stack_pred, step):
        self._loss.update_scalar(loss)
        self._symbol_acc.update_tensor(
            target=seq[:, 1:],
            pred=seq_pred[:, :-1],
            mask=mask[:, 1:])
        self._state_acc.update_tensor(
            target=state[:, 1:],
            pred=state_pred[:, :-1])
        self._stack_acc.update_tensor(
            target=stack[:, 1:],
            pred=stack_pred[:, :-1])
        if "STEP" in LOG_LEVEL:
            self._writer.add_scalar(
                f"[Full]{self._tag}LossStep", self._loss.result(), step)
            self._writer.add_scalar(
                f"[Full]{self._tag}SymbolAccuracyStep",
                self._symbol_acc.result(), step)
            self._writer.add_scalar(
                f"[Full]{self._tag}StateAccuracyStep",
                self._state_acc.result(), step)
            self._writer.add_scalar(
                f"[Full]{self._tag}StackAccuracyStep",
                self._stack_acc.avg_result(), step)
            for i, r in enumerate(self._stack_acc.result()):
                self._writer.add_scalar(
                    f"[Full]{self._tag}StackPos{i}AccuracyStep", r, step)

    def reset_state(self, epoch):
        if "EPOCH" in LOG_LEVEL:
            self._writer.add_scalar(
                f"[Full]{self._tag}LossEpoch", self._loss.result(), epoch)
            self._writer.add_scalar(
                f"[Full]{self._tag}SymbolAccuracyEpoch",
                self._symbol_acc.result(), epoch)
            self._writer.add_scalar(
                f"[Full]{self._tag}StateAccuracyEpoch",
                self._state_acc.result(), epoch)
            self._writer.add_scalar(
                f"[Full]{self._tag}StackAccuracyEpoch",
                self._stack_acc.avg_result(), epoch)
            for i, r in enumerate(self._stack_acc.result()):
                self._writer.add_scalar(
                    f"[Full]{self._tag}StackPos{i}AccuracyEpoch", r, epoch)
        print(f"[Full SUM] {self._tag} | Epoch {epoch} | "
              f"SymbolAcc. {str(self._symbol_acc)} | "
              f"StateAcc. {str(self._state_acc)}"
              f" | StackAcc. {str(self._stack_acc)} | "
              f"Loss {str(self._loss)}")

        if (self._state_acc.result() > self.best_state or
                self._symbol_acc.result() > self.best_symbol or
                np.mean(self._stack_acc.result()) > self.best_stack or
                self._loss.result() < self._best_loss):
            self.best_state = self._state_acc.result()
            self.best_stack = np.mean(self._stack_acc.result())
            self.best_stack_list = self._stack_acc.result()
            self.best_symbol = self._symbol_acc.result()
            self._best_loss = self._loss.result()
            self._best_epoch = epoch

        self._loss.reset()
        self._symbol_acc.reset()
        self._state_acc.reset()
        self._stack_acc.reset()

    def print(self, epoch, step):
        print(f"[Full] {self._tag} Epoch {epoch} | Step {step} | "
              f"SymbolAcc. {str(self._symbol_acc)} | "
              f"StateAcc. {str(self._state_acc)} | "
              f"StackAcc. {str(self._stack_acc)} | "
              f"Loss {str(self._loss)}")

    def early_stop(self, epoch):
        if epoch - self._best_epoch > 10:
            return True
        if abs(self._symbol_acc.result() - 1) > 1e-7:
            return False
        if abs(self._state_acc.result() - 1) > 1e-7:
            return False

        for r in self._stack_acc.result():
            if abs(r - 1) > 1e-7:
                return False
        return True


class ParsingGeneralMetric:
    def __init__(self, writer, tag):
        raise RuntimeError("deprecated class")
        self._seq_acc = _Average()
        self._tok_acc = _Average()
        self._loss = _Average()
        self._writer = writer
        self._tag = tag
        self.best_acc = 0
        self._best_epoch = 0

    def update_state(self, loss, target, pred, step=0):
        target = target.detach().cpu().numpy()
        self._tok_acc.update_tensor(target, pred, mask=None)
        self._loss.update_scalar(loss)
        pred = pred.detach().cpu().numpy()
        seq_match = int(np.equal(pred, target).astype(int).all())
        self._seq_acc.update_scalar(seq_match)
        if "STEP" in LOG_LEVEL:
            self._writer.add_scalar(
                f"{self._tag}LossStep", self._loss.result(), step)
            self._writer.add_scalar(
                f"{self._tag}SeqAccStep", self._seq_acc.result(), step)
            self._writer.add_scalar(
                f"{self._tag}TokAccStep", self._tok_acc.result(), step)

    def reset_state(self, epoch):
        if "EPOCH" in LOG_LEVEL:
            self._writer.add_scalar(
                f"{self._tag}LossEpoch", self._loss.result(), epoch)
            self._writer.add_scalar(
                f"{self._tag}SeqAccEpoch", self._seq_acc.result(), epoch)
            self._writer.add_scalar(
                f"{self._tag}TokAccEpoch", self._tok_acc.result(), epoch)
        print(f"{self._tag} | Epoch {epoch} | Seq Acc. {self._seq_acc} | "
              f"Tok Acc. {self._tok_acc} | Loss {self._loss}")

        if self.best_acc < self._seq_acc.result():
            self.best_acc = self._seq_acc.result()
            self._best_epoch = epoch

        self._seq_acc.reset()
        self._tok_acc.reset()
        self._loss.reset()

    def print(self, epoch, step):
        print(f"{self._tag} | Epoch {epoch} | Step {step} | "
              f"Seq Acc. {self._seq_acc} | "
              f"Tok Acc. {self._tok_acc} | Loss {self._loss}")

    def early_stop(self, epoch):
        if epoch - self._best_epoch > 10:
            return True
        return False
