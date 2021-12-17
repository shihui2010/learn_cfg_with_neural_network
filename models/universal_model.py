from typing import Dict, Any
import torch
import torch.nn as nn
from torch.nn import Parameter
from models.transformer import Encoder


def _get_ffn(input_size, output_size):
    hidden_size = max(input_size, output_size) * 2
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.Sigmoid(),
        nn.Linear(hidden_size, output_size))


class UniversalModel(nn.Module):
    def __init__(self, sequential_model: nn.Module, hps: Dict):
        super(UniversalModel, self).__init__()
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.emb_layer = nn.Embedding(num_embeddings=hps["vocab_size"],
                                      embedding_dim=hps["emb_dim"])
        self._seq_model = sequential_model
        self._hidden_size = self._seq_model.hidden_size
        self._state_size = hps["state_size"]
        self._stack_unit = hps["stack_vocab_size"]
        self._stack_n = hps["max_recur"]
        if hps["latent_factorization"]:
            self.state_pred = _get_ffn(self._hidden_size, self._state_size)
            self.stack_pred = [
                _get_ffn(self._hidden_size, self._stack_unit).to(self.device)
                for _ in range(self._stack_n)]
        else:
            self.state_pred = _get_ffn(self._state_size * hps["multiplier"],
                                       self._state_size)
            self.stack_pred = _get_ffn(self._stack_unit * hps["multiplier"],
                                       self._stack_unit)
        self.symbol_pred = _get_ffn(self._hidden_size, hps["vocab_size"])
        self.classify_nn = _get_ffn(self._hidden_size, 1)
        if not hps["train_sigma"]:
            self.optim = torch.optim.Adam(self.parameters(), lr=0.001)
        self._sigma_sym = Parameter(torch.ones([1]).float())
        self._sigma_state = Parameter(torch.Tensor([1]).float())
        self._sigma_stack = Parameter(torch.Tensor([1]).float())
        self._eps = torch.Tensor([0.1]).to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.bce_loss_fn = nn.BCEWithLogitsLoss()
        self.to(self.device)
        self.hps = hps
        if hps["train_sigma"]:
            self.optim = torch.optim.Adam(self.parameters(), lr=0.001)
        self.hps["seq_model"] = type(self._seq_model).__name__
        self.hps["seq_model_hps"] = self._seq_model.hps

    def count_tv(self):
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return n_params

    def _train(self, loss):
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(self.parameters(), 0.5)
        self.optim.step()

    def freeze_automaton(self):
        param_count = self.count_tv()
        for p in self.emb_layer.parameters():
            p.requires_grad = False
        for p in self._seq_model.parameters():
            p.requires_grad = False
        assert param_count != self.count_tv(), "Failed to freeze model"

    def freeze_peripheral(self):
        param_count = self.count_tv()
        for p in self.symbol_pred.parameters():
            p.requires_grad = False
        for p in self.classify_nn.parameters():
            p.requires_grad = False
        for p in self.state_pred.parameters():
            p.requires_grad = False
        if isinstance(self.stack_pred, list):
            for m in self.stack_pred:
                for p in m.parameters():
                    p.requires_grad = False
        else:
            for p in self.stack_pred.parameters():
                p.requires_grad = False
        assert param_count != self.count_tv()

    def _forward_unimplemented(self):
        """avoid IDE checking"""
        pass

    def forward(self):
        raise NotImplementedError("Call language_model, classifier, "
                                  "full_supervision, or pda_model instead")

    def _encode_seq(self, seq):
        seq = torch.from_numpy(seq).long().to(self.device)
        seq_emb = self.emb_layer(seq).float()
        return seq, self._seq_model(seq_emb)

    def _get_loss(self, pred, target, sigma=None):
        loss = self.loss_fn(pred.reshape(-1, pred.shape[-1]),
                            target.reshape(-1))
        if sigma is None:
            return loss
        return (0.5 / torch.square(torch.abs(sigma) + self._eps) * loss +
                torch.log(torch.abs(sigma) + self._eps))

    def _pda_forward(self, hidden_states):
        if self.hps["latent_factorization"]:
            return self._pda_forward_implicit_factorize(hidden_states)
        return self._pda_forward_explicit_factorize(hidden_states)

    def _pda_forward_explicit_factorize(self, hidden_states):
        """
        :param hidden_states: shape [batch_size, seq_len, hidden_size]
        :return: state_pred [batch_size, seq_len, state_size]
                 stack_pred [batch_size, seq_len, max_recur, vocab_size]
        """
        ss = self._state_size * self.hps["multiplier"]
        state_pred = self.state_pred(hidden_states[:, :, :ss])
        su = self._stack_unit * self.hps["multiplier"]
        stack_pred = list()
        for idx in range(ss, ss + su * self._stack_n, su):
            stack_pred.append(
                self.stack_pred(hidden_states[:, :, idx: idx + su]))
        stack_pred = torch.stack(stack_pred).permute(1, 2, 0, 3)
        return state_pred, stack_pred

    def _pda_forward_implicit_factorize(self, hidden_states):
        """
        :param hidden_states: shape [batch_size, seq_len, hidden_size]
        :return: state_pred [batch_size, seq_len, state_size]
                 stack_pred [batch_size, seq_len, max_recur, vocab_size]
        """
        state_pred = self.state_pred(hidden_states)
        stack_pred = list()
        for idx in range(self._stack_n):
            stack_pred.append(self.stack_pred[idx](hidden_states))
        stack_pred = torch.stack(stack_pred).permute(1, 2, 0, 3)
        return state_pred, stack_pred

    def language_model(self, seq, train: bool):
        seq, hidden_states = self._encode_seq(seq)
        predictions = self.symbol_pred(hidden_states)
        loss = self._get_loss(predictions[:, :-1], seq[:, 1:])
        if train:
            self._train(loss)
        return predictions, loss

    def classifier(self, seq, labels, train: bool, pad_token=None):
        _, hidden_states = self._encode_seq(seq)

        if pad_token is not None:
            last_idx = [s.tolist().index(pad_token) for s in seq]
            last_idx = torch.tensor(last_idx).to(self.device)
            last_idx = torch.reshape(last_idx, [-1, 1, 1])
            hs = torch.take_along_dim(hidden_states, last_idx, 1).squeeze(1)
            # [batch size, hidden size]
        else:
            hs = hidden_states[:, -1]
            # [batch size, hidden size]
        pred = self.classify_nn(hs).squeeze(dim=-1)
        labels = torch.from_numpy(labels).float().to(self.device)
        loss = self.bce_loss_fn(pred, labels)
        if train:
            self._train(loss)
        return pred, loss

    def full_supervision(self, seq, state, stack, train: bool,
                         lm: bool = True):
        seq, hidden_states = self._encode_seq(seq)
        state_pred, stack_pred = self._pda_forward(hidden_states)
        state = torch.from_numpy(state).long().to(self.device)
        stack = torch.from_numpy(stack).long().to(self.device)
        loss = (self._get_loss(state_pred[:, :-1], state[:, 1:],
                               self._sigma_state)
                + self._get_loss(stack_pred[:, :-1], stack[:, 1:],
                                 self._sigma_stack))
        symbol_pred = None
        if lm:
            symbol_pred = self.symbol_pred(hidden_states)
            loss = loss + self._get_loss(symbol_pred[:, :-1], seq[:, 1:],
                                         self._sigma_sym)
        if train:
            self._train(loss)
        if lm:
            return symbol_pred, state_pred, stack_pred, loss
        return state_pred, stack_pred, loss

    def pda_model(self, seq, state, stack, train: bool):
        return self.full_supervision(seq, state, stack, train, lm=False)

    def stack_model(self, seq, state, stack, train: bool):
        """only for loss function visualization"""
        seq, hidden_states = self._encode_seq(seq)
        state_pred, stack_pred = self._pda_forward(hidden_states)
        stack = torch.from_numpy(stack).long().to(self.device)
        loss = self._get_loss(stack_pred[:, :-1], stack[:, 1:])
        return stack_pred, loss


class LSTMModel(nn.Module):
    def __init__(self, emb_dim, hidden_size, n_layers):
        super(LSTMModel, self).__init__()
        self.rnn_layer = nn.LSTM(input_size=emb_dim,
                                 hidden_size=hidden_size,
                                 num_layers=n_layers,
                                 batch_first=True)
        self.hidden_size = hidden_size
        self.hps = {"emb_dim": emb_dim,
                    "hidden_size": hidden_size,
                    "n_layers": n_layers}

    def _forward_unimplemented(self, *x: Any) -> None:
        pass

    def forward(self, seq_emb):
        return self.rnn_layer(seq_emb)[0]


class TransformerModel(nn.Module):
    def __init__(self, emb_dim, n_layers, n_heads,
                 causal=False, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.layer = Encoder(
            hidden_size=emb_dim,
            num_layers=n_layers,
            num_heads=n_heads,
            total_key_depth=n_heads * 2,
            total_value_depth=n_heads * 2,
            filter_size=32,
            max_length=1000,
            input_dropout=dropout,
            layer_dropout=dropout,
            attention_dropout=dropout,
            relu_dropout=dropout,
            use_mask=causal)
        self.hps = {"emb_dim": emb_dim,
                    "n_layers": n_layers,
                    "n_heads": n_heads,
                    "dropout": dropout}
        self.hidden_size = emb_dim

    def _forward_unimplemented(self, *x: Any) -> None:
        pass

    def forward(self, seq_emb):
        return self.layer(seq_emb)[0]
