import numpy as np
import re
import os


class SCANParser:
    def __init__(self):
        """
        C  := CP S | CP V | S | V
        CP := S and | S after | V and | V after
        S  := V twice | V thrice
        V  := VP D | VP left | VP right | D | U
        VP := D opposite | D around | U opposite | U around
        D  := U left | U right
        U  := walk | look | run | jump | turn
        """
        pass

    def shift_reduce(self, sequence):
        seq = ["<start>"]
        stack = [[]]
        for i, word in enumerate(sequence):
            word = sequence[i]
            this_stack = self.copy_stack(stack[-1])
            if word in ['walk', 'look', 'run', 'jump', 'turn']:
                this_stack.append("U")
                seq.append(word)
                stack.append(this_stack)
            elif word in ['left', 'right']:
                if this_stack[-1] == 'U':
                    seq.append(word)
                    this_stack.pop(-1)
                    this_stack.append("D")
                    stack.append(this_stack)
                elif this_stack[-1] == "VP":
                    seq.append(word)
                    this_stack.pop(-1)
                    this_stack.append("V")
                    stack.append(this_stack)
                else:
                    raise ValueError(
                        f"Parsing error({i}, {word}): {sequence}")
            elif word in ["opposite", "around"]:
                if this_stack[-1] not in ["U", "D"]:
                    raise ValueError(
                        f"Parsing error({i}, {word}): {sequence}")
                seq.append(word)
                this_stack.pop(-1)
                this_stack.append("VP")
                stack.append(this_stack)
            elif word in ["twice", "thrice"]:
                if this_stack[-1] != "V":
                    if this_stack[-1] in ["D", "U"]:
                        seq.append("<reduce>")
                        new_stack = self.copy_stack(this_stack)
                        new_stack[-1] = "V"
                        stack.append(new_stack)
                        # apply rule V = D | U
                        this_stack = self.copy_stack(new_stack)
                    else:
                        raise ValueError(
                            f"Parsing error({i}, {word}): {sequence}")
                seq.append(word)
                this_stack.pop(-1)
                this_stack.append("S")
                stack.append(this_stack)
            elif word in ["and", "after"]:
                if this_stack[-1] not in ["S", "V", "D", "U"]:
                    raise ValueError(
                        f"Parsing error({i}, {word}): {sequence}")
                seq.append(word)
                this_stack.pop(-1)
                this_stack.append("CP")
                stack.append(this_stack)
            else:
                raise ValueError(f"Parsing error: unknown word {word}")
        if len(stack[-1]) >= 2:
            if stack[-1][-1] in ["S", "V"] and stack[-1][-2] == "CP":
                seq.append("<reduce>")
                this_stack = self.copy_stack(stack[-1])
                this_stack.pop(-1)
                this_stack[-1] = "C"
                stack.append(this_stack)

        if not len(stack) == len(seq):
            for s, t in zip(seq, stack):
                print(s, t)
            raise ValueError("Length mismatch")
        if not len(stack[-1]) == 1 and stack[-1][-1] == "C":
            for s, t in zip(seq, stack):
                print(s, t)
            raise ValueError("open ended")
        return seq, stack

    def copy_stack(self, stack):
        return [w for w in stack]


class Batcher:
    alphabet = ["<start>", "<reduce>", "<pad>",
                "and", "after", "twice", "thrice",
                "left", "right", "opposite", "around", "walk", "look",
                "run", 'jump', "turn"]
    stack_sym = ["C", "CP", "S", "V", "VP", "D", "U", "PAD"]

    word2id = {w: i for i, w in enumerate(alphabet)}
    stack2id = {w: i for i, w in enumerate(stack_sym)}
    vocab_size = len(word2id)
    state_size = 2
    stack_vocab_size = len(stack_sym)
    pad_state = 0
    valid_state = 1

    def __init__(self):
        self.parser = SCANParser()
        train_seq, train_stk, train_ml, train_mr = self._load('train')
        test_seq, test_stk, test_ml, test_mr = self._load('test')
        self.maxlen = max(train_ml, test_ml)
        self.max_rec = max(train_mr, test_mr)
        self.train_seq, self.train_stack = self._pad(train_seq, train_stk)
        self.test_seq, self.test_stack = self._pad(train_seq, train_stk)
        self._train_pool = np.arange(self.train_seq.shape[0])
        np.random.seed(2021)
        np.random.shuffle(self._train_pool)
        self._test_pool = np.arange(self.test_seq.shape[0])
        print("=" * 10, "Dataset Info", "=" * 10)
        print(f"{self.train_seq.shape[0]} Training samples")
        print(f"{self.test_seq.shape[0]} Test samples")
        print(f"Max sequence length {self.maxlen}, "
              f"Max Recursion {self.max_rec}\n")

    def _load(self, split):
        sequences = list()
        stacks = list()
        maxlen = 0
        max_recur = 0
        dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "SCAN/simple_split")
        with open(f"{dir}/tasks_{split}_simple.txt") as fp:
            for line in fp:
                text = re.findall("IN: (.*) OUT:", line)[0].split()
                seq, stk = self.parser.shift_reduce(text)
                sequences.append(seq)
                stacks.append(stk)
                maxlen = max(len(seq), maxlen)
                max_recur = max(max_recur, max(len(i) for i in stk))
        return sequences, stacks, maxlen, max_recur

    def _pad(self, seq, stk):
        n_samples = len(seq)
        seq_np = np.ones([n_samples, self.maxlen],
                         dtype=np.int8) * self.word2id['<pad>']
        stk_np = np.ones([n_samples, self.maxlen, self.max_rec],
                         dtype=np.int8) * self.stack2id['PAD']
        for i, (s, t) in enumerate(zip(seq, stk)):
            wids = [self.word2id[w] for w in s]
            seq_np[i, :len(wids)] = wids
            for j, stack in enumerate(t):
                tids = [self.stack2id[w] for w in stack]
                stk_np[i, j, :len(tids)] = tids
        return seq_np, stk_np

    def cls_train_samples(self, batch_size):
        raise NotImplementedError("SCAN has no classification task")

    def cls_test_samples(self, batch_size):
        raise NotImplementedError("SCAN has no classification task")

    def lm_train_samples(self, batch_size):
        yield from self._lm_iterator(
            self.train_seq, self._train_pool, batch_size, True)

    def lm_test_samples(self, batch_size):
        yield from self._lm_iterator(
            self.test_seq, self._test_pool, batch_size, False)

    def fs_train_samples(self, batch_size):
        yield from self._fs_iterator(
            self.train_seq, self.train_stack, self._train_pool,
            batch_size, True)

    def fs_test_samples(self, batch_size):
        yield from self._fs_iterator(
            self.test_seq, self.test_stack, self._test_pool,
            batch_size, False)

    def _lm_iterator(self, seq_all, pool, batch_size, shuffle=True):
        for start_idx in np.arange(0, len(pool), step=batch_size):
            idx = pool[start_idx: start_idx + batch_size]
            seq_batch = np.take(seq_all, idx, axis=0)
            yield seq_batch, None
        if shuffle:
            np.random.shuffle(pool)

    def _fs_iterator(self, seq_all, stack_all, pool, batch_size, shuffle=True):
        for start_idx in np.arange(len(pool), step=batch_size):
            idx = pool[start_idx: start_idx + batch_size]
            seq_batch = np.take(seq_all, idx, axis=0)
            stack_batch = np.take(stack_all, idx, axis=0)
            state_batch = np.where(seq_batch != self.word2id['<pad>'],
                                   self.valid_state, self.pad_state)
            yield seq_batch, state_batch, stack_batch, None
        if shuffle:
            np.random.shuffle(pool)
