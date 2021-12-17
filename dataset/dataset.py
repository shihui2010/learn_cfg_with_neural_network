from typing import List, Union
from abc import abstractmethod
from itertools import product
from copy import deepcopy, copy
import numpy as np
import os
import pickle


class BasePDA:
    start = 0
    end = 1
    pad = 1
    pad_state = 0

    @abstractmethod
    def corrupt_seq(self, seq: List[int]):
        ...

    @abstractmethod
    def accept(self, seq: List[int], partial: bool = False):
        ...

    @abstractmethod
    def enumerate(self, maxlen, beam_size):
        ...

    def copy_stack(self, stack, max_rec):
        res = np.ones([max_rec]) * self.pad_state
        res[:len(stack)] = stack
        return res

    @staticmethod
    def copy_pop(seq: List):
        res = deepcopy(seq)
        res.pop(-1)
        return res

    @staticmethod
    def copy_append(seq: List[Union[List, int]], a: Union[List, int]):
        res = deepcopy(seq)
        res.append(copy(a))
        return res

    def terminates(self, state: int, stack: List[int]):
        return state == self._term_st and len(stack) == 0

    @property
    @abstractmethod
    def _term_st(self):
        ...

    @property
    @abstractmethod
    def vocab_size(self):
        ...

    @property
    @abstractmethod
    def state_size(self):
        ...

    @property
    def stack_vocab_size(self):
        return self.vocab_size

    @abstractmethod
    def __name__(self):
        ...


class AnBn(BasePDA):
    def __init__(self):
        """
        (state 1, a, epsilon) -> (a, state 1)
        (state 1, b, a) -> (epsilon, state 2)
        (state 2, b, a) -> (epsilon, state 2)
        (state 2, a, epsilon) -> (a, state 1)
        """
        super(AnBn, self).__init__()
        self._a = 2
        self._b = 3
        self._state1 = 1
        self._state2 = 2
        self._max_recur = None

    @property
    def _term_st(self):
        return self._state2

    @property
    def vocab_size(self):
        return 4

    @property
    def state_size(self):
        return 3

    def __name__(self):
        return "<AnBn>"

    def sampling(self, maxlen, n_samples):
        # generate short an + bn sequences
        anbn = list()
        for n in range(1, maxlen // 2 - 1):
            seq = np.empty([n * 2], dtype=np.int8)
            seq[:n] = self._a
            seq[-n:] = self._b
            state = np.empty([n * 2], dtype=np.int8)
            state[:n] = self._state1
            state[-n:] = self._state2
            stack = [[] for _ in range(n * 2)]
            for i in range(n):
                stack[i] = [self._a] * (i + 1)
                stack[-i - 1] = [self._a] * i
            anbn.append((seq, state, stack))
        for seq, state, stack in anbn:
            new_seq = [self.start]
            new_seq.extend(seq)
            new_state = [self._state1]
            new_state.extend(state)
            new_stack = [[]]
            new_stack.extend(stack)
            new_stack = np.stack(
                [self.copy_stack(x, maxlen // 2) for x in new_stack],
                axis=0)
            yield new_seq, new_state, new_stack

        # min(5, maxlen // 4) to avoid max recursion to be too small
        n_splits = np.random.choice(np.arange(0, min(5, maxlen // 4)),
                                    n_samples - len(anbn))
        for n_seg in n_splits:
            seq_len = np.random.randint(n_seg, maxlen // 2 - 1)
            pos = sorted(np.random.choice(
                np.arange(0, seq_len), n_seg, replace=False))
            seq = [self.start]
            state = [self._state1]
            stack = [[]]
            for start, end in zip(pos[:-1], pos[1:]):
                n = end - start
                seq.extend(anbn[n - 1][0])
                state.extend(anbn[n - 1][1])
                stack.extend(deepcopy(anbn[n - 1][2]))

            if n_seg == 0:
                n = seq_len
            else:
                n = seq_len - pos[-1]
            seq.extend(anbn[n - 1][0])
            state.extend(anbn[n - 1][1])
            stack.extend(deepcopy(anbn[n - 1][2]))
            stack = np.stack([self.copy_stack(x, maxlen // 2) for x in stack],
                             axis=0)
            yield seq, state, stack

    def enumerate(self, maxlen, beam_size=1000):
        yield from self.sampling(maxlen, beam_size)

    def _deprecated_enumerate(self, maxlen, beam_size=1000):
        cur_symbols = [self.start]
        cur_states = [self._state1]
        cur_stacks = [[]]
        queue = [(cur_symbols, cur_states, cur_stacks)]
        while len(queue):
            cur_symbols, cur_states, cur_stacks = queue.pop(0)
            if len(cur_symbols) >= maxlen:
                continue
            for symbol, next_state, next_stack in self._step(
                    cur_states[-1], cur_stacks[-1]):
                new_seq = self.copy_append(cur_symbols, symbol)
                new_state = self.copy_append(cur_states, next_state)
                new_stack = self.copy_append(cur_stacks, next_stack)
                queue.append((new_seq, new_state, new_stack))
                if self.terminates(new_state[-1], new_stack[-1]):
                    stack_padded = np.stack(
                        [self.copy_stack(x, maxlen // 2) for x in new_stack],
                        axis=0)
                    yield new_seq, new_state, stack_padded

            if len(queue) > beam_size:
                indices = np.random.choice(
                    np.arange(len(queue)), int(beam_size / 2))
                queue = [queue[i] for i in indices]

    def _step(self, state: int, stack: List[int]) -> (int, int, List[int]):
        if state == self._state1:
            yield self._a, self._state1, self.copy_append(stack, self._a)
            if len(stack):
                yield self._b, self._state2, self.copy_pop(stack)
        elif state == self._state2:
            if len(stack):
                yield self._b, self._state2, self.copy_pop(stack)
            else:
                yield self._a, self._state1, self.copy_append(stack, self._a)
        else:
            raise ValueError(f"Invalid state {state}")

    @staticmethod
    def translate(seq: List[int]):
        return "".join([{2: 'a', 3: 'b'}[w] for w in seq if w in [2, 3]])

    def accept(self, seq: List[int], partial: bool = False):
        stack = []
        state = self._state2  # such that empty sequence is accepted
        if self.end in seq:
            return self.accept(seq[:seq.index(self.end)], partial=False)
        for symbol in seq:
            if symbol == self.start:
                continue
            if symbol == self._a:
                if state == self._state2 and len(stack):
                    return False
                state = self._state1
                stack.append(self._a)
            elif symbol == self._b:
                if len(stack) == 0:
                    return False
                stack.pop(-1)
                state = self._state2
        if partial or self.terminates(state, stack):
            return True
        return False

    def corrupt_seq(self, seq: List[int]):
        seq = deepcopy(seq)
        idx = np.random.choice(range(1, len(seq)))
        if seq[idx] == self._a:
            seq[idx] = self._b
        elif seq[idx] == self._b:
            seq[idx] = self._a
        else:
            raise ValueError("shouldn't corrupt padded sequence")
        assert not self.accept(seq, partial=False), f"failed to corrupt {seq}"
        return seq


class Parity(BasePDA):
    def __init__(self):
        """
        accepts the sequence containing even-number of 0s, DFA
        (state 1, 1, epsilon) -> (epsilon, state 1)
        (state 1, 0, epsilon) -> (epsilon, state 2)
        (state 2, 0, epsilon) -> (epsilon, state 1)
        (state 2, 1, epsilon) -> (epsilon, state 2)
        """
        super(Parity, self).__init__()
        self._1 = 2
        self._0 = 3
        self._state1 = 1
        self._state2 = 2
        self._max_recur = 1

    @property
    def _term_st(self):
        return self._state1

    @property
    def vocab_size(self):
        return 4

    @property
    def state_size(self):
        return 3

    def __name__(self):
        return "<Parity>"

    def sampling(self, maxlen, n_samples):
        n_zeros = np.random.randint(1, maxlen // 2 - 1, n_samples) * 2
        for n in n_zeros:
            seq_len = np.random.randint(n + 1, maxlen)
            pos = np.random.choice(range(1, seq_len), n, replace=False)
            pos = sorted(pos)
            seq = np.ones([seq_len], dtype=np.int8) * self._1
            seq[0] = self.start
            state = np.ones([seq_len], dtype=np.int8) * self._state1
            for start, end in zip(pos[:-1], pos[1:]):
                seq[start] = self._0
                this_state = 3 - state[start - 1]
                state[start: end] = this_state
            seq[pos[-1]] = self._0
            state[pos[-1]:] = self._term_st
            stack = np.ones([seq_len, self._max_recur],
                            dtype=np.int8) * self.pad_state
            yield list(seq), state, stack

    def enumerate(self, maxlen, beam_size=1000):
        yield from self.sampling(maxlen, beam_size)

    def _deprecated_numerate(self, maxlen, beam_size=1000):
        queue = [[[self.start], [self._state1]]]
        while len(queue):
            cur_symbols, cur_states = queue.pop(0)
            if len(cur_symbols) >= maxlen:
                continue
            for symbol, state in self._step(cur_states[-1]):
                new_seq = self.copy_append(cur_symbols, symbol)
                new_state = self.copy_append(cur_states, state)
                queue.append([new_seq, new_state])
                if self.terminates(new_state[-1], []):
                    stack = np.empty([len(new_state), self._max_recur])
                    stack.fill(self.pad_state)
                    yield new_seq, new_state, stack
            if len(queue) > beam_size:
                indices = np.random.choice(
                    np.arange(len(queue)), int(beam_size / 2))
                queue = [queue[i] for i in indices]

    def _step(self, state: int) -> (int, int):
        if state == self._state1:
            yield self._1, self._state1
            yield self._0, self._state2
        elif state == self._state2:
            yield self._1, self._state2
            yield self._0, self._state1
        else:
            raise ValueError(f"Invalid state {state}")

    @staticmethod
    def translate(seq: List[int]):
        return "".join([{2: "1", 3: "0"}[w] for w in seq if w in [2, 3]])

    def accept(self, seq: List[int], partial: bool = False):
        if self.end in seq:
            return self.accept(seq[:seq.index(self.end)], partial=False)
        if partial or sum([1 for w in seq if w == self._0]) % 2 == 0:
            return True
        return False

    def corrupt_seq(self, seq: List[int]):
        seq = deepcopy(seq)
        idx = np.random.choice(range(1, len(seq)))
        if seq[idx] == self._1:
            seq[idx] = self._0
        elif seq[idx] == self._0:
            seq[idx] = self._1
        else:
            raise ValueError("shouldn't corrupt padded sequence")
        assert not self.accept(seq, partial=False), f"failed to corrupt {seq}"
        return seq


class DYCK(BasePDA):
    def __init__(self, k, m):
        """
        (state 1, si, epsilon) -> (si, state 1)
        (state 1, s(i+k), si) -> (epsilon, state 1)
        si pairs with s(i+k)
        param m: maximum number of recursions
        param k: number of types of brackets
        """
        self._max_recur = m
        self._vocabs = {i + 2: f"s{i}" for i in range(k)}
        self._vocabs.update({i + 2 + k: f"e{i}" for i in range(k)})
        self._k = k
        self._state1 = 1

    @property
    def _term_st(self):
        return self._state1

    @property
    def vocab_size(self):
        return self._k * 2 + 2

    @property
    def state_size(self):
        return 2

    def __name__(self):
        return f"<DYCK: m={self._max_recur}, k={self._k}>"

    def sampling(self, maxlen, n_samples):
        blocks = {n: self._block(n, n * 10) for n in range(1, self._max_recur)}
        for _ in range(n_samples):
            length = max(1, np.random.randint(maxlen // 2))
            segments = [0]
            while sum(segments) < length:
                segments.append(
                    np.random.randint(1, self._max_recur))
            segments = np.cumsum(segments)
            segments[-1] = length
            seq = np.ones(length * 2 + 1) * self.start
            state = np.ones_like(seq) * self._state1
            for x1, x2 in zip(segments[:-1], segments[1:]):
                seg_id = np.random.choice(range(len(blocks[x2 - x1])))
                seq[1 + x1 * 2: 1 + x2 * 2] = copy(blocks[x2 - x1][seg_id])
            stack = [[]]
            for token in seq[1:]:
                # print(token, stack, self._k + 2)
                if token < self._k + 2:
                    stack.append(self.copy_append(stack[-1], token))
                else:
                    stack.append(self.copy_pop(stack[-1]))
            stack_padded = np.stack(
                [self.copy_stack(x, self._max_recur) for x in stack],
                axis=0)
            yield seq.tolist(), state.tolist(), stack_padded

    def _block(self, n, n_samples):
        """create sequences like left brackets * x + right brackets * x"""
        if n < 5 and self._k < 5:
            n_samples = min(n_samples, self._k ** n)
        # print(np.arange(2, self._k + 2), n_samples)
        seq = np.random.choice(np.arange(2, self._k + 2), size=[n_samples, n])
        # print(seq.shape, seq, seq[:, ::-1].shape)
        full_seq = np.hstack([seq, seq[:, ::-1] + self._k])
        return full_seq

    def enumerate(self, maxlen, beam_size=1000):
        yield from self.sampling(maxlen, beam_size)

    def _deprecated_enumerate(self, maxlen, beam_size):
        queue = [[[self.start], [[]]]]
        while len(queue):
            cur_symbols, cur_stacks = queue.pop(0)
            if len(cur_symbols) >= maxlen:
                continue
            for symbol, next_stack in self._step(cur_stacks[-1]):
                new_seq = self.copy_append(cur_symbols, symbol)
                new_stack = self.copy_append(cur_stacks, next_stack)
                queue.append((new_seq, new_stack))
                if self.terminates(self._state1, new_stack[-1]):
                    stack_padded = np.stack(
                        [self.copy_stack(x, self._max_recur)
                         for x in new_stack],
                        axis=0)
                    new_state = np.ones_like(new_seq)  # self._state1 = 1
                    yield new_seq, new_state, stack_padded
            if len(queue) > beam_size:
                indices = np.random.choice(
                    np.arange(len(queue)), int(beam_size / 2))
                queue = [queue[i] for i in indices]

    def _step(self, stack: List[int]) -> (int, List[int]):
        if len(stack) == self._max_recur:
            symbol = self._back_bracket(stack[-1])
            new_stack = self.copy_pop(stack)
            yield symbol, new_stack
        else:
            for symbol in range(2, 2 + self._k):
                new_stack = self.copy_append(stack, symbol)
                yield symbol, new_stack
            if len(stack):
                symbol = self._back_bracket(stack[-1])
                new_stack = self.copy_pop(stack)
                yield symbol, new_stack

    def _back_bracket(self, symbol: int):
        assert symbol < self._k + 2, f"Unable to find back bracket " \
            f"for {symbol} (k={self._k}"
        return self._k + symbol

    def translate(self, seq: List[int]):
        return "".join(self._vocabs[w] for w in seq if w > 1)

    def accept(self, seq: List[int], partial: bool = False):
        stack = []
        if self.end in seq:
            return self.accept(seq[:seq.index(self.end)], partial=False)
        for symbol in seq:
            if symbol == self.start:
                continue
            if symbol < self._k + 2:
                if len(stack) == self._max_recur:
                    return False
                stack.append(symbol)
            else:
                if len(stack) == 0 or symbol != self._back_bracket(stack[-1]):
                    return False
                stack.pop(-1)
        if partial or self.terminates(self._state1, stack):
            return True
        return False

    def corrupt_seq(self, seq: List[int]):
        seq = deepcopy(seq)
        idx = np.random.choice(range(1, len(seq)))
        random_symbol = np.random.choice(
            list(w for w in self._vocabs.keys() if w != seq[idx]))
        seq[idx] = random_symbol
        assert not self.accept(seq, partial=False), f"failed to corrupt {seq}"
        return seq


class Palindrome(BasePDA):
    def __init__(self, k: int, m: int, n: int):
        """
        (w{n]cw[n]^R)m language
        :param k: vocabulary size
        :param m: maximum recursion
        :param n: maximum word length (maximum stack size)
        """
        self._max_recur = m
        self._vocabs = {i + 2: f"w{i}" for i in range(k)}
        self._vocabs[k + 2] = "c"
        self._c = k + 2
        self._k = k
        self._max_word_len = n
        self._state1 = 1
        self._state2 = 2

    @property
    def _term_st(self):
        return self._state2

    @property
    def vocab_size(self):
        return self._k + 3

    @property
    def state_size(self):
        return 3

    def __name__(self):
        return f"<Palindrome: m={self._max_recur}, " \
            f"k={self._k}, n={self._max_word_len}>"

    def _count_c(self, seq):
        return sum(1 for x in seq if x == self._c)

    def enumerate(self, maxlen, beam_size=1000):
        yield from self.sampling(maxlen, beam_size)

    def _deprecated_enumerate(self, maxlen, beam_size):
        queue = [([self.start], [self._state1], [[]])]
        while len(queue):
            cur_seq, cur_state, cur_stack = queue.pop(0)
            if len(cur_seq) >= maxlen:
                continue
            if self._count_c(cur_seq) > self._max_recur:
                continue
            for symbol, next_state, next_stack in self._step(
                    cur_state[-1], cur_stack[-1]):
                new_seq = self.copy_append(cur_seq, symbol)
                new_state = self.copy_append(cur_state, next_state)
                new_stack = self.copy_append(cur_stack, next_stack)
                queue.append((new_seq, new_state, new_stack))
                if self.terminates(next_state, next_stack):
                    if len(new_seq) == 2 or \
                            self._count_c(new_seq) > self._max_recur:
                        continue
                    stack_padded = np.stack(
                        [self.copy_stack(x, self._max_word_len)
                         for x in new_stack],
                        axis=0)
                    yield new_seq, new_state, stack_padded
            if len(queue) > beam_size:
                indices = np.random.choice(
                    np.arange(len(queue)), int(beam_size / 2))
                queue = [queue[i] for i in indices]

    def _step(self, state: int, stack: List[int]) -> (int, int, List[int]):
        if len(stack) == self._max_word_len:
            yield self._c, self._state2, copy(stack)
        else:
            if state == self._state1:
                yield self._c, self._state2, copy(stack)
                for i in range(2, self._k + 2):
                    yield i, self._state1, self.copy_append(stack, i)
            elif state == self._state2:
                if len(stack):
                    yield stack[-1], self._state2, self.copy_pop(stack)
                else:
                    for i in range(2, self._k + 2):
                        yield i, self._state1, self.copy_append(stack, i)

    def translate(self, seq: List[int]):
        return "".join(self._vocabs[w] for w in seq if w in self._vocabs)

    def sampling(self, maxlen, target_num):
        print("[Palindrome] creating words")
        chars = np.arange(2, 2 + self._k)
        words = list()
        for length in range(self._max_word_len):
            if self._k ** length < 500:
                words.extend(product(chars, repeat=length))
            else:
                words.extend(np.random.choice(
                    chars, size=[500 * length, length],
                    replace=True).tolist())
        print(f"[Palindrome] {len(words)} words have been created")
        for c in np.random.randint(1, self._max_recur, size=target_num):
            ws = np.random.choice(words, size=c)
            while sum(len(w) for w in ws) * 2 + c + 2 >= maxlen:
                c = max(c - 1, 0)
                ws = np.random.choice(words, size=c)
            seq = [self.start]
            state = [self._state1]
            stack = [[]]
            for w in ws:
                for i, symbol in enumerate(w):
                    seq.append(symbol)
                    state.append(self._state1)
                    stack.append(w[:i + 1])
                seq.append(self._c)
                state.append(self._state2)
                stack.append(w)
                for i, symbol in enumerate(w[::-1]):
                    seq.append(symbol)
                    state.append(self._state2)
                    stack.append(w[::-1][:i + 1])
            seq.append(self.end)
            state.append(self.pad_state)
            stack.append([])
            stack_padded = np.stack(
                [self.copy_stack(x, self._max_word_len)
                 for x in stack], axis=0)
            yield seq, state, stack_padded

    def accept(self, seq: List[int], partial: bool = False):
        stack = []
        state = self._state1
        if self.end in seq:
            return self.accept(seq[:seq.index(self.end)], partial=False)
        count_c = sum(1 for x in seq if x == self._c)
        if count_c > self._max_recur:
            return False
        for symbol in seq:
            if symbol == self.start:
                continue
            if state == self._state1:
                if symbol == self._c:
                    state = self._state2
                    continue
                if len(stack) >= self._max_word_len:
                    return False
                stack.append(symbol)
            else:
                if len(stack) == 0:
                    if symbol != self._c:
                        state = self._state1
                        stack.append(symbol)
                else:
                    if stack[-1] == symbol:
                        stack.pop(-1)
                    else:
                        return False
        if partial or self.terminates(state, stack):
            return True
        return False

    def corrupt_seq(self, seq: List[int]):
        seq = deepcopy(seq)
        idx = np.random.choice([i for i in range(1, len(seq))
                                if seq[i] != self._c])
        random_symbol = np.random.choice(
            [w for w in self._vocabs if w not in [self._c, seq[idx]]])
        seq[idx] = random_symbol
        assert not self.accept(seq, partial=False), f"failed to corrupt {seq}"
        return seq


class Batcher:
    def __init__(self, pda: Union[BasePDA, None], maxlen=100, load_from=None):
        print("======Dataset Information======")
        if load_from is None:
            print(f"creating samples from {type(pda).__name__}")
            self._pda = pda
            self.vocab_size = pda.vocab_size
            self.state_size = pda.state_size
            self.stack_vocab_size = pda.vocab_size
            self._maxlen = maxlen
            return

        print(f"Reading samples from {load_from}")
        with open(f"{load_from}/params.txt", "r") as fp:
            self.vocab_size = int(fp.readline().strip())
            self.max_rec = int(fp.readline().strip())
            self._maxlen = int(fp.readline().strip())
            n_samples = int(fp.readline().strip())
            self.state_size = int(fp.readline().strip())
            self.stack_vocab_size = self.vocab_size

        # data = np.load(load_from + "/data.npz")
        self._pos_seq = np.empty([n_samples, self._maxlen], dtype=np.int8)
        self._pos_state = np.empty([n_samples, self._maxlen], dtype=np.int8)
        self._pos_stack = np.empty([n_samples, self._maxlen, self.max_rec],
                                   dtype=np.int8)
        self._pos_mask = np.empty([n_samples, self._maxlen, self.vocab_size],
                                  dtype=np.int8)
        self._neg_seq = np.empty([n_samples, self._maxlen], dtype=np.int8)

        loaded = set()
        count, idx = 0, 0
        with open(f"{load_from}/data.npz", "rb") as fp:
            while True:
                try:
                    data = pickle.load(fp)
                    count += 1
                    seq_repr = "s".join(str(i) for i in data[0])
                    if seq_repr in loaded:
                        continue
                    loaded.add(seq_repr)
                    self._pos_seq[idx] = data[0]
                    self._pos_state[idx] = data[1]
                    self._pos_stack[idx] = data[2]
                    self._pos_mask[idx] = data[3]
                    self._neg_seq[idx] = data[4]
                    idx += 1
                except EOFError:
                    break

        assert count == n_samples, f"{n_samples} stated, {idx} samples loaded"
        self._pos_seq = self._pos_seq[:idx]
        self._pos_state = self._pos_state[:idx]
        self._pos_stack = self._pos_stack[:idx]
        self._pos_mask = self._pos_mask[:idx]
        self._neg_seq = self._neg_seq[:idx]

        n_samples = idx
        np.random.seed(2021)
        self._all_idx = np.arange(n_samples)
        self._train_pos_idx = np.random.choice(self._all_idx, n_samples // 2)
        self._train_neg_idx = self._train_pos_idx
        # print(self._pos_seq.shape, self._pos_seq.dtype)
        # print(self._pos_state.shape, self._pos_state.dtype)
        # print(self._pos_stack.shape, self._pos_stack.dtype)
        # print(self._pos_mask.shape, self._pos_mask.dtype)

        print(f"{self._pos_seq.shape[0]} sampled loaded, "
              f"symbol size {self.vocab_size}, "
              f"state size {self.state_size}, "
              f"max recursion {self.max_rec}")

    @property
    def pad_token(self):
        return BasePDA.pad

    def dump(self, dir_name: str, beam_size: int):
        dir_name = os.path.join(os.path.dirname(__file__), dir_name)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        else:
            raise FileExistsError("log directory already exist")
        fp = open(dir_name + "/data.npz", "ab")

        np.random.seed(2021)
        n_samples = 0
        max_rec = 0
        for seq, state, stack in self._pda.enumerate(
                maxlen=self._maxlen, beam_size=beam_size):
            pickle.dump(
                [self._pad_seq(seq), self._pad_seq(state),
                 self._pad_stack(stack),
                 self._pad_mask(self._label_masks(seq)),
                 self._pad_seq(self._pda.corrupt_seq(seq))], fp)
            max_rec = max(max_rec, stack.shape[-1])
            n_samples += 1
            if n_samples % 1000 == 0:
                print(f"{n_samples} sequences created  ", end="\r")
        print(f"{n_samples} samples created, "
              f"max length {self._maxlen}, max recursion {max_rec}")
        fp.close()

        with open(f"{dir_name}/params.txt", "w") as fp:
            fp.write(f"{self.vocab_size}\n"
                     f"{max_rec}\n"
                     f"{self._maxlen}\n"
                     f"{n_samples}\n"
                     f"{self.state_size}\n"
                     f"{self._pda.__name__()}")
        print(f"Data dumped to {dir_name}")

    def _label_masks(self, seq: List[int]):
        mask = np.zeros([len(seq), self._pda.vocab_size])
        for step in range(1, len(seq)):
            for w in range(1, self._pda.vocab_size):  # skip start/end
                part_seq = copy(seq)[:step + 1]
                part_seq[step] = w
                if self._pda.accept(part_seq, partial=True):
                    mask[step, w] = 1
        return mask

    def _cls_iter(self, batch_size, pos_idx, neg_idx, shuffle):
        half_bs = batch_size // 2
        for start in np.arange(0, len(pos_idx), half_bs):
            pos = np.take(self._pos_seq,
                          pos_idx[start: start + half_bs], axis=0)
            neg = np.take(self._neg_seq,
                          neg_idx[start: start + half_bs], axis=0)
            label = np.ones([pos.shape[0] * 2])
            label[pos.shape[0]:] = 0
            yield np.concatenate([pos, neg], axis=0), label
        if shuffle:
            np.random.shuffle(pos_idx)
            np.random.shuffle(neg_idx)

    def _pad_seq(self, seq):
        return np.pad(seq, [0, self._maxlen - len(seq)], mode='constant',
                      constant_values=self._pda.pad).astype(np.int8)

    def _pad_stack(self, stack):
        res = np.ones([self._maxlen, stack.shape[1]]) * self._pda.pad_state
        res[:len(stack)] = stack
        return res.astype(np.int8)

    def _pad_mask(self, mask):
        res = np.zeros([self._maxlen, mask.shape[1]])
        res[:mask.shape[0]] = mask
        res[mask.shape[0]:, self._pda.end] = 1
        return res.astype(np.int8)

    def cls_train_samples(self, batch_size):
        yield from self._cls_iter(batch_size, self._train_pos_idx,
                                  self._train_neg_idx, shuffle=True)

    def cls_test_samples(self, batch_size):
        yield from self._cls_iter(batch_size, self._all_idx,
                                  self._all_idx, shuffle=False)

    def _lm_iter(self, batch_size, idx_pool, shuffle):
        for start in np.arange(0, len(idx_pool), step=batch_size):
            symbol = np.take(self._pos_seq,
                             idx_pool[start: start + batch_size], axis=0)
            mask = np.take(self._pos_mask,
                           idx_pool[start: start + batch_size], axis=0)
            yield symbol, mask
        if shuffle:
            np.random.shuffle(idx_pool)

    def lm_train_samples(self, batch_size):
        yield from self._lm_iter(batch_size, self._train_pos_idx, True)

    def lm_test_samples(self, batch_size):
        yield from self._lm_iter(batch_size, self._all_idx, False)

    def _fs_iter(self, batch_size, idx_pool, shuffle):
        for start in np.arange(0, len(idx_pool), batch_size):
            idx = idx_pool[start: start + batch_size]
            seq_batch = np.take(self._pos_seq, idx, axis=0)
            mask = np.take(self._pos_mask, idx, axis=0)
            state_batch = np.take(self._pos_state, idx, axis=0)
            stack_batch = np.take(self._pos_stack, idx, axis=0)
            yield seq_batch, state_batch, stack_batch, mask
        if shuffle:
            np.random.shuffle(idx_pool)

    def fs_train_samples(self, batch_size):
        yield from self._fs_iter(batch_size, self._train_pos_idx, True)

    def fs_test_samples(self, batch_size):
        yield from self._fs_iter(batch_size, self._all_idx, False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="palindrome")
    parser.add_argument("--k", type=int, help="vocab size")
    parser.add_argument("--m", type=int, help="max recursion")
    parser.add_argument("--n", type=int, help="max word length for palindrome")
    parser.add_argument("--maxlen", type=int)
    parser.add_argument("--beam_size", default=100000, type=int)
    command_args = parser.parse_args()

    def dump_data(args):
        if args.name == "anbn":
            pda = AnBn()
            log_dir = f"anbn_ml{args.maxlen}"
        elif args.name == "parity":
            pda = Parity()
            log_dir = f"parity_ml{args.maxlen}"
        elif args.name == "dyck":
            pda = DYCK(k=args.k, m=args.m)
            log_dir = f"DYCK_k{args.k}_m{args.m}_" \
                f"ml{args.maxlen}"
        elif args.name == "palindrome":
            pda = Palindrome(k=args.k, m=args.m, n=args.n)
            log_dir = f"Palindrome_k{args.k}_m{args.m}_" \
                f"n{args.n}_ml{args.maxlen}"
        else:
            raise AttributeError(f"Unknown PDA {args.name}")

        batcher = Batcher(pda=pda, maxlen=args.maxlen)
        batcher.dump(log_dir, args.beam_size)

    dump_data(command_args)
