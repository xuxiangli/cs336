"""
Byte-pair encoding tokenizer.
"""

import time
import os
import heapq
import pickle
import regex as re

from collections import defaultdict, Counter
from typing import Iterable, Iterator
from multiprocessing import Pool

from pretokenization_example import find_chunk_boundaries


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class TokenPair:

    def __init__(self, pair, freq, pos=None):
        self.pair = pair
        self.freq = freq
        self.pos = pos if pos else set()
        return


    def __lt__(self, other):
        if not isinstance(other, TokenPair):
            raise NotImplementedError
        if self.freq > other.freq:
            return True
        elif self.freq == other.freq and self.pair > other.pair:
            return True
        return False


def get_pretoken_freq_split(input_string: str):

    return Counter(input_string.split(' '))


def get_pretoken_freq_gpt2(input_string: str):

    return Counter(re.findall(PAT, input_string))


def pretokenize(
    input_path,
    start,
    end,
    special_tokens,
    pretoken_style: str = 'gpt2'
):

    with open(input_path, "rb") as file:
        file.seek(start)
        chunk = file.read(end - start).decode("utf-8")

    if special_tokens:
        strings = re.split(
            '|'.join(re.escape(token) for token in special_tokens), 
            chunk
        )
    else:
        strings = [chunk]
    del chunk

    word_freq = Counter()
    match pretoken_style:
        case "gpt2":
            for string in strings:
                word_freq.update(get_pretoken_freq_gpt2(string))

        case "split":
            for string in strings:
                word_freq.update(get_pretoken_freq_split(string))

        case _:
            raise ValueError(
                "pretoken_style should be either split or gpt2."
            )

    return word_freq


def pretokenize_text(
    input_text,
    special_tokens,
    pretoken_style: str = 'gpt2'
):

    if special_tokens:
        strings = re.split(
            '|'.join(re.escape(token) for token in special_tokens), 
            input_text
        )
    else:
        strings = [input_text]

    word_freq = Counter()
    match pretoken_style:
        case "gpt2":
            for string in strings:
                word_freq.update(get_pretoken_freq_gpt2(string))

        case "split":
            for string in strings:
                word_freq.update(get_pretoken_freq_split(string))

        case _:
            raise ValueError(
                "pretoken_style should be either split or gpt2."
            )

    return word_freq


def get_pretoken(counter):

    pretokens = {}
    idx_pretoken = 0
    for word, freq in counter.items():
        # tuple of bytes
        pretoken = tuple(b.to_bytes() for b in word.encode("utf-8"))  
        pretokens[idx_pretoken] = [freq, pretoken]
        idx_pretoken += 1

    return pretokens


def merge_bpe(pretokens, vocab_size, special_tokens, verbose=False):

    t0 = time.perf_counter()

    # Initialize vocab, merges, tokenpairs and heap
    vocab = {idx: idx.to_bytes() for idx in range(256)}  # in bytes
    if special_tokens:
        for idx, special_token in enumerate(special_tokens):
            vocab[256 + idx] = special_token.encode("utf-8")  # in bytes

    merges = []

    tokenpairs = {}
    for idx_pretoken in pretokens.keys():
        freq, pretoken = pretokens[idx_pretoken]
        for b1, b2 in zip(pretoken[:-1], pretoken[1:]):
            pair = (b1, b2)
            if pair in tokenpairs:
                tokenpairs[pair].freq += freq
                tokenpairs[pair].pos.add(idx_pretoken)
            else:
                tokenpairs[pair] = TokenPair(pair, freq, {idx_pretoken})

    tokenpairs_heap = [
        TokenPair(tokenpair.pair, tokenpair.freq) 
        for tokenpair in tokenpairs.values()
    ]
    heapq.heapify(tokenpairs_heap)

    # if verbose:
    #     print("len(pairs): ", len(tokenpairs))

    t1 = time.perf_counter()
    
    # Merge
    idx_start = 256 + len(special_tokens) if special_tokens else 256
    for idx in range(idx_start, vocab_size):

        # pop the heap until find true max tokenpair
        while True:
            max_tokenpair = heapq.heappop(tokenpairs_heap)
            if max_tokenpair.pair in tokenpairs:
                if max_tokenpair.freq == tokenpairs[max_tokenpair.pair].freq:
                    break
                else:
                    # if stale info, push the updated one back
                    heapq.heappush(
                        tokenpairs_heap,
                        TokenPair(
                            pair=max_tokenpair.pair,
                            freq=tokenpairs[max_tokenpair.pair].freq
                        )
                    )

        max_pair = max_tokenpair.pair
        new_token = max_pair[0] + max_pair[1]
        positions = tokenpairs[max_pair].pos

        # if verbose:
        #     print("pretokens: ", pretokens)
        #     print("pairs: ", {tokenpair.pair: tokenpair.freq for tokenpair in tokenpairs.values()})
        #     print("max_pair: ", max_pair)
        #     print("positions: ", positions)
        #     print("\n")

        vocab[idx] = new_token
        merges.append(max_pair)

        # Merge in pretokens, update other tokenpairs
        new_tokenpairs = {}
        for idx_pretoken in positions:
            freq, pretoken = pretokens[idx_pretoken]
            length = len(pretoken)
            new_pretoken = []

            # Go over pretokens to find match pairs
            i = 0
            while i < length:
                if i < length - 1 and pretoken[i:i+2] == max_pair:
                    new_pretoken.append(new_token)

                    # Reduce freq of pairs containing merged token
                    # skip if after a merged pair
                    if i > 0 and not (i > 1 and pretoken[i-2:i] == max_pair):
                        pair = pretoken[i-1:i+1]
                        tokenpairs[pair].freq -= freq
                        if tokenpairs[pair].freq == 0:
                            tokenpairs.pop(pair)

                    if i < length - 2:
                        pair = pretoken[i+1:i+3]
                        tokenpairs[pair].freq -= freq
                        if tokenpairs[pair].freq == 0:
                            tokenpairs.pop(pair)

                    i += 2
                else:
                    new_pretoken.append(pretoken[i])
                    i += 1

            new_pretoken = tuple(new_pretoken)
            pretokens[idx_pretoken] = [freq, new_pretoken]

            # Add new token pairs due to merges:
            new_length = len(new_pretoken)
            for i in range(new_length):
                if new_pretoken[i] != new_token:
                    continue

                # skip if after new_token
                if i > 0 and new_pretoken[i-1] != new_token:
                    pair = new_pretoken[i-1:i+1]
                    if pair in new_tokenpairs:
                        new_tokenpairs[pair].freq += freq
                        # if new_tokenpairs[pair].pos[-1] != idx_pretoken:
                        new_tokenpairs[pair].pos.add(idx_pretoken)
                    else:
                        new_tokenpairs[pair] = TokenPair(pair, freq, {idx_pretoken})

                if i < new_length-1:
                    pair = new_pretoken[i:i+2]
                    if pair in new_tokenpairs:
                        new_tokenpairs[pair].freq += freq
                        new_tokenpairs[pair].pos.add(idx_pretoken)
                    else:
                        new_tokenpairs[pair] = TokenPair(pair, freq, {idx_pretoken})

        for pair, tokenpair in new_tokenpairs.items():
            tokenpairs[pair] = tokenpair
            heapq.heappush(tokenpairs_heap, TokenPair(tokenpair.pair, tokenpair.freq))

        tokenpairs.pop(max_pair)

        if not tokenpairs:
            break

    t2 = time.perf_counter()

    return vocab, merges, t1 - t0, t2 - t1


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str] | None = None,
    pretoken_style: str = "gpt2",
    num_chunk: int = 16,
    verbose: bool = False
):
    """
        Train a tokenizer using byte-pair encoding.

        Parameters:
        -----------
        input_path: str
            Path to a text file with BPE tokenizer training data.
        vocab_size: int
            Size of the final vocabulary, including the initial byte
            vocabulary, items from merging, and special tokens.
        special_tokens: list[str]
            A list of special tokens as strings.
        
        Return:
        -------
        vocab: dict[int, bytes]
            The tokenizer vocabulary, a mapping fron token ID to token bytes
        merges: list[tuple[bytes, bytes]]
            A list of BPE merges produced from traning. Each element is
            a tuple of two bytes <token1>, <token2> to be merged.
    """

    t0 = time.perf_counter()

    # Split the text
    with open(input_path, "rb") as file:
        boundaries = find_chunk_boundaries(
            file, 
            num_chunk, 
            special_tokens[0].encode("utf-8") if special_tokens else b'\n'
        )

    # Allocate pretokenization works
    params = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        params.append([input_path, start, end, special_tokens, pretoken_style])

    # processes = min(num_proc, os.cpu_count()) if num_proc != 0 else os.cpu_count()
    with Pool(processes=os.cpu_count()) as pool:
        counters = pool.starmap(pretokenize, params)

    total_counter = Counter()
    for counter in counters:
        total_counter.update(counter)
    del counters

    pretokens = get_pretoken(total_counter)

    t1 = time.perf_counter()

    vocab, merges, t_heap, t_merge = merge_bpe(pretokens, vocab_size, special_tokens, verbose)

    if verbose:
        # print("Final pretokens: ", {word: freq for freq, word in pretokens.values()})
        print(f"pretokenize {t1-t0:.1f}s | heapify {t_heap:.1f}s | merges {t_merge:.1f}s.")
        print(f"Merges end with {len(vocab)} words in the vocabulary.")

    return vocab, merges


def train_bpe_text(
    input_text, 
    vocab_size: int,
    special_tokens: list[str] | None = None,
    pretoken_style: str = "gpt2",
    verbose: bool = False
):

    t0 = time.perf_counter()

    counter = pretokenize_text(input_text, special_tokens, pretoken_style)

    pretokens = get_pretoken(counter)
    del counter

    t1 = time.perf_counter()

    vocab, merges, t_heap, t_merge = merge_bpe(pretokens, vocab_size, special_tokens, verbose)

    if verbose:
        # print("Final pretokens: ", {word: freq for freq, word in pretokens.values()})
        print(f"pretokenize {t1-t0:.1f}s | heapify {t_heap:.1f}s | merges {t_merge:.1f}s.")
        print(f"Merges end with {len(vocab)} words in the vocabulary.")

    return vocab, merges


class Tokenizer:

    def __init__(self, vocab, merges, special_tokens=None, max_memory=10**6, pretoken_style='gpt2'):
        self.vocab = vocab.copy()
        self.inv_vocab = {token: idx for idx, token in vocab.items()}

        # pretoken of words met already
        self.memory = {}

        self.merges = merges
        self.inv_merges = {merge: idx for idx, merge in enumerate(merges)}

        if special_tokens:
            self.special_tokens = sorted(special_tokens, reverse=True)
        else:
            self.special_tokens = []

        max_id = max(self.vocab.keys()) + 1
        for token in self.special_tokens:
            token = token.encode("utf-8")
            if token not in self.inv_vocab:
                self.vocab[max_id] = token
                self.inv_vocab[token] = max_id
                max_id += 1

        self.max_memory = max_memory

        match pretoken_style:
            case 'gpt2':
                pass

            case _:
                raise ValueError("pretoken_style should be gpt2.")


    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        return cls(vocab, merges, special_tokens)


    def encode(self, text: str):
        if self.special_tokens:
            strings = re.split(
                '('+'|'.join(re.escape(token) for token in self.special_tokens)+')', 
                text
            )
        else:
            strings = [text]

        ids = []
        for string in strings:
            if string in self.special_tokens:
                ids.append(self.inv_vocab[string.encode("utf-8")])
            else:
                words = re.findall(PAT, string)
                for word in words:
                    pretoken = tuple(b.to_bytes() for b in word.encode("utf-8"))
                    pretoken = self._merge(pretoken)

                    for token in pretoken:
                        ids.append(self.inv_vocab[token])

        return ids


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            if self.special_tokens:
                strings = re.split(
                    '('+'|'.join(re.escape(token) for token in self.special_tokens)+')', 
                    text
                )
            else:
                strings = [text]

            for string in strings:
                if string in self.special_tokens:
                    yield self.inv_vocab[string.encode("utf-8")]
                else:
                    words = re.findall(PAT, string)
                    for word in words:
                        pretoken = tuple(b.to_bytes() for b in word.encode("utf-8"))
                        pretoken = self._merge(pretoken)

                        for token in pretoken:
                            yield self.inv_vocab[token]


    def decode(self, ids: list[int]):
        strings = [self.vocab[id] for id in ids]
        return b''.join(strings).decode("utf-8", errors="replace")


    def _merge(self, pretoken: tuple[bytes]):
        if pretoken in self.memory:
            return self.memory[pretoken]

        temp = pretoken

        pairs = {pair for pair in zip(pretoken[:-1], pretoken[1:]) if pair in self.inv_merges}
        while pairs:
            first_pair = min(pairs, key=lambda x: self.inv_merges[x])
            new_token = b''.join(first_pair)

            pairs.remove(first_pair)

            length = len(pretoken)
            new_pretoken = []

            # Go over pretokens to find match pairs
            i = 0
            while i < length:
                if i < length - 1 and pretoken[i:i+2] == first_pair:

                    if i > 0:
                        prefix = (new_pretoken[-1], new_token)
                        if prefix in self.inv_merges:
                            pairs.add(prefix)

                    if i < length - 2 and \
                        not (i < length - 3 and pretoken[i+2:i+4] == first_pair):
                        suffix = (new_token, pretoken[i+2])
                        if suffix in self.inv_merges:
                            pairs.add(suffix)

                    new_pretoken.append(new_token)
                    i += 2
                else:
                    new_pretoken.append(pretoken[i])
                    i += 1
            pretoken = tuple(new_pretoken)

        if len(self.memory) > self.max_memory:
            self.memory = {}
        self.memory[temp] = pretoken
    
        return pretoken