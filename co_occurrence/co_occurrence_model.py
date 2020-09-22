import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from .tokenizer_spacy import Tokenizer

class CoOccurrenceModel(object):
    def __init__(self, tokenizer_model='en_core_web_lg'):
        self.id2vp = {}
        self.vp2id = {}
        self.id2np = {}
        self.np2id = {}
        self.id2word = {}
        self.word2id = {}
        self.wid2pid = defaultdict(list)
        self.vp_freq = defaultdict(int)
        self.np_freq = defaultdict(int)
        self.word_freq = defaultdict(int)
        self.vp_np_freq = defaultdict(int)
        self.co_vp_np = defaultdict(list)
        self.co_np_vp = defaultdict(list)

    def _get_dics(self, pos):
        if pos == 'VP':
            return self.id2vp, self.vp2id, self.vp_freq
        else:
            return self.id2np, self.np2id, self.np_freq

    def _format(self):
        for (vid, nid), freq in self.vp_np_freq.items():
            self.co_vp_np[self.id2vp[vid]].append(
                [
                    self.vp_freq[vid],
                    self.id2np[nid],
                    self.np_freq[nid],
                    freq
                ]
            )

            self.co_np_vp[self.id2np[nid]].append(
                [
                    self.np_freq[nid],
                    self.id2vp[vid],
                    self.vp_freq[vid],
                    freq
                ]
            )

        dicts = [self.co_vp_np, self.co_np_vp]
        for d in dicts:
            for phrase, co_items in d.items():
                d[phrase] = sorted(co_items, key=lambda x: x[3], reverse=True)
