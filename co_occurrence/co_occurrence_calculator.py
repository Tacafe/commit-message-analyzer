import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from .tokenizer_spacy import Tokenizer

class CoOccurrenceCalculator(object):
    def __init__(self, tokenizer_model='en_core_web_lg'):
        self.tokenizer = Tokenizer(model=tokenizer_model)
        self.corpus_phrases = []
        self.id2vp = {}
        self.vp2id = {}
        self.id2np = {}
        self.np2id = {}
        self.id2word = {}
        self.word2id = {}
        self.wid2pid = defaultdict(list)
        self.vp_id = -1
        self.np_id = -1
        self.word_id = -1
        self.vp_freq = defaultdict(int)
        self.np_freq = defaultdict(int)
        self.word_freq = defaultdict(int)
        self.vp_np_freq = defaultdict(int)

    def calculate(self, corpus_path):
        print('Extracting VP and NP...')
        for i, line in tqdm(enumerate(self._gen_file(corpus_path))):
            phrases = self.tokenizer.chunknize(line)
            for phrase, pos in phrases:
                self._register_phrase(phrase, pos)
            vps = [self.vp2id[phrase] for phrase, pos in phrases if pos == "VP"]
            nps = [self.np2id[phrase] for phrase, pos in phrases if pos == "NP"]
            self.corpus_phrases.append([vps, nps])

            # if i > 4000:
            #     break

        n = 2
        pos = "VP"
        print(f'Get more than {n} times occured {pos}')
        important_vps = self._get_more_than_n_phrase(pos, n)

        print('Caluclating co-occurrence...')
        for vps, nps in tqdm(self.corpus_phrases):
            v_imp = [vp for vp in vps if vp in important_vps]
            if len(v_imp) == 0: continue
            for vp in v_imp:
                for np in nps:
                    self.vp_np_freq[vp, np] += 1

        return self._output()

    def _gen_file(self, file_path):
        with open(file_path, 'r') as f:
            for line in f:
                yield line.lower()

    def _register_phrase(self, phrase, pos):
        p2id, id2p, freq_dic = self._get_dics(pos)
        if not phrase in p2id: p2id[phrase] = self._get_max_id(pos)
        pid = p2id[phrase]
        if not id in id2p: id2p[pid] = phrase
        freq_dic[pid] += 1

        for word in phrase.split(' '):
            if not (word, pos) in self.word2id:
                self.word_id += 1
                self.word2id[word, pos] = self.word_id
                self.id2word[self.word_id] = [word, pos]
            wid = self.word2id[word, pos]
            self.word_freq[wid] += 1
            self.wid2pid[wid].append(pid)

    def _get_more_than_n_phrase(self, pos, n):
        _, _, dic = self._get_dics(pos)
        filtered = dict(filter(lambda x: x[1] > n, dic.items()))
        return [vid for vid, freq in sorted(filtered.items(), key=lambda x: x[1], reverse=True)]

    def _get_max_id(self, pos):
        if pos == 'VP':
            self.vp_id += 1
            return self.vp_id
        else:
            self.np_id += 1
            return self.np_id

    def _get_dics(self, pos):
        if pos == 'VP':
            return self.vp2id, self.id2vp, self.vp_freq
        else:
            return self.np2id, self.id2np, self.np_freq

    def _output(self):
        '''
        returns list(CoOccurrence)
        '''
        co_occurrences = defaultdict(list)

        for (vid, nid), freq in tqdm(self.vp_np_freq.items()):
            co_occurrences[self.id2vp[vid], self.vp_freq[vid]].append(
                [
                    self.id2np[nid],
                    self.np_freq[nid],
                    freq
                ]
            )
        return co_occurrences
