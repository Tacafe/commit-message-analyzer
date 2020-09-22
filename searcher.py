import os
import pickle
from collections import defaultdict

class Searcher():
    def __init__(self, model_filepath):
        self.d = defaultdict(list)
        self.co_model = self._from_model(model_filepath)

    def _from_model(self, model_filepath):
        print(f'Loading {model_filepath}')
        with open(model_filepath, 'rb') as f:
            return pickle.load(f)

    def get_co(self, word, top_n=10, threshold=0):
        word = word.lower()
        vps = self._get_phrases(word, "VP")
        nps = self._get_phrases(word, "NP")

        co = defaultdict(list)
        nps_with_vp = []
        for vp in vps:
            items = [child[1] for child in sorted(self.co_model.co_vp_np[vp], key=lambda x: x[2], reverse=True)[:top_n]]
            nps_with_vp += items

        vps_with_np = []
        for np in nps:
            items = [child[1] for child in sorted(self.co_model.co_np_vp[np], key=lambda x: x[2], reverse=True)[:top_n]]
            vps_with_np += items

        co["asVP"] = list(set(nps_with_vp))
        co["asNP"] = list(set(vps_with_np))
        for k, co_phrases in co.items():
            print(k, ','.join(co_phrases))

        return co

    def _get_phrases(self, word, pos, top_n=10, threshold=0):
        if not (word, pos) in self.co_model.word2id: return []

        wid = self.co_model.word2id[word, pos]
        pids = list(set(self.co_model.wid2pid[wid]))
        id2p, p2id, p_freq = self.co_model._get_dics(pos)
        phrases =  [[id2p[pid], p_freq[pid]] for pid in pids if pid in p_freq and p_freq[pid] > threshold]
        return [phrase for [phrase, freq] in sorted(phrases, key=lambda x: x[1], reverse=True)[:top_n]]

def debug():
    s = Searcher('commit_msg_vp_np_6M.model')
    s.get_co('clarify')

if __name__ == '__main__':
    debug()
