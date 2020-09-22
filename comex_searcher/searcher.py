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
        phrase = word.lower()
        nps_with_vp = self._get_co(word, "VP", top_n)
        vps_with_np = self._get_co(word, "NP", top_n)
        return {
            "asVP": nps_with_vp,
            "asNP": vps_with_np
            }

    def get_phrases_include_word(self, word, top_n=100):
        word = word.lower()
        vps = self._get_phrases(word, "VP", top_n)
        nps = self._get_phrases(word, "NP", top_n)
        return {
            "asVP": vps,
            "asNP": nps
        }

    def get_phrases_include_word_and_co(self, word, top_n=10, threshold=0):
        word = word.lower()
        vps = self._get_phrases(word, "VP", 10)
        nps = self._get_phrases(word, "NP", 10)

        co = {}
        nps_with_vp = defaultdict(list)
        for vp in vps:
            items = [child[1] for child in sorted(self.co_model.co_vp_np[vp], key=lambda x: x[2], reverse=True)[:top_n]]
            nps_with_vp[vp] += items

        vps_with_np = defaultdict(list)
        for np in nps:
            items = [child[1] for child in sorted(self.co_model.co_np_vp[np], key=lambda x: x[2], reverse=True)[:top_n]]
            vps_with_np[np] += items

        co["asVP"] = nps_with_vp
        co["asNP"] = vps_with_np
        return co

    def _get_co(self, phrase, pos, top_n=10, threshold=0):
        co_dic = self.co_model.co_vp_np if pos == 'VP' else self.co_model.co_np_vp
        return [child[1] for child in sorted(co_dic[phrase], key=lambda x: x[2], reverse=True)[:top_n] if phrase in co_dic]

    def _get_phrases(self, word, pos, top_n=10, threshold=0):
        if not (word, pos) in self.co_model.word2id: return []

        wid = self.co_model.word2id[word, pos]
        pids = list(set(self.co_model.wid2pid[wid]))
        id2p, p2id, p_freq = self.co_model._get_dics(pos)
        phrases =  [[id2p[pid], p_freq[pid]] for pid in pids if pid in p_freq and p_freq[pid] > threshold]
        return [phrase for [phrase, freq] in sorted(phrases, key=lambda x: x[1], reverse=True)[:top_n]]

def debug_get_co():
    s = Searcher('commit_msg_vp_np_6M.model')
    phrases = s.get_co('test')
    for asPOS, phrases in phrases.items():
        print(asPOS, ', '.join(phrases))

def debug_phrases():
    s = Searcher('commit_msg_vp_np_6M.model')
    phrases = s.get_phrases_include_word('document')
    for asPOS, phrases in phrases.items():
        print(asPOS, ', '.join(phrases))

def debug():
    s = Searcher('commit_msg_vp_np_6M.model')
    co = s.get_co('document')
    for asPOS, co_phrases in co.items():
        print(asPOS)
        for phrase, children in co_phrases.items():
            print(phrase)
            print(', '.join(children))

if __name__ == '__main__':
    debug_get_co()
    # debug_phrases()
