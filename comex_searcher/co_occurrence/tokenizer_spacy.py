import spacy, ginza
from spacy.matcher import Matcher
from spacy.util import filter_spans
from collections import defaultdict
from itertools import *

class Tokenizer():
    def __init__(self, model='ja_ginza'):
        self.nlp = spacy.load(model)
        self.model = model

        self.vp_matcher = Matcher(self.nlp.vocab)
        vp_pattern = [
            {'POS': 'ADP', 'OP': '*'},
            {'POS': 'VERB', 'OP': '?'},
            {'POS': 'ADV', 'OP': '*'},
            {'POS': 'AUX', 'OP': '*'},
            {'POS': 'VERB', 'OP': '+'}
        ]
        self.vp_matcher.add("VP", None, vp_pattern)

        self.np_matcher = Matcher(self.nlp.vocab)
        np_pattern = [
            {'POS': 'DET', 'OP': '*'},
            {'POS': 'ADJ', 'OP': '*'},
            {'POS': 'NOUN', 'OP': '*'},
            {'POS': 'NOUN', 'OP': '?'}
        ]
        self.np_matcher.add("NP", None, np_pattern)

    def tokenize(self, sentence, attributes=['lemma_', 'pos_', 'dep_']):
        if attributes == []: return self.nlp(sentence)
        return [[getattr(token, attr) for attr in attributes] for token in self.nlp(sentence)]

    def chunknize(self, sentence):
        if self.model == 'ja_ginza':
            return self._ginza_bunsetu(sentence)
        else:
            return self._spacy_chunk(sentence)

    def _ginza_bunsetu(self, sentence):
        return [(chunk.text, chunk.label_) for chunk in ginza.bunsetu_spans(self.nlp(sentence))]

    def _spacy_chunk(self, sentence):
        # call the matcher to find matches
        doc = self.nlp(sentence)
        return self._get_matched(doc, self.np_matcher, 'NP') + self._get_matched(doc, self.vp_matcher, 'VP')

    def _get_matched(self, doc, matcher, tag):
        matches = matcher(doc)
        # phrases = [[doc[start:end], tag] for _, start, end in matches if doc[start:end]]
        g = groupby(matches, lambda prop: prop[1])
        max_id = -1
        previous_appended = None
        unique = []
        for k, v in g:
            l = list(v)
            candidate = l[-1]
            if candidate[2] > max_id:
                unique.append(candidate)
                if previous_appended and candidate[1] == previous_appended[1]:
                    unique.remove(previous_appended)
                max_id = candidate[2]
                previous_appended = candidate

        phrases = [(doc[start:end].text, tag) for match_id, start, end in unique if not doc[start:end].text == ""]
        return phrases

def debug():
    t = Tokenizer(model='en_core_web_lg')
    sentence = 'responsive web design is very hard and important'
    print(t.chunknize(sentence))

if __name__ == '__main__':
    debug()
