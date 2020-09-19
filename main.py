import os
import pickle
from co_occurrence.co_occurrence_calculator import CoOccurrenceCalculator
from co_occurrence.co_occurrence_network import CoOccurrenceNetwork

def main(model_filepath=None):
    curdir = os.path.dirname(os.path.abspath(__file__))
    corpus_filepath = os.path.join(curdir, 'corpus', 'commits.csv')

    if not os.path.exists(corpus_filepath):
        print('No commit message file, first crawl commit message')
        exit()

    co_occurrences = None
    if not os.path.exists(model_filepath):
        coc = CoOccurrenceCalculator(tokenizer_model='en_core_web_lg')
        co_occurrences = coc.calculate(corpus_filepath)
        with open(model_filepath, 'wb') as f:
            pickle.dump(coc, f)
        print(f'Saved CoOccurenceCalculator > {model_filepath}')
    else:
        with open(model_filepath, 'rb') as f:
            coc = pickle.load(f)
            co_occurrences = coc._output()
        print(f'Loaded CoOccurrenceCalculator < {model_filepath}')

    rows = ['\t'.join(['vp', 'vp_freq', 'np', 'np_freq', 'vp_np_freq'])]
    for (verb, v_freq), childs in sorted(co_occurrences.items(), key=lambda x: x[0][1], reverse=True):
        for noun, n_freq, co_freq in sorted(childs, key=lambda x: x[2], reverse=True):
            rows.append(f'{verb}\t{v_freq}\t{noun}\t{n_freq}\t{co_freq}')

    with open('co_occurences.csv', 'w') as f:
        f.write('\n'.join(rows))

    # Graph出力
    graph_filename = 'co_occurrence_graph.png'
    CoOccurrenceNetwork(co_occurrences).draw(graph_filename, parent_node_count=30, child_node_count=5)


if __name__ == '__main__':
    main('commit_msg_vp_np_6M.model')
