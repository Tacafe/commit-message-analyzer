import os
import pickle
from co_occurrence.co_occurrence_model_builder import CoOccurrenceModelBuilder
from co_occurrence.co_occurrence_model import CoOccurrenceModel
from co_occurrence.co_occurrence_network import CoOccurrenceNetwork

def main(model_filepath=None):
    curdir = os.path.dirname(os.path.abspath(__file__))
    corpus_filepath = os.path.join(curdir, 'corpus', 'commits.csv')

    if not os.path.exists(corpus_filepath):
        print('No commit message file, first crawl commit message')
        exit()

    co_model = None
    if not os.path.exists(model_filepath):
        cmb = CoOccurrenceModelBuilder(tokenizer_model='en_core_web_lg')
        co_model = cmb.build(corpus_filepath)
        with open(model_filepath, 'wb') as f:
            pickle.dump(co_model, f)
        print(f'Saved CoOccurenceCalculator > {model_filepath}')
    else:
        with open(model_filepath, 'rb') as f:
            co_model = pickle.load(f)
        print(f'Loaded CoOccurrenceCalculator < {model_filepath}')

    # Graph出力
    # graph_filename = 'co_occurrence_graph.png'
    # CoOccurrenceNetwork(model).draw(graph_filename, parent_node_count=30, child_node_count=5)


if __name__ == '__main__':
    main('commit_msg_vp_np_6M.model')
