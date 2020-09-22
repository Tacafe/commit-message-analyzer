import os
import math
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

class CoOccurrenceNetwork(object):
    def __init__(self, co_occurrences):
        self.co_occurrences = co_occurrences
        self.G = nx.Graph()

        self.parent_nodes = []
        self.parent_word_frequencies = []
        self.child_word_frequencies = []
        self.weights = []

        self.size_rate = 15000
        self.weight_rate = 0.01

        self.range_of_parent_node_size = [0.5, 0.5]
        self.range_of_chilid_node_size = [0.3, 0.5]
        self.range_of_edge_weight = [0.1, 0.5]

    def draw(self, output_filepath, parent_node_count=2, child_node_count=30):
        print('Drawing graph...')
        self._to_graph(parent_node_count, child_node_count)
        self._normalize()
        self._output(output_filepath)

    def _to_graph(self, parent_node_count, child_node_count):
        draw_target_co = sorted(self.co_occurrences.items(), key=lambda x: x[0][1], reverse=True)[:parent_node_count]

        for (parent_word, parent_word_freq), childs in draw_target_co:
            self.G.add_node(parent_word, node_color='r', node_size=parent_word_freq)
            self.parent_nodes.append(parent_word)
            self.parent_word_frequencies.append(float(parent_word_freq))

            for [child_word, child_word_freq, score] in childs[:child_node_count]:
                label = child_word
                self.G.add_node(label, node_color='y', node_size=child_word_freq)
                weight = -1 * math.log(score)
                self.G.add_edge(parent_word, label, weight=weight)
                self.child_word_frequencies.append(float(child_word_freq))
                self.weights.append(float(weight))

    def _normalize(self):
        for node, attr in self.G.nodes.items():
            self.G.nodes[node]['node_size'] = self._normalized_node_size(node, int(attr['node_size']))

        for edge, attr in self.G.edges.items():
            self.G.edges[edge]['weight'] = self._normalized_edge_weight(float(attr['weight']))

    def _normalized_node_size(self, node, node_size):
        word_frequencies = self.parent_word_frequencies if node in self.parent_nodes else self.child_word_frequencies
        range_of_node_size = self.range_of_parent_node_size if node in self.parent_nodes else self.range_of_chilid_node_size

        min_size = min(word_frequencies)
        max_size = max(word_frequencies)
        min_range = min(range_of_node_size)
        max_range = max(range_of_node_size)

        return self.size_rate * self._min_max_normalization(node_size, max_size, min_size, max_range, min_range)

    def _normalized_edge_weight(self, weight):
        min_score = min(self.weights)
        max_score = max(self.weights)
        min_range = min(self.range_of_edge_weight)
        max_range = max(self.range_of_edge_weight)
        return self.weight_rate * self._min_max_normalization(weight, max_score, min_score, max_range, min_range)

    def _min_max_normalization(self, value, max_value, min_value, max_norm, min_norm):
        if max_value - min_value == 0: return max_norm
        return ((value - min_value)/(max_value - min_value)) * (max_norm - min_norm) + min_norm

    def _output(self, output_filepath):
        plt.figure(figsize=(15, 15))
        fixed = np.random.seed(0)
        pos = nx.spring_layout(self.G, k=0.1, fixed=fixed)

        # 正規化されたnode_sizeを描画サイズに拡大
        node_size = [attr['node_size'] for node, attr in self.G.nodes(data=True)]

        node_color = [attr['node_color'] for node, attr in self.G.nodes(data=True)]

        nx.draw_networkx_nodes(self.G,
                               pos,
                               node_size=node_size,
                               node_color=node_color,
                               alpha=0.7)

        nx.draw_networkx_labels(self.G,
                                pos,
                                # fontsize=50,
                                font_family='IPAexGothic',
                                font_weight='bold')

        # edge_width = [attr['weight'] for (p, c, attr) in self.G.edges(data=True)]
        nx.draw_networkx_edges(self.G,
                               pos,
                               alpha=0.9,
                               edge_color='darkgrey')

        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_filepath, bbox_inches='tight')
        print(f' > Output "{output_filepath}" to {output_filepath}')

    def _output_by_pagerank(self, output_filepath):
        plt.figure(figsize=(15, 15))
        # nodeの配置方法の指定
        # ※これを指定しないと描画のたびにノードの集合の位置が変わる
        seed = 0
        fixed = np.random.seed(seed)
        pos = nx.spring_layout(self.G, k=0.5, fixed=fixed)
        # nodeの大きさと色をページランクアルゴリズムによる重要度により変える
        pr = nx.pagerank(self.G)
        nx.draw_networkx_nodes(
            self.G,
            pos,
            node_color=list(pr.values()),
            cmap=plt.cm.rainbow,
            alpha=0.7,
            node_size=[100000*v for v in pr.values()])
        # 日本語ラベルの設定
        nx.draw_networkx_labels(self.G,
                                pos,
                                fontsize=50,
                                font_family='IPAexGothic',
                                font_weight='bold')
        # エッジ太さを係数により変える
        edge_width = [d['weight'] * 10 for (u, v, d) in self.G.edges(data=True)]
        nx.draw_networkx_edges(self.G, pos, alpha=0.9, edge_color='darkgrey', width=edge_width)

        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_filepath, bbox_inches='tight')
        print(f' > Output "{output_filepath}" to {output_filepath}')
