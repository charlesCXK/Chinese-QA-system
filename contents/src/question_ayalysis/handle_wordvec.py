# -*- coding: utf-8 -*-
'''
训练词向量
'''
import sys
sys.path.append("..")
import re
import os
import logging
import multiprocessing

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

'''
训练词向量
维度数：400，迭代次数：15，8核CPU并行处理
'''
def word_to_vec(file_in, file_out1, file_out2):
	logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
	logging.root.setLevel(level=logging.INFO)
	model = Word2Vec(LineSentence(file_in), size=4, window=5, min_count=1, workers=multiprocessing.cpu_count(), iter=100)
	model.save(file_out1)
	model.wv.save_word2vec_format(file_out2, binary=False)


if __name__ == '__main__':
	word_to_vec('data/all_raw.txt', 'model/word.model', 'model/word.vector')