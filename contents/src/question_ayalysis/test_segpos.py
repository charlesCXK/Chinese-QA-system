# -*- coding: utf-8 -*-
'''
对测试问题进行词性标注
'''

import sys
import os
import jienba
import jieba.posseg as psg
from pyltp import Postagger
from pyltp import Segmentor
#sys.path.append("..")

# 根据分词结果进行标注
def LTP_postag(line):
    LTP_DATA_DIR = '../ltp_data'        # ltp model的目录
    pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')  # 词性标注模型路径，模型名称为`pos.model`
    seg_model_path = os.path.join(LTP_DATA_DIR, "cws.model")

    segmentor = Segmentor()
    postagger = Postagger()             # 初始化实例

    segmentor.load(seg_model_path)
    postagger.load(pos_model_path)      # 加载模型

    words = segmentor.segment(line)     # 分词
    postags = postagger.postag(words)  # 词性标注

    segmentor.release()
    postagger.release()                 # 释放模型

    for i in range(len(words)):
        words[i] += ('/' + postags[i])
    return words


if __name__ == '__main__':
    with open('data/final_test.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    question = lines    
    for i in range(len(lines)):
        print("handling {} of {}".format(i+1, len(lines)))
        question[i] = LTP_postag(question[i])

    # 写文件
    with open('data/final_test_pos.txt', 'w', encoding='utf-8') as f:
        for i in range(len(lines)):
            for ele in question[i]:
                f.write(ele + '\t')
            f.write('\n')
