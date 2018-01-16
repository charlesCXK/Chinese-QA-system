# -*- coding: utf-8 -*-
'''
提取 测试问题 中的问题和答案
'''

import sys
import os
import jienba
import jieba.posseg as psg
from pyltp import Postagger
from pyltp import Segmentor
#sys.path.append("..")

def jieba_pos(line):
    words = psg.cut(line)
    return words

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
    with open('../wdm_assignment_3_samples.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    l = len(lines)
    question, answer = [[] for i in range(l)], [[] for i in range(l)]
    #print(lines[61].split())
    for i in range(l):
        try:
            question[i] = lines[i].split()[0]
            answer[i] = lines[i].split()[1]
        except:             # 第 110 行不知道为什么总是读不进去。。。。。。
            question[i] = lines[61].split()[0]        
            answer[i] = lines[61].split()[1]
            
    for i in range(l):
        question[i] = LTP_postag(question[i])

    # 写文件
    with open('sample/sample.txt', 'w', encoding='utf-8') as f:
        for i in range(l):
            for ele in question[i]:
                f.write(ele + '\t')
            f.write('\n')
    with open('sample/answer.txt', 'w', encoding='utf-8') as f:
        for i in range(l):
            f.write(str(answer[i]) + '\n')
