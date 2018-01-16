# -*- coding: utf-8 -*-
'''
处理标注好词性的问题，获得原始问题
'''
import jieba
import jieba.posseg as pseg

def get_raw(infile, outfile):
	with open(outfile, 'w', encoding='utf-8') as wr:
		with open(infile, 'r', encoding='utf-8') as f:
			lines = f.readlines()
			for line in lines:
				new_s = []
				sp = line.split()
				for ele in sp:
					new_s.append(ele.split('/')[0])
				for ele in new_s:
					wr.write(ele + '\t')
				wr.write('\n')

def jieba_pos(infile, outfile):
	with open(outfile, 'w', encoding='utf-8') as wr:
		with open(infile, 'r', encoding='utf-8') as f:
			lines = f.readlines()
			for line in lines:
				res = pseg.cut(line.strip())
				new_s = []
				for w in res:
					new_s.append(w.word+'/'+w.flag)
				for ele in new_s:
					wr.write(ele + '\t')
				wr.write('\n')

if __name__ == '__main__':
	get_raw('final_test_pos.txt', 'final_test_seg.txt')
	#jieba_pos('train-questions-raw.txt', 'train-questions.txt')