#--*-- coding:utf-8 --*--
import sys
import re
import os
import logging
import jieba
import multiprocessing


# 提取出中文
def exact_chinese(infile, outfile):
	file_in = open(infile, 'r', encoding='utf-8')
	file_out = open(outfile, 'w', encoding='utf-8')

	while(1):
		line = file_in.readline()
		if not line:
			break
		tmp = re.findall('[\n\u4e00-\u9fa5]', line)
		if tmp:
			file_out.write("".join(tmp))
	file_in.close()
	file_out.close()

# 结巴分词，精确模式
def separate_words(infile, outfile):
	file_in = open(infile, 'r', encoding='utf-8')
	file_out = open(outfile, 'w', encoding='utf-8')
	while(1):
		line = file_in.readline()
		if not line:
			continue
		seg_list = jieba.cut(line.strip(), cut_all=False)
		file_out.write(' '.join(seg_list) + '\n')
	file_in.close()
	file_out.close()

# 根据分词结果建立倒排索引，方便查找
def reverse_index(infile, outfile):
	index_dict = {}		# 倒排索引的字典，value是一个列表，记录了出现这个词的行数
	line_number = 0		# 记录行数
	file_in = open(infile, 'r', encoding='utf-8')
	file_out = open(outfile, 'w', encoding='utf-8')
	while(1):
		line = file_in.readline()
		line_number += 1
		if line_number%10000 == 0:
			print("reverse_index: has processed {}W/30W".format(line_number/10000))
		if not line:
			break
		seg = line[0:-1].split(' ')
		for word in seg:
			if word in index_dict:			# python3中has_key()函数换成了 "in"
				if line_number == index_dict[word][-1][0]:
					index_dict[word][-1][1] += 1
				else:
					index_dict[word].append([line_number,1])
			else:
				index_dict[word] = []
				index_dict[word].append([line_number,1])
		#print(index_dict["欧几里得"])

	for key,value in index_dict.items():
		file_out.write(key + ' ')
		for ele in value:
			file_out.write(str(ele[0]) + ' ' + str(ele[1]) + ' ')
		file_out.write('\n') 
	file_in.close()
	file_out.close()

def handle_raw(infile):
	with open(infile, 'r', encoding='utf-8') as f:
		raw = f.read()
	raw = raw.replace('\n', ' ')
	pat = re.compile(r'<doc.*?>(.*?)</doc>')
	res = pat.findall(raw)
	l = len(res)

	with open('wiki_chs.txt', 'w', encoding='utf-8') as f:
		for i in range(l):
			f.write(str(res[i]) + '\n')

if __name__ == "__main__":
	separate_words('wiki_jian.txt', 'wiki_separate.txt')
	reverse_index('wiki_separate.txt', 'wiki_index.txt')
	# 对测试问题分词
	#separate_words('../question_ayalysis/data/final_test.txt', '../question_ayalysis/data/final_test_seg.txt')
	print("hello!")