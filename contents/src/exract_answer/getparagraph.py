# encoding=utf-8
import math

totalline = 1348439

def calcidf(infile):
	wordindex = {}
	wordidf = {}
	fin = open(infile, 'r')
	t = 0
	for line in fin.readlines():
		print t
		t = t+1
		terms = line.strip().split()
		word = terms[0]
		wordindex[word] = []
		i = 1
		while i+1 < len(terms):
			wordindex[word].append([int(terms[i]), int(terms[i+1])])
			i = i + 2
		wordidf[word] = math.log(totalline / (len(terms)*0.5))
	return wordindex, wordidf


#with open('wiki_jian.txt', 'r') as fin:
#	totalline = 0
#	for line in fin.readlines():
#		totalline = totalline + 1
#	print totalline		# 300644
invertedindex = 'wiki_index.txt'
wordindex, wordidf = calcidf(invertedindex)
sentencetfidf = {}
answerlines = []
with open('final_q_out.txt', 'r') as fques:
	t = 0
	for question in fques.readlines():
		print t
		t = t+1
		answerlines.append([])
		terms = question.strip().split('\t')
		totalnoun = 0
		totalnoun2 = 0
		sentencetotnoun = {}
		sentencetotnoun2 = {}
		for i in range(3):
			k = terms[i].find('/')
			word = terms[i][0:k]
			if not wordindex.has_key(word):
				continue
			if terms[i][k+1] == 'n':
				print word
				if k+2 < len(terms[i]) and terms[i][k+2] in "rstz":
					totalnoun = totalnoun + 1
				totalnoun2 = totalnoun2 + 1
				for j in wordindex[word]:
					if sentencetotnoun2.has_key(j[0]):
						sentencetotnoun2[j[0]] = sentencetotnoun2[j[0]] + 1
					else:
						sentencetotnoun2[j[0]] = 1
					if k+2 < len(terms[i]) and terms[i][k+2] in "rstz":
						if sentencetotnoun.has_key(j[0]):
							sentencetotnoun[j[0]] = sentencetotnoun[j[0]] + 1
						else:
							sentencetotnoun[j[0]] = 1
						if sentencetfidf.has_key(j[0]):
							sentencetfidf[j[0]] += wordidf[word] * 10 * j[1]
						else:
							sentencetfidf[j[0]] = wordidf[word] * 10 * j[1]
					else: 
						if sentencetfidf.has_key(j[0]):
							sentencetfidf[j[0]] += wordidf[word] * j[1]
						else:
							sentencetfidf[j[0]] = wordidf[word] * j[1]
		print str(totalnoun)
		if totalnoun2 > 0:
			for key in sentencetotnoun2:
				if sentencetotnoun2[key] == totalnoun2:
					answerlines[-1].append([sentencetfidf[key], key])
		else:
			for key in sentencetotnoun:
				if sentencetotnoun[key] == totalnoun:
					answerlines[-1].append([sentencetfidf[key], key])
		answerlines[-1].sort(key=lambda func: -func[0])
with open('final_answerlines.txt', 'w') as fout, open('wiki_separate.txt', 'r') as fin:
	wikilines = fin.readlines()
	print len(wikilines)
	for i in range(len(answerlines)):
		print i
		fout.write(str(i) + "：\n")
		for j in range(min(5, len(answerlines[i]))):
			fout.write(wikilines[int(answerlines[i][j][1])-1])
		fout.write('\n')
