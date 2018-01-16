# -*- coding: utf-8 -*-
import math

def loadsynonym():
	synonym = {}
	sum_cat = {}
	f = open('synonym.txt', 'r')
	for line in f.readlines():
		terms = line.strip().split()
		if len(terms) == 1 and terms[0] == '':
			break
		cat = terms[0]
		for i in range(len(cat)):
			s = cat[0:i]
			if sum_cat.get(s) is None:
				sum_cat[s] = 1
			else:
				sum_cat[s] += 1
		for i in range(1, len(terms)):
			if synonym.get(terms[i]) is None:
				synonym[terms[i]] = [terms[0]]
			else:
				synonym[terms[i]].append(terms[0])
	return synonym, sum_cat

def calc(sa, sb):
	# Sim(A,B) = a*cos(n*pi/180)*[(n-k+1)/n]
	a = 0.45
	b = 0.65
	c = 0.8
	d = 0.96
	e = 0.5
	f = 0.1
	g = 20
	if sa[0] != sb[0]:
		return f
	if sa[1] != sb[1]:
		n = sum_cat[sa[0: 1]]
		k = abs(ord(sa[1]) - ord(sb[1]))
		return a * math.cos(n * math.pi * math.pi / 180 / 180) * (n - k + 1) / n
	if sa[2] != sb[2] or sa[3] != sb[3]:
		n = sum_cat[sa[0: 2]]
		k = abs(int(sa[2: 4]) - int(sb[2: 4]))
		return b * math.cos(n * math.pi * math.pi / 180 / 180) * (n - k + 1) / n
	if sa[4] != sb[4]:
		n = sum_cat[sa[0: 4]]
		k = abs(ord(sa[4]) - ord(sb[4]))
		return c * math.cos(n * math.pi * math.pi / 180 / 180) * (n - k + 1) / n
	if sa[5] != sb[5] or sa[6] != sb[6]:
		n = sum_cat[sa[0: 5]]
		k = abs(int(sa[5: 7]) - int(sb[5: 7]))
		return d * math.cos(n * math.pi * math.pi / 180 / 180) * (n - k + g) / (n + g)
	if sa[7] == '#':
		return e
	return 1.0

def get_similar(termA, termB):
    if termB.find('/') == -1 or termA.find('/') == -1:
        return 0.0
    grA = termA[termA.find('/'):]
    grB = termB[termB.find('/'):]
    termA = termA[0: termA.find('/')]
    termB = termB[0: termB.find('/')]
    if termA == termB:
        return 1.0
    ga = synonym.get(termA)
    gb = synonym.get(termB)
    if (ga is None) or (gb is None):
        if grA == grB:
            return 0.2
        return 0.0
    ret = 0.0
    for i in range(len(ga)):
        for j in range(len(gb)):
            ret = max(ret, calc(ga[i], gb[j]))
    return ret

def get_sentence_similar(question, sentence):
	ret = 0.0
	ct = 0
	for term in question:
		maxv = 0.0
		for qterm in sentence:
			maxv = max(maxv, get_similar(term, qterm))
		if ct > 4:
			ret += maxv
		else:
			ret += coef[ct] * maxv
		ct += 1
	return ret

def corres(term, qtype):
	x = term.find('/')
	term += '   '
	if x == -1:
		return 0
	if qtype == 'other':
		if term[x+1] == 'n' and term[x+2] not in ['t','s','r']:
			return True
		else:
			return False
	if qtype == 'number':
		if term[x+1] == 'm':
			term = term.decode('utf-8')
			for i in range(term.find('/')):
				if term[i] in unicode('1234567890一二三四五六七八九十', 'utf-8'):
					return True
			return False
		else:
			return False
	if qtype == 'person':
		if term[x+1] == 'n' and term[x+2] == 'r':
			return True
		else:
			return False
	if qtype == 'place':
		if term[x+1] == 'n' and term[x+2] == 's':
			return True
		else:
			return False
	if qtype == 'organization':
		if term[x+1] == 'n' and term[x+2] == 't':
			return True
		else:
			return False
	if qtype == 'time':
		if term[x+1] == 't':
			return True
		else:
			return False

synonym, sum_cat = loadsynonym()
#f = open('synonymfile.txt', 'w')
#for key in synonym:
#	f.write(str(key) + '\t')
#	for item in synonym[key]:
#		f.write(' ' + str(item))
#	f.write('\n')
#f.close()
dis = [80, 70, 60, 50, 40]
coef = [10.0, 8.7, 3.4, 2.3, 1.2]
f = open('final_answer6500-7999.txt', 'w')
f2 = open('final_test.txt')
fkey = open('final_q_out.txt')
with open('final_test_cut.txt', 'r') as fques, open('final_answerlines_cut6500-7999.txt', 'r') as fin, open('final_test_kind.txt', 'r') as ftype:
	n = 0
	for line in fques.readlines():
		print n
		if n >= 8000:
			break
		if n < 6500:
			n = n+1
			question = f2.readline()
			qtype = ftype.readline()
			qkeys = fkey.readline()
			continue
		n = n+1
		line = line
		qterms = line.strip().split()
		qkeys = fkey.readline().split('\t')
		if qkeys[0] == 'xxx':
			del qkeys[0]
		tmpanswer = []
		sentences = fin.readline()
		while sentences[0] != '\n':
			if line[0] in "0123456789" and len(line) <= 10 :
				sentences = fin.readline()
				continue
			sentences = sentences.split()
			for i in range(len(sentences)):
				if sentences[i] == "。/x":
					sentences[i] = "。"
			sentences = ' '.join(sentences)
			sentences = sentences.split('。')
			for i in range(0, len(sentences)):
				sentence = sentences[i].split()
				#m = 0
				#for qkey in qkeys:
				#	if qkey[qkey.find('/')+1] == 'n' and qkey in sentence:
				#		m = m + 1
				if (len(qkeys) > 0 and qkeys[0] in sentence) or (len(qkeys) > 1 and qkeys[1] in sentence):
					if i > 0 and sentence[0][sentence[0].find('/')+1] in "rv":
						before = sentences[i-1].split()
						before.extend(sentence)
						v = get_sentence_similar(qterms, before)
						#print ' '.join(sentence), v
						for term in qterms:
							if term in before:
								x = term.find('/')
								if term[x+1] == 'n':
									v += 2
									if x+2 < len(term) and term[x+2] in "tsr":
										v += 1
						tmpanswer.append([v, before])
					else:
						v = get_sentence_similar(qterms, sentence)
						#print ' '.join(sentence), v
						for term in qterms:
							if term in sentence:
								x = term.find('/')
								if term[x+1] == 'n':
									v += 2
									if x+2 < len(term) and term[x+2] in "tsr":
										v += 1
						tmpanswer.append([v, sentence])
			sentences = fin.readline()
			#v = get_sentence_similar(qterms, sentences.split())
			#answer.append([v, sentences])
			#sentences = fin.readline()
		i = 1
		while True:
			answer = []
			for item in tmpanswer:
				if qkeys[i][qkeys[i].find('/')+1] in "na" and qkeys[i] in item[1]:
					answer.append(item)
			if not len(answer) > 0:
				answer = tmpanswer
				break
			else:
				#if len(answer) <= 3:
				#	break 
				tmpanswer = answer
				i = i+1
		answer.sort(key=lambda func: -func[0])
		qtype = ftype.readline().strip()
		#maxv = 0
		ans = []
		for t in range(min(5, len(answer))):
			s = answer[t][1]
			val = [0 for index in range(10000)]
			for i in range(len(s)):
				maxs = 0
				cc = 0
				for j in range(len(qterms)):
					similar = get_similar(s[i], qterms[j])
					if similar > maxs:
						maxs = similar
						cc = j
				if maxs > 0.5:
					for k in range(i-4, i+4):
						if k >= 0 and k < len(s):
							if cc > 5:
								val[k] += maxs * dis[abs(i-k)]
							else:
								val[k] += maxs * coef[cc-1] * dis[abs(i-k)]
			for i in range(len(s)):
				#if val[i] > maxv and corres(s[i], qtype):
					#maxv = val[i]
				if corres(s[i], qtype):
					ans.append([s[i], val[i]])
		ans.sort(key=lambda func: func[0])
		unians = []
		if ans:
			unians.append([ans[0][1], ans[0][0]])
		for i in range(1, len(ans)):
			if ans[i][0] in qterms:
				continue
			if ans[i][0] == ans[i-1][0]:
				unians[-1][0] += ans[i][1]
			else:
				unians.append([ans[i][1], ans[i][0]])
		unians.sort(key=lambda func: -func[0])
		question = f2.readline().strip()
		if len(ans) > 0:
			f.write(question + '\t')
			if qtype == 'number':
				for i in range(min(len(unians),2)):
					f.write(str(unians[i][1][:unians[i][1].find('/')]) + ' ')
			else:
				for i in range(min(len(unians),3)):
					f.write(str(unians[i][1][:unians[i][1].find('/')]) + ' ')
			f.write('\n')
		else:
			f.write(str(question) + "不知道" + '\n')
f.close()