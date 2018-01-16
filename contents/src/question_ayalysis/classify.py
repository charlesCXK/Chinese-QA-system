# -*- coding: utf-8 -*-
'''
对问题进行分类
'''

import pandas as pd
from sklearn import svm
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split



# 对问题进行分类
class classify(object):
    """docstring for classify"""
    def __init__(self):
        '''
        下面这些是人工标注的规则
        '''
        self.categorys = ["person", "time", "place", "number", "organization", "other"]                 # 问题的几种类别
        self.categorys_index = {"person": 0, "time": 1, "place": 2, "number": 3, "organization": 4, "other": 5}
        self.w_word = ["什么", "为什么", "谁", "何", "何时", "哪儿", \
        "哪里", "为何", "第几", "几", "多少", \
        "怎么", "怎的", "怎样", "如何", "哪"]         # 疑问词
        self.rule_w_word = {"谁": "person", "哪位": "person", "哪一年": "time", "哪年": "time", \
        "哪个月": "time", "哪一天": "time", "几号": "time", "哪里": "place", \
        "哪儿": "place", "多少": "number", "第几": "number", "多少倍": "number", \
        "哪种":"other"}                         # 疑问词的规则模板
        self.rule_central_word = {"人":"person", "作者": "person", "诗人": "person", "作家": "person", "女性":"person", "男性":"person",\
        "演员":"person", "校长": "person", "学家": "person", "旅行家":"person", "化学家": "person","生物学家": "person","名字": "person", "国家": "place", "国籍": "place", \
        "省份": "place", "地区": "place", "城市": "place", "地点": "place", "大洲": "place","大洋": "place", "景点": "place", "山峰": "place","行星": "other","行星": "other",\
        "海拔": "number", "年龄": "number", "大学": "organization", "机构": "organization", "学院": "organization","气体":"other",\
        "组织": "organization", "俱乐部": "organization", "少数民族": "other", "语言": "other", "单位":"other", "食物":"other","关系":"other","语":"other",\
        "年份": "time", "年": "time", "成分":"other", "内容":"other",  "诗句":"other", "句":"other", "产业":"other","标志":"other", "河":"other", "药":"other"}            # 根据中心词制定模板
        self.rule_character = {"nh": "person", "ns": "place", "nl": "place", "ni": "organization",\
        "m": "number", "nt": "time", "nz": "other",}                                                  # 根据词性的模板

        self.stop_words = self.read_stop_words('data/stop_words.txt')    # 读取停用词
        self.word_map = self.mapping('model/word.vector')                # 获得所有词（不带有词性标记）的向量表示
        self.features, self.lab = self.read_x()         # 以上两个用来写csv文件
        #self.questions = self.read_questions()      # 读取问题
    
    # 读取停用词
    def read_stop_words(self, file):
        words = []
        with open(file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            words = [line[:-1] for line in lines]           # 去掉换行符
        return words

    # 将词映射到向量空间，返回映射的词典
    def mapping(self, file):
        map_res = {}
        #counter = 1     # 用于对词语编号
        with open(file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            l = int(lines[0].split()[0])         # 获得词数
            w_num = int(lines[0].split()[1])     # 获得向量维数
            for i in range(l):
                sp = lines[i+1].split()
                t_list = []
                for j in range(w_num):
                    t_list.append(sp[j+1])
                map_res[sp[0]] = t_list
        map_res['\ufeff'] = [0, 0, 0, 0]
        #print(map_res)
        return map_res

    # 读问题（已经做好词性标注的），返回所有问题的列表
    def read_questions(self, file):
        ques = []
        with open(file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

            l = len(lines)
            ques = [[] for i in range(l)]
            
            for i in range(l):
                if len(lines[i]) == 0:
                    continue
                ques[i] = lines[i].split()         # 分割出每个词  
                #print(ques[i])
                try:
                    if ques[i][-1] == '\n':
                        ques[i] = ques[i][:-1]
                except:
                    #print(ques[i])
                    pass

            #ques = [line.split('\t')[:-1] for line in lines]
            #print(ques[59][-1])
            for i in range(len(ques)):
                if len(ques[i]) == 0:
                    print(i)
        return ques             # 返回的是字符串

    # 找到问题中的疑问词，中心词，关键词
    def find_word(self, ques):
        w_word, center_word, key_word = (-1, ""), "", []
        # 找疑问词
        l = len(ques)       # 问题长度
        for target in self.w_word:
            for i in range(l):
                if ques[i].startswith(target):          # 找到疑问词，返回位置下标和词
                    w_word = (i, ques[i].split('/')[0])
                    break

        # 找中心词。定语+名词（代词） 状语+动词（形容词）这两种结构中的“+”后面的就是中心语
        w_pos = w_word[0]                               # 疑问词的位置
        if w_pos != -1:
            for i in range(w_pos, l):
                attr = ques[i].split("/")[1]            # 从疑问词向后找，它的属性，名次，人名。。。
                tmp = ques[i].split("/")
                if len(attr)>0 and (attr[0]=='n' or tmp[0]=="组织" or (tmp[0]=="一" and ques[i+1].split("/")[0]=="年") or tmp[1]=="ws" or tmp[0]=="年" or tmp[0]=="句"):        # 名词。"组织"被标位动词了  ws是英文
                    if tmp[0]=="一":
                        center_word = ques[i+1]
                    else:
                        center_word = ques[i]
                    break
            if center_word == "":           # 上面没有找到
                for i in range(w_pos, -1, -1):              # 这里第二个指标是-1，因为取不到它
                    attr = ques[i].split("/")[1]            # 从疑问词向前找，它的属性，名次，人名。。。
                    tmp = ques[i].split("/")
                    if len(attr)>0 and (attr[0]=='n' or tmp[0]=="组织" or (tmp[0]=="一" and ques[i+1].split("/")[0]=="年") or tmp[1]=="ws" or tmp[0]=="年" or tmp[0]=="句"):        # 名词。"组织"被标位动词了  ws是英文
                        if tmp[0]=="一":
                            center_word = ques[i+1]
                        else:
                            center_word = ques[i]
                        break
        if center_word == "":                           # 还没找到中心词
            #print(ques)
            if "是/v" in ques:
                s_pos = len(ques)-1       # 找到这个东西的位置
                while s_pos > 0:
                    if ques[s_pos] == "是/v":
                        break
                    s_pos -= 1
                #print("是/v : ", w_pos)
                i = s_pos
                while i >= 0:          # 从后向前遍历 是......
                    try:
                        if ques[i][0]=='/' and ques[i][1]=='/':     # 处理"//wp"这样的问题
                            attr = '/'
                        else:
                            attr = ques[i].split('/')[1]
                        if attr[0] == 'n':
                            center_word = ques[i]
                            break
                    except:
                        print(ques[i])
                    i -= 1
        l = len(ques)       # 问题长度，恢复过来
        # 找关键词（按重要性排序）
        if center_word:
            key_word.append(center_word)                # 先放中心词
        else:
            key_word.append("xxx")                      # 没有中心词，放入xxx
        # 找名词
        nouns = []
        for i in range(l):
            try:
                attr = ques[i].split('/')[1]
            except:
                print(ques, i)
            if len(attr)>0 and attr[0]=='n' and ques[i]!=center_word:
                nouns.append(ques[i])
        key_word += nouns
        # 找动词
        verbs = []
        for i in range(l):
            word = ques[i].split("/")[0]
            attr = ques[i].split("/")[1]
            if len(attr) > 0 and attr[0]=='v' and not (ques[i] in self.stop_words):
                verbs.append(ques[i])                   # 动词
        key_word += verbs
        # 剩下的词全放进去
        for i in range(l):
            if not ques[i] in key_word:
                key_word.append(ques[i])
        return w_word, center_word, key_word

    # 根据问题生成特征向量
    def generate_vector(self, ques):
        v = []
        w_word, center_word, key_word = self.find_word(ques)
        # 加入疑问词
        w_pos = w_word[0]
        if w_pos > -1:
            w_list = self.word_map[ques[w_pos].split('/')[0]]		# 不考虑词性
        else:
            l = len(self.word_map["是"])			# 只是寻找一个词，判断向量维度数
            w_list = []
            for i in range(l):
                w_list.append(0)
        v += w_list
        # 加入中心词
        if center_word:
            c_list = self.word_map[center_word.split('/')[0]]
        else:
            l = len(self.word_map["是"])
            c_list = []
            for i in range(l):
                c_list.append(0)
        v += c_list
        return v

    # 读特征值以及label
    def read_x(self):
        questions = self.read_questions('data/train_questions.txt')      # 获取问题列表
        q_len = len(questions)
        features = [self.generate_vector(q) for q in questions]                    # 获取 q_len 个向量
        # 标签：0-5
        lab = []
        # 获取标签
        labels = [[] for i in range(6)]                 # 六种标签
        with open('data/train-label.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                lab.append(self.categorys_index[line.strip()])
                for i in range(6):
                    labels[i].append(1 if self.categorys_index[line.strip()] == i else 0) 
        return features, labels

    '''
    def write_csv(self):
        # 列名 + 列值
        f0, f1, f2, f3 = [], [], [], []
        for i in range(len(self.features)):
            f0.append(self.features[i][0])
            f1.append(self.features[i][1])
            f2.append(self.features[i][2])
            f3.append(self.features[i][3])
        dataframe = pd.DataFrame({'x0':f0,'x1':f1,'x2':f2,'x3':f3,'label':self.lab})
        dataframe.to_csv("train.csv",index=True,sep=' ')
    '''

    # 对问题进行分类预测，返回预测的结果——flag列表
    def predict(self, infile, outfile):
        # 读问题
        questions = self.read_questions(infile)
        q_len = len(questions)          # 问题长度
        flag = [-1 for i in range(q_len)]        # 输出是 q_len 个类别

        # 基于规则分类
        for i in range(q_len):          # 对问题进行枚举
            w_word, center_word, key_word = self.find_word(questions[i])
            # 根据疑问词-规则来分类
            for (w, kind) in self.rule_w_word.items():
                if w_word[1].startswith(w):
                    flag[i] = self.categorys_index[kind]
                    break
            # 根据中心词-规则来分类
            if flag[i] == -1:
                for (center, kind) in self.rule_central_word.items():
                    try:
                        if center_word.split('/')[0]==center or center_word.split('/')[0][-1]==center:
                            flag[i] = self.categorys_index[kind]
                            break
                    except:
                        pass
            if flag[i]==-1 and not "是/v" in questions[i]:           # 根据中心词的词性来预测
                if center_word:
                    pro = center_word.split('/')[1]
                    for (p, kind) in self.rule_character.items():
                        if pro.startswith(p):
                            flag[i] = self.categorys_index[kind]
                            break
                        #print(center_word)
            if flag[i]==-1 and "是/v" in questions[i]:               # ******这里感觉逻辑略微有点混乱......
                try:
                    if center_word:
                        pro = center_word.split('/')[1]
                        for (p, kind) in self.rule_character.items():
                            if pro.startswith(p):
                                flag[i] = self.categorys_index[kind]
                                break
                except:
                    print(questions[i])
        # 基于DNN分类
        counter = 0						# 统计多少问题不能预测 
        res = dnn(infile, is_test=True)      # 预测结果
        for i in range(q_len):
            if flag[i] != -1:           # 基于规则已经获得结果
                continue
            flag[i] = res[i]
            # 统计有多少是无法预测结果的
            if flag[i] == -1:
                counter += 1
                print(i, questions[i])
        print("{}个问题不能预测".format(counter))

        # 写文件
        with open(outfile, 'w', encoding='utf-8') as f:
            for i in range(q_len):
                f.write(str(self.categorys[flag[i]]) + '\n')
        return flag

def addLayer(inputData,inSize,outSize,activity_function = None):  
    Weights = tf.Variable(tf.random_normal([inSize,outSize]))   
    basis = tf.Variable(tf.random_uniform([1,outSize], -1, 1))    
    weights_plus_b = tf.matmul(inputData,Weights)+basis  
    if activity_function is None:  
        ans = weights_plus_b  
    else:  
        ans = activity_function(weights_plus_b)
    return ans  

def dnn(infile, is_test=False):
	# 读问题
    classifier = classify()
    questions = classifier.read_questions(infile)
    q_len = len(questions)          # 问题长度
    flag = [-1 for i in range(q_len)]        # 输出是 q_len 个类别
    features = [classifier.generate_vector(q) for q in questions]          # 为每个问题生成特征向量
    x, y = features, classifier.lab
    x, y = np.array(x), np.matrix(classifier.lab).T
    x = np.matrix(x)

    if is_test == False:
        x_data , x_test , y_data , y_test = train_test_split(x, y, test_size = 0.99)

    # 本次是否是训练
    is_train = False

    xs = tf.placeholder(tf.float32,[None, 8]) # 样本数未知，特征数为1，占位符最后要以字典形式在运行中填入  
    ys = tf.placeholder(tf.float32,[None, 6])  # 输出二维
    keep_prob = tf.placeholder(tf.float32)  
      
    l1 = addLayer(xs, 8, 10,activity_function=None)
    l2 = addLayer(l1, 10, 20,activity_function=tf.nn.sigmoid)  
    l3 = addLayer(l2,20, 10,activity_function=tf.nn.sigmoid)  
    l4 = addLayer(l3, 10, 6,activity_function=tf.nn.softmax)


    y = l4
    loss = tf.reduce_sum(-tf.reduce_sum(ys * tf.log(y),reduction_indices=[1]))  # loss  
    train =  tf.train.GradientDescentOptimizer(0.0001).minimize(loss) # 选择梯度下降法  
        
    saver=tf.train.Saver()
    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)
        # 自己训练模型
        if is_test == False:
	        if is_train: 
	            run_step = 10000
	            for i in range(run_step):  
	                sess.run(train,feed_dict={xs:x_data,ys:y_data})  
	                if i%50 == 0:  
	                    print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
	            # 保存模型
	            saver=tf.train.Saver(max_to_keep=1)
	            saver.save(sess,'model/net.ckpt')
	        else:			# 读取已经训练好的模型
	            saver.restore(sess, 'model/net.ckpt')
	            print("load saver success!")

	        # 下面开始进行预测
	        l = y_test.shape		# y_test的大小
	        new_y = []
	        for i in range(l[0]):	# 这么多个测试样本
	        	tmp_list = y_test[i].tolist()
	        	for j in range(6):
	        		if tmp_list[0][j] == 1:
	        			new_y.append(j)
	        			break
	        new_y = np.matrix(new_y).T 		# ()*1的矩阵

	        res = sess.run(fetches=y, feed_dict={xs: x_test})
	        new_res = []
	        for ele in res:
	            mmax = -1111
	            index = -1
	            for i in range(6):
	                if ele[i] > mmax:
	                    index, mmax  = i, ele[i]
	            new_res.append(index)		# 加入更大值的下标
	        #print(new_res)
	        new_res = np.matrix(new_res).T
	        print(classification_report(new_res, new_y))
        else:			# 对测试问题进行分类
        	try:
	        	saver.restore(sess, 'model/net.ckpt')
	        	print("load saver success!")			# 读取模型成功

		        l = x.shape		# 测试问题大小
		       	res = sess.run(fetches=y, feed_dict={xs: x})
		        new_res = []
		        for ele in res:
		            mmax = -1111
		            index = -1
		            for i in range(6):
		                if ele[i] > mmax:
		                    index, mmax  = i, ele[i]
		            new_res.append(index)		# 加入更大值的下标
		        return new_res
	        except:
	        	print("load fail......")				# 读取失败

# 提取问题关键词
def exract_key(infile='data/final_test_pos.txt', outfile='data/问题关键词.txt'):
    with open(outfile, 'w', encoding='utf-8') as wr:
        classifier = classify()
        with open(infile, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        questions = [line.strip() for line in lines]
        for ques in questions:
            ques = ques.split()
            w_word, center_word, key_word = classifier.find_word(ques)
            for ele in key_word:
                wr.write(str(ele) + '\t')
            wr.write('\n')

if __name__ == '__main__':
    classifier = classify()
    #classifier.predict('data/final_test_pos.txt', 'data/final_test_kind.txt')
    #dnn('data/train_questions.txt')
    exract_key('final_test_cut.txt', 'q_out.txt')