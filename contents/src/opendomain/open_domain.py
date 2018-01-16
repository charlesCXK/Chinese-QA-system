# --*-- coding:utf-8 --*--
from urllib import request
from bs4 import BeautifulSoup  

import re
import sys
import requests

# 获取网页子链接标题、链接、简介
def get_main_info(html):
    titles, links, contents = [], [], []
    soup = BeautifulSoup(html.text, 'lxml')
    ans = soup.select('div.result')           # 获取答案那一块的html源码
    #print(ans)
    ans_len = len(ans)  
    #print(ans_len)
    ans_len = min(ans_len, 8)                       # 几个子链接长度
    for i in range(ans_len):
        soup = BeautifulSoup(str(ans[i]), 'lxml')
        title = soup.find('a').text             # 获取子网页的题目

        link = soup.find('a').get('href')       # 获得子网页的链接        

        if "淘宝" not in title and "好网站导航" not in title:		# 过滤掉
	        titles.append(title)
	        links.append(link)
	        try:
	            main_content = soup.find('div', class_ = "c-abstract").text     # 获取网页简介
	        except:     # 如果有图片或者视频，就不一样
	            main_content = soup.find('div', class_ = "c-span-last")
	            if main_content:
	                main_content = main_content.text
	        contents.append(main_content)
    return titles, links, contents

# 百度搜索，记录网页及其他信息
def search_baidu(start=0, end=8000, conduct = False):
    if conduct == False:
        return

    base_url = "http://www.baidu.com/s?wd="

    # 读取问题
    with open("test.txt", "r", encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines()]

    headers = {'content-type': 'application/json',
           'User-Agent': 'Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10_6_8; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50'}

    with open("passage"+'_'+str(start)+'_'+str(end)+".txt", "w", encoding='utf-8') as f:
        cur = start                                     # 当前问题的编号
        for line in lines[start:end]:               # 处理这一段问题
            cur += 1                # 对问题计数
            print("handling {}".format(cur))
            sys.stdout.flush()
            line = line.replace(" ", "%20")                 # 改成URL格式的空格
            line = line.replace(",", "%2C")                 # 改成URL格式的空格
            if line != "":
                print(base_url+line)                        # 输出查询的完整URL
                while True:                                 # 反复申请，直到成功
                    try:
                        url = base_url + line #+ '&pn=10'               # 查这个问题
                        req = request.Request(url, headers=headers)
                        break
                    except:
                        print("error")
                        time.sleep(1)
                html = requests.get(url)          # 获取网页
                html.encoding = ('utf-8')
                #print(html.text)
                titles, links, contents = get_main_info(html)
                #print(titles, links, contents)
                #break
            f.write("<question id={}>\n".format(cur))       # 写问题
            for i in range(5):                              # 写5个子网页
                f.write("<content>\n")
                f.write("{}\n{}\n{}".format(str(titles[i]) if i < len(titles) else "",\
                    str(links[i]) if i < len(links) else "",\
                    str(contents[i]) if i < len(contents) else ""))
                f.write("\n</content>\n")
            f.write("</question>\n")
            #print("123113")

# 找到答案的结尾
def find_end(begin, line):
    end = begin + 1             # 结尾
    while(end < len(line) and line[end] == ' '):
        end += 1
    while end < len(line) and line[end:].startswith("更多") == False\
        and not (end+2<len(line) and line[end]==" " and line[end+1].isdigit() and line[end+2].isdigit())\
        and not line[end:].startswith("～")\
        and not line[end:].startswith("？")\
        and not line[end:].startswith("?")\
        and not line[end:].startswith("！")\
        and not line[end:].startswith("!")\
        and not line[end:].startswith("。"):
        end += 1
    return end

# 模糊匹配
def is_match(a, b):
    if a[-1]=='?' or a[-1]=='？' or a[-1]=='.' or a[-1]=='。':
        a = a[:-1]
    if a[-3:] in b:
        match = True
        last_pos = b.find(a[-3:])+2
    else:
        match = False
        last_pos = 0
    return match, last_pos

def get_ans(conduct=False):
    if conduct == False:
    	return
    global dict
    dict = {}
    with open('test.txt', 'r', encoding='utf-8') as f:
        question = [line.strip() for line in f.readlines()]
    with open('label.txt', 'r', encoding='utf-8') as f:
        labels = [line.strip() for line in f.readlines()]
    # 读取所以子网页信息
    with open('passage.txt', 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines()]
    with open('../question_ayalysis/data/close.txt', 'r', encoding='utf-8') as f:
        li = f.readlines()
        for i in range(len(li)):
            ele = li[i].strip().split()
            dict[ele[0]] = ele[1]

    wr = open('1500012741_open.txt', 'w', encoding='utf-8')          # 写文件

    l = len(lines)          # 文档长度
    best_ans = [[] for i in range(len(question))]       # 最佳答案
    zhuanye_ans = [[] for i in range(len(question))]    # 专业答案
    other_ans = [[] for i in range(len(question))]      # 其他答案，比如一站到底
    answer = [[] for i in range(len(question))]
    titles = [[] for i in range(len(question))]			# 网页子标题
    abstracts = [[] for i in range(len(question))]		# 简述
    question_id = -1
    it = 0
    while it < l:
        # 汇报进度
        if it%1000 == 0:
            print("handling {}%".format(it/l))

        if lines[it].startswith('<question'):       #问题id
            pat = re.compile(r'\d+\.?\d*')            # 数字
            question_id = int(pat.findall(lines[it])[0]) - 1
            #print(question_id)
            is_found = False
            #print(question_id)
            it += 1
        elif lines[it].startswith('<content>'):      # 下面是网页标题、链接等信息
            it += 1
            #print(question_id)
            titles[question_id].append(lines[it])		# 加入标题
            #print(titles[question_id])
            it += 2
        elif lines[it].startswith('</content') or lines[it].startswith('/question'):        # 结尾
            it += 1
        else:
            index = 0                  # 取答案的起始位置
            abstracts[question_id].append(lines[it])			# 加入网页简述
            if "最佳答案" in lines[it]:
                index = lines[it].find("最佳答案") + 4
                #index = lines[it].find(':', index) + 1          # 答案的起始位置  
                end = find_end(index, lines[it])
                best_ans[question_id].append(lines[it][index:end].strip(":.。 ?？，,"))
            if "参考答案" in lines[it]:
                index = lines[it].find("参考答案") + 4
                #index = lines[it].find(':', index) + 1          # 答案的起始位置  
                end = find_end(index, lines[it])
                best_ans[question_id].append(lines[it][index:end].strip(":.。 ?？，,"))                
            if "[专业]答案" in lines[it]:
                index = lines[it].find("[专业]答案") + 6
                #index = lines[it].find(':', index) + 1          # 答案的起始位置
                end = find_end(index, lines[it])
                zhuanye_ans[question_id].append(lines[it][index:end].strip(":.。 ?？，,"))
            match, last_pos = is_match(question[question_id], lines[it])

            if match:			# 能够匹配
                tl = len(question[question_id])         # 问题长度
                index = last_pos + 1
            else:
                index = 0
            end = find_end(index, lines[it])						# 结尾部分
            ans = lines[it][index:end].strip(":.。 ?？，,")          # 最后的答案
            if index and not is_found and end-index < 200 and end-index > 1:      # 只要第一次才会替换原有答案
                #print(index, end, ans)
                answer[question_id] = ans
                is_found = True
            if best_ans[question_id]:            # 存在最佳答案，替换掉
                #print(best_ans[question_id])
                answer[question_id] = best_ans[question_id][0]
            elif zhuanye_ans[question_id]:
                answer[question_id] = zhuanye_ans[question_id][0]
            if len(titles[question_id])==5:
                if len(best_ans[question_id])==0 and len(zhuanye_ans[question_id])==0:
                    que = question[question_id]
                    if que[-1]=='.' or que[-1]=='。' or que[-1]=='?' or que[-1]=='？':
                        que = que[:-1]
                    if str(que) in dict:
                        answer[question_id] = dict[str(que)]
                if len(answer[question_id]) == 0:       # 还是没找到答案
                    pass
                    #answer[question_id] = abstracts[question_id][0]
            it += 1
    for i in range(len(answer)):
    	#print(len(titles[i]))
        if len(answer[i]) > 20:
            answer[i] = answer[i][:21]
        #wr.write(str(i+1) + '\t' + str(question[i]) + '\t' + str(answer[i]) + '\n')
        wr.write(str(answer[i]) + '\n')
    wr.close()



if __name__ == '__main__':
    search_baidu(0, 8000, conduct = False)
    get_ans(conduct = True)