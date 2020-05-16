# coding: utf-8

"""
实体消除歧义，待更新
"""


import pandas as pd
import requests
from bs4 import BeautifulSoup
from copy import copy
import fool
import numpy as np
import tqdm

# 找出test_data.csv中前25条样本所有的人物名称，以及人物所在文档的上下文内容
test_data = pd.read_csv('../data/info_extract/test_data.csv', encoding='gb2312', header=0)

# 存储人物以及上下文信息（key为人物ID，value为人物名称、人物上下文内容）
person_name = {}

# 观察上下文的窗口大小
window = 10

# 遍历前25条样本
for i in range(25):
    sentence = copy(test_data.iloc[i, 1])
    words, ners = fool.analysis(sentence)
    ners[0].sort(key=lambda x: x[0], reverse=True)
    for start, end, ner_type, ner_name in ners[0]:
        if ner_type == 'person':
            # TODO：提取实体的上下文
            context = sentence[max(0, start - window):start] + sentence[end - 1:min(len(sentence), end - 1 + window)]
            person_name[i] = (ner_name, context)


# print(person_name)

def calculate_sentence_similarity(string1, string2):
    """
    计算两个句子的相似度(基于unigram, bigram, trigram的重复程度)
    """
    # unigram
    unigram_set1 = set(string1)
    unigram_set2 = set(string2)
    unigram_intersection = unigram_set1 & unigram_set2
    unigram_union = unigram_set1 | unigram_set2
    unigram_similarity_score = len(unigram_intersection) / len(unigram_union)

    # bigram
    bigram_set1 = {string1[idx:idx + 2] for idx in range(len(string1) - 1)}
    bigram_set2 = {string2[idx:idx + 2] for idx in range(len(string2) - 1)}
    bigram_intersection = bigram_set1 & bigram_set2
    bigram_union = bigram_set1 | bigram_set2
    bigram_similarity_score = len(bigram_intersection) / len(bigram_union)

    # trigram
    trigram_set1 = {string1[idx:idx + 3] for idx in range(len(string1) - 2)}
    trigram_set2 = {string2[idx:idx + 3] for idx in range(len(string2) - 2)}
    trigram_intersection = trigram_set1 & trigram_set2
    trigram_union = trigram_set1 | trigram_set2
    trigram_similarity_score = len(trigram_intersection) / len(trigram_union)

    return unigram_similarity_score + bigram_similarity_score + trigram_similarity_score


# 利用爬虫得到每个人物名称对应的URL
# TODO：找到每个人物实体的词条内容。
# # 初始化结果列表
result_data = []
# # 循环爬取

try:
    with tqdm(person_name.items()) as tq:
        for k, v in tq:
            # 拼接生成serach_url
            current_name = v[0]
            search_url = 'https://baike.baidu.com/item/' + current_name
            # 获取当前人名对应实体编号
            current_name_id = ner_dict_new[current_name]
            #     print(current_name_id, current_name)
            # 请求数据
            headers = {
                'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743.116 Safari/537.36'}
            web_data = requests.get(search_url, headers=headers)
            web_data.encoding = 'utf-8'
            # 解析数据
            soup = BeautifulSoup(web_data.text, 'lxml')
            # # 查找是否有百科搜到的title
            check_title = soup.select('dd.lemmaWgt-lemmaTitle-title > h1')

            if check_title:
                # 将样本中人物上下文与爬取词条结果进行对比，选择最接近的词条
                current_context = v[1]
                # 获取对应节点列表
                entries_node = soup.select('li.item *')
                # 获取词条列表
                entries_list = [i.text for i in entries_node]
                #         print(entries_list)
                # 获取词条链接列表
                entries_url_list = [search_url if i.get('class') else 'https://baike.baidu.com' + i.get('href') for i in
                                    entries_node]
                # 判断是否获取到了词条
                if entries_list:
                    # # 计算所有词条与当前人物名称的相似度
                    entries_similarity_list = [calculate_sentence_similarity(current_context, entry) for entry in
                                               entries_list]
                    #                     print(entries_similarity_list)
                    # # 获取最大相似度所在的索引
                    max_similarity_index = np.argmax(entries_similarity_list)
                    # # 获取最大相似度对应的词条链接
                    max_similarity_entry_url = entries_url_list[max_similarity_index]
                    # # 将该词条链接添加到结果列表
                    result_data.append((current_name_id, max_similarity_entry_url))
                else:
                    result_data.append((current_name_id, search_url))
            else:
                result_data.append((current_name_id, 'N/A'))
except KeyboardInterrupt:
    tq.close()
    raise

# print(result_data)

# 输出结果
result_df = pd.DataFrame(result_data, columns=['实体编号', 'URL'])
result_df.to_csv('../submit/entity_disambiguation_submit.csv', index=False)
result_df