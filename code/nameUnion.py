# coding: utf-8

"""
编写main_extract函数，实现对实体的名称提取“主体名称”的功能。
"""

import jieba
import re


def remove_word(seg,stop_word,suffix):
    """
    筛选掉停用词和
    :param seg: 实体
    :param stop_word: 停用词列表
    :param suffix: 后缀列表
    :return:
    """
    temp_seg_lst = [word for word in seg if word not in stop_word]
    seg_lst = [word for word in temp_seg_lst if word not in suffix]
    return seg_lst



def city_prov_ahead(seg,d_city_province):
    """
    实现公司名称中地名提前
    :param seg: 实体
    :param d_city_province: 城市、省份名
    :return:
    """
    city_prov_lst = [word for word in seg if word in d_city_province]
    seg_lst = [word for word in seg if word not in d_city_province]
    return city_prov_lst+seg_lst



def my_initial():
    """
    初始化，加载词典
    :return: 后缀、停用词、地方列表
    """
    fr1 = open(r"../data/dict/co_City_Dim.txt", encoding='utf-8')
    fr2 = open(r"../data/dict/co_Province_Dim.txt", encoding='utf-8')
    # fr3 = open(r"../data/dict/company_business_scope.txt", encoding='utf-8')
    fr4 = open(r"../data/dict/company_suffix.txt", encoding='utf-8')

    # 城市名
    d_city_province = [re.sub(r'(\r|\n)*','',line) for line in fr1.readlines()]
    # 省份名
    l2_tmp = [re.sub(r'(\r|\n)*','',line) for line in fr2.readlines()]
    d_city_province.extend(l2_tmp)
    # 公司后缀
    lines4 = fr4.readlines()
    suffix = [re.sub(r'(\r|\n)*','',line) for line in lines4]

    # 停用词
    fr = open(r'../data/dict/stopwords.txt', encoding='utf-8')
    stop_word = fr.readlines()
    stop_word_after = [re.sub(r'(\r|\n)*', '', item) for item in stop_word]
    stop_word_after[-1] = stop_word[-1]
    stop_word = stop_word_after
    return suffix, stop_word, d_city_province



def main_extract(input_str,stop_word,suffix,d_city_province):
    """
    从输入的“公司名”中提取主体
    :param input_str: 名称
    :param stop_word: 停用词列表
    :param suffix: 后缀列表
    :param d_city_province: 地方列表
    :return: 实体
    """
    seg = jieba.lcut(input_str) # 分词处理
    seg_lst = remove_word(seg,stop_word,suffix)
    seg_lst = city_prov_ahead(seg_lst, d_city_province)
    return seg_lst


if __name__ == '__main__':
    # 测试实体统一
    suffix, stop_word, d_city_province = my_initial()
    company_name = "河北银行股份有限公司"
    lst = main_extract(company_name,stop_word,suffix,d_city_province)
    company_name = ''.join(lst)  # 对公司名提取主体部分，将包含相同主体部分的公司统一为一个实体
    print(company_name)

