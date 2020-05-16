# coding: utf-8

"""
构造关系识别训练特征包括句子的tfidf特征以及提取句法特征（1、企业实体间距离；2、企业实体间句法距离；
3、企业实体分别和关键触发词的距离；4、实体的依存关系类别）。利用已经提取好的tfidf特征以及parse特征，
建立分类器进行分类任务，识别测试数据集的实体是否有关系，并生成关系对。
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from pyltp import Parser
from pyltp import Segmentor
from pyltp import Postagger
from itertools import combinations
from scipy import sparse
from tqdm import tqdm
import networkx as nx
import re
import pandas as pd
import numpy as np
from NER import Recognizer

from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


# 构造关系识别训练特征
# 实体符号加入分词词典
with open('../data/user_dict.txt', 'w', encoding='utf-8') as fw:
    for v in ['一', '二', '三', '四', '五', '六', '七', '八', '九', '十']:
        fw.write(v + '号企业 ni\n')
# 加载模型，加载自定义词典
segmentor = Segmentor()
segmentor.load_with_lexicon('../data/sltp-model/ltp_data_v3.4.0/cws.model', '../data/user_dict.txt')

# 加载停用词
fr = open(r'../data/dict/stopwords.txt', 'r', encoding='utf-8')
stop_word = fr.readlines()
stop_word = [re.sub(r'(\r|\n)*', '', word) for word in stop_word]


def filtered_segment(s):
    """
    将ner_1089_等实体替换成“N号企业”，避免ner_1089_这样的字符被分开
    """
    tmp_ner_dict = {}
    num_lst = ['一', '二', '三', '四', '五', '六', '七', '八', '九', '十']

    # 将公司代码替换为特殊称谓，保证分词词性正确
    for i, ner in enumerate(list(set(re.findall(r'(ner_\d\d\d\d_)', s)))):
        try:
            tmp_ner_dict[num_lst[i] + '号企业'] = ner
        except IndexError:
            # 定义错误情况的输出
            print('替换出错!')

        s = s.replace(ner, num_lst[i] + '号企业')
    words = segmentor.segment(s)
    result_list = [tmp_ner_dict.get(word) if tmp_ner_dict.get(word) else word for word in words if
                   word not in stop_word]
    result_segment = ' '.join(result_list)
    return result_segment


# 获取训练数据和测试数据
recognizer = Recognizer()
X_train_sentence,y_train,X_test_sentence = recognizer.recognize()
corpus_train = X_train_sentence['ner'].map(filtered_segment).tolist()
corpus_test = X_test_sentence['ner'].map(filtered_segment).tolist()

# 提取tfidf特征
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(corpus_train)
X_test = vectorizer.transform(corpus_test)
print(X_train.shape)
print(X_test.shape)

# 提取句法特征
# 1、企业实体间距离
# 2、企业实体间句法距离
# 3、企业实体分别和关键触发词的距离
# 4、实体的依存关系类别
postagger = Postagger()
postagger.load_with_lexicon('../data/ltp-models/ltp_data_v3.4.0/pos.model', '../data/user_dict.txt')  # 加载模型
segmentor = Segmentor()
segmentor.load_with_lexicon('../data/ltp-models/ltp_data_v3.4.0/cws.model', '../data/user_dict.txt')  # 加载模型

def shortest_path(arcs_ret, source, target):
    """
    求出两个词最短依存句法路径，不存在路径返回-1
    arcs_ret：句法分析结果表格
    source：实体1
    target：实体2
    """
    G = nx.DiGraph()
    # 为这个网络添加节点...
    for i in list(arcs_ret.index):
        G.add_node(i)
        # 在网络中添加带权重的边...（注意，我们需要的是无向边）
        G.add_edge(arcs_ret.loc[i, 2], i)

    G.to_undirected()  # 转换成无向图

    try:
        # 利用nx包中shortest_path_length方法实现最短距离提取
        distance = nx.shortest_path_length(G, source=source, target=target)
        return distance
    except:
        return -1

def parse(s):
    """
    对语句进行句法分析，并返回句法结果
    """
    tmp_ner_dict = {}
    num_lst = ['一', '二', '三', '四', '五', '六', '七', '八', '九', '十']

    # 将公司代码替换为特殊称谓，保证分词词性正确
    for i, ner in enumerate(list(set(re.findall(r'(ner_\d\d\d\d_)', s)))):
        try:
            tmp_ner_dict[num_lst[i] + '号企业'] = ner
        except IndexError:
            # 定义错误情况的输出
            print('替换出错!')

        s = s.replace(ner, num_lst[i] + '号企业')
    words = segmentor.segment(s)
    tags = postagger.postag(words)
    parser = Parser()  # 初始化实例
    parser.load('../data/ltp-models/ltp_data_v3.4.0/parser.model')  # 加载模型
    arcs = parser.parse(words, tags)  # 句法分析
    arcs_lst = list(map(list, zip(*[[arc.head, arc.relation] for arc in arcs])))

    # 句法分析结果输出
    parse_result = pd.DataFrame([[a, b, c, d] for a, b, c, d in zip(list(words), list(tags), arcs_lst[0], arcs_lst[1])],
                                index=range(1, len(words) + 1))
    parser.release()  # 释放模型

    # 提取企业实体依存句法类型
    # 投资关系关键词
    key_words = ["收购", "竞拍", "转让", "扩张", "并购", "注资", "整合", "并入", "竞购",
                 "竞买", "支付", "收购价","收购价格", "承购", "购得", "购进","购入", "买进",
                 "买入", "赎买", "购销", "议购", "函购", "函售", "抛售", "售卖", "销售",
                 "转售"]
    # 提取关键词和对应句法关系提取特征
    # 初始化企业实体的索引列表
    ner_index_list = []
    # 初始化关键词的索引列表
    keyword_index_list = []

    # 遍历句法分析表格的索引
    for idx in parse_result.index:
        if parse_result.loc[idx, 0].endswith('号企业'):
            ner_index_list.append(idx)

        if parse_result.loc[idx, 0] in key_words:
            keyword_index_list.append(idx)

    # 1)在句子中的关键词数量
    parse_feature1 = len(keyword_index_list)

    # 2)若关键词存在
    # # 初始化判断与关键词有直接关系的'X号企业'句法类型为'S..'的数量
    parse_feature2 = 0
    # # 初始化判断与关键词有直接关系的'X号企业'句法类型为'.OB'的数量    A B C D
    parse_feature3 = 0

    # 遍历出现在句子中的关键词索引
    for i in keyword_index_list:
        # 遍历出现在句子中的实体索引
        for j in ner_index_list:
            # 如果实体句法类型以S开头(主语)或者OB结尾(宾语)
            if parse_result.loc[j, 3].startswith('S') or parse_result.loc[j, 3].endswith('OB'):
                # 若关键词对应句法关联索引为实体索引，则parse_feature2数量+1
                if parse_result.loc[i, 2] == j:
                    parse_feature2 += 1
                # 若实体索引对应句法关联索引为关键词索引，则parse_feature3数量+1
                if parse_result.loc[j, 2] == i:
                    parse_feature3 += 1

    # 3)实体与关键词之间距离的平均值，最大值和最小值
    ner_keyword_pair_list = [(ner_index, keyword_index) for ner_index in ner_index_list for keyword_index in
                             keyword_index_list]
    ner_keyword_distance_list = [abs(pair[0] - pair[1]) for pair in ner_keyword_pair_list]

    parse_feature4 = np.mean(ner_keyword_distance_list) if ner_keyword_distance_list else 0
    parse_feature5 = max(ner_keyword_distance_list) if ner_keyword_distance_list else 0
    parse_feature6 = min(ner_keyword_distance_list) if ner_keyword_distance_list else 0

    # 4)实体与关键词之间句法距离的平均值，最大值和最小值
    ner_keyword_parse_distance_list = [shortest_path(parse_result, pair[0], pair[1]) for pair in ner_keyword_pair_list]

    parse_feature7 = np.mean(ner_keyword_parse_distance_list) if ner_keyword_parse_distance_list else 0
    parse_feature8 = max(ner_keyword_parse_distance_list) if ner_keyword_parse_distance_list else 0
    parse_feature9 = min(ner_keyword_parse_distance_list) if ner_keyword_parse_distance_list else 0

    # 5)实体之间距离的平均值，最大值和最小值
    ner_pair_list = list(combinations(ner_index_list, 2))
    ner_distance_list = [abs(pair[0] - pair[1]) for pair in ner_pair_list]

    parse_feature10 = np.mean(ner_distance_list) if ner_distance_list else 0
    parse_feature11 = max(ner_distance_list) if ner_distance_list else 0
    parse_feature12 = min(ner_distance_list) if ner_distance_list else 0

    # 6)实体之间句法距离的平均值，最大值和最小值
    ner_parse_distance_list = [shortest_path(parse_result, pair[0], pair[1]) for pair in ner_pair_list]

    parse_feature13 = np.mean(ner_parse_distance_list) if ner_parse_distance_list else 0
    parse_feature14 = max(ner_parse_distance_list) if ner_parse_distance_list else 0
    parse_feature15 = min(ner_parse_distance_list) if ner_parse_distance_list else 0

    return [parse_feature1, parse_feature2, parse_feature3,
            parse_feature4, parse_feature5, parse_feature6,
            parse_feature7, parse_feature8, parse_feature9,
            parse_feature10, parse_feature11, parse_feature12,
            parse_feature13, parse_feature14, parse_feature15]

def get_extra_feature(corpus):
    """
    获取语料数据的额外特征
    """
    # 初始化存放所有语料数据额外特征的列表
    extra_features_list = []
    # 遍历语料，计算各条数据对应的额外特征，并添加到列表
    try:
        with tqdm(corpus) as t:
            for i in t:
                current_extra_features = parse(i)
                extra_features_list.append(current_extra_features)
        return np.array(extra_features_list)
    except KeyboardInterrupt:
        t.close()
        raise

# 对训练解和测试集分别拼接tf-idf特征与额外特征
X_train_extra_features = get_extra_feature(corpus_train)
X_test_extra_features = get_extra_feature(corpus_test)

X_train = sparse.hstack((X_train, sparse.csr_matrix(X_train_extra_features)), format='csr')
X_test = sparse.hstack((X_test, sparse.csr_matrix(X_test_extra_features)), format='csr')

print(X_train.shape)
print(X_test.shape)


# 利用已经提取好的tfidf特征以及parse特征，建立分类器进行分类任务
# 建立分类器
# 定义需要遍历的参数
tuned_parameters = {"C": [0.001, 0.003, 0.01, 0.003, 0.1, 0.3, 1, 3]}
scores = ['precision', 'recall', "f1"]

# 选择模型
lr = LogisticRegression()

# 利用GridSearchCV搜索最佳参数
for score in scores:
    print("# Tuning hyper-parameters for %s---------" % score)
    clf = GridSearchCV(LogisticRegression(),
                       tuned_parameters,
                       cv=5,
                       scoring='%s' % score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:\n")
    print(clf.best_params_)
    print("Grid scores on development set:\n")
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print("Detailed classification report:\n")
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.\n")

# 对Test_data进行分类
best_model = clf.best_estimator_
y_pred_test = best_model.predict(X_test)


# 识别测试数据集的实体关系

# 初始化最终的结果列表
info_extract_list = []
# 先利用预测的标签，提取出有关系的句子
predict_positive_examples = pd.Series(corpus_test)[pd.Series(y_pred_test).astype(bool)]
# 遍历各个句子
for example in predict_positive_examples:
    # # 正则提取该句子所有的实体符号，并去重
    current_entity_list = re.findall('ner_(\d\d\d\d)_', example)
    current_entity_set = set(current_entity_list)
    # 当前句子的实体组合关系
    current_entity_pair_list = list(combinations(current_entity_set, 2))
    # 将当前句子的实体组合关系扩充到结果列表中
    info_extract_list.extend(current_entity_pair_list)

# 生成表格
info_extract_df = pd.DataFrame(info_extract_list, columns=['实体1', '实体2'])
info_extract_df.to_csv(r'..\result\info_extract_submit.csv', index=False)





if __name__ == "__main__":
    print(parse('2016年 ner_1614_ 有限公司 瑞峰 张家港 光伏 科技 有限公司 支付 设备 款 人民币 4 515 770.00 元'))
    print(parse('2016年 ner_1614_ 有限公司瑞峰张家港光伏科技有限公司支付设备款人民币4515770.00元'))

    print(parse('资产 出售 资产 购买 方案 简要 介绍 资产 出售 公司 拟 控股 股东 ner_1625_ 出售 ner_1463_ 100% 股权'))
    print(parse('资产出售资产购买方案简要介绍资产出售公司拟控股股东 ner_1625_ 出售 ner_1463_ 100%股权'))

    print(parse('集团 ner_1089_ 股份 有限公司 持有 ner_1616_ 股份 有限公司 股票 本期 变动 系 买卖 一揽子 沪 深 300 指数 成份股 致'))
    print(parse('本集团及 ner_1089_ 股份有限公司持有 ner_1616_ 大股份有限公司股票的本期变动系买卖一揽子沪深300指数成份股所致'))
