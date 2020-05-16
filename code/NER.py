# coding: utf-8

"""
命名实体识别：将每句句子中实体识别出来，存入实体词典，并用特殊符号替换语句。
"""

import pandas as pd
import re
import fool
from copy import copy
from nameUnion import main_extract, my_initial


train_data = pd.read_csv('../data/info_extract/train_data.csv', encoding='gb2312', header=0)
test_data = pd.read_csv('../data/info_extract/test_data.csv', encoding='gb2312', header=0)


class Recognizer():
    def __init__(self):
        self.ner_id = 1001
        self.ner_dict_new = {}  # 存储所有实体统一后结果与其编号对应关系
        self.ner_dict_reverse_new = {}  # 存储所有实体统一后结果与其未统一前的实体名称对应关

    def recognize(self):
        suffix, stop_word, d_city_province = my_initial()

        # 处理test数据，利用开源工具进行实体识别和并使用实体统一函数存储实体
        test_data['ner'] = None
        for i in range(len(test_data)):
            sentence = copy(test_data.iloc[i, 1])
            sentence = sentence.strip()
            sentence = sentence.replace(' ', '')

            #  调用fool进行实体识别，得到words和ners结果
            words, ners = fool.analysis(sentence)
            ners[0].sort(key=lambda x: x[0], reverse=True)
            for start, end, ner_type, ner_name in ners[0]:
                if ner_type == 'company' or ner_type == 'person':
                    # 调用实体统一函数，存储统一后的实体
                    # 并自增ner_id
                    ner_lst = main_extract(ner_name, stop_word, suffix, d_city_province)
                    company_main_name = ''.join(ner_lst)

                    # 存储所有实体统一后结果与其未统一前的实体名称对应关系
                    if company_main_name not in self.ner_dict_reverse_new:
                        self.ner_dict_reverse_new[company_main_name] = [ner_name]
                    else:
                        if ner_name not in self.ner_dict_reverse_new[company_main_name]:
                            self.ner_dict_reverse_new[company_main_name].append(ner_name)
                    # 存储所有实体统一后结果与其编号对应关系
                    if company_main_name not in self.ner_dict_new:
                        self.ner_dict_new[company_main_name] = self.ner_id
                        self.ner_id += 1

                    # 在句子中用编号替换实体名
                    sentence = sentence[:start] + ' ner_' + str(self.ner_dict_new[company_main_name]) + '_ ' + sentence[end - 1:]
            test_data.iloc[i, -1] = sentence
        X_test_sentence = test_data[['ner']]


        # 处理train数据，利用开源工具进行实体识别和并使用实体统一函数存储实体
        train_data['ner'] = None
        for i in range(len(train_data)):
            # 判断正负样本
            if train_data.iloc[i, :]['member1'] == '0' and train_data.iloc[i, :]['member2'] == '0':
                sentence = copy(train_data.iloc[i, 1])
                sentence = sentence.strip()
                sentence = sentence.replace(' ', '')

                # 调用fool进行实体识别，得到words和ners结果
                words, ners = fool.analysis(sentence)
                ners[0].sort(key=lambda x: x[0], reverse=True)
                for start, end, ner_type, ner_name in ners[0]:
                    if ner_type == 'company' or ner_type == 'person':
                        # 调用实体统一函数，存储统一后的实体
                        # 并自增ner_id
                        ner_lst = main_extract(ner_name, stop_word, suffix, d_city_province)
                        company_main_name = ''.join(ner_lst)

                        # 存储所有实体统一后结果与其未统一前的实体名称对应关系
                        if company_main_name not in self.ner_dict_reverse_new:
                            self.ner_dict_reverse_new[company_main_name] = [ner_name]
                        else:
                            if ner_name not in self.ner_dict_reverse_new[company_main_name]:
                                self.ner_dict_reverse_new[company_main_name].append(ner_name)
                        # 存储所有实体统一后结果与其编号对应关系
                        if company_main_name not in self.ner_dict_new:
                            self.ner_dict_new[company_main_name] = self.ner_id
                            self.ner_id += 1

                        # 在句子中用编号替换实体名
                        sentence = sentence[:start] + ' ner_' + str(self.ner_dict_new[company_main_name]) + '_ ' + sentence[end - 1:]
                train_data.iloc[i, -1] = sentence
            else:
                # 将训练集中正样本已经标注的实体，直接使用编码替换(免去用工具进行实体识别步骤)
                sentence = copy(train_data.iloc[i, :]['sentence'])
                sentence = sentence.strip()
                sentence = sentence.replace(' ', '')

                for ner_name in [train_data.iloc[i, :]['member1'], train_data.iloc[i, :]['member2']]:
                    # 调用实体统一函数，存储统一后的实体
                    # 并自增ner_id
                    ner_lst = main_extract(ner_name, stop_word, suffix, d_city_province)
                    company_main_name = ''.join(ner_lst)

                    # 存储所有实体统一后结果与其未统一前的实体名称对应关系
                    if company_main_name not in self.ner_dict_reverse_new:
                        self.ner_dict_reverse_new[company_main_name] = [ner_name]
                    else:
                        if ner_name not in self.ner_dict_reverse_new[company_main_name]:
                            self.ner_dict_reverse_new[company_main_name].append(ner_name)

                    # 存储所有实体统一后结果与其编号对应关系
                    if company_main_name not in self.ner_dict_new:
                        self.ner_dict_new[company_main_name] = self.ner_id
                        self.ner_id += 1

                    # 在句子中用编号替换实体名
                    sentence = re.sub(company_main_name, ' ner_%s_ ' % (str(self.ner_dict_new[company_main_name])), sentence)
                train_data.iloc[i, -1] = sentence
        X_train_sentence = train_data[['ner']]
        y_train = train_data[['tag']]
        # train_num = len(train_data)

        return X_train_sentence,y_train,X_test_sentence


if __name__ == "__main__":
    recognizer = Recognizer()
    info_extract_entity_list = []
    # 遍历ner_dict_new, ner_dict_reverse_new进行生成
    for k in recognizer.ner_dict_new:
        entity_id = recognizer.ner_dict_new[k]
        entity_names = '|'.join(recognizer.ner_dict_reverse_new[k])
        info_extract_entity_list.append([entity_id, entity_names])

    # # 生成表格
    info_extract_entity_df = pd.DataFrame(info_extract_entity_list, columns=['实体编号', '实体名'])
    info_extract_entity_df.to_csv('../result/info_extract_entity.csv', index=False)




