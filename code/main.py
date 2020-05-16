# coding: utf-8

"""
操作图数据库来表示实体间的关系，通过cypher语句操作图数据库noe4j的增删改查。可以参考“https://cuiqingcai.com/4778.html”。
"""

from py2neo import Node, Relationship, Graph
import pandas as pd


info_extract_df = pd.read_csv(r'..\result\info_extract_submit.csv',header=None)
info_extract_list = [zip(line[0],line[1]) for line in info_extract_df]

graph = Graph(
    "http://localhost:7474",
    username="neo4j",
    password="person"
)

for v in info_extract_list:
    a = Node('Company', name=v[0])
    b = Node('Company', name=v[1])

    # 本次不区分投资方和被投资方，无向图
    r = Relationship(a, 'INVEST', b)
    s = a | b | r
    graph.create(s)
    r = Relationship(b, 'INVEST', a)
    s = a | b | r
    graph.create(s)


# 将实体关系插入图数据库，并查询某节点的3层投资关系，即三个节点组成的路径（如果有的话）。
# 如果无法找到3层投资关系，则查询出任意指定节点的投资路径。
graph.run("match data=(na:Company{name:'1018'})-[*1..3]->(nb:Company) return data").data()