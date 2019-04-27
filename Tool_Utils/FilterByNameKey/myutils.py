#!/usr/bin/python3
# -*- coding:utf-8 -*-
# author: liyangmei
# date: 2019/4/11

"""
including some utilities for other py files
"""


def rawdata2list(infile):
    """
    原始句子文件转化为列表
    句子：[sentenceid,entity1,entity2,sentence]
    """
    sentences = []
    with open(infile, "r", encoding="UTF-8") as fr:
        for line in fr.readlines():
            sentences.append(line.strip().split("\t"))
    fr.close()
    return sentences


def rawdata2dict(infile):
    """
    原始关系文件转化为字典
    key:句子编号
    value：关系int值
    """
    relations = {}
    with open(infile, "r", encoding="UTF-8") as fr:
        for line in fr.readlines():
            line = line.strip().split("\t")
            if " " in line[1]:
                line[1] = (line[1].split(" "))[0]
            relations[line[0]] = int(line[1])
    fr.close()
    return relations


def list2pickle(thelist, outfile):
    import pickle
    with open(outfile, "wb") as fw:
        pickle.dump(thelist, fw)


def pickel2list(infile):
    import pickle
    with open(infile, "rb") as fr:
        return pickle.load(fr)

