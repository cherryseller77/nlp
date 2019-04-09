# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 16:37:56 2019

@author: peng
"""

import pandas as pd
from gensim.models import KeyedVectors
from collections import defaultdict
import jieba
import re
import torch
import numpy as np
import tensorflow as tf
from torch import nn



data_dir = 'atec_nlp_sim_train_add.csv'
preprocessed_data_dir = r'G:\tecent'
dict_path = r'dict.txt'
word2vec_dir = 'sgns.zhihu.bigram'
maxSeqLength = 15



def seg_sentence(sentence):
    """
    对句子进行分词
    :param sentence:句子，String
    """
    #加载新词典
    jieba.load_userdict(dict_path)
    sentence_seged = jieba.cut(sentence.strip())
    out_str = ""
    for word in sentence_seged:
            if word != " ":
                out_str += word
                out_str += " "
    return out_str
    
def preprocessing(data_df,fname):
    """
    :param data_df:需要处理的数据集
    :param fname:
    :return:
    """
    # 记录词汇表词频
    vocabs = defaultdict(int)
    for index, row in data_df.iterrows():
        # 分别遍历每行的两个句子，并进行分词处理
        for col_name in ["s1", "s2"]:
            #提取中文
            seg_str = re.sub("[^\u4e00-\u9fa5]", " ", row[col_name]) 
            # 分词
            seg_str = seg_sentence(seg_str)
            for word in seg_str.split(" "):
                vocabs[word] = vocabs[word] + 1
                data_df.at[index, col_name] = seg_str

    data_df.to_csv(preprocessed_data_dir + '{}.csv'.format(fname), sep='\t', header=None,index=None,encoding='utf-8')
    return vocabs


def getembedding(sentence,model):
    s1 = []
    s2 = []
    for index,row in sentence.iterrows():
        for col_name in ['s1']:
            s1.append(row[col_name].split())
        for col_name in ['s2']:
            s2.append(row[col_name].split())
    s11 = np.zeros((len(s1),maxSeqLength),dtype = 'int32')
    s22 = np.zeros((len(s2),maxSeqLength),dtype = 'int32')
    rownumber,indexnumber = 0,0
    for i in s1:
        for j in i:
            if indexnumber <maxSeqLength:
                try:
                    s11[rownumber][indexnumber] = model.index(j)
                except ValueError:
                    s11[rownumber][indexnumber] = model.index('1')
            indexnumber += 1
        rownumber += 1
        indexnumber = 0
        if rownumber >= len(s1):
            break
    rownumber,indexnumber = 0,0
    for i in s2:
        for j in i:
            if indexnumber <maxSeqLength:
                try:
                    s22[rownumber][indexnumber] = model.index(j)
                except ValueError:
                    s22[rownumber][indexnumber] = model.index('1')
            indexnumber += 1
        rownumber += 1
        indexnumber = 0
        if rownumber >= len(s2):
            break
    np.save(preprocessed_data_dir + '{}.npy'.format('\s1'),s11)
    np.save(preprocessed_data_dir + '{}.npy'.format('\s2'),s22)
    return s11,s22



if __name__ == '__main__':
    data = pd.read_csv(data_dir, sep='\t', header=None,names=["index", "s1", "s2", "label"])
    a = preprocessing(data,r'\pt')
    model = KeyedVectors.load_word2vec_format(word2vec_dir, binary=False)
    wordindex = model.index2word
    s1,s2 = getembedding(data,wordindex)
    weights = torch.FloatTensor(model.syn0)
    embedding = nn.Embedding.from_pretrained(weights)
    a = torch.LongTensor(b)
