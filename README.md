# nlp

蚂蚁金服nlp比赛



主要用了siamese模型和esim模型

# 数据
  比赛总共10万条数据，正负样本比例1：5左右
# 我的方案
  1.清洗数据，仅保留中文汉字，将数据按6：2：2设置训练集，验证集，测试集
  
  2.提取特征：
            a.长度距离
            b.编辑距离
            c.相同词数
            d.通过siamese产生的曼哈顿距离，余弦距离（仅用char级别）
            e.改进的esim模型分类概率输出（仅用char级别）
            
  3.esim模型：
    句子A,B经过embeding矩阵后，得到词向量表示，其中300维知乎问答训练的固定词向量，100维10万条数据训练的不固定词向量，共400维拼接。
    然后经过2层双向GRU进行编码，随后进入attention交互层，得到新的编码表示
    从新的编码中分离出相似向量与不相似向量，句子A.B相似矩阵，不相似矩阵分别经过一层lstm提取特征后，在通过最大池化和平均池化提取特征
    最后将相似特征与不相似特征concat，经过2层MLP得到概率输出
  
    
    
