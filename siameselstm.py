# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 10:08:06 2019

@author: peng
"""

from data import *
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
import torch.optim as optim
from torch.utils import data

s1 = np.load('s1.npy')
s2 = np.load('s2.npy')
model = KeyedVectors.load_word2vec_format(word2vec_dir, binary=False)
data = pd.read_csv(data_dir, sep='\t', header=None,names=["index", "s1", "s2", "label"])

class LSTMEncoder(nn.Module):
    def __init__(self, vocab_size, opt, layer=1,weight=None):
        super(LSTMEncoder, self).__init__()
        self.layer = layer
        self.vocab_size = vocab_size
        self.opt = opt
        self.name = 'sim_encoder'
        self.embedding_table = nn.Embedding.from_pretrained(weight, freeze=True)
        self.lstm_rnn = nn.LSTM(input_size=self.opt.embedding_dims, hidden_size=self.opt.hidden_dims,\
                                bias=True,num_layers=self.layer,batch_first = True)
        self.decode = nn.Linear(2,2)
        # self.linear = nn.Linear(self.opt.hidden_dims,self.opt.hidden_dims)
        # self.linear2 = nn.Linear(self.opt.hidden_dims,self.opt.hidden_dims//2)
        # self.dropout = nn.Dropout(p=0.2)
        # self.softplus = torch.nn.Softplus()

    def initialize_hidden_plus_cell(self, batch_size):
        zero_hidden = Variable(torch.randn(self.layer, batch_size, self.opt.hidden_dims))
        zero_cell = Variable(torch.randn(self.layer, batch_size, self.opt.hidden_dims))
        return zero_hidden, zero_cell

    def forward(self,sen1,sen2):
        """ Performs a forward pass through the network. """
        sen_emb1 = self.embedding_table(sen1)
        sen_emb2 = self.embedding_table(sen2)
#        zero_hidden, zero_cell = self.initialize_hidden_plus_cell(batch_size=batch_size)
        output1, (h1, c1) = self.lstm_rnn(sen_emb1)
        output2, (h2, c2) = self.lstm_rnn(sen_emb2)
#        return h1[-1],h2[-1]
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
#        return abs(cos(h1[-1],h2[-1]))
#        return h1[-1]-h2[-1]
        c = abs(cos(h1[-1],h2[-1]))
        res = torch.exp(-torch.sum(torch.abs(h1[-1]-h2[-1]),1))
        encoding = torch.stack((c,res),dim=1)
        output = self.decode(encoding)
        return output
        
        return res
#        self.prediction = torch.exp(-torch.norm((h1[self.layer -1] - h2[self.layer -1])))
#        return self.prediction
#        return torch.sigmoid(self.prediction)

class p(object):
    def __init__(self):
        self.learning_rate = 0.1
        self.hidden_dims = 32
        self.embedding_dims = 300
        self.beta_1 = 0.9

   
batchSize  = 64
opt = p()   
weights = torch.FloatTensor(model.syn0)     
lstma = LSTMEncoder(15,opt,weight = weights)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(lstma.parameters(), lr=0.1)



train_sen1 = s1[0:50000]
train_sen2 = s2[0:50000]
train_label = data['label'][0:50000]
test_sen1 = s1[50000:63131]
test_sen2 = s2[50000:63131]
test_label = data['label'][50000:63131].reset_index(drop=True)

#def getlabel(label):
#    a = []
#    for i in label:
#        if i == 1:
#            a.append([1,0])
#        else:
#            a.append([0,1])
#    return a
#    
#test_label = getlabel(test_label)
#train_label = getlabel(train_label)
train_sen1 = torch.LongTensor(train_sen1)
train_sen2 = torch.LongTensor(train_sen2)
train_label = torch.LongTensor(train_label)
train_set = torch.utils.data.TensorDataset(train_sen1,train_sen2,train_label)
train_iter = torch.utils.data.DataLoader(train_set, batch_size=batchSize,
                                         shuffle=True)
test_sen1 = torch.LongTensor(test_sen1)
test_sen2 = torch.LongTensor(test_sen2)
test_label = torch.LongTensor(test_label)
test_set = torch.utils.data.TensorDataset(test_sen1,test_sen2,test_label)
test_iter = torch.utils.data.DataLoader(test_set, batch_size=batchSize,
                                         shuffle=True)
for i in range(10000):
    train_loss, test_losses = 0, 0
    train_acc, test_acc = 0, 0
    acc = []
    TP,FP,FN = 0,0,0
    for sen1,sen2,label in train_iter:
#        TP,FP,FN = 0,0,0
#        acc = []
        train_acc = 0
        lstma.zero_grad()
        score= lstma(sen1,sen2)
        loss = loss_function(score, label)
#        print(score)
        loss.backward()
        optimizer.step()
        train_loss += loss
#        print(loss)
        a = torch.argmax(score.data,dim =1)
#        print(a)
#        print(score)
#        print(label)
#        print(label)
#        for item in score:
#            if item.item()>0.5:
#                acc.append(1)
#            else:
#                acc.append(0)
        for pre,tar in zip(a,label):
            if int(pre.item()) == 1 and int(tar.item()) ==1:
                TP += 1
            if int(pre.item()) == 1 and int(tar.item()) ==0:
                FP += 1
            if int(pre.item()) and int(tar.item()) ==1:
                FN += 1
    if TP !=0 and FP !=0 and FN !=0:
        recall = TP/(TP+FN)
        precision = TP/(TP+FP)
        f1 = 2*recall*precision/(precision+recall)
        print('训练集分数')
        print(f1)
#    print(recall)
#    print(precision)
    TP,FP,FN = 0,0,0
    for sen1,sen2,label in test_iter:
#        TP,FP,FN = 0,0,0
        acc = []
        score = lstma(sen1,sen2)
        loss = loss_function(score, label)
        b = torch.argmax(score.data,dim =1)
#        for item in score:
#            if item.item()>0.5:
#                acc.append(1)
#            else:
#                acc.append(0)
        for pre,tar in zip(b,label):
            if int(pre.item()) == 1 and int(tar.item()) == 1:
                TP += 1
            if int(pre.item()) == 1 and int(tar.item()) ==0:
                FP += 1
            if int(pre.item()) == 0 and int(tar.item()) ==1:
                FN += 1
    if TP !=0 and FP !=0 and FN !=0:
        recall = TP/(TP+FN)
        precision = TP/(TP+FP)
        f1 = 2*recall*precision/(precision+recall)
        print('测试集误差')
        print(f1)
#    recall = TP/(TP+FN)
#    precision = TP/(TP+FP)
#    f1 = 2*recall*precision/(precision+recall)          
#    print(f1)
#    print(recall)
#    print(precision)
#    print(n)
#    print(acc/36)
#class SiameseClassifier(nn.Module):
#    def __init__(self, vocab_size, opt, pretrained_embeddings=None, is_train=False):
#        super(SiameseClassifier, self).__init__()
#        self.opt = opt
#        self.encoder_a = self.encoder_b = LSTMEncoder(vocab_size, self.opt, is_train)
#        # Initialize pre-trained embeddings, if given
#        if pretrained_embeddings is not None:
#            None
#
#
#    def forward(self,input_data_sen1,input_data_sen2):
#        hidden_a, cell_a = self.encoder_a.initialize_hidden_plus_cell(input_data_sen1.size()[0])
#        output_a,_,_ = self.encoder_a(input_data_sen1,hidden_a, cell_a )
#
#
#        hidden_b, cell_b = self.encoder_b.initialize_hidden_plus_cell(input_data_sen1.size()[0])
#        output_b, _, _ = self.encoder_a(input_data_sen2, hidden_b, cell_b)
#
#
#        self.encoding_a = output_a[-1]
#        self.encoding_b = output_b[-1]
#
#
#        self.prediction = torch.exp(-torch.norm((self.encoding_a - self.encoding_b), 1,1))
#        return self.prediction
#        
