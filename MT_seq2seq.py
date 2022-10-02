#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 11:01:07 2022

"""

import spacy
import torch
import torch.nn as nn
import random
import time,math
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from torchtext.datasets import Multi30k
from torchtext.data  import Field,BucketIterator
from torch.optim import Adam

'''

This experiment is divided into three parts, data preparation, Seq2Seq model establishment and model training.
Seq2Seq model is composed of encoder and decoder. 
There is an improvement in Seq2Seq model, which is to reverse source sequence.

CrossEntropyLoss is chosen as loss function. Loss function is expressed as perplexity eventually.

'''



####################################################
#######         data preparation         ###########
####################################################


# define field
# create a tokenizer
spacy_en=spacy.load("en_core_web_sm")# English tokenizer
spacy_fr=spacy.load("fr_core_news_sm")# French tokenizer
 
def en_seq(text):
    return [word.text for word in spacy_en.tokenizer(text)]

def fr_seq(text):
    return [word.text for word in spacy_fr.tokenizer(text)]
 
    
# source sequence reversal
#def fr_seq(text):
    #return [word.text for word in spacy_fr.tokenizer(text)][::-1]
 
# process source sequence
SRC=Field(tokenize=fr_seq,
         init_token="<sos>",
         eos_token="<eos>",
         lower=True)
 
# process target sequence
TRG=Field(tokenize=en_seq,
         init_token="<sos>",
         eos_token="<eos>",
         lower=True)


# define dataset and process using field
train_data,valid_data,test_data=Multi30k.splits(exts=('.fr','.en'),
                                                    fields=(SRC,TRG),root = '/Applications/anaconda3/lib/python3.8/site-packages/torchtext/data')
#(path = '/Applications/anaconda3/lib/python3.8/site-packages/torchtext/data/multi30k',exts=("/train.de","/train.en"),
                                               #fields=(SRC,TRG))
   
# check the size of training, validation and testing samples                                            
print(f"Number of training examples: {len(train_data.examples)}")
print(f"Number of validation examples: {len(valid_data.examples)}")
print(f"Number of testing examples: {len(test_data.examples)}")


# check if the source sequences are reversed
print(vars(train_data.examples[1]))


# createa a vocabulary list for source and target sequence
# require every word appears at least twice
SRC.build_vocab(train_data,min_freq=2)
TRG.build_vocab(train_data,min_freq=2)


'''
After establishing the vocabulary list, we build an iterative index. 
If batch_size = 1, it does not matter that the text length is not equal.
But if batch_size> 1, there is a problem, the same batch of samples are not the same length.
Then when we are training, generally we will make up for the sentence with pad operation.
Fortunately the tortchtext iterator automatically helps the pad to fill. 
The BucketIterator iterator is to choose the most appropriate length as the fixed length of all sentences.
Use pad operation when lower than the length.
Clip when above this length. 
This fixed length is based on the length of all the samples in the dataset to get the most appropriate one
'''

BATCH_SIZE = 128
cuda=torch.cuda.is_available()
 
train_iterator, valid_iterator, test_iterator=BucketIterator.splits(
    (train_data,valid_data,test_data),
    batch_size=BATCH_SIZE,
    device=torch.device('cuda' if cuda else 'cpu')
)


# test the data iterator
for example in train_iterator:
    break
print(example.src.device,example.trg.device,len(example))
print(example.src.shape,example.trg.shape)

example.src=example.src.permute(1,0)
example.trg=example.trg.permute(1,0)
print(example.src.shape,example.trg.shape)


# check the index of pad of source and target sequence
TRG.vocab.stoi["<pad>"],SRC.vocab.stoi["<pad>"]





####################################################
#######         model establishment      ###########
####################################################

'''
Seq2Seq model is mainly two parts, namely the encoder and decoder. 
We have 3 modules here, encoder, decoder, and Seq2Seq, which sets two parts. 
Among them, the network depth of the source and target is 4 layers.
'''


class  Encoder(nn.Module):
    # src_vocab_size is the size of French vocabulary list.
    # emb_model is the dimension of word vector.
    # hidden_size is the dimension of hidden vector.
    # n_layers is the depth of LSTM networks.
    def __init__(self,src_vocab_size,emb_model,hidden_size,n_layers,dropout):
        super(Encoder,self).__init__()
        
        self.embed=nn.Embedding(src_vocab_size,emb_model,padding_idx=1)
        self.lstm=nn.LSTM(input_size=emb_model,hidden_size=hidden_size,num_layers=n_layers,batch_first=True,dropout=dropout)
    
    def forward(self,src):
        src=self.embed(src)
        output,(h_n,c_n)=self.lstm(src)
        
        
        return output,(h_n,c_n)# h_n,c_n hidden state and memory of last time step

# test, parameter
emb_model=256
hidden_size=512
n_layers=4
dropout=0.5
src_vocab_size=len(SRC.vocab)

# dimension of input samples
example.src.shape

# create source sequence model
enModel=Encoder(src_vocab_size,emb_model,hidden_size,n_layers,dropout)
if(cuda):
    enModel=enModel.cuda()
output,(h_n,c_n)=enModel(example.src)
 
 
output.shape,h_n.shape,c_n.shape





'''
The target is established below. 
The start of the target is almost the same as the source.
But the difference from the source is that the input of the target is not specific.
And it adopts the form of teaching.
Either self-learning (the output of the previous time step is input at the current time step)
Or teaching (the input at the current time step is the target).
Therefore, in the target, we cannot give a sequence of words at a time.
But give one word by one, that is, the input seq_len=1.
The output of the final model we need to map it into the vector space of English words.
Another thing worth mentioning is that if lstm does not pass in h and c, it will help you create a h and c with all 0s by default.
But we need to pass in here, and the incoming h and c cannot start with batch.
The form is [n_layers*n_direction,batch_size,hidden_size]
'''




class Decoder(nn.Module):
    #trg_vocab_size is the size of vocabulary list of target
    #emb_dim is the dimension of word vector (we set it the same size with the source)
    #hidden_size is the dimension of hidden layer of target (we set it the same size with the source)
    #n_layers is the number of layers
    def __init__(self,trg_vocab_size,emb_dim,hidden_size,n_layers,dropout):
        super(Decoder,self).__init__()
        
        self.emb=nn.Embedding(trg_vocab_size,emb_dim,padding_idx=1)
        self.lstm=nn.LSTM(emb_dim,hidden_size,num_layers=n_layers,batch_first=True,dropout=dropout)
        self.classify=nn.Linear(hidden_size,trg_vocab_size)
    
    def forward(self,trg,h_n,c_n):
        trg=trg.unsqueeze(1)
        trg=self.emb(trg)
        output,(h_n,c_n)=self.lstm(trg,(h_n,c_n))
        output=self.classify(output.squeeze())
        return output,(h_n,c_n) #return (h_n,c_n) is for the use of next decoder

#test, parameter
trg_vocab_size=len(TRG.vocab)

Demodel=Decoder(trg_vocab_size,emb_model,hidden_size,n_layers,dropout)
if cuda:
    Demodel=Demodel.cuda()
    
    
# check the shape of target
example.trg.shape

trg=example.trg[:,1]# assump it is teaching this time
 
output,(h_n,c_n)=Demodel(trg,h_n,c_n)
h_n.shape,c_n.shape



'''
We use a tensor to save the output of each time step mapped to the English word vector space. 
In fact, the output of each Decode.
The classification with the highest probability of each output is the word that needs to be translated.
We need to traverse each word in the sentence.
It is also worth mentioning that because we set <sos> at the beginning of the sentence, it does not participate in the operation.
'''


class Seq2Seq(nn.Module):
    def __init__(self,encoder,decoder):
        super(Seq2Seq,self).__init__()
        self.encoder=encoder
        self.decoder=decoder
    
    def forward(self,src,trg,teach_rate=0.5):
        #src [bacth,seq_len]
        #trg  [bacth,seq_len]
        #teach_rate threshold of teaching
        batch_size=trg.shape[0]
        trg_seqlen=trg.shape[1]
        
        #save the result of every output
        outputs_save=torch.zeros(batch_size,trg_seqlen,trg_vocab_size)
        if(cuda):
            outputs_save=outputs_save.cuda()
        # encode for source
        _,(h_n,c_n)=self.encoder(src)
        
        # the first input to decoder is <sos>
        trg_i=trg[:,0]
        #trg_i [batch]
        for i in range(1,trg_seqlen):
            output,(h_n,c_n)=self.decoder(trg_i,h_n,c_n)
            #output[batch trg_vocab_size]
            outputs_save[:,i,:]=output
            #  generate a random probability (namely whether teaching or not)
            probability=random.random()
            
            # get the result of prediction of time step
            top=output.argmax(1)
            #top[batch]
            # input the next time step
            trg_i=trg[:,i] if probability>teach_rate else top
        return outputs_save


# test
model=Seq2Seq(enModel,Demodel)
if(cuda):
    model=model.cuda()
outputs=model(example.src,example.trg)

outputs.shape




####################################################
##############         training      ###############
####################################################




epochs=10
optim=Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index = 1)# pad does not participate in the calculation of the loss function

model

# parameter initialization
def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)
        
model.apply(init_weights)


'''
Establish training and test functions.
The main function of model.train() and model.eval() is that 
during training, dropout will randomly discard neurons,
and in the testing phase, we will not drop neurons.
'''


def train(model,train_iter,optim,criterion):
    model.train()# dropout works
    epoch_loss=0
    for i,example in enumerate(train_iter):
        src=example.src.permute(1,0)
        trg=example.trg.permute(1,0)
        #src[batch seqlen]
        #trg[batch seqlen]
 
        optim.zero_grad()
        output=model(src,trg)#output[batch trg_seqlen trq_vocab_size]
        #<sos> and pad don't participate in calculation（criterion has already set ignore）
        output=output[:,1:,:].reshape(-1,trg_vocab_size)
        trg=trg[:,1:].reshape(-1)
        #output[batch*(trg_len-1),trg_vocab_size]
        #trg[batch*(trg_len-1)]
        loss=criterion(output,trg)
        loss.backward()
        optim.step()
        epoch_loss+=loss.item()
    return epoch_loss/len(train_iter)

def evaluate(model,test_iter,criterion):
    model.eval()# dropout works
    epoch_loss=0
    
    with torch.no_grad():
        for i,example in enumerate(test_iter):
            
            src=example.src.permute(1,0)
            trg=example.trg.permute(1,0)
            #src[batch seqlen]
            #trg[batch seqlen]
 
 
            # teahing can't be realized
            output=model(src,trg,0)#output[batch trg_seqlen trq_vocab_size]
            
            #<sos> and pad don't participate in calculation（criterion has already set ignore）
            output=output[:,1:].reshape(-1,trg_vocab_size)
            trg=trg[:,1:].reshape(-1)
            #output[batch*(trg_len-1),trg_vocab_size]
            #trg[batch*(trg_len-1)]
            loss=criterion(output,trg)
            epoch_loss+=loss.item()
 
    return epoch_loss/len(test_iter)

# Count one training epoch for each iterator:
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# Our loss function is simply expressed as: perplexity
for epoch in range(epochs):
    start_time=time.time()
    
    train_loss=train(model,train_iterator,optim,criterion)
    valid_loss=evaluate(model,valid_iterator,criterion)
    
    end_time=time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

test_loss = evaluate(model, test_iterator, criterion)
print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
























