"""
A model that is similar to the one from
Rocktaschel et al. "Reasoning about entailment with neural attention."
approaches 2015-state-of-art results on the anssel-wang task (with
token flags).

The implementation is inspired by https://github.com/shyamupa/snli-entailment/blob/master/amodel.py


"""

from keras.layers.core import Layer
from keras.layers import GRU, Dropout, Lambda, Dense, RepeatVector, TimeDistributedDense, Activation, Reshape, Permute, Flatten
from keras.regularizers import l2
from keras import backend as K
import pysts.kerasts.blocks as B

spad=60

def config(c):
    c['dropout'] = 4/5
    c['dropoutfix_inp'] = 0
    c['dropoutfix_rec'] = 0
    c['l2reg'] = 1e-4
    c['rnnact'] = 'tanh'
    c['rnninit'] = 'glorot_uniform'
    c['sdim'] = 2


def get_last_time_dim(X):
    ans=X[:, -1, :]
    return ans

def get_R(X):
    Y, alpha = X.values()  
    ans=K.T.batched_dot(Y,alpha)
    return ans

def get_first_sentence(X):
    spad = K.spad
    return X[:, :spad, :]  

def entailment_embedding(model, inputs,N=608, spad=60, l2reg=1e-4, pfx=''):
    setattr(K, 'spad',spad)
    model.add_node(Lambda(get_last_time_dim, output_shape=(N,)), name=pfx+'h_n', input=inputs[1])
    model.add_node(Lambda(get_first_sentence, output_shape=(N,spad)), name=pfx+'Y', input=inputs[0])
    model.add_node(Permute((2,1)), name=pfx+"Yp", input=pfx+'Y')
    model.add_node(Dense(N,W_regularizer=l2(l2reg)),name=pfx+'Wh_n', input=pfx+'h_n')
    model.add_node(RepeatVector(spad), name=pfx+'Wh_n_cross_e', input=pfx+'Wh_n')
    model.add_node(TimeDistributedDense(N,W_regularizer=l2(l2reg)), name=pfx+'WY', input=pfx+'Yp')
    model.add_node(Activation('tanh'), name=pfx+'M', inputs=[pfx+'Wh_n_cross_e', pfx+'WY'], merge_mode='sum')
    model.add_node(TimeDistributedDense(1,activation='linear'), name=pfx+'alpha', input=pfx+'M')
    model.add_node(Lambda(get_R, output_shape=(N,1)), name=pfx+'_r', inputs=[pfx+'Yp',pfx+'alpha'], merge_mode='join')
    model.add_node(Permute((2,1)), name=pfx+"_rp", input=pfx+'_r')
    model.add_node(Flatten(input_shape = (N,1)),name=pfx+'r', input=pfx+'_rp')
    model.add_node(Dense(N,W_regularizer=l2(l2reg)), name=pfx+'Wr', input=pfx+'r')
    model.add_node(Dense(N,W_regularizer=l2(l2reg)), name=pfx+'Wh', input=pfx+'h_n')
    outputs=[pfx+'Wr',pfx+'Wh']
    return outputs


def prep_model(model, N, s0pad, s1pad, c):
    model.add_node(name="embmerge", inputs=['e0','e1'], merge_mode='concat', layer=Activation('linear'))
    B.rnn_input(model, N, spad, dropout=c['dropout'], dropoutfix_inp=c['dropoutfix_inp'], dropoutfix_rec=c['dropoutfix_rec'],
               sdim=2, rnnbidi=True, return_sequences=True,
               rnn=GRU, rnnact='tanh', rnninit='glorot_uniform', rnnbidi_mode='sum',
               rnnlevels=1,
               inputs=['embmerge'])
    rnn_outputs=['e0s_', 'e1s_']
    outputs = entailment_embedding(model, rnn_outputs,2*N,spad,c['l2reg'])
    return outputs

