"""
A simple model based on skipthoughts sentence embeddings.

To set up:
    * Execute the "Getting started" wgets in its README
    * set config['skipthoughts_datadir'] to directory with downloaded files
    * make skipthoughts.py from https://github.com/ryankiros/skip-thoughts/blob/master/skipthoughts.py 
        available via import skipthoughts

Inner working: First we compute skipthought embedding of both inputs; then we merge them (multiply & subtract), cancatenate, and compute result (1 MLP layer).
"""

from __future__ import print_function
from __future__ import division


from keras.models import Graph
from keras.layers.core import Activation, Dense, Dropout
from keras.regularizers import l2

import pysts.embedding as emb
import pysts.loader as loader
import pysts.kerasts.blocks as B
from pysts.kerasts.objectives import pearsonobj

import numpy as np


def config(c):
    # XXX:
    c['skipthoughts_datadir'] = "/storage/ostrava1/home/nadvorj1/skip-thoughts/"

    # disable GloVe
    c['embdim'] = None
    # disable Keras training
    c['ptscorer'] = None

    c["skipthoughts_uni_bi"] = "combined"

    # These values were selected by (random) tuning:
    c['merge_sum'] = True
    c['merge_mul'] = False
    c['merge_diff'] = True
    c['merge_absdiff'] = False
    c['l2reg'] = 0.0001
    c['dropout'] = 0.2


class STModel:
    """ Quacks (a little) like a Keras model. """

    def __init__(self, c, output):
        self.c = c

        # xxx: this will probably break soon
        if output == 'classes':
            self.output = output
            self.out_act = "sigmoid"
            self.c['loss'] = pearsonobj
            self.output_width = 6
        else:
            self.output = 'score'
            self.out_act = "sigmoid"
            self.c['loss'] = 'binary_crossentropy'
            c['balance_class'] = True
            self.output_width = 1

        self.st = emb.SkipThought(c=self.c)
        self.N = self.st.N

        self.model = self.prep_model()

    def prep_model(self, do_compile=True):
        dropout = self.c["dropout"]  # XXX

        model = Graph()
        model.add_input(name='e0', input_shape=(self.N,))
        model.add_input(name='e1', input_shape=(self.N,))
        model.add_node(name="e0_", input="e0", layer=Dropout(dropout))
        model.add_node(name="e1_", input="e1", layer=Dropout(dropout))

        merges = []
        if self.c.get("merge_sum"):
            model.add_node(name='sum', inputs=['e0_', 'e1_'], layer=Activation('linear'), merge_mode='sum')
            model.add_node(name="sum_", input="sum", layer=Dropout(dropout))
            merges.append("sum_")

        if self.c.get("merge_mul"):
            model.add_node(name='mul', inputs=['e0_', 'e1_'], layer=Activation('linear'), merge_mode='mul')
            model.add_node(name="mul_", input="mul", layer=Dropout(dropout))
            merges.append("mul_")

        if self.c.get("merge_absdiff"):
            merge_name = B.absdiff_merge(model, ["e0_", "e1_"], pfx="", layer_name="absdiff", )
            model.add_node(name="%s_" % merge_name, input=merge_name, layer=Dropout(dropout))
            merges.append("%s_" % merge_name)

        if self.c.get("merge_diff"):
            merge_name = B.absdiff_merge(model, ["e0_", "e1_"], pfx="", layer_name="diff")
            model.add_node(name="%s_" % merge_name, input=merge_name, layer=Dropout(dropout))
            merges.append("%s_" % merge_name)

        model.add_node(name='hidden', inputs=merges, merge_mode='concat',
                       layer=Dense(self.output_width, W_regularizer=l2(self.c['l2reg'])))
        model.add_node(name='out', input='hidden', layer=Activation(self.out_act))
        model.add_output(name=self.output, input='out')

        if do_compile:
            model.compile(loss={self.output: self.c['loss']}, optimizer=self.c["opt"])
        return model

    def fit(self, gr, **kwargs):
        self.precompute_embeddings(gr)

        self.e0, self.e1, _, _, y = loader.load_embedded(self.st, gr["s0"], gr["s1"], gr[self.output], balance=True, ndim=1)

        # fit_kwargs = {}
        # if self.c['balance_class']:
        #     targets = list(set(gr[self.output]))
        #     ratios = dict([(target, 1.0 / np.sum(gr[self.output] == target)) for target in targets])
        #     s = sum(ratios.values())
        #     for k, v in ratios.items():
        #         ratios[k] = v / s
        #     fit_kwargs = {"class_weight": {self.output: ratios}}

        self.model.fit({'e0': self.e0, 'e1': self.e1, self.output: y},
                       batch_size=self.c["batch_size"], nb_epoch=self.c["nb_epoch"],
                       verbose=2,
                       # **fit_kwargs
                       )

    def load_weights(self, *args, **kwargs):
        if not hasattr(self, "model"):
            self.model = self.prep_model()
        self.model.load_weights(*args, **kwargs)

    def save_weights(self, *args, **kwargs):
        self.model.save_weights(*args, **kwargs)

    def precompute_embeddings(self, gr):
        sentences = [" ".join(words) for words in gr["s0"] + gr["s1"]]
        self.st.batch_embedding(sentences)

    def predict(self, gr):
        self.precompute_embeddings(gr)
        e0, e1, _, _, _ = loader.load_embedded(self.st, gr["s0"], gr["s1"], gr[self.output], balance=False, ndim=1)

        result = self.model.predict({'e0': e0, 'e1': e1})
        return result


def prep_model(vocab, c, output='score'):
    return STModel(c, output)
