# -*- coding: utf-8 -*-
"""
Calculate and graph symmetric Kullback-Liebler divergence in LDA
across range of number of topics.
"""

import os
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import scipy.stats as stats
import time

from gensim import matutils
from gensim.corpora import Dictionary, MmCorpus
from gensim.models import LdaMulticore
from os import path

datadir = path.abspath(path.join(os.getcwd(), "data"))

fin = path.join(datadir, "reuters21578.dict.txt")
vocabulary = Dictionary.load_from_text(fin)

fin = path.join(datadir, "reuters21578.mm.bz2")
mm = MmCorpus(fin)

def sym_kl(p, q):
    return np.sum([stats.entropy(p, q), stats.entropy(p, q)])
    
def arun(corpus, dictionary, min_topics=10, max_topics=100, step=10):
    l = np.array([sum(cnt for _, cnt in doc) for doc in corpus])
    
    kl = []
    for n in range(min_topics, max_topics+step, step):
        print("starting multicore LDA for num_topics={}".format(n))
        st = time.clock()
        lda = LdaMulticore(corpus=corpus,
                           id2word=vocabulary,
                           num_topics=n,
                           passes=20,
                           workers=mp.cpu_count()-1)
        el = time.clock()-st
        print("multicore LDA finished in {:.2f}s!".format(el))
        
        m1 = lda.expElogbeta
        _, cm1, _ = np.linalg.svd(m1)
        
        lda_topics = lda[corpus]
        m2 = matutils.corpus2dense(lda_topics, lda.num_topics).transpose()
        cm2 = l.dot(m2)
        cm2 = cm2 + 0.0001
        cm2norm = np.linalg.norm(l)
        cm2 = cm2/cm2norm
        kl.append(sym_kl(cm1, cm2))
        
    return kl
    
  
kl = arun(mm, vocabulary, min_topics=5, max_topics=100, step=5)
x = list(range(5,100+5,5))  
labels = ["{}".format(e) for e in x]

plt.plot(x, kl)
plt.xticks(x, labels)
plt.title("Number of LDA Topics & KL Divergence")
plt.ylabel("KL Divergence")
plt.xlabel("Num. of Topics")
plt.savefig("sym_kl_div.png", bbox_inches="tight")