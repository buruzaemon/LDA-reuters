# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 21:58:58 2015

@author: buruzaemon
"""
import os
from gensim.corpora import Dictionary, MmCorpus
from gensim.models import TfidfModel, LdaModel
from os import path
from reuterscorpus import ReutersCorpus

def main():
    DATADIR = path.abspath(path.join(os.getcwd(), "data"))
    INFILE = path.join(DATADIR, "reuters21578.tar.gz")
    OUT_BOW = path.join(DATADIR, "reuters_bow.mm")
    OUT_VOCAB = path.join(DATADIR, "reuters_wordids.txt")
    OUT_TFIDFMODEL = path.join(DATADIR, "reuters.tfidf_model")
    OUT_TFIDF_CORPUS = path.join(DATADIR, "reuters_tfidf.mm")

    rc = ReutersCorpus(INFILE)
    rc.dictionary.filter_extremes(no_below=20, no_above=0.1)
    
    # save the bag-of-words corpus
    MmCorpus.serialize(OUT_BOW, rc, progress_cnt=10000)

    # serialize vocabulary as text
    rc.dictionary.save_as_text(OUT_VOCAB)

    # we are now done using the corpus
    #del rc
    
    # load back the id->word mapping directly from file
    # this seems to save more memory, compared to keeping the corpus.dictionary 
    # object from above
    vocab = Dictionary.load_from_text(OUT_VOCAB)

    # initialize corpus reader and word->id mapping
    mm = MmCorpus(OUT_BOW)

    # build tfidf, ~50min
    tfidf = TfidfModel(mm, id2word=vocab, normalize=True)
    tfidf.save(OUT_TFIDFMODEL)

    # save tfidf vectors in matrix market format
    MmCorpus.serialize(OUT_TFIDF_CORPUS, tfidf[mm], progress_cnt=10000)
    
    mm = MmCorpus(OUT_TFIDF_CORPUS)
    lda = LdaModel(corpus=mm, id2word=vocab, num_topics=50, passes=10)
    
    print(lda.print_topics(8))
    
if __name__ == "__main__":
    print("here we go...")
    main()
    print("all pau!")