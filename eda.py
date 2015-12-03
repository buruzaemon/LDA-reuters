# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 21:58:58 2015

@author: buruzaemon
"""
import glob
import os
import re
from bs4 import BeautifulSoup
from nltk import regexp_tokenize
from os import path


# glob .sgm files
DATADIR = path.abspath(path.join(os.getcwd(), "data"))
DATAEXT = "*000.sgm" 

PATT = r'''(?x)    # set flag to allow verbose regexps
    ([A-Z]\.)+          # abbreviations, e.g. U.S.A.
    | \w+(-\w+)*        # words with optional internal hyphens
    | \$?\d+(\.\d+)?%?  # currency and percentages, e.g. $12.40, 82%
    | \.\.\.            # ellipsis
    | [][.,;"'?():-_`]  # these are separate tokens; includes ], [
'''

files_iter = glob.iglob(path.join(DATADIR, DATAEXT))

documents = []

for fname in files_iter:
    with open(fname, "r", encoding='ASCII') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')
        for body in soup.find_all("body"):
            txt = re.sub(r"\n\x03?", " ", body.text)
            doc = regexp_tokenize(txt, PATT)
            documents.append(doc)
        
        