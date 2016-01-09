from glob import iglob
from nltk.corpus import stopwords
from os import path

__stopwords = set(stopwords.words("english"))
for f in iglob(path.join(path.dirname(__file__), "*.txt")):
    with open(f) as fin:
        __stopwords.update([e.strip() for e in fin.readlines()])

def stopwords():
    return __stopwords
