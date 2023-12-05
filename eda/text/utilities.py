# -*- coding: utf-8 -*-
""" eda.text.utilities provides functions for cleaning and tokenizing the
    Reuters 21578 corpus.
"""
import nltk
import re
from .stopwords import stopwords

__LEMM = nltk.WordNetLemmatizer()

__STOPWORDS = stopwords()

__PATT = r'''(?x)
# entities
(?<=[\<\+])([a-zA-Z0-9 \.]+)(?=[\>|\+])

# contractions
| (?<=[ ])(n't|'d|'m|'s|'ll|'re|'ve)(?=[ ])

# currency and percentages, e.g. $12.40, 82%
| (?<=[\$ ])([0-9]+[0-9,\.]+[0-9]+)(?=[% ])

# abbreviations, e.g. U.S.A.
| ([A-Z]\.)+

# major cities / countries, attempt #1
| (?<![a-zA-Z])((Buenos|Cape|East(ern)?|El|Fort|Hong|Las|Los|New|North(ern)?|Puerto|Port|Rio(\sde)?|Saint|San|St\.|São|Sierra|Saudi|South(ern)?|United|West(ern)?)[\s][A-Z][a-z]+)(?![a-zA-Z])

# major cities / countries, attempt #2
| (?<![a-zA-Z])(([A-Z][a-z]+)\s(City|Island(s)|Republic))(?![a-zA-Z])

# words with optional internal hyphens
| \w+([-/]\w+)*

# ellipsis
| \.\.\.

# these are separate tokens; includes ], [
| [!"#$%&'()*+,-./:;<=>?@[]\^_{|}~]
'''

def clean(text):
    """ Cleans the argument text, in the context of the Reuters 21578 dataset.

        - replace tabs with whitepace
        - replace new-line characters with whitespace
        - deletes double-quotation marks
        - deletes those &#03; control characters
        - surrounds negative-contraction n't with whitespace
        - surrounds the other contractions with whitespace
    """
    # Gensim Dictionary is tab-delimited,
    # so we cannot allow allow any tabs in the text
    cleaned = re.sub(r'\t', r' ', text)

    # we don't need line breaks in the text
    cleaned = re.sub(r'\n', r' ', cleaned)

    # same for quotation marks
    cleaned = re.sub(r'"', r'', cleaned)

    # delete those control chars in *.sgm
    cleaned = re.sub(r'\x03', r'', cleaned)

    # surround negative contractions w/ whitespace
    cleaned = re.sub(r"(?<=(ai|ca|do|is|wo))(n't)(?!=[a-z])", r' \2 ', cleaned, flags=re.I)
    cleaned = re.sub(r"(?<=(are|did|had|has|sha|was))(n't)(?!=[a-z])", r' \2 ', cleaned, flags=re.I)
    cleaned = re.sub(r"(?<=(does|have|must|need|were))(n't)(?!=[a-z])", r' \2 ', cleaned, flags=re.I)
    cleaned = re.sub(r"(?<=(could|might|ought|would))(n't)(?!=[a-z])", r' \2 ', cleaned, flags=re.I)
    cleaned = re.sub(r"(?<=should)(n't)(?!=[a-z])", r' \1 ', cleaned, flags=re.I)

    # surround pronoun contractions w/ whitespace
    cleaned = re.sub(r"(?<=i)('d|'ll|'m|'ve)(?!=[a-z])", r' \1 ', cleaned, flags=re.I)
    cleaned = re.sub(r"(?<=he)('d|'ll|'s)(?!=[a-z])", r' \1 ', cleaned, flags=re.I)
    cleaned = re.sub(r"(?<=she)('d|'ll|'s)(?!=[a-z])", r' \1 ', cleaned, flags=re.I)
    cleaned = re.sub(r"(?<=we)('d|'ll|'re|'ve)(?!=[a-z])", r' \1 ', cleaned, flags=re.I)
    cleaned = re.sub(r"(?<=they)('d|'ll|'re|'ve)(?!=[a-z])", r' \1 ', cleaned, flags=re.I)
    cleaned = re.sub(r"(?<=you)('d|'ll|'re|'ve)(?!=[a-z])", r' \1 ', cleaned, flags=re.I)

    return cleaned.strip()

def tokenize(text):
    """ Returns the tokens in the given text.

        Tokenization is via NLTK regexp_tokenize.
        Tokens will be lower-cased, with stopwords removed.
        Lemmatization via NLTK lemmative.
    """
    #tokens = [t.lower() for t in nltk.regexp_tokenize(text, __PATT)]
    tokens = [t.lower() for t in nltk.tokenize.wordpunct_tokenize(text)]
    return [__LEMM.lemmatize(t) for t in tokens if t not in __STOPWORDS]
