# -*- coding: utf-8 -*-
""" eda.text.utilities provides functions for cleaning and tokenizing the
    Reuters 21578 corpus.
"""

import re
import spacy

nlp = spacy.load("en_core_web_md")

NER_KEEP = [
    "ORG", 
    "GPE", 
    "PERSON", 
    "NORP",
    "PRODUCT", 
    "WORK_OF_ART", 
    "FACILITY", 
    "EVENT", 
    "LOC", 
    "LAW", 
    "MONEY", 
    "DATE"
]

NER_IGNORE = [
    "CARDINAL", 
    "QUANTITY", 
    "ORDINAL", 
    "PERCENT", 
    "TIME"
]

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

    return cleaned.strip()


def tokenize(text):
    """ Returns the tokens in the given text.

        Tokenization and filtering via spaCy's English NLP pipeline.
    """
    token_acc = []
    entity_acc = []

    for token in nlp(text):
        if token.ent_iob_ in ("B", "I") and token.ent_type_ in NER_KEEP:
            entity_acc.append( token.text )
        else:
            if len(entity_acc) > 0:
                token_acc.append( "_".join([e.lower() for e in entity_acc]) )
                entity_acc = []
            if not (token.ent_type_ in NER_IGNORE or
                    token.is_punct or
                    token.is_space or
                    token.is_bracket or
                    token.is_stop):
                token_acc.append( token.text.lower() )
    if len(entity_acc) > 0:
        token_acc.append( "_".join([e.lower() for e in entity_acc]) )

    return token_acc
