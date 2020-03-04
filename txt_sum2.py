import re
import nltk
from html import unescape
from contractions import CONTRACTION_MAP
import spacy
from collections import Counter
import pandas as pd
import numpy as np
import json
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


nlp = spacy.load('en_core_web_md')

def parse_document(document):
    document = re.sub("\n", " ", document).strip()
    sentences = nltk.sent_tokenize(document)
    sentences = [sent.strip() for sent in sentences]
    return sentences

def tokenize_text(text):
    return [token.strip() for token in nltk.word_tokenize(text)]

def expand_contraction(text, contraction_mapping):
    pattern = re.compile("({})".format("|".join(contraction_mapping.keys())), 
                         flags=re.IGNORECASE|re.DOTALL)
    def expand(contraction):
        match = contraction.group(0)
        first_char = match[0]
        exp_contraction = contraction_mapping.get(match.lower())
        if exp_contraction:
            exp_contraction = first_char + exp_contraction[1:]
        else:
            exp_contraction = match
        return exp_contraction
    
    exp_text = pattern.sub(expand, text)
    return exp_text

def lemmatize_text(text, nlp):
    doc = nlp(text)
    lemmatized_tokens = [token.lemma_ if token.lemma_ != "-PRON-" else token.text.lower() for token in doc]
    return " ".join(lemmatized_tokens)

def rm_special_chars(text):
    tokens = tokenize_text(text)
    pattern = re.compile("[^a-zA-Z0-9]")
    filtered_tokens = [pattern.sub("", token) for token in tokens if pattern.sub("", token)]
    return " ".join(filtered_tokens)

def rm_stopwords(text, stopwords=nltk.corpus.stopwords.words("english")):
    tokens = tokenize_text(text)
    filtered_tokens = [token for token in tokens if token.lower() not in stopwords]
    return " ".join(filtered_tokens)

def normalize_corpus(corpus, tokenize=False, lemmatize=True,
                     contraction_mapping=CONTRACTION_MAP, nlp=nlp, 
                     stopwords=nltk.corpus.stopwords.words("english")):
    norm_corpus = []
    for text in corpus:
        txt = unescape(text)
        txt = expand_contraction(txt, contraction_mapping)
        if lemmatize:
            txt = lemmatize_text(txt, nlp)
        else:
            txt = txt.lower()
        txt = rm_special_chars(txt)
        txt = rm_stopwords(txt, stopwords)
        if tokenize:
            txt = tokenize_text(txt)
            norm_corpus.append(txt)
        else:
            norm_corpus.append(txt)
    return norm_corpus

def build_feature_matrix(documents, feature_type="frequency", ngram_range=(1,1)):
    feature_type = feature_type.strip().lower()
    
    if feature_type == "binary":
        vect = CountVectorizer(binary=True, ngram_range=ngram_range)
    elif feature_type == "frequency":
        vect = CountVectorizer(binary=False, ngram_range=ngram_range)
    elif feature_type == "tfidf":
        vect = TfidfVectorizer(ngram_range=ngram_range)
    else:
        raise Exception("Possible feature types: 'binary', 'frequency', 'tfidf'")
    
    feat_matrix = vect.fit_transform(documents).astype(float)
    
    return vect, feat_matrix

doc_raw = (
"""The abusive male himself might be unseen, but the fear he spreads is in plain sight in “The Invisible Man,” Leigh Whannell’s sophisticated sci-fi-horror that dares to turn a woman’s often silenced trauma from a toxic relationship into something unbearably tangible. Charged by a constant psychological dread that surpasses the ache of any visible bruise, Whannell’s ingenious genre entry amplifies the pain of its central character Cecilia Kass (Elisabeth Moss) at every turn, making sure that her visceral scars sting like our own. Sometimes, to an excruciating degree.

It's not an easy feat to accomplish. Partly because Whannell’s playground has its boundaries set within a pre-existing property that ought to be handled with care—James Whale’s circa 1933 pre-code classic, adapted from H.G. Wells’ 1897 novel—that is, if we learned anything from various lackluster studio remakes of recent years. But mostly because we are in the era of #MeToo, with the once-protected monsters of the real world finally being exposed for what they are, their terrorizing powers examined in stupendous films like Kitty Green’s “The Assistant”—a long-delayed revolution that shouldn’t be cheapened or misused. Thankfully, the Australian writer/director behind the wildly successful “Saw” and “Insidious” franchises, comes equipped with both sufficient visual panache—“The Invisible Man” recalls David Fincher’s Bay Area-set masterwork “Zodiac” and the mazy quality of James Cameron’s spine-tingling “Terminator 2: Judgment Day” when you least expect it—and fresh ideas to fashion the classic Universal Movie Monster with timeless and timely anxieties. And he does so in startlingly well-considered ways, updating something familiar with an inventive take. 

It wouldn’t be a stretch to suggest that part of what Green prioritized with her masterpiece is also what lends “The Invisible Man” (and eventually, its visible woman robbed out of options) its cumulative strength—an unforgiving emphasis on the loneliness emotional violence births in the mistreated. There is a constant in all the sharply edited, terrifying set pieces lensed by Stefan Duscio with elegant, clever camera moves in bedrooms, attics, restaurants and secluded mansions: a vigilant focus on Cecilia’s isolation. That isolation, intensified by Benjamin Wallfisch’s fiendish score, happens to be her concealed assailant’s sharpest knife. A deadly weapon others refuse to see and acknowledge.

One relief is, Whannell doesn’t ever leave us in a state of bewilderment in front of his mean, handsomely-styled and absorbing thriller. We believe Cecilia through and through, when others, perhaps understandably, refuse to do so, questioning her sanity instead. (Sure, “the crazy woman no one will listen to” is a long-exploited cliché, but rest assured, in Whannell’s hands, this by-design bug eventually leads to a deeply earned conclusion.) And yes, at least we as the audience are by her side, all the way from the film’s taut opening when Cecilia wakes up with a long-harbored purpose next to her sleeping enemy, but not showing traces of Julia Roberts’ fragility. Instead, we detect something both mighty and vulnerable in her, closer to Sarah Connor of "The Terminator" in spirit, when she forcefully runs through the woods to escape her cruel partner Adrian (Oliver Jackson-Cohen), gets picked up by her sister Alice (Harriet Dyer) after some heart-stopping setbacks and takes refuge with her childhood best friend James (Aldis Hodge)—a resourceful cop living with his teenaged daughter Sydney (Storm Reid), who dreams of going to a design school they can’t afford.

The initially agoraphobic Cecilia finally claims her freedom back, at least briefly, when the moneyed scientist Adrian commits suicide, leaving Cecilia a healthy sum that would finance both her future and Sydney’s choice of college. Of course, if something is too good to be true, it probably is, no matter what Adrian’s brother Tom (a brilliantly sinister Michael Dorman) claims, handling his late sibling’s estate and inheritance. In that, Cecilia soon puts the pieces of the puzzle together, discovering that Adrian had invented an armor of invisibility (dear reader, this good-looking piece of scientific artifact is the premise, not a spoiler), which he would be using for a complex scheme of gaslighting as a sadistic form of revenge—a reality she can’t prove to anyone. There will be floating knives, pulled comforters, and eerie footprints. You might let out a scream or two.

The certified contemporary queen of unhinged screen heroines—just consider “Her Smell,” “The Handmaid’s Tale,” “Us” and the upcoming “Shirley” collectively—Moss excels in these creepy scenes with her signature verve. As Cecilia who resourcefully fights an undetectable authority that ruins her life and controls her psychological wellbeing, Moss continues to deliver what we crave from woman characters: the kind of messy yet sturdy intricacy many of today’s thinly conceived you-go-girl female superheroes continue to lack. Whannell’s script and direction generously allow Moss the room to stretch those complex, varied muscles, while casually winking at an empowered final girl for this side of the 21st century.""")

step1 = parse_document(doc_raw)
step2 = normalize_corpus(step1, lemmatize=True)
v, m = build_feature_matrix(step2, feature_type="tfidf")
sim_mat = (m * m.T)
import networkx
sim_graph = networkx.from_scipy_sparse_matrix(sim_mat)



networkx.draw_networkx(sim_graph, cmap=plt.get_cmap('jet'))
plt.show()


scrs = networkx.pagerank(sim_graph)
ranked_sents = sorted(scrs.items(), key=lambda x: x[1], reverse=True)

top_s_idxs = [x[0] for x in ranked_sents[:6]]
top_s_idxs.sort()
top_s_idxs

for idx in top_s_idxs:
    print(step1[idx])