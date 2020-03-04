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

nlp = spacy.load('en_core_web_md')

import gensim
from gensim.models import KeyedVectors

def parse_document(document):
    document = re.sub("\n", " ", document).strip()
    sentences = nltk.sent_tokenize(document)
    sentences = [sent.strip() for sent in sentences]
    return sentences

to_summarize = """ The abusive male himself might be unseen, but the fear he spreads is in plain sight in “The Invisible Man,” Leigh Whannell’s sophisticated sci-fi-horror that dares to turn a woman’s often silenced trauma from a toxic relationship into something unbearably tangible. Charged by a constant psychological dread that surpasses the ache of any visible bruise, Whannell’s ingenious genre entry amplifies the pain of its central character Cecilia Kass (Elisabeth Moss) at every turn, making sure that her visceral scars sting like our own. Sometimes, to an excruciating degree.
It's not an easy feat to accomplish. Partly because Whannell’s playground has its boundaries set within a pre-existing property that ought to be handled with care—James Whale’s circa 1933 pre-code classic, adapted from H.G. Wells’ 1897 novel—that is, if we learned anything from various lackluster studio remakes of recent years. But mostly because we are in the era of #MeToo, with the once-protected monsters of the real world finally being exposed for what they are, their terrorizing powers examined in stupendous films like Kitty Green’s “The Assistant”—a long-delayed revolution that shouldn’t be cheapened or misused. Thankfully, the Australian writer/director behind the wildly successful “Saw” and “Insidious” franchises, comes equipped with both sufficient visual panache—“The Invisible Man” recalls David Fincher’s Bay Area-set masterwork “Zodiac” and the mazy quality of James Cameron’s spine-tingling “Terminator 2: Judgment Day” when you least expect it—and fresh ideas to fashion the classic Universal Movie Monster with timeless and timely anxieties. And he does so in startlingly well-considered ways, updating something familiar with an inventive take. 
It wouldn’t be a stretch to suggest that part of what Green prioritized with her masterpiece is also what lends “The Invisible Man” (and eventually, its visible woman robbed out of options) its cumulative strength—an unforgiving emphasis on the loneliness emotional violence births in the mistreated. There is a constant in all the sharply edited, terrifying set pieces lensed by Stefan Duscio with elegant, clever camera moves in bedrooms, attics, restaurants and secluded mansions: a vigilant focus on Cecilia’s isolation. That isolation, intensified by Benjamin Wallfisch’s fiendish score, happens to be her concealed assailant’s sharpest knife. A deadly weapon others refuse to see and acknowledge.
One relief is, Whannell doesn’t ever leave us in a state of bewilderment in front of his mean, handsomely-styled and absorbing thriller. We believe Cecilia through and through, when others, perhaps understandably, refuse to do so, questioning her sanity instead. (Sure, “the crazy woman no one will listen to” is a long-exploited cliché, but rest assured, in Whannell’s hands, this by-design bug eventually leads to a deeply earned conclusion.) And yes, at least we as the audience are by her side, all the way from the film’s taut opening when Cecilia wakes up with a long-harbored purpose next to her sleeping enemy, but not showing traces of Julia Roberts’ fragility. Instead, we detect something both mighty and vulnerable in her, closer to Sarah Connor of "The Terminator" in spirit, when she forcefully runs through the woods to escape her cruel partner Adrian (Oliver Jackson-Cohen), gets picked up by her sister Alice (Harriet Dyer) after some heart-stopping setbacks and takes refuge with her childhood best friend James (Aldis Hodge)—a resourceful cop living with his teenaged daughter Sydney (Storm Reid), who dreams of going to a design school they can’t afford.
The initially agoraphobic Cecilia finally claims her freedom back, at least briefly, when the moneyed scientist Adrian commits suicide, leaving Cecilia a healthy sum that would finance both her future and Sydney’s choice of college. Of course, if something is too good to be true, it probably is, no matter what Adrian’s brother Tom (a brilliantly sinister Michael Dorman) claims, handling his late sibling’s estate and inheritance. In that, Cecilia soon puts the pieces of the puzzle together, discovering that Adrian had invented an armor of invisibility (dear reader, this good-looking piece of scientific artifact is the premise, not a spoiler), which he would be using for a complex scheme of gaslighting as a sadistic form of revenge—a reality she can’t prove to anyone. There will be floating knives, pulled comforters, and eerie footprints. You might let out a scream or two.
The certified contemporary queen of unhinged screen heroines—just consider “Her Smell,” “The Handmaid’s Tale,” “Us” and the upcoming “Shirley” collectively—Moss excels in these creepy scenes with her signature verve. As Cecilia who resourcefully fights an undetectable authority that ruins her life and controls her psychological wellbeing, Moss continues to deliver what we crave from woman characters: the kind of messy yet sturdy intricacy many of today’s thinly conceived you-go-girl female superheroes continue to lack. Whannell’s script and direction generously allow Moss the room to stretch those complex, varied muscles, while casually winking at an empowered final girl for this side of the 21st century. """

from gensim.summarization import summarize

def summarize_text(text, summary_ratio=0.5):
    summary = summarize(text, split=True, ratio=summary_ratio)
    for sentence in summary:
        print(sentence)

sent_to_summ = parse_document(to_summarize)
prep_to_summ = " ".join(sent_to_summ)

summarize_text(prep_to_summ, 0.3)