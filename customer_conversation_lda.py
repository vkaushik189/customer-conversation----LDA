# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 21:47:23 2018

@author: kaushik
"""
"""
Steps for Lda:
    1. Get the raw data in proper foramt
    2. Preprocess the data(i.e. lowercase,remove punctation, stopwords, lemmatize)
    3. Convert the df to [[what, is, time, today],[when, are , you, coming],[how, are, you]] format using (for w in w .split)
    4. Create a dictionary from the above list
    5. Create a doc_term matrix from above dictionary and list
    6. Apply LDA on the above doc_term matrix
"""


import pandas as pd
import numpy as np
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn
import glob
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import gensim
from gensim import corpora


#reading the files and getting it into a dataframe format
path = "C:\\git\\LDA and LSA --ubuntu dialog conversations\\test"
allfiles = glob.glob(path + "/*.tsv")
frame = pd.DataFrame()
list_ = []
for file in allfiles:
    df = pd.read_csv(file,index_col=None, sep='\t', header=None, error_bad_lines=False, names = ['date','person1','person2','conversation'])
    list_.append(df)
frame = pd.concat(list_)

data = pd.DataFrame(frame.iloc[:,3]).reset_index(drop=True)

x=0
y=0
newdf = pd.DataFrame(np.nan, index = range(0,int((data.shape[0]/4))), columns = ['conversation'])
for i in range(0, int((data.shape[0]/4))):
    newdf.conversation[x] = data.conversation[y] + " " +data.conversation[y+1] + " " + data.conversation[y+2] + " " + data.conversation[y+3]
    x += 1
    y += 4
    


#the data is in good format now, Let's start preprocessing it
#initially lowercase and remove the punctuations
def clean(text):
    text=text.lower()
    text=re.sub("\\n"," ",text)
    text=re.sub("\d{1,}","",text)
    text=re.sub("\.{1,}",".",text)
    text=re.sub("\:{1,}","",text)
    text=re.sub("\;|\=|\%|\^|\_|\*|\'"," ",text)
    text=re.sub("\""," ",text)
    text=re.sub("\'{2,}","",text)
    text=re.sub("\/|\!"," ",text)
    text=re.sub("\?"," ",text)
    text=re.sub("\#"," ",text)
    text=re.sub("\,|\@|\|\*"," ",text)
    text=re.sub("\(|\)"," ",text)
    text=re.sub("\S+jpg"," ",text)
    text=re.sub("\S*wikip\S+","",text) 
    text=re.sub("\[.*?\]"," ",text)
    text=re.sub("\-"," ",text)
    text=re.sub("[.,]"," ",text)
    '''text=re.sub("\"|:|@|,|\/|\=|;|\.|\'|\?|\!|\||\+|\~|\-|\#"," ",text)
    text=re.sub("\_"," ",text)
    '''
    text=re.sub("www\S+","",text)
    text=re.sub("http","",text)
    text=re.sub("com","",text)
    text=re.sub(r'[^\x00-\x7F]+',' ', text) # remove non ascii
    text=re.sub("\s+"," ",text)
    text = ' '.join( [w for w in text.split() if len(w)>1])
    text = text.strip()
    return text

newdf['conversation'] = newdf['conversation'].apply(lambda x: clean(x))


#apply stopwords and lemmatize them
stopw = set(stopwords.words('english'))
lemma = WordNetLemmatizer()
def preprocess(doc):
    doc = " ".join([w for w in doc.split() if w not in stopw])
    doc = " ".join(lemma.lemmatize(word) for word in doc.split())
    return doc

#to get the dataframe in a list format, separated by conversations 
df_list = [preprocess(doc) for doc in newdf['conversation']]

#to get the dataframe in words separated format(to be able to create a dict)
df_list_fordict = [preprocess(doc).split() for doc in newdf['conversation']]



#now create a dictionary from the above list(list(separated words)) file
dictionary = corpora.Dictionary(df_list_fordict)
print(len(dictionary))
print(dictionary)

#now create a doc-term matrix from the abpve dictionary and list[list(separatedwords)] file
doc_term_matrix = [dictionary.doc2bow(doc) for doc in df_list_fordict]
print(len(doc_term_matrix))
print(doc_term_matrix[:3])


#Now let's do LDA
LDA = gensim.models.ldamodel.LdaModel
lda_model = LDA(doc_term_matrix, id2word = dictionary, num_topics = 5, iterations = 20, passes=150, minimum_probability=0.0)
#alpha = high --> docs are composed of more topics
#alpha = low --> docs are composed of less topics
#Beta = high --> topics are composed of large number of words
#Beta = low --> topics are composed of low number of words


#to display the topics
lda_model.print_topics(num_topics = 5, num_words=7)

#to know the distribution of each topic
lda_corpus = lda_model[doc_term_matrix]
#shows the topic distibution of each doc
print(lda_corpus[6])