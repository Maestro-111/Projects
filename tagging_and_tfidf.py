from collections import Counter
from math import log
import numpy as np
from math import sqrt
import string
import nltk
import re
from nltk.stem.porter import *


def tagging_a(): # caution - tooks a couple of mins to run
    stemmer = PorterStemmer()
    
    corps_tagged = [i for i in nltk.corpus.brown.tagged_words(tagset='universal') if i[1] == "NOUN"] 

    stems = 0

    res = {stemmer.stem(tag[0]):{1:0,-1:0} for tag in corps_tagged} # stem each word

    # for each word, 1 is reponsible for the plural endings, and -1 for the singulars

    i = 0

    for word in res.keys():
        for tag in corps_tagged:
            if word in tag[0]: # if our stem in a word
                if tag[0][-1] == "s": # see if it edns with an s, if so increase the dic[1]
                    res[word][1] += 1
                else: # else dic[-1]
                    res[word][-1] += 1
            else:
                continue
        i +=1

    plurals = []

    for key,value in res.items():
        if value[1] > value[-1]:
            plurals.append(key)



    return plurals


def tagging_d():
    corps_tagged_bias = nltk.bigrams(nltk.corpus.brown.tagged_words(tagset='universal'))

    res = []


    for t1,t2 in corps_tagged_bias: # if the next one is NOUN get the tag
        if t1[1] == "NOUN":
            res.append(t2[1])


    return dict(sorted(Counter(res).items(), key = lambda x: x[1], reverse = True)) # sort them, to find out the most common tags following after nouns
            

    

def tagging_c():
    res = {}

    corps_tagged = nltk.corpus.brown.tagged_words(tagset='universal')

    for word, tag in corps_tagged: # store each tag in a dictioary, count their frequencies
        if tag not in res:
            res[tag] = 1
        else:
            res[tag] += 1

    s = sorted(res.items(), key = lambda x : x[1], reverse = True) # sort
    
    return s[:20] # 1st 20


def tagging_b():
    words = sorted([i for i in list(set(nltk.corpus.brown.words())) if i not in string.punctuation + " "]) # analyze text?? for puncs
    corps_tagged = sorted([i for i in nltk.corpus.brown.tagged_words(tagset='universal') if i[0] not in string.punctuation + " "], key = lambda x: x[0])
    dic_b = {}
    i = 0
    j = 0

    while j < len(words) and i < len(corps_tagged):
        word = words[j]
        t = corps_tagged[i]

        if t[0] == word:
            if word not in dic_b:
                dic_b[word] = {t[1]:1}
            else:
                current = dic_b[word]
                if t[1] not in current:
                    dic_b[word][t[1]] = 1
                else:
                    dic_b[word][t[1]] = dic_b[word][t[1]]+1

        else:
            j += 1
            continue
        
        i += 1


    m = [-float('inf'), {}] # note that there are multiple words with a maximum number of distinct tags
    
    for key,value in dic_b.items():
        s = len(value.values())

        if s == m[0]: # is just like the max - store it
            m[1][key] = (s, value)


        if s > m[0]: # if we found new max - delete all the values below new threshhold

            copy = m[1].copy()

            for key_1,value_1 in m[1].items():
                if value_1[0] < s:
                    copy.pop(key_1)
    
            m[1] = copy
            m[0] = s
            m[1][key] = (s,value)

    return m[1] # word: (the max num of dist tags, and the tags)
                    



def frequency_vector(docs):
    rule = lambda x : x.split()
    docs = list(map(rule, docs))

    y = []

    for doc in docs:
        y += doc
        
    vocab = list(set(y))
    
    l = [Counter(doc) for doc in docs]
    
    matrix = [[0] * len(vocab) for i in range(len(docs))]

    for i in range(len(matrix)):
        document = l[i]
        for j in range(len(matrix[0])):
            word_f = document.get(vocab[j], 0) + 1
            matrix[i][j] = word_f
            
    return matrix,vocab


def tf_idf(docs):
    rule = lambda x : x.split()
    docs = list(map(rule, docs)) # split all strings
    y = []
    N = 0
    
    for doc in docs:
        y += doc
        N += 1
        
    vocab = list(set(y)) # create vocabuary

    l = [Counter(doc) for doc in docs]
    
    idfs = {}
    
    freqs = []

    id_doc = 0
    
    for d in l: # for each Counter object
        s = sum(d.values()) # total number of words in a doc
        d_doc = {}
        
        for word,count in d.items(): # count frequency of a particular word
            d_doc[word] = count/s

        freqs.append((id_doc, d_doc)) # make sure to save a number of a doc
        id_doc += 1

    freqs = dict(freqs)

    for word in vocab: # idfs will be only one for each unique word
        c = 0
        for doc in docs:
            if word in doc: # count the num of docs where thr word occurs
                c += 1
        idfs[word] = log(N/c)
    #print(idfs)
    

    matrix = [[0]*len(vocab) for i in range(N)] # vectorization

    for doc_n in range(N):
        documnet_freq = freqs[doc_n] # get the particular doc
        for i in range(len(vocab)): # for each word in thshis doc
            tf_idf = documnet_freq.get(vocab[i], 0) * idfs[vocab[i]] # compuet tf * idf
            matrix[doc_n][i] = tf_idf
            
    return matrix,vocab
        
"""
# task 1
print("one")
print(tagging_a())
print()
print(tagging_b())
print()
print(tagging_c())
print()
print(tagging_d())
print()
print("two")
print()

# task 2, not in paper


def compute_cos(vec1,vec2):
    scalar = np.dot(vec1,vec2)
    length = lambda x : sqrt(sum([y**2 for y in x]))
    l1,l2 = length(vec1),length(vec2)
    return scalar/(l1*l2)




d1 = "information information data train"
d2 = "computer information cpu computer"
d3 =  "computer retrieval information"

res = tf_idf([d1,d2,d3])

vectorized_text,voc = res[0],res[1]

print(voc)
print()


i = 0 
for v in vectorized_text:
    print(f"doc number {i}, vector: {v} \n")
    i += 1

print()
print(compute_cos(vectorized_text[0],vectorized_text[0])) # between first and first
print(compute_cos(vectorized_text[0],vectorized_text[1])) # between first and second
print(compute_cos(vectorized_text[0],vectorized_text[2])) # between first and third
#print(compute_cos(vectorized_text[1],vectorized_text[2]))
print()
print("three")
print()

# task 3
path = "corp_4.txt" # path

corp = []

file = open(path,"r").readlines()
for l in file:
    l = "".join([i for i in l if i not in string.punctuation])
    corp.append(l)

res = tf_idf(corp)

vectorized_text,voc = res[0],res[1]

print(voc)
print()

i = 0 
for v in vectorized_text:
    print(f"doc number {i}, vector: {v} \n")
    i += 1

res = frequency_vector(corp)

vectorized_text,voc = res[0],res[1]


print(voc)
print()

i = 0 
for v in vectorized_text:
    print(f"doc number {i}, vector: {v} \n")
    i += 1

"""
from sklearn.naive_bayes import GaussianNB

d1 = "follow-up metting"
d2 = "free cash get money"
d3 = "money money money"
d4 = "dinner plans"
d5 = "get cash now"
d6 = "get money now"


x = [d1,d2,d3,d4,d5,d6]
res = frequency_vector(x)

m,v = res[0],res[1]

for i in range(len(m)-1):
    m[i] = [m[i],"not spam"] if i in [0,3] else [m[i], "spam"]


#for i in m:
#    print(i)


model = GaussianNB()
data = m[:len(m)-1]
print(data)
x = [i[0] for i in data]
y = [i[1] for i in data]
print(x)
print(y)
model.fit(x,y)
print(model.predict([m[-1]]))
