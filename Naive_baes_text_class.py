import nltk
import re
import numpy as np
import pandas as pd
from collections import Counter
from math import log
from sklearn.naive_bayes import GaussianNB
import os
from nltk.corpus import stopwords
import string


stopwords = set(stopwords.words('english'))


def readFolderContent(path): # first we read labed data
    sents = [] # parsed sentences
    labels = [] # lst of lables
    j = 0
    file_list = os.listdir(path) # get the folder
    
    for filename in sorted(file_list):
        with open(path + '/' + filename, 'r', encoding = 'utf-8') as infile: # each text file
            lines = infile.readlines()[1:] # seperate each sentence

            for sent in lines: # for each sentence
                
                sp = re.split(r"\n|\t", sent) # split by  \n or by \t
                cur_dic = {}
                
                if len(sp) < 3:
                    r = sp[0].split()
                    label = r[0]
                    words = r[1:]

                else:
                    words = sp[1].split()
                    label = sp[0]

                if label == "###":
                    continue
                
                words = [i for i in words if (i not in stopwords and i not in string.punctuation.replace("-", ""))] # wihtout puncs and stop words

                sents.append(" ".join(words))
                cur_dic[label] = [words,j] # memorize that jth senetence has jth label, j is the number of the sentence 
                labels.append(cur_dic) 
                j += 1
                
    return [sents,labels],j # also attach index of the lastly appended labeled sent


def readFolderContent_unlabled(path,ind): # read the unlabled data
    sents = []
    labels = []
    j = ind # j is the index of the lastly appended labeld sentence, with this we will say that j+1th sentece has None as label
    file_list = os.listdir(path)
    for filename in sorted(file_list):
        with open(path + '/' + filename, 'r', encoding = 'utf-8') as infile:
            lines = infile.readlines()[1:]

            for sent in lines:
                cur_dic = {}
                label = None # for each sentnece we can say that its label is None
                sent = sent.strip()
                sp = re.split(r"\n|\t", sent)
                
                if "#" in sp[0]:
                    continue

                sp[0] = " ".join([word for word in "".join([i for i in sp[0] if (i not in string.punctuation.replace("-", ""))]).split() if word not in stopwords]) # wihtout puncs and stop words
                sents.append(sp[0])
                cur_dic[label] = [sp[0].split(),j]
                labels.append(cur_dic)
                j += 1
                
    return sents,labels
            
            

def get_mat(labels,corps):
    data = []
    score = tf_idf(corps) # corps is corpus of docs, where doc is just the sentence
    matrix,vocab = score[0],score[1] 


    # now, we will have to restore the structure - for each document we should assign its own label
    # dic is just the dic of the follwoing form {label:[words of the sentence, index of the sentence (j)]}
    for dic in labels:
        label,values = list(dic.items())[0][0],list(dic.items())[0][1]
        words,doc_id = values
        document = matrix[doc_id] # selct jth document
        inds = [vocab.index(word) for word in vocab] # get inds of the words presented in our document
        scores = [document[i] for i in inds] # get the words by inds
        data.append([label.strip() if label is not None else label] + scores)
    return data,vocab




def tf_idf(docs):
    rule = lambda x : x.split()
    docs = list(map(rule, docs)) # split all strings
    y = []
    N = 0
    
    for doc in docs:
        y += doc
        N += 1
        
    vocab = sorted(list(set(y))) # create vocabuary

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


def create_naive_baes(data):
    clf = GaussianNB()
    X = data.iloc[:,1:data.shape[1]] # except for the label
    Y = data["label"]
    clf.fit(X, Y)
    print("Score of the model: ", clf.score(X,Y))
    return clf


path_1 = "C:/text_class/labeled_dataset"
path_2 = "C:/text_class/unlabeled_dataset"


rock = readFolderContent(path_1)

r,ind = rock[0],rock[1]

# ind is the index of the last labeled tuple in our dataframe


rock_2 = readFolderContent_unlabled(path_2,ind)
unlabled,l = rock_2[0],rock_2[1] # sentences of the unlabeld data, and their labels (just None to maintain the stucture of our dataframe)


x,y = r[0],r[1] # sentences of the labled data, and their labels


r  = get_mat(y+l,x+unlabled) # create matrix in the following form [ [label, vectorized_sentence], [label, vectorized_sentence]]
labeled,v = r[0],r[1]
data = pd.DataFrame(labeled,columns = ["label"]+v) # create a pd.DataFrame
train = data.iloc[:ind, :] # we use for train everything before ind
test = data.iloc[ind:, 1:] # and for our predcitions we will use unlabled data (here label will be None)





model = create_naive_baes(train)


print("Predictions with Naive Baes:\n")

print(model.predict(test)) # see the predictions

print()

preds = model.predict(test)

for i in range(len(unlabled)):
    print(f"label: {preds[i]}, sentence: {unlabled[i]}")











