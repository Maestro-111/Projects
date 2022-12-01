import nltk
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from math import log
import os
from nltk.corpus import stopwords
import string
from nltk.stem.snowball import SnowballStemmer
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamulticore import LdaMulticore
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import gensim
from gensim.models import Phrases
from gensim.models.coherencemodel import CoherenceModel
from matplotlib.ticker import FuncFormatter
import warnings
import time
import math
import re
from collections import defaultdict
from datetime import datetime
import matplotlib.dates
from gensim.test.utils import datapath
import random
import swifter




TOPICS = 21 # limit, till which we are going to search up for the number of topics we want
STEP = 3
toys = 2000


warnings.simplefilter('ignore')


def measure_time(func):
    def wrapper(*args, **kwargs):
        a = time.time()
        res = func(*args,**kwargs)
        b = time.time()
        print(f"{round((b-a)/3600, 3)} long,long hours were spent on {func.__name__}\n")
        return res
    
    return wrapper



# python 3.9 for gensim!

stopwords = set(stopwords.words('english'))
stemmer = nltk.PorterStemmer()  # our stemmer
regex = re.compile('[%s]' % re.escape(string.punctuation+"—”“’"+"0123456789"+"\n\r\/"))


@measure_time
def readFolderContent(path): # read the files
    file_list = os.listdir(path)
    res = []
    
    for filename in sorted(file_list):
        p = path + '/' + filename

        current_file = pd.read_csv(p)
        smth = current_file.iloc[0:toys, :]
        res.append(smth)

        #res.append(current_file)
        break # temporary shit
    #print(pd.concat(res).head()["date"])

    f = pd.concat(res).reset_index(drop = True).loc[:,["date","content"]]
    return f



# note that we use (), not [], because generators will work a bit faster than list comprehesnipns

def text_process(st):
    st = regex.sub('', st) # remove puncs
    st = (token.strip().lower() for token in st.split() if token.strip().lower() not in stopwords) 
    st = (stemmer.stem(token) for token in st)
    st = (token for token in st if token)
    
    st = list(st)

    return " ".join(st)

#print(text_process("Elin went down for the walk!!,/ it has a lot! i the"))

@measure_time
def add_colos(data): # for each docment we will add collocaions of biagrams and triagrams. Like if we have red and wine frequently appeqaring together, we will add red_wine as single word to the document
    df = data
    text_clean= []
    
    for index, row in df.iterrows():
        text_clean.append(row['content'].split())

    bigram = Phrases(text_clean)
    trigram = Phrases(bigram[text_clean])


    for idx in range(len(text_clean)):
        for token in bigram[text_clean[idx]]:
            if '_' in token:
                text_clean[idx].append(token)
        for token in trigram[text_clean[idx]]:           
            if '_' in token:
                text_clean[idx].append(token)
                
    return text_clean



@measure_time
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3): # before the modeling part
    
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):

        model=LdaMulticore(corpus=corpus,id2word=dictionary, num_topics=num_topics,\
                        random_state=100,
                        alpha='symmetric',
                        per_word_topics=True,
                        workers = None) # build the nodel
                
        
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='u_mass') # get its coherence
        coherence_values.append(coherencemodel.get_coherence()) 

        print(f"we are on {num_topics} topics!")
        print()
        
    return model_list, coherence_values
	




@measure_time
def modeling(text_clean):
    
    dictionary = Dictionary(text_clean)
    corpus = [dictionary.doc2bow(doc) for doc in text_clean]


    
    print('Number of unique tokens: %d' % len(dictionary))
    print('Number of documents: %d' % len(corpus))
    print()

    start = 2
    limit = TOPICS
    step = STEP
    
    model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=corpus, texts=text_clean, start=start, limit=limit, step=step)
    # get the coherence list and models list
    
    
    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Topics Num")
    plt.ylabel("Coherence")
    plt.legend(("coherence_values"), loc='best')
    plt.show()

    n_t = int(input("Enter the desired number of topics: ")) # after we see the coherence values, enter the desired number of topics
    print()

    ind = 0

    print("Number of topics: ", list(range(start,limit,step)))
    print()
    
    for topic_num in range(start,limit,step): # if we select 6, but we trained 5,7,9 and so on, adjust the number to 5
        if topic_num <= n_t:
            ind += 1
            continue
        else:
            model = model_list[ind-1]
            n_t = topic_num-step
            break
    else:
        model = model_list[-1]
        n_t = topic_num
        

    print(f"adjusted number of topics: {n_t}")
    print()    

    print("Model has been trained!\n")
 
    d = gensimvis.prepare(model, corpus, dictionary)

    pyLDAvis.save_html(d, 'LDA_Visualization.html') # make the visualization

    return model,corpus,dictionary,n_t


def decorate(ls, length):
    res = [0]*length

    for i in range(len(ls)):
        ind,perc = ls[i]
        res[ind] = perc
        
    return res



@measure_time
def mod_per_doc(model, corpus, n_t):
    """
    In this function we collect the data for the graphs and construct word topic distrubition
    """
    
    dominant_topics = []
    topic_percentages = []
    words_topics = {}
    doc_topics = {i:0 for i in range(n_t)} # we will count the docs for each topic
    
    for i, corp in enumerate(corpus): # corp is the document
        topic_percs, wordid_topics, wordid_phivalues = model[corp]


        for ind,score in wordid_topics:
            if ind in words_topics: # if this word is a part of the topic - count it up
                words_topics[ind].append(score)
            else:
                words_topics[ind] = [score]

        for ind,value in topic_percs: # if this theme is presented in the documents, count it up
            doc_topics[ind] += 1
        
                
        dominant_topic = sorted(topic_percs, key = lambda x: x[1], reverse=True)[0][0] # get the most signifficant topic for each document
        dominant_topics.append((i, dominant_topic))
        topic_percentages.append(topic_percs)

    values = [[i]+decorate(topic_percentages[i],n_t) for i in range(len(corpus))] 
    columns = ["index of doc"]+["Topic# " + str(i) for i in range(n_t)]

    """
    df1 and df2 are the dataframes representing doc-topic and word-topic distr respectively
    """

    df1 = pd.DataFrame((values), columns=columns)

    
    values1 = []

    for key in words_topics:
        sc = words_topics[key]
        bar = [0]*n_t
        for lst in sc:
            for ind in lst:
                bar[ind] += 1
                
        over = sum(bar)
        
        try:
            smth = [key]+[i/over for i in bar]
        except ZeroDivisionError:
            values1.append([key]+[bar])
        
        values1.append(smth)
        
    columns = ["index of word"]+["Topic# " + str(i) for i in range(n_t)]

    df2 = pd.DataFrame(values1, columns=columns)

    most_popular = [i for (i,count) in sorted(doc_topics.items(), key = lambda x : x[1], reverse = True)][0:4]


    print("Document/Topic distribution: ")
    print(df1.head(n=10))
    print()
    print("Word/Topic distribution: ")
    print(df2.head(n=10))
    print()
    
    return(dominant_topics, topic_percentages, doc_topics,most_popular)


@measure_time
def most_sig_topic(model, dominant_topics, topic_percentages,doc_topics,n_t):
    """
    in this function we will use the gathered data to construct the graphs
    """

    tops = list(doc_topics.keys())
    freqs = list(doc_topics.values())
    
    df = pd.DataFrame(dominant_topics, columns=['Document_Id', 'Dominant_Topic'])
    dominant_topic_in_each_doc = df.groupby('Dominant_Topic').size()


    plt.bar(tops,freqs,width=.2)
    plt.title('Number of Documents for each Topic', fontdict=dict(size=10))
    plt.ylabel('Number of Documents')
    plt.xlabel('Topics')
    plt.show()

    topic_weightage_by_doc = pd.DataFrame([dict(t) for t in topic_percentages])
    df_topic_weightage_by_doc = topic_weightage_by_doc.sum().to_frame(name='count').reset_index()
    
    df_dominant_topic_in_each_doc = dominant_topic_in_each_doc.to_frame(name='count').reset_index()
    topic_top3words = [(i, topic) for i, topics in model.show_topics(formatted=False,num_topics =n_t)  # top 3 words for each topics
                                 for j, (topic, wt) in enumerate(topics) if j < 3]

    
    df_top3words_stacked = pd.DataFrame(topic_top3words, columns=['topic_id', 'words'])
    df_top3words = df_top3words_stacked.groupby('topic_id').agg(', \n'.join)
    non_ind = df_top3words.copy()
    df_top3words.reset_index(level=0,inplace=True)

    
    fig, ax1 = plt.subplots(1, 1, figsize=(40, 20), dpi=120, sharey=True)
    ax1.bar(x='Dominant_Topic', height='count', data=df_dominant_topic_in_each_doc, width=.2, color='firebrick')
    ax1.set_xticks(range(df_dominant_topic_in_each_doc.Dominant_Topic.unique().__len__()))
    tick_formatter = FuncFormatter(lambda x, pos: 'Topic ' + str(x)+ '\n' + df_top3words.loc[df_top3words.topic_id==x, 'words'].values[0])
    ax1.xaxis.set_major_formatter(tick_formatter)

    ax1.set_title('Number of Documents by Dominant Topic', fontdict=dict(size=10))
    ax1.set_ylabel('Number of Documents')

    plt.show()

    return non_ind






@measure_time
def filtered_df(df): # filert the dataframe for only 2016-2017 years
        def d(s):
            try:
                
                pat = re.findall(r"(?:2016|2017){1}-.+",s)
                if pat:
                    return 1
                return None
            except TypeError as v:
                return None

                
        df["check"] = df["date"].apply(lambda x : d(x))
        res = []
        for i in range(len(df)):
            if df.loc[i, "check"] == 1:
                res.append(i)
                
        return res



def to_date(tup):
    year,month = tup

    return str(year)+"-"+str(month) 



def sort_dict(d): # sort the dict (by the dates)
    k = list(d.keys())
    dates = []
    res= {}

    for dic in d.values():
        c = {tuple(map(int, date.split("-"))): v for date, v in dic.items()}
        c = {to_date(k): v for k, v in sorted(c.items(), key=lambda item: item[0])}
        dates.append(c)

    for i in range(len(k)):
        res[k[i]] = dates[i]

    return res

    



@measure_time
def sec_graph(df, model, corpus, inds, key_words_df,topics_ind=[0,1,2,3]): # make up time series graph
    print(key_words_df)
    
    n_t = len(topics_ind)

    try:
        
        labels = {i:key_words_df.loc[i,"words"] for i in topics_ind}
    except KeyError:
        print(topics_ind)
        print()
        print(key_words_df)
        exit()

    
    themes = {} 


    for i, corp in enumerate(corpus): # inds is the list of indices of thw rows which are in either 2016 or 2017
        if i not in inds:
            continue

        topic_percs, wordid_topics, wordid_phivalues = model[corp]

        d = re.findall(r"\d{4}-\d{2}",df.loc[i, "date"])[0] # seperare the year-month

        for ind,perc in topic_percs:

            if ind not in topics_ind: # topics inds are the indicies of topics we are intresetd in (4 most popular)
                continue

            # basically, count the documents for these topics
            
            if ind not in themes:
                themes[ind] = {}
            else:
                pass
            
            if d not in themes[ind]:                
                themes[ind][d] =[perc]
            else:
                themes[ind][d].append(perc)
             
    themes = dict(themes)

    for topic_id, dic_dates in themes.items(): # sinnce there can be many occurences for the same date, we will get the average 
        for date, percs in dic_dates.items():
            avg = sum(percs)/len(percs)
            dic_dates[date] = avg

    #print(themes)

    themes = sort_dict(themes) # sort the dictionary based on the data

    
    ops = ["-go","s:m",":o","-*"] # markers

    for topic_id, dic_dates in themes.items():
        marker = random.choice(ops)
        ops.remove(marker)

        xs = []
        ys = []

        for date,average in dic_dates.items(): # apply strptime to convert the oject to date
            tim = datetime.strptime(date, "%Y-%m")
            xs.append(tim)
            ys.append(average)

        dates = matplotlib.dates.date2num(xs) # then converte to this format to be able to use plot_date functinon
        plt.plot_date(dates, ys,marker,label=f"Topic # {labels[topic_id]}")

    plt.legend()
    plt.title('Trend', fontdict=dict(size=10))
    plt.ylabel('Popularity')
    plt.xlabel('Time')
    plt.show()



def save_m(model):
    temp_file = datapath("model")
    model.save(temp_file)
    return temp_file


def load_m(temp_file):
    return LdaModel.load(temp_file)
    


path = "C:/text_min_p/corpus"

def main(path):
    data = readFolderContent(path) # reaad folder
    
    data["content"] = data["content"].apply(text_process) # apply processing func on the content

    only_for_spec = data # copy
    
    inds = filtered_df(data) # get inds for the rows in 2016-2017
    
    data  = add_colos(data) # add colocartions


    model,corpus,dictionary,num = modeling(data) # modeling

    temp_file = save_m(model)

    dominant_topics, topic_percentages, doc_topics, most_popular = mod_per_doc(model,corpus,num) # gather data for the graphs and construct word and document topic distributions

    key_words_df = most_sig_topic(model, dominant_topics, topic_percentages,doc_topics,num) # grpahs 1,2

    sec_graph(only_for_spec, model, corpus, inds,key_words_df,most_popular) # time series graph




if __name__ == "__main__":
    main(path)













