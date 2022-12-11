import string
import nltk
from nltk.corpus import stopwords
import random
import math

N = 5 # num of sample texts
random.seed(42)

def optional_parser(raw_text):
    return nltk.word_tokenize(raw_text)

def get_words_bia(raw_text, p = False, lan = "eng"): # this funcs works a bit different with corpuses, for this we can specify the flag
    if not p:
        pass
    else:
        file = open(raw_text, 'r', encoding = "utf-8") # read file
        raw_text = optional_parser(file.read()) # get tokens
        file.close() # close



    stop_words = set(stopwords.words("english")) if lan == "eng" else set(stopwords.words("russian"))
    
    rule = lambda a: True if (a not in (string.punctuation+"«»") and a not in stop_words) else False

    
    words = [i.lower() for i in raw_text if rule(i)] # remove stop words and puncs

    biagrams = [] # I decided to get the biagrams manually here, although we can do with nltk in one line

    i,j = 0,1

    while j < len(words):
        pair = (words[i],words[j])
        biagrams.append(pair)
        i += 1
        j += 1
        
    #print(biagrams)
    return biagrams,list(set(words)) # return biagrams and vocab



def frequency(biag,words): 
    res = {}
    for word in words: # for each word

        cur = [] # list all words following current word
        
        for pair in biag: # for each pair
        
            if word == pair[0]: # if word is in the 1st place - then some word follows this word
                cur.append(pair[1]) # store this word in the cur list

        res[word] = {} 
        
        for analog in cur: # now, calculate probaility for each word to be selected
            prob = cur.count(analog)/len(cur)
            res[word][analog] = prob

    #print(res)
            
    return res
        
                



def generate_text(dic,words, bound = 100): # bound mean max of 100 words to generate
    r_ind = random.randrange(0,len(words)) 
    count_bad = 0
    cur_word = words[r_ind] # chose randonly word from vocab
    
    bound = min(bound, len(dic)) # if the length of vocab is less than bound

    res = [cur_word]
    
    while bound-1: # cuz we alredy selected one word
        pairs = dic[cur_word] # get the pos words

        if len(pairs) == 0: # the only case if that it is the last word in the text if it has nothing behind it
            return " ".join(res),count_bad # return generated text

        # here we'll do the following - we will chose each word which has the difference between the mean of probabilities and some random value from uniform dist
        mean = sum(pairs.values())/len(pairs) # probs mean
        otkl = random.uniform(0,mean) # random value
        cur = [] # list all words which passed this "test". 
        
        for key,value in pairs.items():
            if otkl >= abs(value - mean): # if current prob minus mean is less than or = to the random value, meaning that this difference is not that signifficant
                cur.append(key)
        
        if len(cur) == 0: # if all words have very low prob
            cur_word = random.choice(list(pairs.keys())) # just chose randomly
            count_bad += 1 
        else:
            chose = {k: v for k, v in sorted(pairs.items(), key=lambda item: item[1])} # sort the seleted words
            choice = math.ceil(random.uniform(0, len(chose))) # chose nth the most frequent word
            while choice and len(chose) > 0: # we will pop the items from dic untill we reach the nth word
                item = chose.popitem()
                cur_word = item[0]
                choice -= 1
            
        res.append(cur_word)
        bound -= 1


    return " ".join(res),count_bad # return text
        
        

def spec_to_p(dic):
    i = 100
    for key,value in dic.items():
        if not i:
            break
        print(key,value)
        i -= 1


def main(path):  # the func we imported
    biag,words = get_words_bia(path, True) # get words and their biagrans
    f = frequency(biag,words) # get freq (cond probabilities)
    result_test,c = generate_text(f, words) # and generate the result
    return result_test

    

if __name__ == "__main__":
    text = main("sample.txt")
    print(text)
    

