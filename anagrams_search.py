import numpy as np
import re
import collections
import time
import string
import random
import sys
"""
name: Elin Huliiev
student id : 163890189
Program description:

The main goal is to organize the data in such way, that the time for finding all anograms for one word would be O(1)
We are going to define overall complexity later
So, firstly we are grouping our words in hash table. Notice, that anograms will be having the same hash always since letters in words are the same. Our hash table is a big list of lists, we are calculating the hash for each word, it becomes index and based on this index we place the word into corresponding list.
For example, words binary and brainy will be in the same list since their hash is the same. This is huge advantage now, because previously we has to scan the whole array to find all anograms and that is O(N^2), but now it is enougth to just go through the new list since if some word has anogarms they will be in the same list for 100%
However, we still have to scan the whole list, even the size of it is way less, to find annograms. To fix this every time we will sort the list based on the words that we have. To clarify this - in our intermediate hash table values are in this format [[],[],[],[(word, frequncy), (word, frequency)]]
Them we sort each non empty array by words so that annograms will be close to each other. Imagine words binary and brainy - in the array that is not sorted we don't know the location of binary, if we are at brainy and looking for the anograms for it. However, if we sort the array
binary and briany will be right to each other, since we are sorting by letters and they have the same letters! So, for any word, if is has anogarms, in a sorted array it is enougth to look at the right word - if is not anogarm - then there is no anogrma to the right for this word and same for the left.
In the other words - for any word it is enought to look and right and left only and then stop - it is O(1). Now, we have to sort each array in our intermediate hash table  - it is O(i*klogk) where i is the numebr of array in table and k is the length of the array. Finally, we have to scan each word
so it becomes O(i*(klogk + k)). It is overall complexity of the getting_anograms function.

"""


def measure_time(func): # for measuring time
    def wrapper(txt):
        a = time.time()
        c = func(txt)
        b = time.time()
        print(f"\nTime spent on this function - {func.__name__}: {b-a}, in seconds\n")
        return c
    return wrapper


@measure_time # ignore this function
def generate_w(k):
    x = list(string.ascii_lowercase)
    f = []
    for i in range(k):
        res = ""
        for ii in range(7):
            ind = random.randint(0, len(x)-1)
            res += x[ind]
        f.append(res)
    return f
          




class HashMap(): # Haah Table
    def __init__(self, maximum = 100):
        self.max = maximum
        self.arr = [[] for i in range(self.max)]

        
    def get_hash(self, key):
        return sum([ord(i) for i in key]) % self.max


    def __setitem__(self, key, val):
        h = self.get_hash(key)
        self.arr[h].append(val)

    def __getitem__(self, word): # find function
        result = []
        h = self.get_hash(word)
        massive = self.arr[h]

        for some_word, its_anogram, frequncies in massive:
            if (its_anogram == word):
                result.append(f"The initial word - |{word}| was found and its frequnecy in the text is: |{frequncies}|")
                continue
            
            if (sorted(word) == sorted(its_anogram) and word != its_anogram):
                result.append(f"The anogram of initial word - |{word}| was found: anogram: |{its_anogram}| and its frequnecy for the |{word}| in text is : |{frequncies}|")
                continue
            
            continue
        
        if len(massive) == 0 or len(result) == 0:
            return f"Word - {word} is absent"
        
            
        return result                

    def __iter__(self):
        return iter(self.arr)


    def __str__(self):
        return str(self.arr)

    def __len__(self):
        return len(self.arr)

    def size(self):
        k = 0
        for i in self.arr:
            k += len(i)
        return k
        

def little_rep(st): # for deleting punc
    res = ""
    puncs = '''!()[]{};:'"\,<>.?@#$%^&*_~'''
    for i in st:
        if i not in puncs:
            res += i
    return res
 

@measure_time
def text_process(txt): # text processing
    lst = []
    try:
        with open(txt, 'r', encoding = 'utf-8') as text:
            for line in text:
                line = little_rep(line.strip().lower())
                possible_nums = re.findall(r'\d+', line)
                for num in possible_nums:
                   line = line.replace(num, "")
                line =  line.strip()
                line = line.split()
                if line != []:
                    lst += line
                else:
                    continue
    except FileNotFoundError as f:
        print("wrong file name")
        return 
    
    return lst


def get_left(lst, i): # go to the left
    res = []
    t = i
    while i >= 1:
        if sorted(lst[t][0]) == sorted(lst[i-1][0]) and lst[t][0] != lst[i-1][0]: # if word to the left is anagram - move on to the left more, see this function used in getting_anograms
            res.append([lst[i-1][0], lst[i-1][1]]) # set the word by hash to the new hash map object
            i -= 1
            continue
        else:
            break # else there is no need to analyze the rest - they are not anagrams
        
    return res


def get_right(lst, i): # the same as for the left
    res  = []
    t = i
    while i < len(lst)-1:
        if sorted(lst[t][0]) == sorted(lst[i+1][0]) and lst[t][0] != lst[i+1][0]:
            res.append([lst[i+1][0], lst[i+1][1]])
            i += 1
            continue
        else:
            break
        
    return res



def frequency(array,num): # for counting frequencyt
    i = np.where(array[0] == num)[0][0]
    return array[1][i]




@measure_time
def group_in_hash(txt): # "grouping words by hash"
    if txt is None:
        return
    map_object = HashMap(900) # create object Hash Table
    unique = (i for i in list(set(txt))) # only unique words and make it generator to consume less memory
    
    fu = lambda x : int("".join([str(ord(u)) for u in x])) # lambda function to convert words into the sequence of ints, or decoding the word
    
    txt = np.array(list(map(fu,txt))) # we convert each word in text to number, we do this to speed up frequency calculation
    
    (hell, counts) = np.unique(txt, return_counts=True) # count all frequncies, hell is the set of unique elements, in our case set of the unqiue numbers and our lambda function guarntees us that all word will be having unique "codes"
    
    table = (hell, counts)
    for word in unique:
        code = fu(word) # produce the "code" of the word
        map_object[word] = (word, frequency(table, code)) #look up for its frequncy and add it along with the word to the Hash Table
    return map_object



@measure_time
def getting_anograms(map_object):
    if map_object is None:
        return

    """
    map_object2 = HashMap(200)
    for possible_annogram in map_object: # for each list in Hahs Table
        if possible_annogram == []:
            continue
        else:
            massive = possible_annogram
            massive = sorted(massive, key = lambda y: sorted(y[0])) # sort the list by the fisrt element, where fitsrt element is just the word
            for i in range(len(massive)): # for each word in massive
                map_object2 = get_left(massive,i,map_object2) # and we update it with new possible anogarms by going to the left and right, this is O(1)
                map_object2 = get_right(massive,i,map_object2)
                continue

    
    return map_object2
    """

    res = []
    
    for possible_annogram in map_object: 
        if possible_annogram == []:
            continue
        else:
            massive = possible_annogram
            massive = sorted(massive, key = lambda y: sorted(y[0]))

            for i in range(len(massive)): # for each word in massive
                sub = [massive[i][0]]
                sub += get_left(massive,i)
                sub += get_right(massive,i)
                res.append(sub)
    
    return res
            
    


    
        
        
def print_st(map_object): # print hash Table
    for i in map_object:
        if i != []:
            print(i)
    return 
    



def helper(): # calling all the functions
    processed = text_process("chapter.txt") # note tha path
    #processed = text_process("C:/pro/war_and_p.txt") #this is much bigger text, you may try it also
    getting_the_hash = group_in_hash(processed)
    analysis = getting_anograms(getting_the_hash)
    if analysis is None:
        return -1
    else:
        print_st(analysis)
        return analysis 


def main():
    a = helper() # hash object
    if a == -1:
        print("try again")
        return

    print()
    print(a["night"]) # part 2, getting the words from the table
    print(a["apple"])
    print(a["pie"])
    print(a["no"])
    print(a["war"])
    print(a["death"])
    print(a["rats"])
    return 


if __name__ == '__main__':
   main()



