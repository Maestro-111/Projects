
import math
import time
import json
import re
import sys
#import numpy as np
#from numpy import linalg as la
#import os
#from numba import njit, jit

# внизу глобальные переменные для регулировки чувствительности алгоритма
NUMBER_OF_MINS = 5 
NUMBER_OF_ADR = 15 # к-то возможных адресов, если больше то повтор (??) 
DETECT_OF_INACCURACY = math.pi
DETECT_OF_INACCURACY_2 = math.pi ** 2

def count_t(some_func): # декоратор, для вычесления времени работы
    def call_func():
        point1 = time.time()
        res = some_func()
        point2 = time.time()
        print("\nMinutes: " + str((point2-point1)/60))
    return call_func


def main_menu():
    #.encode(sys.stdout.encoding, errors = 'replace')
    print("%56s" % "------------------------------")
    print("%43s" % "Menu")
    print("%57s" % "------------------------------\n")
    print("Options: ")
    print("\nEnter 1, in order to select English")
    print("Введите 2, для выбора русского языка")
    print("Введiть 3, щоб обрати українську мову\n")
    try :
        a = int(input("Your choice: "))
    except ValueError as v:
        print("\n" ,v, "\n")
        print("Error\n")
        return 0
    while a != 1 and a !=2 and a != 3:
        print("\nEnter 1, in order to select English")
        print("Введите 2, для выбора русского языка")
        print("Введiть 3, щоб обрати українську мову\n")
        try :
            a = int(input("Your choice: "))
        except ValueError as v:
            print("\n" ,v, "\n")
            print("Error\n")
            return 0
    if a == 1:
        return "eng"
    elif a == 2:
        return "ru"
    else:
        return "ua"

def choice(lan):# функция выбора, если юзер при неточном запросе захочет повторить
    print("\nOptions: ")
    if lan == 'eng':
        print("\nEnter 1 if you would like to try again\n")
        print("Enter 2 if you would not like to try again\n")
        out = int(input("Here: "))
        while out != 1 and out != 2:
            print("\nEnter 1 if you would like to try again\n")
            print("Enter 2 if you would not like to try again\n")
            out = int(input("Here: "))
    elif lan == "ru":
        print("\nВведите 1 если вы хотите повторить\n")
        print("Введите 2 если вы не хотите повторить\n")
        out = int(input("Введите тут: "))
        while out != 1 and out != 2:
            print("\nВведите 1 если вы хотите повторить\n")
            print("Введите 2 если вы не хотите повторить\n")
            out = int(input("Введите тут: "))
    elif lan == "ua":
        print("\nВведіть 1 якщо ви хочете продовжити\n")
        print("Введіть 2 якщо ви не хочете продовжити\n")
        out = int(input("Введіть тут: "))
        while out != 1 and out != 2:
            print("\nВведіть 1 якщо ви хочете продовжити\n")
            print("Введіть 2 якщо ви не хочете продовжити\n")
            out = int(input("Введіть тут: "))
    return out


def enter_the_house_number(lan):
    if lan == "eng":
        print("\nEnter the number bewlow\n")
        try:
            out = int(input("Here: "))
            return out
        except ValueError:
            return 0
    elif lan == "ru":
        print("\nВведите номер здесь внизу\n")
        try:
            out = int(input("Здесь: "))
            return out
        except ValueError:
            return 0
    elif lan == "ua":
        print("\nБудь ласка введіть номер унизу\n")
        try:
            out = int(input("Тут: "))
            return out
        except ValueError:
            return 0
        
     
                
                  
        

def ana_inner(lst, lan):
    if lst == [] and lan == 0:
        return -1
    
    if len(lst) == 2 and lst[0] == True:
        return DETECT_OF_INACCURACY
    if len(lst) == 4 and lst[0] == True:
        return DETECT_OF_INACCURACY_2

    
    if lan == "eng":
        key = "Stop"
        if len(lst) == 0:
            return -1
        print("Print Stop, if you need to quit\n")
        for x in lst:
            print(x+' <---> is this your request? \n')
            u = input("Enter your response here: ")
            while u != 'Yes' and u!= 'No' and u!= key:
                
                print("\nprint Yes or No or Stop\n")
                u = input("Enter your response here: ")
            if u == "Yes":
                return u, x
            elif u == "No" and lst.index(x)+1 <= len(lst)-1:
                print("\nYou probably meant:\n")
                continue
            elif u == key:                
                break
            else:
                break
        if u == "No" or u == key:
            return -1
    elif lan == "ru":
        key = "Стоп"
        if len(lst) == 0:
            return -1
        print("Введите Стоп, если Вы хотите выйти\n")
        for x in lst:
            print(x+' <---> это Ваш запрос?\n')
            u = input("Введите Ваш ответ здесь: ")
            while u != 'Да' and u!= 'Нет' and u!= key:
                print("\nВведите Да или Нет\n")
                u = input("Введите Ваш ответ здесь: ")
            if u == "Да":
                return u, x
            elif u == "Нет" and lst.index(x)+1 <= len(lst)-1:
                print("\nВозможно Вы имели в виду:\n")
                continue
            elif u == key:                
                break   
            else:
                break
        if u == "Нет" or u == key:
            return -1
    else:
        key = "Стоп"
        if len(lst) == 0:
            return -1
        print("Введіть Стоп, якщо Ви бажаете вийти\n")
        for x in lst:
            print(x+' <---> це те, що Ви мали на увазі?\n')
            u = input("Введіть вашу відповідь тут: ")
            while u != 'Так' and u!= 'Ні' and u!= key:
                print("\nВведіть Так або Ні\n")
                u = input("Введіть вашу відповідь тут: ")
            if u == "Так":
                return u, x
            elif u == "Ні" and lst.index(x)+1 <= len(lst)-1:
                print("\nМожливо, Ви мали на увазі:\n")
                continue
            elif u == key:                
                break   
            else:
                break
        if u == "Ні" or u == key:
            return -1
        
def analyze_language(lst_of_distinct, adr):
    for letter in lst_of_distinct:
        if letter in adr:
            return False
        else:
            continue
    return True

#@njit
def analyze_little_str(some_str):
    try:
        int(some_str)
        return True
    except ValueError:
        return False


def detect_str(some_str : str): # детектим индексы украинских і, І, чтобы в последствии их заменить
    letter = "і"
    letter_2 = "І"
    lst_of_ind_1 = []
    lst_of_ind_2 = []
    ind = 0
    while ind < len(some_str):
        if some_str[ind] == letter:
            lst_of_ind_1.append(ind)
        elif some_str[ind] == letter_2:
            lst_of_ind_2.append(ind)
        ind += 1
    return [lst_of_ind_1, lst_of_ind_2]

def user_input(lan): # обработка юзеровской строки
    if lan == 0:
        return 0, 0
    if lan == "eng":
        u = input("\nPlease, enter adress here: ")
    elif lan == "ru": # замена 
        u = input("\nПожалуйста, введите адрес здесь: ")
        inds = detect_str(u)
        for x in (inds[0]):
            u = u.replace(u[x], "i")
        for x in (inds[1]):
            u = u.replace(u[x], "I")
    else:
        u = input("\nБудь ласка, введіть адрессу тут: ")
    cop = (u + '.')[:-1]
    if lan == "ru": # заменяем укр буквы на англ буквы I и і, чтобы pypy3 не ругался
        inds_2 = detect_str(cop)
        for x in inds_2[0]:
            cop = cop.replace(cop[x], "i")
        for x in inds[1]:
            cop = cop.replace(cop[x], "I")
    point = u.split()
    for x in range(len(point)):
        point[x] = point[x].replace('.', ' ')
        if not(point[x].isalpha() or point[x].isdigit()):
            point[x] = point[x].replace(',', ' ')
        else:
            continue
    u = " ".join(point)
    punctuations = '''!()[]{};:'"\,<>.?@#$%^&*_~'''
    no_punct2 = ""
    for pp in u:
        if pp not in punctuations:
            no_punct2+=pp
    u = no_punct2
    #u = u.replace("/", " ")
    u = u.lower()
    return u, cop



def spec_for_algo2(user_txt): # функция необходима для работы КМП - она разбивает строку юзера на 5 частей
    r = user_txt.split()
    if len(r) == 1:
        return [str(r[0])+"" for i in range(5)]
    elif len(r) == 2:
        a,b,c,d = r[0],r[math.floor(len(r)*0.25)], r[math.floor(len(r)*0.75)], r[-1]
        return [a,b,c,d]
    else:
        a,b,c,d,f = r[0],r[math.floor(len(r)*0.25)], r[math.floor(len(r)/2)], r[math.floor(len(r)*0.75)],  r[-1]
        return [a,b,c,d,f]
                                        
def text_process(txt, option): # обработка адрессов из справочника
    if option == 0:
        return 0, 0
    lst_of_ru = ["ы","э", "ъ", "ё"]
    lst_of_ua = ["ї", "і", "є", "ґ"]
    data = []
    data2 = []
    with open(txt, 'r', encoding='utf-8-sig') as f:
        text = json.load(f)

    val = list(text.values())
    for x in val:
        for xx in x:
            for xxx in xx:
                if xxx == "Data":
                    data.append(xx[xxx])
    for y in data:
        for yy in y:
            if yy["language"] == option:
                if option == 'ru' and analyze_language(lst_of_ua, yy["StreetName"]):
                    a = yy["CityName"]
                    b = yy["StreetName"]
                    data2.append(a+" " + b)
                elif option == 'ua' and analyze_language(lst_of_ru, yy["StreetName"]):
                    a = yy["CityName"]
                    b = yy["StreetName"]
                    data2.append(a+" " + b)
                else:
                    a = yy["CityName"]
                    b = yy["StreetName"]
                    data2.append(a+" " + b)
                    continue

    lst_of_str = data2 # в адресах из базы тоже меняем укр буквы на англ, чтобы везде была аналогичная замена
    if option == "ru":
        for x in range(len(lst_of_str)):
            lst_of_inds = detect_str(lst_of_str[x])
            for y in lst_of_inds[0]:
                lst_of_str[x] = lst_of_str[x].replace(lst_of_str[x][y], "i")
            for yy in lst_of_inds[1]:
                lst_of_str[x] = lst_of_str[x].replace(lst_of_str[x][yy], "I")
                
    punctuations = '''!()[]{};:'"\,<>.?@#$%^&*_~'''
    for y in range(len(lst_of_str)):
        lst_of_str[y] = lst_of_str[y].replace('\n','')
        
    for z in range(len(lst_of_str)):
        no_punct = ""
        for zz in lst_of_str[z]:
            if zz not in punctuations:
                no_punct+=zz
        lst_of_str[z] = no_punct
            
    lst_of_str_2 = lst_of_str.copy()
    for w in range(len(lst_of_str)):
        lst_of_str[w] = lst_of_str[w].lower()

    return lst_of_str, lst_of_str_2



    


def algo_kmp(text_1, massive, detect = False, lst_detect = False): # ищет прообраз каждого элемента, их всего 5, в справочнике
    key_list = []
    final_list = []
    j = 0
    i = 1
    sim = 0
    elements = []
    elements_2 = []
    #print(text_1)
    if detect == False and lst_detect == False:
        for words in text_1: # выбираем подстроки из разбитых слов, что бы сделать алгоритм менее чуствительным к опечаткам
            if len(words) >= 5:
                le = math.floor(len(words)/2)
                el = math.floor(len(words)*0.25)
                lel = math.floor(len(words)*0.75)
                f_50 = words[:el]
                s_50 = words[el:le]
                r_50 = words[le:lel]
                d_50 = words[lel:-1]
                elements.append(f_50)
                elements.append(s_50)
                elements.append(r_50)
                elements.append(d_50)
            elif len(words) == 4:
                le = math.floor(len(words)/2)
                a = words[:le]
                b = words[le:]
                elements.append(a)
                elements.append(b)
            elif len(words) == 3:
                elements.append(words[::])
            elif len(words) == 2:
                elements.append(words[::])
            else:
                f_50 = words[:1]
                elements.append(f_50)
    elif lst_detect == False and detect == True:
        for words in text_1.split():
            elements.append(words)
    elif lst_detect == True and detect == False:
        elements = text_1
    #print(elements)
    scg = 0
    for y in elements:
        p = [0]*len(y) # мапссив суфиксов, ну или здесь алгоритм будет смотреть на какой именно элемент ему сдвигать, что бы не пропустить возможный прообраз
        while i < len(y):
            if y[j] == y[i]:
                p[i] = j+1
                i += 1
                j += 1
            else:
                if j == 0:
                    p[i] = 0
                    i += 1
                else:
                    j = p[j-1]
        # на каждой итерации самого верхнего цыкла фор, массив пи будет высчитыватся для каждого из элементов
        for x in massive:
            m = len(y)
            n = len(x)
            i = 0
            j = 0
            while i < n:
                if detect == False and lst_detect == False:
                    pass
                    #print(i,j,n,y)
                if x[i] == y[j]:  # x это один из адрессов, а игрек это один из 5 элементов
                    i += 1
                    j += 1
                    if j == m: # Если дошли до конца и нашли образ
                        sim+=1
                        key_list.append([x,y,'Образ найден'])
                        break
                else:
                    if j > 0:
                        j = p[j-1]
                    #if j == (m-1): # ????, при адресе Ломоносова 121, возможен баг
                        #break
                    else: # только если индекс J не ссылается на первый элемент 
                        i += 1
            if i == n and j != m:
                key_list.append([x,y,'Образ не найден'])
    #print(key_list)       
    sg = []
    for adress in massive:
        num = 0
        for element in key_list:
            if (adress == element[0]) and element[2] == 'Образ найден': # для каждого адресса считаем к-то найденных образов и групируем все в словарь
                num+=1
        sg.append([adress, num])
    sg = dict(sg)
    #print(sg)
    val = 0
    for value in sg.values():
        val += value
    if val == 0:
        return []
    else:
        detect = max(sg.values())
        mob = sorted(set([i for i in sg.values()]))
        #print(mob)
        if len(mob) == 1:
            sec_dec = mob[0]
        elif len(mob) == 2 and mob[0] != 0:
            sec_dec = mob[0]
        elif len(mob) == 2 and mob[0] == 0:
            sec_dec = detect
        else:
            sec_dec = mob[len(mob)-2]
        for key, value in sg.items():
            if value == detect or value == sec_dec:
                final_list.append(key)
        return final_list



def distance(a, b): # ахождение метрики Левенштейна, где ищется именно метрика
    n, m = len(a), len(b)
    if n > m:
        # убедимся что n <= m, чтобы использовать минимум памяти O(min(n, m)) - признаюсь честно - этот "быстрый" алгоритм я подсмотрел в гугле - он работает быстрей, чем если бы мы сделали тупо по формуле с рекрусиоными вызовами, 
        a, b = b, a
        n, m = m, n
    current_row = range(n + 1)  # здесь важно то, что для нахождение конкретной метрики, необязательно "запоминать" предыдущею строку
    for i in range(1, m + 1):
        previous_row, current_row = current_row, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete, change = previous_row[j] + 1, current_row[j - 1] + 1, previous_row[j - 1] # операции изменения строки: добавить элемент, убарать элемент или поменять элемент
            if a[j - 1] != b[i - 1]:
                change += 1
            current_row[j] = min(add, delete, change) # выбираем минимальную операцию
    return current_row[n]



#?
def magic_function(lst_of_addresses, user_input): # функция для вывода больше 20 адресов, при "плохом" запросе.
    #print(lst_of_addresses)
    lst_1 = []
    lst_2 = []
    lst_3 = []
    inter = []
    flag_1 = "Number"
    lst_of_addresses_2 = lst_of_addresses.copy()
    #print(user_input)
    for x in range(len(user_input)):
        if not (user_input[x].isalpha()):
            lst_1.append(user_input[x])
    user_input = list(set(user_input).difference(set(lst_1)))
    for x in range(len(lst_of_addresses_2)):
        for xx in lst_of_addresses_2[x].split():
            if not (xx.isalpha()):
                lst_2.append(xx)
    #print(lst_2)
    #print("")
    for adr in range(len(lst_of_addresses_2)):
        ind = {}
        lst_of_addresses_2[adr] = lst_of_addresses_2[adr].split()
        count_2 = 0
        for elem in lst_of_addresses_2[adr]:
            ind[count_2] = elem
            count_2 += 1
        lst_of_addresses_2[adr] = list(set(lst_of_addresses_2[adr]).difference(set(lst_2)))
        lst_of_addresses_3 = lst_of_addresses_2[adr].copy()
        #print(ind)
        for key, value in ind.items():
            if value in lst_of_addresses_2[adr]:
                    try:
                        lst_of_addresses_3[key] = value
                        continue
                    except IndexError:
                        lst_of_addresses_3.append(value)
        lst_of_addresses_2[adr] = lst_of_addresses_3
        lst_of_addresses_2[adr]  = " ".join(lst_of_addresses_2[adr])
    #print(lst_of_addresses_2)
    if len(list(set(lst_of_addresses_2))) == 1:
        return [flag_1, list(set(lst_of_addresses_2))]

def matrix_construct(lst_of_words, er, matrix, one = False):
    if one == False:
        for x in range(len(lst_of_words)): # если в адресе есть n-ое слово, мы аппендим 1, а иначе 0
            for xx in range(len(er)):
                if er[xx] in lst_of_words[x]:
                    matrix[x][xx] = 1
                else:
                    continue
        return matrix
    else:
        for x in range(1): # если в адресе есть n-ое слово, мы аппендим 1, а иначе 0
            for xx in range(len(er)):
                if er[xx] in lst_of_words:
                    matrix[x][xx] = 1
                else:
                    continue
        return matrix


def my_inner(vector_1, vector_2):
    length = len(vector_1)
    ind = 0
    sum_of = 0
    while ind <= (length - 1):
        res = vector_1[ind] * vector_2[ind]
        sum_of += res
        ind+=1
    return sum_of

def dot_product(matrix_1, matrix_2, lst_2):
    tuy = []
    temp = 0
    lst_of_indices = []
    count_1 = 0
    for x in matrix_2:
        for xx in x:
            if xx == 1:
                lst_of_indices.append(count_1)
            count_1 += 1
    for x in matrix_2: # здесь, мы считаем скалярн произведение строк в матрицах, где опять же - строки это предложения - юзеровский адрес с каждым адресом в справочнике
        if sum(x) == 0:
            return []
        for y in matrix_1:
            temporary = 0
            for ind in lst_of_indices:
                temporary += y[ind]
            if temporary == 0:
                temp += 1
                continue  
            res = my_inner(x,y)
            tuy.append([lst_2[temp], res])
            temp+=1

    tuy = dict(tuy) # для удобного анализа - словар
    return tuy




                
#@njit
def cal_inner(txt, lst, lst_2, approximation = False): # алгоритм нахождения скалярного произведения векторов-адресов, для превращении адреса в ветор, я использивал Bag of Words
    point3 = time.time()
    if txt == 0  and lst == 0 and lst_2 == 0:
        return []
    seperate = spec_for_algo2(txt)
    txt = txt.split()
    lst_of_words = []
    lst_of_words_2 = []
    er = {}
    for x in lst:
        y = x.split()
        lst_of_words.append(y)
    for x in lst_of_words:
        for word in x:
            lst_of_words_2.append(word)
    for x in lst_of_words_2:
        if x not in er:
            er[x] = 1
        else:
            er[x] += 1
    #print(len(er))
    matrix = [0]*len(lst_of_words) # матрица, где каждый row репрезентует предложение, то есть адрес, а каждая column есть слово, уникальное слово    
    for x in range(len(lst_of_words)):
        matrix[x]=[0]*len(er)
    er = list(map(str, er))
    point1 = time.time()
    matrix = matrix_construct(lst_of_words, er, matrix)
    point2 = time.time()
    #print("\nMinutes: " + str((point2-point1)/60))
    matrix_2 = [0] * 1 # тоже самое для юзеровской строки, но здесь у нас только один row, так как только одно предолжение-адрес
    for x in range(1):
        matrix_2[x] = [0] * len(er)
    lst_of_bad_words = []
    for z in range(len(txt)):
        t = 0
        if len(txt[z]) == 1 and not(analyze_little_str(txt[z])):
            lst_of_bad_words.append(txt[z])
            continue
        for zz in range(len(er)):
            if txt[z] == er[zz]:
                t+=1
        if t >= 1:
            continue
        else:
            lst_of_bad_words.append(txt[z])
    txt = list(set(txt).difference(set(lst_of_bad_words)))
    lst_of_mins = []

    #print(lst_of_bad_words)
    #### ОСНОВА
    for x in range(len(lst_of_bad_words)):
        if (len(lst_of_bad_words[x]) == 1 and not(analyze_little_str(lst_of_bad_words[x]))) or (len(lst_of_bad_words)==1 and analyze_little_str(lst_of_bad_words[x])):
            continue
        b = {}
        metr = algo_kmp(lst_of_bad_words[x], lst, detect = True)
        #print(metr, lst_of_bad_words[x])
        if len(metr) == 0:
            pass
        elif len(metr) >= NUMBER_OF_ADR:
            continue
        else:
            bob = {}
            word = []
            for st in metr:
                for ts in st.split():
                    word.append(ts)
            word = list(set(word))
            sec_metr = algo_kmp(lst_of_bad_words[x], word, detect = True)
            for adr in range(len(sec_metr)):
                lst_of_mins.append(sec_metr[adr])
            continue
        for xx in range(len(er)):
            a = distance(lst_of_bad_words[x], er[xx])
            b[er[xx]] = a
        min_dis = min(b.values())
        t = 0
        for value in b.values():
            if value == min_dis:
                t+=1

        """
        #служебная информац
        
        lst_of_dis = list(b.values())
        dict_of_dis = {}
        for values in lst_of_dis:
            if values not in dict_of_dis:
                dict_of_dis[values] = 1
            else: 
                dict_of_dis[values] += 1
        """
        if t >= NUMBER_OF_MINS:
            continue
        else:
            for key, value in b.items():
                if  value == min(b.values()) or (value>min(b.values()) and value <= ((len(key)//2)-1)):
                    lst_of_mins.append(key)
        ####
    txt += lst_of_mins
    #print(txt)
    matrix_2 = matrix_construct(txt, er, matrix_2, one = True)
    tuy = dot_product(matrix, matrix_2, lst_2)  # для удобного анализа - словарь
    if isinstance(tuy, list) and len(tuy) == 0:
        return []
    #print(tuy)
    sorted_np = []
    detect = max(tuy.values())
    #print(detect)
    for key, value in tuy.items():
        if value == detect:
            sorted_np.append(key)
    #print(sorted_np)
    if len(sorted_np) > NUMBER_OF_ADR:
        approximation = True
        #print(sorted_np)
        what_should_we_do = magic_function(sorted_np, txt)
        if what_should_we_do != None:
            return [approximation, sorted_np] + what_should_we_do
        return [approximation, sorted_np]
    new_dict = {}
    matrix_3 = [0]*len(sorted_np)
    for x in sorted_np:
        for xx in x.split():
            if xx in new_dict:
                new_dict[xx]+=1
            else:
                new_dict[xx] = 1
    new_dict = list(map(str, new_dict))
    for p in range(len(sorted_np)):
        matrix_3[p] = [0] * len(new_dict)
    for x in range(len(sorted_np)):
        for xx in range(len(new_dict)):
            if new_dict[xx] in sorted_np[x]:
                matrix_3[x][xx] = 1
            else:
                continue
    glob = []
    spa = []
    #print(sorted_np)
    for x in range(len(matrix_3)):
        if len(matrix_3) == 1:
            return sorted_np
        lst_of_dots = []
        maxis = []
        for xx in range(len(matrix_3)):
            dot = my_inner(matrix_3[x],matrix_3[xx])
            a = sorted_np[x]
            b = sorted_np[xx]
            c = (b,dot)
            lst_of_dots.append([[a,b],dot])
        #print(lst_of_dots)
        for y in lst_of_dots:
            maxis.append(y[1])
        maxis = max(maxis)
        for y in lst_of_dots:
            if y[1] == maxis:
                glob.append(y[0])
    #print(glob)
    for x in range(len(glob)):
        if len(glob[x]) == 1:
            if glob[x][0] not in spa:
                spa.append(glob[x][0])
        else:
            w = {}
            w[0] = glob[x][0]
            w[1] = glob[x][1]
            glob[x] = list(set(glob[x]))
            if len(glob[x]) == 1:
                if glob[x][0] not in spa:
                    spa.append(glob[x][0])
            else:
                if glob[x][0] != w[0]:
                    glob[x][0],glob[x][1] = glob[x][1], glob[x][0]
                if (glob[x][0] not in spa) and (glob[x][1] not in spa):
                    c = (glob[x][0],glob[x][1])
                    spa.extend(c)
    sorted_np  = spa
    #print(len(('\n1',sorted_np)))
    sorted_2 = sorted_np.copy()
    for x in range(len(sorted_2)):
        sorted_2[x] = sorted_2[x].lower()
    res = algo_kmp(seperate, sorted_2)
    spare = []
    for x in res:
        for y in sorted_np:
            if x == y.lower():
                spare.append(y)
    sorted_np = spare
    lst_of_str_with_nums = []
    dict_of_nums = {}
    new_lst_of_srt = []
    # финальная сортировка адресов, так как на этом моменте их минимум 2 и до 15. Сортировка по номерам домов
    for x in sorted_np:
        if not(x.isalpha()):
            lst_of_str_with_nums.append(x)
    sorted_np = list(set(sorted_np).difference(set(lst_of_str_with_nums)))
    for x in lst_of_str_with_nums:
        nums = re.findall(r'\d+', x) # отделяем строки с числами и потом их сортируем
        nums = [int(i) for i in nums]
        if len(nums) == 1:
            nums = nums[0]
        else:
            sorted_np.append(x)
            continue
        dict_of_nums[lst_of_str_with_nums.index(x)] = nums
    #print(dict_of_nums)
    vals = list(set((dict_of_nums.values())))
    vals.sort()
    #print(vals)
    for x in vals:
        for key,value in dict_of_nums.items():
            if x == value:
                new_lst_of_srt.append(lst_of_str_with_nums[key])
                if len(vals) == 1:
                    continue
                else:
                    break
    new_lst_of_srt += sorted_np # плюс чистые строки
    #print(new_lst_of_srt)
    point4 = time.time()
    print("\nSeconds: " + str((point4-point3)))
    return new_lst_of_srt
            

# серия вспомогательных функций, где и происходит вызов всего           

def _help1(detect = False, detect_2 = False, detect_3 = False):
    try:
        if detect == False and detect_2 == False and detect_3 == False:
            ident = main_menu()
            a, z = user_input(ident)
            #b,c = text_process("C:/task_12/adresses_3.json", ident)
            b,c = text_process("C:/Users/La_Admin/Desktop/winetime_task/adresses.json", ident) # в корень диска С
            d = cal_inner(a, b, c)
            return [d, ident, z]
        elif detect != False and detect_2 != False and detect_3 == False:
            a, z = user_input(detect)
            lst_of_low = [x.lower() for x in detect_2]
            # b,c = text_process("adresses_3.json", detect)
            d = cal_inner(a, lst_of_low, detect_2)
            return [d, detect, z]
        else:
            # ернуть копию маленькйо и большой строки 
            a = enter_the_house_number(detect)
            if a == 0:
                return 10
            a = str(a)
            lst_of_low = [x.lower() for x in detect_2]
            new_input = detect_3[0] + " " + a
            z = (new_input + '.')[:-1]
            new_input = new_input.lower()
            d = cal_inner(new_input, lst_of_low, detect_2)
            return [d, detect, z]
            
    except FileNotFoundError:
        return 7
    
def part_1():
    a = _help1()
    if a == 7:
        return 8
    answ = ana_inner(a[0], a[1])
    if answ == DETECT_OF_INACCURACY and a[1] == "eng":
        print("\nYour request ---> " + a[2] + " <--- is not clear, please repeat")
        choice_of_user = choice(a[1])
        while choice_of_user == 1:
            a = _help1(a[1], a[0][1])
            if True not in a[0]:
                answ = ana_inner(a[0], a[1])
                break
            else:
                print("\nYour request ---> " + a[2] + " <--- is not clear, please repeat")
                choice_of_user = choice(a[1])
        else:
            return math.e, a[1]
    elif answ == DETECT_OF_INACCURACY and a[1] == "ru":
        print("\nВаш запрос ---> " + a[2] +  " <--- неточен, пожалуйста повторите")
        choice_of_user = choice(a[1])
        while choice_of_user == 1:
            a = _help1(a[1], a[0][1])
            if True not in a[0]:
                answ = ana_inner(a[0], a[1])
                break
            else:
                print("\nВаш запрос ---> " + a[2] +  " <--- неточен, пожалуйста повторите")
                choice_of_user = choice(a[1])
        else:
            return math.e, a[1]
    elif answ == DETECT_OF_INACCURACY and a[1] == "ua":
        print("\nВаш запит ---> " + a[2] + " <--- не дуже точний, будь ласка повторіть знову")
        choice_of_user = choice(a[1])
        while choice_of_user == 1:
            a = _help1(a[1], a[0][1])
            if True not in a[0]:
                answ = ana_inner(a[0], a[1])
                break
            else:
                print("\nВаш запит ---> " + a[2] + " <--- не дуже точний, будь ласка повторіть знову")
                choice_of_user = choice(a[1])
        else:
            return math.e, a[1]
    
    if answ == DETECT_OF_INACCURACY_2 and a[1] == "eng":
        print("\nYour request ---> " + a[2] + " <--- is not clear, please enter the house number")
        choice_of_user = choice(a[1])
        while choice_of_user == 1:
            a = _help1(a[1], a[0][1], a[0][3])
            if a == 10:
                return 11
            if True not in a[0]:
                answ = ana_inner(a[0], a[1])
                break
            else:
                print("\nYour request ---> " + a[2] + " <--- is not clear, please enter the house number")
                choice_of_user = choice(a[1])
        else:
            return math.e, a[1]
    
    elif answ == DETECT_OF_INACCURACY_2 and a[1] == "ru":
        print("\nВаш запрос ---> " + a[2] +  " <--- неточен, пожалуйста введить номер дома")
        choice_of_user = choice(a[1])
        while choice_of_user == 1:
            a = _help1(a[1], a[0][1], a[0][3])
            if a == 10:
                return 11
            if True not in a[0]:
                answ = ana_inner(a[0], a[1])
                break
            else:
                print("\nВаш запрос ---> " + a[2] +  " <--- неточен, пожалуйста введить номер дома")
                choice_of_user = choice(a[1])
        else:
            return math.e, a[1]
        
    elif answ == DETECT_OF_INACCURACY_2 and a[1] == "ua":
        print("\nВаш запит ---> " + a[2] + " <--- не дуже точний, будь ласка введіть номер будинку")
        choice_of_user = choice(a[1])
        while choice_of_user == 1:
            a = _help1(a[1], a[0][1], a[0][3])
            if a == 10:
                return 11
            if True not in a[0]:
                answ = ana_inner(a[0], a[1])
                break
            else:
                print("\nВаш запит ---> " + a[2] + " <--- не дуже точний, будь ласка введіть номер будинку")
                choice_of_user = choice(a[1])
        else:
            return math.e, a[1]
            
            
    if answ == -1 and a[1] == "eng":
        print("\nNo such adress: " + "\n")
        return 0, a[2]
    elif answ == -1  and a[1] == "ru":
        print("\nТакой адрес отсутствует: " + "\n")
        return 0, a[2]
    elif answ == -1 and a[1] == "ua":
        print("\nТака адресса відсутня: " + "\n")
        return 0, a[2]
    elif answ == -1 and a[1] == 0:
        return 0, ""
    else:
        return answ[0], answ[1], a[2], a[1]

@count_t  # дек  
def main(): # мейн запускает все три части, где часть три это часть один (векторы), а часть 1 это алгоритм КМП ну и часть 2 это алгоритм нахождения метрики
    z = part_1()
    if z == 8:
        print("\nWrong file name\n")
        return
    if z == 11:
        print("\nError\n")
        return
    if z[0] == math.e and z[1] == "eng":
        print("\nThank you")
        return
    elif z[0] == math.e and z[1] == "ru":
        print("\nСпасибо")
        return
    elif z[0] == math.e and z[1] == "ua":
        print("\nДякую")
        return
    try:
        if z[3] == "eng":
            print("\n" + z[1])
            print("\nThank you")
        elif z[3] == "ru":
            print("\n" + z[1])
            print("\nСпасибо")
        elif z[3] == "ua":
            print("\n" + z[1])
            print("\nДякую")
    except IndexError:
        print(z[1])

if __name__ == "__main__":
       main()
