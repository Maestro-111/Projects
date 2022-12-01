import numpy as np
import time # I imported time in orderd to see how much time program requires
import openpyxl # module for reading the excel files
import warnings
warnings.simplefilter("ignore") # this is required in order to skip some of the python warnings, related to the read of xlxs files


def cal_time(some_fun): # here small decorator is defined. It calculates the time of program termination
    def inner(similarity,file,u):
        point1 = time.time()
        some_fun(similarity,file,u)
        point2 = time.time()
        print("Minutes: " + str((point2-point1)/60))
    return inner

def cal_sim(n): # here we just set an initial similarity matrix
    similarity = [0] * n
    for z in range(n):
        similarity[z] = [0] * n
    return similarity

@cal_time
def algo(similarity,file,u): # u is max_row, so we could have done that like sheet.max_row but here we know that we have 4883 vectors
    book = openpyxl.open(file, read_only = True)
    sheet = book.active
    look = []
    for linel in range(u): # fisrt of all, we read our file 
        candidate = sheet[linel+1]
        wow = (n.value for n in candidate if n.value != None) # however, instead of lists, we will append generators, since they require less memory
        look.append(wow)
    z = list(map(list, look)) # now we converte generators into the lists. Map in python works very fast
    for vector1 in z: # now, we can actually calculate the inner products
        for vector2 in z:
            x = z.index(vector1)
            y = z.index(vector2)
            similarity[x][y] = np.dot(vector1,vector2)


    outcome = [] # we will append the strings into this list
    for ress in similarity:
        candd = max(ress) # define a max in a row
        for indd in ress:
            if indd == candd: # catch the max and append vectors in a list
                xx = similarity.index(ress)+1
                yy = ress.index(indd)+1
                outcome.append("Article " + str(xx) + " is related to the Article " + str(yy) + ". ")

    document = open("myfile.txt", "w") # finally, we will write everything in the text file
    for lune in outcome:
        document.write(lune+'\n')
    document.close()



def main(): # main function that calls two functions
    a = cal_sim(4883)
    b = algo(a,"vot.xlsx",4883)
main() # calling the main

























