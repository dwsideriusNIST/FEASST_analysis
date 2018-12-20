import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import re


data=[]


# Function checks if the string
# contains special character #
def run(string):

    # Make own character set and pass
    # this as argument in compile method
    regex = re.compile('#')

    # Pass the string in search
    # method of regex object.
    if not (re.match(regex, string)):
        data.append(line.split(' '))



file_object = open("colMat","r") #read collection matrix
#file_object = open("colMatrw.txtrw","r") #read collection matrix
lines = file_object.read().split("\n") #split collection matrix by line
lines = np.asarray(lines) #convert lines list into array




for line in lines: #iterate over lines until find line without '#'
        run(line)


data = np.concatenate((data), axis=0)
data =np.delete(data, -1)

data = np.asarray(data)  #Converts data list into array.
data = np.reshape(data, (301,13)) #must provide matrix dimensions
data = data.astype(np.float)



print(data[:,0])


plt.figure(1)
plt.plot(data[:,0], data[:,1])
plt.ylabel('ln(pi)')
#plt.ylabel('PE')
plt.xlabel('N')
plt.show()



file_object.close()
