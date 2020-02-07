import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np
import os

def plotFTFN(P, R, folderName, subFolderArray):
    
    print("geldi")

    for i in range(0,len(P)):
        subFolderArray[i] = subFolderArray[i].split('/')[0]
        plt.plot(P[i], R[i], label = subFolderArray[i], marker='o', markerfacecolor='blue', markersize=3, linewidth=2)
        axes = plt.gca()
        axes.set_xlim([0,1])
        axes.set_ylim([0,1])
        axes.set_xlabel('Recall')
        axes.set_ylabel('Precision')
        axes.set_title(folderName)

    plt.legend()
    plt.show() 
    
    #return True


paths = glob.glob('*/')
print(paths)


for i in paths:
    os.chdir(i)
    k = glob.glob('*/')
    folder = i.split('/')[0]
    subfolder = k
    print(k)
    totalP = []
    totalR = []
    
    for j in k:

        os.chdir(j)
        print(j)
        csvFiles = glob.glob('*/')
        extension = 'csv'
        result = glob.glob('*.{}'.format(extension))
        result.sort()
        for l in result:
            print(l)

        arrayP =[]
        arrayR = []

        for n in result:
            #print(i)
            #plotFTFN(pd.read_csv(i))
            csvDF = pd.read_csv(n)

            arrayP.append(csvDF['P'][0])
            arrayR.append(csvDF['R'][0])

        #plotFTFN(arrayP,arrayR)
        
        totalP.append(arrayP)
        totalR.append(arrayR) 
        
        os.chdir("..")

    plotFTFN(totalP, totalR, folder, subfolder)

    os.chdir("..")
