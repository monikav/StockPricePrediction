#!/usr/bin/python
import json
import os
import nltk
import numpy as np


def lowercaseWord(word):
    return word.lower()

def loadModel(wordVecFile):
    wordVec = np.zeros([0,100])
    with open(wordVecFile) as file:
        for i in file:
            i = i.strip().split()
            i = list(map(float,i))
            wordVec = np.vstack((wordVec,np.array(i).flatten()))
    return wordVec

def featureNormalise(senShape, desiredLen):
    rows = senShape.shape[0]
    columns = senShape.shape[1]
    if columns < desiredLen:
        return np.hstack((np.zeros([rows, desiredLen - columns]), senShape)).flatten()
    else:
        return senShape[:, -desiredLen:].flatten()

def genFeatMat(gloveEmb, wordIndex, stocksPrice, numWords=60, model="test", index=0):
    directory = './input/'
    newsInput = [f for f in os.listdir(directory) if f.startswith('news_reuters.csv')]
    count = 0
    shape = gloveEmb.shape[1]
    finalFeat = np.zeros([0, numWords * shape])
    depVar = []
    for file in newsInput:
        count = 0
        with open(directory+file) as f:
            if model == 'test' and not index:
                f.seek(125000,0)           # seek to end of file; f.seek(0, 2) is legal
            if model == 'validation' and not index:
                f.seek(100000,0)
            if model == 'train' and index :
                f.seek(50000,0)
            if model == 'test' and index :
                f.seek(137500,0)
            if model == 'validation' and index :
                f.seek(112500,0)
            for line in f:
                if model == 'test' and count == 12500: break
                if model == 'train' and count== 50000: break
                if model == 'validation' and count == 12500:break
                line = line.strip().split(',')
                if len(line) != 6: continue
                company, companyName, CurrDate, newsHL, newsBody ,newsImpact= line
                if company not in stocksPrice: continue
                if CurrDate not in stocksPrice[company]: continue
                count += 1
                print(count)
                resultTkn = nltk.word_tokenize(newsHL) + nltk.word_tokenize(newsBody)
                resultTkn = [lowercaseWord(t) for t in resultTkn]
                sentNp = np.zeros([shape, 0])
                for t in resultTkn:
                    if t not in wordIndex: continue
                    sentNp = np.hstack((sentNp, np.matrix(gloveEmb[wordIndex[t]]).T))
                finalFeat = np.vstack((finalFeat, featureNormalise(sentNp, numWords)))
                count+=1
                depVar.append(round(stocksPrice[company][CurrDate], 6))
    finalFeat = np.array(finalFeat)
    depVar = np.matrix(depVar)
    featMat = np.concatenate((finalFeat, depVar.T), axis=1)
    featFile = './input/featureMatrix_' + str(index) + "_" + model + '.csv'
    np.savetxt(featFile, featMat, fmt="%s")

def splitData(gloveEmb, wordIndexFile, numWords=60):
    with open('./input/stockPrices.json') as pricesFile:
        reqPrice = json.load(pricesFile)
    with open(wordIndexFile) as indexFile:
        wordIndex = json.load(indexFile)

    genFeatMat(gloveEmb, wordIndex, reqPrice, numWords, "validation", 0)
    genFeatMat(gloveEmb, wordIndex, reqPrice, numWords, "test", 0)
    genFeatMat(gloveEmb, wordIndex, reqPrice, numWords, "validation", 1)
    genFeatMat(gloveEmb, wordIndex, reqPrice, numWords, "test", 1)

def main(modelPath, wordIndex):
    gloveEmbModel = loadModel(modelPath)
    splitData(gloveEmbModel, wordIndex, 30)


if __name__ == "__main__":
    modelPath = './input/wordEmbeddingsVocab.csv'
    indexPath = "./input/word2idx.json"
    main(modelPath, indexPath)
