#!/usr/bin/python
import os
import json
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
import operator
from datetime import datetime
from nltk.corpus import reuters

class Glove:
    def __init__(self, Dlength, lenWord, conSZ):
        self.Dlength = Dlength
        self.lenWord = lenWord
        self.conSZ = conSZ

    def fitGlove(self, paragraphs, covarMat=None, lrngRt=10e-5, rglrRate=0.1, maxX=100, alpha=0.75, numEpochs=10, gradientDes=False, useTheano=True):
        currentTime = datetime.now()
        lenWord = self.lenWord
        Dlength = self.Dlength

        if os.path.exists(covarMat):
            covarMatrix = np.load(covarMat)
        else:
            covarMatrix = np.zeros((lenWord, lenWord))
            numOfSent = len(paragraphs)
            print ("Number of sentences to process:", numOfSent)
            processedCount = 0
            for eachSentence in paragraphs:
                processedCount += 1
                if processedCount % 10000 == 0:
                    print ("Processed", processedCount, "/", numOfSent)
                numWords = len(eachSentence)
                for eachWord in range(numWords):
                    wordIndex = eachSentence[eachWord]

                    start = max(0, eachWord - self.conSZ)
                    end = min(numWords, eachWord + self.conSZ)

                    if eachWord - self.conSZ < 0:
                        sentScore = 1.0 / (eachWord + 1)
                        covarMatrix[wordIndex,0] += sentScore
                        covarMatrix[0,wordIndex] += sentScore
                    if eachWord + self.conSZ > numWords:
                        sentScore = 1.0 / (numWords - eachWord)
                        covarMatrix[wordIndex,1] += sentScore
                        covarMatrix[1,wordIndex] += sentScore

                    for wordInd in range(start, eachWord):
                        if wordInd == eachWord: continue
                        tempIndex = eachSentence[wordInd]
                        sentScore = 1.0 / abs(eachWord - wordInd) # this is +ve
                        covarMatrix[wordIndex,tempIndex] += sentScore
                        covarMatrix[tempIndex,wordIndex] += sentScore
            np.save(covarMat, covarMatrix)

        normalizedCovar = np.zeros((lenWord, lenWord))
        normalizedCovar[covarMatrix < maxX] = (covarMatrix[covarMatrix < maxX] / float(maxX)) ** alpha
        normalizedCovar[covarMatrix >= maxX] = 1
        covarLog = np.log(covarMatrix + 1)

        print( "time to build co-occurrence costMat:", (datetime.now() - currentTime))
        # initialize weights
        initialWeights = np.random.randn(lenWord, Dlength) / np.sqrt(lenWord + Dlength)
        intialBias = np.zeros(lenWord)
        normalConst = np.random.randn(lenWord, Dlength) / np.sqrt(lenWord + Dlength)
        lenConst = np.zeros(lenWord)
        meanCovar = covarLog.mean()

        if gradientDes and useTheano:
            theanoWeights = theano.shared(initialWeights)
            theanoBias = theano.shared(intialBias)
            theanoConst = theano.shared(normalConst)
            theanoLen = theano.shared(lenConst)
            theanoLog = T.matrix('covarLog')
            theanoNormal = T.matrix('normalizedCovar')

            params = [theanoWeights, theanoBias, theanoConst, theanoLen]

            theanoDel = theanoWeights.dot(theanoConst.T) + T.reshape(theanoBias, (lenWord, 1)) + T.reshape(theanoLen, (1, lenWord)) + meanCovar - theanoLog
            theanoCost = ( theanoNormal * theanoDel * theanoDel ).sum()

            gradients = T.grad(theanoCost, params)

            costUp = [(p, p - lrngRt * g) for p, g in zip(params, gradients)]

            trainFun = theano.function(
                inputs=[theanoNormal, theanoLog],
                updates=costUp,
            )

        allCosts = []
        sentence_indexes = range(len(paragraphs))
        for numEpoch in range(numEpochs):
            epochDel = initialWeights.dot(normalConst.T) + intialBias.reshape(lenWord, 1) + lenConst.reshape(1, lenWord) + meanCovar - covarLog
            epochCost = ( normalizedCovar * epochDel * epochDel ).sum()
            allCosts.append(epochCost)
            print( "Epoch:", numEpoch, "epochCost:", epochCost)

            if gradientDes:
                if useTheano:
                    trainFun(normalizedCovar, covarLog)
                    initialWeights = theanoWeights.get_value()
                    intialBias = theanoBias.get_value()
                    normalConst = theanoConst.get_value()
                    lenConst = theanoLen.get_value()

                else:
                    # Paramerters Updation
                    prevWeights = initialWeights.copy()
                    for eachWord in range(lenWord):
                        initialWeights[eachWord] -= lrngRt * (normalizedCovar[eachWord, :] * epochDel[eachWord, :]).dot(normalConst)
                    initialWeights -= lrngRt * rglrRate * initialWeights

                    for eachWord in range(lenWord):
                        intialBias[eachWord] -= lrngRt * normalizedCovar[eachWord, :].dot(epochDel[eachWord, :])
                    intialBias -= lrngRt * rglrRate * intialBias

                    for wordInd in range(lenWord):
                        normalConst[wordInd] -= lrngRt * (normalizedCovar[:, wordInd] * epochDel[:, wordInd]).dot(prevWeights)
                    normalConst -= lrngRt * rglrRate * normalConst

                    for wordInd in range(lenWord):
                        lenConst[wordInd] -= lrngRt * normalizedCovar[:, wordInd].dot(epochDel[:, wordInd])
                    lenConst -= lrngRt * rglrRate * lenConst

            else:
                for eachWord in range(lenWord):
                    costMat = rglrRate * np.eye(Dlength) + (normalizedCovar[eachWord, :] * normalConst.T).dot(normalConst)
                    costVect = (normalizedCovar[eachWord,:]*(covarLog[eachWord,:] - intialBias[eachWord] - lenConst - meanCovar)).dot(normalConst)
                    initialWeights[eachWord] = np.linalg.solve(costMat, costVect)

                for eachWord in range(lenWord):
                    biasDen = normalizedCovar[eachWord,:].sum()
                    biasNum = normalizedCovar[eachWord,:].dot(covarLog[eachWord,:] - initialWeights[eachWord].dot(normalConst.T) - lenConst - meanCovar)
                    intialBias[eachWord] = biasNum / biasDen / (1 + rglrRate)

                for wordInd in range(lenWord):
                    costMat = rglrRate * np.eye(Dlength) + (normalizedCovar[:, wordInd] * initialWeights.T).dot(initialWeights)
                    costVect = (normalizedCovar[:,wordInd]*(covarLog[:,wordInd] - intialBias - lenConst[wordInd] - meanCovar)).dot(initialWeights)
                    normalConst[wordInd] = np.linalg.solve(costMat, costVect)

                for wordInd in range(lenWord):
                    lenConDen = normalizedCovar[:,wordInd].sum()
                    lenConNum = normalizedCovar[:,wordInd].dot(covarLog[:,wordInd] - initialWeights.dot(normalConst[wordInd]) - intialBias  - meanCovar)
                    lenConst[wordInd] = lenConNum / lenConDen / (1 + rglrRate)

        self.weights = initialWeights
        self.normalConst = normalConst
        plt.plot(allCosts)
        plt.show()

    def dumpWeights(self, weightFileLoc):
        savedWeights = [self.weights, self.normalConst.T]
        np.savez(weightFileLoc, *savedWeights)

def funLower(word):
    return word.lower()

def parseNews(countNews):
    finalSent = []
    wordDict = {'START': 0, 'END': 1}
    wordList = ['START', 'END']
    wordIndex = 2
    wordCountDict = {0: float('inf'), 1: float('inf')}
    iterCount = 0
    for news in reuters.fileids():
        i = reuters.words(news)
        words = [funLower(t) for t in i]
        for token in words:
            if token not in wordDict:
                wordDict[token] = wordIndex
                wordList.append(token)
                wordIndex += 1
            index = wordDict[token]
            wordCountDict[index] = wordCountDict.get(index, 0) + 1
        indexSentence = [wordDict[x] for x in words]
        finalSent.append(indexSentence)
        iterCount += 1
        print(iterCount)

    sortedWordIndex = sorted(wordCountDict.items(), key=operator.itemgetter(1), reverse=True)
    wordIndexTrunc = {}
    newIndex = 0
    updatedIndexDict = {}
    for index, count in sortedWordIndex[:countNews]:
        currWrd = wordList[index]
        print(currWrd, count)
        wordIndexTrunc[currWrd] = newIndex
        updatedIndexDict[index] = newIndex
        newIndex += 1
    wordIndexTrunc['UNKNOWN'] = newIndex
    unknown = newIndex

    newsTrunc = []
    for i in finalSent:
        if len(i) > 1:
            updatedSent = [updatedIndexDict[index] if index in updatedIndexDict else unknown for index in i]
            newsTrunc.append(updatedSent)
    return newsTrunc, wordIndexTrunc

def main(modelPath, wordIndexPath, sentencesPath):
    covarMatrix = "./input/covarMatrix.npy"
    if not os.path.isfile(wordIndexPath):
        news, wordIndex = parseNews(countNews=2000)
        with open(wordIndexPath, 'w') as f:
            json.dump(wordIndex, f)
        with open(sentencesPath, 'w') as f:
            json.dump(news, f)
    else:
        with open(wordIndexPath) as f:
            wordIndex = json.load(f)
        with open(sentencesPath) as f:
            news = json.load(f)

    newsLen = len(wordIndex)
    model = Glove(50, newsLen, 10)
    model.fitGlove(
        news,
        covarMat=covarMatrix,
        lrngRt=3 * 10e-5,
        rglrRate=0.01,
        numEpochs=20,
        gradientDes=True,
        useTheano=True
    )
    model.dumpWeights(modelPath)

if __name__ == '__main__':
    modelPath = './input/glove_model_50.npz'
    wordIndexPath = './input/word2idx.json'
    newsPath = './input/sentences.json'
    main(modelPath, wordIndexPath, newsPath)
