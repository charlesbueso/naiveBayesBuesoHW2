from cgi import test
from cmath import log
from logging import logProcesses
from operator import neg
import string, re
import math

allData = {}
posData = {}
negData = {}
trainingData = {}
developmentData = {}
testData = {}
negTokens = ['isn\'t','doesn\'t', 'can\'t', 'not', 'no', 'never', 'weren\'t', 'wasn\'t', 'ain\'t', 'don\'t']

#function for negative tokenization
def negTokenization(line):
    allWords = re.findall(r"[\w']+|[.,!?;]", line)
    index = 0
    new = ''
    while allWords:
        if allWords[index] in negTokens:
            new = new + ' ' + allWords[index]
            del allWords[index]
            if allWords == '': break
            try:
                while allWords[index] not in string.punctuation:
                        new = new + ' ' + 'NOT' + allWords[index] 
                        del allWords[index]
            except: break
        else:
            new = new + ' ' + allWords[index]
            del allWords[index]
    return new

##Preparing data
#Dictionary with positive and negative reviews as keys, sentiment as value.
with open("posData.txt") as file:
    for line in file:
        ''.join(x for x in line if x.isalpha() or x == "'")
        line = negTokenization(line)
        allData[line] = 'pos'
        posData[line] = True
with open("negData.txt") as file:
    for line in file:
        ''.join(x for x in line if x.isalpha() or x == "'")
        line = negTokenization(line)
        allData[line] = 'neg'
        negData[line] = True
with open("stopWords.txt", "r") as f:
    stopWords = {k:v for k, *v in map(str.split, f)}  

#Split data
keyPosData = list(posData)
keyNegData = list(negData)
counter = 0
while counter < 5331:
    #70% training
    if 0 <= counter < 3732:
        trainingData[keyPosData[counter]] = 'pos'
        trainingData[keyNegData[counter]] = 'neg'
        counter += 1
    #15% development
    elif 3732 <= counter < 4531:
        developmentData[keyPosData[counter]] = 'pos'
        developmentData[keyNegData[counter]] = 'neg'
        counter += 1
    #15% test
    elif 4531 <= counter < 5331:
        testData[keyPosData[counter]] = 1
        testData[keyNegData[counter]] = 1
        counter += 1

#print(len(trainingData), len(testData), len(developmentData))

def trainNaiveBayes(D):
    nDoc = len(D)
    nC = nDoc/2
    #logpriors
    logpriorPcPos = math.log(nC/nDoc)
    logpriorPcNeg = math.log(nC/nDoc)
    vocabularyD = {}
    bigDocPos = {}
    bigDocNeg = {}
    #making bag of words with frequencies, also pos and neg dicts
    for review in D:
        if D[review] == 'pos':
            #optimize by deleting duplicates on each review
            delDuplicates = review.split()
            delDuplicates = " ".join(sorted(set(delDuplicates), key=delDuplicates.index))
            for word in delDuplicates.split():
                if word not in stopWords and word.isalpha():
                    if word in vocabularyD:
                        vocabularyD[word] = vocabularyD[word] + 1
                    else: 
                        vocabularyD[word] = 1
                    
                if word not in stopWords and word.isalpha():
                    if word in bigDocPos:
                        bigDocPos[word] = bigDocPos[word] + 1
                    else: bigDocPos[word] = 1
        else:
            delDuplicates = review.split()
            delDuplicates = " ".join(sorted(set(delDuplicates), key=delDuplicates.index))
            for word in delDuplicates.split():
                if word not in stopWords and word.isalpha():
                    if word in vocabularyD:
                        vocabularyD[word] = vocabularyD[word] + 1
                    else: vocabularyD[word] = 1

                if word not in stopWords and word.isalpha():
                    if word in bigDocNeg:
                        bigDocNeg[word] = bigDocNeg[word] + 1
                    else: bigDocNeg[word] = 1    

    #loglikelihoods
    loglikelihoodPos = {}
    loglikelihoodNeg = {}
    for word in vocabularyD:
        try: countWcPos = bigDocPos[word]
        except: countWcPos = 0
        sumPos = 0
        for i in bigDocPos: sumPos + bigDocPos[i]
        #using add 1 smoothing
        loglikelihoodPos[word] = round(math.log((countWcPos+1)/(sumPos + len(vocabularyD))), 5)
        #loglikelihoodPos[word] = math.log((countWcPos+1)/(sumPos + len(vocabularyD)))

        try: countWcNeg = bigDocNeg[word]
        except: countWcNeg = 0
        sumNeg = 0
        for i in bigDocNeg: sumNeg + bigDocNeg[i]
        #using add 1 smoothing
        loglikelihoodNeg[word] = round(math.log((countWcNeg+1)/(sumNeg + len(vocabularyD))), 5)
        #loglikelihoodNeg[word] = math.log((countWcNeg+1)/(sumNeg + len(vocabularyD)))

    return logpriorPcPos, logpriorPcNeg, loglikelihoodPos, loglikelihoodNeg, vocabularyD

def testNaiveBayes(testdoc, logpriorPos, logpriorNeg, loglikelihoodPos, loglikelihoodNeg, V):

    posCounter = 0
    negCounter = 0
    classifiedReviews = {}

    for review in testdoc:
        sumCPos = logpriorPos
        sumCNeg = logpriorNeg
        for word in review.split():
            if word.isalpha() and word in V:
                sumCPos = sumCPos * (loglikelihoodPos[word])
                sumCNeg = (sumCNeg * (loglikelihoodNeg[word]))

        if sumCPos > sumCNeg: 
            classifiedReviews[review] = 'pos'
            posCounter += 1
        elif sumCNeg > sumCPos: 
            classifiedReviews[review] = 'neg'
            negCounter += 1
        #elif sumCNeg == sumCPos: print(sumCNeg)
    
    return classifiedReviews

logpriorPos, logpriorNeg, loglikelihoodPos, loglikelihoodNeg, V = trainNaiveBayes(trainingData)
classifiedReviews = testNaiveBayes(developmentData, logpriorPos, logpriorNeg, loglikelihoodPos, loglikelihoodNeg, V)


good = 0
wrong = 0
for i in classifiedReviews:
    if allData[i] == classifiedReviews[i]:
        good += 1
    else: wrong += 1

print(good, wrong, good+wrong)