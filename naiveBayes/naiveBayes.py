import string, re

allData = {}
posData = {}
negData = {}
trainingData = {}
developmentData = {}
testData = {}

##Preparing data
#Dictionary with positive and negative reviews as keys, sentiment as value.
with open("posData.txt") as file:
    for line in file:
        ''.join(x for x in line if x.isalpha() or x == "'")
        allData[line] = 'pos'
        posData[line] = True
with open("negData.txt") as file:
    for line in file:
        ''.join(x for x in line if x.isalpha() or x == "'")
        allData[line] = 'neg'
        negData[line] = True

#Split data
keyPosData = list(posData)
keyNegData = list(negData)
counter = 0
while counter < 5331:
    #70% training
    if 0 <= counter < 3732:
        trainingData[keyPosData[counter]] = 1
        trainingData[keyNegData[counter]] = 1
        counter += 1
    #15% development
    elif 3732 <= counter < 4531:
        developmentData[keyPosData[counter]] = 1
        developmentData[keyNegData[counter]] = 1
        counter += 1
    #15% test
    elif 4531 <= counter < 5331:
        testData[keyPosData[counter]] = 1
        testData[keyNegData[counter]] = 1
        counter += 1

print(len(trainingData), len(testData), len(developmentData))

def trainNaiveBayes(D, C):
    return

def testNaiveBayes(testdoc, logprior, loglikelihood, C, V):
    return
