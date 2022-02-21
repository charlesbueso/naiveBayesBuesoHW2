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

#merge two dictionaries
def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z

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

                    if word in bigDocNeg:
                        bigDocNeg[word] = bigDocNeg[word] + 1
                    else: bigDocNeg[word] = 1    

    #Removing infrequent words
    oldKeysPos = list(bigDocPos.keys())
    oldKeysNeg = list(bigDocNeg.keys())
    removeBy = 70
    for i in oldKeysPos:
        try:
            if bigDocPos[i] < removeBy:
                del bigDocPos[i]
        except: pass

    for i in oldKeysNeg:
        try:
            if bigDocNeg[i] < removeBy:
                del bigDocNeg[i]
        except: pass

    #adding features from positive and negative words datasets
    weigth = 125
    with open('posWords.txt') as posWords:
            for line in posWords:
                line.strip('\n')
                if line not in bigDocPos:
                    bigDocPos[line] = weigth
                if line not in vocabularyD:
                    vocabularyD[line] = weigth
    with open('negWords.txt') as negWords:
            for line in negWords:
                line = line.strip('\n')
                if line not in bigDocNeg:
                    bigDocNeg[line] = weigth
                if line not in vocabularyD:
                    vocabularyD[line] = weigth

    #log likelihoods
    loglikelihoodPos = {}
    loglikelihoodNeg = {}
    sumV = 0

    for word in vocabularyD:
        countWcPos = 0
        try: countWcPos = bigDocPos[word]
        except: countWcPos = 0
        sumPos = 0
        for i in bigDocPos: sumPos = sumPos + bigDocPos[i]
        #using add 1 smoothing
        loglikelihoodPos[word] = round(math.log((countWcPos+1)/(sumPos + len(vocabularyD))), 7)
        
        countWcNeg = 0
        try: countWcNeg = bigDocNeg[word]
        except: countWcNeg = 0
        sumNeg = 0
        for i in bigDocNeg: sumNeg = sumNeg + bigDocNeg[i]
        #using add 1 smoothing
        loglikelihoodNeg[word] = round(math.log((countWcNeg+1)/(sumNeg + len(vocabularyD))), 7)

    return logpriorPcPos, logpriorPcNeg, loglikelihoodPos, loglikelihoodNeg, vocabularyD

def testNaiveBayes(testdoc, logpriorPos, logpriorNeg, loglikelihoodPos, loglikelihoodNeg, V):
    posCounter = 0
    negCounter = 0
    classifiedReviews = {}

    for review in testdoc:
        sumCPos = logpriorPos
        sumCNeg = logpriorNeg
        for word in review.split():
            if word in V:
                sumCPos = sumCPos * (loglikelihoodPos[word])
                sumCNeg = sumCNeg * (loglikelihoodNeg[word])

        if sumCPos > sumCNeg: classifiedReviews[review] = 'pos'
        elif sumCNeg > sumCPos: classifiedReviews[review] = 'neg'

    return classifiedReviews


###Calling all functions, printing results to console.
logpriorPos, logpriorNeg, loglikelihoodPos, loglikelihoodNeg, V = trainNaiveBayes(merge_two_dicts(trainingData, developmentData))
classifiedReviews = testNaiveBayes(testData, logpriorPos, logpriorNeg, loglikelihoodPos, loglikelihoodNeg, V)

#results
goodPos = 0
wrongPos = 0
goodNeg = 0
wrongNeg = 0
for i in classifiedReviews:
    if allData[i] == classifiedReviews[i] == 'pos':
        goodPos += 1
    elif allData[i] == classifiedReviews[i] == 'neg':
        goodNeg += 1
    elif allData[i] == 'pos' and classifiedReviews[i] == 'neg':
        wrongPos += 1
    elif allData[i] == 'neg' and classifiedReviews[i] == 'pos':
        wrongNeg += 1  

accuracy = (goodPos+goodNeg)*100 / 1600
print("{cp} positive movie reviews were classified correctly, and {cn} negative reviews were classified correctly.".format(cp = goodPos, cn = goodNeg))
print("{ip} positive reviews were classified incorrectly, and {ineg} negative reviews were classified incorrectly.".format(ip = wrongPos, ineg = wrongNeg)) 
print("{nc} were not classified (kept as neutral). The system has {acc}% accuracy, or {c} over 1600.".format(nc = len(testData) - (goodPos+goodNeg+wrongPos+wrongNeg), acc = round(accuracy,3), c = (goodPos+goodNeg)))