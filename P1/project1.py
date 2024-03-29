import numpy as np
import matplotlib.pyplot as plt

# process file
def getCorpus(filename):
    corpus = []
    #remove = ",.!?;:#$%&()*+-/<=>@[]^_`{}~<>|\\"
    #l = list(remove)
    for line in open(filename, 'r'):
        #define the start word <s>
        newLine = ['<s>']
        for word in line.split():
            #if word not in l:
            newLine.append(word.lower())
        # stop word
        newLine.append('</s>')
        corpus.append(newLine)
    return corpus

# get bigram corpus
def getBigramCorpus(corpus):
    bigramCorpus = []
    for line in corpus:
        newLine = []
        for i in range(len(line) - 1):
            newLine.append((line[i].lower(), line[i + 1].lower()))
        bigramCorpus.append(newLine)
    return bigramCorpus

# treat the word that appears once as <unk>
def getCorpusWithUnk(corpus, unigramCount):
    temp = []
    output = []
    for word in unigramCount:
        if unigramCount[word] == 1:
            temp.append(word)
    for i in range(len(corpus)):
        output.append([])
        for j in range(len(corpus[i])):
            if corpus[i][j] in temp:
                output[i].append("<unk>")
            else:
                output[i].append(corpus[i][j])
    return output

# get unigram probabilities and count
# return as tuple
def getUnigramProbs(corpus):
    unigramProbs = {}
    unigramCount = {}
    totalCount = 0
    for line in corpus:
        for word in line:
            if word.lower() not in unigramProbs:
                unigramProbs[word.lower()] = 0
                unigramCount[word.lower()] = 0
            unigramProbs[word.lower()] += 1
            unigramCount[word.lower()] += 1
            totalCount += 1
    for word in unigramProbs:
        unigramProbs[word] /= 1.0 * totalCount
    return unigramProbs, unigramCount

# get bigram probabilities
def getBigramProbs(corpus, unigramCount):
    bigramProbs = {}
    for line in corpus:
        for i in range(len(line) - 1):
            # bigram tuple
            bigramTuple = (line[i].lower(), line[i + 1].lower())
            if bigramTuple not in bigramProbs:
                bigramProbs[bigramTuple] = 0
            bigramProbs[bigramTuple] += 1
    for word in bigramProbs:
        bigramProbs[word] = 1.0 * bigramProbs[word] / unigramCount[word[0]]
    return bigramProbs

# return unigram probabilities with add-k smoothing 
def unigramAddK(corpus, k):
    unigramProbs = {}
    totalCount = 0
    for line in corpus:
        for word in line:
            if word.lower() not in unigramProbs:
                unigramProbs[word.lower()] = 0
            unigramProbs[word.lower()] += 1
            totalCount += 1
    for word in unigramProbs:
        unigramProbs[word] = 1.0 * (unigramProbs[word] + k) / (totalCount + k * len(unigramProbs))
    return unigramProbs

# return bigram probabilities with add-k smoothing
def bigramAddK(corpus, k, unigramCount):
    bigramProbs = {}
    for line in corpus:
        for i in range(len(line) - 1):
            bigramTuple = (line[i].lower(), line[i + 1].lower())
            if bigramTuple not in bigramProbs:
                bigramProbs[bigramTuple] = 0
            bigramProbs[bigramTuple] += 1
    for word in bigramProbs:
        bigramProbs[word] = 1.0 * (bigramProbs[word] + k) / (unigramCount[word[0]] + k * len(unigramCount))
    return bigramProbs

# compute the perplexity for unigram
def unigramPerplexity(line, unigramProbs):
    result = []
    for word in line:
        if word in unigramProbs:
            result.append(-np.log(unigramProbs[word]))
        else:
            result.append(-np.log(unigramProbs['<unk>']))
    return np.exp(1.0 / len(result) * sum(result))

# compute the perplexity for bigram
# first if is for add-k: if a pair of word tuple has count 0, simply use the numerator as k (use this together with add-k)
# else is for bigram perplexity calculation without add-k smoothing
def bigramPerplexity(line, bigramProbs, unigramProbs, unigramCount = [], k = 0):
    result = [-np.log(unigramProbs['<s>'])]
    if len(unigramCount) > 0 and k > 0:
        for bigramTuple in line:
            if bigramTuple in bigramProbs:
                result.append(-np.log(bigramProbs[bigramTuple]))
            else:
                first = bigramTuple[0]
                if first not in unigramCount:
                    first = '<unk>'
                result.append(-np.log(k / (unigramCount[first] + k * len(unigramCount))))
    else:
        for bigramTuple in line:
            if bigramTuple in bigramProbs:
                result.append(-np.log(bigramProbs[bigramTuple]))
            elif (bigramTuple[0], '<unk>') in bigramProbs:
                result.append(-np.log(bigramProbs[(bigramTuple[0], '<unk>')]))
            elif ('<unk>', bigramTuple[1]) in bigramProbs:
                result.append(-np.log(bigramProbs[('<unk>', bigramTuple[1])]))
            else:
                result.append(-np.log(bigramProbs[('<unk>', '<unk>')]))
    return np.exp(1.0 / len(result) * sum(result))

# classify the reviews using unigram perplexity, 0 for truthful, 1 for deceptive
def unigramClassifier(corpus, truthfulProbs, deceptiveProbs):
    result = []
    for line in corpus:
        truthfulPerplexity = unigramPerplexity(line, truthfulProbs)
        deceptivePerplexity = unigramPerplexity(line, deceptiveProbs)
        #print('t',truthfulPerplexity)
        #print('d',deceptivePerplexity)
        if truthfulPerplexity <= deceptivePerplexity:
            result.append(0)
        else:
            result.append(1)
    return result

# classify the reviews using bigram perplexity, 0 for truthful, 1 for deceptive
# need to get the unigram probability with unk first
def bigramClassifier(corpus, truthfulProbs, deceptiveProbs, unigramTruthfulProbsUnk, unigramDeceptiveProbsUnk, \
    unigramTruthfulCount=[], unigramDeceptiveCount=[], k = 0):
    result = []
    for line in corpus:
        truthfulPerplexity = bigramPerplexity(line, truthfulProbs, unigramTruthfulProbsUnk, unigramTruthfulCount, k)
        deceptivePerplexity = bigramPerplexity(line, deceptiveProbs, unigramDeceptiveProbsUnk, unigramDeceptiveCount, k)
        #print('t',truthfulPerplexity)
        #print('d',deceptivePerplexity)
        if truthfulPerplexity <= deceptivePerplexity:
            result.append(0)
        else:
            result.append(1)
    return result

# compute the accuracy
def accuracy(pred, real):
    return 1.0 * sum([1 for i in range(len(pred)) if pred[i] == real[i]]) / len(pred)

if __name__ == "__main__":
    # process training data
    corpusTruthfulTrain = getCorpus('DATASET/train/truthful.txt')
    corpusDeceptiveTrain = getCorpus('DATASET/train/deceptive.txt')
    unigramTruthfulTrain = getUnigramProbs(corpusTruthfulTrain)
    unigramDeceptiveTrain = getUnigramProbs(corpusDeceptiveTrain)

    #bigramTruthfulTrain = getBigramProbs(corpusTruthfulTrain, unigramTruthfulTrain[1])
    #bigramDeceptiveTrain = getBigramProbs(corpusDeceptiveTrain, unigramDeceptiveTrain[1])

    corpusTruthfulTrainUnk = getCorpusWithUnk(corpusTruthfulTrain, unigramTruthfulTrain[1])
    corpusDeceptiveTrainUnk = getCorpusWithUnk(corpusDeceptiveTrain, unigramDeceptiveTrain[1])
    unigramTruthfulTrainUnk = getUnigramProbs(corpusTruthfulTrainUnk)
    unigramDeceptiveTrainUnk = getUnigramProbs(corpusDeceptiveTrainUnk)
    bigramTruthfulTrainUnk = getBigramProbs(corpusTruthfulTrainUnk, unigramTruthfulTrainUnk[1])
    bigramDeceptiveTrainUnk = getBigramProbs(corpusDeceptiveTrainUnk, unigramDeceptiveTrainUnk[1])

    #bigramCorpusTruthfulTrainUnk = getBigramCorpus(corpusTruthfulTrainUnk)
    #bigramCorpusDeceptiveTrainUnk = getBigramCorpus(corpusDeceptiveTrainUnk)

    # validation
    corpusTruthfulVal = getCorpus('DATASET/validation/truthful.txt')
    corpusDeceptiveVal = getCorpus('DATASET/validation/deceptive.txt')
    unigramTruthfulVal = getUnigramProbs(corpusTruthfulVal)
    unigramDeceptiveVal = getUnigramProbs(corpusDeceptiveVal)

    #corpusTruthfulValUnk = getCorpusWithUnk(corpusTruthfulVal, unigramTruthfulVal[1])
    #corpusDeceptiveValUnk = getCorpusWithUnk(corpusDeceptiveVal, unigramDeceptiveVal[1])
    #unigramTruthfulValUnk = getUnigramProbs(corpusTruthfulValUnk)
    #unigramDeceptiveValUnk = getUnigramProbs(corpusDeceptiveValUnk)
    #bigramTruthfulValUnk = getBigramProbs(corpusTruthfulValUnk, unigramTruthfulValUnk[1])
    #bigramDeceptiveValUnk = getBigramProbs(corpusDeceptiveValUnk, unigramDeceptiveValUnk[1])

    bigramCorpusTruthfulVal = getBigramCorpus(corpusTruthfulVal)
    bigramCorpusDeceptiveVal = getBigramCorpus(corpusDeceptiveVal)
    #bigramCorpusTruthfulValUnk = getBigramCorpus(corpusTruthfulValUnk)
    #bigramCorpusDeceptiveValUnk = getBigramCorpus(corpusDeceptiveValUnk)
    '''
    # compute the validation accuracy
    unigramTruthfulValPerp = unigramClassifier(corpusTruthfulVal, unigramTruthfulTrainUnk[0], unigramDeceptiveTrainUnk[0])
    #print(unigramTruthfulValPerp)
    print('accuracy for unigram validation/truthful:'),
    print(accuracy(unigramTruthfulValPerp, [0 for i in range(len(unigramTruthfulValPerp))]))

    unigramDeceptiveValPerp = unigramClassifier(corpusDeceptiveVal, unigramTruthfulTrainUnk[0], unigramDeceptiveTrainUnk[0])
    #print(unigramDeceptiveValPerp)
    print('accuracy for unigram validation/deceptive:'),
    print(accuracy(unigramDeceptiveValPerp, [1 for i in range(len(unigramDeceptiveValPerp))]))

    bigramTruthfulValPerp = bigramClassifier(bigramCorpusTruthfulVal, bigramTruthfulTrainUnk, bigramDeceptiveTrainUnk, unigramTruthfulTrainUnk[0], unigramDeceptiveTrainUnk[0])
    print('accuracy for bigram validation/truthful:'),
    print(accuracy(bigramTruthfulValPerp, [0 for l in range(len(bigramTruthfulValPerp))]))

    bigramDeceptiveValPerp = bigramClassifier(bigramCorpusDeceptiveVal, bigramTruthfulTrainUnk, bigramDeceptiveTrainUnk, unigramTruthfulTrainUnk[0], unigramDeceptiveTrainUnk[0])
    print('accuracy for bigram validation/deceptive:'),
    print(accuracy(bigramDeceptiveValPerp, [1 for l in range(len(bigramDeceptiveValPerp))]))
    '''

    '''
    # select best add-k: first obtain (0.01, 0.69140625); then obatin best->(0.06, 0.890625)
    k = 0
    best = -1
    bestK = 0
    while (k < 1.01):
        #print(k)
        unigramTruthfulTrainAddK = unigramAddK(corpusTruthfulTrainUnk, k)
        unigramDeceptiveTrainAddK = unigramAddK(corpusDeceptiveTrainUnk, k)
        bigramTruthfulTrainAddk = bigramAddK(corpusTruthfulTrainUnk, k, unigramTruthfulTrainUnk[1])
        bigramDeceptiveTrainAddk = bigramAddK(corpusDeceptiveTrainUnk, k, unigramDeceptiveTrainUnk[1])

        unigramTruthfulValPerp = unigramClassifier(corpusTruthfulVal, unigramTruthfulTrainAddK, unigramDeceptiveTrainAddK)
        #print(unigramTruthfulValPerp)
        print('accuracy for unigram validation/truthful for k = %.3f:' %k),
        print(accuracy(unigramTruthfulValPerp, [0 for i in range(len(unigramTruthfulValPerp))]))

        unigramDeceptiveValPerp = unigramClassifier(corpusDeceptiveVal, unigramTruthfulTrainAddK, unigramDeceptiveTrainAddK)
        #print(unigramDeceptiveValPerp)
        print('accuracy for unigram validation/deceptive for k = %.3f:' %k),
        print(accuracy(unigramDeceptiveValPerp, [1 for i in range(len(unigramDeceptiveValPerp))]))

        bigramTruthfulValPerp = bigramClassifier(bigramCorpusTruthfulVal, bigramTruthfulTrainAddk, bigramDeceptiveTrainAddk, \
            unigramTruthfulTrainAddK, unigramDeceptiveTrainAddK, unigramTruthfulTrainUnk[1], unigramDeceptiveTrainUnk[1], k)
        print('accuracy for bigram validation/truthful for k = %.3f:' %k),
        print(accuracy(bigramTruthfulValPerp, [0 for l in range(len(bigramTruthfulValPerp))]))

        bigramDeceptiveValPerp = bigramClassifier(bigramCorpusDeceptiveVal, bigramTruthfulTrainAddk, bigramDeceptiveTrainAddk, \
            unigramTruthfulTrainAddK, unigramDeceptiveTrainAddK, unigramTruthfulTrainUnk[1], unigramDeceptiveTrainUnk[1], k)
        print('accuracy for bigram validation/deceptive for k = %.3f:' %k),
        print(accuracy(bigramDeceptiveValPerp, [1 for l in range(len(bigramDeceptiveValPerp))]))

        print('\n\n')
        temp1 = accuracy(bigramTruthfulValPerp, [0 for l in range(len(bigramTruthfulValPerp))])
        temp2 = accuracy(bigramDeceptiveValPerp, [1 for l in range(len(bigramDeceptiveValPerp))])
        if best < (temp1+temp2)/2:
            best = (temp1+temp2)/2
            bestK = k
        
        k += 0.1
    print(bestK,best)'''

    
    k = 0.06
    unigramTruthfulTrainAddK = unigramAddK(corpusTruthfulTrainUnk, k)
    unigramDeceptiveTrainAddK = unigramAddK(corpusDeceptiveTrainUnk, k)
    bigramTruthfulTrainAddk = bigramAddK(corpusTruthfulTrainUnk, k, unigramTruthfulTrainUnk[1])
    bigramDeceptiveTrainAddk = bigramAddK(corpusDeceptiveTrainUnk, k, unigramDeceptiveTrainUnk[1])

    bigramTruthfulValPerp = bigramClassifier(bigramCorpusTruthfulVal, bigramTruthfulTrainAddk, bigramDeceptiveTrainAddk, \
    unigramTruthfulTrainAddK, unigramDeceptiveTrainAddK, unigramTruthfulTrainUnk[1], unigramDeceptiveTrainUnk[1], k)
    #print('accuracy for bigram validation/truthful for k = %.3f:' %k),
    #print(accuracy(bigramTruthfulValPerp, [0 for l in range(len(bigramTruthfulValPerp))]))
    truthfulOutCorrect = []
    truthfulOutIncorrect = []
    for i in range(len(bigramTruthfulValPerp)):
        if (bigramTruthfulValPerp[i] == 0):
            truthfulOutCorrect.append(i)
        else:
            truthfulOutIncorrect.append(i)
    print('truthful output correct:', len(truthfulOutCorrect))
    print('truthful output incorrect:', len(truthfulOutIncorrect))

    bigramDeceptiveValPerp = bigramClassifier(bigramCorpusDeceptiveVal, bigramTruthfulTrainAddk, bigramDeceptiveTrainAddk, \
    unigramTruthfulTrainAddK, unigramDeceptiveTrainAddK, unigramTruthfulTrainUnk[1], unigramDeceptiveTrainUnk[1], k)
    #print('accuracy for bigram validation/deceptive for k = %.3f:' %k),
    #print(accuracy(bigramDeceptiveValPerp, [1 for l in range(len(bigramDeceptiveValPerp))]))
    deceptiveOutCorrect = []
    deceptiveOutIncorrect = []
    for i in range(len(bigramDeceptiveValPerp)):
        if (bigramDeceptiveValPerp[i] == 1):
            deceptiveOutCorrect.append(i)
        else:
            deceptiveOutIncorrect.append(i)
    print('deceptive output correct:', len(deceptiveOutCorrect))
    print('deceptive output incorrect:', len(deceptiveOutIncorrect))

    nbTruthCorrect = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 62, 63, 64, 66, 67, 68, 69, 70, 71, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 91, 92, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 109, 110, 111, 112, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127]
    nbTruthIncorrect = [14, 16, 61, 65, 72, 83, 90, 93, 108, 113]

    nbDecepCorrect = [0, 1, 2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 69, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 82, 83, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 105, 106, 107, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 122, 123, 125, 126, 127]
    nbDecepIncorrect = [3, 5, 22, 55, 68, 71, 81, 89, 104, 110, 121, 124]

    print(len(nbTruthCorrect),len(nbTruthIncorrect),len(nbDecepCorrect),len(nbDecepIncorrect))

    bothCorrect = []
    bothIncorrect = []
    lmCorrect_nbIncorrect = []
    lmIncorrect_nbCorrect = []
    for i in range(len(corpusTruthfulVal)):
        if i in truthfulOutCorrect and i in nbTruthCorrect:
            bothCorrect.append(i)
        elif i in truthfulOutCorrect and i in nbTruthIncorrect:
            lmCorrect_nbIncorrect.append(i)
        elif i in truthfulOutIncorrect and i in nbTruthCorrect:
            lmIncorrect_nbCorrect.append(i)
        elif i in truthfulOutIncorrect and i in nbTruthIncorrect:
            bothIncorrect.append(i)
    
    #valCorpus = []
    #for line in open('DATASET/validation/truthful.txt', 'r'):
    #    valCorpus.append(line)

    #outputFile1 = open('truthful-val.txt', 'w')
    
    #outputFile1.write('lm correct but nb incorrect:')
    #for i in lmCorrect_nbIncorrect:
    #    outputFile1.write('%d, %s' %(i, valCorpus[i]))
    #outputFile1.write('lm incorrect but nb correct:')
    #for i in lmIncorrect_nbCorrect:
    #    outputFile1.write('%d, %s' %(i, valCorpus[i]))
    #outputFile1.close()
    
    print('both correct:', len(bothCorrect))
    print('both incorrect:', len(bothIncorrect))
    print('lm correct but nb incorrect:', len(lmCorrect_nbIncorrect))
    print('lm incorrect but nb correct:', len(lmIncorrect_nbCorrect))
    
    bothCorrect = []
    bothIncorrect = []
    lmCorrect_nbIncorrect = []
    lmIncorrect_nbCorrect = []
    for i in range(len(corpusDeceptiveVal)):
        if i in deceptiveOutCorrect and i in nbDecepCorrect:
            bothCorrect.append(i)
        elif i in deceptiveOutCorrect and i in nbDecepIncorrect:
            lmCorrect_nbIncorrect.append(i)
        elif i in deceptiveOutIncorrect and i in nbDecepCorrect:
            lmIncorrect_nbCorrect.append(i)
        elif i in deceptiveOutIncorrect and i in nbDecepIncorrect:
            bothIncorrect.append(i)

    print('both correct:', len(bothCorrect))
    print('both incorrect:', len(bothIncorrect))
    print('lm correct but nb incorrect:', len(lmCorrect_nbIncorrect))
    print('lm incorrect but nb correct:', len(lmIncorrect_nbCorrect))

    #valCorpus = []
    #for line in open('DATASET/validation/deceptive.txt', 'r'):
    #    valCorpus.append(line)
    #outputFile2 = open('deceptive-val.txt', 'w')
    #outputFile2.write('lm correct but nb incorrect:')
    #for i in lmCorrect_nbIncorrect:
    #    outputFile2.write('%d, %s' %(i, valCorpus[i]))
    #outputFile2.write('lm incorrect but nb correct:')
    #for i in lmIncorrect_nbCorrect:
    #    outputFile2.write('%d, %s' %(i, valCorpus[i]))
    #outputFile2.close()


    '''
    # test predication
    corpusTest = getCorpus('DATASET/test/test.txt')
    bigramCorpusTest = getBigramCorpus(corpusTest)
    
    # add k here
    k = 0.06
    unigramTruthfulTrainAddK = unigramAddK(corpusTruthfulTrainUnk, k)
    unigramDeceptiveTrainAddK = unigramAddK(corpusDeceptiveTrainUnk, k)
    bigramTruthfulTrainAddk = bigramAddK(corpusTruthfulTrainUnk, k, unigramTruthfulTrainUnk[1])
    bigramDeceptiveTrainAddk = bigramAddK(corpusDeceptiveTrainUnk, k, unigramDeceptiveTrainUnk[1])
    bigramTestPerp = bigramClassifier(bigramCorpusTest, bigramTruthfulTrainAddk, bigramDeceptiveTrainAddk, \
            unigramTruthfulTrainAddK, unigramDeceptiveTrainAddK, unigramTruthfulTrainUnk[1], unigramDeceptiveTrainUnk[1], k)

    #unigramTestPerp = unigramClassifier(corpusTest, unigramTruthfulTrainUnk[0], unigramDeceptiveTrainUnk[0])
    #bigramTestPerp = bigramClassifier(bigramCorpusTest, bigramTruthfulTrainUnk, bigramDeceptiveTrainUnk, unigramTruthfulTrainUnk[0], unigramDeceptiveTrainUnk[0])

    #outputFile1 = open('test-unigram-output.csv', 'w')
    #outputFile1.write('Id,Prediction\n')
    #for i in range(len(unigramTestPerp)):
        #outputFile1.write(str(i) + ',' + str(unigramTestPerp[i]) + '\n')
    #outputFile1.close()

    outputFile2 = open('test-bigram.csv', 'w')
    outputFile2.write('Id,Prediction\n')
    for i in range(len(bigramTestPerp)):
        outputFile2.write(str(i) + ',' + str(bigramTestPerp[i]) + '\n')
    outputFile2.close()'''

    '''# overview of the training data
    sortedUnigramTPr = sorted(unigramTruthful[0].items(), key=lambda x: x[1], reverse=True)
    sortedUnigramTCo = sorted(unigramTruthful[1].items(), key=lambda x: x[1], reverse=True)
    #print('Top 20 unigram probabilities for Truthful: ', sortedUnigramTPr[:50])
    print('Corresponding count: ', sortedUnigramTCo[:50])

    sortedUnigramDPr = sorted(unigramDeceptive[0].items(), key=lambda x: x[1], reverse=True)
    sortedUnigramDCo = sorted(unigramDeceptive[1].items(), key=lambda x: x[1], reverse=True)
    #print('Top 20 unigram probabilities for Deceptive: ', sortedUnigramDPr[:50])
    print('Corresponding count: ', sortedUnigramDCo[:50])
    
    n = len(sortedUnigramTCo)
    m = n/100
    outY = []
    for j in range(100):
        temp = sortedUnigramTCo[j*m:(j+1)*m]
        table1 = []
        table2 = []
        for i in temp:
            table1.append(i[0])
            table2.append(i[1])
        outY.append(1.*sum(table2)/len(table2))
    plt.bar(range(len(outY)), outY, align='center')
    plt.xlabel('%')
    plt.ylabel('average')
    plt.title('Frequency distribution for train/truthful')
    plt.show()

    n = len(sortedUnigramDCo)
    m = n/100
    outY = []
    for j in range(100):
        temp = sortedUnigramDCo[j*m:(j+1)*m]
        table1 = []
        table2 = []
        for i in temp:
            table1.append(i[0])
            table2.append(i[1])
        outY.append(1.*sum(table2)/len(table2))
    plt.bar(range(len(outY)), outY, align='center')
    plt.xlabel('%')
    plt.ylabel('average')
    plt.title('Frequency distribution for train/deceptive')
    plt.show()
    '''
    
    '''
    # plot the frequency for train truthful
    table1 = []
    table2 = []
    for i in sortedUnigramTCo[:50]:
        table1.append(i[0])
        table2.append(i[1])
    plt.bar(range(len(table1)), table2, align='center')
    plt.xticks(range(len(table1)), table1, size='xx-small')
    plt.xlabel('word')
    plt.ylabel('frequency')
    plt.title('Frequency of top 50 words in train/truthful')
    for index,data in enumerate(table2):
        plt.text(x=index-0.5, y =data+1.5, s=data , fontdict=dict(fontsize=7))
    plt.show()
    
    # plot the frequency for train deceptive
    table1 = []
    table2 = []
    for i in sortedUnigramDCo[:50]:
        table1.append(i[0])
        table2.append(i[1])
    plt.bar(range(len(table1)), table2, align='center')
    plt.xticks(range(len(table1)), table1, size='xx-small')
    plt.xlabel('word')
    plt.ylabel('frequency')
    plt.title('Frequency of top 50 words in train/deceptive')
    for index,data in enumerate(table2):
        plt.text(x=index-0.5, y =data+1.5, s=data , fontdict=dict(fontsize=7))
    plt.show()
    ''' 