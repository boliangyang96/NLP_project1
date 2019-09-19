#process file
def getCorpus(filename):
    corpus = []
    for line in open(filename, 'r'):
        #define the start word <s>
        #newLine = ['<s>']
        newLine = []
        for word in line.split():
            newLine.append(word.lower())
        # append stop word
        #new_line.append('</s>')
        corpus.append(newLine)
    return corpus

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
        unigramProbs[word] /= (1.0 * totalCount)  
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

# test one line file
c = getCorpus('test.txt')
u = getUnigramProbs(c)
b = getBigramProbs(c, u[1])
print(b)

# process training data
corpusTruthful = getCorpus('DATASET/train/truthful.txt')
corpusDeceptive = getCorpus('DATASET/train/deceptive.txt')
unigramTruthful = getUnigramProbs(corpusTruthful)
unigramDeceptive = getUnigramProbs(corpusDeceptive)
bigramTruthful = getBigramProbs(corpusTruthful, unigramTruthful[1])
bigramDeceptive = getBigramProbs(corpusDeceptive, unigramDeceptive[1])

sortedUnigramTPr = sorted(unigramTruthful[0].items(), key=lambda x: x[1], reverse=True)
sortedUnigramTCo = sorted(unigramTruthful[1].items(), key=lambda x: x[1], reverse=True)
#print('Top 20 unigram probabilities for Truthful: ', sortedUnigramTPr[:20])
print('Corresponding count: ', sortedUnigramTCo[:20])

sortedUnigramDPr = sorted(unigramDeceptive[0].items(), key=lambda x: x[1], reverse=True)
sortedUnigramDCo = sorted(unigramDeceptive[1].items(), key=lambda x: x[1], reverse=True)
#print('Top 20 unigram probabilities for Deceptive: ', sortedUnigramDPr[:20])
print('Corresponding count: ', sortedUnigramDCo[:20])