import numpy as np

# process file
def getCorpus(filename):
    corpus = []
    for line in open(filename, 'r'):
        #define the start word <s>
        newLine = ['<s>']
        for word in line.split():
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
    for word in unigramCount:
        if unigramCount[word] == 1:
            temp.append(word)
    for i in range(len(corpus)):
        for j in range(len(corpus[i])):
            if corpus[i][j] in temp:
                corpus[i][j] = "<unk>"
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
    return np.exp(1.0 / len(unigramProbs) * np.sum(result))

# compute the perplexity for bigram
def bigramPerplexity(line, bigramProbs, startProb):
    result = [-np.log(startProb)]
    for bigramTuple in line:
        if bigramTuple in bigramProbs:
            result.append(-np.log(bigramProbs[bigramTuple]))
        elif (bigramTuple[0], '<unk>') in bigramProbs:
            result.append(-np.log(bigramProbs[(bigramTuple[0], '<unk>')]))
        elif ('<unk>', bigramTuple[1]) in bigramProbs:
            result.append(-np.log(bigramProbs[('<unk>', bigramTuple[1])]))
        else:
            result.append(-np.log(bigramProbs[('<unk>', '<unk>')]))
    return np.exp(1.0 / len(bigramProbs) * np.sum(result))

# classify the reviews using unigram perplexity, 0 for truthful, 1 for deceptive
def unigramClassifier(corpus, truthfulProbs, deceptiveProbs):
    result = []
    for line in corpus:
        truthfulPerplexity = unigramPerplexity(line, truthfulProbs)
        deceptivePerplexity = unigramPerplexity(line, deceptiveProbs)
        if truthfulPerplexity <= deceptivePerplexity:
            result.append(0)
        else:
            result.append(1)
    return result

# classify the reviews using bigram perplexity, 0 for truthful, 1 for deceptive
# need to get the unigram probability with unk first
def bigramClassifier(corpus, truthfulProbs, deceptiveProbs, unigramTruthfulProbsUnk, unigramDeceptiveProbsUnk):
    result = []
    for line in corpus:
        truthfulPerplexity = bigramPerplexity(line, truthfulProbs, unigramTruthfulProbsUnk['<s>'])
        deceptivePerplexity = bigramPerplexity(line, deceptiveProbs, unigramDeceptiveProbsUnk['<s>'])
        if truthfulPerplexity <= deceptivePerplexity:
            result.append(0)
        else:
            result.append(1)
    return result

if __name__ == "__main__":
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
    #print('Top 20 unigram probabilities for Truthful: ', sortedUnigramTPr[:50])
    print('Corresponding count: ', sortedUnigramTCo[:50])

    sortedUnigramDPr = sorted(unigramDeceptive[0].items(), key=lambda x: x[1], reverse=True)
    sortedUnigramDCo = sorted(unigramDeceptive[1].items(), key=lambda x: x[1], reverse=True)
    #print('Top 20 unigram probabilities for Deceptive: ', sortedUnigramDPr[:50])
    print('Corresponding count: ', sortedUnigramDCo[:50])