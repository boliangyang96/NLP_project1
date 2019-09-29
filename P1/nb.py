from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords

## Constant
NLTK = stopwords.words('english')

## Global settings
N_GRAM = (1,2)  ## tuple with lower and upper bound; eg: (1,1) is unigram, (1,2) is combination of unigram and bigram
STOPWORDS = NLTK   ## values: None, NLTK
VECTORIZER = CountVectorizer   ## values: CountVectorizer, TfidfVectorizer
NB_METHOD = MultinomialNB   ## values: GaussianNB, MultinomialNB, BernoulliNB


def parse_dataset(f, x=None, y=None, label=None):
    if x is None: x = list()
    if y is None: y = list()

    for line in f:
        x.append(line.strip())
        y.append(label)

    return x,y


if __name__ == "__main__":
    f_tr = open("./DATASET/train/truthful.txt")
    f_de = open("./DATASET/train/deceptive.txt")
    f_val_t = open("./DATASET/validation/truthful.txt")
    f_val_f = open("./DATASET/validation/deceptive.txt")
    f_test = open("./DATASET/test/test.txt")

    x_train, y_train = parse_dataset(f_tr, label=True)
    x_train, y_train = parse_dataset(f_de, x_train, y_train, label=False)
    x_val, y_val = parse_dataset(f_val_t, label=True)
    x_val, y_val = parse_dataset(f_val_f, x_val, y_val, label=False)
    x_test, _ = parse_dataset(f_test)

    ## vectorize
    vectorizer = VECTORIZER(ngram_range=N_GRAM, stop_words=STOPWORDS)
    x_transform = vectorizer.fit_transform(x_train).toarray()

    ## train
    nb = NB_METHOD()
    clf = nb.fit(x_transform, y_train)

    ## test
    x_val_trans = vectorizer.transform(x_val).toarray()
    y_val_pred = clf.predict(x_val_trans)
    print(sum(y_val_pred == y_val))  ## correct
    print(len(y_val))  ## total
    print(sum(y_val_pred == y_val) / len(y_val))  ## accuracy
