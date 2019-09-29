from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords

import csv

## Constant
NLTK = stopwords.words('english')

## Global settings
N_GRAM = (1,2)  ## tuple with lower and upper bound; eg: (1,1) is unigram, (1,2) is combination of unigram and bigram
STOPWORDS = 'english'   ## values: None, 'english', NLTK
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

    """
    csv output code
    with open("nb_result.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow(["Id","Prediction"])
        t_count = f_count = 0
        for i,y in enumerate(y_val):
            if y:
                writer.writerow([t_count, y_val_pred[i]])
                t_count += 1
            else:
                writer.writerow([f_count, y_val_pred[i]])
                f_count += 1
    """

    """ txt result output """
    tr_t = list()
    tr_f = list()
    de_t = list()
    de_f = list()
    tr_count = de_count = 0
    for i,y in enumerate(y_val):
        if y:
            if y_val_pred[i]: tr_t.append(tr_count)
            else: tr_f.append(tr_count)
            tr_count += 1
        else:
            if y_val_pred[i]: de_f.append(de_count)
            else: de_t.append(de_count)
            de_count += 1

    print("truthful output correct:", tr_t)
    print("truthful output incorrect:", tr_f)
    print("deceptive output correct:", de_t)
    print("deceptive output incorrect:", de_f)
