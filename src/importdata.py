import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

#this script uses CountVectorizer to process corpus.

def main():

    corpus = importasline('../data/shakespear.txt')
    vectorizer = CountVectorizer(min_df=1)
    X = vectorizer.fit_transform(corpus)
    analyze = vectorizer.build_analyzer()
    Y = [[vectorizer.vocabulary_[x] for x in analyze(corpus[i])] for i in range(len(corpus))]
    print(Y)
    print(corpus[0:3])



def importaspoem(filename):

    corpus = []
    article = ''
    with open(filename) as file:
        for line in file:
            if(len(line.strip())<=4):
                #if the length of a sentence is smaller than 4,
                #then that means an end of poem is met.
                corpus.append(article)
                article = ''
            else:
                article += ' '
                article += line.strip()

    return corpus



def importasline(filename):

    corpus = []
    article = ''
    with open(filename) as file:
        for line in file:
            if(len(line.strip())>4):

                corpus.append(line.strip())

    return corpus



if __name__ == "__main__":
    main()