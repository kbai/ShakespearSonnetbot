import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

#this script uses CountVectorizer to process corpus.

def main():

    corpus = importasline()
    print(corpus[0:10])

    vectorizer = CountVectorizer(min_df=1)
    vectorizer
    X = vectorizer.fit_transform(corpus[0:10])

    analyze = vectorizer.build_analyzer()
    print(vectorizer.get_feature_names())

    print(X[0:10].toarray())


def importaspoem():

    corpus = []
    article = ''
    with open('../data/shakespear.txt') as file:
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



def importasline():

    corpus = []
    article = ''
    with open('../data/shakespear.txt') as file:
        for line in file:
            if(len(line.strip())>4):

                corpus.append(line.strip())

    return corpus



if __name__ == "__main__":
    main()