import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

#this script uses CountVectorizer to process corpus.

def main():

    corpus = importasline()
    vectorizer = CountVectorizer(min_df=1)
    X = vectorizer.fit_transform(corpus)
    analyze = vectorizer.build_analyzer()
    Y = [[vectorizer.vocabulary_[x] for x in analyze(corpus[i])] for i in range(len(corpus))]
    print(Y)
    print(corpus[0:3])



def importaspoem(ignorehyphen):

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
                if ignorehyphen:
                    article += line.replace('-','').replace("'",'').strip()
                else:
                    article += line.replace("'",'').strip()

    return corpus



def importasline(ignorehyphen):

    corpus = []
    article = ''
    with open('../data/shakespear.txt') as file:
        for line in file:
            if(len(line.strip())>4):
                if ignorehyphen:
                    corpus.append(line.replace('-','').replace("'",'').strip())
                else:
                    corpus.append(line.replace("'",'').strip())

    return corpus



if __name__ == "__main__":
    main()