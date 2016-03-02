import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

#this script uses CountVectorizer to process corpus.

def main():
    print('kkk')
    corpus = [
        'This is the first document.',
        'This is the second second document.',
        'And the third one.',
        'Is this the first document?',
    ]
    corpus = []
    article = ''
    with open('../data/shakespear.txt') as file:
        for line in file:
            if(len(line.strip())<=4):
                #if the length of a sentence is smaller than 4,
                #then that means an end of poem is met.
                corpus.append(article)
            else:
                article += line.strip()
    print(corpus[2])
  #  vectorizer = CountVectorizer(min_df=1)
  #  X = vectorizer.fit_transform(corpus)

  #  analyze = vectorizer.build_analyzer()
  #  print(vectorizer.get_feature_names())

  #  print(X.toarray())



if __name__ == "__main__":
    main()