import random
import numpy as np
from importdata import importasline
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
np.set_printoptions(threshold=40)


class Markov():
    def __init__(self, m, corpus_in,name):
        self.trans_ = np.random.rand(m + 2, m + 2)
        self.m_ = m  # number of POS
        self.start_ = m  # we put start token as the m+1 th token
        self.end_ = m + 1  # we put end token as the m+2 th token
        self.trans_[:, self.start_] = 0.0
        self.trans_[self.end_, :] = 0.0
        self.corpus = corpus_in
        self.y = []
        self.epsilon = 0.0
        self.filename = name
        self.Markov_table = np.zeros((self.m_ + 2,self.m_ + 2))
        self.secondordermarkov_mapping = np.zeros((self.m_ + 2,self.m_ + 2)) - 1
        self.inversetable = []

        self.generating_markov_trans_table()
        self.generatingsecondordermarkov_mapping()
        self.generatingsecondordermarkov()



    def generatingsecondordermarkov(self):
        counter = 0
        self.m2_ = len(self.inversetable)
        self.M2start_ = self.m2_
        self.M2end_ = self.m2_+1
        self.secondordermarkov_table = np.zeros((self.m2_+2,self.m2_+2))

        for article in self.corpus:
            i = self.secondordermarkov_mapping[self.start_,article[0]]
            j = self.secondordermarkov_mapping[article[0],article[1]]
            self.secondordermarkov_table[i,j]+=1
            self.secondordermarkov_table[self.M2start_,i] += 1
            i = self.secondordermarkov_mapping[article[-2],article[-1]]
            j = self.secondordermarkov_mapping[article[-1],self.end_]
            self.secondordermarkov_table[i,j]+=1
            self.secondordermarkov_table[j,self.M2end_] += 1

            for k in range(len(article) - 2):
                    i = self.secondordermarkov_mapping[article[k],article[k+1]]
                    j = self.secondordermarkov_mapping[article[k+1],article[k+2]]
                    self.secondordermarkov_table[i,j]+=1

        for l in range(self.m2_+1):
            if(np.sum(self.secondordermarkov_table[l,:])==0):
                print(self.inversetable[l])
            else:
                print('yes')
            self.secondordermarkov_table[l,:] = self.secondordermarkov_table[l,:]/np.sum(self.secondordermarkov_table[l,:])


    def generatingsecondordermarkov_mapping(self):
        counter = 0
        for article in self.corpus:
            counter = self.insert(self.start_, article[0],self.secondordermarkov_mapping,counter)
            counter = self.insert(article[-1], self.end_,self.secondordermarkov_mapping,counter)
            for i in range(len(article) - 1):
                counter = self.insert(article[i], article[i+1],self.secondordermarkov_mapping,counter)

    def insert(self,x,y,table,counter):
        if(table[x,y]<-0.5):
            table[x,y] = counter
            counter += 1
            self.inversetable.append([x,y])
            return counter
        else:
            return counter

    def generating_markov_trans_table(self):
        for article in self.corpus:
            self.Markov_table[self.start_, article[0]] += 1
            self.Markov_table[article[-1],self.end_] += 1
            for i in range(len(article) - 1):
                self.Markov_table[article[i],article[i+1]] += 1
                print(article[i],article[i+1])
        for l in range(self.m_+1):
            self.Markov_table[l,:] = self.Markov_table[l,:]/np.sum(self.Markov_table[l,:])


    def savemodel(self):
        np.savetxt('../model/'+self.filename+'trans.txt',self.trans_)
        np.savetxt('../model/'+self.filename+'obs.txt',self.obs_)


    def loadmodel(self):
        self.trans_ = np.loadtxt('../model/'+self.filename+'trans.txt')
        self.obs_ = np.loadtxt('../model/'+self.filename+'obs.txt')


    def analyzing_word(self,words):
        df = pd.DataFrame((self.obs_).transpose(),index=words)
        df.to_csv('../model/'+self.filename+'withword.txt',index=True,header=True,sep=' ')







    def generating_random_line(self):
        line = []
        linew = []
        currentstate = self.M2start_
        while currentstate!=self.M2end_:
            currentstate = random_distr(self.secondordermarkov_table[currentstate,:])
            line.append(currentstate)
        for token in line[:-1]:
            word = self.inversetable[token][1]
            linew.append(word)

        return line,linew



    def find_max_Y(self):
        for line in self.corpus:
            plen, path, max_path = self.viterbi(line)
            self.y.append(max_path)


def main():

    corpus = importasline('../data/shakespear_modified.txt',ignorehyphen = True)
   

    vectorizer = CountVectorizer(min_df=1)
    X = vectorizer.fit_transform(corpus)
    analyze = vectorizer.build_analyzer()
    Y = [[vectorizer.vocabulary_[x] for x in analyze(corpus[i])] for i in range(len(corpus))]
    words = vectorizer.get_feature_names()
    num_of_hidden_states = 1000
    print(len(words))
    hmm = Markov( len(words), Y, 'modelnhidden1000groupA')
    print(len(hmm.inversetable))
    for i in range(20):
        [line,linew]=hmm.generating_random_line()
        print(linew)
        robotpoem = ''
        for j in linew[:-1]:
            robotpoem+=' '+words[j]+' '
        print(robotpoem)
            
    #print(corpus)
    #print(Y)
    #print(hmm.Markov_table)
    #print(hmm.secondordermarkov_mapping)
    #print(hmm.secondordermarkov_table)
    #print(hmm.inversetable)



def random_distr(l):
    r = random.uniform(0, 1)
    s = 0
    item = 0
    for  prob in l:
        s += prob
        if s >= r:
            return item
        item+=1
    return item

if __name__ == "__main__":
    main()