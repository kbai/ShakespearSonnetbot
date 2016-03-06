import random
import numpy as np
from importdata import importasline
from sklearn.feature_extraction.text import CountVectorizer



class modelhmm():
    def __init__(self,m,n):
        self.obs_ = np.random.rand( m , n )
        self.trans_ = np.random.rand(m + 2 , m + 2)
        self.m_ = m   # number of POS
        self.n_ = n   # number of words
        self.start_ = m
        self.end_ = m+1
        self.trans_[:,self.start_] = 0.0
        self.trans_[self.end_,:] = 0.0

        for i in range(m):
            self.obs_[i,:] = self.obs_[i,:]/np.sum(self.obs_[i,:])
            self.trans_[i,:] = self.trans_[i,:]/np.sum(self.trans_[i,:])

        print self.trans_
        #we store transition possibility from starting state and to end state
        #at the end of the transition matrix





    def viterbi(self,data):

        logobs = np.log(self.obs_)
        logtrans = np.log(self.trans_)

        ns = self.m_ # #POS
        nd = len(data)
        plen = np.zeros((nd,ns))
        path = np.zeros((nd,ns))
        plen[0,:] = logobs[:,data[0]] + logtrans[self.start_,0:ns]
        for ii in range(1,nd):
            for jj in range(0,ns):
                maxp = - float('inf')
                for kk in range(0,ns):
                    tmpp = plen[ii-1,kk] + logtrans[kk,jj] + logobs[jj,data[ii]]
                    if(tmpp > maxp):
                        path[ii,jj] = kk
                        maxp = tmpp
                plen[ii,jj] = maxp
        minp = -float('inf')
        max_path = np.full((nd,),0)
        max_end_tag = -1
        for ii in range(0,ns):
            if (plen[-1,ii] > minp):
                minp = plen[-1,ii]
                max_end_tag = ii
        max_path[-1] = max_end_tag
        for ii in range(nd,1,-1):
            max_path[ii-2] = path[ii-1,max_path[ii-1]]

        return plen,path,max_path

def main():
    corpus = importasline()
    vectorizer = CountVectorizer(min_df=1)
    X = vectorizer.fit_transform(corpus)
    analyze = vectorizer.build_analyzer()
    Y = [[vectorizer.vocabulary_[x] for x in analyze(corpus[i])] for i in range(len(corpus))]
    print(Y)
    words = vectorizer.get_feature_names()
    num_of_hidden_states = 10
    print(len(words))
    hmm = modelhmm(num_of_hidden_states,len(words))
    plen,path,max_path = hmm.viterbi(Y[0])
    print(max_path)


if __name__ == "__main__":
    main()


