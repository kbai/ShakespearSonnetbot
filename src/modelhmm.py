import random
import numpy as np
from importdata import importasline
from sklearn.feature_extraction.text import CountVectorizer
import warnings
warnings.filterwarnings('ignore')



class modelhmm():
    def __init__(self,m,n):
        self.obs_ = np.random.rand(m,n)               # obser matrix
        self.trans_ = np.random.rand(m + 2 , m + 2)   # trans matrix : transition from row label to column label
        self.m_ = m   # number of POS
        self.n_ = n   # number of words
        self.start_ = m
        self.end_ = m + 1
        self.trans_[:,self.start_] = 0.0
        self.trans_[self.end_,:] = 0.0
        self.alpha_set = list([])
        self.beta_set = list([])

        for i in range(m):
            self.obs_[i,:] = self.obs_[i,:]/np.sum(self.obs_[i,:])
            self.trans_[i,:] = self.trans_[i,:]/np.sum(self.trans_[i,:])

        ###print self.trans_, 'self.trans_'
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


    def training(self,sequence_set):
        return sequence_set

    def forward_backward(self,data):
    #  Each row in data is a sequence
        
        self.alpha_set = list([])
        self.beta_set = list([])
        for it,sequence in enumerate(data):
            Mj = len(sequence)   # Mj denotes the length of sequence
            '''
            forward algorithm
            alpha:
                row : hidden state
                column : number of columns == number of sequence 
            '''
            alpha = np.zeros((self.m_, Mj))

            alpha[:, 0] =  [A * O for A,O in zip(self.trans_[self.start_, 0:4],self.obs_[:,sequence[0]])]
            assert self.trans_[self.start_, 0:4][0] * self.obs_[:,sequence[0]][0] == alpha[0, 0], 'zip should be pointwise pick'

            for ii in range(1, Mj):
                alpha_tmp = [self.obs_[label, ii] * np.dot(alpha[:, ii - 1],self.trans_[0:4, label]) for label in range(self.m_)]
                assert self.m_ == len(alpha_tmp), 'self.m_ == len(alpha_tmp) should hold'
                alpha[:,ii] = np.array(alpha_tmp)
            
            self.alpha_set.append(alpha)

            
            '''
            backward algorithm
            beta:
                row : hidden state
                column : number of columns == number of sequence 
            '''
            beta = np.zeros((self.m_,Mj))
            beta[:,Mj-1] = np.ones(self.m_)
            for ii in reversed(range(0,Mj-1)):
                tmp = [\
                                [ b * A * O \
                                    for b,A,O in zip(beta[:,ii + 1], self.trans_[label, 0:4], self.obs_[:, sequence[ii + 1]])\
                                ]\
                            for label in range(self.m_)\
                       ]
                beta_tmp = [sum(line) for line in tmp]
                assert self.m_ == len(beta_tmp), 'self.m_ == len(beta_tmp) should hold '
                beta[:, ii] = np.array(beta_tmp)
                
            self.beta_set.append(beta)

        
        return self.alpha_set, self.beta_set


def main():
    corpus = importasline('../data/shakespear.txt')
    vectorizer = CountVectorizer(min_df=1)
    X = vectorizer.fit_transform(corpus)
    analyze = vectorizer.build_analyzer()
    num_of_hidden_states = 4
    # each element in Y contains words in a line, the label of word starts from 0
    Y = [[vectorizer.vocabulary_[x] for x in analyze(corpus[i])] for i in range(len(corpus))]
    words = vectorizer.get_feature_names()
    
    hmm = modelhmm(num_of_hidden_states,len(words))
    plen,path,max_path = hmm.viterbi(Y[0])
    hmm.forward_backward(Y)


if __name__ == "__main__":
    main()


