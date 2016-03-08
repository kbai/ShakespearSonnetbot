from importdata import importasline
from sklearn.feature_extraction.text import CountVectorizer

'''
	This code set implement grouping schemes
	Each grouping schemes returns data sets for training

	Note:
			filename can only take '../data/shakespear_modified.txt' or
'''



def grouping1(filename):

	###
    ###	scheme	ABAB CDCD EFEF GG
    ###

    corpus = importasline(filename)
    vectorizer = CountVectorizer(min_df=1)
    X = vectorizer.fit_transform(corpus)
    analyze = vectorizer.build_analyzer()

    dataSet = [[vectorizer.vocabulary_[x] for x in analyze(corpus[i])] for i in range(len(corpus))] 
    assert 0 == len(dataSet) % 14, 'number of lines should be multiple of 14' 
    
    groupA = dataSet[ 0: :14] + dataSet[ 2: :14]
    groupB = dataSet[ 1: :14] + dataSet[ 3: :14]
    groupC = dataSet[ 4: :14] + dataSet[ 6: :14]
    groupD = dataSet[ 5: :14] + dataSet[ 7: :14]
    groupE = dataSet[ 8: :14] + dataSet[10: :14]
    groupF = dataSet[ 9: :14] + dataSet[11: :14]
    groupG = dataSet[12: :14] + dataSet[13: :14]

    assert groupA[0] == dataSet[0], 'groupA[0] == dataSet[0] should hold'
    assert groupE[1] == dataSet[22], 'groupA[8] == dataSet[22] should hold'

    return groupA, groupB, groupC, groupD, groupE, groupF, groupG

def grouping2(filename):

	###
    ###	scheme	AAAA BBBB CCCC DD
    ###

    corpus = importasline(filename)
    vectorizer = CountVectorizer(min_df=1)
    X = vectorizer.fit_transform(corpus)
    analyze = vectorizer.build_analyzer()

    dataSet = [[vectorizer.vocabulary_[x] for x in analyze(corpus[i])] for i in range(len(corpus))] 
    assert 0 == len(dataSet) % 14, 'number of lines should be multiple of 14' 
    
    groupA = dataSet[ 0: :14] + dataSet[ 1: :14] + dataSet[ 2: :14] + dataSet[ 3: :14]
    groupB = dataSet[ 4: :14] + dataSet[ 5: :14] + dataSet[ 6: :14] + dataSet[ 7: :14]
    groupC = dataSet[ 8: :14] + dataSet[ 9: :14] + dataSet[10: :14] + dataSet[11: :14]
    groupD = dataSet[12: :14] + dataSet[13: :14]

    assert groupB[0] == dataSet[4], 'groupB[0] == dataSet[4] should hold'
    assert groupC[1] == dataSet[22], 'groupC[1] == dataSet[22] should hold'

    return groupA, groupB, groupC, groupD

def grouping3(filename):

	###
    ###	scheme	AAAA AAAA AAAA DD
    ###

    corpus = importasline(filename)
    vectorizer = CountVectorizer(min_df=1)
    X = vectorizer.fit_transform(corpus)
    analyze = vectorizer.build_analyzer()

    dataSet = [[vectorizer.vocabulary_[x] for x in analyze(corpus[i])] for i in range(len(corpus))] 
    assert 0 == len(dataSet) % 14, 'number of lines should be multiple of 14' 
    
    groupA = dataSet[ 0: :14] + dataSet[ 1: :14] + dataSet[ 2: :14] + dataSet[ 3: :14] + \
    		 dataSet[ 4: :14] + dataSet[ 5: :14] + dataSet[ 6: :14] + dataSet[ 7: :14] + \
    		 dataSet[ 8: :14] + dataSet[ 9: :14] + dataSet[10: :14] + dataSet[11: :14]
    groupB = dataSet[12: :14] + dataSet[13: :14]

    assert groupB[1] == dataSet[26], 'groupB[1] == dataSet[26]'

    return groupA, groupB


def main():
	
	grouping3('../data/shakespear_modified.txt')

if __name__ == "__main__":
    main()

