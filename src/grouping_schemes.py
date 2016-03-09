from importdata import importasline
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

'''
	This code set implement grouping schemes
	Each grouping schemes returns data sets for training

	Note:
			filename can only take '../data/shakespear_modified.txt' 
								   '../data/spenser_modified.txt'
								   '../data/all_modified.txt'
'''

def grouping1(filename):

	###
    ###	scheme	ABAB CDCD EFEF GG
    ###

    dataSet = importasline(filename)

    assert 0 == len(dataSet) % 14, 'number of lines should be multiple of 14' 
    num_poem = len(dataSet) / 14
    groupA = [] ; groupB = [] ; groupC = [] ; groupD = [] ; groupE = [] ; groupF = [] ; groupG = []

    for it in range(num_poem):
    	start = it * 14
    	groupA.append(dataSet[start + 0]) ; groupA.append(dataSet[start + 2])
    	groupB.append(dataSet[start + 1]) ; groupB.append(dataSet[start + 3])
    	groupC.append(dataSet[start + 4]) ; groupC.append(dataSet[start + 6])
    	groupD.append(dataSet[start + 5]) ; groupD.append(dataSet[start + 7])
    	groupE.append(dataSet[start + 8]) ; groupE.append(dataSet[start +10])
    	groupF.append(dataSet[start + 9]) ; groupF.append(dataSet[start +11])
    	groupG.append(dataSet[start +12]) ; groupG.append(dataSet[start +13])

    assert groupA[1] == dataSet[2], 'groupA[0] == dataSet[0] should hold'
    assert groupC[1] == dataSet[6], 'groupA[8] == dataSet[22] should hold'

    return groupA, groupB, groupC, groupD, groupE, groupF, groupG

def grouping2(filename):

	###
    ###	scheme	AAAA BBBB CCCC DD
    ###

    dataSet = importasline(filename)

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

    dataSet = importasline(filename)

    assert 0 == len(dataSet) % 14, 'number of lines should be multiple of 14' 
    
    groupA = dataSet[ 0: :14] + dataSet[ 1: :14] + dataSet[ 2: :14] + dataSet[ 3: :14] + \
    		 dataSet[ 4: :14] + dataSet[ 5: :14] + dataSet[ 6: :14] + dataSet[ 7: :14] + \
    		 dataSet[ 8: :14] + dataSet[ 9: :14] + dataSet[10: :14] + dataSet[11: :14]
    groupB = dataSet[12: :14] + dataSet[13: :14]

    assert groupB[1] == dataSet[26], 'groupB[1] == dataSet[26]'

    return groupA, groupB






def main():
    [groupA,groupB,groupC,groupD,groupE,groupF,groupG]=grouping1('../data/shakespear_modified.txt')
    np.savetxt('../data/grouping1/groupA.txt',groupA,fmt='%s')
    np.savetxt('../data/grouping1/groupB.txt',groupB,fmt='%s')
    np.savetxt('../data/grouping1/groupC.txt',groupC,fmt='%s')
    np.savetxt('../data/grouping1/groupD.txt',groupD,fmt='%s')
    np.savetxt('../data/grouping1/groupE.txt',groupE,fmt='%s')
    np.savetxt('../data/grouping1/groupF.txt',groupF,fmt='%s')
    np.savetxt('../data/grouping1/groupG.txt',groupG,fmt='%s')







if __name__ == "__main__":
    main()

