import random
import numpy as np

class modelhmm():
    def __init__(self,m,n,corpus):
        self.A = random.random(m,n)
        self.O = random.random(n,n)


