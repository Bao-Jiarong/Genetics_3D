'''
 *  Author : Bao Jiarong
 *  Contact: bao.salirong@gmail.com
 *  Created  On: 2020-05-29
 *  Modified On: 2020-05-29
 '''
import random
import numpy as np

def uniform(low,high):
    return int(low+((high-low)*random.random()))

def randfloat(low,high):
    return low+((high-low)*random.random())

def rand_floats(low,high,n):
    a =[]
    for i in range(n):
        t = low+((high-low)*random.random())
        a.append(round(t,5))
    return a

class Algo:
    #--------------------------------------------------------
    # Constructor
    #---------------------------------------------------------
    def __init__(self,f,T,low,high,p_size):
        self.f      =  f
        self.T      =  T
        self.low    =  low
        self.high   =  high
        self.p_size =  p_size
