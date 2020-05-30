'''
 *  Author : Bao Jiarong
 *  Contact: bao.salirong@gmail.com
 *  Created  On: 2020-05-29
 *  Modified On: 2020-05-29
 '''
import random
import numpy as np
from .algo import *

class EV(Algo):
    #--------------------------------------------------------
    # Constructor
    #--------------------------------------------------------
    def __init__(self,f,T,low,high,p_size,eps=1e-3):
        Algo.__init__(self,f,T,low,high,p_size)
        self.eps = eps

    #---------------------------Genetic Operators-------------
    def creation(self):
        return randfloat(self.low,self.high)

    def crossover(self,a,b,alpha = 0.25):
        return alpha*a+(1-alpha)*b

    def mutation(self,a):
        r = random.random() + self.eps
        return a + r * a

    def reproduction(self,px,py):
        n = len(px)
        p1 = []
        p2 = []

        for i in range(int(n/4)):
            p1.append(self.creation())
            p2.append(self.creation())

        for i in range(int(n/4)):
            a = px[uniform(0,n)]
            b = px[uniform(0,n)]

            c = py[uniform(0,n)]
            d = py[uniform(0,n)]
            tx = self.crossover(a,b)
            ty = self.crossover(c,d)
            p1.append(tx)
            p2.append(ty)

        for i in range(int(n/4)):
            a = px[int(randfloat(0,n))]
            b = py[int(randfloat(0,n))]
            t1 = self.mutation(a)
            t2 = self.mutation(b)
            p1.append(t1)
            p2.append(t2)
            return p1,p2

    #-----------------------------Replacement------------------
    def uniform_replacement(self,px,py,p1x,p1y):
        N = len(px)
        n1 = len(px)
        n2 = len(p1x)
        t1 = []
        t2 = []
        for i in range(N):
            a = uniform(0,n1)
            b = uniform(0,n2)
            x = px[a]
            y = py[a]

            x1 = p1x[b]
            y1 = p1y[b]

            if self.f(x,y)>self.f(x1,y1):
                best = x1,y1
            else:
                best = x,y
            t1.append(best[0])
            t2.append(best[1])
        return t1,t2

    def elitist_replacement(self,px,py,p1x,p1y):
        A1 = px + p1x
        A2 = py + p1y
        n = len(px)
        m = len(A1)
        B = []
        for i in range(m):
            B.append(self.f(A1[i],A2[i]))

        # Way 1:
        # t = 0
        # for i in range(m):
        #     k = B[i]
        #     for j in range(i,m,1):
        #         if k>B[j]:
        #             k=B[j]
        #             t = j
        #     x = B[i]
        #     B[i] = B[t]
        #     B[t] = x
        #
        #     x = A1[i]
        #     A1[i] = A1[t]
        #     A1[t] = x
        #
        #     x = A2[i]
        #     A2[i] = A2[t]
        #     A2[t] = x

        # Way 2:
        # A1 = [x for _,x in sorted(zip(B,A1))]
        # A2 = [x for _,x in sorted(zip(B,A2))]

        # Way 3:
        A1_ = []
        A2_ = []
        for b,x,y in sorted(zip(B,A1,A2))[:n]:
            A1_.append(x)
            A2_.append(y)

        return A1_,A2_

    #------------------------------Selection--------------------
    def tournament_selection(self,px,py,s,k):
        Rx = []
        Ry = []
        n = len(px)
        for i in range(s):
            a = uniform(0,n)
            c = uniform(0,n)
            for j in range(k):
                b = uniform(0,n)
                d = uniform(0,n)
                x1 = px[a]
                x2 = px[b]

                y1 = py[c]
                y2 = py[d]
                if self.f(x1,y1)>self.f(x2,y2):
                    a = b
                    c = d
            Rx.append(px[a])
            Ry.append(py[c])
        return Rx,Ry

    def roulette_wheel(self,px,py,s):
        m = len(px)
        n = 0
        A = []
        B = []
        for i in range(m):
            n = n + self.f(px[i],py[i])      # 求和
        for i in range(m):
            A.append(self.f(px[i],py[i])/(n+self.eps))    # 求每个数占的比例
        for i in range(m):
            b = 0
            for j in range(i):
                b = b + A[j]    # 求和
            B.append(b)
        p1 = []
        p2 = []
        for n in range(s):
            r = uniform(0,1)
            for i in range(m):
                if r <= B[i]:
                    p1.append(px[i])
                    p2.append(py[i])
        return p1,p2

    #--------------------------Evolutionary_Programming----------
    def evolutionary_programming(self):
        err  = []
        px = rand_floats(self.low,self.high,self.p_size)
        py = rand_floats(self.low,self.high,self.p_size)
        n = len(px)

        for t in range (self.T):
            p1x = []
            p1y = []
            for i in range(n):
                p1x.append(self.mutation(px[i]))
                p1y.append(self.mutation(py[i]))
            # px,py = uniform_replacement(f,px,py,p1x,p1y)
            px,py = self.elitist_replacement(px,py,p1x,p1y)

            # Early stopping
            if abs(self.f(px[0],py[0])) < self.eps:
                break

        return px[0],py[0]

    #------------------------Evolution_Strategy------------------
    def evolution_strategy(self,u=15,k=25,lamda=40):
        err = []
        px = rand_floats(self.low,self.high,self.p_size)
        py = rand_floats(self.low,self.high,self.p_size)

        for t in range (self.T):
            p1x = []
            p1y = []

            p2x = []
            p2y = []
            Cx,Cy = self.roulette_wheel(px,py,u)
            n = len(Cx)

            for i in range(lamda):
                a = Cx[uniform(0,n)]
                b = Cx[uniform(0,n)]

                a1 = Cy[uniform(0,n)]
                b1 = Cy[uniform(0,n)]
                p1x.append(self.crossover(a,b))
                p1y.append(self.crossover(a1,b1))

            k = len(p1x)
            for i in range(k):
                p2x.append(self.mutation(p1x[i]))
                p2y.append(self.mutation(p1y[i]))

            px,py = self.elitist_replacement(px,py,p2x,p2y)

            # Early stopping
            if abs(self.f(px[0],py[0])) < self.eps:
                break

        return px[0],py[0]

    #------------------------Genetic_Algorithm ------------------
    def genetic_algorithm(self,s):
        err = []
        t = 0
        px = rand_floats(self.low,self.high,self.p_size)
        py = rand_floats(self.low,self.high,self.p_size)
        n = len(px)
        p3 = []
        while t < self.T:
            p1x,p1y = self.roulette_wheel(px,py,s)
            p2x,p2y = self.reproduction(p1x,p1y)
            px,py = self.elitist_replacement(px,py,p2x,p2y)
            t = t + 1
        for i in range(len(px)):
            p3.append(self.f(px[i],py[i]))
        index = np.argmin(p3)

        return px[index],py[index]
