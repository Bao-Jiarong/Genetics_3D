'''
 *  Author : Bao Jiarong
 *  Contact: bao.salirong@gmail.com
 *  Created  On: 2020-05-29
 *  Modified On: 2020-05-29
 '''

import math
import sys
import numpy as np
import src.ev as envo
#------------------- Test functions for optimization ---------------
# link:
# https://en.wikipedia.org/wiki/Test_functions_for_optimization

#-------------------------------------------------------------------
# Global minimum : f(0,0)=0
def ackley(x,y):
    a = -20*np.exp(-0.2*math.sqrt(0.5*(x**2+y**2)))\
        -np.exp(0.5*(np.cos(2*np.pi*x)+(np.cos(2*np.pi*y))))+np.e+20
    return a

#-------------------------------------------------------------------
# Global minimum : f(3,0.5)=0
def beale(x,y):
    a = ((1.5-x+x*y)**2)+(2.25-x+x*(y**2))**2+(2.625-x+x*(y**3))**2
    return a

#-------------------------------------------------------------------
# Global minimum : f(0,-1)=3
def goldstein_price(x,y):
    a = (1+((x+y+1)**2)*(19-14*x+3*(x**2)-14*y+6*x*y+3*(y**2)))\
        *(30+((2*x-3*y)**2)*(18-32*x+12*(x**2)+48*y-36*x*y+27*(y**2)))
    return a

#-------------------------------------------------------------------
# Global minimum : f(1,3)=0
def booth(x,y):
    a = ((x+2*y-7)**2)+(2*x+y-5)**2
    return a

#-------------------------------------------------------------------
# Global minimum : f(-10,1)=0
def bukin(x,y):
    a = 100*math.sqrt(abs(y-0.01*(x**2)))+0.01*abs(x+10)
    return a

#-------------------------------------------------------------------
# Global minimum : f(0,0)=0
def matyas(x,y):
    a = 0.26*(x**2+y**2)-0.48*x*y
    return a

#-------------------------------------------------------------------
# Global minimum :f(1,1)=0
def levi(x,y):
    a = np.sin(3*np.pi*x)**2+((x-1)**2)*(1+np.sin(3*np.pi*y)**2)\
        +((y-1)**2)*(1+np.sin(2*np.pi*y)**2)
    return a

#-------------------------------------------------------------------
# Global minimum : f(3,2) = 0;
#                  f(-2.805118,3.131312) = 0;
#                  f(-3.779310,-3.283186)= 0;
#                  f(3.584428,-1.848126) = 0
def himmelblau(x,y):
    a = (x**2+y-11)**2+(x+y**2-7)**2
    return a

#-------------------------------------------------------------------
# Global minimum : f(0,0)=0
def three_hump_camel(x,y):
    a = 2*x**2-1.05*x**4+((x**6)/6)+x*y+y**2
    return a

#-------------------------------------------------------------------
# Global minimum : f(pi ,pi )=-1
def easom(x,y):
    a = -np.cos(x)*np.cos(y)*np.exp(-((x-np.pi)**2+(y-np.pi)**2))
    return a

#-------------------------------------------------------------------
# Global minimum : f(1.34941,-1.34941) = -2.06261;
#                  f(1.34941,1.34941)  = -2.06261;
#                  f(-1.34941,1.34941) = -2.06261;
#                  f(-1.34941,-1.34941)= -2.06261
def cross_in_tray(x,y):
    a = -0.0001*(abs((np.sin(x)*np.sin(y)*np.exp(abs(100-(math.sqrt((x**2+y**2)/np.pi))))))+1)**0.1
    return a

#-------------------------------------------------------------------
# Global minimum : f(512,404.2319)=-959.6407
def eggholder(x,y):
    a = -(y+47)*np.sin(math.sqrt(abs(x/2+(y+47))))-x*np.sin(math.sqrt(abs(x-(y+47))))
    return a

#-------------------------------------------------------------------
# Global minimum : f( 8.05502, 9.66459) = -19.2085;
#                  f(-8.05502, 9.66459) = -19.2085;
#                  f( 8.05502,-9.66459) = -19.2085;
#                  f(-8.05502,-9.66459) = -19.2085
def holder_table(x,y):
    a = -abs(np.sin(x)*np.cos(y)*np.exp(abs(1-(math.sqrt(x**2+y**2))/np.pi)))
    return a

#-------------------------------------------------------------------
# Global minimum : f(-0.54719,-1.54719)=-1.9133
def mccormick(x,y):
    a = np.sin(x+y)+(x-y)**2-1.5*x+2.5*y+1
    return a

#-------------------------------------------------------------------
# Global minimum : f(0,0)=0
def schaffer_n2(x,y):
    a = 0.5 + (np.sin(x**2-y**2)**2-0.5)/(1+0.001*(x**2+y**2))**2
    return a

#-------------------------------------------------------------------
# Global minimum : f(0, 1.25313)=0.292579;
#                  f(0,-1.25313)=0.292579
def schaffer_n4(x,y):
    a = 0.5 + (np.cos(np.sin(abs(x**2-y**2)))**2-0.5)/(1+0.001*(x**2+y**2))**2
    return a

#------------------------------------------------------------------
func = {
        "ackley"          : ackley,
        "beale"           : beale,
        "goldstein_price" : goldstein_price,
        "booth"           : booth,
        "bukin"           : bukin,
        "matyas"          : matyas,
        "levi"            : levi,
        "himmelblau"      : himmelblau,
        "three_hump_camel": three_hump_camel,
        "easom"           : easom,
        "cross_in_tray"   : cross_in_tray,
        "eggholder"       : eggholder,
        "holder_table"    : holder_table,
        "mccormick"       : mccormick,
        "schaffer_n2"     : schaffer_n2,
        "schaffer_n4"     : schaffer_n4}

name = sys.argv[1]
f=func[name]

#-----------  Evolutionary Algorithms ------------------------------
ev = envo.EV(f     = ackley,    # the function to be minimized
             T     = 100,       # maximum number of iterations
             low   =-100,       # search domain
             high  = 100,       # search domain
             p_size= 50,        # population size
             eps   = 1e-3)

print("Evolutionary Programming")
x,y = ev.evolutionary_programming()
print("x =",round(x,5),"y =",round(y,5),"f(x,y) =",round(f(x,y),5))

print("Evolution Strategy")
x,y = ev.evolution_strategy(u     = 15,
                            k     = 25,
                            lamda = 40)
print("x =",round(x,5),"y =",round(y,5),"f(x,y) =",round(f(x,y),5))

print("Genetic Algorithm")
x,y = ev.genetic_algorithm(s=10)
print("x =",round(x,5),"y =",round(y,5),"f(x,y) =",round(f(x,y),5))
