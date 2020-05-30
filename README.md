## Genetics 3D in Python
The implemented genetics 3D are:

* Evolutionary Programming (EP),
* Evolution Strategy (ES),
* Genetic Algorithm (GA).

All of the implemented algorithms can be used to find the minimum of 3D function.  
For example : f(x,y) = x^2+y^2-4

### Requirement
```
python==3.7.0
numpy==1.18.1
```
### How to use

Open test.py you will find some examples
```
import src.ev as envo

def ackley(x,y):
    a = -20*np.exp(-0.2*math.sqrt(0.5*(x**2+y**2)))\
        -np.exp(0.5*(np.cos(2*np.pi*x)+(np.cos(2*np.pi*y))))+np.e+20
    return a

ev = envo.EV(f     = ackley,
             T     = 100,
             low   = -100,
             high  = 100,
             p_size= 50,
             eps   = 1e-3)

x,y = ev.evolutionary_programming()
print("x =",round(x,5),"y =",round(y,5),"f(x) =",round(ackley(x,y),5))
```
