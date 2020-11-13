import random
import math

class matrix():
    def __init__(self, d=None, dim=None, rand=False):
        if(dim != None):
            self.shape = dim
            self.data = [[random.uniform(-1,1) if rand else 0.0 for i in range(dim[1])] for i in range(dim[0])]
        else:
            self.data = d
            if(not isinstance(self.data[0], list)):
                self.shape = [1, len(self.data)]
            else:
                self.shape = [len(self.data), len(self.data[0])]

    def __str__(self):
        return "matrix("+str(self.data)+")"

    def __add__(self, other):
        if(isinstance(other, matrix)):
            return ewadd(self, other)
        else:
            return add(self, other)

    def __sub__(self, other):
        if(isinstance(other, matrix)):
            return ewadd(self, mul(other,-1))
        else:
            return add(self, other*-1)

    def __mul__(self, other):
        if(isinstance(other, matrix)):
            return ewmul(self, other)
        else:
            return mul(self, other)

def itr(x):
    dim = x.shape
    if(dim[0] == 1):
         for i in range(dim[1]):
            yield (0,i)
    else:
        for i in range(dim[0]):
            for j in range(dim[1]):
                yield(i,j)

def mapf(x, f):
    if(x.shape[0] == 1):
        return matrix([f(i) for i in x.data])
    else:
        z = matrix(dim=x.shape)
        for i,j in itr(x):
            z.data[i][j] = f(x.data[i][j])
        return z

def add(x, c):
    f = lambda x : x+c
    return mapf(x, f)

def mul(x, c):
    f = lambda x: x*c
    return mapf(x, f)

def ewadd(x, y):
    assert x.shape == y.shape
    if(x.shape[0] == 1):
        return matrix([x.data[i]+y.data[i] for i in range(x.shape[1])])
    z = matrix(dim=x.shape)
    for i, j in itr(x):
        z.data[i][j] = x.data[i][j] + y.data[i][j]
    return z

def ewmul(x, y):
    assert x.shape == y.shape
    z = matrix(dim=x.shape)
    for i, j in itr(x):
        z.data[i][j] = x.data[i][j] * y.data[i][j]
    return z

def dot(x, y):
    #vector dot product
    if(1 in x.shape and 1 in y.shape):
        a = flatten(x).data
        b = flatten(y).data
        return sum([i*j for (i,j) in zip(a, b)])
    #matrix multiplication
    else:
        dimx = x.shape
        dimy = y.shape
        prod = matrix(dim=[dimx[0],dimy[1]])
        assert dimx[1] == dimy[0]
        for i in range(dimx[0]):
            for j in range(dimy[1]):
                for k in range(dimx[1]):
                    prod.data[i][j] += x.data[i][k] * y.data[k][j]
        return prod

def transposed(x):
    dim = x.shape
    if(dim[0] == 1):
        z = matrix([[x.data[i]] for i in range(len(x.data))])
    else:
        z = matrix([[x.data[j][i] for j in range(len(x.data))] for i in range(len(x.data[0]))])
    z.shape = dim[::-1]
    return z

def relu(x):
    f = lambda x,i: max(0,x)
    return mapf(x,f)

def drelu(x):
    f = lambda x,i: int(x > 0)
    return mapf(x, f)

def flatten(x):
    flat = []
    if(x.shape[0] == 1):
        return x
    for i in range(len(x.data)):
        for j in range(len(x.data[i])):
            flat.append(x.data[i][j])
    return matrix(flat)

def tsum(x):
    sm=0
    for i,j in itr(x):
        sm += x.data[i][j]
    return sm
