import numpy as np


def phi_1(x):
    return 0.9*int(x<=1/np.sqrt(2))+0.1*int(x<=1)

def phi_2(x):
    return int(x<=1)

def phi_3(x):
    return int(x<=1)-0.5*int(x<=1/np.sqrt(2))

def phi_4(x):
    return int(x<=1)-0.9*int(x<=1/np.sqrt(2))

def phi_5(x):
    return int(x<=1)*(1-x)**3

def phi_6(x):
    return int(x<=1)*(1-x)**6