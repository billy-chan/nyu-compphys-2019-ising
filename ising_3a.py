# Imports
from pylab import *
from numpy.random import *
import numpy as np
import matplotlib.pyplot as plt

# Set a seed so that we obtain reproducable results.
seed(5)

# Define the set of constants we are working with. 
N = 20
J = 1
T = 1
kb = 1
beta = 1 / (kb * T)
steps = 10000

# Setup an array that represents a fully magnetized system.
s = np.ones((N, N),int)

def energy(s):
	
	s1 = s[:-1,:]*s[1:,:]
	s2 = s[:,:-1]*s[:,1:]
	
	E = -J*(sum(s1) + sum(s2))
	
	return E

def energy_check(s):
	I = 0
	for i in range(N-1):
		for j in range(N):
			I+=s[i,j]*s[i+1,j]	
	for i in range(N):
		for j in range(N-1):
			I+=s[i,j]*s[i,j+1]
	return -J*I

# Setup lists to hold the magnetization value of system through each flip.
Eplot = []
Mplot = []
E1 = energy(s)
M = sum(s)

# Setup a loop to perform the flipping of dipoles.
for k in range(steps):
	i = randint(N)
	j = randint(N)
	s[i,j] *=-1
	E2 = energy(s)
	dE = E2 - E1
	
    # Metropolis check for accepting/declining a flip.
	if dE>0:
		if random()<exp(-beta*dE):
			E1 = E2
			M = sum(s)
		else:
			s[i,j]*=-1
	else:
		E1 = E2
		M = sum(s)
	Mplot.append(M / 400)

plot(Mplot)
