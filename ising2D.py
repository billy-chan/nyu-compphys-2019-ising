"""
# Coders: Chan, Mishra, Tahir, Tsega
# File: ising2D.py
# Description: Setup a class which will host the primary functions of the program.
# Date: 2019/12/05
"""

# Imports and packages.
import numpy as np 

class Ising2D:
    
    # Create and initialize the system parameters.
    def __init__(self,system,J,N,T,H,steps):     
        if system == "aligned":
            self.system = np.ones((N,N), dtype=int)                     # NxN matrix of dipoles all with spin up.
        elif system == "random":
            self.system = (2*np.random.randint(2, size = (N,N)) - 1)    # NxN matrix of dipoles with either spin up or down.
        else:
            print("Invalid system argument: input 'aligned' or 'random'")
            
        self.J      = J         # Exchange energy.
        self.N      = N         # Lattice points or dipoles.
        self.T      = T         # Temperature of the system.
        self.H      = H         # External magnetic field.
        self.steps  = steps     # Number of monte carlo steps.
        
        self.E_tot  = self.getTotalEnergy()
        self.M      = np.sum(self.system.astype(float))/(self.N**2)

    # Obtain the spins of neighboring dipoles.        
    def getNeighbors(self,i,j):
        neighborhood =  self.system[(i+1)%self.N, j] + self.system[i,(j+1)%self.N] + self.system[(i-1)%self.N, j] + self.system[i,(j-1)%self.N]
        return neighborhood

    # Obtain the energy of a given dipole and its neighbors, for a given state.    
    def getEnergy(self,i,j):
        s = self.system[i,j]
        neighborhood = self.getNeighbors(i,j)
        energy = ((-s * neighborhood) - (self.H * s))
        return energy

    # Obtain the energy of the system.        
    def getTotalEnergy(self):
        energy = 0
        for i in range(self.N):
            for j in range(self.N):
                energy += self.getEnergy(i,j)
        return energy/2
        
    # Obtain the magnetization of the system.        
    def getMagnetization(self):
        self.M = np.sum(self.system.astype(float))/(self.N**2)
    
    # Performs the flipping and checking via the Metropolis algorithim. This is no different 
    # than what we have already seen from Newman and other examples of the classic algorithim.
    def metropolisStep(self,i,j):
        
        # Calculate the contribution to the energy at the specified lattice point.
        beta = 1/self.T
        E_i = self.getEnergy(i,j)
        
        # Flips the dipole at a given position.
        self.system[i,j] *= -1
        
        # Calculate the contribution to the energy again, and check the difference post-flip.
        E_f = self.getEnergy(i,j)
        dE = E_f - E_i
        
        # If the change in energy is negative, accept the flip.
        if dE < 0:
            self.E_tot += dE
        
        # If the Boltzmann condition is satisfied, accept the flip.
        elif np.random.rand() < np.exp(-beta*dE):
            self.E_tot += dE
            
        # Reject the flip otherwise.    
        else:
            self.system[i,j] *= -1
        self.getMagnetization()
        
    # Picks an arbitrary dipole within the lattice.            
    def run(self):
        for l in range(self.steps):
            i = np.random.randint(self.N)
            j = np.random.randint(self.N)
            self.metropolisStep(i,j)
    
    # Define the temperature-space, since temperature is our independent variable in every scenario.
    def changingTemp(self,temps):
        a1 = np.linspace(0.0001, 2, 20)
        a2 = np.linspace(2, 3, 20)
        a3 = np.linspace(3,10,100)
        temp_array = np.concatenate((a1,a2,a3))
        
        if temps == "cool":
            temps = np.flip(temp_array)
        elif temps == "heat":
            temps = temp_array
        else:
            print("Error: invalid temps argument")
            return
        
        E = []
        M = []
        for i in np.nditer(temps, order='C'):
            self.T = i
            self.run()
            E.append(self.E_tot)
            M.append(self.M)
        return E, M, temps


