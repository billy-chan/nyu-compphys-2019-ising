import numpy as np 
import matplotlib.pyplot as plt

class Ising2D:
    def __init__(self,system,J,N,T,H,steps):
        #initialize the system variables 
        
        self.system = system      # N x N matrix of dipoles with spin up (1) or spin down (-1)
        self.J      = J 
        self.N      = N           # Number of lattice points (dipoles) in the system
        self.T      = T           # Temperature of system
        self.H      = H           # External magnetic field
        self.steps  = steps       # Number of monte carlo steps 
        #self.beta   = 1/T
        
        #initialize system parameters
        
        self.E_tot  = self.getTotalEnergy()
        self.M      = np.sum(self.system.astype(float))/(self.N**2)
        
    def getNeighbors(self,i,j):
        neighborhood =  self.system[(i+1)%self.N, j] + self.system[i,(j+1)%self.N] + self.system[(i-1)%self.N, j] + self.system[i,(j-1)%self.N]
        return neighborhood
        
    def getEnergy(self,i,j):
        s = self.system[i,j]
        neighborhood = self.getNeighbors(i,j)
        energy = ((-s * neighborhood) - (self.H * s))
        return energy
        
    def getTotalEnergy(self):
        energy = 0
        for i in range(self.N):
            for j in range(self.N):
                energy += self.getEnergy(i,j)
        return energy
    
    def getMagnetization(self):
        self.M = np.sum(self.system.astype(float))/(self.N**2)
    
    def metropolisStep(self,i,j):
        #calculate the contribution to energy at the specified lattice point before flipping spin
        #print(self.system)
        #print(self.E_tot)
        beta = 1/self.T
        E_i = self.getEnergy(i,j)
        #print('initial energy:', E_i)
        #flip the spin at position i,j
        self.system[i,j] *= -1
        #calculate the contribution to energy at specified lattice point after flipping spin
        E_f = self.getEnergy(i,j)
        #print('final energy:', E_f)
        dE = E_f - E_i
        #if the change in energy is negative, update the system energy and keep flipped spin. Do the same if Boltzman condition is satisfied. If not flip the spin back and do not update energy
        if dE < 0:
            #print('flip accepted')
            self.E_tot += dE
        elif np.random.rand() < np.exp(-beta*dE):
            #print('flip accepted by chance')
            self.E_tot += dE
        else:
            #print('flip rejected')
            self.system[i,j] *= -1
        self.getMagnetization()
            
    def run(self):
        for l in range(self.steps):
            i = np.random.randint(self.N)
            j = np.random.randint(self.N)
            self.metropolisStep(i,j)
    
    def test(self):
        M = []
        for l in range(self.steps):
            i = np.random.randint(self.N)
            j = np.random.randint(self.N)
            self.metropolisStep(i,j)
            M.append(self.M)
        return M
            
    def changingTemp(self,temps):
        E = []
        M = []
        for i in np.nditer(temps):
            self.T = i
            self.run()
            E.append(self.E_tot)
            M.append(self.M)
        return E,M
        
        
        
        
N = 25
system_aligned = np.ones((N,N), dtype=int)
system_random = 2*np.random.randint(2, size = (N,N)) - 1
T = 10
J = 1
H = -2
steps = 5000
temps = np.linspace(0.0001,50,100)

Ising = Ising2D(system_random,J,N,T,H,steps)
#Ising.run()
#M = Ising.test()
#plt.plot(M)
#plt.show()

E,M = Ising.changingTemp(temps)


#plt.scatter(temps,E, color = 'r')
plt.scatter(temps,M, color = 'b')
plt.show()
        