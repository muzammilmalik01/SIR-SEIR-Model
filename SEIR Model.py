#import numpy as np
import matplotlib.pyplot as plt

class SEIR:
    def __init__(self, S, E, I, R, beta, alpha, sigma, gamma, N):
        self.S = S
        self.E = E
        self.I = I
        self.R = R
        self.beta = beta 
        self.alpha = alpha 
        self.sigma = sigma 
        self.gamma = gamma
        self.N = N

    def dSdt(self):
        return -self.beta * (self.S * self.I) / self.N + self.gamma*R

    def dEdt(self):
        return self.beta * (self.S * self.I) / self.N - (self.alpha * self.E)

    def dIdt(self):
        return self.alpha * self.E - self.sigma * self.I

    def dRdt(self):
        return self.sigma * self.I - self.gamma*R

    def runge_kutta(self, h):
        # Calculate the intermediate values k1, k2, k3, k4, l1, l2, l3, l4, m1, m2, m3, m4
        k1 = h * self.dSdt()
        l1 = h * self.dEdt()
        m1 = h * self.dIdt()
        n1 = h * self.dRdt()
        k2 = h * self.dSdt() + k1/2
        l2 = h * self.dEdt() + l1/2
        m2 = h * self.dIdt() + m1/2
        n2 = h * self.dRdt() + n1/2
        k3 = h * self.dSdt() + k2/2
        l3 = h * self.dEdt() + l2/2
        m3 = h * self.dIdt() + m2/2
        n3 = h * self.dRdt() + n2/2
        k4 = h * self.dSdt() + k3
        l4 = h * self.dEdt() + l3
        m4 = h * self.dIdt() + m3
        n4 = h * self.dRdt() + n3
        # Calculate the updated values of S, I, E, and R
        self.S += (1/6) * (k1 + 2*k2 + 2*k3 + k4)
        self.E += (1/6) * (l1 + 2*l2 + 2*l3 + l4)
        self.I += (1/6) * (m1 + 2*m2 + 2*m3 + m4)
        self.R += (1/6) * (n1 + 2*n2 + 2*n3 + n4)

    def run_simulation(self, num_iterations, h):
        # Initialize lists to store values for plotting
        S_values = []
        E_values = []
        I_values = []
        R_values = []
        t_values = []
        total_pop_values = []
        t = 0
        # Run the simulation for the given number of iterations
        for i in range(num_iterations):
            # Use Runge-Kutta method to update values
            self.runge_kutta(h)
            # Append current values to lists for plotting
            S_values.append(self.S)
            E_values.append(self.E)
            I_values.append(self.I)
            R_values.append(self.R)
            t += h
            t_values.append(t)
            total_pop_values.append(self.S + self.E + self.I + self.R)
            
    # Plot the results
        plt.plot(t_values, S_values, label='S')
        plt.plot(t_values, E_values, label='E')
        plt.plot(t_values, I_values, label='I')
        plt.plot(t_values, R_values, label='R')
        plt.plot(t_values, total_pop_values, label='Total population')
        plt.xlabel('Time (days)')
        plt.ylabel('Number of people')
        plt.legend()
        plt.show()

# Set the initial values for S, E, I, R
S = 500000
E = 50
I = 5
R = 0
num_iterations = 90
h = 1

# Set the values for beta, alpha, sigma, gamma.
beta = 0.5 #S to E
alpha = 0.1 #E to I   
sigma = 0.1 #I to R
gamma = 0.1 #R to S

# Set the total population
N = S+I+E+R

# Create an instance of the SEIR class
model = SEIR(S,E,I,R,beta,alpha,sigma,gamma,N)

# Run the simulation for 10 time steps with a time step size of 0.1
model.run_simulation(num_iterations, h)