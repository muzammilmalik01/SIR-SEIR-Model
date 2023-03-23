import numpy as np
import matplotlib.pyplot as plt

# Define the ODEs for the SIR model
def dSdt(S, I, R, beta, N):
    return -beta * S * I / N

def dIdt(S, I, R, beta, sigma, N):
    return beta * S * I / N - sigma * I

def dRdt(I, nu):
    return nu * I

# Define the Runge-Kutta function
def runge_kutta(S, I, R, h, beta, nu, N):
    # Calculate the intermediate values k1, k2, k3, k4, l1, l2, l3, l4, m1, m2, m3, m4
    k1 = h * dSdt(S, I, R, beta, N)
    l1 = h * dIdt(S, I, R, beta, nu, N)
    m1 = h * dRdt(I, nu)
    k2 = h * dSdt(S + k1/2, I + l1/2, R + m1/2, beta, N)
    l2 = h * dIdt(S + k1/2, I + l1/2, R + m1/2, beta, nu, N)
    m2 = h * dRdt(I + l1/2, nu)
    k3 = h * dSdt(S + k2/2, I + l2/2, R + m2/2, beta, N)
    l3 = h * dIdt(S + k2/2, I + l2/2, R + m2/2, beta, nu, N)
    m3 = h * dRdt(I + l2/2, nu)
    k4 = h * dSdt(S + k3, I + l3, R + m3, beta, N)
    l4 = h * dIdt(S + k3, I + l3, R + m3, beta, nu, N)
    m4 = h * dRdt(I + l3, nu)

    # Calculate the updated values of S, I, and R
    S_new = S + (1/6) * (k1 + 2*k2 + 2*k3 + k4)
    I_new = I + (1/6) * (l1 + 2*l2 + 2*l3 + l4)
    R_new = R + (1/6) * (m1 + 2*m2 + 2*m3 + m4)

    return S_new, I_new, R_new

# Define the main function
def main():
    # Set the initial values of S, I, and R
    S = 999
    I = 1
    R = 0

    # Set the parameters for the model
    beta = 0.5 #S to I
    sigma = 1/14 #I to R
    N = S + I + R
    h = 0.1

    # Set the number of iterations
    iterations = 1000

    # Create lists to store the values of S, I, and R at each iteration
    S_values = []
    I_values = []
    R_values = []

    # Run the Runge-Kutta method for the specified number of iterations
    for i in range(iterations):
        S, I, R = runge_kutta(S, I, R, h, beta, sigma, N)
        S_values.append(S)
        I_values.append(I)
        R_values.append(R)

    # Print the final values of S, I, and R
    print("S:", S)
    print("I:", I)
    print("R:", R)

    # Create a range of time values for the x-axis
    t = np.arange(0, iterations * h, h)

    # Plot the values of S, I, and R on the same graph
    plt.plot(t, S_values, label="S")
    plt.plot(t, I_values, label="I")
    plt.plot(t, R_values, label="R")

    # Add a legend and labels to the plot
    plt.legend()
    plt.xlabel("Time (days)")
    plt.ylabel("Population")

    # Show the plot
    plt.show()
    
# Calling the Main function
main()