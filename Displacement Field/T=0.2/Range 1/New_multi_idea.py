import numpy as np
import matplotlib.pyplot as plt
import math as m
import time 
from scipy.optimize import minimize
from multiprocessing import Pool
import csv
import json

# function for obtaining k_x,k_y
def f(N):
    n = np.arange(0,N,1)
    k_x = np.zeros(len(n))
    
    for i in range (len(n)):
        k_x[i] = (2*np.pi*n[i])*(1./(N*a))
    
    k_y = [[0]*N for _ in range (N)]   
    for i in range (N):
        for j in range (N):
            k_y[i][j]= ((4*np.pi*n[j]/(N)) - (2*np.pi*n[i]/N))* (1./(np.sqrt(3)*a))
    return k_x, k_y

#(k_x[i],k_y[i][j]) are the coordinates in momentum space.
#The size of the grid is N X N = N^2 lattice points

def epsilon_P(X, Y, k_x, k_y, N, J_1):
    E_k = [[0]*N for _ in range (N)]    #E_k is a 1D matrix of size N 
    for i in range (N):
        for j in range (N):
            e1 = 0
            e2 = 0
            e1 = -8*X*J*(np.cos(k_x[i]*a)+np.cos((k_x[i]*a/2)+(k_y[i][j]*np.sqrt(3)*a/2))+np.cos((k_x[i]*a/2)-(k_y[i][j]*np.sqrt(3)*a/2)))
            e2 = -8*Y*J_1*(np.cos(k_y[i][j]*a*np.sqrt(3))+np.cos((3*k_x[i]*a/2)-(k_y[i][j]*np.sqrt(3)*a/2))+np.cos((-3*k_x[i]*a/2)-(k_y[i][j]*np.sqrt(3)*a/2)))
            E_k[i][j]= e1+e2
    return (E_k)

#Define fermi function

# for flavors p=3 and 4
def fermi_n(e_k, mu, d, T):
    k_b =1
    b = 1./(k_b*T)
    c = (e_k -mu-d)*b
    if (c>10):
        return (0)
    else:
        return (1./(m.exp((e_k-mu-d)*b)+1))

# for flavors 1 and 2
def fermi_p(e_k, mu, d, T):
    k_b =1
    b = 1./(k_b*T)
    c = (e_k -mu+d)*b
    if (c>10):
        return (0)
    else:
        return (1./(m.exp((e_k-mu+d)*b)+1))

# defining the consistent equations

def Seq_1(X, Y, mu, P, d, T, J_1):
    sol = 0
    E_k = epsilon_P(X, Y, k_x, k_y, N, J_1)
    for i in range(len(k_x)):
        for j in range(len(k_y)):
            sol = sol + 2*fermi_n(E_k[i][j], mu, d, T)+ 2*fermi_p(E_k[i][j], mu, d, T)
    return (sol/ (N*N))

def Seq_2(X, Y, mu, P, d, T, J_1):
    sol =0
    c = 3
    E_k = epsilon_P(X, Y, k_x, k_y, N, J_1)
    for i in range(len(k_x)):
        for j in range(len(k_x)):
            sol = sol + 2*(fermi_p(E_k[i][j], mu, d, T)+fermi_n(E_k[i][j], mu, d, T))*(np.cos(k_x[i]*a)+np.cos(k_x[i]*a*0.5 + k_y[i][j]*a*np.sqrt(3)*0.5)+np.cos(k_x[i]*a*0.5 - k_y[i][j]*a*np.sqrt(3)*0.5))
    return (sol/(4*N*N*c))

def Seq_3(X, Y, mu, P, d, T, J_1):
    sol =0
    c = 3
    E_k = epsilon_P(X, Y, k_x, k_y, N, J_1)
    for i in range(len(k_x)):
        for j in range(len(k_x)):
            sol = sol + 2*(fermi_p(E_k[i][j], mu, d, T)+fermi_n(E_k[i][j], mu, d, T))*(np.cos(k_y[i][j]*a*np.sqrt(3))+np.cos((3*k_x[i]*a/2)-(k_y[i][j]*np.sqrt(3)*a/2))+np.cos((-3*k_x[i]*a/2)-(k_y[i][j]*np.sqrt(3)*a/2)))
    
    return (sol/(4*N*N*c))

def Seq_4(X, Y, mu, P, d, T, J_1):
    sol = 0
    E_k = epsilon_P(X, Y, k_x, k_y, N, J_1)
    for i in range (len(k_x)):
        for j in range (len(k_x)):
            sol = sol + (fermi_p(E_k[i][j], mu, d, T)-fermi_n(E_k[i][j], mu, d, T))

    return (2*sol/(N*N))

# def average energy
def avg_eng(X, Y, mu, P, d, T, J_1):
    c = 3
    sol_1 = (16*J*c*(N**2)*(X**2))+(16*J_1*c*(N**2)*(Y**2))+ ((J+J_1)*c*(N**2)) +(mu*N*N)
    E_k = epsilon_P(X, Y, k_x, k_y, N, J_1)
    sol = 0
    for i in range (len(k_x)):
        for j in range (len(k_y)):
            sol = sol + 2*(fermi_p(E_k[i][j], mu, d, T)*(E_k[i][j]-mu+d)) + 2*(fermi_n(E_k[i][j], mu, d, T)*(E_k[i][j]-mu-d))
    return (sol_1 + sol)

# Defining the objective function
def objective(x, d, T, J_1):
    X  = x[0]
    Y  = x[1]
    mu = x[2]
    P  = x[3]
    return (avg_eng(X, Y, mu, P, d, T, J_1))

# Defining constraints 
def constraint1(x, d, T, J_1):
    X  = x[0]
    Y  = x[1]
    mu = x[2]
    P  = x[3]
    return ( Seq_1(X, Y, mu, P, d, T, J_1)- 1 )

def constraint2(x, d, T, J_1):
    X  = x[0]
    Y  = x[1]
    mu = x[2]
    P  = x[3]
    return ( Seq_2(X, Y, mu, P, d, T, J_1) - X )

def constraint3(x, d, T, J_1):
    X  = x[0]
    Y  = x[1]
    mu = x[2]
    P  = x[3]
    return ( Seq_3(X, Y, mu, P, d, T, J_1) - Y )

def constraint4(x, d, T, J_1):
    X  = x[0]
    Y  = x[1]
    mu = x[2]
    P  = x[3]
    return ( Seq_4(X, Y, mu, P, d, T, J_1) - P )


# Define optimization

def optimization(args):
    x_0, d, T, J_1 = args
    # Defining the bounds
    bonds = ((-1, 1),(-1, 1),(-2, 2),(-1,1))
    con1   = {'type': 'eq', 'fun': constraint1, 'args':(d, T, J_1)}
    con2   = {'type': 'eq', 'fun': constraint2, 'args':(d, T, J_1)}
    con3   = {'type': 'eq', 'fun': constraint3, 'args':(d, T, J_1)}
    con4   = {'type': 'eq', 'fun': constraint4, 'args':(d, T, J_1)}
    
    cons  = (con1, con2, con3, con4)
    
    initial_guess =x_0
    #performing optimization
    opt = minimize(objective, initial_guess, args = (d, T, J_1), method = 'SLSQP', bounds=bonds, constraints=cons)
    print("Done for", J_1)
    print("Done for", d)
    return([J_1, d, opt.x])

# Creating input values 

def Big_list(X_range, Y_range, mu_range, P_range, d_range, ratio, T):
    Grid = [([X, Y, mu, P], d, T, J_1) for J_1 in ratio for d in d_range for X in X_range for Y in Y_range for mu in mu_range for P in P_range]
    return Grid

# Optimizing for each value in Grid

def grid_optimize(Grid):
    
    def collect_result(result):
        results.append(result)

    with Pool(processes=256) as pool:
        input_values = Grid
        results = []
        for value in input_values:
            pool.apply_async(optimization, args=(value,), callback=collect_result) 
        pool.close()
        pool.join ()
    return (results)

def convert_to_list(nested_data):
    if isinstance(nested_data, np.ndarray):
        return nested_data.tolist()
    elif isinstance(nested_data, list):
        return [convert_to_list(item) for item in nested_data]
    else:
        return nested_data

def minima_finder(results, T):
    
    E = []
    ans = []
    minima_Energy = []
    
    results = convert_to_list(results)
    
    with open('all_results.json', 'w') as file:
        json.dump(results, file, indent=4)
        
    ratio = list({sub[0] for sub in results})
    delta = list({sub[1] for sub in results})
    
    for i in ratio:
        for j in delta :
            J_1  = i
            d    = j
            avg_energy = []
            dum_sol = []
            minima_E = 0
            error = 0.01
            for sol in results:
                if (sol[0] == J_1 and sol[1] == d):
                    if ((np.abs(constraint1(sol[2], d, T, J_1))<error) and (np.abs(constraint2(sol[2], d, T, J_1))<error) and (np.abs(constraint3(sol[2], d, T, J_1))<error)and(np.abs(constraint4(sol[2], d, T, J_1))<error)):
                        avg_energy.append(avg_eng(sol[2][0],  sol[2][1], sol[2][2], sol[2][3], d, T, J_1))
                        dum_sol.append(sol)
            minima_E = min(avg_energy)
            for k in range(len(avg_energy)):
                index = 0
                if (avg_energy[k]==minima_E):
                    index = k
                    ans.append(dum_sol[index])
                    #print("Done!")
                    #print("The sol is", dum_sol[index])
                    #print("The min E is", minima_E)
                    #print("\n")
                    
            E.append([J_1, d, avg_energy])
            minima_Energy.append([J_1, d ,minima_E])
    return(E, ans, minima_Energy)           



        

a = 1
J = 1
N = 100
T = 0.2
k_x, k_y = f(N)

start = time.time()

Grid = Big_list(np.linspace(-1,1,4),np.linspace(-1,1,4),np.linspace(-2,2,4),np.linspace(-1,1,4), [0, 0.25, 0.5, 0.75,1], [0, 0.075, 0.1, 0.125], T)

print("Started Opti")

if __name__ == "__main__":
    sol  = grid_optimize(Grid)
    
end = time.time()
print('Time taken is', end-start)

sorted_data = sorted(sol, key=lambda x: x[0])
sorted_data = sorted(sorted_data, key=lambda x:x[1])

E, ans, minima_Energy = minima_finder(sorted_data, T)

E = convert_to_list(E)
ans = convert_to_list(ans)
minima_Energy = convert_to_list(minima_Energy)

with open('meta_E.json', 'w') as file:
    json.dump(E, file, indent=4)

with open('sol.json', 'w') as file:
    json.dump(ans, file, indent=4)

with open('minima_E.json', 'w') as file:
    json.dump(minima_Energy, file, indent=4)
    
    
