import numpy as np
import matplotlib.pyplot as plt
import math as m
from scipy.optimize import minimize
from multiprocessing import Pool  
import time
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

def epsilon_NNN(X, Y, k_x, k_y, N, J_1):
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
def fermi(e_k, mu, T):
    k_b =1
    b = 1./(k_b*T)
    d = (e_k -mu)*b
    if (d>10):
        return (0)
    else:
        return (1./(m.exp((e_k-mu)*b)+1))

# Define consistent eq 1
def Seq_1(X, Y, mu, T, J_1):
    sum = 0
    E_k = epsilon_NNN(X, Y, k_x, k_y, N, J_1)
    for i in range (len(k_x)):
        for j in range (len(k_x)):
            sum = sum + fermi (E_k[i][j], mu, T)
    return (sum/(N**2))

#Define consistent eq 2
def Seq_2(X, Y, mu, T, J_1):
    sum =0
    c = 3
    E_k = epsilon_NNN(X, Y, k_x, k_y, N, J_1)
    for i in range(len(k_x)):
        for j in range(len(k_x)):
            sum = sum + (fermi(E_k[i][j], mu, T)*(np.cos(k_x[i]*a)+np.cos(k_x[i]*a*0.5 + k_y[i][j]*a*np.sqrt(3)*0.5)+np.cos(k_x[i]*a*0.5 - k_y[i][j]*a*np.sqrt(3)*0.5)))
    return (sum/(N*N*c))

#Define consistent eq 3
def Seq_3(X, Y, mu, T, J_1):
    sum =0
    c = 3
    E_k = epsilon_NNN(X, Y, k_x, k_y, N, J_1)
    for i in range(len(k_x)):
        for j in range(len(k_x)):
            sum = sum + fermi(E_k[i][j], mu, T)*(np.cos(k_y[i][j]*a*np.sqrt(3))+np.cos((3*k_x[i]*a/2)-(k_y[i][j]*np.sqrt(3)*a/2))+np.cos((-3*k_x[i]*a/2)-(k_y[i][j]*np.sqrt(3)*a/2)))
    
    return (sum/(N*N*c))

# Average energy of the system 

def avg_eng(X, Y, mu, T, J_1):
    c = 3
    sol_1 = (16*J*c*(N**2)*(X**2))+(16*J_1*c*(N**2)*(Y**2))+(mu*(N**2)+ (J+J_1)*c*(N**2))
    E_k = epsilon_NNN(X, Y, k_x, k_y, N, J_1)
    sol = 0
    for i in range (len(k_x)):
        for j in range (len(k_y)):
            sol = sol + (fermi(E_k[i][j], mu, T)*(E_k[i][j]-mu))
    return ((4*sol)+ sol_1)   

# Defining the objective function
def objective(x, T, J_1):
    X  = x[0]
    Y  = x[1]
    mu = x[2]
    ans = avg_eng(X,Y,mu,T,J_1)
    return (ans)

# Defining constraints 
def constraint1(x, T, J_1):
    X  = x[0]
    Y  = x[1]
    mu = x[2]
    return ( Seq_1(X, Y, mu, T, J_1)-0.25 )

def constraint2(x, T, J_1):
    X  = x[0]
    Y  = x[1]
    mu = x[2]
    return ( Seq_2(X, Y, mu, T, J_1)-X )

def constraint3(x, T, J_1):
    X  = x[0]
    Y  = x[1]
    mu = x[2]
    return ( Seq_3(X, Y, mu, T, J_1)-Y )

# Define optimization

def optimization(args):
    x_0, T, J_1 = args
    # Defining the bounds
    bonds = ((-1, 1),(-1, 1),(-2, 2))
    con1   = {'type': 'eq', 'fun': constraint1, 'args':(T, J_1)}
    con2   = {'type': 'eq', 'fun': constraint2, 'args':(T, J_1)}
    con3   = {'type': 'eq', 'fun': constraint3, 'args':(T, J_1)}

    cons  = (con1, con2, con3)
    
    initial_guess =x_0
    #performing optimization
    opt = minimize(objective, initial_guess, args = (T,J_1), method = 'SLSQP', bounds=bonds, constraints=cons)
    return(opt.x)

# creating the grid
def grid(X_range, Y_range, mu_range):
    Grid = []
    for i in range (len(X_range)):
        for j in range (len(Y_range)):
            for k in range (len(mu_range)):
                Grid.append([X_range[i], Y_range[j], mu_range[k]])

    return Grid

#finding the minima
def grid_optimize(args):
    T, J_1, Grid = args
    avg_energy = []
    with Pool(processes=252) as pool:
        input_values = [([Grid[i][0], Grid[i][1], Grid[i][2]], T, J_1) for i in range (len(Grid))]
        results = pool.map(optimization, input_values)
        pool.close()
        pool.join ()
    return (results)

# Finding the global minima in the range
def minima(args):
    T, J_1, results = args
    avg_energy = []
    for i in range (len(results)):
        avg_energy.append(avg_eng(results[i][0], results[i][1], results[i][2], T, J_1))
    for i in range (len(avg_energy)):
        if (avg_energy[i]==min(avg_energy)):
            print(" Done computing for T=", T)
            print(" The minimum energy is ", avg_energy[i])
            print(" The solution is ", results[i])
            print('\n')
            E_avg = avg_energy[i]
            sol   = results[i]
    G_E.append(avg_energy)
    ans.append(convert_to_list(results))
    return ([T, E_avg, sol])

# obtaining data for vs ratio plots at T
def data(args):
    J_1, temp, Grid = args
    final_arr  = []
    for i in range (len(temp)):
        T = temp[i]
        res = grid_optimize((T, J_1, Grid))
        A   = minima((T, J_1, res))
        final_arr.append(A)
    return (final_arr)

def convert_to_list(nested_data):
    if isinstance(nested_data, np.ndarray):
        return nested_data.tolist()
    elif isinstance(nested_data, list):
        return [convert_to_list(item) for item in nested_data]
    else:
        return nested_data

a = 1
N = 100
k_x, k_y = f(N)
J = 1
J_1 = 0.125

Grid = grid(np.linspace(-1, 1, 6), np.linspace(-1, 1, 6),np.linspace(-2, 2, 7))

A  = np.linspace(0.1, 0.6, 10)
B  = np.linspace(0.61, 0.9, 10)
C  = np.linspace(0.91, 1.5, 10)
temperature = np.concatenate((A,B,C))
G_E  = []
ans = []
start = time.time()
final = data((J_1, temperature, Grid))


csv_file_name = 'Tvs_meta_128.csv'
with open(csv_file_name, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(final)

np.savetxt('meta_avg_eng.txt',np.array(G_E))

json_file_name = 'meta_avg_eng.json'
with open(json_file_name, 'w') as file:
    json.dump(G_E, file, indent=4)

json_file_name = 'meta_solution.json'
with open(json_file_name, 'w') as file:
    json.dump(ans, file, indent=4)

end = time.time()
print("Time taken is", end-start)
