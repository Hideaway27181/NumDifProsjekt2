import numpy as np

#shape functions(lagrange basis functions) Psi_i on Reference element
def Psi_0(x):
    return 2*(x-1/2)*(x-1)

def Psi_1(x):
    return 4*x*(1-x) 

def Psi_2(x):
    return 2*x*(x-1/2)

def make_partition(N):
    x = np.linspace(0,1,N+1)
    h = x[1]-x[0]

    return x,h 

def map_to_element_k(x_k,h,tau):
    '''
    map from reference element to K_k
    input: 
    - x (vector of xi's)
    - k (element nr k)
    - h (stepsize)
    -tau (element in [0,1])    
    '''
    return x_k + h*tau

def index_mapping(k, alpha):
    '''
    maps from local to global index
    input: 
    - k: element
    - alpha: local index
    '''
    return 2*k + alpha

def elemental_A(h):
    '''
    make the Elemental matrix A_{K_k}, for a given stepsize h
    input: 
    -h: elementsize
    
    '''
    A = 1/(3*h)*np.array([[7, -8,1], [-8,16,-8],[1,-8,7]])
    return A

import numpy as np

def elemental_load_vec(f, h, x_k):
    """
    Computes the elemental load vector b_{K_k} using Simpson's Rule.
    
    Input:
    - f: The source function f(x).
    - h: elementsize
    - x_k: left endpoint of K_k
    
    Returns:
    - b_local: numpy array (3,)
        The computed elemental load vector.
    """
    # Quadrature points and weights for Simpson's Rule
    xi = np.array([0, 0.5, 1])  # Reference element quadrature points
    weights = np.array([1/6, 4/6, 1/6])  # Simpson's rule weights
    
    # Quadratic shape functions at quadrature points
    psi_funcs = [
        Psi_0, Psi_1, Psi_2]
    
    # Compute b_local using quadrature rule
    b_local = np.zeros(3)
    for i in range(3):  # Iterate over shape functions
        for j in range(3):  # Iterate over quadrature points
            x_mapped = map_to_element_k(x_k,h,xi[j])
            b_local[i] += weights[j] * f(x_mapped) * psi_funcs[i](xi[j])
    
    return h * b_local




# Example usage: Compute for f(x) = sin(pi*x) over an element [0, 0.1]
f = lambda x: 1
h_example = 0.1
x_k_example = 0  # Left endpoint of the element

# Compute the elemental matrix and load vector
A_k = elemental_A(h_example)
b_k = elemental_load_vec(f, h_example, x_k_example)

# Print results
print("Elemental Stiffness Matrix A_{K_k}:\n", A_k)
print("\nElemental Load Vector b_{K_k}:\n", b_k)