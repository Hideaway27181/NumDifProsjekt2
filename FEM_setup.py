import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sci

#shape functions(lagrange basis functions) Psi_i on Reference element
def Psi_0(x):
    return 2*(x-1/2)*(x-1)

def Psi_1(x):
    return 4*x*(1-x) 

def Psi_2(x):
    return 2*x*(x-1/2)

def make_partition(N):
    ''' 
    function to make mesh partition
    input: 
    N: number of elements

    reurns: 
    x: array of nodes: [0,h,2h,...,1] on the elements
    h: stepsize
    '''
    x = np.linspace(0,1,2*N+1)
    h = x[1]-x[0] #1/2N

    return x,2*h 

def map_to_element_k(x_k,h,tau):
    '''
    map from reference element to K_k
    input: 
    - x (vector of xi's)
    - k (element nr k)
    - h (stepsize)
    -tau (point in [0,1])    
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
    make the Elemental stiffness matrix A_{K_k}, for a given stepsize h
    input: 
    -h: elementsize
    
    '''
    A = 1/(3*h)*np.array([[7, -8,1], [-8,16,-8],[1,-8,7]])
    return A


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

def extended_matrix(N,elemental_matrix):
    """
    Assembles the extended global matrix A from given elemental matrices .
    
    input:
    - N: number of elements.
    - elemental_matrix: function to build the elemental matrix

    Returns:
    - A_global: numpy array ((2N+1)x(2N+1))
        The assembled global stiffness matrix.
    """
    _, h = make_partition(N)  # Generate mesh and element size
    num_nodes = 2 * N + 1  # Total number of global nodes
    extetnded = sci.lil_matrix((num_nodes, num_nodes))  #initialise sparce lil_matrix for construction
    
    for k in range(N):  # Loop over elements
        local_matrix = elemental_matrix(h)  # Get elemental matrix
        global_indices = [index_mapping(k, alpha) for alpha in range(3)]  # Map local to global

        # Assemble into global matrix
        for i in range(3):
            for j in range(3):
                extetnded[global_indices[i], global_indices[j]] += local_matrix[i, j]
    
    return extetnded.tocsr() #convert to csr for efficient computations


def extended_load_vector(N, f):
    """
    Assembles the extended global load vector b from elemental load vectors.
    
    input:
    - N: Number of elements.
    - f: Source function f(x).

    Returns:
    - b_global: numpy array (2N+1,)
        The assembled global load vector.
    """
    x, h = make_partition(N)  # Generate mesh and element size
    num_nodes = 2 * N + 1  # Total number of global nodes
    b_global = np.zeros(num_nodes)  # Initialize global vector
    
    for k in range(N):  # Loop over elements
        x_k = x[2*k]  # Left endpoint of the element
        b_local = elemental_load_vec(f, h, x_k)  # Compute local load vector
        global_indices = [index_mapping(k, alpha) for alpha in range(3)]  # Map local to global

        # Assemble into global vector
        for i in range(3):
            b_global[global_indices[i]] += b_local[i]
    
    return b_global


def apply_Dirichlet_conditions(A_global, b_global):
    """
    Applies homogeneous Dirichlet boundary conditions (u(0) = u(1) = 0)
    by removing the first and last rows/columns of the stiffness matrix
    and the first and last entries of the load vector.

    """
    A_reduced = A_global[1:-1, 1:-1]  # Remove first and last rows/columns
    b_reduced = b_global[1:-1]  # Remove first and last elements of the load vector
    
    return A_reduced, b_reduced

def fem_solver(f,N, dirichlet = True):

    '''
    function to solve the linear system Au = b, for the solution u
    A is the stiffness matrix, b is the load vector
    input: 
    - N: number of nodes
    - f: source function

    returns:
    - u: solution to poisson problem in sparce format
    '''
    A = extended_matrix(N, elemental_A)
    b = extended_load_vector(N,f)

    if dirichlet == True:
        A,b = apply_Dirichlet_conditions(A,b)
    u = sci.linalg.spsolve(A,b)

    return u

def plot_FEM_and_exact(u_aprox, u_exact, N, direchlet=True):
    '''
    Plots the approximated FEM solution against the exact solution.
    Plots absolute error and shows max norm of errors

    input:
    - u_aprox: solution from FEM (assumed to be full vector of length 2N+1)
    - u_exact: callable exact solution u(x)
    - N: number of elements
    '''

    x_nodes = np.linspace(0, 1, 2*N + 1)  # FEM global nodes (where u_aprox is defined)

    exact_values = u_exact(x_nodes) 

    if direchlet:
        x_nodes = x_nodes[1:-1]
    exact_values = u_exact(x_nodes)   # Evaluate exact solution at same points
    # print(exact_values.shape)
    errors = np.abs(exact_values - u_aprox)
    
    plt.plot(x_nodes, u_aprox, 'o-', label='FEM Approximation')
    plt.plot(x_nodes, exact_values, '--', label='Exact Solution')
    plt.plot(x_nodes, errors, label = f'|u_exact - u_aprox|, max = {np.max(errors)}')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title(f'FEM vs Exact Solution (N = {N})')
    plt.grid(True)
    plt.legend()
    plt.show()

