import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from FEM_setup import fem_solver, extended_matrix, elemental_A,make_partition
from scipy.sparse import bmat

def elemental_mass_matrix_lagrende(h):
    '''
    function to create the elemental mass matrix for the legrendre basis. with stepsize h
    F_ij is the intergral over K of psi_i*psi_j dtau
    
    '''
    F = np.array([[2/15,1/15,-1/30 ],[1/15,8/15,1/15],[-1/30,1/15,2/15]])
    return 1/h*F


def build_kkt_system(M, K, alpha, yd):
    """
    Build the KKT system matrix A (shape=(3*ndofs,3*ndofs)) and rhs (shape=(3*ndofs,))
    for:
       [ M        0     -K ] [ y ]   [ M*yd ]
       [ 0   alpha*M   M ] [ u ] = [   0   ]
       [K       -M     0 ] [λ ]   [   0   ]
       
    Here, y,u,λ each have dimension 'ndofs'. 
    'yd' is a vector of length ndofs with the interpolated desired state.
    """
    ndofs = M.shape[0]

    
    zero = lil_matrix((ndofs, ndofs), dtype=float)
    
    block_11 = M
    block_12 = zero
    block_13 = -K
    
    block_21 = zero
    block_22 = alpha * M
    block_23 = M
    
    block_31 = K
    block_32 = -M
    block_33 = zero
    
    A = bmat([
        [block_11, block_12, block_13],
        [block_21, block_22, block_23],
        [block_31, block_32, block_33]
    ]).tocsr()
    
    # Right-hand side
    b_top = M @ yd
    b_mid = np.zeros(ndofs)
    b_bot = np.zeros(ndofs)
    b = np.concatenate([b_top, b_mid, b_bot])
    
    return A, b

def solve_optimal_control_1d(y_d,num_elems=4, alpha=1e-2):
    """
    1) Generate mesh
    2) Assemble interior mass/stiffness
    3) Interpolate y_d
    4) Build KKT system
    5) Solve
    6) Return the solutions (y, u, lambda) plus the mesh for plotting
    """

    num_elems = int(num_elems)
    #make the extended mass matrix for legrendre polynomial basis
    M = extended_matrix(num_elems, elemental_mass_matrix_lagrende) [1:-1, 1:-1] #apply dirichle conditions
    K = extended_matrix(num_elems,elemental_A)[1:-1,1:-1] #slice to apply dirichle contitions
    
    ndofs = M.shape[0]
    
    # Interpolate y_d on the nodes:
    nodes = np.linspace(0, 1, 2*num_elems + 1)   # Global node array
    yd_full = y_d(nodes)                  # Evaluate y_d on all nodes
    yd_vals = yd_full[1:-1]

    A, b = build_kkt_system(M, K, alpha, yd_vals)
    
    sol = spsolve(A, b)
    
    # The solution 'sol' is [y, u, lambda], each of length ndofs:
    y = sol[0:ndofs]
    u = sol[ndofs:2*ndofs]
    lam = sol[2*ndofs:3*ndofs]
    
    return nodes, y, u, lam

