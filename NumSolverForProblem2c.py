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
    
# def generate_mesh(num_elements):
    """
    Create a simple uniform mesh of [0,1] with 'num_elements' elements.
    Each element is quadratic (P2), so we have 2 degrees of freedom per element interior,
    plus 1 overlap at boundaries. 
    We'll store the node coordinates in an array of length 2*num_elements+1.
    """
    # For uniform spacing:
    nodes = np.linspace(0, 1, 2*num_elements + 1)  # e.g. for num_elements=4 -> 9 nodes
    return nodes

# def local_matrices_linear(x0, x1):
    """
    Return the local mass and stiffness matrices for a quadratic element [x0, x1]
    with linear.  In 1D (P2), each local matrix is 3x3.
    We assume equally spaced local nodes in [x0, x1], i.e. at x0, (x0+x1)/2, x1.
    
    For polynomial basis of degree <=2 on [x0, x1], 
    we can use known formulas or do a small numeric integration. 
    Below we use known exact formula for P2 on a segment of length h = x1 - x0:
    
      M_loc = (h/30) * [[4,   2,  -1],
                        [2,  16,   2],
                        [-1,  2,   4]]
      K_loc = (1/h)* (h/6) * [[2, -2,  0],
                              [-2, 4, -2],
                              [0, -2,  2]]
            = (1/h)*(h/6)*...
      (However, the exact entries can differ by sign conventions or reference. 
       We'll just do the standard approach.)
    """
    h = x1 - x0
    M_loc = (h/30.0) * np.array([
        [ 4,  2, -1],
        [ 2, 16,  2],
        [-1,  2,  4]
    ], dtype=float)
    K_loc = (1.0/h) * (h/6.0) * np.array([
        [  2, -2,  0],
        [ -2,  4, -2],
        [  0, -2,  2]
    ], dtype=float)
    return M_loc, K_loc



# def assemble_global_matrices(nodes):
    """
    Assemble the global mass (M) and stiffness (K) matrices for quadratic (P2) elements
    on the interior DOFs (i.e., excluding the Dirichlet boundary nodes at x=0 and x=1).
    
    We'll number the global nodes 0..2*N, but remove node=0 and node=2*N from the system.
    That leaves 'ndofs = 2*N - 1' interior nodes for y (and similarly for u).
    
    Returns:
      M, K   (each shape=(ndofs, ndofs))
    """
    num_elements = len(nodes)//2  # each element uses 2 intervals in the node indexing
    # Actually, if nodes has length 2*num_elements + 1,
    # then num_elements = (len(nodes)-1)//2

    num_elems = (len(nodes)-1)//2
    ndofs = 2*num_elems - 1  # interior DOFs after removing boundary
    
    # We can store with a LIL sparse format for easy assembly:
    M = lil_matrix((ndofs, ndofs), dtype=float)
    K = lil_matrix((ndofs, ndofs), dtype=float)
    
    # Map from local dof index {0,1,2} in element e to global interior dof index
    # The boundary nodes at index=0 and index=2*num_elems are omitted.
    # So a typical element e has local nodes: global j0=2*e, j1=2*e+1, j2=2*e+2.
    # Among these, if j=0 or j=2*N, that's a boundary node => skip in interior indexing.
    
    # We'll build an array 'interior_index' of length = 2*num_elems+1 that maps
    # the global node index => interior index or -1 if boundary:
    interior_index = np.full(len(nodes), -1, dtype=int)
    count = 0
    for j in range(1, len(nodes)-1):
        interior_index[j] = count
        count += 1
    
    # Loop over elements:
    for e in range(num_elems):
        # local node indices in the global numbering:
        j0 = 2*e
        j1 = 2*e + 1
        j2 = 2*e + 2
        
        x0 = nodes[j0]
        x1 = nodes[j2]
        M_loc, K_loc = local_matrices_linear(x0, x1)
        
        # Indices in interior dof space (may be -1 if boundary):
        ig0 = interior_index[j0]
        ig1 = interior_index[j1]
        ig2 = interior_index[j2]
        
        # We add contributions to M,K only if ig>=0 => interior
        local_map = [ (0, ig0), (1, ig1), (2, ig2) ]
        for a, Ia in local_map:
            if Ia < 0:  # boundary node
                continue
            for b, Ib in local_map:
                if Ib < 0:
                    continue
                M[Ia, Ib] += M_loc[a,b]
                K[Ia, Ib] += K_loc[a,b]
    return M.tocsr(), K.tocsr()

def build_kkt_system(M, K, alpha, yd):
    """
    Build the KKT system matrix A (shape=(3*ndofs,3*ndofs)) and rhs (shape=(3*ndofs,))
    for:
       [ M        0     -K ] [ y ]   [ M*yd ]
       [ 0   alpha*M   -M ] [ u ] = [   0   ]
       [-K       -M     0 ] [λ ]   [   0   ]
       
    Here, y,u,λ each have dimension 'ndofs'. 
    'yd' is a vector of length ndofs with the interpolated desired state.
    """
    ndofs = M.shape[0]
    
    # We’ll build A in block form:
    #   A = [[ M,           0,         -K        ],
    #        [ 0,    alpha*M,         -M        ],
    #        [-K,          -M,         0         ]]
    #
    # and b = [M*yd,  0,  0].
    
    # For convenience, convert to lil for easy block assembly:
    
    zero = lil_matrix((ndofs, ndofs), dtype=float)
    
    block_11 = M
    block_12 = zero
    block_13 = -K
    
    block_21 = zero
    block_22 = alpha * M
    block_23 = -M
    
    block_31 = -K
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
    
    # print(y)
    # print(u)
    # print(lam)
    return nodes, y, u, lam

