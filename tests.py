from FEM_setup import *

### define f1 and f2, test source functions with exact solutions exac1 and exact 2: 
def f1(x):
    '''source function f(x) = 1'''
    return np.ones_like(x)
def exact1(x):
    return 1/2*x*(1-x)

def f2(x):
    return -np.exp(-x)*(np.sin(np.pi*x)*(1-np.pi**2) - 2*np.pi*np.cos(np.pi*x))
    
def exact2(x):
    return np.exp(-x)*(np.sin(np.pi*x))



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
    print(x_nodes, x_nodes[1:-1])
    exact_values = u_exact(x_nodes) 
    print(exact_values.shape)
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

    print(f'u exact =  {exact_values}')
    print(f'u_aprox = {u_aprox}')


#tests: 
u1 = solver(f1,10)
plot_FEM_and_exact(u1, exact1, 10)

u2 = solver(f2, 10)
plot_FEM_and_exact(u2, exact2, 10)