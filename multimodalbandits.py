import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds
from scipy import stats
import time

DEBUG = False
RUN_RUNTIME_EXPERIMENT = False
RUN_REGRET_EXPERIMENT = False


#__________________Helpful auxiliary functions__________________

def compute_modes(G,mu):
    f = []
    for i in range(mu.shape[0]):
        if (mu[i]- max([mu[k] for k in G[i]]) > 0):
            f.append(i)
    return(f)

def modes_neighborhood(G,mu):
    f = []
    for i in range(mu.shape[0]):
        if (mu[i]-max([mu[k] for k in G[i]]) > 0) and not (i in f):
            f.append(i)
            for k in G[i]:
                if not (k in f):
                    f.append(k)
    f.sort()
    return(f)

def check_modes(G,mu,m):
# Check that the set of modes of mu is equal to M
    f = []
    for i in range(mu.shape[0]):
        if (mu[i]- max([mu[k] for k in G[i]]) > 0):
            f.append(i)
    if DEBUG: print('Modes location check:', f == m)
    
def generate_line_graph(n):
# Generate a line graph with n nodes
    G = nx.Graph()
    for i in range(n-1):
        G.add_edge(i,i+1)
    return(G)
        
def generate_multimodal_function(G,m,mmax,sigma2):
# Generate a multimodal function over graph G with sets of modes m using a gaussian mixture function
# Note: if sigma2 is too large, the gaussian peaks might "merge" so that the generated function is less than len(m)-multimodal
    mu = np.zeros(G.number_of_nodes())
    d = dict(nx.all_pairs_shortest_path_length(G))
    for j in m:
        for i,_ in enumerate(mu):
            mu[i] = mu[i] + (1 + (j == mmax))*np.exp(-(0.5/(sigma2))*d[i][j])
    if DEBUG: print('Reward function: ', np.round(mu,3))
    return(mu)

def generate_spread_out_modes(n_arms, n_modes): 
    # Maximizes the number of points outside of the neighborhood of the modes to have a more representative runtime
    if n_modes * 3 - 1 > n_arms:
        raise ValueError(f"Cannot generate {n_modes} spread out modes for {n_arms} arms.")
    
    modes = []
    available_positions = list(range(n_arms))
    
    # Start with the extremes
    modes.append(0)
    modes.append(n_arms - 1)
    available_positions = available_positions[2:-2]  # Remove the extremes and their neighbors
    
    # Add remaining modes
    for _ in range(n_modes - 2):
        if not available_positions:
            raise ValueError("Not enough positions to place modes.")
        
        # Choose the middle of the largest gap
        gaps = [available_positions[i+1] - available_positions[i] for i in range(len(available_positions)-1)]
        largest_gap_index = gaps.index(max(gaps))
        new_mode = (available_positions[largest_gap_index] + available_positions[largest_gap_index+1]) // 2
        
        modes.append(new_mode)
        
        # Remove the new mode and its neighbors from available positions
        available_positions = [p for p in available_positions if abs(p - new_mode) > 1]
    
    return sorted(modes)

def divergence(mu,lam):
    # Compute the gaussian divergence between distributions \nu(mu_1),....,\nu(mu_L) and \nu(lambda_1),....,\nu(lambda_L), the output being a vector of divergences
    return((1/2)*(mu-lam)**2)
    


#__________________Main dynamic programming algorithm__________________

def regression_graph(G,mu,eta,p,k,N):
# Find the minimizer of the weighted divergence with set of peaks p and maximizer k on a line graph, discretization number N
    # Number of nodes
    K = mu.shape[0]
    # Best node
    kstar = np.argmax(mu)
    # Discretize the space of lambda with a grid of size N
    grid = np.linspace(np.min(mu),np.max(mu),N)
    # write the tree as rooted at k
    T = nx.bfs_tree(G, k)
    # Initialize the values of f,fstar,fsquare, lambdastar
    f = np.zeros([K,len(grid),2])
    fstar = np.zeros([K,N,2]) #(-1,+1)
    fsquare = np.zeros([K,N])
    v = np.zeros([K,N])
    lambdastar = np.zeros(mu.shape[0])
    # Loop over the nodes sorted by decreasing depth to compute the f values
    for ell in reversed(list(T.nodes())):
        # If ell is the maximizer of mu, then eta = +infty
        e = 10 ** 10 if (ell == kstar) else eta[ell] 
        # Compute the value of f
        for i,z in enumerate(grid): # f[ell,i,0]=f_ell(grid[i]=z,-1), f[ell,i,1]=f_ell(grid[i]=z,1)
            if ell in p: # ell can be a mode
                f[ell,i,0] = e*divergence(mu[ell],z) 
                f[ell,i,1] = e*divergence(mu[ell],z)
                for j in T.successors(ell): 
                    f[ell,i,1] += fsquare[j,i]
                    f[ell,i,0] += fsquare[j,i]
            else: # ell cannot be a mode, and needs to have a neighbour whose reward is higher
                f[ell,i,0] = e*divergence(mu[ell],z) 
                for j in T.successors(ell): f[ell,i,0] += fsquare[j,i]
                if any(T.successors(ell)):
                    f[ell,i,1] = e*divergence(mu[ell],z) + sum([fsquare[j,i] for j in T.successors(ell)]) + min([min(fstar[j,i,1],f[j,i,0]) - fsquare[j,i] for j in T.successors(ell)])
                else:
                    f[ell,i,1] = 10 ** 10
        # Compute the value of fstar and fsquare
        fstar[ell,0,0] = f[ell,0,0]
        for i in range(1,N): fstar[ell,i,0] = min(fstar[ell,i-1,0],f[ell,i,0]) #min_{w \leq mu_*} [...] = fstar[ell,N-1,0]=min_{i=0,...,N-1} f[ell,i,0]
        fstar[ell,N-1,1] = 10**10
        for i in range(1,N): fstar[ell,N-1-i,1] = min(fstar[ell,N-i,1],f[ell,N-i,1])
        for i in range(N): fsquare[ell,i] = min(fstar[ell,i,0],fstar[ell,i,1])
    lambdastar[k] = max(mu)
    for ell in list(T.nodes()):
        if ell == k:
            continue
        # Find parent
        parent = next(T.predecessors(ell))        
        parent_grid_index = np.where(grid == lambdastar[parent])[0][0]
        
        # Check if the value of ell is constrained by the past
        grandparent_list = list(T.predecessors(parent))
        ell_is_constrained = False
        if grandparent_list :
            grandparent = grandparent_list[0]
            if lambdastar[parent] > lambdastar[grandparent] and parent not in p:
                child_terms = [
                    min(fstar[j, parent_grid_index, 1], f[j, parent_grid_index, 0]) - fsquare[j, parent_grid_index]
                    for j in T.successors(parent)
                ]
                min_child_index = np.argmin(child_terms)
                constrained_child = list(T.successors(parent))[min_child_index]
                
                if ell == constrained_child:
                    ell_is_constrained = True
        
        # Apply appropriate formula if ell is constrained
        if ell_is_constrained:
            # Ensure λ_ℓ is strictly greater than parent's λ
            if fstar[ell, parent_grid_index, 1] <= fstar[ell, parent_grid_index, 0]:
                lambdastar[ell] = grid[parent_grid_index + 1 + np.argmin(f[ell, parent_grid_index+1:, 1])]
            else:
                lambdastar[ell] = lambdastar[parent]
        else:
            # No constraint on ell
            if fstar[ell, parent_grid_index, 1] <= fstar[ell, parent_grid_index, 0]:
                lambdastar[ell] = grid[parent_grid_index + 1 + np.argmin(f[ell, parent_grid_index+1:, 1])]
            else:
                lambdastar[ell] = grid[np.argmin(f[ell, :parent_grid_index+1, 0])]
    # Debug information
    if DEBUG:    
        for ell in range(K):
            print("Node",ell)
            print("Can Be Mode", ell in p)
            print("f minus ",np.round(f[ell,:,0],3))
            print("f plus",np.round(f[ell,:,1],3))
            print("f star minus ",np.round(fstar[ell,:,0],3))
            print("f star plus",np.round(fstar[ell,:,1],3))
            print("f square",np.round(fsquare[ell,:],3))
            print("v", np.round(v[ell,:],3))
        print("Modes",p)
        print("Grid:",np.round(grid,3))
        print("Mu vector",np.round(mu,3))
        print("Eta vector",np.round(eta,3))
        print("Optimal Solution ",np.round(lambdastar,3))
        print("Optimal Value",np.round(fsquare[k,N-1],3))
        check_modes(G,lambdastar,p)
    return(lambdastar)

    
def regression_approx_ratio(G,mu,lambdastar,eta,k,N):
    # Compute an approximation ratio for the algorithm (i.e. we are guaranteed that the algorithm works better than this)
    v = sum( eta*divergence(mu,lambdastar))
    err = nx.eccentricity(G,k)*(1/N)*(max(mu)-min(mu))*sum(2*eta*np.abs(lambdastar-mu))
    return(v/(v-err))

     
def regression_all(G,mu,eta,N,nb_modes):
     # Computes explicitly the solution of PGL(k) when k is in the neighborhood of a mode or mu has strictly less than m modes, performs dynamic programming for the other k's
     kstar = np.argmax(mu)
     m = compute_modes(G,mu)
     K=mu.shape[0]
     lambdastar = np.ones(K)*np.max(mu)
     vstar = sum(eta*divergence(mu,lambdastar)) 
     if nb_modes > len(m): #if mu is strictly less than m-modal, we have no constraints besides lambda[k]=lambda[kstar] for k different than kstar
         for k in [k_val for k_val in range(K) if k_val != kstar]:
             lambdastar_new=np.copy(mu)
             lambdastar_new[k]=mu[kstar]
             if (vstar > sum(eta*divergence(mu,lambdastar_new))):
                 lambdastar = lambdastar_new
                 vstar = sum(eta*divergence(mu,lambdastar))
     else:
         neighborhood=modes_neighborhood(G,mu)
         neighborhood.remove(kstar)
         for k in range(K):
             if k in neighborhood:
                 lambdastar_new=np.copy(mu)
                 lambdastar_new[k]=mu[kstar]
                 if (vstar > sum(eta*divergence(mu,lambdastar_new))):
                     lambdastar = lambdastar_new
                     vstar = sum(eta*divergence(mu,lambdastar))
             else:
                 if (k != kstar):
                     for j in m:
                         p = list(m);p.remove(j);p.append(k)
                         lambda_new=np.copy(mu)
                         lambda_new[k]=mu[kstar]
                         if vstar>sum(eta*divergence(mu,lambda_new)): #if this is not the case there is no point to test this case
                             lambdastar_new = regression_graph(G,mu,eta,p,k,N)
                             if (vstar > sum(eta*divergence(mu,lambdastar_new))):
                                 lambdastar = lambdastar_new
                                 vstar = sum(eta*divergence(mu,lambdastar_new))
     return(lambdastar,vstar)
    

#__________________Projected subgradient descent__________________

def subgradient_descent(G,mu,N,I,nb_modes):
    # Uses values of penalization and step size suggested by the analysis
    kstar = np.argmax(mu)
    Delta = mu[kstar] - mu    
    K=mu.shape[0]
    eta = np.zeros(K)
    gamma=0
    non_zero_gaps = Delta[np.nonzero(Delta)]
    if non_zero_gaps.size == 0:
        return (np.ones(K) / K, 0.0)
    for k in range(K):
        if (Delta[k] > 0):
            eta[k] = 1/divergence(mu[k],mu[kstar])
            if 2*Delta[k]*eta[k] > gamma:
                gamma=2*Delta[k]*eta[k]               
    B=eta.dot(Delta)/np.min(non_zero_gaps)
    C=np.linalg.norm(Delta)+gamma*K**(3/2)*(mu[kstar]-np.min(mu))**2 #for gaussian distributions we can take A(mu)=mu^*-mu_*
    eta_mean = eta/I
    delta=np.sqrt(K*B**2/(I*C**2))
    for i in range(I-1):
        (lambdastar,vstar) = regression_all(G,mu,eta,N,nb_modes)
        subgradient = Delta - gamma*divergence(mu,lambdastar)*(sum(eta*divergence(mu,lambdastar)) < 1)  
        eta = eta - delta*subgradient
        eta[eta < 0] = 0
        eta_mean += eta/I
    eta_mean[kstar] = 0
    (lambdastar,vstar) = regression_all(G,mu,eta_mean,N,nb_modes)
    eta_final=eta_mean/sum(eta_mean*divergence(mu,lambdastar))
    return(eta_final,sum(eta_final*Delta))

def subgradient_descent_timed(G,mu,N,I,nb_modes):
    start_time = time.time()
    kstar = np.argmax(mu)
    Delta = mu[kstar] - mu    
    K=mu.shape[0]
    eta = np.zeros(K)
    gamma=0
    non_zero_gaps = Delta[np.nonzero(Delta)]
    if non_zero_gaps.size == 0:
        return (np.ones(K) / K, 0.0)
    for k in range(K):
        if (Delta[k] > 0):
            eta[k] = 1/divergence(mu[k],mu[kstar])
            if 2*Delta[k]*eta[k] > gamma:
                gamma=2*Delta[k]*eta[k]               
    B=eta.dot(Delta)/np.min(Delta[np.nonzero(Delta)])
    C=np.linalg.norm(Delta)+gamma*K**(3/2)*(mu[kstar]-np.min(mu))**2
    eta_mean = eta/I
    delta=np.sqrt(K*B**2/(I*C**2))
    
    print(f"Initial setup time: {time.time() - start_time:.4f} seconds")
    
    total_regression_time = 0
    total_update_time = 0
    
    for i in range(I-1):
        regression_start = time.time()
        (lambdastar,vstar) = regression_all(G,mu,eta,N,nb_modes)
        total_regression_time += time.time() - regression_start
        
        update_start = time.time()
        subgradient = Delta - gamma*divergence(mu,lambdastar)*(sum(eta*divergence(mu,lambdastar)) < 1)
        eta = eta - delta*subgradient
        eta[eta < 0] = 0
        eta_mean += eta/I
        total_update_time += time.time() - update_start
        
        if i % 100 == 0:
            print(f"Iteration {i}, current objective: {sum(eta*Delta):.4f}")
    
    eta_mean[kstar] = 0
    final_regression_start = time.time()
    (lambdastar,vstar) = regression_all(G,mu,eta_mean,N,nb_modes)
    total_regression_time += time.time() - final_regression_start
    eta_final=eta_mean/sum(eta_mean*divergence(mu,lambdastar))
    end_time = time.time()
    runtime = end_time - start_time
    
    print(f"Total runtime: {runtime:.4f} seconds")
    print(f"Total regression time: {total_regression_time:.4f} seconds")
    print(f"Total update time: {total_update_time:.4f} seconds")
    
    return(eta_final,sum(eta_final*Delta),runtime)

def slsqp(G,mu,N,nb_modes): # Minimizing function from python, can be used instead of subgradient descent
    n=len(mu)
    kstar = np.argmax(mu)
    Delta = mu[kstar] - mu
    K=mu.shape[0]
    eta0 = np.zeros(K)
    for k in range(K):
        if (Delta[k] > 0):
            eta0[k] = 1/divergence(mu[k],mu[kstar])
    bounds=Bounds([0]*n,[np.inf]*n)
    cons = ({'type': 'ineq', 'fun': lambda eta:regression_all(G,mu,eta,N,nb_modes)[1] -1,
             'jac' : lambda eta:divergence(mu,regression_all(G,mu,eta,N,nb_modes)[0])})
    objective = lambda eta: eta @ Delta
    sol=minimize(objective, eta0, jac=lambda eta:Delta, constraints=cons, options={'maxiter':100}, bounds=bounds)
    return(sol)


#__________________Experiments from the paper__________________

def runtime_experiment(n_arms_list, n_modes_list, N_list, num_trials):
    results = {}
    plot_data = {}
    
    plt.style.use('seaborn-whitegrid') 
    plt.rcParams['legend.frameon'] = True
    plt.rcParams.update({
        'font.size': 15,
        'axes.labelsize': 20,
        'axes.titlesize': 20,
        'legend.fontsize': 15,
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'lines.linewidth': 2,
        'lines.markersize': 8,
        'figure.figsize': (10, 5)
    })
    
    for n_modes in n_modes_list:
        for N in N_list:
            key = (n_modes, N)
            plot_data[key] = {'x': [], 'y': []}
    
    for n_arms in n_arms_list:
        for n_modes in n_modes_list:
            if n_modes * 3 - 1 <= n_arms:
                for N in N_list:
                    key = (n_arms, n_modes, N)
                    results[key] = []
                    
                    for trial in range(num_trials):
                        print(f"\nTrial {trial+1} for Arms: {n_arms}, Modes: {n_modes}, N: {N}")
                        G = generate_line_graph(n_arms)
                        try:
                            modes = generate_spread_out_modes(n_arms, n_modes)
                        except ValueError as e:
                            print(f"Couldn't generate modes: {str(e)}")
                            continue
                        
                        max_mode = np.random.choice(modes)
                        mu = generate_multimodal_function(G, modes, max_mode, 1)
                        
                        max_mu = np.max(mu)
                        for mode in modes:
                            mu[mode] += max_mu
                        
                        print(f"Generated modes: {modes}")
                        print(f"Max mode: {max_mode}")
                        print(f"Generated mu (after adjustment): {mu}")
                    
                        try:
                            _, _, runtime = subgradient_descent_timed(G, mu, N, 100, n_modes) #100 iterations of subgradient descent
                            results[key].append(runtime)
                        except Exception as e:
                            print(f"Error occurred: {str(e)}")
                            print(f"mu: {mu}")
                            print(f"modes: {modes}")
                            print(f"max_mode: {max_mode}")
    
    # Calculate average runtimes and store plot data
    for key, runtimes in results.items():
        n_arms, n_modes, N = key
        avg_runtime = np.mean(runtimes)
        plot_key = (n_modes, N)
        plot_data[plot_key]['x'].append(n_arms)
        plot_data[plot_key]['y'].append(avg_runtime)
    
    # Perform log-log regression and plot
    fig, axs = plt.subplots(len(N_list), 1, figsize=(10, 5*len(N_list)))
    if len(N_list) == 1:
        axs = [axs]
    
    for i, N in enumerate(N_list):
        for n_modes in n_modes_list:
            key = (n_modes, N)
            x = np.array(plot_data[key]['x'])
            y = np.array(plot_data[key]['y'])
            if len(x) > 1:  # Need at least two points for regression
                log_x = np.log(x)
                log_y = np.log(y)
                slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)
                axs[i].plot(x, y, 'o-', label=f'{n_modes} modes (slope: {slope:.2f})')
                print(f"N={N}, {n_modes} modes: log-log slope = {slope:.2f}, R^2 = {r_value**2:.2f}")
        
        axs[i].set_xlabel('Number of arms')
        axs[i].set_ylabel('Average runtime (s)')
        axs[i].set_title(f'N = {N}')
        axs[i].legend()
        axs[i].set_xscale('log')
        axs[i].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('runtime_analysis.png')
    plt.show()
    
    return results, plot_data

if RUN_RUNTIME_EXPERIMENT:
    num_trials = 5
    n_arms_list = [20,25,30,35,40,45,50,55,60,65,70]
    n_modes_list = [2, 3, 4, 5]
    N_list = [100]
    results, plot_data = runtime_experiment(n_arms_list, n_modes_list, N_list, num_trials)

# The following functions can be used to plot the runtime with respect to the number of modes or the number of discretization points

def analyze_complexity_n_modes(results, plot_data, n_arms_list, n_modes_list, N_list):
    # Reorganize the data to plot the runtime w.r.t. number of modes
    fig, axs = plt.subplots(len(N_list), 1, figsize=(10, 5*len(N_list)))
    if len(N_list) == 1:
        axs = [axs]

    for i, N in enumerate(N_list):
        for n_arms in n_arms_list:
            x = []
            y = []
            for n_modes in n_modes_list:
                key = (n_arms, n_modes, N)
                if key in results:
                    x.append(n_modes)
                    y.append(np.mean(results[key]))
            
            if len(x) > 1:
                log_x = np.log(x)
                log_y = np.log(y)
                slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)
                axs[i].plot(x, y, 'o-', label=f'{n_arms} arms (slope: {slope:.2f})')
                print(f"N={N}, {n_arms} arms: log-log slope (n_modes) = {slope:.2f}, R^2 = {r_value**2:.2f}")
        
        axs[i].set_xlabel('Number of modes')
        axs[i].set_ylabel('Average runtime (s)')
        axs[i].set_title(f'N = {N}')
        axs[i].legend()
        axs[i].set_xscale('log')
        axs[i].set_yscale('log')

    plt.tight_layout()
    plt.savefig('complexity_n_modes.png')
    plt.show()

def analyze_complexity_N(results, plot_data, n_arms_list, n_modes_list, N_list):
    # Reorganize the data to plot the runtime w.r.t. number of discretization points
    fig, axs = plt.subplots(len(n_modes_list), 1, figsize=(10, 5*len(n_modes_list)))
    if len(n_modes_list) == 1:
        axs = [axs]

    for i, n_modes in enumerate(n_modes_list):
        for n_arms in n_arms_list:
            x = []
            y = []
            for N in N_list:
                key = (n_arms, n_modes, N)
                if key in results:
                    x.append(N)
                    y.append(np.mean(results[key]))
            
            if len(x) > 1:
                log_x = np.log(x)
                log_y = np.log(y)
                slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)
                axs[i].plot(x, y, 'o-', label=f'{n_arms} arms (slope: {slope:.2f})')
                print(f"n_modes={n_modes}, {n_arms} arms: log-log slope (N) = {slope:.2f}, R^2 = {r_value**2:.2f}")
        
        axs[i].set_xlabel('Number of discretization points (N)')
        axs[i].set_ylabel('Average runtime (s)')
        axs[i].set_title(f'n_modes = {n_modes}')
        axs[i].legend()
        axs[i].set_xscale('log')
        axs[i].set_yscale('log')

    plt.tight_layout()
    plt.savefig('complexity_N.png')
    plt.show()

# analyze_complexity_n_modes(results, plot_data, n_arms_list, n_modes_list, N_list)
# analyze_complexity_N(results, plot_data, n_arms_list, n_modes_list, N_list)

# # # Access plot data for a specific N and number of modes
# N = 100
# n_modes = 2
# x_values = plot_data[(n_modes, N)]['x']
# y_values = plot_data[(n_modes, N)]['y']

# print(f"For N={N} and {n_modes} modes:")
# print(f"Number of arms: {x_values}")
# print(f"Average runtimes: {y_values}")


#__________________OSSB implementation__________________

class MultimodalOSSB:
    def __init__(self, G, K, T, m, true_means, N=100, I=100, strategy="multimodal"):
        """
        Args:
            G: NetworkX graph structure
            K: Number of arms
            T: Time horizon
            m: Number of modes allowed
            N: Number of discretization points (default=100)
            I: Subgradient descent iterations (default=100)
            strategy : 'multimodal' (with subgradient descent), 'multimodal slsqp' (with SLSQP), 'local' or 'classical' (classical uses the Graves-Lai solution for bandits without structure)
        """
        self.G = G
        self.K = K
        self.T = T
        self.m = m
        self.N = N
        self.I = I
        self.strategy = strategy
        
        # True means and optimal mean (for regret calculation)
        self.true_means = np.asarray(true_means)
        self.optimal_mean = np.max(self.true_means)
        self.cumulative_regret = 0.0
        self.regret_history = []  # Stores regret at each timestep
        
        # Track empirical means and pull counts
        self.mu_hat = np.zeros(K)
        self.N_pulls = np.zeros(K, dtype=int)
        
    def classical_eta(self):
        """Classical Graves-Lai exploration rates (1 / divergence)."""
        kstar = np.argmax(self.mu_hat)
        Delta = self.mu_hat[kstar] - self.mu_hat
        eta = np.zeros(self.K)
        for k in range(self.K):
            if k != kstar and Delta[k] > 0:
                eta[k] = 1 / (divergence(self.mu_hat[k], self.mu_hat[kstar]))
        return eta
    
    def local_eta(self):
        """local search rates (1 / divergence in the neighborhood of modes)."""
        kstar = np.argmax(self.mu_hat)
        Delta = self.mu_hat[kstar] - self.mu_hat
        eta = np.zeros(self.K)
        neighborhood=modes_neighborhood(self.G,self.mu_hat)
        for k in range(self.K):
            if k in neighborhood and k != kstar and Delta[k] > 0:
                eta[k] = 1 / (divergence(self.mu_hat[k], self.mu_hat[kstar]))
        return eta

    def select_arm(self, t):
        """Select arm using the desired strategy."""
        if self.strategy == "classical":
            eta = self.classical_eta()
        elif self.strategy == "local":
            eta = self.local_eta()
        elif self.strategy == 'local2':
            eta = self.local2_eta()
        elif self.strategy == "multimodal":
            eta, _ = subgradient_descent(
                G=self.G,
                mu=self.mu_hat,
                N=self.N,
                I=self.I,
                nb_modes=self.m
            )
        elif self.strategy == "multimodal slsqp":
            eta = slsqp( 
                G=self.G,
                mu=self.mu_hat,
                N=self.N,
                nb_modes=self.m).x
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
            
        exploration_done = all(
            self.N_pulls[k] >= eta[k] * np.log(t + 1)
            for k in range(self.K)
        )
        
        return (np.argmax(self.mu_hat) if exploration_done 
                else np.argmin(self.N_pulls/(eta + 1e-10)))


    def update(self, arm, reward):
        """Update statistics after arm pull."""
        n = self.N_pulls[arm]
        self.N_pulls[arm] += 1    
        self.mu_hat[arm] = (n*self.mu_hat[arm]+reward)/(n+1)
        instant_regret = self.optimal_mean - self.true_means[arm]
        self.cumulative_regret += instant_regret
        self.regret_history.append(self.cumulative_regret)
        
    def get_regret(self):
        """Return cumulative regret up to the current timestep."""
        return self.cumulative_regret



def run_trials(true_means, graph, m, K, T, strategy, num_trials):
    """Run multiple trials and collect regret histories."""
    all_regrets = []
    t_init=time.time()
    for trial in range(num_trials):
        # Initialize bandit
        bandit = MultimodalOSSB(
            G=graph,
            K=K,
            T=T,
            true_means=true_means,
            m=m,
            strategy=strategy
        )
        
        # Run bandit algorithm
        for t in range(T):
            arm = bandit.select_arm(t)
            reward = np.random.normal(bandit.true_means[arm], 1.0)
            bandit.update(arm, reward)
            
        # Store regret history
        all_regrets.append(bandit.regret_history)
        # Prints progress at each percent of the total number of trials
        if trial%(num_trials/100) == 0 and strategy == "multimodal slsqp":
            print("percentage done:",trial/num_trials,'time elapsed:',time.time()-t_init)
        
    return np.array(all_regrets)

def plot_results(mmslsqp_regrets, local_regrets, classical_regrets, T, num_trials):
    """Plot regret curves with empirical confidence intervals."""
    plt.figure(figsize=(10, 6))
    # 97.5% quantile of standard Gaussian 
    quantile = stats.norm.ppf(0.975, loc=0, scale=1)
    
    # # Multimodal (with projected subgradient descent) curve
    # mm_mean = np.mean(mm_regrets, axis=0)
    # mm_std = np.std(mm_regrets, axis=0)
    # plt.plot(mm_mean, label="Multimodal OSSB")
    # plt.fill_between(
    #     range(T), mm_mean - mm_std, mm_mean + mm_std,
    #     alpha=0.2
    # )
    
    # Multimodal slsqp curve
    mmslsqp_mean = np.mean(mmslsqp_regrets, axis=0)
    mmslsqp_std = quantile/np.sqrt(num_trials)*np.std(mmslsqp_regrets, axis=0)
    plt.plot(mmslsqp_mean, label="Multimodal OSSB", marker='o', markevery=num_trials/25)
    plt.fill_between(
        range(T), mmslsqp_mean - mmslsqp_std, mmslsqp_mean + mmslsqp_std,
        alpha=0.2
    )
    # Local curve
    local_mean = np.mean(local_regrets, axis=0)
    local_std = quantile/np.sqrt(num_trials)*np.std(local_regrets, axis=0)
    plt.plot(local_mean, label="Local search OSSB" ,marker='^', markevery=num_trials/25)
    plt.fill_between(
        range(T), local_mean - local_std, local_mean + local_std,
        alpha=0.2
    )
    
    
    # Classical curve
    classical_mean = np.mean(classical_regrets, axis=0)
    classical_std = quantile/np.sqrt(num_trials)*np.std(classical_regrets, axis=0)
    plt.plot(classical_mean, label="Classical OSSB", marker='s', markevery=num_trials/25)
    plt.fill_between(
        range(T), classical_mean - classical_std,
        classical_mean + classical_std, alpha=0.2
    )
    
    plt.xlabel("Time Step",fontsize=20)
    plt.ylabel("Cumulative Regret",fontsize=20)
    plt.legend(fontsize="15", loc="upper left")
    plt.grid(True)
    plt.show()
    
    
if RUN_REGRET_EXPERIMENT:
    num_trials = 50
    K = 7
    # Create a line graph with K nodes
    G = nx.path_graph(K)
    T = 500
    m = 2  # Allow 2 modes
    true_means=generate_multimodal_function(G,[0,6],6,1)
    # mm_regrets = run_trials(true_means, G, m=m, K=K, T=T, 
    #                         strategy="multimodal", num_trials=num_trials)
    mmslsqp_regrets = run_trials(true_means, G, m=m, K=K, T=T, 
                            strategy="multimodal slsqp", num_trials=num_trials)
    local_regrets = run_trials(true_means, G, m=m, K=K, T=T, 
                            strategy="local", num_trials=num_trials)
    classical_regrets = run_trials(true_means, G, m=m, K=K, T=T,
                                  strategy="classical", num_trials=num_trials)
    
    
    #Plot results
    plot_results(mmslsqp_regrets, local_regrets, classical_regrets, T, num_trials)
    
