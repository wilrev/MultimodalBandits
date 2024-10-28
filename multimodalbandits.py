import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import multimodalbandits_oldcode as mbo
from math import inf
from scipy.optimize import nnls, minimize, Bounds
from scipy import stats
import time

DEB = False

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
#check that the set of modes of mu is equal to M
    f = []
    for i in range(mu.shape[0]):
        if (mu[i]- max([mu[k] for k in G[i]]) > 0):
            f.append(i)
    if DEB: print('Modes location check:', f == m)
    
def generate_line_graph(n):
#generate a line graph with n nodes
    G = nx.Graph()
    for i in range(n-1):
        G.add_edge(i,i+1)
    if DEB: print('Generated graph:', G.edges())
    return(G)
        
def generate_multimodal_function(G,m,mmax,sigma2):
#generate a multimodal function over graph G with sets of modes m using a gaussian mixture function
#note: if sigma2 is too large, the gaussian peaks might "merge" so that the generated function is less than len(m)-multimodal
    mu = np.zeros(G.number_of_nodes())
    d = dict(nx.all_pairs_shortest_path_length(G))
    for j in m:
        for i,_ in enumerate(mu):
            mu[i] = mu[i] + (1 + (j == mmax))*np.exp(-(0.5/(sigma2))*d[i][j])
    if DEB: print('Reward function: ', np.round(mu,3))
    return(mu)

def divergence(mu,lam):
    #compute the gaussian divergence between distributions \nu(mu_1),....,\nu(mu_L) and \nu(lambda_1),....,\nu(lambda_L), the output being a vector of divergences
    return((1/2)*(mu-lam)**2)
    
def regression_graph(G,mu,eta,p,k,N):
#find the minimizer of the weighted divergence with set of peaks p and maximizer k on a line graph, discretization number N
    #number of nodes
    K = mu.shape[0]
    #best node
    kstar = np.argmax(mu)
    #discretize the space of lambda with a grid of size N
    grid = np.linspace(np.min(mu),np.max(mu),N)
    #write the tree as rooted at k
    T = nx.bfs_tree(G, k)
    #initialize the values of f,fstar,fsquare, lambdastar
    f = np.zeros([K,len(grid),2])
    fstar = np.zeros([K,N,2]) #(-1,+1)
    fsquare = np.zeros([K,N])
    v = np.zeros([K,N])
    lambdastar = np.zeros(K)
    #loop over the nodes sorted by decreasing depth to compute the f values
    for ell in reversed(list(T.nodes())):
        #if ell is the maximizer of mu, then eta = +infty
        e = 10 ** 10 if (ell == kstar) else eta[ell] 
        #compute the value of f
        for i,z in enumerate(grid):
            if ell in p: #ell can be a mode
                f[ell,i,0] = e*divergence(mu[ell],z)
                f[ell,i,1] = e*divergence(mu[ell],z)
                for j in T.successors(ell): f[ell,i,1] += fsquare[j,i]
            else: #ell cannot be a mode, and needs to have a neighbour whose reward is higher
                f[ell,i,0] = e*divergence(mu[ell],z) 
                for j in T.successors(ell): f[ell,i,0] += fsquare[j,i]
                if any(True for _ in T.successors(ell)):
                    f[ell,i,1] = e*divergence(mu[ell],z) + sum([fsquare[j,i] for j in T.successors(ell)]) + min([(fstar[j,i,1] - fsquare[j,i]) for j in T.successors(ell)])
                else:
                    f[ell,i,1] = 10 ** 10
        #compute the value of fstar and fsquare
        fstar[ell,0,0] = f[ell,0,0]
        fstar[ell,N-1,1] = f[ell,N-1,1]
        for i in range(1,N): fstar[ell,i,0] = min(fstar[ell,i-1,0],f[ell,i,0])
        for i in range(1,N): fstar[ell,N-1-i,1] = min(fstar[ell,N-i,1],f[ell,N-1-i,1])                    
        for i in range(N): fsquare[ell,i] = min(fstar[ell,i,0],fstar[ell,i,1])
    #loop over the nodes sorted by decreasing depth
    lambdastar[k] = max(grid)
    pred=0
    for ell in list(T.nodes()):
        for i in T.predecessors(ell): pred=i
        if (ell == k):
            lambdastar[ell] = max(mu)+1/N
        else:
            for i in range(N): v[ell,i] = f[ell,i, int(grid[i] > lambdastar[pred])] 
            lambdastar[ell] = grid[np.argmin(v[ell,:])]
    #debug information
    if DEB:    
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
    #compute an approximation ratio for the algorithm (i.e. we are guaranteed that the algorithm works better than this)
    v = sum( eta*divergence(mu,lambdastar))
    err = nx.eccentricity(G,k)*(1/N)*(max(mu)-min(mu))*sum(2*eta*np.abs(lambdastar-mu))
    return(v/(v-err))

     
def regression_all(G,mu,eta,N,nb_modes):
     t=time.time()
     #computes explicitly the solution of PFGL(k) when k is in the neighborhood of a mode or mu has strictly less than m modes, performs dynamic programming for the other k's
     kstar = np.argmax(mu)
     m = compute_modes(G,mu)
     K=mu.shape[0]
     lambdastar = np.ones(K)*np.max(mu)
     vstar = sum(eta*divergence(mu,lambdastar)) 
     if nb_modes > len(m): #if mu is strictly less than m-modal, we have no constraints
         for k in range(K):
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
     # t2=time.time()
     # print(t2-t)
     return(lambdastar,vstar)
    

def subgradient_descent(G,mu,N,I,nb_modes):
    #uses values of penalization and step size suggested by the analysis
    kstar = np.argmax(mu)
    Delta = mu[kstar] - mu    
    #print("Delta",Delta)
    K=mu.shape[0]
    eta = np.zeros(K)
    gamma=0
    for k in range(K):
        if (Delta[k] > 0):
            eta[k] = 1/divergence(mu[k],mu[kstar])
            if 2*Delta[k]*eta[k] > gamma:
                gamma=2*Delta[k]*eta[k]               
    B=eta.dot(Delta)/np.min(Delta[np.nonzero(Delta)])
    C=np.linalg.norm(Delta)+gamma*K**(3/2)*(mu[kstar]-np.min(mu))**2 #for gaussian distributions we can take A(mu)=mu^*-mu_*
    eta_mean = eta/I
    delta=np.sqrt(K*B**2/(I*C**2))
    print(delta,gamma)
    for i in range(I-1):
        (lambdastar,vstar) = regression_all(G,mu,eta,N,nb_modes)
        #print("Div",divergence(mu,lambdastar))
        subgradient = Delta - gamma*divergence(mu,lambdastar)*(sum(eta*divergence(mu,lambdastar)) < 1)
        #print( subgradient )    
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
    if sum(eta_mean*divergence(mu,lambdastar))==0:
        print('ATTENTION',eta_mean,divergence(mu,lambdastar),mu,lambdastar)
    eta_final=eta_mean/sum(eta_mean*divergence(mu,lambdastar))
    end_time = time.time()
    runtime = end_time - start_time
    
    print(f"Total runtime: {runtime:.4f} seconds")
    print(f"Total regression time: {total_regression_time:.4f} seconds")
    print(f"Total update time: {total_update_time:.4f} seconds")
    
    return(eta_final,sum(eta_final*Delta),runtime)

def slsqp(G,mu,N,nb_modes,tol): #minimizing function from python, to compare with subgradient descent
    n=len(mu)
    delta=mbo.gap(mu)
    delta_modif=np.where(delta==0,1,delta)   
    eta0=2/delta_modif**2
    bounds=Bounds([0]*n,[np.inf]*n)
    cons = ({'type': 'ineq', 'fun': lambda eta:  regression_all(G,mu,eta,N,nb_modes)[1] -1,
             'jac' : lambda eta:divergence(mu,regression_all(G,mu,eta,N,nb_modes)[0])})
    objectif = lambda eta: eta @ delta
    #jac=
    sol=minimize(objectif, eta0, jac=lambda eta:delta, constraints=cons, tol=tol, options={'maxiter':10**3}, bounds=bounds)
    return(sol)

def generate_valid_modes(n_arms, n_modes):
    modes = []
    available_positions = list(range(n_arms))
    while len(modes) < n_modes and available_positions:
        mode = np.random.choice(available_positions)
        modes.append(mode)
        # Remove the chosen mode and its neighbors from available positions
        available_positions = [p for p in available_positions if abs(p - mode) > 1]
    return sorted(modes)

def generate_spread_out_modes(n_arms, n_modes): 
    #maximize the number of points outside of the neighborhood of the modes to have a more representative runtime
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

def run_experiment(n_arms_list, n_modes_list, N_list, num_trials=1):
    results = {}
    plot_data = {}
    
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
                            _, _, runtime = subgradient_descent_timed(G, mu, N, 10, n_modes) #10 iterations of gradient descent
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
                # axs[i].plot(x, np.exp(intercept) * x**slope, '--')
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

# # Run the experiment
n_arms_list = [5,6,7,8,9,10]
n_modes_list = [2, 3, 4]
N_list = [100, 200, 300]

results, plot_data = run_experiment(n_arms_list, n_modes_list, N_list)
#uncomment above to run the experiment

def analyze_complexity_n_modes(results, plot_data, n_arms_list, n_modes_list, N_list):
    #reorganize the data to plot the runtime w.r.t. number of modes
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
    #reorganize the data to plot the runtime w.r.t. number of discretization points
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

analyze_complexity_n_modes(results, plot_data, n_arms_list, n_modes_list, N_list)
analyze_complexity_N(results, plot_data, n_arms_list, n_modes_list, N_list)

# # Access plot data for a specific N and number of modes
N = 100
n_modes = 2
x_values = plot_data[(n_modes, N)]['x']
y_values = plot_data[(n_modes, N)]['y']

print(f"For N={N} and {n_modes} modes:")
print(f"Number of arms: {x_values}")
print(f"Average runtimes: {y_values}")

