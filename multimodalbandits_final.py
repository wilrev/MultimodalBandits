import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds
from scipy import stats
import time
from tqdm import tqdm
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from joblib import Parallel, delayed

DEBUG = False
RUNTIME_EXPERIMENT = False
RUNTIME_IMPROVED_DP_EXPERIMENT = False
REGRET_EXPERIMENT = True


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

def get_random_modes(G, num_modes, k_star):
    modes = {k_star}
    forbidden = {k_star} | set(G.neighbors(k_star))
    available_nodes = [n for n in G.nodes() if n not in forbidden]
    
    while len(modes) < num_modes and available_nodes:
        new_mode = np.random.choice(available_nodes)
        modes.add(new_mode)
        forbidden.update({new_mode} | set(G.neighbors(new_mode)))
        available_nodes = [n for n in available_nodes if n not in forbidden]
        
    return list(modes)

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
    v = np.full([K, N], -1, dtype=int) # v[ell, i] will store the identity of the child that minimizes g_j(z_i) for parent ell
    lambdastar = np.zeros(mu.shape[0])
    # Loop over the nodes sorted by decreasing depth to compute the f values
    for ell in reversed(list(T.nodes())): # could use reversed(list(nx.topological_sort(T))), could also use list(nx.dfs_postorder_nodes(T, source=k))
        # If ell is the maximizer of mu, then eta = +infty
        e = 10**10 if (ell == kstar) else eta[ell] 
        # Compute the value of f
        for i,z in enumerate(grid): # f[ell,i,0]=f_ell(grid[i]=z,-1), f[ell,i,1]=f_ell(grid[i]=z,1)
            if ell in p: # ell can be a mode
                f[ell,i,0] = e*divergence(mu[ell],z) 
                f[ell,i,1] = e*divergence(mu[ell],z)
                for j in T.successors(ell): 
                    f[ell,i,1] += fsquare[j,i]
                    f[ell,i,0] += fsquare[j,i]
            else: # ell cannot be a mode, and needs to have a neighbour whose reward is higher or equal
                f[ell,i,0] = e*divergence(mu[ell],z) + sum([fsquare[j,i] for j in T.successors(ell)])
                children = list(T.successors(ell)) 
                if children:
                    # Explicitly find the minimum g_cost and the winning child
                    min_g_cost = float('inf')
                    winning_child = -1
                    for child in children:
                        g_cost = min(fstar[child, i, 1], f[child, i, 0]) - fsquare[child, i]
                        if g_cost < min_g_cost:
                            min_g_cost = g_cost
                            winning_child = child
                    
                    # Store the winner's identity
                    v[ell, i] = winning_child

                    # Compute f[ell,i,1] using the found minimum
                    f[ell,i,1] = e*divergence(mu[ell],z) + sum([fsquare[j,i] for j in children]) + min_g_cost
                else:
                    f[ell,i,1] = 10**10
        # Compute the value of fstar and fsquare
        fstar[ell,0,0] = f[ell,0,0]
        for i in range(1,N): fstar[ell,i,0] = min(fstar[ell,i-1,0],f[ell,i,0]) #min_{w \leq mu^\star} [...] = fstar[ell,N-1,0]=min_{i=0,...,N-1} f[ell,i,0]
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
                 constrained_child = v[parent, parent_grid_index]
                
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
        print(" ....... regression_graph ....... p =  ",p)
        for ell in list(T.nodes()):
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
         if kstar in neighborhood:
             neighborhood.remove(kstar)
             remote_arms = [k for k in range(K) if k != kstar and k not in neighborhood]
         for k in neighborhood:
             lambdastar_new=np.copy(mu)
             lambdastar_new[k]=mu[kstar]
             if (vstar > sum(eta*divergence(mu,lambdastar_new))):
                 lambdastar = lambdastar_new
                 vstar = sum(eta*divergence(mu,lambdastar))
                 
         if nb_modes == 1:
             return (lambdastar, vstar) # This is the optimal solution in the unimodal case
      
         remote_arms = [k for k in range(K) if k != kstar and k not in neighborhood]
         for k in remote_arms:
             for j in m:
                 p = list(m);p.remove(j);p.append(k)
                 lambda_new=np.copy(mu)
                 lambda_new[k]=mu[kstar]
                 if vstar>sum(eta*divergence(mu,lambda_new)): # If this is not the case, it is unecessary to go further
                     lambdastar_new = regression_graph(G,mu,eta,p,k,N)
                     if (vstar > sum(eta*divergence(mu,lambdastar_new))):
                         lambdastar = lambdastar_new
                         vstar = sum(eta*divergence(mu,lambdastar))
     return (lambdastar,vstar)
    

#__________________fast Dynamic Programming algorithm__________________

def fast_minimization(g,b,c):
# Compute the minimization of sum_{v \in C(\ell)} g(b_v,c_v) across all ({\bf b},{\bf c}) \in B_ell(b,c) \times C_ell(b,c) using "fast minimization"
    vstar = 10**10
    bstar = [0 for i in range( g.shape[0] )] 
    cstar = [0 for i in range( g.shape[0] )] 
    if (b==0) and (c==0):
        vstar = np.sum(g[:,0,0])
    elif (b==1) and (c==0): 
        vstar = np.min(  g[:,1,0] - g[:,0,0] ) + np.sum(g[:,0,0])
        bstar[np.argmin(  g[:,1,0] - g[:,0,0] ) ] = 1
    elif (b==0) and (c==1): 
        vstar = np.min(  g[:,0,1] - g[:,0,0] )  + np.sum(g[:,0,0])
        cstar[np.argmin(  g[:,0,1] - g[:,0,0] ) ] = 1
    elif (b==1) and (c==1): 
        #first case, b_v = c_v for some node v 
        v11 = np.argmin(g[:,1,1] - g[:,0,0])
        D11 = g[v11,1,1] - g[v11,0,0]
        #second case, there exists two distinct nodes w_1,w_2 such that b_{w_1} = 1 = c_{w_2}, b_{w_2} = 0 = c_{w_1}
        #in this case one must be careful if both A01 and A10 are minimized at the same entry, if so, we must chose the first best and second best 
        #note: this case is only possible if there are at least two entries
        if (g.shape[0] > 1):
            A10 = g[:,1,0] - g[:,0,0]
            v10 = A10.argsort()
            A01 = g[:,0,1] - g[:,0,0]
            v01 = A01.argsort()
            if (v10[0] == v01[0]):
                D1001 = min(  A10[v10[0]] + A01[v01[1]] ,  A10[v10[1]] + A01[v01[0]])
            else:
                D1001 = A10[v10[0]] + A01[v01[0]]
            vstar = min(D11,D1001)+ np.sum(g[:,0,0])
            if D11 < D1001: #or D11 <= D1001? not the main issue but to keep in mind
                bstar[v11] = 1
                cstar[v11] = 1
            else:
                if (v01[0] == v10[0]):
                    if (A10[v10[0]] + A01[v01[1]] <  A10[v10[1]] + A01[v01[0]]):
                        bstar[v10[0]] = 1
                        cstar[v01[1]] = 1
                    else:
                        bstar[v10[1]] = 1
                        cstar[v01[0]] = 1
                else:
                        bstar[v10[0]] = 1
                        cstar[v01[0]] = 1
        else:
            bstar[0] = 1
            cstar[0] = 1
            vstar = D11+ np.sum(g[:,0,0])
    return(vstar,bstar,cstar)


def fast_dynamic_programming(G,mu,eta,N,nb_modes):
    # Find the minimal value of the weighted divergence with m modes, discretization number N
    M = compute_modes(G,mu)
    K = mu.shape[0]
    kstar = np.argmax(mu)

   # Case number 1: when the maximal entry of confusing parameter lambda does lie in the neighbourhood of a mode of mu, or when the number of modes of lambda is strictly greater than that of mu
   # Computes explicitly the solution of PGL(k) when k is in the neighborhood of a mode or mu has strictly less than m modes, performs dynamic programming for the other k's
    lambdastar1 = np.ones(K)*np.max(mu)
    vstar1 = sum(eta*divergence(mu,lambdastar1)) 
    if nb_modes > len(M): # if mu is strictly less than m-modal, we have no constraints besides lambda[k]=lambda[kstar] for k different than kstar
        for k in [k_val for k_val in range(K) if k_val != kstar]:
            lambdastar_new = np.copy(mu)
            lambdastar_new[k]= mu[kstar]
            if (vstar1 > sum(eta*divergence(mu,lambdastar_new))):
                lambdastar1 = lambdastar_new
                vstar1 = sum(eta*divergence(mu,lambdastar1))
        return (lambdastar1,vstar1) 
    else:
        neighborhood = modes_neighborhood(G,mu)
        for k in [k_val for k_val in neighborhood if k_val != kstar]:
            lambdastar_new = np.copy(mu)
            lambdastar_new[k] = mu[kstar]
            if (vstar1 > sum(eta*divergence(mu,lambdastar_new))):
                lambdastar1 = lambdastar_new
                vstar1 = sum(eta*divergence(mu,lambdastar1))
        if nb_modes == 1:
            return (lambdastar1,vstar1) # This is the best solution in the unimodal setting

    # Case number 2: when the maximal entry of confusing parameter lambda does not lie in the neighbourhood of a mode of mu
    # Discretize the space of lambda with a grid of size N
    grid = np.linspace(np.min(mu),np.max(mu),N)
    T = nx.bfs_tree(G, kstar)

    # Initialize value arrays
    h = np.zeros([K,len(grid),3,2,2])
    hplus = np.zeros([K,len(grid),2,2])
    hminus = np.zeros([K,len(grid),2,2])
    hequal = np.zeros([K,len(grid),2,2])
    hstar = np.zeros([K,len(grid),2,2])
    hplusequal = np.zeros([K,len(grid),2,2])
    
    # Pointers to store the index that minimizes hminus, hplus
    # Shape: (K, N, b=2, c=2)
    ptr_hminus = np.zeros([K, len(grid), 2, 2], dtype=int)
    ptr_hplus = np.zeros([K, len(grid), 2, 2], dtype=int)

 
    max_deg = max((len(list(T[ell])) for ell in T.nodes()), default=0)
    # Pointers for the a==1 case
    # Shape: (K, N, b=2, c=2, max_deg)
    best_bs_a1 = np.zeros([K, len(grid), 2, 2, max_deg], dtype=int)
    best_cs_a1 = np.zeros([K, len(grid), 2, 2, max_deg], dtype=int)
    best_w_idx_a1 = np.full([K, len(grid), 2, 2], -1, dtype=int)
    champion_w_idx_a1 = np.full([K, len(grid), 2, 2], -1, dtype=int)

    # Forward pass over the nodes, sorted by decreasing depth
    for ell in reversed(list(T.nodes())):
        children = list(T[ell])
        num_children = len(children)

        if not(children):  # leaf
            for a in range(3):
                for b in range(2):
                    for c in range(2):
                        is_max = ((a == 0) and (b==1) and (ell not in M))
                        valid = (a != 1) and (b == (a==0)*(ell not in M)) and (c == (a!=0)*(ell in M))
                        for i,z in enumerate(grid):
                            if (not valid) or (is_max and (i != N-1)):
                                h[ ell,i,a,b,c ] = 1e10
                            else:
                                h[ ell,i,a,b,c ] = eta[ell]*divergence(mu[ell],z)
                                
        else:  # Internal node
            # For each grid point i and each of the 4 child-flag configurations,
            # find the candidate child  that offers the minimum penalty.
            if num_children > 0:
                for i in range(N):
                    for b_child in range(2):
                        for c_child in range(2):
                            # Calculate penalties h_ge - h_star for all children at once.
                            penalties = hplusequal[children, i, b_child, c_child] - hstar[children, i, b_child, c_child]
                            candidate_w_idx_a1[ell, i, b_child, c_child] = np.argmin(penalties)

            for i,z in enumerate(grid):
                # Construct the candidate set W_ell(z)
                candidate_indices = {
                    candidate_w_idx_a1[ell, i, b_c[0], b_c[1]]
                    for b_c in [(0,0), (0,1), (1,0), (1,1)] if champion_w_idx_a1[ell, i, b_c[0], b_c[1]] != -1
                }
                # Compute h[ell,i,a,b,c] for each a,b,c
                for a in range(3):
                    for b in range(2):
                        for c in range(2):
                            is_max = ((a == 0) and (b==1) and (ell not in M))
                            div = 1e10 if is_max and (i != N-1) else eta[ell]*divergence(mu[ell],z)

                            if a == 0:
                                h[ell,i,a,b,c] = div + fast_minimization(hminus[children,i,:,:], b -1*(ell not in M), c)[0]

                            elif a == 1:
                                # G_base: shape (num_children, 2, 2) containing hstar for each child
                                G_base = np.array([ hstar[w, i, :, :] for w in children ])
                                best_val = 1e20
                                best_bs, best_cs, best_w_idx = None, None, -1
                                
                                # Loop over the small candidate set W_ell(z) (at most 4 children)
                                for idx_w in candidate_indices:
                                    w = children[idx_w]
                                    G = G_base.copy()
                                    G[idx_w, :, :] = hplusequal[w, i, :, :]
                                    val, bs_cand, cs_cand = fast_minimization(G, b, c - 1*(ell in M))
                                    
                                    if val < best_val:
                                        best_val, best_bs, best_cs, best_w_idx = val, bs_cand, cs_cand, idx_w
                                
                                h[ell,i,1,b,c] = div + best_val
                                # Store pointers
                                if best_bs is not None:
                                    best_bs_a1[ell,i,b,c,:num_children] = np.array(best_bs[:num_children], dtype=int)
                                    best_cs_a1[ell,i,b,c,:num_children] = np.array(best_cs[:num_children], dtype=int)
                                    best_w_idx_a1[ell,i,b,c] = int(best_w_idx)

                            elif a == 2:
                                h[ell,i,a,b,c] = div + fast_minimization(hstar[children,i,:,:], b, c-1*(ell in M))[0]

        # Compute the aggregated tables and store pointers for backtracking
        for b in range(2):
            for c in range(2):
                # Compute hplus and store pointer
                # Base case: i = N-1
                hplus[ell, N-1, b, c] = 1e10
                ptr_hplus[ell, N-1, b, c] = -1
                
                # Iterate downwards from i = N-2 to 0
                for i in range(N-2, -1, -1):
                    val_from_ip1 = min(h[ell, i+1, 0, b, c], h[ell, i+1, 1, b, c])
                    if val_from_ip1 <= hplus[ell, i+1, b, c]:
                        hplus[ell, i, b, c] = val_from_ip1
                        ptr_hplus[ell, i, b, c] = i + 1
                    else:
                        hplus[ell, i, b, c] = hplus[ell, i+1, b, c]
                        ptr_hplus[ell, i, b, c] = ptr_hplus[ell, i+1, b, c]

                # Compute hminus and store pointer
                # Base case: i = 0
                hminus[ell, 0, b, c] = 1e10
                ptr_hminus[ell, 0, b, c] = -1
                
                # Iterate upwards from i = 1 to N-1
                for i in range(1, N):
                    val_from_im1 = h[ell, i-1, 2, b, c]
                    if val_from_im1 <= hminus[ell, i-1, b, c]:
                        hminus[ell, i, b, c] = val_from_im1
                        ptr_hminus[ell, i, b, c] = i - 1
                    else:
                        hminus[ell, i, b, c] = hminus[ell, i-1, b, c]
                        ptr_hminus[ell, i, b, c] = ptr_hminus[ell, i-1, b, c]

                # Compute hequal, hstar, hplusequal
                for i in range(N):
                    hequal[ell,i,b,c] = h[ell,i,2,b,c]
                for i in range(N):
                    hstar[ell,i,b,c] = min(hminus[ell,i,b,c], hequal[ell,i,b,c], hplus[ell,i,b,c])
                for i in range(N):
                    hplusequal[ell,i,b,c] = min(hequal[ell,i,b,c], hplus[ell,i,b,c])

    # Backtracking 
    istar = [0 for i in range(K)]
    astar = [0 for i in range(K)]
    bstar = [0 for i in range(K)]
    cstar = [0 for i in range(K)]

    istar[kstar] = N-1
    bstar[kstar] = 1
    cstar[kstar] = 1
    astar[kstar] = 0
    theoretical_vstar2 = h[kstar, N-1, 0, 1, 1]
    if DEBUG:
        print(f"\nDEBUG: Theoretical optimal value from forward pass (vstar2) = {theoretical_vstar2:.8f}")

    # Compute the optimal solution using stored pointers where needed
    for ell in list(T.nodes()):
        children = list(T[ell])
        if children:
            if astar[ell] == 0:
                (vs,bs,cs) = fast_minimization(hminus[children,istar[ell],:,:], bstar[ell] -1*(ell not in M), cstar[ell])
                for i,v in enumerate(children):
                    bstar[v] = bs[i]
                    cstar[v] = cs[i]
                    istar[v] = np.argmin(h[ v,:,2,bstar[v],cstar[v]]  +  (1e10)*(grid >=  grid[istar[ell]]))

            elif astar[ell] == 1:
                # Read stored bs/cs and chosen special child index
                num_children = len(children)
                bs = best_bs_a1[ell, istar[ell], bstar[ell], cstar[ell], :num_children].astype(int)
                cs = best_cs_a1[ell, istar[ell], bstar[ell], cstar[ell], :num_children].astype(int)
                wi = int(best_w_idx_a1[ell, istar[ell], bstar[ell], cstar[ell]])
    
                for i,v in enumerate(children):
                    bstar[v] = int(bs[i])
                    cstar[v] = int(cs[i])
                    
                    i_parent = istar[ell]
                    b_child = bstar[v]
                    c_child = cstar[v]

                    if i == wi:
                        # Special child logic, use hplusequal
                        val_plusequal = hplusequal[v, i_parent, b_child, c_child]
                        if hplus[v, i_parent, b_child, c_child] == val_plusequal:
                            istar[v] = ptr_hplus[v, i_parent, b_child, c_child]
                        else: # hequal must be the minimum
                            istar[v] = i_parent
                    else:
                        # Other children logic, use hstar
                        val_star = hstar[v, i_parent, b_child, c_child]

                        # Check branches in order. If values are identical, this establishes a preference.
                        # Preference: z_v > z_ell (hplus) -> z_v = z_ell (hequal) -> z_v < z_ell (hminus)
                        if hplus[v, i_parent, b_child, c_child] == val_star:
                            istar[v] = ptr_hplus[v, i_parent, b_child, c_child]
                        elif hequal[v, i_parent, b_child, c_child] == val_star:
                            istar[v] = i_parent
                        else: # hminus must be the minimum
                            istar[v] = ptr_hminus[v, i_parent, b_child, c_child]

            elif astar[ell] == 2:
                (vs,bs,cs) = fast_minimization(hstar[children,istar[ell],:,:], bstar[ell], cstar[ell]-1*(ell in M))
                for i,v in enumerate(children):
                    bstar[v] = bs[i]
                    cstar[v] = cs[i]

                    i_parent = istar[ell]
                    b_child = bstar[v]
                    c_child = cstar[v]
                    
                    # Same logic as the other children in a==1
                    val_star = hstar[v, i_parent, b_child, c_child]
                    
                    if hplus[v, i_parent, b_child, c_child] == val_star:
                        istar[v] = ptr_hplus[v, i_parent, b_child, c_child]
                    elif hequal[v, i_parent, b_child, c_child] == val_star:
                        istar[v] = i_parent
                    else: # hminus must be the minimum
                        istar[v] = ptr_hminus[v, i_parent, b_child, c_child]

            for i,v in enumerate(children):
                if istar[v] <= istar[ell]:
                    astar[v] = 2
                elif h[v,istar[v],1,bstar[v],cstar[v]] <  h[v,istar[v],0,bstar[v],cstar[v]]:
                    astar[v] = 1
                else:
                    astar[v] = 0

    lambdastar2 = np.array([grid[istar[i]] for i in range(K)])
    vstar2 = np.sum(eta*divergence(mu,lambdastar2))
    if DEBUG:
        print(f"DEBUG: Value from reconstructed lambda (vstar2_recalc) = {vstar2:.8f}")

    # pick the best of cases 1 and 2
    if vstar1 < vstar2:
        vstar = vstar1
        lambdastar = lambdastar1
    else:
        vstar = vstar2
        lambdastar = lambdastar2

    return (lambdastar,vstar)


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
    C=np.linalg.norm(Delta)+gamma*K**(3/2)*(mu[kstar]-np.min(mu))**2 # for gaussian distributions with variance 1 we can take A(mu)=mu^*-mu_*
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
    neighborhood=modes_neighborhood(G,mu)
    for k in range(K):
        if (Delta[k] > 0):
            eta0[k] = 1/divergence(mu[k],mu[kstar])
    bounds=Bounds([0]*n,[np.inf]*n)
    cons = ({'type': 'ineq', 'fun': lambda eta:regression_all(G,mu,eta,N,nb_modes)[1] -1,
             'jac' : lambda eta:divergence(mu,regression_all(G,mu,eta,N,nb_modes)[0])})
    objective = lambda eta: eta @ Delta
    sol=minimize(objective, eta0, jac=lambda eta:Delta, constraints=cons, options={'maxiter':100}, bounds=bounds)
    return(sol)


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
        # Classical Graves-Lai exploration rates (1 / divergence)
        kstar = np.argmax(self.mu_hat)
        Delta = self.mu_hat[kstar] - self.mu_hat
        eta = np.zeros(self.K)
        for k in range(self.K):
            if k != kstar and Delta[k] > 0:
                eta[k] = 1 / (divergence(self.mu_hat[k], self.mu_hat[kstar]))
        return eta
    
    def local_eta(self):
        # local search rates (1 / divergence in the neighborhood of modes)
        kstar = np.argmax(self.mu_hat)
        Delta = self.mu_hat[kstar] - self.mu_hat
        eta = np.zeros(self.K)
        neighborhood=modes_neighborhood(self.G,self.mu_hat)
        for k in range(self.K):
            if k in neighborhood and k != kstar and Delta[k] > 0:
                eta[k] = 1 / (divergence(self.mu_hat[k], self.mu_hat[kstar]))
        return eta

    def select_arm(self, t):
        # Select arm using the desired strategy
        if self.strategy == "classical":
            eta = self.classical_eta()
        elif self.strategy == "local":
            eta = self.local_eta()
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
        # Update statistics after arm pull
        n = self.N_pulls[arm]
        self.N_pulls[arm] += 1    
        self.mu_hat[arm] = (n*self.mu_hat[arm]+reward)/(n+1)
        instant_regret = self.optimal_mean - self.true_means[arm]
        self.cumulative_regret += instant_regret
        self.regret_history.append(self.cumulative_regret)
        
    def get_regret(self):
        # Return cumulative regret up to the current timestep
        return self.cumulative_regret



#__________________Experiments from the paper__________________

# Runtime Experiment (OriginaL DP, varying the number of modes and nodes)

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
                        modes = generate_spread_out_modes(n_arms, n_modes)
                        
                        max_mode = np.random.choice(modes)
                        mu = generate_multimodal_function(G, modes, max_mode, 1)
                        
                        max_mu = np.max(mu)
                        for mode in modes:
                            mu[mode] += max_mu
                        
                        print(f"Generated modes: {modes}")
                        print(f"Max mode: {max_mode}")
                        print(f"Generated mu (after adjustment): {mu}")
                        _, _, runtime = subgradient_descent_timed(G, mu, N, 100, n_modes) #100 iterations of subgradient descent
                        results[key].append(runtime)
                            
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
        #axs[i].set_title(f'N = {N}')
        axs[i].legend()
        axs[i].set_xscale('log')
        axs[i].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('runtime_analysis.png')
    plt.show()
    
    return results, plot_data

if RUNTIME_EXPERIMENT:
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
        #axs[i].set_title(f'N = {N}')
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
        #axs[i].set_title(f'n_modes = {n_modes}')
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



# Runtime experiment (Original DP vs Improved DP)

def runtime_DP_single_trial(trial_seed, N, num_modes, name, graph_func, K):

    np.random.seed(trial_seed) 
    G = graph_func(K)
    
    k_star = np.random.randint(K)
    
    modes_to_set = get_random_modes(G, num_modes, k_star)
    mu = generate_multimodal_function(G, modes_to_set, k_star, 1.0)
    eta = np.random.rand(K)
    nb_modes_in_mu = len(compute_modes(G, mu))

    start = time.time()
    regression_all(G, mu, eta, N, nb_modes_in_mu)
    slow_time = time.time() - start

    start = time.time()
    fast_dynamic_programming(G, mu, eta, N, nb_modes_in_mu)
    fast_time = time.time() - start
    
    return slow_time, fast_time

def runtime_DP(num_trials=50, N=100, num_modes=3, seed_base=0):
    results = []

    print(f"\n--- Experiment 1: Varying K for random trees (modes={num_modes}) ---")
    node_counts = [100, 400, 700, 1000, 1300, 1600, 1900]
    graph_types = {
        'Random Tree': lambda n: nx.random_tree(n, seed=seed_base) # Use seed_base for graph generation
    }
    for name, graph_func in graph_types.items():
        for K in node_counts:
            print(f"Running Experiment 1: {name}, K={K}")
            # Parallelize the trials for this specific (name, K) configuration
            trial_results = Parallel(n_jobs=-1, verbose=5)(
                delayed(runtime_DP_single_trial)(
                    seed_base + i, N, num_modes, name, graph_func, K
                ) for i in range(num_trials)
            )
            # Collect results
            for slow_time, fast_time in trial_results:
                results.append({
                    'Experiment': name,
                    'Parameter': K,
                    'Original DP': slow_time,
                    'Improved DP': fast_time
                })

    print(f"\n--- Experiment 2: Varying branching factor of balanced trees (modes={num_modes}) ---")
    height = 3
    bf_list = [2, 4, 6, 8, 10, 12]
    for bf in bf_list:
        G_base = nx.balanced_tree(bf, height)
        K = G_base.number_of_nodes()
        print(f"Running Experiment 2: Balanced tree, d={bf}, K={K}")
        # Parallelize trials for this specific (bf, K)
        trial_results = Parallel(n_jobs=-1, verbose=5)(
            delayed(runtime_DP_single_trial)(
                seed_base + i, N, num_modes, 
                f'Balanced Tree (h={height})', 
                lambda k: nx.balanced_tree(bf, height), K 
            ) for i in range(num_trials)
        )
        # Collect results
        for slow_time, fast_time in trial_results:
            results.append({
                'Experiment': f'Balanced Tree (h={height})',
                'Parameter': bf,
                'Original DP': slow_time,
                'Improved DP': fast_time
            })

    # Plotting the results
    df = pd.DataFrame(results)
    sns.set_theme(style="whitegrid")
    color_palette = ["#1f77b4", "#ff7f0e"] # Blue, Orange
    markers_list = ["o", "x"]
    algs = ['Original DP', 'Improved DP']

    handles = [
        Line2D([0], [0], marker=markers_list[i], linestyle='None', markerfacecolor=color_palette[i], markeredgecolor=color_palette[i],
               markersize=8, label=algs[i])
        for i in range(len(algs))
    ]

    # Determine which experiments to plot (Exp 1 and 2)
    exp_names_to_plot = ['Random Tree', f'Balanced Tree (h={height})']
    df_plot = df[df['Experiment'].isin(exp_names_to_plot)].copy()

    if not df_plot.empty:
        print("\n--- Generating plot for Experiments 1 & 2 ---")
        df_melted = df_plot.melt(id_vars=['Experiment', 'Parameter'], value_vars=['Original DP', 'Improved DP'],
                                var_name='Algorithm', value_name='Runtime (s)')

        g = sns.catplot(data=df_melted, x='Parameter', y='Runtime (s)', hue='Algorithm',
                        col='Experiment', kind='point', col_wrap=2, sharex=False, sharey=False,
                        palette=color_palette, markers=markers_list, hue_order=algs,
                        aspect=1.5, dodge=True, legend=False)

        axes = g.axes.flatten()
        for ax in axes:
            title = ax.get_title()
            ax.set_xlabel("Number of Nodes (K)" if 'Random Tree' in title else "Branching factor (d)")
            ax.set_title("")

        # Set y-axis label only on the far-left plots for clarity
        if len(axes) > 0: axes[0].set_ylabel("Average Runtime (s)")
        if len(axes) > 2: axes[2].set_ylabel("Average Runtime (s)") # In case there are more than 2 plots

        plt.subplots_adjust(right=0.85)
        g.fig.legend(handles=handles, title='Algorithm', loc='center left', bbox_to_anchor=(0.87, 0.5))
        
        plt.show()

    return df

if RUNTIME_IMPROVED_DP_EXPERIMENT:
    runtime_DP()


# Regret Experiment (multimodal OSSB vs local/classical OSSB)

def run_trials(true_means, graph, m, K, T, strategy, num_trials):
    all_regrets = []
    t_init = time.time()
    for trial in tqdm(range(num_trials)):
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
            
    return np.array(all_regrets)

def plot_results(mmslsqp_regrets, local_regrets, classical_regrets, T, num_trials):
    # Plot regret curves with empirical confidence intervals
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
    plt.plot(mmslsqp_mean, label="Multimodal OSSB", marker='o', markevery=T//10)
    plt.fill_between(
        range(T), mmslsqp_mean - mmslsqp_std, mmslsqp_mean + mmslsqp_std,
        alpha=0.2
    )
    # Local curve
    local_mean = np.mean(local_regrets, axis=0)
    local_std = quantile/np.sqrt(num_trials)*np.std(local_regrets, axis=0)
    plt.plot(local_mean, label="Local search OSSB" ,marker='^', markevery=T//10)
    plt.fill_between(
        range(T), local_mean - local_std, local_mean + local_std,
        alpha=0.2
    )
    
    
    # Classical curve
    classical_mean = np.mean(classical_regrets, axis=0)
    classical_std = quantile/np.sqrt(num_trials)*np.std(classical_regrets, axis=0)
    plt.plot(classical_mean, label="Classical OSSB", marker='s', markevery=T//10)
    plt.fill_between(
        range(T), classical_mean - classical_std,
        classical_mean + classical_std, alpha=0.2
    )
    
    plt.xlabel("Time Step",fontsize=20)
    plt.ylabel("Cumulative Regret",fontsize=20)
    plt.legend(fontsize="15", loc="upper left")
    plt.grid(True)
    plt.show()
    
    
if REGRET_EXPERIMENT:
    num_trials = 50
    K = 7
    # Create a line graph with K nodes
    G = nx.path_graph(K)
    T = 500
    m = 2  # Allow 2 modes
    true_means = generate_multimodal_function(G,[0,6],6,1)
    # mm_regrets = run_trials(true_means, G, m=m, K=K, T=T, 
    #                         strategy="multimodal", num_trials=num_trials)
    mmslsqp_regrets = run_trials(true_means, G, m=m, K=K, T=T, 
                            strategy="multimodal slsqp", num_trials=num_trials)
    local_regrets = run_trials(true_means, G, m=m, K=K, T=T, 
                            strategy="local", num_trials=num_trials)
    classical_regrets = run_trials(true_means, G, m=m, K=K, T=T,
                                  strategy="classical", num_trials=num_trials)
    
    # If an error occured while running the experiment, run_trials still outputs regret history until that error
    actual_trials = min(len(mmslsqp_regrets), len(local_regrets), len(classical_regrets))
    print(f"Plotting results using {actual_trials} completed trials")

    # Plot results using the actual number of completed trials
    plot_results(mmslsqp_regrets[:actual_trials], 
            local_regrets[:actual_trials], 
            classical_regrets[:actual_trials], T, actual_trials)