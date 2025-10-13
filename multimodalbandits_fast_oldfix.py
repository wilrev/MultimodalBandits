import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds
from scipy import stats
import time

DEBUG = False
RUN_RUNTIME_EXPERIMENT = False
RUN_REGRET_EXPERIMENT = False



def fast_dynamic_programming(G,mu,eta,N,nb_modes):
# Find the minimal value of the weighted divergence with m modes discretization number N
    # Set of modes of mu
    M = compute_modes(G,mu)
    # Number of nodes
    K = mu.shape[0]
    # Best node
    kstar = np.argmax(mu)
# Case number 1: when the maximal entry of confusing parameter lambda does lie in the neighbourhood of a mode of mu, or when the number of modes of lambda is strictly greater than that of mu)
    # Computes explicitly the solution of PGL(k) when k is in the neighborhood of a mode or mu has strictly less than m modes, performs dynamic programming for the other k's
    lambdastar1 = np.ones(K)*np.max(mu)
    vstar1 = sum(eta*divergence(mu,lambdastar1)) 
    if nb_modes > nb_modes: #if mu is strictly less than m-modal, we have no constraints besides lambda[k]=lambda[kstar] for k different than kstar
         for k in [k_val for k_val in range(K) if k_val != kstar]:
             lambdastar_new=np.copy(mu)
             lambdastar_new[k]=mu[kstar]
             if (vstar1 > sum(eta*divergence(mu,lambdastar_new))):
                 lambdastar1 = lambdastar_new
                 vstar1 = sum(eta*divergence(mu,lambdastar1))
    else:
        neighborhood=modes_neighborhood(G,mu)
        for k in [k_val for k_val in neighborhood if k_val != kstar]:
            lambdastar_new=np.copy(mu)
            lambdastar_new[k]=mu[kstar]
            if (vstar1 > sum(eta*divergence(mu,lambdastar_new))):
                lambdastar1 = lambdastar_new
                vstar1 = sum(eta*divergence(mu,lambdastar1))
# Case number 2: when the maximal entry of confusing parameter lambda does not lie in the neighbourhood of a mode of mu
    # Discretize the space of lambda with a grid of size N
    grid = np.linspace(np.min(mu),np.max(mu),N)
    # write the tree as rooted at k
    T = nx.bfs_tree(G, kstar)
    # Initialize the values of h(ell,z,a,b,c),hminus(ell,z,b,c),hequal(ell,z,b,c),hplus(ell,z,b,c),hstar(ell,z,b,c) 
    h = np.zeros([K,len(grid),3,2,2])
    hplus = np.zeros([K,len(grid),2,2])
    hminus = np.zeros([K,len(grid),2,2])
    hequal = np.zeros([K,len(grid),2,2])
    hstar = np.zeros([K,len(grid),2,2])
    hplusequal = np.zeros([K,len(grid),2,2])
    hdelta = np.zeros([K,len(grid),2,2])
    # Loop over the nodes sorted by decreasing depth to compute the f values
    for ell in reversed(list(T.nodes())):
        #to store the deltas
        g = np.zeros([np.size(T[ell]),2,2])
        # Compute the values of h(ell,z,a,b,c) using the values of descendents
        if not(T[ell]): # case 1: ell is a leaf, in this case the value is computable by hand
            for a in range(3): 
                for b in range(2): 
                    for c in range(2): 
                        is_max = ((a == 0) and (b==1) and (ell not in M)) 
                        valid = (a != 1) and (b == (a==0)*(ell not in M) ) and (c == (a!=0)*(ell in M) )
                        for i,z in enumerate(grid): 
                            if (not valid) or (is_max and (i != N-1)):
                                h[ ell,i,a,b,c ] = 10 ** 10
                            else:
                                h[ ell,i,a,b,c ] = eta[ell]*divergence(mu[ell],z)
        else: #case 2: ell is an internal node 
            #first compute the value of hdelta
            for i,z in enumerate(grid):
                for b in range(2):
                    for c in range(2):
                        hdelta[ell,i,b,c] = np.min(hplusequal[T[ell],i,b,c] - hstar[T[ell],i,b,c])/np.size(T[ell])
                for a in range(3): 
                    for b in range(2): 
                        for c in range(2): 
                            is_max = ((a == 0) and (b==1) and (ell not in M)) 
                            div = 10 ** 10 if is_max and (i != N-1) else eta[ell]*divergence(mu[ell],z)
                            if (a==0):
                                h[ell,i,a,b,c] = div + fast_minimization(hminus[T[ell],i,:,:],b -1*(ell not in M),c)[0]
                            elif (a==1): 
                                h[ell,i,a,b,c] = div + fast_minimization(hstar[T[ell],i,:,:] + hdelta[ell,i,:,:],b,c-1*(ell in M))[0]
                            elif (a==2): 
                                h[ell,i,a,b,c] = div + fast_minimization(hstar[T[ell],i,:,:],b,c-1*(ell in M))[0] 
        # Compute the values of hminus(ell,z,b,c),hequal(ell,z,b,c),hplus(ell,z,b,c),hstar(ell,z,b,c) by appropriate minimization
        for b in range(2):
            for c in range(2):
                hplus[ell,-1,b,c] = 10 ** 10 
                hminus[ell,0,b,c] = 10 ** 10 
                for i in range(1,N): hplus[ell,N-1-i,b,c] = min(hplus[ell,N-i,b,c], min( h[ell,N-i,0,b,c],h[ell,N-i,1,b,c])) 
                for i in range(1,N): hminus[ell,i,b,c] = min(hminus[ell,i-1,b,c], h[ell,i-1,2,b,c]) 
                for i in range(N): hequal[ell,i,b,c] = h[ell,i,2,b,c]
                for i in range(N): hstar[ell,i,b,c] =  min( hminus[ell,i,b,c]  , hequal[ell,i,b,c] , hplus[ell,i,b,c] )
                for i in range(N): hplusequal[ell,i,b,c] =  min( hequal[ell,i,b,c] , hplus[ell,i,b,c] )
    
    #initialize the values of the nodes flag
    istar = [0 for i in range(K)] 
    astar = [0 for i in range(K)]
    bstar = [0 for i in range(K)]
    cstar = [0 for i in range(K)]
    #compute the flags for the root node
    istar[kstar] = N-1
    bstar[kstar] = 1
    cstar[kstar] = 1
    astar[kstar] = 1 if (h[kstar,istar[kstar],1,bstar[kstar],cstar[kstar]] <  h[ kstar,istar[kstar],0,bstar[kstar],cstar[kstar] ] ) else 0
    #compute the optimal solution using the values of h
    for ell in list(T.nodes()):
        if (T[ell]): #if node ell is not a leaf we find the value of the flags of its successors
            if (astar[ell] == 0):
                 (vs,bs,cs) = fast_minimization(hminus[T[ell],i,:,:],bstar[ell] -1*(ell not in M),cstar[ell])
                 for i,v in enumerate(T[ell]):
                    bstar[v] = bs[i]
                    cstar[v] = cs[i]
                    istar[v] = np.argmin(h[ v,:,2,bstar[v],cstar[v]]  +  (10 * 10)*(grid >=  grid[istar[ell]]))
            elif (astar[ell] == 1):
                 (vs,bs,cs) = fast_minimization(hminus[T[ell],i,:,:],bstar[ell] ,cstar[ell]-1*(ell in M))
                 wi = np.argmin([hplus[ w,:,bstar[w],cstar[w] ] - hstar[ w,:,bstar[w],cstar[w] ] for w in T[ell]])
                 for i,v in enumerate(T[ell]):
                    bstar[v] = bs[i]
                    cstar[v] = cs[i]
                    if (i == wi): 
                        if (hplus[ v,istar[ell],bstar[v],cstar[v] ] ==  hplusequal[ v,istar[ell],bstar[v],cstar[v] ]  ):
                            istar[v] = np.argmin(  np.minimum(  h[ v,:,0,bstar[v],cstar[v] ], h[ v,:,1,bstar[v],cstar[v] ])  +  (10 * 10)*(grid <=  grid[istar[ell]]))
                        else:
                            istar[v] = istar[ell]
                    else:
                        if (hplus[ v,istar[ell],bstar[v],cstar[v] ] ==  hstar[ v,istar[ell],bstar[v],cstar[v] ]  ):
                            istar[v] = np.argmin(  np.minimum(h[ v,:,0,bstar[v],cstar[v] ], h[ v,:,1,bstar[v],cstar[v] ])  +  (10 * 10)*(grid <=  grid[istar[ell]]))
                        elif (hminus[ v,istar[ell],bstar[v],cstar[v] ] ==  hstar[ v,istar[ell],bstar[v],cstar[v] ]  ):
                            istar[v] = np.argmin(h[ v,:,2,bstar[v],cstar[v]]  +  (10 * 10)*(grid >=  grid[istar[ell]]))
                        else:
                            istar[v] = istar[ell]
            elif (astar[ell] == 2):
                 (vs,bs,cs) = fast_minimization(hminus[T[ell],i,:,:],bstar[ell] ,cstar[ell]-1*(ell in M))
                 for i,v in enumerate(T[ell]):
                    bstar[v] = bs[i]
                    cstar[v] = cs[i]
                    if (hplus[ v,istar[ell],bstar[v],cstar[v] ] ==  hstar[ v,istar[ell],bstar[v],cstar[v] ]  ):
                        istar[v] = np.argmin(  np.minimum(h[ v,:,0,bstar[v],cstar[v] ], h[ v,:,1,bstar[v],cstar[v] ])  +  (10 * 10)*(grid <=  grid[istar[ell]]))
                    elif (hminus[ v,istar[ell],bstar[v],cstar[v] ] ==  hstar[ v,istar[ell],bstar[v],cstar[v] ]  ):
                        istar[v] = np.argmin(h[ v,:,2,bstar[v],cstar[v]]  +  (10 * 10)*(grid >=  grid[istar[ell]]))
                    else:
                        istar[v] = istar[ell]

            for i,v in enumerate(T[ell]):
                if (istar[v] <= istar[ell]):
                    astar[v] = 2
                elif (h[ v,istar[v],2,bstar[v],cstar[v]] <  h[ v,istar[v],0,bstar[v],cstar[v] ] ):
                    astar[v] = 1
                else:
                    astar[v] = 0
    lambdastar2 = np.array( [grid[istar[i]] for i in range(K) ])
    vstar2 =  np.sum(eta*divergence(mu,lambdastar2))
#Choose the case with the lowest value between case 1 and case 2 and return the corresponding minimizer
    if (vstar1 < vstar2):
        vstar = vstar1
        lambdastar = lambdastar1
    else:
        vstar = vstar2
        lambdastar = lambdastar2

    # Debug information
    if DEBUG:    
        for ell in reversed(list(T.nodes())):
            print("\n **************************** \n")
            print("Node",ell)
            print("Successors",T[ell])
            print("Is A Mode", ell in M)
            print("Is A Leaf", (not T[ell]))
            print("Flags (z,a,b,c) = ",grid[istar[ell]],astar[ell],bstar[ell], cstar[ell])
            for a in range(3): 
                for b in range(2): 
                    for c in range(2): 
                        print("h(.,",a,",",b,",",c,"), = ", np.round(h[ell,:,a,b,c],3))
            for b in range(2): 
                for c in range(2): 
                    print("hplus(.,",b,",",c,"), = ", np.round(hplus[ell,:,b,c],3))
            for b in range(2): 
                for c in range(2): 
                    print("hminus(.,",b,",",c,"), = ", np.round(hminus[ell,:,b,c],3))
            for b in range(2): 
                for c in range(2): 
                    print("hequal(.,",b,",",c,"), = ", np.round(hequal[ell,:,b,c],3))
            for b in range(2): 
                for c in range(2): 
                    print("hstar(.,",b,",",c,"), = ", np.round(hstar[ell,:,b,c],3))
        print("Modes",M)
        print("Grid:",np.round(grid,3))
        print("Mu vector",np.round(mu,3))
        print("Eta vector",np.round(eta,3))
    return( (lambdastar,vstar) ) 





def fast_minimization(g,b,c):
# Compute the minimization of sum_{v \in C(\ell)} g(b_v,c_v) across all ({\bf b},{\bf c}) \in B_ell(b,c) \times C_ell(b,c) using "fast minimization"
    vstar = 10 ** 10
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
            if D11 < D1001:
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


def slow_minimization(g,b,c):
    vstar = 10 ** 10
    for i in range(n):
        for j in range(n): 
            bvec = [b if i == l else 0 for l in range(n) ] 
            cvec = [c if j == l else 0 for l in range(n) ] 
            s = sum([ g[l,bvec[l],cvec[l]] for l in range(n) ] )
            if (s < vstar):
                bstar = bvec
                cstar = cvec
                vstar = s
    return(vstar,bstar,cstar)

##test fast minimization 
#n = 50
#k = 5
#b = 1
#c = 1
#g = np.random.randint(0,k,size=(n,2,2)) 
#start_slow = time.time()
#(vstar,bstar,cstar) = slow_minimization(g,b,c)
#end_slow = time.time()
#start_fast = time.time()
#(vstar2,bstar2,cstar2) = fast_minimization(g,b,c)
#end_fast = time.time()
#print("Minimal value (slow)", (vstar,bstar,cstar), " Computing time", end_slow - start_slow)
#print("Minimal value (fast)", (vstar2,bstar2,cstar2), " Computing time", end_fast - start_fast)
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
    v = np.full([K, N], -1, dtype=int) # v[ell, i] will store the identity of the child that minimizes g_j(z_i) for parent ell
    lambdastar = np.zeros(mu.shape[0])
    # Loop over the nodes sorted by decreasing depth to compute the f values
    for ell in reversed(list(nx.topological_sort(T))): #previously reversed(list(T.nodes())), could also use list(nx.dfs_postorder_nodes(T, source=k))
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
                    f[ell,i,1] = 10 ** 10
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


# older version with more minimal fix (not storing the winning children in v)

# def regression_graph(G,mu,eta,p,k,N):
# # Find the minimizer of the weighted divergence with set of peaks p and maximizer k on a line graph, discretization number N
#     # Number of nodes
#     K = mu.shape[0]
#     # Best node
#     kstar = np.argmax(mu)
#     # Discretize the space of lambda with a grid of size N
#     grid = np.linspace(np.min(mu),np.max(mu),N)
#     # write the tree as rooted at k
#     T = nx.bfs_tree(G, k)
#     # Initialize the values of f,fstar,fsquare, lambdastar
#     f = np.zeros([K,len(grid),2])
#     fstar = np.zeros([K,N,2]) #(-1,+1)
#     fsquare = np.zeros([K,N])
#     v = np.zeros([K,N])
#     lambdastar = np.zeros(mu.shape[0])
#     # Loop over the nodes sorted by decreasing depth to compute the f values
#     for ell in reversed(list(T.nodes())):
#         # If ell is the maximizer of mu, then eta = +infty
#         e = 10 ** 10 if (ell == kstar) else eta[ell] 
#         # Compute the value of f
#         for i,z in enumerate(grid): # f[ell,i,0]=f_ell(grid[i]=z,-1), f[ell,i,1]=f_ell(grid[i]=z,1)
#             if ell in p: # ell can be a mode
#                 f[ell,i,0] = e*divergence(mu[ell],z) 
#                 f[ell,i,1] = e*divergence(mu[ell],z)
#                 for j in T.successors(ell): 
#                     f[ell,i,1] += fsquare[j,i]
#                     f[ell,i,0] += fsquare[j,i]
#             else: # ell cannot be a mode, and needs to have a neighbour whose reward is higher
#                 f[ell,i,0] = e*divergence(mu[ell],z) 
#                 for j in T.successors(ell): f[ell,i,0] += fsquare[j,i]
#                 children = list(T.successors(ell)) 
#                 if children:
#                     f[ell,i,1] = e*divergence(mu[ell],z) + sum([fsquare[j,i] for j in children]) + min([min(fstar[j,i,1],f[j,i,0]) - fsquare[j,i] for j in children])
#                 else:
#                     f[ell,i,1] = 10 ** 10
#         # Compute the value of fstar and fsquare
#         fstar[ell,0,0] = f[ell,0,0]
#         for i in range(1,N): fstar[ell,i,0] = min(fstar[ell,i-1,0],f[ell,i,0]) #min_{w \leq mu_*} [...] = fstar[ell,N-1,0]=min_{i=0,...,N-1} f[ell,i,0]
#         fstar[ell,N-1,1] = 10**10
#         for i in range(1,N): fstar[ell,N-1-i,1] = min(fstar[ell,N-i,1],f[ell,N-i,1])
#         for i in range(N): fsquare[ell,i] = min(fstar[ell,i,0],fstar[ell,i,1])
#     lambdastar[k] = max(mu)
#     for ell in list(T.nodes()):
#         if ell == k:
#             continue
#         # Find parent
#         parent = next(T.predecessors(ell))        
#         parent_grid_index = np.where(grid == lambdastar[parent])[0][0]
        
#         # Check if the value of ell is constrained by the past
#         grandparent_list = list(T.predecessors(parent))
#         ell_is_constrained = False
#         if grandparent_list :
#             grandparent = grandparent_list[0]
#             if lambdastar[parent] > lambdastar[grandparent] and parent not in p:
#                 child_terms = [
#                     min(fstar[j, parent_grid_index, 1], f[j, parent_grid_index, 0]) - fsquare[j, parent_grid_index]
#                     for j in T.successors(parent)
#                 ]
#                 min_child_index = np.argmin(child_terms)
#                 constrained_child = list(T.successors(parent))[min_child_index]
                
#                 if ell == constrained_child:
#                     ell_is_constrained = True
        
#         # Apply appropriate formula if ell is constrained
#         if ell_is_constrained:
#             # Ensure λ_ℓ is strictly greater than parent's λ
#             if fstar[ell, parent_grid_index, 1] <= fstar[ell, parent_grid_index, 0]:
#                 lambdastar[ell] = grid[parent_grid_index + 1 + np.argmin(f[ell, parent_grid_index+1:, 1])]
#             else:
#                 lambdastar[ell] = lambdastar[parent]
#         else:
#             # No constraint on ell
#             if fstar[ell, parent_grid_index, 1] <= fstar[ell, parent_grid_index, 0]:
#                 lambdastar[ell] = grid[parent_grid_index + 1 + np.argmin(f[ell, parent_grid_index+1:, 1])]
#             else:
#                 lambdastar[ell] = grid[np.argmin(f[ell, :parent_grid_index+1, 0])]
#     # Debug information
#     if DEBUG:    
#         for ell in range(K):
#             print("Node",ell)
#             print("Can Be Mode", ell in p)
#             print("f minus ",np.round(f[ell,:,0],3))
#             print("f plus",np.round(f[ell,:,1],3))
#             print("f star minus ",np.round(fstar[ell,:,0],3))
#             print("f star plus",np.round(fstar[ell,:,1],3))
#             print("f square",np.round(fsquare[ell,:],3))
#             print("v", np.round(v[ell,:],3))
#         print("Modes",p)
#         print("Grid:",np.round(grid,3))
#         print("Mu vector",np.round(mu,3))
#         print("Eta vector",np.round(eta,3))
#         print("Optimal Solution ",np.round(lambdastar,3))
#         print("Optimal Value",np.round(fsquare[k,N-1],3))
#         check_modes(G,lambdastar,p)
#     return(lambdastar)
    
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
                         if vstar>sum(eta*divergence(mu,lambda_new)): # If this is not the case, it is unecessary to go further
                             lambdastar_new = regression_graph(G,mu,eta,p,k,N)
                             if (vstar > sum(eta*divergence(mu,lambdastar_new))):
                                 lambdastar = lambdastar_new
                                 vstar = sum(eta*divergence(mu,lambdastar_new))
     vstar = sum(eta*divergence(mu,lambdastar))  
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
        #axs[i].set_title(f'N = {N}')
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
    t_init = time.time()
    try:
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
            if trial%(num_trials/100) == 0 and strategy == "multimodal slsqp":
                print("percentage done:",trial/num_trials,'time elapsed:',time.time()-t_init)
    
    except Exception as e:
        print(f"Error occurred during trial {len(all_regrets)}: {str(e)}")
        if len(all_regrets) > 0:
            print(f"Returning partial results from {len(all_regrets)} completed trials")
            return np.array(all_regrets)
        else:
            raise e  # Re-raise if no trials completed
            
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

#for i in [3,6,14,16]:
for i in [6]: # previous counter examples: seed=6 with [0,6] peaks and K=10, or (simpler) seed=14 with [0,5] peaks and K=7
    np.random.seed(i)
    print("Seed",i)
    K = 10    # Create a line graph with K nodes
    G = nx.path_graph(K)
    nb_modes = 2  # Allow 2 modes
    mu = generate_multimodal_function(G,[0,6],6,1)
    N = 100
    eta = np.random.rand(K)
    #mu=np.flip(mu) 
    #eta=np.flip(eta)
    #prior code worked when we flipped mu and eta, indicating an issue in the implementation
    #test new method vs old method
    start_new = time.time()
    lambdastar, vstar = fast_dynamic_programming(G,mu,eta,N,nb_modes)
    end_new = time.time()
    start_old = time.time()
    lambdastarold,vstarold=regression_all(G,mu,eta,N,nb_modes)
    end_old = time.time()
    if 1:
        print("Mu function", np.round(mu,3))
        print("New strategy vs Old strategy")
        print("Optimal solution", np.round(lambdastar,3))
        print("Optimal solution", np.round(lambdastarold,3))
        print("Value", vstar) 
        print("Value", vstarold) 
        if 0:
            print("Computing time", end_new - start_new)
            print("Computing time", end_old - start_old)
   

