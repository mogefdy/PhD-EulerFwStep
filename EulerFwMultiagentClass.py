import numpy as np
from matplotlib import pyplot as plt
import importlib
import inspect
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
import cluster_helpers


class EulerForwardStepper:

    def __init__(self, xmin:float = 0, xmax:float = 10, t_min:float = 0, t_max:float = 1.5, x_num:float = 500, t_res:float = 1, init_cond:np.array = None, stepper:str = "opinion_dynamic", torus:bool = False, torus_ub = 1, phi_val = 2, sigma:float = 0.1, stochastic:bool=False):
        #assume init_cond is a vector of values. If init condition not given, assume fixed x_num
        
        self.curr_t = t_min
        self.t_max=t_max
        self.timestep = t_res
        self.stochastic = stochastic
        self.step_count = 0
        self.torus = torus
        if torus_ub:
            self.torus_ub=torus_ub
        else:
            self.torus_ub=xmax
        
        if init_cond:
            self.init_cond = init_cond
            self.curr_cond = init_cond
            self.xmin = init_cond[0]
            self.xmax = init_cond[-1]
            self.x_num = len(self.init_cond)
        else:
            self.xmin = xmin
            self.xmax = xmax
            self.x_num = x_num
            self.curr_cond = np.linspace(self.xmin, self.xmax, self.x_num)
            self.init_cond = self.curr_cond
        if self.torus:
            self.curr_cond = np.mod(self.curr_cond,self.torus_ub)

        self.trajectories = [self.curr_cond]
        self.rolling_n_clusters = []
        self.rolling_n_noise = []
        self.rolling_labels = []

        phis = importlib.import_module("phis")
        phi_functions = inspect.getmembers(phis, inspect.isfunction)
        self.phi = phi_functions[phi_val-1][1] #the phis are 1-indexed. This could be solved by using a dict.
        self.phi = np.vectorize(self.phi)

        self.step_funcs = {"opinion_dynamic":self.opinion_dynamic, "rk4":self.rk4_step, "heun":self.heun_step}
        self.step_func = self.step_funcs[stepper]


        if stochastic:
            self.sigma = sigma
            self.randns = np.random.randn(int((t_max-t_min)/t_res)+1, x_num) * np.sqrt(t_res) #+1 for Special case when t_res divides t_max-t_min as loop condition is non-strict.
        
    def opinion_dynamic(self, old_state:np.array, dt:float, phi):
        all_diffs = old_state - old_state.reshape(-1,1)
        change = -(phi(abs(all_diffs)) *all_diffs).sum(axis=0)
        change = change/len(old_state)*dt
        if self.stochastic:
            change+=self.randns[self.step_count]*self.sigma
        new_state = old_state+change
        return new_state


    def milstein_step(self, old_state:np.array, dt:float, phi):
        assert self.stochastic, "Milstein is applicable only to SDEs"
        change = np.zeros(len(old_state))
        for i,x in enumerate(old_state):
            for j,y in enumerate(old_state):
                change[i]-= phi(abs(x-y))*(x-y)
        change = change/len(old_state)*dt
        change+=self.randns[self.step_count]*self.sigma_func(old_state)
        change+=self.sigma_func(old_state)*self.d_sigma_func(old_state)/2*(self.randns[self.step_count]**2-dt)
        new_state = old_state+change
        
        return new_state
    
    def heun_step(self, old_state:np.array, dt:float, phi):
        assert not self.stochastic, "Heun's method is not applicable to SDEs"
        mid_est = self.opinion_dynamic(old_state, dt, phi)
        mid_change_1 = mid_est-old_state
        mid_change_2 = self.opinion_dynamic(mid_est, dt, phi)-old_state

        new_state = old_state+(mid_change_1+mid_change_2)/2
        
        return new_state
    

    def rk4_step(self, old_state:np.array, dt:float, phi):
        assert not self.stochastic, "Runge-Kutta is not applicable to SDEs"

        est_1 = self.opinion_dynamic(old_state, dt, phi)
        mid_change_1 = est_1-old_state
        est_2 = self.opinion_dynamic(old_state+mid_change_1/2, dt, phi)
        mid_change_2 = est_2-old_state
        est_3 = self.opinion_dynamic(old_state+mid_change_2/2, dt, phi)
        mid_change_3 = est_3-old_state
        est_4 = self.opinion_dynamic(est_3, dt, phi)
        mid_change_4 = est_4-old_state

        new_state = old_state+1/6*(mid_change_1+2*mid_change_2+2*mid_change_3+mid_change_4)
        
        return new_state
    

    def next_step(self):
        self.curr_cond = self.step_func(self.curr_cond, self.timestep, self.phi)
        self.step_count+=1
        if self.torus:
            self.curr_cond = np.mod(self.curr_cond,self.torus_ub)

    def simulate(self):
        while self.curr_t<=self.t_max:
            self.curr_t+=self.timestep
            self.next_step()
            self.trajectories.append(self.curr_cond)

    def clusters_each_step(self, eps = 0.5, min_samples = 10, algo = "DBSCAN", max_clusters = 10, max_iter=300, tol = 1e-4): #Only DBSCAN will generate n_noise. 
        self.rolling_n_clusters = []
        self.rolling_labels = []
        if algo=="DBSCAN":
            self.rolling_n_noise = []

            for t in self.trajectories:
                (labels, n_clusters_, n_noise_) = cluster_helpers.get_DBSCAN_cluster_info(state = t, eps = eps, min_samples = min_samples)

                self.rolling_n_clusters.append(n_clusters_)
                self.rolling_n_noise.append(n_noise_)
                self.rolling_labels.append(labels)

        elif algo=="KMeans":
            for t in self.trajectories:
                (labels, n_clusters) = cluster_helpers.get_KMeans_cluster_info(state = t, max_clusters = max_clusters, tol = tol)

                self.rolling_labels.append(labels)
                self.rolling_n_clusters.append(n_clusters)
        else: 
            print("No accepted algorithm was given. Please use either \"DBSCAN\" or \"KMeans\".")


    def get_cluster(self, cluster_no = 0):
        try:
            mask = (np.stack(self.rolling_labels) == cluster_no)
        except ValueError:
            print("The labels attribute is currently empty. Please run clusters_each_step() first to generate cluster labels")
        else:
            return mask.nonzero() #returns 2 arrays of x and y coordinates of all indices of an element which is a member of the given cluster. Note this 
    #a. does not account for the number of clusters, and thus the label of each cluster, changing, 
    #b. gives results in the form (timestep, agent number)

    def get_com(self, cluster_no = 0): #need to run after clusters_each_step
        coms = []
        mask = self.get_cluster(cluster_no)
        for time, pos_vec in enumerate(self.trajectories):
            com = np.mean(pos_vec[mask[1][(mask[0]==time).nonzero()]]) #(mask[0]==time).nonzero() is a vector of the position in the mask which correspond to the time being the given time.
            coms.append(com)
        return coms


    def stop_at_next_clustering(self, max_future = 200, algo = "DBSCAN", eps = 0.5, min_samples = 5, max_clusters = 5, max_iter=300, tol = 1e-4):
        init_t = self.curr_t
        if algo == "DBSCAN":
            init_clusters = cluster_helpers.get_DBSCAN_cluster_info(self.curr_cond, eps = eps, min_samples = min_samples)[1]
        elif algo == "KMeans":
            init_clusters = cluster_helpers.get_KMeans_cluster_info(self.curr_cond, max_clusters = max_clusters, max_iter = max_iter, tol = tol)[1]
        curr_clusters = init_clusters

        while curr_clusters == init_clusters and self.curr_t<init_t + max_future:
            self.curr_t+=self.timestep
            self.next_step()
            if algo=="DBSCAN":
                curr_clusters =  cluster_helpers.get_DBSCAN_cluster_info(self.curr_cond, eps = eps, min_samples = min_samples)[1]
            elif algo == "KMeans":
                curr_clusters =  cluster_helpers.get_KMeans_cluster_info(self.curr_cond, max_clusters = max_clusters, max_iter = max_iter, tol = tol)[1]
            #self.next_step()
            self.trajectories.append(self.curr_cond)
        print(init_clusters)
        print(curr_clusters)