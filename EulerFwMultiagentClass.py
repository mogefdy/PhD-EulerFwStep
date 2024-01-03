import numpy as np
from matplotlib import pyplot as plt
import importlib
import inspect


class EulerForwardStepper:

    def __init__(self, xmin:float = 0, xmax:float = 10, t_min:float = 0, t_max:float = 1.5, x_num:float = 500, t_res:float = 1, init_cond:np.array = None, stepper:str = "opinion_dynamic", torus:bool = False, torus_ub = 1, phi_val = 2, sigma = 0.1, stochastic=False):
        #assume init_cond is a vector of values. If init condition not given, assume fixed x_num
        
        self.curr_t = t_min
        self.t_max=t_max
        self.torus = torus
        self.torus_ub=torus_ub
        self.timestep = t_res
        self.stochastic = stochastic
        self.step_count = 0
        
        if init_cond:
            self.init_cond = init_cond
            self.curr_cond = init_cond
            self.xmin = init_cond[0]
            self.xmax = init_cond[-1]
        else:
            self.xmin = xmin
            self.xmax = xmax
            self.x_num = x_num
            self.curr_cond = np.linspace(self.xmin, self.xmax, self.x_num)
            self.init_cond = self.curr_cond
        if self.torus:
            self.curr_cond = np.mod(self.curr_cond,self.torus_ub)

        self.trajectories = [self.curr_cond]

        phis = importlib.import_module("phis")
        phi_functions = inspect.getmembers(phis, inspect.isfunction)
        self.phi = phi_functions[phi_val-1][1] #the phis are 1-indexed. This should be solved by converting phi_functions (list of tuples) to a dict.

        self.step_funcs = {"opinion_dynamic":self.opinion_dynamic}
        self.step_func = self.step_funcs[stepper]


        if stochastic:
            self.sigma = sigma
            self.randns = np.random.randn(int((t_max-t_min)/t_res), x_num) * np.sqrt(t_res)
        
    def opinion_dynamic(self, old_state:np.array, dt:float, phi):
        change = np.zeros(len(old_state))
        for i,x in enumerate(old_state):
            for j,y in enumerate(old_state):
                change[i]-= phi(abs(x-y))*(x-y)
        change = change/len(old_state)*dt
        if self.stochastic:
            change+=self.randns[self.step_count]*self.sigma
        new_state = old_state+change
        
        return new_state
    

    def next_step(self):
        self.curr_cond = self.step_func(self.curr_cond, self.timestep, self.phi)
        self.step_count+=1
        if self.torus:
            self.curr_cond = np.mod(self.curr_cond,self.torus_ub)

    def simulate(self):
        while self.curr_t<self.t_max:
            self.curr_t+=self.timestep
            self.next_step()
            self.trajectories.append(self.curr_cond)