import numpy as np
from matplotlib import pyplot as plt

class EulerForwardStepper:

    def __init__(self, xmin:float = 0, xmax:float = 10, t_min:float = 0, t_max:float = 1.5, x_res:float = 500, t_res:float = 1, init_cond:np.array = None, stepper:str = "opinion_dynamic", torus:bool = False):
        #assume init_cond is a vector of values. If init condition not given, assume fixed resolution
        
        self.curr_t = t_min
        self.t_max=t_max
        self.torus = torus
        self.torus_ub=1
        
        self.timestep = t_res
        if init_cond:
            self.init_cond = init_cond
            self.curr_cond = init_cond
            self.xmin = init_cond[0]
            self.xmax = init_cond[-1]
        else:
            self.xmin = xmin
            self.xmax = xmax
            self.resolution = x_res
            self.curr_cond = np.linspace(self.xmin, self.xmax, self.resolution)
            self.init_cond = self.curr_cond

        self.trajectories = [self.curr_cond]
        self.changes = []



        self.step_funcs = {"opinion_dynamic":self.opinion_dynamic}
        self.step_func = self.step_funcs[stepper]
        
    def phi(self, x:float)->float:
        return int(x<=1)
    
    def opinion_dynamic(self, old_state:np.array, dt:float, phi):
        import IPython;IPython.embed
        change = np.zeros(len(old_state))
        for i,x in enumerate(old_state):
            for j,y in enumerate(old_state):
                change[i]-= phi(abs(x-y))*(x-y)
        new_state = old_state+change/len(old_state)*dt
        return new_state
    

    def next_step(self):
        self.curr_cond = self.step_func(self.curr_cond, self.timestep, self.phi)
        if self.torus:
            self.curr_cond = np.mod(self.curr_cond,self.torus_ub)

    def simulate(self):
        while self.curr_t<self.t_max:
            self.curr_t+=self.timestep
            self.next_step()
            self.trajectories.append(self.curr_cond)