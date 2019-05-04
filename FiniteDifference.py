import numpy as np 
import scipy.linalg as linalg
import math
from OptionClass import OptionClass

class FiniteDifferenceMethod(OptionClass):
    
    def __init__(self,spot,strike,maturity,risk_free,volatility,dividend = 0,call = True,early_exercise = False,s_grids = 100,t_grids = 100):
        super().__init__(spot,strike,maturity,risk_free,volatility,dividend,call,early_exercise)
        self.M = s_grids
        self.N = t_grids
        self.smax = max(self.s,self.k) * 2
        self.dt = self.t/self.N
        self.ds = self.smax/self.M
        self.i_values = np.arange(self.M)
        self.j_values = np.arange(self.N)
        self.boundary = np.linspace(0,self.smax,self.M+1)
        self.grids = np.zeros((self.M+1,self.N+1))
        
    def _set_boundary_(self):
        pass
        
    def _set_backward_matrix_(self):
        pass
    
    def _backward_process_(self):
        pass
    
    def _interpolate_(self):
        return np.interp(self.s, self.boundary, self.grids[:,0])
    
    def _compute_(self):
        self._set_boundary_()
        self._set_backward_matrix_()
        self._backward_process_()
    
    def cal_price(self):
        self._compute_()
        self.price = self._interpolate_()
        return self.price
    
    def _delta_(self):
        s_up_value = np.interp(self.s+self.ds,self.boundary,self.grids[:,0])
        s_down_value = np.interp(self.s-self.ds,self.boundary,self.grids[:,0])
        return (s_up_value - s_down_value) / (self.ds*2)
    
    def _gamma_(self):
        s_up_value = np.interp(self.s+self.ds,self.boundary,self.grids[:,0])
        s_down_value = np.interp(self.s-self.ds,self.boundary,self.grids[:,0])
        s_value = self._interpolate_()
        return (s_up_value - 2 * self.price+ s_down_value)/(self.ds**2)
    
    def _theta_(self):
        s_value = self._interpolate_()
        s_right_value = np.interp(self.s,self.boundary,self.grids[:,1])
        return (s_right_value - self.price)/self.dt
    
    def _vega_(self):
        v0 = self.v
        dv = v0 * 0.001
        self.v = v0 + dv 
        p1 = self.cal_price()
        self.v = v0 - dv 
        p2 = self.cal_price()
        self.v = v0
        self._compute_()
        return (p1 - p2)/(2*dv)
    
    def _rho_(self):
        r0 = self.r
        dr = self.r * 0.001
        self.r = r0 + dr 
        p1 = self.cal_price()
        self.r = r0 - dr 
        p2 = self.cal_price()
        self.r = r0
        self._compute_()
        return (p1 - p2)/(2*dr)
    
    def Greeks(self):
        p = self.cal_price()
        return dict({'price':self.price,'delta':self._delta_(),'theta':self._theta_(),'gamma':self._gamma_(),'vega':self._vega_(),'rho':self._rho_()})


class CrankNicolson(FiniteDifferenceMethod):
    def _set_boundary_(self):
        if self.is_call:
            self.grids[:, -1] = np.maximum(self.boundary - self.k, 0)
            self.grids[-1,:] = self.smax - self.k 
        else:
            self.grids[:, -1] = np.maximum(self.k - self.boundary,0)
            self.grids[1,:] = self.k
    
    
    
    def _set_backward_matrix_(self):
    
        self.alpha = 0.25*self.dt*((self.v**2)*(self.i_values**2) -(self.r-self.q)*self.i_values)
        self.beta = -self.dt*0.5*((self.v**2)*(self.i_values**2) +self.r)
        self.gamma = 0.25*self.dt*((self.v**2)*(self.i_values**2) +(self.r-self.q)*self.i_values)
        
        M1 = -np.diag(self.alpha[2:self.M], -1) + np.diag(1-self.beta[1:self.M]) - np.diag(self.gamma[1:self.M-1], 1)
        M2 = np.diag(self.alpha[2:self.M], -1) + np.diag(1+self.beta[1:self.M]) + np.diag(self.gamma[1:self.M-1], 1)
        
        M1[-1,-2:] = M1[-1,-2:] + np.array([self.gamma[-1], -2 * self.gamma[-1]])
        M2[-1,-2:] = M2[-1,-2:] + np.array([-self.gamma[-1], 2 * self.gamma[-1]])
        
        self.M1 = M1
        self.M2 = M2

    def _backward_process_(self):
        
        for j in reversed(range(self.N)):
            P, L, U = linalg.lu(self.M1)
            x1 = linalg.solve(L,np.dot(self.M2,self.grids[1:-1, j+1]))
            x2 = linalg.solve(U, x1)
            
            if not self.early_exercise:
                self.grids[1:-1, j] = x2
            else:
                self.grids[1:-1, j] = np.maximum(x2,self.grids[1:-1,-1])


class BarrierOption(CrankNicolson):

    def __init__(self,spot,strike,maturity,risk_free,volatility,barrier,dividend = 0,call = True,early_exercise = False,s_grids = 100,t_grids = 100,knock_out = True):
        super().__init__(spot,strike,maturity,risk_free,volatility,dividend,call ,early_exercise ,s_grids ,t_grids)
        self.H = barrier
        self.knock_out = knock_out
        self.down = (self.H < self.s)
        self.nH = int(self.H/self.ds)
        
    def _set_boundary_(self):
        if self.is_call:
            self.grids[:, -1] = np.maximum(self.boundary - self.k, 0)
            self.grids[-1,:] = self.smax - self.k 
        else:
            self.grids[:, -1] = np.maximum(self.k - self.boundary,0)
            self.grids[1,:] = self.k
        
        if self.down:
            self.grids[:self.nH+1,-1] = 0
        else:
            self.grids[self.nH:,-1] = 0
        
        
    def _backward_process_(self):
        M = np.linalg.inv(self.M1).dot(self.M2)
          # Barrier Option values
        
        if self.knock_out:
            for j in reversed(range(self.N)):
                xb = M.dot(self.grids[1:-1,j+1]) # Value for trace Back
                if not self.early_exercise:
                    self.grids[1:-1, j] = xb
                else:
                    self.grids[1:-1, j] = np.maximum(xb,self.grids[1:-1,-1])
                
                if self.down:
                        self.grids[:self.nH+1,j] = 0
                if not self.down:
                        self.grids[self.nH:,j] = 0
 
        else:  
            # Set An Vanilla Option 
            grids1 = np.zeros((self.M+1,self.N+1))
            if self.is_call:
                grids1[:, -1] = np.maximum(self.boundary - self.k, 0)
                grids1[-1,:] = self.smax - self.k 
            else:
                grids1[:, -1] = np.maximum(self.k - self.boundary,0)
                grids1[1,:] = self.k            
            
            for j in reversed(range(self.N)):
                xb = M.dot(self.grids[1:-1,j+1])
                x1 = M.dot(grids1[1:-1,j+1])

                self.grids[1:-1, j] = xb
                grids1[1:-1,j] = x1
            
                if self.down:
                    self.grids[:self.nH+1,j] = 0
                if not self.down:
                    self.grids[self.nH:,j] = 0      
            self.grids = grids1 - self.grids   
            