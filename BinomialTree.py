import numpy as np 
import math 
from OptionClass import OptionClass
import BlackScholes as BS


class BinomialTreeMethod(OptionClass):
    def __init__(self,spot,strike,maturity,risk_free,volatility,dividend = 0,nsteps = 252,call = True, early_exercise = False):
        super().__init__(spot,strike,maturity,risk_free,volatility,dividend,call, early_exercise)
        self.N = nsteps
        self.dt = self.t/self.N
        self.D = math.exp(-self.r*self.dt)
        self.u = math.exp(self.v*self.dt**0.5)
        self.d = 1/self.u
        self.pu = (math.exp((self.r-self.q)*self.dt)-self.d)/(self.u-self.d)
        self.pd = 1-self.pu
        self.op_tree = np.zeros((self.N+1,self.N+1))
        
    def _initAttr_(self):
        self.dt = self.t/self.N
        self.D = math.exp(-self.r*self.dt)
        self.u = math.exp(self.v*self.dt**0.5)
        self.d = 1/self.u
        self.pu = (math.exp((self.r-self.q)*self.dt)-self.d)/(self.u-self.d)
        self.pd = 1-self.pu
        self.op_tree = np.zeros((self.N+1,self.N+1))

    def _spot_tree_(self):
        mu = mu = np.arange(self.N+1)
        mu = np.resize(mu,(self.N+1,self.N+1))
        md = np.transpose(mu)
        
        self.exp_tree = mu - 2*md
        self.sp_tree = self.s * self.u**self.exp_tree

    def _op_tree_(self):
        pass
    
    def _backward_process_(self):
        pass
    
    def _compute_(self):
        self._initAttr_()
        self._spot_tree_()
        self._op_tree_()
        self._backward_process_()
        self.price = self.op_tree[0,0]
        
    def cal_price(self):
        self._compute_()
        return self.price
    
    def _delta_(self):
        return (self.op_tree[0,1] - self.op_tree[1,1])/(self.sp_tree[0,1] - self.sp_tree[1,1])
        
    def _gamma_(self):
        return ((self.op_tree[0,2] - self.op_tree[1,2])/(self.sp_tree[0,2]-self.sp_tree[1,2])-(self.op_tree[1,2] - self.op_tree[2,2])/(self.sp_tree[1,2] - self.sp_tree[2,2]))/(self.sp_tree[0,2] - self.sp_tree[2,2])*2

    def _theta_(self):    
        return (self.op_tree[1,2] - self.op_tree[0,0])/(2*self.dt)
    
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

    def Greeks(self):
        return dict({'price':self.cal_price(),'delta':self._delta_(),'theta':self._theta_(),'gamma':self._gamma_(),'vega':self._vega_(),'rho':self._rho_()})



class VanillaOption(BinomialTreeMethod):
    
    def _op_tree_(self):
        self.op_tree = np.maximum(self.sp_tree - self.k ,0) if self.is_call else np.maximum(self.k - self.sp_tree ,0)

    def _backward_process_(self):
        for j in range(self.N,0,-1):
            if not self.early_exercise:
                self.op_tree[:j,j-1] = self.D * (self.pu * self.op_tree[:j,j] + self.pd * self.op_tree[1:(j+1),j])
            else:
                self.op_tree[:j,j-1] = self.D * np.maximum(self.pu * self.op_tree[:j,j] + self.pd * self.op_tree[1:(j+1),j],self.op_tree[:j,j-1])



class ShoutOption(VanillaOption):
    
    def _op_tree_(self):
        for i in range(0,self.N):
            self.op_tree[:,i] = BS.EuropeanOption(self.sp_tree[:,i],self.sp_tree[:,i],(self.t - i*self.dt),self.r,self.v,self.q,self.is_call).cal_price()
        self.op_tree[:,-1] = np.maximum(self.sp_tree[:,-1] - self.k,0) if self.is_call \
                            else np.maximum(self.k - self.sp_tree[:,-1],0)
    
    def _backward_process_(self):
        for j in range(self.N,0,-1):
            self.op_tree[:j,j-1] = self.D * np.maximum(self.pu * self.op_tree[:j,j] + self.pd * self.op_tree[1:(j+1),j],self.op_tree[:j,j-1])



class BarrierOption(BinomialTreeMethod):
    def __init__(self,spot,strike,maturity,risk_free,volatility,barrier_up = None ,barrier_down = None,dividend = 0,nsteps = 252,call = True, early_exercise = False):
        super().__init__(spot,strike,maturity,risk_free,volatility,dividend,nsteps,call, early_exercise)
        self.Hu = barrier_up 
        self.Hd = barrier_down
        
    

    def _initAttr_(self):
        self.dt = self.t/self.N
        self.D = math.exp(-self.r*self.dt)
        self.u = math.exp(self.v*self.dt**0.5)
        self.d = 1/self.u
        self.pu = (math.exp((self.r-self.q)*self.dt)-self.d)/(self.u-self.d)
        self.pd = 1-self.pu
        self.op_tree_in = np.zeros((self.N+1,self.N+1))
        self.op_tree_out = np.zeros((self.N+1,self.N+1))
        self.in_Hu = np.floor(np.log(self.Hu/self.s)/np.log(self.u)) if self.Hu is not None else None
        self.out_Hu = np.ceil(np.log(self.Hu/self.s)/np.log(self.u))  if self.Hu is not None else None
        self.in_Hd = np.ceil(np.log(self.Hd/self.s)/np.log(self.u))   if self.Hd is not None else None
        self.out_Hd = np.floor(np.log(self.Hd/self.s)/np.log(self.u))  if self.Hd is not None else None




    
    def _op_tree_(self):
        self.op_tree_in = np.maximum(self.sp_tree - self.k ,0) if self.is_call else np.maximum(self.k - self.sp_tree ,0)
        self.op_tree_out = np.maximum(self.sp_tree - self.k ,0) if self.is_call else np.maximum(self.k - self.sp_tree ,0)
        
        if self.Hu is not None:
            self.op_tree_in = np.where(self.exp_tree >= self.in_Hu,0,self.op_tree_in) 
            self.op_tree_out = np.where(self.exp_tree >= self.out_Hu,0,self.op_tree_out)
        if self.Hd is not None:
            self.op_tree_in = np.where(self.exp_tree <= self.in_Hd,0,self.op_tree_in) 
            self.op_tree_out = np.where(self.exp_tree <= self.out_Hd,0,self.op_tree_out)
    
    def _backward_process_(self):
        for j in range(self.N,0,-1):
            self.op_tree_in[:j,j-1] = self.D * (self.pu * self.op_tree_in[:j,j] + self.pd * self.op_tree_in[1:(j+1),j])
            self.op_tree_out[:j,j-1] = self.D * (self.pu * self.op_tree_out[:j,j] + self.pd * self.op_tree_out[1:(j+1),j])
            if self.Hu is not None:
                self.op_tree_in[:,j-1] = np.where(self.exp_tree[:,j-1] >= self.in_Hu,0,self.op_tree_in[:,j-1]) 
                self.op_tree_out[:,j-1] = np.where(self.exp_tree[:,j-1] >= self.out_Hu,0,self.op_tree_out[:,j-1])
            if self.Hd is not None:
                self.op_tree_in[:,j-1] = np.where(self.exp_tree[:,j-1] <= self.in_Hd,0,self.op_tree_in[:,j-1]) 
                self.op_tree_out[:,j-1] = np.where(self.exp_tree[:,j-1] <= self.out_Hd,0,self.op_tree_out[:,j-1])
            
    def _compute_(self):
        self._initAttr_()
        self._spot_tree_()
        self._op_tree_()
        self._backward_process_()
        self.price = np.array([self.op_tree_in[0,0],self.op_tree_out[0,0]])
        
    def cal_price(self):
        self._compute_()
        return np.interp(self.Hu,self.u**np.array(self.in_Hu,self.out_Hu),self.price)

















                                        

def LookbackFloatTree(spot,maturity,risk_free,volatility,dividend,nsteps,call = True,early_exercise = False,strike =None):                    
    # Calculate Basic Input 
    s,t,r,v,q,N,dt = spot,maturity,risk_free,volatility,dividend,nsteps,maturity/nsteps
    D = np.exp(-r*dt)
    u =np.exp(v*np.sqrt(dt))
    d = 1/u
    pu = (np.exp((r-q)*dt)-d)/(u-d)
    pd = 1-pu   

    # Construct Spot Price Tree & Option Price Tree
    exp_tree = SpotBinomialTree(u,nsteps)  # Tree of power of u for spot process
    sp_tree = s *  (u ** exp_tree)
    
    if call:
        pre_value = []
        for i in range(N+1):
            Nmin_max = min(exp_tree[i,N],0)   # At maturity,maximum power of u of the Smin
            Nmin_min = -i                     # At maturity,minimum power of u of the Smin  
            cat_list = np.array(list(range(Nmin_min,Nmin_max+1,1)))  # list of power of u for minimum price of a terminal node 
            spot_cat_value = s*u**cat_list     # list of spot price of terminal nodes
            option_cat_value = sp_tree[i,N] - spot_cat_value # list of catagory of option value for a terminal node
            pre_value.append(dict(zip(cat_list,option_cat_value))) # list of dictionary of option values of terminal nodes 

        
        for j in range(N-1,-1,-1): # j means time
            new_value = []
            for i in range(j+1): # i means nodes in time i 
                smin_max = min(exp_tree[i,j],0) # At j, maximum power of u of the Smin
                smin_min = -i                   # At j,minimum power of u of the Smin
                cat_list = np.array(list(range(smin_min,smin_max+1,1))) 
                option_cat_value = [(pu*pre_value[i][l] + pd*pre_value[i+1][l-1 if l == exp_tree[i,j] else l])*D for l in cat_list]
                
                if early_exercise:
                    intinsic_cat_value = np.array([sp_tree[i,j] - s*u**l for l in cat_list])
                    option_cat_value = np.maximum(option_cat_value,intrinsic_cat_value)
                    
                new_value.append(dict(zip(cat_list,option_cat_value)))
            pre_value = new_value
    
    else:
        pre_value = []
        for i in range(N+1):
            Nmax_min = max(exp_tree[i,N],0)
            Nmax_max = N - i
            cat_list = np.array(list(range(Nmax_min,Nmax_max+1,1)))
            spot_cat_value = s*u**cat_list
            option_cat_value = spot_cat_value - sp_tree[i,N]
            pre_value.append(dict(zip(cat_list,option_cat_value)))

        
        for j in range(N-1,-1,-1): # j means time
            new_value = []
            for i in range(j+1): # i means nodes in time i 
                smax_min = max(exp_tree[i,j],0)
                smax_max = j - i
                cat_list = np.array(list(range(smax_min,smax_max+1,1)))
                option_cat_value = np.array([(pu*pre_value[i][l+1 if l == exp_tree[i,j] else l] + pd*pre_value[i+1][l])*D for l in cat_list])
                
                if early_exercise:
                    intinsic_cat_value = np.array([s*u**l - sp_tree[i,j] for l in cat_list])
                    option_cat_value = np.maximum(option_cat_value,intrinsic_cat_value)
                    
                new_value.append(dict(zip(cat_list,option_cat_value)))
            pre_value = new_value
     
   # return pre_value
    return pre_value[0][0]




















#def LookbackFixTree(spot,strike,maturity,risk_free,volatility,dividend,nsteps,call = True,early_exercise = False):
#    input_kwarg = {'spot':spot,'strike':strike,'maturity':maturity,'risk_free':risk_free,'volatility':volatility,'dividend':dividend,'call':cal#l, 'early_excercise':early_exercise}











#def BarrierOptionTree(spot,strike,maturity, risk_free,volatility,barrier,nsteps,dividend = 0,call = True, knock_out = True, down = True):
    # Basic Iuput
 #   s,k,t,r,v,H,q,N = spot,strike,maturity, risk_free,volatility,barrier,dividend,nsteps
  #  D = np.exp(-r*dt)
   # u =np.exp(v*np.sqrt(dt))
   # d = 1/u
   # pu = (np.exp((r-q)*dt)-d)/(u-d)
   # pd = 1-pu
    
   # oH = H
   # iH
    
    #if knock_out:
        
        