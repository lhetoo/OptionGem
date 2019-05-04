import numpy as np 
from OptionClass import OptionClass
class MonteCarlo(OptionClass):
    
    def __init__(self,spot,strike,maturity,risk_free,volatility,dividend = 0,call = True,early_exercise = False,num_paths = 10000,num_t = 252,seed = 0):
        super().__init__(spot,strike,maturity,risk_free,volatility,dividend,call,early_exercise)
        self.N = num_paths
        self.nt = num_t
        self.dt = self.t/self.nt
        self.random_num = None
        self.price = None
        self.seed = seed
        self.intrinsic_value = np.zeros((self.nt+1,self.N))
    
        
    def set_random_number(self,anti_paths = True, mo_match = True):
        if self.seed is not None:
            np.random.seed(self.seed)
        if anti_paths:
            rand = np.random.standard_normal((self.nt,int(self.N/2)))
            rand = np.hstack([rand,-rand])
        else:
            rand = np.random.standard_normal((self.nt,self.N))
    
        if mo_match:
            rand = (rand - rand.mean())/rand.std()
        
        self.rand_num = rand
    
    
    def set_spot_paths(self):
    
        self.spot_paths = np.zeros(((self.nt + 1), self.N))
        self.spot_paths[0] = self.s
        for i in range(1,self.nt+1):
            self.spot_paths[i] = self.spot_paths[i-1] * np.exp((self.r-self.q-self.v**2/2)*self.dt + self.v*np.sqrt(self.dt) * self.rand_num[i-1])
    
    def _set_intrinsic_value_(self):
        pass
    
    def _cal_option_price_(self):
        if self.early_exercise:
            h = self.intrinsic_value
            V = h.copy()         # Option Values
            for i in range(self.nt-1,0,-1):
                condition = h[i] > 0 
                sreg = self.spot_paths[i][condition]
                vreg = V[i+1][condition]        # Early Exercise Exist 
                if len(sreg) >= 10:    
                    reg = np.polynomial.polynomial.polyfit(sreg,vreg * np.exp(-self.r*self.dt),2)
                    C = np.polynomial.polynomial.polyval(c= reg,x = sreg)
                    V[i][condition == False] = V[i+1][condition == False] * np.exp(-self.r*self.dt)
                    V[i][condition] = np.where(C > h[i][condition], V[i+1][condition] * np.exp(-self.r*self.dt), h[i][condition])
                else:
                    print(i)
                    V[i] = V[i+1] * np.exp(-self.r*self.dt)
            self.price = V[1].mean() * np.exp(-self.r*self.dt)
            
        else:
            self.price = np.mean(self.intrinsic_value[-1])* np.exp(-self.r*self.t)
            #self.price = np.exp(-self.r*self.t)*np.mean(-np.exp(abs(self.rand_num.sum(0))*np.sqrt(self.dt)*self.v + (self.r - self.q - self.v**2/2)*self.t) * self.s + self.spot_paths[-1])
    
    def cal_price(self):
        if self.random_num is None:
            self.set_random_number()
        self.set_spot_paths()
        self._set_intrinsic_value_()
        self._cal_option_price_()
        return self.price
    
    
    def _delta_(self):
        p = self.price
        s0 = self.s
        ds = s0 * 0.001
        self.s = s0 + ds 
        p1 = self.cal_price()
        self.s = s0 - ds 
        p2 = self.cal_price()
        self.s = s0
        self.price = p
        return (p1 - p2)/(2*ds)
    
    def _gamma_(self):
        s0 = self.s
        ds = s0 * 0.001
        self.s = s0 + ds 
        p1 = self.cal_price()
        self.s = s0 - ds 
        p2 = self.cal_price()
        self.s = s0
        p0 = self.cal_price()
        return (p1 -2*p0 + p2)/(ds**2)
    
    def _theta_(self):
        p = self.price
        t0 = self.t
        dt = t0 * 0.001
        self.t = t0 - dt
        self.dt = self.t/self.nt
        p1 = self.cal_price()
        self.t = t0 + dt
        self.dt = self.t/self.nt
        p2 = self.cal_price()
        self.t = t0
        self.dt = self.t/self.nt
        self.price = p
        return (p1 - p2)/(2*dt)

    
    def _vega_(self):
        p = self.price
        v0 = self.v
        dv = v0 * 0.001
        self.v = v0 + dv 
        p1 = self.cal_price()
        self.v = v0 - dv 
        p2 = self.cal_price()
        self.v = v0
        self.price = p
        return (p1 - p2)/(2*dv)
    
    def _rho_(self):
        p = self.price
        r0 = self.r
        dr = self.r * 0.001
        self.r = r0 + dr 
        p1 = self.cal_price()
        self.r = r0 - dr 
        p2 = self.cal_price()
        self.r = r0
        self.price = p
        return (p1 - p2)/(2*dr)
    
    def Greeks(self):
        p = self.cal_price()
        return dict({'price':self.price,'delta':self._delta_(),'theta':self._theta_(),'gamma':self._gamma_(),'vega':self._vega_(),'rho':self._rho_()})    
        
################################################################################################################


class VanillaOption(MonteCarlo):
    
    def _set_intrinsic_value_(self):
        self.intrinsic_value = np.maximum(self.spot_paths - self.k , 0) if self.is_call else np.maximum(self.k - self.spot_paths , 0)

##################################################################################################################

class LookbackFloatOption(MonteCarlo):
    def __init__(self,spot,maturity,risk_free,volatility,dividend = 0,call = True,s_minmax = None,early_exercise = False,num_paths = 10000,num_t = 252,seed = 0,strike = None):
        super().__init__(spot,strike,maturity,risk_free,volatility,dividend,call,early_exercise,num_paths,num_t,seed)
        self.s_minmax = self.s if s_minmax is None else s_minmax
    
    def _set_intrinsic_value_(self):
        
        if self.is_call:
            smin = np.ones(self.N) * np.minimum(self.s,self.s_minmax)
            for i in range(self.nt):
                smin = np.minimum(smin,self.spot_paths[i+1])
                self.intrinsic_value[i+1] = self.spot_paths[i+1] - smin * np.exp(-0.586 * self.v* np.sqrt(self.dt))
            
        else:
            smax = np.ones(self.N) * np.maximum(self.s,self.s_minmax)
            for i in range(self.nt):
                smax = np.maximum(smax,self.spot_paths[i+1])
                self.intrinsic_value[i+1] = smax*np.exp(0.586*self.v*np.sqrt(self.dt)) - self.spot_paths[i+1]                
                
                
####################################################################################################################
class LookbackFixOption(LookbackFloatOption):

    def _set_intrinsic_value_(self):
    
        if self.is_call:
            smax = np.ones(self.N) * np.maximum(self.s,self.s_minmax)
            for i in range(self.nt):
                smax = np.maximum(smax,self.spot_paths[i+1])
                self.intrinsic_value[i+1] = np.maximum(smax-self.k*np.exp(-0.586*self.v*np.sqrt(self.dt)),0) 
        else:
            smin = np.ones(self.N) * np.minimum(self.s,self.s_minmax)
            for i in range(self.nt):
                smin = np.minimum(smin,self.spot_paths[i+1])
                self.intrinsic_value[i+1] = np.maximum(self.k*np.exp(0.586*self.v*np.sqrt(self.dt))-smin,0)                
#####################################################################################################################

class BarrierOption(MonteCarlo):
    
    def __init__(self,spot,strike,maturity,risk_free,volatility,barrier,dividend = 0,call = True,knock_out = True,early_exercise = False,num_paths = 10000,num_t =252,seed = None):
        super().__init__(spot,strike,maturity,risk_free,volatility,dividend,call,early_exercise,num_paths,num_t,seed)
        self.H = barrier
        self.knock_out = knock_out
        self.down = (self.H < self.s)
        
    def _set_intrinsic_value_(self):
    
        is_knock = np.zeros(self.N)

        if self.knock_out:
            for i in range(self.nt):
                is_knock = np.logical_or(is_knock,np.where(self.spot_paths[i+1] <= self.H,1,0)) if self.down else np.logical_or(is_knock, np.where(self.spot_paths[i+1] >= self.H,1,0)) 
                self.intrinsic_value[i+1] = (1 - is_knock) * np.maximum(self.spot_paths[i+1] - self.k,0) if self.is_call else (1- is_knock) * np.maximum(self.k - self.spot_paths[i+1],0)
        
        else:
            for i in range(self.nt):
                is_knock = np.logical_or(is_knock,np.where(self.spot_paths[i+1] <= self.H,1,0)) if self.down else np.logical_or(is_knock, np.where(self.spot_paths[i+1] >= self.H,1,0)) 
                self.intrinsic_value[i+1] = is_knock * np.maximum(self.spot_paths[i+1] - self.k,0) if self.is_call else is_knock * np.maximum(self.k - self.spot_paths[i+1],0)


#########################################################################################################################
class AsianOption(MonteCarlo):
    
    def _set_intrinsic_value_(self):
        avg_price = (self.spot_paths.cumsum(0).T/np.arange(1,self.nt+2)).T
        self.intrinsic_value = np.maximum(avg_price - self.k,0) if self.is_call \
                                    else np.maximum(self.k - avg_price,0)
        






                