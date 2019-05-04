import numpy as np 
from scipy.stats import norm
from OptionClass import OptionClass

#################################################################################################
class BlackScholesOption(OptionClass):
    def __init__(self,spot,strike,maturity,risk_free,volatility,dividend = 0,call = True,early_exercise = False):
        super().__init__(spot,strike,maturity,risk_free,volatility,dividend,call,early_exercise)
        if self.k is not None:
            self.d1 = (np.log(self.s/self.k)+(self.r-self.q+self.v**2/2)*self.t)/(self.v*np.sqrt(self.t))
            self.d2 = self.d1-self.v*np.sqrt(self.t)
    
    def cal_price(self):
        pass
    
    def delta(self):
        p = self.price if self.price is not None else self.cal_price()
        s0 = self.s
        ds = s0 * 0.001
        self.s = s0 + ds 
        p1 = self.cal_price()
        self.s = s0 - ds 
        p2 = self.cal_price()
        self.s = s0
        self.price = p
        return (p1 - p2)/(2*ds)
    
    def gamma(self):
        p = self.price if self.price is not None else self.cal_price()
        s0 = self.s
        ds = s0 * 0.001
        self.s = s0 + ds 
        p1 = self.cal_price()
        self.s = s0 - ds 
        p2 = self.cal_price()
        self.s = s0
        p0 = self.cal_price()
        self.price = p
        return (p1 -2*p0 + p2)/(ds**2)
    
    def theta(self):
        p = self.price if self.price is not None else self.cal_price()
        t0 = self.t
        dt = t0 * 0.001
        self.t = t0 - dt
        p1 = self.cal_price()
        self.t = t0 + dt
        p2 = self.cal_price()
        self.t = t0
        self.price = p
        return (p1 - p2)/(2*dt)

    
    def vega(self):
        p = self.price if self.price is not None else self.cal_price()
        v0 = self.v
        dv = v0 * 0.001
        self.v = v0 + dv 
        p1 = self.cal_price()
        self.v = v0 - dv 
        p2 = self.cal_price()
        self.v = v0
        self.price = p
        return (p1 - p2)/(2*dv)
    
    def rho(self):
        p = self.price if self.price is not None else self.cal_price()
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
        p = self.price if self.price is not None else self.cal_price()
        return dict({'price':self.price,'delta':self.delta(),'theta':self.theta(),'gamma':self.gamma(),'vega':self.vega(),'rho':self.rho()}) 



###############################################################################################################    
class EuropeanOption(BlackScholesOption):

    def cal_price(self):
        s,k,t,r,v,q,d1,d2 = self.s,self.k,self.t,self.r,self.v,self.q,self.d1,self.d2
        if self.is_call:
            price = s*np.exp(-q*t)*norm.cdf(d1)-k*np.exp(-r*t)*norm.cdf(d2)
        else:
            price = k*np.exp(-r*t)*norm.cdf(-d2)-s*np.exp(-q*t)*norm.cdf(-d1)
        self.price = price
        return self.price
    
    def delta(self):
        s,k,t,r,v,q,d1,d2 = self.s,self.k,self.t,self.r,self.v,self.q,self.d1,self.d2
        return np.exp(-q*t)*norm.cdf(d1) if self.is_call else -np.exp(-q*t)*norm.cdf(-d1)

    def theta(self):
        s,k,t,r,v,q,d1,d2 = self.s,self.k,self.t,self.r,self.v,self.q,self.d1,self.d2
        return -np.exp(-q*t)*s*norm.pdf(d1)*v/(2*np.sqrt(t))-r*k*np.exp(-r*t)*norm.cdf(d2)+q*s*np.exp(-q*t)*norm.cdf(d1) if self.is_call \
                else -np.exp(-q*t)*s*norm.pdf(d1)*v/(2*np.sqrt(t))+r*k*np.exp(-r*t)*norm.cdf(-d2)-q*s*np.exp(-q*t)*norm.cdf(-d1)

    def rho(self):
        s,k,t,r,v,q,d1,d2 = self.s,self.k,self.t,self.r,self.v,self.q,self.d1,self.d2
        return k*t*np.exp(-r*t)*norm.cdf(d2) if self.is_call else -k*t*np.exp(-r*t)*norm.cdf(-d2)
    
    def gamma(self):
        s,k,t,r,v,q,d1,d2 = self.s,self.k,self.t,self.r,self.v,self.q,self.d1,self.d2
        return np.exp(-q*t)*norm.pdf(d1)/(s*v*np.sqrt(t))
        
    def vega(self):
        s,k,t,r,v,q,d1,d2 = self.s,self.k,self.t,self.r,self.v,self.q,self.d1,self.d2
        return s*np.exp(-q*t)*norm.pdf(d1)*np.sqrt(t)

    
    def Greeks(self): 
        return  {'price':self.cal_price(),'delta':self.delta(),'gamma':self.gamma(),'theta':self.theta(),'vega':self.vega(),'rho':self.rho()}


############################################################################################################################################
class BinarynOption(BlackScholesOption):
    def __init__(self,spot,strike,maturity,risk_free,volatility,dividend = 0,call = True,cash = True,early_exercise = False):
        super().__init__(spot,strike,maturity,risk_free,volatility,dividend,call,early_exercise)
        self.is_cash = cash
        
    def cal_price(self):
        s,k,t,r,v,q,d1,d2 = self.s,self.k,self.t,self.r,self.v,self.q,self.d1,self.d2

        if self.is_cash:
            price = np.exp(-r*t)*norm.cdf(d2) if self.is_call else np.exp(-r*t)*norm.cdf(-d2)
        else:
            price = s*np.exp(-q*t)*norm.cdf(d1) if self.is_call else np.exp(-q*t)*norm.cdf(-d1)
        self.price = price
        return self.price
        
        
        
##############################################################################################################################################
class BarrierOption(BlackScholesOption):

    def __init__(self,spot,strike,maturity,risk_free,volatility,barrier,dividend = 0,call = True,knock_out = True,adj_ob = None,early_exercise = False):
        super().__init__(spot,strike,maturity,risk_free,volatility,dividend,call,early_exercise)
        self.knock_out = knock_out
        self.H = barrier
        self.down = (self.H <= self.s)
        self.M = adj_ob
        
    def cal_price(self):    
        s,k,t,r,v,q,H,down,knock_out = self.s,self.k,self.t,self.r,self.v,self.q,self.H,self.down,self.knock_out
        if self.M is not None:
            if down:
                H = H*np.exp( - 0.5826 * v * (t/self.M)**0.5)
            else:
                H = H*np.exp(  0.5826 * v * (t/self.M)**0.5)
            
            
        lamb = (r-q+v**2/2)/v**2
        y = np.log(H**2/(s*k))/(v*np.sqrt(t)) + lamb*v*np.sqrt(t)
        x1 = np.log(s/H)/(v*np.sqrt(t)) + lamb*v*np.sqrt(t)
        y1 = np.log(H/s)/(v*np.sqrt(t)) + lamb*v*np.sqrt(t)
        C = EuropeanOption(s,k,t,r,v,q,self.is_call).cal_price()


        # For Call Option
        if self.is_call:
            if H <= k:
                cdi = s*np.exp(-q*t)*(H/s)**(2*lamb)*norm.cdf(y) - k*np.exp(-r*t)*(H/s)**(2*lamb-2)*norm.cdf(y-v*np.sqrt(t))
                cdo = C - cdi 
                cui = C
                cuo = 0
            
            else:   
                cdo = s*norm.cdf(x1)*np.exp(-q*t) - k*np.exp(-r*t)*norm.cdf(x1-v*np.sqrt(t)) - s*np.exp(-q*t)*(H/s)**(2*lamb)*norm.cdf(y1) + k*np.exp(-r*t)*(H/s)**(2*lamb-2)*norm.cdf(y1-v*np.sqrt(t))
                cdi = C - cdo 
                cui = s*norm.cdf(x1)*np.exp(-q*t) - k*np.exp(-r*t)*norm.cdf(x1-v*np.sqrt(t)) - s*np.exp(-q*t)*(H/s)**(2*lamb)*(norm.cdf(-y)-norm.cdf(-y1)) + k*np.exp(-r*t)*(H/s)**(2*lamb-2)*(norm.cdf(-y+v*np.sqrt(t))-norm.cdf(-y1+v*np.sqrt(t)))
                cuo = C - cui
                
            price = (cdo if knock_out else cdi) if down else (cuo if knock_out else cui)
                
        # For Put Option
        else:
            if H >= k:
                pdo = 0  
                pdi = C
                pui = -s*np.exp(-q*t)*(H/s)**(2*lamb)*norm.cdf(-y) + k*np.exp(-r*t)*(H/s)**(2*lamb-2)*norm.cdf(-y+v*np.sqrt(t))
                puo = C - pui
                
            else:
                
                pdi = -s*norm.cdf(-x1)*np.exp(-q*t) + k*np.exp(-r*t)*norm.cdf(-x1+v*np.sqrt(t)) + s*np.exp(-q*t)*(H/s)**(2*lamb)*(norm.cdf(y)-norm.cdf(y1)) - k*np.exp(-r*t)*(H/s)**(2*lamb-2)*(norm.cdf(y-v*np.sqrt(t))-norm.cdf(y1-v*np.sqrt(t)))
                pdo = C - pdi
                puo = -s*norm.cdf(-x1)*np.exp(-q*t) + k*np.exp(-r*t)*norm.cdf(-x1+v*np.sqrt(t)) + s*np.exp(-q*t)*(H/s)**(2*lamb)*norm.cdf(-y1) - k*np.exp(-r*t)*(H/s)**(2*lamb-2)*norm.cdf(-y1+v*np.sqrt(t))
                pui = C - puo
            price = (pdo if knock_out else pdi) if down else (puo if knock_out else pui)
        self.price = price
        return self.price
        

class LookbackFloatOption(BlackScholesOption):
    
    def __init__(self,spot,maturity,risk_free,volatility,dividend = 0,call = True,early_exercise = False,strike = None,s_minmax = None):
        super().__init__(spot,strike,maturity,risk_free,volatility,dividend,call,early_exercise )
        self.s_minmax = s_minmax
        
        
    def cal_price(self):
        s,t,r,v,q = self.s,self.t,self.r,self.v,self.q
        sm= s if self.s_minmax is None else self.s_minmax # maximum or minimum of previous stock prices, depends on call or put
        self.sm = sm
        
        if self.is_call:
            smin = np.minimum(sm,s)
            a1 = (np.log(s/smin) + (r-q+v**2/2)*t)/(v*np.sqrt(t))
            a2 = a1 - v*np.sqrt(t)
            a3 = (np.log(s/smin)+ (-r+q+v**2/2)*t)/(v*np.sqrt(t))
            y1 = -np.log(s/smin) *2*(r-q-v**2/2)/v**2
            if r == q:
                price = s*np.exp(-q*t)*norm.cdf(a1) - smin*np.exp(-r*t)*norm.cdf(a2)
            else:
                price = s*np.exp(-q*t)*norm.cdf(a1) - s*np.exp(-q*t)*v**2/(2*(r-q))*norm.cdf(-a1) \
                        -smin*np.exp(-r*t)*(norm.cdf(a2)-v**2/(2*(r-q))*np.exp(y1)*norm.cdf(-a3))

        else:
            smax = np.maximum(sm,s)
            b1 = (np.log(smax/s)+ (-r+q+v**2/2)*t)/(v*np.sqrt(t))
            b2 = b1 - v*np.sqrt(t)
            b3 = (np.log(smax/s) + (r-q-v**2/2)*t)/(v*np.sqrt(t))
            y1 = np.log(smax/s) *2*(r-q-v**2/2)/v**2
            if r == q:
                price = -s*np.exp(-q*t)*norm.cdf(a2)  + smax*np.exp(-r*t)*norm.cdf(a1)
            else:
                price = -s*np.exp(-q*t)*norm.cdf(b2) + s*np.exp(-q*t)*v**2/(2*(r-q))*norm.cdf(-b2) \
                        +smax*np.exp(-r*t)*(norm.cdf(b1)-v**2/(2*(r-q))*np.exp(y1)*norm.cdf(-b3))
                        
        self.price = price
        return self.price 

class LookbackFixOption(LookbackFloatOption):

    def cal_price(self):
        s,k,t,r,v,q = self.s,self.k,self.t,self.r,self.v,self.q
        sm= s if self.s_minmax is None else self.s_minmax
        input_kwarg = {'spot':s,'strike':k,'maturity':t,'risk_free':r,'volatility':v,'dividend':q}
        if self.is_call:
            price = LookbackFloatOption(**input_kwarg, call = False, s_minmax = np.maximum(sm,k)).cal_price() + s*np.exp(-q*t) - k*np.exp(-r*t)
        else:
            price = LookbackFloatOption(**input_kwarg, call = True, s_minmax = np.minimum(sm,k)).cal_price() - s*np.exp(-q*t) + k*np.exp(-r*t)
        self.price = price
        return self.price
        
class AsianOption(BlackScholesOption):
    
    def cal_price(self):
        s,k,t,r,v,q = self.s,self.k,self.t,self.r,self.v,self.q
        M1 = (np.exp(r - q)*t - 1)*s / (r-q)/t
        M2 = 2 *np.exp((2*(r-q)+v**2)*t)*s**2/(r-q+v**2)/(2*r-2*q+v**2)/t**2 + \
            2*s**2 /(r-q)/t**2 *(1/(2*(r-q)+v**2) - np.exp((r-q)*t)/(r-q+v**2))
        v1 = np.sqrt(np.log(M2/M1**2)/t)
        F0 = M1
        d1 = (np.log(F0/k) + v1**2/2*t)/(v1 * t**0.5)
        d2 = d1 - v1*t**0.5
        self.v1 = v1
        self.price = np.exp(-r*t)*(F0*norm.cdf(d1) - k*norm.cdf(d2)) if self.is_call \
                    else np.exp(-r*t)*(-F0*norm.cdf(-d1) + k*norm.cdf(-d2))
        return self.price
        