import BinomialTree as BT 
import FiniteDifference as FD
import BlackScholes as BS
import MonteCarlo as MC
import numpy as np 
from scipy.stats import norm                           
            
def VanillaOption(spot,strike,maturity,risk_free,volatility,dividend = 0,call = True,method = 'BS',early_exercise = False,Greeks = False, method_coef = None):
    
    input_kwarg = {'spot':spot,'strike':strike,'maturity':maturity,'risk_free':risk_free,'volatility':volatility,'dividend':dividend,'call':call,'early_exercise' : early_exercise}
    if method_coef is not None:
        input_kwarg.update(method_coef)
    
    if method == 'BS':
        obj = BS.EuropeanOption(**input_kwarg)

    elif method == 'BT':
        obj = BT.VanillaOption(**input_kwarg)
    
    elif method == 'FD':
        obj = FD.CrankNicolson(**input_kwarg)
    
    elif method == 'MC':
        obj = MC.VanillaOption(**input_kwarg)
    
    return obj.cal_price() if not Greeks else obj.Greeks()
        
        
def BinaryOption(spot,strike,maturity,risk_free,volatility,dividend = 0,call = True,cash = True):     
    s,k,t,r,v,q = spot,strike,maturity,risk_free,volatility,dividend
    d1 = (np.log(s/k)+(r-q+v**2/2)*t)/(v*np.sqrt(t))
    d2 = d1-v*np.sqrt(t)
    if cash:
        value = np.exp(-r*t)*norm.cdf(d2) if call else np.exp(-r*t)*norm.cdf(-d2)
    else:
        value = s*np.exp(-q*t)*norm.cdf(d1) if call else np.exp(-q*t)*norm.cdf(-d1)
    return value

        
def BarrierOption(spot,strike,maturity, risk_free,volatility,barrier,dividend = 0,call = True, knock_out = True,method = 'BS',early_exercise = False,method_coef = None,Greeks = False):
    input_kwarg = {'spot':spot,'strike':strike,'maturity':maturity,'risk_free':risk_free,'volatility':volatility,'dividend':dividend,'call':call,'barrier':barrier,'knock_out':knock_out}
    if method_coef is not None:
        input_kwarg.update(method_coef)
        
    if method == 'BS':
        anw = BS.BarrierOption(**input_kwarg).cal_price() if not Greeks else BS.BarrierOption(**input_kwarg).Greeks()
    
    elif method == 'MC':
        anw = MC.BarrierOption(**input_kwarg).cal_price() if not Greeks else MC.BarrierOption(**input_kwarg).Greeks()
    return anw 
        
    
def LookbackFloatOption(spot,maturity,risk_free,volatility,dividend = 0, call = True,s_minmax = None,strike = None, method = 'BS',early_exercise = False,method_coef = None,Greeks = False):
    
    input_kwarg = {'spot':spot,'maturity':maturity,'risk_free':risk_free,'volatility':volatility,'dividend':dividend,'call':call,'s_minmax':s_minmax}
    if method_coef is not None:
        input_kwarg.update(method_coef)
    if method == 'BS':
        anw = BS.LookbackFloatOption(**input_kwarg).cal_price() if not Greeks else BS.LookbackFloatOption(**input_kwarg).Greeks()
    
    elif method == 'MC':
        anw = MC.LookbackFloatOption(**input_kwarg).cal_price() if not Greeks else MC.LookbackFloatOption(**input_kwarg).Greeks()
    return anw     
        
def LookbackFixOption(spot,strike,maturity,risk_free,volatility,dividend = 0,call = True,s_minmax = None,method = 'BS',early_exercise = False,method_coef = None,Greeks = False):

    input_kwarg = {'spot':spot,'strike':strike,'maturity':maturity,'risk_free':risk_free,'volatility':volatility,'dividend':dividend,'call':call,'s_minmax':s_minmax}
    if method_coef is not None:
        input_kwarg.update(method_coef)
    if method == 'BS':
        anw = BS.LookbackFixOption(**input_kwarg).cal_price() if not Greeks else BS.LookbackFixOption(**input_kwarg).Greeks()
    
    elif method == 'MC':
        anw = MC.LookbackFixOption(**input_kwarg).cal_price() if not Greeks else MC.LookbackFixOption(**input_kwarg).Greeks()
    return anw 

def ImpliedVolatility(spot,strike,maturity,risk_free,option_price,dividend = 0,call = True, est_v = 0.5,it =100):
        s,k,t,r,q,P = spot,strike,maturity,risk_free,dividend,option_price
        vega = lambda v:BS.EuropeanOption(s,k,t,r,v,q,call).vega()
        for i in range(it):
            est_v1 = est_v - (VanillaOption(s,k,t,r,est_v,q,call) - P)/vega(est_v)
            print(est_v)
        return est_v
        
    




    


    