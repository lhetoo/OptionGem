
class OptionClass(object):
    def __init__(self,spot = None,strike = None,maturity = None,risk_free = None,volatility = None,dividend = None,call = True,early_exercise = False):
        self.s = spot
        self.k = strike
        self.t = maturity
        self.r = risk_free
        self.v = volatility
        self.q = dividend
        self.is_call = call
        self.early_exercise = early_exercise
        