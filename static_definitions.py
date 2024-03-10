import numpy as np

class exponent_functions():

    def P_of_S(self, x, amp, exponent):
        return amp * x ** (1-exponent)


    def P_of_T(self, x, amp, exponent):
        return amp * x ** (1-exponent)


    def P_of_L(self, x, amp, exponent):
        return amp * x ** (1-exponent)

    def E_of_S_T(self, x, amp, exponent):
        return amp * x ** exponent


    def E_of_T_S(self, x, amp, exponent):
        return amp * x ** (1/exponent)

    def E_of_S_L(self, x, amp, exponent):
        return amp * x ** exponent

    def E_of_L_S(self, x, amp, exponent):
        return amp * x ** (1/exponent)

    def E_of_T_L(self, x, amp, exponent):
        return amp * x ** exponent

    def E_of_L_T(self, x, amp, exponent):
        return amp * x ** (1/exponent)
    
    def __getitem__(self, name):
        return getattr(self, name)
    


