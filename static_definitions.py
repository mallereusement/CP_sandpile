import numpy as np

class exponent_functions():
    """A class containing functions for power-law and exponential relationships."""


    def P_of_S(self, x, amp, exponent):
        return amp * np.power(x, 1 - exponent)


    def P_of_T(self, x, amp, exponent):
        return amp * np.power(x, 1 - exponent)


    def P_of_L(self, x, amp, exponent):
        return amp * x ** (1 - exponent)

    def E_of_S_T(self, x, amp, exponent):
        return amp * np.power(x, exponent)


    def E_of_T_S(self, x, amp, exponent):
        return amp * np.power(x, 1 / exponent)

    def E_of_S_L(self, x, amp, exponent):
        return amp * np.power(x, exponent)

    def E_of_L_S(self, x, amp, exponent):
        return amp * np.power(x, 1 / exponent)

    def E_of_T_L(self, x, amp, exponent):
        return amp * np.power(x, exponent)

    def E_of_L_T(self, x, amp, exponent):
        return amp * np.power(x, 1 / exponent)
    
    def S_of_f(self, x, amp, exponent):
        return amp * np.power(x, 1 / exponent)
    
    def __getitem__(self, name):
        """Get a function by its name."""
        return getattr(self, name)
    


