import numpy as np


def homogeneous_correction_factor(x=0, radius=-1):
    if np.abs(x)>radius:
        return 0
    rp = 2*radius*np.sin(np.arccos(x/radius))
    if x == 0:
        return 1
    return (2*radius)/rp

def inhomogeneous_correction(x=-1, inner_circle_r=1, outer_circle_r=1):
    r = np.abs(x)
    if r >= outer_circle_r:
        return 0
    elif (r >= inner_circle_r) and (r <= outer_circle_r):
        return 2*outer_circle_r*np.sin(np.arccos(x/outer_circle_r))
    else:
        rp1 = 2*inner_circle_r*np.sin(np.arccos(x/inner_circle_r))
        rp2 = 2*outer_circle_r*np.sin(np.arccos(x/outer_circle_r))
        rp = rp2 - rp1
        return rp
    
def factor_inho(x=0, inner_circle_r=1, outer_circle_r=1):
    _value = inhomogeneous_correction(x=x, 
                                      inner_circle_r=inner_circle_r, 
                                      outer_circle_r=outer_circle_r)
    if x == 0:
        return 1
    if _value == 0:
        _value = np.NaN
    return (2*(outer_circle_r-inner_circle_r)/_value)