"""Mask functions for sponge layers

"""

def Theta(theta,zeta=0.9):
    """This is the mask function.
    
    """
    return (1+zeta)/zeta * np.arctan(zeta*np.sin(theta)/(1+zeta*np.cos(theta)))

def Y(y,Ly,zeta=0.9):
    """simply map cartesian y to theta, then use Theta
    
    """
    theta = np.pi*y/Ly
    big_y = Ly/np.pi * Theta(theta,zeta)
    return big_y

def Z(y, Ly,zeta=0.9):
    theta = np.pi*y/Ly
    return (1-zeta)**2/2 * (1-np.cos(theta))/(1+zeta**2+2*zeta*np.cos(theta))

