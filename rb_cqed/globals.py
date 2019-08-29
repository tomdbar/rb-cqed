import numpy as np

##########################################
# Standard variable definitions.         #
##########################################
d = 3.584*10**(-29)
i = np.complex(0,1)

def R2args(R):
    alpha = np.clip(np.abs(R[0, 0]), 0, 1)
    phi1, phi2 = np.angle(R[0, 0]), np.angle(R[1, 0])
    beta = np.sqrt(1 - alpha ** 2)
    return alpha, beta, phi1, phi2

class Singleton(type):
    """
    A singleton metaclass to enforce that only one instance of an object exists.  If an object
    has already been instantiated, this instantiated version is returned when a new version
    of the object is requested.
    """
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]