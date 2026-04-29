from pinndicmulti.DIC_importlib import jax, jnp, np
import scipy.stats

from pinndicmulti.segpinndic import DIC_networks

class Domain:
    """Base domain class to be inherited by different domain classes.

    Note all methods in this class are jit compiled / used by JAX,
    so they must not include any side-effects!
    (A side-effect is any effect of a function that doesn’t appear in its output)
    This is why only static methods are defined.
    """

    # required methods

    @staticmethod
    def init_params(*args):
        """Initialise class parameters.
        Returns tuple of dicts ({k: pytree}, {k: pytree}) containing static and trainable parameters"""

        # below parameters need to be defined
        static_params = {
            "xd":None,# dimensionality of x
            }
        raise NotImplementedError

    @staticmethod
    def sample_interior(all_params, batch_shape):
        """Samples interior of domain.
        Returns x_batch points in interior of domain"""
        raise NotImplementedError

    @staticmethod
    def norm_fn(all_params, x):
        """"Applies norm function, for a SINGLE point with shape (xd,)"""# note only used for PINNs, FBPINN norm function defined in Decomposition
        raise NotImplementedError
    
    
class RectangularDomainND(Domain):

    @staticmethod
    def init_params(xmin, xmax):
        
        static_params = {
            "xd":2,
            "xmin":jnp.array(xmin),
            "xmax":jnp.array(xmax),
            }
        return static_params, {}
    
    @staticmethod
    def sample_interior(roi=None):

        assert roi.ndim == 2, "ROI must be (H, W) bool array"
        H, W = roi.shape
        
        # ROI → coordinates
        ys, xs = np.where(roi)
        coords = np.stack([xs, ys], axis=1)   # (N,2)

        return coords

    @staticmethod
    def norm_fn(all_params, x):
        xmin, xmax = all_params["static"]["domain"]["xmin"], all_params["static"]["domain"]["xmax"]
        mu, sd = (xmax+xmin)/2, (xmax-xmin)/2
        x = DIC_networks.norm(mu, sd, x)
        return x
    
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    domain = RectangularDomainND
    