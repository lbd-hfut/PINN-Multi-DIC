import jax.nn
from pinndicmulti.DIC_importlib import jnp, np, jax
from pinndicmulti.segpinndic.utils.logger import logger

class Problem:
    """Base problem class to be inherited by different problem classes.

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
            "dims":None,# (ud, xd)# dimensionality of u and x
            }
        raise NotImplementedError

    @staticmethod
    def sample_constraints(all_params, domain):
        """Samples all constraints.
        Returns [[x_batch, *any_constraining_values, required_ujs], ...]. Each list element contains
        the x_batch points and any constraining values passed to the loss function, and the required
        solution and gradient components required in the loss function, for each constraint."""
        raise NotImplementedError

    @staticmethod
    def loss_fn(all_params, constraints):
        """Computes the PINN loss function, using constraints with the same structure output by sample_constraints"""
        raise NotImplementedError
    

class DIC_MSE(Problem):
    """DIC problem class with MSE loss function"""

    @staticmethod
    def init_params(ref_img, QKBQKT_def, mask, degree, znssd_kernel_size=7):
        static_params = {
            "dims":(2,2),
            "QKBQKT_def": QKBQKT_def,
            "ref_img": ref_img,
            "mask": mask,
            "degree": degree,
        }
        return static_params, {}
    
    @staticmethod
    def sample_constraints(all_params, domain, dim=2):
        x_batch_global = domain.sample_interior(all_params["static"]["problem"]["mask"])
        if dim == 2:
            required_ujs = (
                (0,()),
                (1,()),
                (0,(0,)),
                (0,(1,)),
                (1,(0,)),
                (1,(1,))
            )
        else:
            required_ujs = ((0,()),(1,()))
        return [[x_batch_global, required_ujs]]
    
    @staticmethod
    def loss_fn(all_params, x_batch, uv):
        ref_img = all_params["static"]["problem"]["ref_img"]
        QKBQKT_def = all_params["static"]["problem"]["QKBQKT_def"]
        degree = all_params["static"]["problem"]["degree"]
        u, v = uv[:,0], uv[:,1]

        xref, yref = x_batch[:,0], x_batch[:,1]
        xs, ys = xref + u, yref + v

        # warp defimg
        H, W = QKBQKT_def.shape[:2]

        xs_floor = jax.lax.stop_gradient(jnp.floor(xs)).astype(jnp.int32)
        ys_floor = jax.lax.stop_gradient(jnp.floor(ys)).astype(jnp.int32)

        xs_oob = (xs_floor < 0) | (xs_floor >= W)
        ys_oob = (ys_floor < 0) | (ys_floor >= H)
        mask = xs_oob | ys_oob

        xs_floor = jnp.clip(xs_floor, 0, W - 1)
        ys_floor = jnp.clip(ys_floor, 0, H - 1)

        # (N,6,6)
        QK_B_QKT = QKBQKT_def[ys_floor, xs_floor]

        xd = xs - xs_floor
        yd = ys - ys_floor

        powers = jnp.arange(degree+1)
        x_vec = xd[:, None] ** powers[None, :]
        y_vec = yd[:, None] ** powers[None, :]

        tmp = jnp.einsum("ni,nij->nj", y_vec, QK_B_QKT)
        warp_values = jnp.einsum("ni,ni->n", tmp, x_vec)

        valus = ref_img[yref.astype(jnp.int32), xref.astype(jnp.int32)]

        mse = jnp.mean((warp_values - valus) ** 2)
        return mse
    

class DIC_ZNSSD(Problem):
    """DIC problem class with ZNSSD loss function"""

    @staticmethod
    def init_params(ref_img, QKBQKT_def, mask, degree, znssd_kernel_size=7):
        static_params = {
            "dims":(2,2),
            "QKBQKT_def": QKBQKT_def,
            "ref_img": ref_img,
            "mask": mask,
            "degree": degree,
            "padding": znssd_kernel_size//2,
            "znssd_kernel": jnp.ones(
                (znssd_kernel_size, znssd_kernel_size), dtype=jnp.float32)
        }
        return static_params, {}
    
    @staticmethod
    def sample_constraints(all_params, domain, dim=2):
        x_batch_global = domain.sample_interior(all_params["static"]["problem"]["mask"])
        if dim == 2:
            required_ujs = (
                (0,()),
                (1,()),
                (0,(0,)),
                (0,(1,)),
                (1,(0,)),
                (1,(1,))
            )
        else:
            required_ujs = ((0,()),(1,()))
        return [[x_batch_global, required_ujs]]
    
    @staticmethod
    def loss_fn(all_params, x_batch, uv):
        ref_img = all_params["static"]["problem"]["ref_img"]
        ref_img = ref_img.astype(jnp.float32)
        QKBQKT_def = all_params["static"]["problem"]["QKBQKT_def"]
        degree = all_params["static"]["problem"]["degree"]
        kernel_2d = all_params["static"]["problem"]["znssd_kernel"]
        pad = all_params["static"]["problem"]["padding"]
        u, v = uv[:,0], uv[:,1]

        xref, yref = x_batch[:,0], x_batch[:,1]
        xs, ys = xref + u, yref + v

        # warp defimg
        H, W = QKBQKT_def.shape[:2]

        xs_floor = jax.lax.stop_gradient(jnp.floor(xs)).astype(jnp.int32)
        ys_floor = jax.lax.stop_gradient(jnp.floor(ys)).astype(jnp.int32)
        # xs_oob = (xs_floor < 0) | (xs_floor >= W)
        # ys_oob = (ys_floor < 0) | (ys_floor >= H)
        # oob_mask = xs_oob | ys_oob
        xs_floor = jnp.clip(xs_floor, 0, W - 1)
        ys_floor = jnp.clip(ys_floor, 0, H - 1)
        # (N,6,6)
        QK_B_QKT = QKBQKT_def[ys_floor, xs_floor]

        xd = xs - xs_floor
        yd = ys - ys_floor
        powers = jnp.arange(degree+1)
        x_vec = xd[:, None] ** powers[None, :]
        y_vec = yd[:, None] ** powers[None, :]
        tmp = jnp.einsum("ni,nij->nj", y_vec, QK_B_QKT)
        warp_values = jnp.einsum("ni,ni->n", tmp, x_vec)
        
        yi = yref.astype(jnp.int32)
        xi = xref.astype(jnp.int32)
        def_img = jnp.zeros_like(ref_img).at[yi, xi].set(warp_values)
        roi_img = jnp.zeros_like(ref_img).at[yi, xi].set(1.0)
        ref_pad = jnp.pad(ref_img, ((pad,pad),(pad,pad)), mode='reflect')
        def_pad = jnp.pad(def_img, ((pad,pad),(pad,pad)), mode='reflect')
        roi_pad = jnp.pad(roi_img, ((pad,pad),(pad,pad)), mode='reflect')
        ref_roi = ref_pad * roi_pad
        def_roi = def_pad * roi_pad
        
        k = kernel_2d[None, None, :, :]
        ref4 = ref_roi[None, None, :, :]
        def4 = def_roi[None, None, :, :]
        roi4 = roi_pad[None, None, :, :]
        
        conv = jax.lax.conv_general_dilated
        S_ref = conv(ref4, k, window_strides=(1, 1), padding="VALID")[0, 0]
        S_def = conv(def4, k, window_strides=(1, 1), padding="VALID")[0, 0]
        S_ref2 = conv(ref4 * ref4, k, window_strides=(1, 1), padding="VALID")[0, 0]
        S_def2 = conv(def4 * def4, k, window_strides=(1, 1), padding="VALID")[0, 0]
        S_roi = conv(roi4, k, window_strides=(1, 1), padding="VALID")[0, 0]
        
        eps = 1e-6
        n = jnp.maximum(S_roi, 1.0)
        ref_mean = S_ref / n
        def_mean = S_def / n
        ref_var = jnp.maximum(S_ref2 / n - ref_mean**2, eps)
        def_var = jnp.maximum(S_def2 / n - def_mean**2, eps)
        ref_std = jnp.sqrt(ref_var)
        def_std = jnp.sqrt(def_var)

        znssd_map = ((ref_img - ref_mean)/ref_std*def_std - (def_img - def_mean)) ** 2
        sampled_znssd = znssd_map[yi, xi]
        znssd = jnp.mean(sampled_znssd)
        return znssd
    

    
