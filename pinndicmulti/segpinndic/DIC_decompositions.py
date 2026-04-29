from pinndicmulti.DIC_importlib import jnp, vmap, tree_map, np, plt, mcoll, jax, partial

from pinndicmulti.segpinndic import DIC_windows
from pinndicmulti.segpinndic import DIC_networks
from pinndicmulti.segpinndic.utils.jax_util import tree_index
from pinndicmulti.segpinndic.utils.other import colors


class Decomposition:
    """Base decomposition class to be inherited by different decomposition classes.

    Note all methods in this class are jit compiled / used by JAX,
    so they must not include any side-effects!
    (A side-effect is any effect of a function that doesn’t appear in its output)
    This is why only static methods are defined.
    """

    # required methods

    @staticmethod
    def init_params(*args):
        """Initialise class parameters.
        Returns tuple of dicts ({k: pytree}, {k: pytree}) containing static and trainable parameters.
        The special key, k='subdomain', should be used to specify all subdomain parameters"""

        # below parameters need to be defined
        static_params = {
            "m":None,# total number of models (i.e. subdomains) in domain
            "xd":None,# dimensionality of x
            "subdomain":{
                "pou":None,# pou for each subdomain
                }
            }
        raise NotImplementedError

    @staticmethod
    def norm_fn(params, x):
        """"Applies norm function, for a SINGLE point with shape (xd,) and params for a SINGLE model"""
        raise NotImplementedError

    @staticmethod
    def unnorm_fn(params, u):
        """"Applies unnorm function, for a SINGLE point with shape (ud,) and params for a SINGLE model"""
        raise NotImplementedError

    @staticmethod
    def window_fn(params, x):
        """"Applies window function, for a SINGLE point with shape (xd,) and params for a SINGLE model"""
        raise NotImplementedError

    @staticmethod
    def inside_points(all_params, x_batch):
        """Returns ips, ims, inside_ims, where
        ips, ims = indices of all point-model pairs where point is inside model
        inside_ims = indicies of all models which have at least one point inside them
        """
        raise NotImplementedError

    @staticmethod
    def inside_models(all_params, x_batch, ims):
        """Returns inside_ips, d, where
        inside_ips = indicies of all points which are at least inside one model in ims
        d = average number of points inside each model in ims
        """
        raise NotImplementedError

    @staticmethod
    def plot(all_params, active=None, create_fig=True):
        "Plots decomposition. Returns matplotlib figure"
        raise NotImplementedError


class RectangularDecompositionND(Decomposition):
    """ND hyperrectangular domain.
    Rectangular subdomains can be placed arbitrarily in domain."""

    @staticmethod
    def init_params(subdomain_xs, subdomain_ws, unnorm):
        """Creates hyperrectangular subdomains initialised on a regular grid
        with subdomain centers subdomain_xs and widths subdomain_ws.
        """

        # get dimensionality of DD
        nm = tuple([len(x) for x in subdomain_xs])# shape of rectangular DD grid
        m = np.prod(nm)
        xd = len(subdomain_xs)# number of input dimensions

        # get level params
        ps = RectangularDecompositionND._get_level_params(
            0, xd, subdomain_xs, subdomain_ws, unnorm
        )

        # set constants for rectangular scheduler
        xmins0, xmaxs0 = (ps[0]+ps[2]/2), (ps[1]-ps[3]/2)# center lines of overlapping regions

        params = tree_map(lambda x: jnp.array(x), ps)

        static_params = {
            "m":m,
            "xd":xd,
            "subdomain":{"params":params[:-1],# use special key for subdomain parameters
                         "pou":params[-1]}, # Weight function parameters
            "xmins0":xmins0,
            "xmaxs0":xmaxs0,
            }

        return static_params, {}
    
    @staticmethod
    def _get_level_params(il, xd, subdomain_xs, subdomain_ws, unnorm):

        # get subdomain extents
        xs = np.stack(np.meshgrid(*subdomain_xs, indexing="ij"), 0)# (xd, nm)
        ws = np.stack(np.meshgrid(*subdomain_ws, indexing="ij"), 0)# (xd, nm)
        if xs.shape != ws.shape:
            raise ValueError("shape of subdomain_ws not same as subdomain_xs")
        xmins, xmaxs = xs - (ws/2), xs + (ws/2)

        # get subdomain overlap widths
        wmins, wmaxs = 0.5*(xmaxs-xmins), 0.5*(xmaxs-xmins)# default value for 1 subdomain (xd, nm)
        for i in range(xd):
            # fill in slices
            sl0, sl1 = [slice(None),]*xd, [slice(None),]*xd
            sl0[i] = slice(None, -1); sl1[i] = slice(1, None)
            sl0, sl1 = (i,) + tuple(sl0), (i,) + tuple(sl1)
            wmaxs[sl0] = wmins[sl1] = xmaxs[sl0] - xmins[sl1]
            # fill in edges
            sl0, sl1 = [slice(None),]*xd, [slice(None),]*xd
            sl0[i] = 0; sl1[i] = -1
            sl0, sl1 = (i,) + tuple(sl0), (i,) + tuple(sl1)
            wmins[sl0] = wmaxs[sl0]
            wmaxs[sl1] = wmins[sl1]
        if (wmins <= 0).any() or (wmaxs <= 0).any():
            raise ValueError("some subdomains are not overlapping!")

        # flatten arrays
        f = lambda x: (x.reshape(xd, -1)).T# (m, xd)
        xmins, xmaxs = f(xmins), f(xmaxs)# (m, xd)
        wmins, wmaxs = f(wmins), f(wmaxs)# (m, xd)
        s = (xmins.shape[0], 1)

        # get flag for whether to apply window or not
        if xmins.shape[0] == 1:# 1 subdomain case
            flags = np.zeros(s)
        else:
            flags = np.ones(s)

        # get unnorm parameters
        mu, sd = unnorm
        mu_N = np.repeat(mu[None, :], xmins.shape[0], axis=0)
        sd_N = np.repeat(sd[None, :], xmins.shape[0], axis=0)
        unnorms = np.stack([mu_N, sd_N], axis=1)   # (N,2,2)
        # unnorms = jnp.expand_dims(unnorms, 2) # (N,2,1,2)

        # get pou index
        # important note: each POU MUST cover the entire domain (POU boundary introduces discontinuities)
        pous = il*np.ones(s)

        return [xmins, xmaxs, wmins, wmaxs, flags, unnorms, pous]
    
    @staticmethod
    def norm_fn(params, x):
        params = params["static"]["decomposition"]["subdomain"]["params"]
        xmin, xmax = params[:2]
        mu, sd = (xmax+xmin)/2, (xmax-xmin)/2
        return DIC_networks.norm(mu, sd, x)

    @staticmethod
    def unnorm_fn(params, u):
        params = params["static"]["decomposition"]["subdomain"]["params"]
        mu, sd = params[5]
        return DIC_networks.unnorm(mu, sd, u)

    @staticmethod
    def window_fn(params, x):
        params = params["static"]["decomposition"]["subdomain"]["params"]
        return params[4]*DIC_windows.cosine(*params[:2], x)+(1-params[4])
    
    @staticmethod
    def inside_points(all_params, x_batch):
        m = all_params["static"]["decomposition"]["m"]
        ims = jnp.arange(m)
        batch_size = min(int(1e9/(4*ims.shape[0])), x_batch.shape[0])# limit GPU memory
        all_params = {"params": all_params["static"]["decomposition"]["subdomain"]["params"]}# filter out subdomain params
        return inside_points_batch(all_params, x_batch, ims, batch_size,
                                   RectangularDecompositionND._inside_rectangleND)

    @staticmethod
    def inside_models(all_params, x_batch, ims):
        batch_size = min(int(1e9/(4*ims.shape[0])), x_batch.shape[0])# limit GPU memory
        all_params = {"params": all_params["static"]["decomposition"]["subdomain"]["params"]}# filter out subdomain params
        return inside_models_batch(all_params, x_batch, ims, batch_size,
                                   RectangularDecompositionND._inside_rectangleND)

    @staticmethod
    def _inside_rectangleND(all_params, x_batch, ims):
        "Code for assessing if point is in ND hyperrectangle"

        ps = all_params["params"]
        x_batch = jnp.expand_dims(x_batch, 1)# (n,1,xd)
        xmins = jnp.expand_dims(ps[0][ims], 0)# (1,mc,xd)
        xmaxs = jnp.expand_dims(ps[1][ims], 0)# (1,mc,xd)
        inside = (x_batch >= xmins) & (x_batch <= xmaxs)# (n,mc,xd)
        inside = jnp.all(inside, -1)# (n,mc) keep as bool to reduce memory
        return inside
    
    # helper methods
    @staticmethod
    def plot(all_params, iaxes=[0,1], active=None, show_norm=False, show_window=False, create_fig=True):
        p = all_params["static"]["decomposition"]
        params = {"static":{"decomposition":{"subdomain":p["subdomain"]}}}
        m, xd = p["m"], p["xd"]

        if active is None:
            active = np.ones(m)
            
        a,b = iaxes
        if create_fig: f = plt.figure(figsize=(8,8))
        else: f = plt.gcf()

        # get domain params
        xmins, xmaxs, wmins, wmaxs, *_ = params["static"]["decomposition"]["subdomain"]["params"]
        mus, sds = (xmaxs+xmins)/2, (xmaxs-xmins)/2

        # plot subdomains
        lines = np.empty((m, 4, 2, 2))# 0,1,2,3 counterclockwise from x axes
        lines[:,0,0,0] = lines[:,2,0,0] = lines[:,3,0,0] = lines[:,3,1,0] = xmins[:,a]
        lines[:,0,1,0] = lines[:,1,0,0] = lines[:,1,1,0] = lines[:,2,1,0] = xmaxs[:,a]
        lines[:,0,0,1] = lines[:,0,1,1] = lines[:,1,0,1] = lines[:,3,0,1] = xmins[:,b]
        lines[:,1,1,1] = lines[:,2,0,1] = lines[:,2,1,1] = lines[:,3,1,1] = xmaxs[:,b]
        lws = np.array([[3 if active[im] else 1]*4 for im in range(m)]).flatten()
        alphas = np.array([[1 if active[im] else 0.5]*4 for im in range(m)]).flatten()
        lss = np.array([[":" if active[im]==2 else "-"]*4 for im in range(m)]).flatten()
        cs = np.array([[colors[im]]*4 for im in range(m)]).flatten()
        lines = mcoll.LineCollection(lines.reshape((-1,2,2)),
                                        linewidths=lws,
                                        alpha=alphas,
                                        linestyles=lss,
                                        colors=cs)
        plt.gca().add_collection(lines)

        # plot active norms
        if show_norm:
            alphas = np.array([1 if active[im] else 0 for im in range(m)])
            cs = np.array([colors[im] for im in range(m)])
            plt.scatter(mus[:,a], mus[:,b], color=cs, alpha=alphas, s=100)
            plt.scatter(mus[:,a]+sds[:,a], mus[:,b]+sds[:,b], color=cs, alpha=alphas, s=100, edgecolor="k")

        # plot summed windows (expensive!)
        if show_window:
            xmin, xmax = xmins.min(0), xmaxs.max(0)
            x = np.tile(np.expand_dims((xmax+xmin)/2, 0), (150**2, 1))
            xs = [np.linspace(mi, ma, 150) for mi,ma in zip(xmin[np.array([a,b])],xmax[np.array([a,b])])]
            xxs = np.stack(np.meshgrid(*xs, indexing="ij"), 0)# (2, nm)
            x_ = xxs.reshape((2, 150**2)).T
            x[:,a] = x_[:,0]; x[:,b] = x_[:,1]
            ws = vmap(vmap(RectangularDecompositionND.window_fn, in_axes=(None,0)), in_axes=(0,None))(params, x)
            ww = ws.sum(0).reshape((150,150))
            plt.imshow(ww.T,
                        origin="lower", extent=(xmin[a], xmax[a], xmin[b], xmax[b]),
                        cmap="bwr", vmin=0, vmax=2, zorder=-99)

        # set axis limits / labels / aspect ratio
        xmin, xmax = xmins.min(0), xmaxs.max(0)
        mi, ma = xmin-0.05*(xmax-xmin), xmax+0.05*(xmax-xmin)
        plt.xlim(mi[a], ma[a]); plt.ylim(mi[b], ma[b])
        plt.xlabel(a); plt.ylabel(b)
        plt.gca().set_aspect("equal")
        
        return f


class MultilevelRectangularDecompositionND(RectangularDecompositionND):
    """ND hyperrectangular domain, with multiple DDs at different scales.
    Rectangular subdomains can be placed arbitrarily in domain."""

    def init_params(subdomain_xss, subdomain_wss, unnorm):
        """Creates multiscale hyperrectangular subdomains initialised on a regular grid
        with subdomain centers subdomain_xs and widths subdomain_ws.
        """
        # get dimensionality of DD
        nms = [tuple([len(x) for x in subdomain_xs]) for subdomain_xs in subdomain_xss]# shape of rectangular DD grid
        if False in [len(nm)==len(nms[0]) for nm in nms]:
            raise ValueError("subdomain_xss are not all the same dimensionality")
        m = sum([np.prod(nm) for nm in nms])
        xd = len(subdomain_xss[0])# number of input dimensions

        # get level params
        ps = [[] for _ in range(7)]
        for il,(subdomain_xs, subdomain_ws) in enumerate(zip(subdomain_xss, subdomain_wss)):
            ps_ = RectangularDecompositionND._get_level_params(il, xd, subdomain_xs, subdomain_ws, unnorm)
            for i,p_ in enumerate(ps_): ps[i].append(p_)
        ps = [np.concatenate(p) for p in ps]

        # set constants for rectangular scheduler
        xmins0, xmaxs0 = (ps[0]+ps[2]/2), (ps[1]-ps[3]/2)# center lines of overlapping regions

        params = tree_map(lambda x: jnp.array(x), ps)

        static_params = {
            "m":m,
            "xd":xd,
            "subdomain":{"params":params[:-1],# use special key for subdomain parameters
                         "pou":params[-1]},
            "xmins0":xmins0,
            "xmaxs0":xmaxs0,
            }

        return static_params, {}
    

@partial(jax.jit, static_argnums=(3,4))
def _inside_sum_batch(all_params, x_batch, ims, batch_size, inside_fn):
    """
    Computes summary statistics of which (point, model) pairs satisfy `inside_fn`,
    processing the data in fixed-size batches.
    Statistical information (without generating a specific index)

    Args:
        all_params: Model parameters (pytree).
        x_batch: Array of input points of shape (N, xd).
        ims: Model indices or model-specific data (size M).
        batch_size: Number of points per batch (static for JIT).
        inside_fn: Function (params, x_batch_sub, ims) -> (n, m) boolean array
                   indicating whether each point belongs to each model.

    Returns:
        (s, inside_ips, inside_ims, d):
            s: Total number of (point, model) pairs where inside == True.
            inside_ips: Boolean mask over points (length N).
            inside_ims: Boolean mask over models (length M).
            d: Estimated average spatial density of points per model.
        irange: Starting indices of each batch.
        mask: Boolean mask correcting padding in last batch.
    """
    def batch_step(x):
        i, mask = x
        x_batch_ = jax.lax.dynamic_slice(x_batch, (i,0), (batch_size, x_batch.shape[1]))# (n, xd)
        inside_ = jnp.expand_dims(mask,1)*inside_fn(all_params, x_batch_, ims)# (n, m)
        # s1: Does each point in this batch belong to at least one model?
        # s2: How many points does each model contain in this batch?
        s1, s2 = jnp.any(inside_, axis=1), inside_.sum(0)
        return (s1, s2)# (n), (m)

    # get fully-populated batches by shifting last value of irange
    r = x_batch.shape[0]%batch_size
    shift = batch_size-r if r else 0
    irange = jnp.arange(0, x_batch.shape[0], batch_size)# (k)
    mask = jnp.ones((len(irange), batch_size), dtype=bool)# (k, n)
    irange = irange.at[-1].add(-shift)
    mask = mask.at[-1,:shift].set(False)
    s1, s2 = jax.lax.map(batch_step, (irange, mask)) # auto stack, return ((k, n), (k, m))

    # parse ims and ips
    # inside_ips: Which points belong to at least one model
    # inside_ims: How many points does each model contain?
    inside_ips = jnp.concatenate([s1[:-1].ravel(), s1[-1][shift:]], axis=0)# (n)
    inside_ims = s2.sum(0)# (m) # Does each subdomain fall into any points?
    d = (inside_ims.mean()**(1/x_batch.shape[1]))# average number of points per model
    s = inside_ims.sum() # How many pairs are inside in total?
    inside_ims = inside_ims.astype(bool)
    return (s, inside_ips, inside_ims, d), irange, mask

@partial(jax.jit, static_argnums=(3,4,5))
def _inside_take_batch(all_params, x_batch, ims, batch_size, inside_fn, s, irange, mask):
    """
    Processes a single batch.
    Actually generate the index list of which (point, model) pairs satisfy `inside_fn`.

    Args:
        x: Tuple (i, mask)
            i: starting index of batch
            mask: boolean mask to ignore padded elements in last batch

    Returns:
        s1: Boolean mask (batch_size,) → whether each point is inside any model
        s2: Integer counts (M,) → number of inside points per model
    """
    def batch_step(carry, x):
        i, mask = x
        x_batch_ = jax.lax.dynamic_slice(x_batch, (i,0), (batch_size, x_batch.shape[1]))# (n, xd)
        inside_ = jnp.expand_dims(mask,1)*inside_fn(all_params, x_batch_, ims)# (n, m)
        inside_ = inside_.ravel()# (n*m)
        itake = jnp.cumsum(inside_)-1# (n*m) Give us the index (starting from 0) of this True value.
        ii_ = jnp.expand_dims(inside_,1)*ii.at[:,0].add(i)# (n*m, 2)
        take, s = carry
        take = take.at[s+itake].add(ii_)# (s, 2)
        return (take, s+itake[-1]+1), None

    ix,iy = jnp.meshgrid(jnp.arange(batch_size), jnp.arange(ims.shape[0]), indexing="ij")# (n, m)
    ii = jnp.stack([ix.ravel(), iy.ravel()], axis=1)# (n*m, 2)
    take = jnp.zeros((s,2), dtype=int)# (s, 2)
    (take, _), _ = jax.lax.scan(batch_step, (take, 0), (irange, mask))
    return take # each element is (point_index, model_index)

def inside_points_batch(all_params, x_batch, ims, batch_size, inside_fn):
    """
    Constructs the explicit list of (point_index, model_index) pairs
    where inside_fn is True.

    Args:
        all_params: Model parameters.
        x_batch: Input points (N, xd).
        ims: Model indices or data (M).
        batch_size: Batch size (static).
        inside_fn: Boolean membership function.
        s: Total number of inside pairs (static).
        irange: Batch starting indices.
        mask: Padding mask.

    Returns:
        take: Array of shape (s, 2), where each row is (point_idx, model_idx).
    """
    assert batch_size <= x_batch.shape[0]
    (s, inside_ips, inside_ims, d), irange, mask = _inside_sum_batch(all_params, x_batch, ims, batch_size, inside_fn)
    inside_ims = jnp.arange(ims.shape[0])[inside_ims] # Convert the Boolean mask into a model index
    s = s.item()
    take = _inside_take_batch(all_params, x_batch, ims, batch_size, inside_fn, s, irange, mask)
    # point_indices, model_indices, valid_model_indices
    return take[:,0], take[:,1], inside_ims

def inside_models_batch(all_params, x_batch, ims, batch_size, inside_fn):
    """
        Iterates over batches and accumulates valid (point, model) pairs.

        carry:
            take: Output array being filled
            s: Current write pointer

        x:
            (i, mask) batch start index and padding mask
        """
    assert batch_size <= x_batch.shape[0]
    (s, inside_ips, inside_ims, d), irange, mask = _inside_sum_batch(all_params, x_batch, ims, batch_size, inside_fn)
    inside_ips = jnp.arange(x_batch.shape[0])[inside_ips] # Convert the Boolean mask into a point index
    return inside_ips, d


if __name__ == "__main__":
    
    ## 2D test
    subdomain_xs = [np.linspace(-3,3,4), np.linspace(-2,2,3)]
    subdomain_ws = [3*np.ones(4), 2.2*np.ones(3)]

    decomposition = RectangularDecompositionND
    ps_ = decomposition.init_params(subdomain_xs, subdomain_ws, (0,1))
    all_params = {"static":{"decomposition":ps_[0]}, "trainable":{"decomposition":ps_[1]}}
    m = all_params["static"]["decomposition"]["m"]
    active = np.ones(m)

    active[1] = 0
    active[2] = 2
    decomposition.plot(all_params, active=active, show_norm=True, show_window=True)
    x_batch = np.array([[-3.6,-4.2],
                        [1,2],
                        [3,4]])
    for x in x_batch:
        plt.scatter(x[0], x[1])
    plt.show()
    print(decomposition.inside_models(all_params, x_batch, np.arange(m)))
    print(decomposition.inside_points(all_params, x_batch))

    # single subdomain test
    subdomain_xs = [np.array([0]), np.array([0])]
    subdomain_ws = [np.array([1]), np.array([2])]

    decomposition = RectangularDecompositionND
    ps_ = decomposition.init_params(subdomain_xs, subdomain_ws, (0,1))
    all_params = {"static":{"decomposition":ps_[0]}, "trainable":{"decomposition":ps_[1]}}

    decomposition.plot(all_params)
    plt.show()
    
    
    ## multiscale tests
    subdomain_xss = [[np.linspace(-3,3,4), np.linspace(-2,2,3)],
                     [np.linspace(-3,3,10), np.linspace(-2,2,10)],
                     ]
    subdomain_wss = [[3*np.ones(4), 2.2*np.ones(3)],
                     [1*np.ones(10), 1*np.ones(10)],
                     ]

    decomposition = MultilevelRectangularDecompositionND
    ps_ = decomposition.init_params(subdomain_xss, subdomain_wss, (0,1))
    all_params = {"static":{"decomposition":ps_[0]}, "trainable":{"decomposition":ps_[1]}}
    m = all_params["static"]["decomposition"]["m"]
    active = np.ones(m)

    decomposition.plot(all_params, active=active, show_norm=True, show_window=True)