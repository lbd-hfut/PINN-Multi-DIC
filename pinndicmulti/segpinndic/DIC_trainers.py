from pinndicmulti.DIC_importlib import os, time, pickle, jax, jnp, np, plt, SummaryWriter, \
    jit, vmap, value_and_grad, jvp, partial, random, optax
import IPython.display

from pinndicmulti.segpinndic.utils.logger import switch_to_file_logger, logger
from pinndicmulti.segpinndic.utils.jax_util import tree_index, total_size, str_tensor, partition, combine
from pinndicmulti.segpinndic import DIC_networks, DIC_plot_trainer


class _Trainer:
    "Generic model trainer class"

    def __init__(self, c):
        "Initialise device and output directories"
        # logger.info(c)

        # # initialise summary writer
        writer = SummaryWriter(c.summary_out_dir)
        writer.add_text("constants", str(c).replace("\n","  \n"))# uses markdown

        self.c, self.writer = c, writer

    def _print_summary(self, i, loss, rate, start):
        "Prints training summary"

        logger.info("[i: %i/%i] loss: %.4f rate: %.1f elapsed: %.2f hr %s" % (
               i,
               self.c.n_steps,
               loss,
               rate,
               (time.time()-start)/(60*60),
               self.c.run,
                ))
        self.writer.add_scalar("loss/train", loss, i)
        self.writer.add_scalar("stats/rate", rate, i)

    def _save_figs(self, i, fs):
        "Saves figures"

        if self.c.clear_output: IPython.display.clear_output(wait=True)
        for name,f in fs:
            if self.c.save_figures:
                f.savefig(self.c.summary_out_dir+f"{name}_{i:08d}.png",
                          bbox_inches='tight', pad_inches=0.1, dpi=100, facecolor="white")
            self.writer.add_figure(name, f, i, close=False)
        if self.c.show_figures: plt.show()
        else: plt.close("all")

    def _save_model(self, i, model):
        "Saves a model"

        model = jax.tree_util.tree_map(lambda x: np.array(x) if isinstance(x, jnp.ndarray) else x, model)# convert jax arrays to np
        with open(self.c.model_out_dir+f"model_{i:08d}.jax", "wb") as f:
            pickle.dump(model, f)

    def train(self):

        raise NotImplementedError
    

'''
LABELLING CONVENTIONS: 
    xd = dimensionality of point
    ud = dimensionality of solution
    dims = (ud, xd)
    n = number of points
    m = number of models (i.e. subdomains)
    c = number of constraints

    x = single coordinate (xd)
    x_batch = batch of coordinates (n, xd)
    uj = solution and gradient component list

    j = index in uj
    im = index of model
    ip = index of point
    ic = index of constraint
    i = generic index

    nm = shape of rectangular DDs
    ii = for grid index in nm
'''

def tree_map_dicts(f, *trees):
    "Apply function to top-level dicts in tree(s)"

    is_dict = lambda x: isinstance(x, dict)
    def apply(leaf, *leaves):
        if is_dict(leaf):# if top-level dict
            return f(leaf, *leaves)
        else:
            return leaf# if leaf (i.e. at bottom of tree), return first tree's leaf only (!)
    return jax.tree_util.tree_map(apply, *trees, is_leaf=is_dict)# stop traverse on top-level dicts

def get_jmaps(required_ujs):
    "Generate tree for computing chained jacobians"

    # logger.debug("get_jmaps")

    # build tree of required gradients
    tree = {}
    for iu,ixs in required_ujs:
        # iu: The nth output function u
        # ixs: The sequence of variable indices to be differentiated
        t = tree
        for ix in ixs:
            if ix not in t:
                t[ix] = {}
            t = t[ix]

    # parse tree nodes
    def get_nodes(t, n=(), ks=()):
        ni = len(n)-1 + 1# index of parent node (including u at start)
        for k in t:
            ks_ = ks+(k,)
            if t[k]:
                n += (((ni,k),ks_,0),)# node
                n = get_nodes(t[k], n, ks_)
            else:
                n += (((ni,k),ks_,1),)# leaf
        return n

    # list of chained grad functions
    nodes = get_nodes(tree)
    # logger.debug(nodes)

    # list of grad functions to evaluate
    leaves = tuple((i + 1, node[1]) for i,node in enumerate(nodes) if node[2])
    if not leaves:
        leaves = ((0,()),)# special case where only solution required. tree/nodes are empty in this case
    logger.debug(leaves)

    # get map between required_ujs and list of chained gradients
    jac_is = ()# il (leaf index), io (order index), iu (u index)
    for iu,ixs in required_ujs:
        io = len(ixs)
        il = [leaf[1][:io] for leaf in leaves].index(ixs)# also works for 0,()
        jac_is += ((il, io, iu),)
    logger.debug(jac_is)

    return nodes, leaves, jac_is


# JITTED FUNCTIONS
def FBPINN_model_inner(params, x, norm_fn, network_fn, unnorm_fn, window_fn):
    x_norm = norm_fn(params, x)# normalise
    u_raw = network_fn(params, x_norm)# network
    u = unnorm_fn(params, u_raw)# unnormalise
    w = window_fn(params, x)# window
    return u*w, w, u_raw

def PINN_model_inner(all_params, x, norm_fn, network_fn, unnorm_fn):
    x_norm = norm_fn(all_params, x)# normalise
    u_raw = network_fn(all_params, x_norm)# network
    u = unnorm_fn(u_raw)# unnormalise
    return u, u_raw

def FBPINN_model(all_params, x_batch, takes, model_fns, verbose=True):
    "Defines FBPINN model"

    norm_fn, network_fn, unnorm_fn, window_fn = model_fns
    m_take, n_take, p_take, np_take, npou = takes

    # take x_batch
    x_take = x_batch[n_take]# (s, xd)

    # take subdomain params
    d = all_params
    all_params_take = {
        t_k: {
            cl_k: {
                k: jax.tree_util.tree_map(lambda p:p[m_take], d[t_k][cl_k][k]) if k=="subdomain" else d[t_k][cl_k][k]
                for k in d[t_k][cl_k]
            }
            for cl_k in d[t_k]
        }
        for t_k in ["static", "trainable"]
    }
    f = {
        t_k: {
            cl_k: {
                k: jax.tree_util.tree_map(lambda p: 0, d[t_k][cl_k][k]) if k=="subdomain" else jax.tree_util.tree_map(lambda p: None, d[t_k][cl_k][k])
                for k in d[t_k][cl_k]
            }
            for cl_k in d[t_k]
        }
        for t_k in ["static", "trainable"]
    }

    logger.debug(jax.tree_util.tree_map(lambda x: str_tensor(x), all_params))
    logger.debug(jax.tree_util.tree_map(lambda x: str_tensor(x), all_params_take))

    # batch over parameters and points
    us, ws, us_raw = vmap(FBPINN_model_inner, in_axes=(f,0,None,None,None,None))(all_params_take, x_take, norm_fn, network_fn, unnorm_fn, window_fn)# (s, ud)

    # apply POU and sum
    u = jnp.concatenate([us, ws], axis=1)# (s, ud+1)
    u = jax.ops.segment_sum(u, p_take, indices_are_sorted=False, num_segments=len(np_take))# (_, ud+1)
    wp = u[:,-1:]
    u = u[:,:-1]/wp
    u = jax.ops.segment_sum(u, np_take, indices_are_sorted=False, num_segments=len(x_batch))# (n, ud)
    u = u/npou

    return u, wp, us, ws, us_raw

def PINN_model(all_params, x_batch, model_fns, verbose=True):
    "Defines PINN model"

    norm_fn, network_fn, unnorm_fn = model_fns
    # batch over parameters and points
    u, u_raw = vmap(PINN_model_inner, in_axes=(None,0,None,None,None))(all_params, x_batch, norm_fn, network_fn, unnorm_fn)# (n, ud)
    return u, u_raw

def FBPINN_forward(all_params, x_batch, takes, model_fns):
    "Computes uv of FBPINN model"
    return FBPINN_model(all_params, x_batch, takes, model_fns)[0]

def PINN_forward(all_params, x_batch, model_fns):
    "Computes uv of PINN model"
    return PINN_model(all_params, x_batch, model_fns)[0]

def FBPINN_predict_forward(all_params, x_batch, takes, model_fns, jmaps):
    "Computes gradients of FBPINN model"

    # isolate model function
    def u(x_batch):
        return FBPINN_model(all_params, x_batch, takes, model_fns)[0], ()
    return _get_ujs(x_batch, jmaps, u)

def PINN_predict_forward(all_params, x_batch, model_fns, jmaps):
    "Computes gradients of PINN model"

    # isolate model function
    def u(x_batch):
        return PINN_model(all_params, x_batch, model_fns)[0], ()
    return _get_ujs(x_batch, jmaps, u)

def _get_ujs(x_batch, jmaps, u):

    nodes, leaves, jac_is = jmaps
    vs = jnp.tile(jnp.eye(x_batch.shape[1]), (x_batch.shape[0],1,1))

    # chain required jacobian functions
    fs = [u]
    for (ni, ix), _, _ in nodes:
        fs.append(jacfwd(fs[ni], vs[:,ix]))

    # evaluate required jacobian functions
    jacs = []
    for ie,_ in leaves:
        fin, jac = fs[ie](x_batch)
        jacs.append(jac+(fin,))

    # index required jacobians
    ujs = [jacs[il][io][:,iu:iu+1] for il,io,iu in jac_is]

    logger.debug("fs")
    logger.debug(fs)
    logger.debug("jacs")
    for jac in jacs: logger.debug([j.shape for j in jac])# (n, ud)
    logger.debug("ujs")
    for uj in ujs: logger.debug(str_tensor(uj))

    return ujs

def jacfwd(f, v):
    "Computes jacobian for single x, for all y, fully chained"
    def jacfun(x):
        y, j, aux = jvp(f, (x,), (v,), has_aux=True)
        aux = aux + (y,)
        return j, aux
    return jacfun

def FBPINN_loss(active_params, fixed_params, static_params, takes, x_batch, model_fns, loss_fn):

    # add fixed params to active, recombine all_params
    d, da = active_params, fixed_params
    trainable_params = {
        cl_k: {
            k: jax.tree_util.tree_map(
                    lambda p1, p2:jnp.concatenate([p1,p2],0), 
                    d[cl_k][k], 
                    da[cl_k][k]
                ) if k=="subdomain" else d[cl_k][k]
                for k in d[cl_k]
            }
            for cl_k in d
        }
    all_params = {"static":static_params, "trainable":trainable_params}
    # run FBPINN 
    u = FBPINN_forward(all_params, x_batch, takes, model_fns)
    return loss_fn(all_params, x_batch, u)

def PINN_loss(active_params, static_params, x_batch, model_fns, loss_fn):

    # recombine all_params
    all_params = {"static":static_params, "trainable":active_params}
    # run PINN 
    u = PINN_forward(all_params, x_batch, model_fns)
    return loss_fn(all_params, x_batch, u)

@partial(jit, static_argnums=(0,5,8,9))
def FBPINN_update(optimiser_fn, active_opt_states,
                  active_params, fixed_params, static_params_dynamic, static_params_static,
                  takes, x_batch, model_fns, loss_fn):
    # recombine static params
    static_params = combine(static_params_dynamic, static_params_static)
    # update step
    lossval, grads = value_and_grad(FBPINN_loss, argnums=0)(
        active_params, fixed_params, static_params, takes, x_batch, model_fns, loss_fn)
    updates, active_opt_states = optimiser_fn(grads, active_opt_states, active_params)
    active_params = optax.apply_updates(active_params, updates)
    return lossval, active_opt_states, active_params

@partial(jit, static_argnums=(0, 4, 6, 7))
def PINN_update(optimiser_fn, active_opt_states,
                active_params, static_params_dynamic, static_params_static,
                x_batch, model_fns, loss_fn):
    # recombine static params
    static_params = combine(static_params_dynamic, static_params_static)
    # update step
    lossval, grads = value_and_grad(PINN_loss, argnums=0)(
        active_params, static_params, x_batch, model_fns, loss_fn)
    updates, active_opt_states = optimiser_fn(grads, active_opt_states, active_params)
    active_params = optax.apply_updates(active_params, updates)
    return lossval, active_opt_states, active_params


# For fast test inference only

@partial(jax.jit, static_argnums=(1,3,5))
def _FBPINN_predict_jit(all_params_dynamic, all_params_static, x_batch, jmaps, takes, model_fns):
    all_params = combine(all_params_dynamic, all_params_static)
    return FBPINN_predict_forward(all_params, x_batch, takes, model_fns, jmaps)
def FBPINN_predict_jit(all_params, x_batch, jmaps, takes, model_fns):
    all_params_dynamic, all_params_static = partition(all_params)
    return _FBPINN_predict_jit(all_params_dynamic, all_params_static, x_batch, jmaps, takes, model_fns)

@partial(jax.jit, static_argnums=(1,3,4))
def _PINN_predict_jit(all_params_dynamic, all_params_static, x_batch, jmaps, model_fns):
    all_params = combine(all_params_dynamic, all_params_static)
    return PINN_predict_forward(all_params, x_batch, model_fns, jmaps)
def PINN_predict_jit(all_params, x_batch, jmaps, model_fns):
    all_params_dynamic, all_params_static = partition(all_params)
    return _PINN_predict_jit(all_params_dynamic, all_params_static, x_batch, jmaps, model_fns)


def get_inputs(x_batch, active, all_params, decomposition):
    "Get the inputs to the FBPINN model based on x_batch and the active models"

    # get the ims inside x_batch
    n_take, m_take, training_ims = decomposition.inside_points(all_params, x_batch)# (nc, m)

    # get active_ims and fixed_ims
    # scheduler should return
    # 0 = inactive (but still trained if it intersects with active models)
    # 1 = active
    # 2 = fixed
    # now modify active to
    # 0 = discard (not in current training points)
    # 1 = active
    # 2 = fixed
    active = jnp.array(active).copy()
    assert jnp.isin(active, jnp.array([0,1,2])).all()
    assert active.shape == (all_params["static"]["decomposition"]["m"],)

    active = active.at[active==0].set(1)# set inactive models to active
    mask = jnp.zeros_like(active)# mask out models in training points
    mask = mask.at[training_ims].set(1)
    active = active*mask
    ims_ = jnp.arange(all_params["static"]["decomposition"]["m"])
    active_ims = ims_[active==1]# assume unsorted
    fixed_ims = ims_[active==2]
    # logger.debug("updated active")
    # logger.debug(active)
    # logger.debug("active_ims")
    # logger.debug(active_ims)
    # logger.debug("fixed_ims")
    # logger.debug(fixed_ims)

    # note, numbers in all_ims == numbers in training_ims == numbers in m_take
    # which also means we need all m_take points above
    all_ims = jnp.concatenate([active_ims, fixed_ims])

    # re-index m_take to all_ims index
    inv = jnp.zeros(all_params["static"]["decomposition"]["m"], dtype=int)
    inv = inv.at[all_ims].set(jnp.arange(len(all_ims)))# assumes all_ims is unique
    m_take = inv[m_take]

    # (!) note: make sure n_take, pous (and therefore p_take / np_take) are sorted - makes segment_sum quicker
    # logger.debug("takes")
    # logger.debug(str_tensor(m_take))
    # logger.debug(str_tensor(n_take))

    # get POUs
    pous = all_params["static"]["decomposition"]["subdomain"]["pou"][all_ims].astype(int)
    np = jnp.stack([n_take, pous[m_take,0]], axis=-1).astype(int)# points and pous
    # logger.debug(str_tensor(np))
    npu,p_take = jnp.unique(np, axis=0, return_inverse=True)# unique points and pous (sorted), point-pou takes
    np_take = npu[:,0]
    # logger.debug(str_tensor(p_take))
    # logger.debug(str_tensor(np_take))
    npou = len(jnp.unique(all_params["static"]["decomposition"]["subdomain"]["pou"].astype(int)))# global npou
    # logger.debug(f"Total number of POUs: {npou}")

    takes = (m_take, n_take, p_take, np_take, npou)

    # cut active and fixed parameter trees
    def cut_active(d):
        "Cuts active_ims from param dict"
        return {cl_k: {k: jax.tree_util.tree_map(lambda p:p[active_ims], d[cl_k][k]) if k=="subdomain" else d[cl_k][k]
                for k in d[cl_k]}
                for cl_k in d}
    def cut_fixed(d):
        "Cuts fixed_ims from param dict"
        return {cl_k: {k: jax.tree_util.tree_map(lambda p:p[fixed_ims],  d[cl_k][k]) if k=="subdomain" else d[cl_k][k]
                for k in d[cl_k]}
                for cl_k in d}
    def cut_all(d):
        "Cuts all_ims from param dict"
        return {cl_k: {k: jax.tree_util.tree_map(lambda p:p[all_ims],    d[cl_k][k]) if k=="subdomain" else d[cl_k][k]
                for k in d[cl_k]}
                for cl_k in d}
    def merge_active(da, d):
        "Merges active_ims from param dict da to d"
        for cl_k in d:
            for k in d[cl_k]:
                if k=="subdomain":
                    d[cl_k][k] = jax.tree_util.tree_map(lambda pa, p: p.copy().at[active_ims].set(pa), da[cl_k][k], d[cl_k][k])
                else:
                    d[cl_k][k] = da[cl_k][k]
        return d

    return takes, all_ims, (active, cut_active, cut_fixed, cut_all, merge_active)


def _common_train_initialisation(c, key, all_params, problem, domain, dim=2):

    # print stats
    logger.info("Total number of trainable parameters:")
    for k in all_params["trainable"]:
        logger.info(f'\t{k}: {total_size(all_params["trainable"][k]):,}')

    # initialise optimiser
    optimiser = optax.adam(**c.optimiser_kwargs)
    all_opt_states = optimiser.init(all_params["trainable"])
    # logger.debug("all_opt_states")
    # logger.debug(jax.tree_util.tree_map(lambda x: str_tensor(x), all_opt_states))
    optimiser_fn, loss_fn = optimiser.update, problem.loss_fn

    # get global constraints (training points)
    key, subkey = random.split(key)
    constraints_global = problem.sample_constraints(all_params=all_params, domain=domain, dim=dim)
    
    # parse global constraints
    x_batch_global = constraints_global[0][0]# (n, xd)
    
    required_ujs = constraints_global[0][1]
    # logger.debug("x_batch_global")
    # logger.debug(str_tensor(x_batch_global))

    # get jac maps
    jmaps = get_jmaps(required_ujs)

    return (optimiser, all_opt_states, optimiser_fn, loss_fn, key,
            x_batch_global, jmaps)


class FBPINNTrainer(_Trainer):
    "FBPINN model trainer class"

    def _get_x_batch(self, i, active, all_params, x_batch_global, decomposition):
        "Get the x_batch points from x_batch_global which are inside active models"

        # cut active points out of x_batch_global
        ims = jnp.arange(all_params["static"]["decomposition"]["m"])[active==1]
        training_ips, _d = decomposition.inside_models(all_params, x_batch_global, ims)# (n, mc)
        x_batch = x_batch_global[training_ips]

        # report
        logger.info(f"[i: {i}/{self.c.n_steps}] Average number of points/dimension in active subdomains: {_d:.2f}")
        # logger.debug("x_batch")
        # logger.debug(str_tensor(x_batch))

        return x_batch

    def _get_update_inputs(self, i, active, all_params, all_opt_states, x_batch_global, decomposition):
        "Get inputs to the FBPINN update step based on active models"

        start0 = time.time()
        logger.info(f"[i: {i}/{self.c.n_steps}] Updating active inputs..")

        # check active
        logger.debug(active)
        active = jnp.array(active).copy()
        assert jnp.isin(active, jnp.array([0,1,2])).all()
        assert active.shape == (all_params["static"]["decomposition"]["m"],)

        # get x_batch from x_batch_global based on active
        x_batch = self._get_x_batch(i, active, all_params, x_batch_global, decomposition)

        # get model takes / scheduler cuts given x_batch and active
        takes, _, (active, cut_active, cut_fixed, cut_all, merge_active) = get_inputs(x_batch, active, all_params, decomposition)

        # cut params / opt states (schedule)
        active_params = cut_active(all_params["trainable"]) # cut active params
        fixed_params = cut_fixed(all_params["trainable"])   # cut fixed params
        static_params = cut_all(all_params["static"])       # cut active/fixed params
        active_opt_states = tree_map_dicts(cut_active, all_opt_states)# because all_opt_states has more complex structure
        # logger.debug("active_params")
        # logger.debug(jax.tree_util.tree_map(lambda x: str_tensor(x), active_params))
        # logger.debug("fixed_params")
        # logger.debug(jax.tree_util.tree_map(lambda x: str_tensor(x), fixed_params))
        # logger.debug("static_params")
        # logger.debug(jax.tree_util.tree_map(lambda x: str_tensor(x), static_params))
        # logger.debug("active_opt_states")
        # logger.debug(jax.tree_util.tree_map(lambda x: str_tensor(x), active_opt_states))

        logger.info(f"[i: {i}/{self.c.n_steps}] Updating active inputs done ({time.time()-start0:.2f} s)")

        return active, merge_active, active_opt_states, active_params, fixed_params, static_params, takes, x_batch

    def train(self, dim=2):
        "Train model"

        c, writer = self.c, self.writer

        # generate root key
        key = random.PRNGKey(c.seed)
        np.random.seed(c.seed)

        # define all_params
        all_params = {"static":{},"trainable":{}}

        # initialise domain, problem and decomposition params
        domain, problem, decomposition = c.domain, c.problem, c.decomposition
        for tag, cl, kwargs in zip(["domain", "problem", "decomposition"], [domain, problem, decomposition],
                                   [c.domain_init_kwargs, c.problem_init_kwargs, c.decomposition_init_kwargs]):
            ps_ = cl.init_params(**kwargs)
            if ps_[0]: all_params["static"][tag] = ps_[0]
            if ps_[1]: all_params["trainable"][tag] = ps_[1]
        assert (all_params["static"]["domain"]["xd"] ==\
                all_params["static"]["problem"]["dims"][1] ==\
                all_params["static"]["decomposition"]["xd"])
        logger.info(f'Total number of subdomains: {all_params["static"]["decomposition"]["m"]}')

        # initialise subdomain network params
        network = c.network
        key, *subkeys = random.split(key, all_params["static"]["decomposition"]["m"]+1)
        ps_ = vmap(network.init_params, in_axes=(0, None))(jnp.array(subkeys), *c.network_init_kwargs.values())
        if ps_[0]: all_params["static"]["network"] = tree_index(ps_[0],0)# grab first set of static params only
        if ps_[1]: all_params["trainable"]["network"] = {"subdomain": ps_[1]}# add subdomain key
        # logger.debug("all_params")
        # logger.debug(jax.tree_util.tree_map(lambda x: str_tensor(x), all_params))
        model_fns = (decomposition.norm_fn, network.network_fn, decomposition.unnorm_fn, decomposition.window_fn)

        # initialise scheduler
        scheduler = c.scheduler(all_params=all_params, n_steps=c.n_steps, **c.scheduler_kwargs)

        # common initialisation
        (optimiser, all_opt_states, optimiser_fn, loss_fn, key,
        x_batch_global, jmaps) = _common_train_initialisation(c, key, all_params, problem, domain, dim=dim)

        # train loop
        pstep, fstep = 0, 0 # Cumulative training parameter size, Cumulative FLOPs (floating-point operations)
        start0, start1, report_time = time.time(), time.time(), 0.
        merge_active, active_params, active_opt_states, fixed_params = None, None, None, None
        lossval = None
        for i,active_ in enumerate(scheduler):

            # update active
            if active_ is not None:
                active = active_

                # first merge latest all_params / all_opt_states
                if i != 0:
                    all_params["trainable"] = merge_active(active_params, all_params["trainable"])
                    all_opt_states = tree_map_dicts(merge_active, active_opt_states, all_opt_states)

                # then get new inputs to update step
                active, merge_active, active_opt_states, active_params, fixed_params, static_params, takes, x_batch = \
                     self._get_update_inputs(i, active, all_params, all_opt_states, x_batch_global, decomposition)
                
                # AOT compile update function
                startc = time.time()
                logger.info(f"[i: {i}/{self.c.n_steps}] Compiling update step..")
                static_params_dynamic, static_params_static = partition(static_params)
                update = FBPINN_update.lower(optimiser_fn, active_opt_states,
                                             active_params, fixed_params, static_params_dynamic, static_params_static,
                                             takes, x_batch, model_fns, loss_fn).compile()
                logger.info(f"[i: {i}/{self.c.n_steps}] Compiling done ({time.time()-startc:.2f} s)")
                cost_ = update.cost_analysis()
                p = total_size(active_params["network"])
                f = cost_.get("flops", 0) if cost_ else 0
                logger.info("p, f")
                logger.info((p,f))

            # report initial model
            if i == 0:
                lossval, start1, report_time = \
                self._report(i, start0, start1, report_time,
                            all_params, all_opt_states,
                            active, merge_active, active_opt_states, active_params,
                            lossval)

            # take a training step
            lossval, active_opt_states, active_params = update(active_opt_states,
                                         active_params, fixed_params, static_params_dynamic,
                                         takes, x_batch)# note compiled function only accepts dynamic arguments
            pstep, fstep = pstep+p, fstep+f

            # report
            lossval, start1, report_time = \
            self._report(i + 1, start0, start1, report_time,
                        all_params, all_opt_states,
                        active, merge_active, active_opt_states, active_params,
                        lossval)

        # cleanup
        writer.close()
        logger.info(f"[i: {i+1}/{self.c.n_steps}] Training complete")

        # return trained parameters
        all_params["trainable"] = merge_active(active_params, all_params["trainable"])
        all_opt_states = tree_map_dicts(merge_active, active_opt_states, all_opt_states)

        u, v, exx, exy, eyy = self._predict(all_params, decomposition, x_batch_global, jmaps, model_fns, dim=dim)
        
        return all_params, u, v, exx, exy, eyy, x_batch_global

    def _report(self, i, start0, start1, report_time,
                all_params, all_opt_states,
                active, merge_active, active_opt_states, active_params,
                lossval):
        "Report results"

        c = self.c
        summary_,test_,model_save_ = [(i % f == 0) for f in
                                      [c.summary_freq, c.test_freq, c.model_save_freq]]
        if lossval is None:
            loss_val = None
        else:
            loss_val = lossval.item()
        
        if summary_ or model_save_:

            # print summary
            if i != 0 and summary_:
                rate = c.summary_freq / (time.time()-start1-report_time)
                self._print_summary(i, loss_val, rate, start0)
                start1, report_time = time.time(), 0.

            if i != 0 and model_save_:

                start2 = time.time()

                # merge latest all_params / all_opt_states
                all_params["trainable"] = merge_active(active_params, all_params["trainable"])
                all_opt_states = tree_map_dicts(merge_active, active_opt_states, all_opt_states)

                self._save_model(i, (i, all_params, all_opt_states, active, jnp.array(loss_val) if loss_val is not None else jnp.array(jnp.nan)))

                report_time += time.time()-start2

        return loss_val, start1, report_time
    
    def _predict(self, all_params, decomposition, x_batch_global, jmaps, model_fns, dim=2):
        if x_batch_global.shape[0] > 512**2:
            batch_size = 512**2
            r = x_batch_global.shape[0]%batch_size
            shift = batch_size-r if r else 0
            irange = jnp.arange(0, x_batch_global.shape[0], batch_size)# (k)
            irange = irange.at[-1].add(-shift)
            
            def take_batch(i):
                return jax.lax.dynamic_slice(
                    x_batch_global,
                    (i, 0),
                    (batch_size, x_batch_global.shape[1])
                )
            
            x_batches = jax.vmap(take_batch)(irange)   # (m, batch_size, 2)
        else:
            batch_size = x_batch_global.shape[0]
            x_batches = x_batch_global[None, ...]   # (1, batch_size, 2)
        
        u = v = exx = exy = eyy = jnp.zeros_like(all_params["static"]["problem"]["ref_img"])
        
        if dim == 2:
            for i, x_batch in enumerate(x_batches):
                x_batch = x_batch.astype(jnp.float32)
                logger.info(f"Predicting on batch {i+1}/{x_batches.shape[0]}..")
                active = jnp.zeros(all_params["static"]["decomposition"]["m"], dtype=int)
                takes, _, (active, cut_active, cut_fixed, cut_all, merge_active) = \
                    get_inputs(x_batch, active, all_params, decomposition)
                    
                trainable_params = cut_active(all_params["trainable"])
                static_params = cut_all(all_params["static"])
                all_params_ = {"static":static_params, "trainable":trainable_params}
                
                ujs = FBPINN_predict_jit(all_params_, x_batch, jmaps, takes, model_fns)
                u_, v_, ux_, uy_, vx_, vy_ = ujs
                exx_, exy_, eyy_ = ux_, (uy_+vx_)/2, vy_
                ys = x_batch[:, 1].astype(jnp.int32)
                xs = x_batch[:, 0].astype(jnp.int32)
                u = u.at[ys, xs].set(u_.flatten())
                v = v.at[ys, xs].set(v_.flatten())
                exx = exx.at[ys, xs].set(exx_.flatten())
                exy = exy.at[ys, xs].set(exy_.flatten())
                eyy = eyy.at[ys, xs].set(eyy_.flatten())
        else:
            for i, x_batch in enumerate(x_batches):
                x_batch = x_batch.astype(jnp.float32)
                logger.info(f"Predicting on batch {i+1}/{x_batches.shape[0]}..")
                active = jnp.zeros(all_params["static"]["decomposition"]["m"], dtype=int)
                takes, _, (active, cut_active, cut_fixed, cut_all, merge_active) = \
                    get_inputs(x_batch, active, all_params, decomposition)
                    
                trainable_params = cut_active(all_params["trainable"])
                static_params = cut_all(all_params["static"])
                all_params_ = {"static":static_params, "trainable":trainable_params}
                
                ujs = FBPINN_predict_jit(all_params_, x_batch, jmaps, takes, model_fns)
                u_, v_ = ujs
                ys = x_batch[:, 1].astype(jnp.int32)
                xs = x_batch[:, 0].astype(jnp.int32)
                u = u.at[ys, xs].set(u_.flatten())
                v = v.at[ys, xs].set(v_.flatten())
        return u, v, exx, exy, eyy
    

class PINNTrainer(_Trainer):
    "PINN model trainer class"

    def train(self, dim=2):
        "Train model"

        c, writer = self.c, self.writer

        # generate root key
        key = random.PRNGKey(c.seed)
        np.random.seed(c.seed)

        # define all_params
        all_params = {"static":{},"trainable":{}}

        # initialise domain, problem and decomposition params
        domain, problem = c.domain, c.problem
        for tag, cl, kwargs in zip(["domain", "problem"], [domain, problem],
                                   [c.domain_init_kwargs, c.problem_init_kwargs]):
            ps_ = cl.init_params(**kwargs)
            if ps_[0]: all_params["static"][tag] = ps_[0]
            if ps_[1]: all_params["trainable"][tag] = ps_[1]
        assert (all_params["static"]["domain"]["xd"] ==\
                all_params["static"]["problem"]["dims"][1])

        # initialise network params
        network = c.network
        key, subkey = random.split(key)
        ps_ = network.init_params(key=subkey, **c.network_init_kwargs)
        if ps_[0]: all_params["static"]["network"] = ps_[0]
        if ps_[1]: all_params["trainable"]["network"] = {"subdomain": ps_[1]}# add subdomain key
        # logger.debug("all_params")
        # logger.debug(jax.tree_util.tree_map(lambda x: str_tensor(x), all_params))

        # define unnorm function
        mu_, sd_ = c.decomposition_init_kwargs["unnorm"]
        unnorm_fn = lambda u: DIC_networks.unnorm(mu_, sd_, u)
        model_fns = (domain.norm_fn, network.network_fn, unnorm_fn)

        # common initialisation
        (optimiser, all_opt_states, optimiser_fn, loss_fn, key,
        x_batch_global, jmaps) = _common_train_initialisation(c, key, all_params, problem, domain, dim=dim)

        # get implicit jitted update function
        active_params = all_params["trainable"]
        static_params = all_params["static"]
        active_opt_states = all_opt_states
        x_batch = x_batch_global

        # AOT compile update function
        startc = time.time()
        logger.info(f"[i: {0}/{self.c.n_steps}] Compiling update step..")
        static_params_dynamic, static_params_static = partition(static_params) # 图像和预处理那里需要处理一下
        update = PINN_update.lower(optimiser_fn, active_opt_states,
                                   active_params, static_params_dynamic, static_params_static,
                                   x_batch, model_fns, loss_fn).compile()
        logger.info(f"[i: {0}/{self.c.n_steps}] Compiling done ({time.time()-startc:.2f} s)")
        cost_ = update.cost_analysis()
        p = total_size(active_params["network"])
        f = cost_.get("flops", 0) if cost_ else 0
        logger.debug("p, f")
        logger.debug((p,f))

        # train loop
        pstep, fstep = 0, 0
        start0, start1, report_time = time.time(), time.time(), 0.
        lossval = None
        for i in range(c.n_steps):

            if i == 0:
                # report initial model
                lossval, start1, report_time = \
                self._report(i, start0, start1, report_time,
                            all_params, all_opt_states,
                            active_opt_states, active_params,
                            lossval)

            # take a training step
            lossval, active_opt_states, active_params = update(active_opt_states,
                                       active_params, static_params_dynamic,
                                       x_batch)# note compiled function only accepts dynamic arguments
            pstep, fstep = pstep+p, fstep+f

            # report
            lossval, start1, report_time = \
            self._report(i + 1, start0, start1, report_time,
                        all_params, all_opt_states,
                        active_opt_states, active_params,
                        lossval)

        # cleanup
        writer.close()
        logger.info(f"[i: {i+1}/{self.c.n_steps}] Training complete")

        # return trained parameters
        all_params["trainable"] = active_params
        all_opt_states = active_opt_states

        u, v, exx, exy, eyy = self._predict(all_params, x_batch_global, jmaps, model_fns, dim=dim)
        
        return all_params, u, v, exx, exy, eyy, x_batch_global

    def _report(self, i, start0, start1, report_time,
                all_params, all_opt_states,
                active_opt_states, active_params,
                lossval):
        "Report results"

        c = self.c
        summary_,test_,model_save_ = [(i % f == 0) for f in
                                      [c.summary_freq, c.test_freq, c.model_save_freq]]
        if lossval is None:
            loss_val = None
        else:
            loss_val = lossval.item()
            
        if summary_ or model_save_:

            # print summary
            if i != 0 and summary_:
                rate = c.summary_freq / (time.time()-start1-report_time)
                self._print_summary(i, lossval.item(), rate, start0)
                start1, report_time = time.time(), 0.

            if i != 0 and model_save_:

                start2 = time.time()

                # merge latest params
                all_params["trainable"] = active_params
                all_opt_states = active_opt_states
                
                self._save_model(i, (i, all_params, all_opt_states, jnp.array(loss_val) if loss_val is not None else jnp.array(jnp.nan)))

                report_time += time.time()-start2

        return lossval, start1, report_time
    
    def _predict(self, all_params, x_batch_global, jmaps, model_fns, dim=2):
        if x_batch_global.shape[0] > 512**2:
            batch_size = 512**2
            r = x_batch_global.shape[0]%batch_size
            shift = batch_size-r if r else 0
            irange = jnp.arange(0, x_batch_global.shape[0], batch_size)# (k)
            irange = irange.at[-1].add(-shift)
            
            def take_batch(i):
                return jax.lax.dynamic_slice(
                    x_batch_global,
                    (i, 0),
                    (batch_size, x_batch_global.shape[1])
                )
            
            x_batches = jax.vmap(take_batch)(irange)   # (m, batch_size, 2)
        else:
            batch_size = x_batch_global.shape[0]
            x_batches = x_batch_global[None, ...]   # (1, batch_size, 2)
            
        u = v = exx = exy = eyy = jnp.zeros_like(all_params["static"]["problem"]["ref_img"])
        if dim ==2:
            for i, x_batch in enumerate(x_batches):
                x_batch = x_batch.astype(jnp.float32)
                logger.info(f"Predicting on batch {i+1}/{x_batches.shape[0]}..")
                
                ujs = PINN_predict_jit(all_params, x_batch, jmaps, model_fns)
                u_, v_, ux_, uy_, vx_, vy_ = ujs
                exx_, exy_, eyy_ = ux_, (uy_+vx_)/2, vy_
                ys = x_batch[:, 1].astype(jnp.int32)
                xs = x_batch[:, 0].astype(jnp.int32)
                u = u.at[ys, xs].set(u_.flatten())
                v = v.at[ys, xs].set(v_.flatten())
                exx = exx.at[ys, xs].set(exx_.flatten())
                exy = exy.at[ys, xs].set(exy_.flatten())
                eyy = eyy.at[ys, xs].set(eyy_.flatten())
        else:
            for i, x_batch in enumerate(x_batches):
                x_batch = x_batch.astype(jnp.float32)
                logger.info(f"Predicting on batch {i+1}/{x_batches.shape[0]}..")
                
                ujs = PINN_predict_jit(all_params, x_batch, jmaps, model_fns)
                u_, v_ = ujs
                ys = x_batch[:, 1].astype(jnp.int32)
                xs = x_batch[:, 0].astype(jnp.int32)
                u = u.at[ys, xs].set(u_.flatten())
                v = v.at[ys, xs].set(v_.flatten())
        return u, v, exx, exy, eyy
            
            
    
    