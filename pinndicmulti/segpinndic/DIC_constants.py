from pinndicmulti.DIC_importlib import pickle, np, optax, socket, seed
from pinndicmulti.segpinndic.utils import io

from pinndicmulti.segpinndic import DIC_domains, DIC_decompositions, DIC_networks, DIC_schedulers, DIC_problem
from pinndicmulti.segpinndic.DIC_readImg import BufferManager

class ConstantsBase:

    # note can set members freely, below only for index assignment
    def __getitem__(self, key):
        if key not in vars(self): raise KeyError(f'key "{key}" not defined in class')
        return getattr(self, key)
    def __setitem__(self, key, item):
        if key not in vars(self): raise KeyError(f'key "{key}" not defined in class')
        setattr(self, key, item)

    def __str__(self):
        s = repr(self) + '\n'
        for k in vars(self): s+=f"{k}: {self[k]}\n"
        return s
    
    # calculated variables
    @property
    def summary_out_dir(self):
        return f"{self.work_dir}/summaries/{self.run}/"
    @property
    def model_out_dir(self):
        return f"{self.work_dir}/models/{self.run}/"
    @property
    def calib_out_dir(self):
        return f"{self.work_dir}/calibration/"
    @property
    def reconstruct_out_dir(self):
        return f"{self.work_dir}/reconstruct/"
    @property
    def deformation_out_dir(self):
        return f"{self.work_dir}/deformation/"
    @property
    def fig_out_dir(self):
        return f"{self.work_dir}/figs/{self.run}/"

    def get_outdirs(self):
        io.get_dir(self.summary_out_dir)
        io.get_dir(self.model_out_dir)
        io.get_dir(self.calib_out_dir)
        io.get_dir(self.reconstruct_out_dir)
        io.get_dir(self.deformation_out_dir)
        io.get_dir(self.fig_out_dir)

    def clear_outdirs(self):
        io.clear_dir(self.summary_out_dir)
        io.clear_dir(self.model_out_dir)
        io.clear_dir(self.fig_out_dir)

    def save_constants_file(self):
        "Save a constants to file in self.summary_out_dir"
        with open(self.summary_out_dir + f"constants_{self.run}.txt", 'w') as f:
            for k in vars(self): f.write(f"{k}: {self[k]}\n")
        with open(self.summary_out_dir + f"constants_{self.run}.pickle", 'wb') as f:
            pickle.dump(vars(self), f)

    @property
    def constants_file(self):
        return self.summary_out_dir + f"constants_{self.run}.pickle"
    
    
# main constants class
class Constants(ConstantsBase):

    def __init__(self, DICconfig, roi_id):
        "Defines global constants for model"

        # Define results directories
        self.work_dir = DICconfig.work_dir
        
        # image shape
        self.roi_id = roi_id
        self.pair_idx = 0  # updated per DIC pair by the pipeline
        roi = BufferManager.mask[roi_id]
        roi = np.asarray(roi)
        assert roi.ndim == 2, "ROI must be (H, W) bool array"
        H, W = roi.shape
        
        ys, xs = np.where(roi)
        if len(xs) == 0:
            raise ValueError("ROI is empty")

        xmin_roi = xs.min()
        xmax_roi = xs.max()
        ymin_roi = ys.min()
        ymax_roi = ys.max()
        
        self.roi_bbox = (xmin_roi, xmax_roi, ymin_roi, ymax_roi)

        # Define domain
        self.domain = DIC_domains.RectangularDomainND
        self.domain_init_kwargs = dict(
            xmin=np.array([xmin_roi, ymin_roi]),
            xmax=np.array([xmax_roi, ymax_roi])
        )
        
        # Define problem
        loss_name = DICconfig.loss_fun
        if not hasattr(DIC_problem, loss_name):
            raise ValueError(f"Unknown network: {loss_name}")
        self.problem = getattr(DIC_problem, loss_name)
        self.problem_init_kwargs = dict(
            ref_img = None,
            QKBQKT_def = None,
            mask = None,
            degree = 5,
            znssd_kernel_size = getattr(DICconfig, "znssd_kernel_size", 7),
        )

        # Define domain decomposition
        nx, ny = getattr(DICconfig, "n_subdomains", [4, 8])
        if nx*ny == 1:
            self.subdomain_xs = [
                np.linspace(xmin_roi, xmax_roi, 5), 
                np.linspace(ymin_roi, ymax_roi, 5)
                ]
            self.subdomain_ws = get_subdomain_ws(self.subdomain_xs, 2.4)
        else:
            nodes_xs = [
                np.linspace(xmin_roi, xmax_roi, nx+1), 
                np.linspace(ymin_roi, ymax_roi, ny+1)
            ]
            self.subdomain_ws = [
                1.6 * np.min(np.diff(nodes)) * np.ones(nx_or_ny_current)                 for nodes, nx_or_ny_current in zip(nodes_xs, [nx, ny])
            ]
            self.subdomain_xs = [
                (nodes[:-1] + nodes[1:]) / 2 
                for nodes in nodes_xs
            ]
            
        self.decomposition = DIC_decompositions.RectangularDecompositionND
        self.decomposition_init_kwargs = dict(
            subdomain_xs=self.subdomain_xs,
            subdomain_ws=self.subdomain_ws,
            unnorm=(np.array([0., 0.]), np.array([1., 1.])),
            )

        # Define neural network
        net_name = DICconfig.network
        if not hasattr(DIC_networks, net_name):
            raise ValueError(f"Unknown network: {net_name}")
        self.network = getattr(DIC_networks, net_name)
        self.network_init_kwargs = dict(
            layer_sizes=[2] + DICconfig.hidden_units + [2],
            )

        # Define scheduler
        self.n_steps = getattr(DICconfig, "adam_epochs", 15000)
        scheduler_name = DICconfig.train_schedulers
        if not hasattr(DIC_schedulers, scheduler_name):
            raise ValueError(f"Unknown scheduler: {scheduler_name}")
        self.scheduler = getattr(DIC_schedulers, scheduler_name)
        if scheduler_name == "AllActiveSchedulerND":
            self.scheduler_kwargs = dict()
            scheduler_rename = "All" 
        elif scheduler_name == "PointSchedulerRectangularND":
            self.scheduler_kwargs = dict(
                point=np.array([(xmin_roi+xmax_roi)/2, (ymin_roi+ymax_roi)/2])
            )
            scheduler_rename = "Point"
        elif scheduler_name == "LineSchedulerRectangularND":
            scheduler_rename = "Line"
            if nx > ny:
                self.scheduler_kwargs = dict(
                    point=np.array([xmin_roi]),
                    iaxis=1
                )
            else:
                self.scheduler_kwargs = dict(
                    point=np.array([ymin_roi]),
                    iaxis=0
                )

        # Define optimisation parameters
        self.ns = ((1,),)# batch_shape for placeholder
        self.n_test = (200,)# batch_shape for test data
        self.seed_lr = getattr(DICconfig, "adam_lr", 1e-3)
        self.dic_lr = getattr(DICconfig, "dic_lr", self.seed_lr)
        self.optimiser = optax.adam
        self.optimiser_kwargs = dict(
            learning_rate=self.dic_lr
            )
        self.seed = seed

        # L-BFGS refinement parameters
        self.lbfgs_epochs = getattr(DICconfig, "lbfgs_epochs", 0)
        self.lbfgs_history_size = getattr(DICconfig, "lbfgs_history_size", 10)
        self.lbfgs_maxls = getattr(DICconfig, "lbfgs_maxls", 15)
        self.lbfgs_lr = getattr(DICconfig, "lbfgs_lr", 1.0)

        # seed supervised pre-training
        self.seed_pos = None        # (N,2) seed point coordinates
        self.seed_uv = None         # (N,2) seed point displacements
        self.seed_train_epochs = getattr(DICconfig, "seed_train_epochs", 0)  # number of seed pre-training steps
        self.seed_smooth_lambda = getattr(DICconfig, "seed_smooth_lambda", 0.0)    # gradient-norm penalty weight
        self.seed_smooth_npoints = getattr(DICconfig, "seed_smooth_npoints", 0)    # collocation points for smoothness
        self.seed_sd_threshold = getattr(DICconfig, "seed_sd_threshold", 5.0)      # min sd (pixels) to enable seed training

        # Define summary output parameters
        self.summary_freq = getattr(DICconfig, "summary_freq", 1000)        # outputs train stats to command line
        self.test_freq = getattr(DICconfig, "test_freq", 1000)              # outputs test stats to plot / file / command line
        self.test_flag = getattr(DICconfig, "test_flag", False)              # whether to print test information
        self.model_save_freq = getattr(DICconfig, "model_save_freq", 10000)
        self.show_figures = getattr(DICconfig, "show_figures", False)       # whether to show figures
        self.save_figures = getattr(DICconfig, "save_figures", True)        # whether to save figures
        self.clear_output = getattr(DICconfig, "clear_output", False)       # whether to clear ipython output periodically

        # other constants
        self.hostname = socket.gethostname().lower()
        
        # Define run
        if nx*ny == 1:
            self.run = net_name + "_" + loss_name
        else:
            self.run = net_name + "_" + scheduler_rename + "_" + f"{nx}x{ny}"  + "_" + loss_name

    def load_kwargs(self, **kwargs):
        # overwrite with input arguments
        for key in kwargs.keys(): self[key] = kwargs[key]# invokes __setitem__ in ConstantsBase

def print_c_dicts(c_dicts):
    "Pretty print a list of c_dicts"

    # get full list of keys
    keys = []
    for c_dict in c_dicts[::-1]:
        for k in c_dict.keys():
            if k not in keys: keys.append(k)

    for k in keys:
        print(f"{k}: ",end="")
        for i,c_dict in enumerate(c_dicts):
            if k in c_dict.keys(): item=str(c_dict[k])
            else: item='None'
            if i == len(c_dicts)-1: print(f"{item}",end="")
            else: print(f"{item} | ",end="")
        print("")
        
def get_subdomain_ws(subdomain_xs, width):
    return [width*np.min(np.diff(x))*np.ones_like(x) for x in subdomain_xs]


if __name__ == "__main__":
    from segpinndic.DIC_config import DIC_2D_config_txt
    config = DIC_2D_config_txt("./config/PINN-DIC-2D.txt")
    
    # BufferManager.refImg = np.zeros((100, 200))# dummy image for testing
    # constants = Constants(config)
    # print(constants)