from pinndicmulti.DIC_importlib import jnp, random

class Network:
    """Base neural network class to be inherited by different neural network classes.

    Note all methods in this class are jit compiled / used by JAX,
    so they must not include any side-effects!
    (A side-effect is any effect of a function that doesn’t appear in its output)
    This is why only static methods are defined.
    """

    @staticmethod
    def init_params(*args):
        """Initialise class parameters.
        Returns tuple of dicts ({k: pytree}, {k: pytree}) containing static and trainable parameters"""
        raise NotImplementedError

    @staticmethod
    def network_fn(params, x):
        """Forward model, for a SINGLE point with shape (xd,)"""
        raise NotImplementedError
    
    
class FCN(Network):

    @staticmethod
    def init_params(key, layer_sizes):
        keys = random.split(key, len(layer_sizes)-1)
        params = [FCN._random_layer_params(k, m, n)
                for k, m, n in zip(keys, layer_sizes[:-1], layer_sizes[1:])]
        trainable_params = {"layers": params}
        return {}, trainable_params

    @staticmethod
    def _random_layer_params(key, m, n):
        "Create a random layer parameters"

        w_key, b_key = random.split(key)
        v = jnp.sqrt(1/m)
        w = random.uniform(w_key, (n, m), minval=-v, maxval=v)
        b = random.uniform(b_key, (n,), minval=-v, maxval=v)
        return w,b

    @staticmethod
    def network_fn(params, x):
        params = params["trainable"]["network"]["subdomain"]["layers"]
        for w, b in params[:-1]:
            x = jnp.dot(w, x) + b
            x = jnp.tanh(x)
        w, b = params[-1]
        x = jnp.dot(w, x) + b
        return x
    
# class FourierNet(Network):

#     @staticmethod
#     def init_params(key, layer_sizes, mapping_size=64, scale=10.0):
#         """
#         layer_sizes example:
#         [2,32,32,32,2]

#         Actual network becomes:
#         [2, mapping_size, 32, 32, 32, 2]
#         """

#         in_dim = layer_sizes[0]

#         # split keys
#         key_B, key_layers = random.split(key)

#         # Fourier matrix
#         B = random.normal(key_B, (mapping_size // 2, in_dim)) * scale

#         # construct new layer sizes
#         new_layer_sizes = [mapping_size] + layer_sizes[1:]

#         keys = random.split(key_layers, len(new_layer_sizes)-1)

#         params = [
#             FourierNet._random_layer_params(k, m, n)
#             for k, m, n in zip(keys, new_layer_sizes[:-1], new_layer_sizes[1:])
#         ]

#         static_params = {
#             "fourier": {
#                 "B": B
#             }
#         }

#         trainable_params = {
#             "layers": params
#         }

#         return static_params, trainable_params

#     @staticmethod
#     def _random_layer_params(key, m, n):

#         w_key, b_key = random.split(key)

#         v = jnp.sqrt(1/m)

#         w = random.uniform(w_key, (n, m), minval=-v, maxval=v)
#         b = random.uniform(b_key, (n,), minval=-v, maxval=v)

#         return w, b

#     @staticmethod
#     def _fourier_feature(B, x):
#         """
#         x shape: (xd,)
#         B shape: (mapping_size/2, xd)
#         """

#         x_proj = B @ x

#         return jnp.concatenate(
#             [jnp.sin(x_proj), jnp.cos(x_proj)],
#             axis=0
#         )

#     @staticmethod
#     def network_fn(params, x):

#         B = params["static"]["network"]["fourier"]["B"]

#         layers = params["trainable"]["network"]["subdomain"]["layers"]

#         # Fourier mapping
#         x = FourierNet._fourier_feature(B, x)

#         # MLP
#         for w, b in layers[:-1]:
#             x = jnp.dot(w, x) + b
#             x = jnp.tanh(x)

#         w, b = layers[-1]

#         x = jnp.dot(w, x) + b

#         return x

class FourierNet(Network):

    @staticmethod
    def init_params(key,
                    layer_sizes,
                    mapping_size=64,
                    sigma_list=(1,4,8,16),
                    alpha=0.8):

        in_dim = layer_sizes[0]

        key_B, key_layers = random.split(key)

        K = len(sigma_list)

        m_total = mapping_size // 2

        # ----- compute mk according to Eq.(27) -----

        k_vals = jnp.arange(1, K+1)

        weights = (k_vals / K) ** alpha
        weights = weights / jnp.sum(weights)

        mk_list = jnp.round(m_total * weights).astype(int)

        # ensure sum equals m_total
        diff = m_total - jnp.sum(mk_list)
        mk_list = mk_list.at[-1].add(diff)

        # ----- build multi-scale B -----

        keys = random.split(key_B, K)

        B_list = []

        for i, sigma in enumerate(sigma_list):

            mk = int(mk_list[i])

            Bk = random.normal(
                keys[i],
                (mk, in_dim)
            ) * sigma

            B_list.append(Bk)

        B = jnp.concatenate(B_list, axis=0)

        # encoded input dimension
        encoded_dim = in_dim + 2 * B.shape[0]

        new_layer_sizes = [encoded_dim] + layer_sizes[1:]

        keys = random.split(key_layers, len(new_layer_sizes)-1)

        params = [
            FourierNet._random_layer_params(k, m, n)
            for k, m, n in zip(keys, new_layer_sizes[:-1], new_layer_sizes[1:])
        ]

        static_params = {
            "fourier": {
                "B": B
            }
        }

        trainable_params = {
            "layers": params
        }

        return static_params, trainable_params
    
    @staticmethod
    def _random_layer_params(key, m, n):

        w_key, b_key = random.split(key)

        v = jnp.sqrt(1/m)

        w = random.uniform(w_key, (n, m), minval=-v, maxval=v)
        b = random.uniform(b_key, (n,), minval=-v, maxval=v)

        return w, b
    
    @staticmethod
    def _fourier_feature(B, x):

        x_proj = 2 * jnp.pi * (B @ x)

        return jnp.concatenate(
            [jnp.sin(x_proj), jnp.cos(x_proj)],
            axis=0
        )
        
    @staticmethod
    def network_fn(params, x):

        B = params["static"]["network"]["fourier"]["B"]
        layers = params["trainable"]["network"]["subdomain"]["layers"]

        # Fourier features
        fourier = FourierNet._fourier_feature(B, x)

        # concatenate original coordinates
        x = jnp.concatenate([x, fourier], axis=0)

        # MLP
        for w, b in layers[:-1]:
            x = jnp.dot(w, x) + b
            x = jnp.tanh(x)

        w, b = layers[-1]

        x = jnp.dot(w, x) + b

        return x
    
class ResNet(Network):

    @staticmethod
    def init_params(key, layer_sizes):

        keys = random.split(key, len(layer_sizes)-1)
        out_dim = layer_sizes[-1]
        
        params = [
            ResNet._random_layer_params(k, m, n, out_dim)
            for k, m, n in zip(keys, layer_sizes[:-1], layer_sizes[1:])
        ]
        trainable_params = {"layers": params}

        return {}, trainable_params


    @staticmethod
    def _random_layer_params(key, m, n, out_dim):

        k1, k2 = random.split(key)
        v = jnp.sqrt(1.0 / m)
        # hidden layer
        w = random.uniform(k1, (n, m), minval=-v, maxval=v)
        b = jnp.zeros((n,))
        # residual output head
        w_out = random.uniform(k2, (out_dim, n), minval=-v, maxval=v)
        b_out = jnp.zeros((out_dim,))
        return w, b, w_out, b_out


    @staticmethod
    def network_fn(params, x):

        params = params["trainable"]["network"]["subdomain"]["layers"]
        h = x
        outputs = []
        for i, (w, b, w_out, b_out) in enumerate(params):
            h = jnp.dot(w, h) + b
            if i < len(params)-1:
                h = jnp.tanh(h)
            y_i = jnp.dot(w_out, h) + b_out
            outputs.append(y_i)
        y = sum(outputs) / len(outputs)
        return y
    
class AdaptiveResNet(Network):

    @staticmethod
    def init_params(key, layer_sizes):

        keys = random.split(key, len(layer_sizes)-1)
        out_dim = layer_sizes[-1]
        
        params = [
            AdaptiveResNet._random_layer_params(k, m, n, out_dim)
            for k, m, n in zip(keys, layer_sizes[:-1], layer_sizes[1:])
        ]
        trainable_params = {"layers": params}

        return {}, trainable_params


    @staticmethod
    def _random_layer_params(key, m, n, out_dim):

        k1, k2 = random.split(key)
        v = jnp.sqrt(1.0 / m)
        # hidden layer
        w = random.uniform(k1, (n, m), minval=-v, maxval=v)
        b = jnp.zeros((n,))
        a = jnp.ones_like(b)
        # residual output head
        w_out = random.uniform(k2, (out_dim, n), minval=-v, maxval=v)
        b_out = jnp.zeros((out_dim,))
        a_out = jnp.ones_like(b_out)
        return w, b, a, w_out, b_out, a_out


    @staticmethod
    def network_fn(params, x):

        params = params["trainable"]["network"]["subdomain"]["layers"]
        h = x
        outputs = []
        for i, (w, b, a, w_out, b_out, a_out) in enumerate(params):
            h = jnp.dot(w, h) + b
            
            if i < len(params)-1:
                h = jnp.tanh(h)
                h = a * h
            y_i = jnp.dot(w_out, h) + b_out
            y_i = a_out * y_i
            outputs.append(y_i)
        y = jnp.mean(jnp.stack(outputs), axis=0)
        return y

class AdaptiveFCN(Network):

    @staticmethod
    def init_params(key, layer_sizes):

        keys = random.split(key, len(layer_sizes)-1)
        params = [AdaptiveFCN._random_layer_params(k, m, n)
                for k, m, n in zip(keys, layer_sizes[:-1], layer_sizes[1:])]
        trainable_params = {"layers": params}
        return {}, trainable_params

    @staticmethod
    def _random_layer_params(key, m, n):
        "Create a random layer parameters"

        w_key, b_key = random.split(key)
        v = jnp.sqrt(1/m)
        w = random.uniform(w_key, (n, m), minval=-v, maxval=v)
        b = random.uniform(b_key, (n,), minval=-v, maxval=v)
        a = jnp.ones_like(b)
        return w,b,a

    @staticmethod
    def network_fn(params, x):
        params = params["trainable"]["network"]["subdomain"]["layers"]
        for w, b, a in params[:-1]:
            x = jnp.dot(w, x) + b
            x = a*jnp.tanh(x/a)
        w, b, _ = params[-1]
        x = jnp.dot(w, x) + b
        return x

class SIREN(FCN):

    @staticmethod
    def _random_layer_params(key, m, n):
        "Create a random layer parameters"

        w_key, b_key = random.split(key)
        v = jnp.sqrt(1/m)
        w = random.uniform(w_key, (n, m), minval=-v, maxval=v)
        # b = random.uniform(b_key, (n,), minval=-v, maxval=v)
        b = jnp.zeros((n,))
        return w,b

    @staticmethod
    def network_fn(params, x):
        params = params["trainable"]["network"]["subdomain"]["layers"]
        for w, b in params[:-1]:
            x = jnp.dot(w, x) + b
            x = jnp.sin(x)
        w, b = params[-1]
        x = jnp.dot(w, x) + b
        return x

class AdaptiveSIREN(Network):

    @staticmethod
    def init_params(key, layer_sizes):

        keys = random.split(key, len(layer_sizes)-1)
        params = [AdaptiveSIREN._random_layer_params(k, m, n)
                for k, m, n in zip(keys, layer_sizes[:-1], layer_sizes[1:])]
        trainable_params = {"layers": params}
        return {}, trainable_params

    @staticmethod
    def _random_layer_params(key, m, n):
        "Create a random layer parameters"

        w_key, b_key = random.split(key)
        v = jnp.sqrt(1/m)
        w = random.uniform(w_key, (n, m), minval=-v, maxval=v)
        # b = random.uniform(b_key, (n,), minval=-v, maxval=v)
        b = jnp.zeros((n,))
        c,o = jnp.ones_like(b), jnp.ones_like(b)
        return w,b,c,o

    @staticmethod
    def network_fn(params, x):
        params = params["trainable"]["network"]["subdomain"]["layers"]
        for w,b,c,o in params[:-1]:
            x = jnp.dot(w, x) + b
            x = c*jnp.sin(o*x)
        w,b,_,_ = params[-1]
        x = jnp.dot(w, x) + b
        return x


def norm(mu, sd, x):
    return (x-mu)/sd

def unnorm(mu, sd, x):
    return x*sd + mu

if __name__ == "__main__":

    x = jnp.ones(2)
    key = random.PRNGKey(0)
    layer_sizes = [2,16,32,16,1]
    for NN in [FCN, AdaptiveFCN, SIREN, AdaptiveSIREN]:
        network = NN
        ps_ = network.init_params(key, layer_sizes)
        params = {"static":{"network":ps_[0]}, "trainable":{"network":{"subdomain":ps_[1]}}}
        print(x.shape, network.network_fn(params, x).shape, NN.__name__)