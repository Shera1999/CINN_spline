import math
import matplotlib.pyplot as plt
from scipy.stats import special_ortho_group
import torch
import torch.nn as nn
import FrEIA.framework as ff
import FrEIA.modules as fm
import numpy as np
import torch.nn.functional as F
from typing import Callable


class VBLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_prec=1.0, _map=False, std_init=-9):
        super(VBLinear, self).__init__()
        self.n_in = in_features
        self.n_out = out_features
        self.map = _map
        self.prior_prec = prior_prec
        self.random = None
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.mu_w = nn.Parameter(torch.Tensor(out_features, in_features))
        self.logsig2_w = nn.Parameter(torch.Tensor(out_features, in_features))
        self.std_init = std_init
        self.reset_parameters()

    def enable_map(self):
        self.map = True

    def disenable_map(self):
        self.map = False

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.mu_w.size(1))
        self.mu_w.data.normal_(0, stdv)
        self.logsig2_w.data.zero_().normal_(self.std_init, 0.001)
        self.bias.data.zero_()

    def reset_random(self):
        self.random = None
        self.map = False

    def sample_random_state(self):
        return torch.randn_like(self.logsig2_w).detach().cpu().numpy()

    def import_random_state(self, state):
        self.random = torch.tensor(state, device=self.logsig2_w.device,
                                   dtype=self.logsig2_w.dtype)

    def KL(self):
        return 0.5 * (self.prior_prec * (self.mu_w.pow(2) + self.logsig2_w.exp())
                        - self.logsig2_w - 1 - math.log(self.prior_prec)).sum()

    def forward(self, input):
        if self.training:
            # local reparameterization trick is more efficient and leads to
            # an estimate of the gradient with smaller variance.
            # https://arxiv.org/pdf/1506.02557.pdf
            mu_out = F.linear(input, self.mu_w, self.bias)
            s2_w = self.logsig2_w.exp()
            var_out = F.linear(input.pow(2), s2_w) + 1e-8
            return mu_out + var_out.sqrt() * torch.randn_like(mu_out)

        else:
            if self.map:
                return F.linear(input, self.mu_w, self.bias)

            if self.random is None:
                self.random = torch.randn_like(self.logsig2_w)
            s2_w = self.logsig2_w.exp()
            weight = self.mu_w + s2_w.sqrt() * self.random
            return F.linear(input, weight, self.bias) + 1e-8

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.n_in}) -> ({self.n_out})"

class Subnet(nn.Module):
    """ This class constructs a subnet for the coupling blocks """

    def __init__(self, num_layers, size_in, size_out, internal_size=None, dropout=0.0,
                 layer_class=nn.Linear, layer_args={}, layer_norm=None, layer_act="nn.ReLU"):
        """
            Initializes subnet class.

            Parameters:
            size_in: input size of the subnet
            size: output size of the subnet
            internal_size: hidden size of the subnet. If None, set to 2*size
            dropout: dropout chance of the subnet
        """
        super().__init__()
        if internal_size is None:
            internal_size = size_out * 2
        if num_layers < 1:
            raise(ValueError("Subnet size has to be 1 or greater"))
        self.layer_list = []
        for n in range(num_layers - 1):
            if isinstance(internal_size, list):
                input_dim, output_dim = internal_size[n], internal_size[n+1]
            else:
                input_dim, output_dim = internal_size, internal_size
            if n == 0:
                input_dim = size_in

            self.layer_list.append(layer_class[n](input_dim, output_dim, **(layer_args[n])))

            if dropout > 0:
                self.layer_list.append(nn.Dropout(p=dropout))
            
            if layer_norm is not None:
                self.layer_list.append(eval(layer_norm)(output_dim))

            self.layer_list.append(eval(layer_act)())
        
        # separating last linear/VBL layer
        #output_dim = size_out
        self.layer_list.append(layer_class[-1](output_dim, size_out, **(layer_args[-1]) ))

        self.layers = nn.Sequential(*self.layer_list)

        final_layer_name = str(len(self.layers) - 1)
        for name, param in self.layers.named_parameters():
            if name[0] == final_layer_name and "logsig2_w" not in name:
                param.data.zero_()

    def forward(self, x):
        return self.layers(x)
    
########

class LogTransformation(fm.InvertibleModule):
    def __init__(self, dims_in, dims_c=None, alpha = 0., alpha_logit=0.):
        super().__init__(dims_in, dims_c)
        self.alpha = alpha
        self.alpha_logit = alpha_logit

    def forward(self, x, c=None, rev=False, jac=True):
        x, = x
        if rev:
            z = torch.exp(x) - self.alpha
            #z2 = torch.exp(x[:, -4:])
            #z3 = x[:, 369].reshape(-1, 1)

            #z = torch.cat((z1,z3,z2), dim=1)
            jac = torch.sum( x, dim=1)
        else:
            z = torch.log(x + self.alpha)
            #z3 = x[:,369].reshape(-1, 1)
            #z2 = torch.log(x[:, -4:])     #*(1-2*self.alpha_logit) + self.alpha_logit)

            #z = torch.cat((z1,z3,z2), dim=1)
            jac = - torch.sum( z, dim=1)
        return (z, ), torch.tensor([0.], device=x.device) # jac

    def output_dims(self, input_dims):
        return input_dims
######
class RationalQuadraticSplineBlock(fm.InvertibleModule):

    DEFAULT_MIN_BIN_WIDTH = 1e-3
    DEFAULT_MIN_BIN_HEIGHT = 1e-3
    DEFAULT_MIN_DERIVATIVE = 1e-3

    def __init__(self, dims_in, dims_c=[],
                 subnet_constructor: Callable = None,
                 num_bins: int = 10,
                 bounds_init: float = 1.,
                 permute_soft: bool = False,
                 tails='linear',
                 bounds_type="SOFTPLUS"):

        super().__init__(dims_in, dims_c)
        channels = dims_in[0][0]
        # rank of the tensors means 1d, 2d, 3d tensor etc.
        self.input_rank = len(dims_in[0]) - 1
        # tuple containing all dims except for batch-dim (used at various points)
        self.sum_dims = tuple(range(1, 2 + self.input_rank))
        if len(dims_c) == 0:
            self.conditional = False
            self.condition_channels = 0
        else:
            assert tuple(dims_c[0][1:]) == tuple(dims_in[0][1:]), \
                F"Dimensions of input and condition don't agree: {dims_c} vs {dims_in}."
            self.conditional = True
            self.condition_channels = sum(dc[0] for dc in dims_c)

        split_len1 = channels - channels // 2
        split_len2 = channels // 2
        self.splits = [split_len1, split_len2]
        self.num_bins = num_bins
        if self.DEFAULT_MIN_BIN_WIDTH * self.num_bins > 1.0:
            raise ValueError('Minimal bin width too large for the number of bins')
        if self.DEFAULT_MIN_BIN_HEIGHT * self.num_bins > 1.0:
            raise ValueError('Minimal bin height too large for the number of bins')

        try:
            self.permute_function = {0: F.linear,
                                     1: F.conv1d,
                                     2: F.conv2d,
                                     3: F.conv3d}[self.input_rank]
        except KeyError:
            raise ValueError(f"Data is {1 + self.input_rank}D. Must be 1D-4D.")



        if bounds_type == 'SIGMOID':
            bounds = 2. - np.log(10. / bounds_init - 1.)
            self.bounds_activation = (lambda a: 10 * torch.sigmoid(a - 2.))
        elif bounds_type == 'SOFTPLUS':
            bounds = 2. * np.log(np.exp(0.5 * 10. * bounds_init) - 1)
            self.softplus = nn.Softplus(beta=0.5)
            self.bounds_activation = (lambda a: 0.1 * self.softplus(a))
        elif bounds_type == 'EXP':
            bounds = np.log(bounds_init)
            self.bounds_activation = (lambda a: torch.exp(a))
        elif bounds_type == 'LIN':
            bounds = bounds_init
            self.bounds_activation = (lambda a: a)
        else:
            raise ValueError('Global affine activation must be "SIGMOID", "SOFTPLUS" or "EXP"')

        self.in_channels         = channels
        self.bounds = self.bounds_activation(torch.ones(1, self.splits[1], *([1] * self.input_rank)) * float(bounds))
        self.tails = tails

        if permute_soft:
            w = special_ortho_group.rvs(channels)
        else:
            w = np.zeros((channels, channels))
            for i, j in enumerate(np.random.permutation(channels)):
                w[i, j] = 1.

        # self.w_perm = nn.Parameter(torch.FloatTensor(w).view(channels, channels, *([1] * self.input_rank)),
        #                            requires_grad=False)
        # self.w_perm_inv = nn.Parameter(torch.FloatTensor(w.T).view(channels, channels, *([1] * self.input_rank)),
        #                                requires_grad=False)
        self.w_perm = nn.Parameter(torch.Tensor(w).view(channels, channels, *([1] * self.input_rank)),
                                   requires_grad=False)
        self.w_perm_inv = nn.Parameter(torch.Tensor(w.T).view(channels, channels, *([1] * self.input_rank)),
                                       requires_grad=False)

        if subnet_constructor is None:
            raise ValueError("Please supply a callable subnet_constructor"
                             "function or object (see docstring)")
        self.subnet = subnet_constructor(self.splits[0] + self.condition_channels, (3 * self.num_bins - 1) * self.splits[1])
        self.last_jac = None

    def _unconstrained_rational_quadratic_spline(self,
                                   inputs,
                                   theta,
                                   rev=False):

        inside_interval_mask = torch.all((inputs >= -self.bounds) & (inputs <= self.bounds),
                                         dim = -1)
        outside_interval_mask = ~inside_interval_mask

        masked_outputs = torch.zeros_like(inputs)
        masked_logabsdet = torch.zeros(inputs.shape[0], dtype=inputs.dtype).to(inputs.device)

        min_bin_width=self.DEFAULT_MIN_BIN_WIDTH
        min_bin_height=self.DEFAULT_MIN_BIN_HEIGHT
        min_derivative=self.DEFAULT_MIN_DERIVATIVE


        if self.tails == 'linear':
            masked_outputs[outside_interval_mask] = inputs[outside_interval_mask]
            masked_logabsdet[outside_interval_mask] = 0

        else:
            raise RuntimeError('{} tails are not implemented.'.format(self.tails))
        inputs = inputs[inside_interval_mask]
        theta = theta[inside_interval_mask, :]
        bound = torch.min(self.bounds)

        left = -bound
        right = bound
        bottom = -bound
        top = bound

        #if not rev and (torch.min(inputs) < left or torch.max(inputs) > right):
        #    raise ValueError("Spline Block inputs are not within boundaries")
        #elif rev and (torch.min(inputs) < bottom or torch.max(inputs) > top):
        #    raise ValueError("Spline Block inputs are not within boundaries")

        unnormalized_widths = theta[...,:self.num_bins]
        unnormalized_heights = theta[...,self.num_bins:self.num_bins*2]
        unnormalized_derivatives = theta[...,self.num_bins*2:]

        unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))
        constant = np.log(np.exp(1 - min_derivative) - 1)
        unnormalized_derivatives[..., 0] = constant
        unnormalized_derivatives[..., -1] = constant



        widths = F.softmax(unnormalized_widths, dim=-1)
        widths = min_bin_width + (1 - min_bin_width * self.num_bins) * widths
        cumwidths = torch.cumsum(widths, dim=-1)
        cumwidths = F.pad(cumwidths, pad=(1, 0), mode='constant', value=0.0)
        cumwidths = (right - left) * cumwidths + left
        cumwidths[..., 0] = left
        cumwidths[..., -1] = right
        widths = cumwidths[..., 1:] - cumwidths[..., :-1]

        derivatives = min_derivative + F.softplus(unnormalized_derivatives)

        heights = F.softmax(unnormalized_heights, dim=-1)
        heights = min_bin_height + (1 - min_bin_height * self.num_bins) * heights
        cumheights = torch.cumsum(heights, dim=-1)
        cumheights = F.pad(cumheights, pad=(1, 0), mode='constant', value=0.0)
        cumheights = (top - bottom) * cumheights + bottom
        cumheights[..., 0] = bottom
        cumheights[..., -1] = top
        heights = cumheights[..., 1:] - cumheights[..., :-1]

        if rev:
            bin_idx = self.searchsorted(cumheights, inputs)[..., None]
        else:
            bin_idx = self.searchsorted(cumwidths, inputs)[..., None]

        input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
        input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

        input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
        delta = heights / widths
        input_delta = delta.gather(-1, bin_idx)[..., 0]

        input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
        input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]

        input_heights = heights.gather(-1, bin_idx)[..., 0]

        if rev:
            a = (((inputs - input_cumheights) * (input_derivatives
                                                 + input_derivatives_plus_one
                                                 - 2 * input_delta)
                  + input_heights * (input_delta - input_derivatives)))
            b = (input_heights * input_derivatives
                 - (inputs - input_cumheights) * (input_derivatives
                                                  + input_derivatives_plus_one
                                                  - 2 * input_delta))
            c = - input_delta * (inputs - input_cumheights)

            discriminant = b.pow(2) - 4 * a * c
            
            ######################################################################
            # Mini-Debug terminal
            """
            if not (discriminant >= 0).all():
                print(f"{discriminant=}, \n {a=}, \n {b=}, \n {c=}, \n {theta=}")
                while True:
                    inp = input()
                    print(inp)
                    if inp=="break":
                        break
                    try:
                        print(eval(inp), flush=True)
                    except:
                        print("Cannot do this", flush=True)
            """
            #######################################################################
            
            assert (discriminant >= 0).all()

            root = (2 * c) / (-b - torch.sqrt(discriminant))
            outputs = root * input_bin_widths + input_cumwidths

            theta_one_minus_theta = root * (1 - root)
            denominator = input_delta + ((input_derivatives + input_derivatives_plus_one - 2 * input_delta)
                                         * theta_one_minus_theta)
            derivative_numerator = input_delta.pow(2) * (input_derivatives_plus_one * root.pow(2)
                                                         + 2 * input_delta * theta_one_minus_theta
                                                         + input_derivatives * (1 - root).pow(2))
            logabsdet = - torch.log(derivative_numerator) + 2 * torch.log(denominator)

        else:
            theta = (inputs - input_cumwidths) / input_bin_widths
            theta_one_minus_theta = theta * (1 - theta)

            numerator = input_heights * (input_delta * theta.pow(2)
                                         + input_derivatives * theta_one_minus_theta)
            denominator = input_delta + ((input_derivatives + input_derivatives_plus_one - 2 * input_delta)
                                         * theta_one_minus_theta)
            outputs = input_cumheights + numerator / denominator

            derivative_numerator = input_delta.pow(2) * (input_derivatives_plus_one * theta.pow(2)
                                                         + 2 * input_delta * theta_one_minus_theta
                                                         + input_derivatives * (1 - theta).pow(2))
            logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

        logabsdet = torch.sum(logabsdet, dim=1)

        masked_outputs[inside_interval_mask], masked_logabsdet[inside_interval_mask] = outputs, logabsdet

        return masked_outputs, masked_logabsdet

    def searchsorted(self, bin_locations, inputs, eps=1e-6):
        bin_locations[..., -1] += eps
        return torch.sum(
            inputs[..., None] >= bin_locations,
            dim=-1
        ) - 1

    def _permute(self, x, rev=False):
        '''Performs the permutation and scaling after the coupling operation.
        Returns transformed outputs and the LogJacDet of the scaling operation.'''

        scale = torch.ones(x.shape[-1]).to(x.device)
        perm_log_jac = torch.sum(-torch.log(scale))
        if rev:
            return (self.permute_function(x * scale, self.w_perm_inv),
                    perm_log_jac)
        else:
            return (self.permute_function(x, self.w_perm) / scale,
                    perm_log_jac)

    def forward(self, x, c=[], rev=False, jac=True):
        '''See base class docstring'''
        self.bounds = self.bounds.to(x[0].device)
        
        # For debugging
        # print(np.exp(c[0].cpu().numpy()))
        #self.cond = torch.exp(c[0])
        #self.data = x[0]

        if rev:
            x, global_scaling_jac = self._permute(x[0], rev=True)
            x = (x,)

        x1, x2 = torch.split(x[0], self.splits, dim=1)

        if self.conditional:
            x1c = torch.cat([x1, *c], 1)
        else:
            x1c = x1

        if not rev:
            theta = self.subnet(x1c).reshape(x1c.shape[0], self.splits[1], 3*self.num_bins - 1)
            x2, j2 = self._unconstrained_rational_quadratic_spline(x2, theta, rev=False)
        else:
            theta = self.subnet(x1c).reshape(x1c.shape[0], self.splits[1], 3*self.num_bins - 1)
            x2, j2 = self._unconstrained_rational_quadratic_spline(x2, theta, rev=True)
        log_jac_det = j2
        x_out = torch.cat((x1, x2), 1)

        if not rev:
            x_out, global_scaling_jac = self._permute(x_out, rev=False)
        # add the global scaling Jacobian to the total.
        # trick to get the total number of non-channel dimensions:
        # number of elements of the first channel of the first batch member
        n_pixels = x_out[0, :1].numel()
        log_jac_det += (-1)**rev * n_pixels * global_scaling_jac
        return (x_out,), log_jac_det

    def output_dims(self, input_dims):
        return input_dims
    

class CINN(nn.Module):
    """ cINN model """

    def __init__(self, params, data, cond):
        """ Initializes model class.

        Parameters:
        params: Dict containing the network and training parameter
        data: Training data to initialize the norm layer
        cond: Conditions to the training data
        """
        super(CINN, self).__init__()
        self.params = params
        self.num_dim = data.shape[1]

        self.norm_m = None
        self.bayesian = params.get("bayesian", False)
        self.alpha = params.get("alpha", 1e-8)
        self.alpha_logit = params.get("alpha_logit", 1.0e-6)
        self.log_cond = params.get("log_cond", False)
        self.use_norm = False
        self.pre_subnet = None
        self.cond_dim = cond.shape[1]

        if self.bayesian:
            self.bayesian_layers = []

        self.initialize_normalization(data, cond)
        self.define_model_architecture(self.num_dim)
        print(self.model)

    def forward(self, x, c, rev=False, jac=True):
        if self.log_cond:
            c_norm = torch.log10(c) # use log10 for all the models (add a rescaling option)
        else:
            c_norm = c
        if self.pre_subnet:
            c_norm = self.pre_subnet(c_norm)
        return self.model.forward(x, c_norm, rev=rev, jac=jac)

    def get_layer_class(self, lay_params):
        lays = []
        for n in range(len(lay_params)):
            if lay_params[n] == 'vblinear':
                lays.append(VBLinear)
            if lay_params[n] == 'linear':
                lays.append(nn.Linear)
        return lays

    def get_layer_args(self, params):
        layer_class = params["sub_layers"]
        layer_args = []
        for n in range(len(layer_class)):
            n_args = {}
            if layer_class[n] == "vblinear":
                if "prior_prec" in params:
                    n_args["prior_prec"] = params["prior_prec"]
                if "std_init" in params:
                    n_args["std_init"] = params["std_init"]
            layer_args.append(n_args)
        return layer_args

    def get_constructor_func(self, params):
        """ Returns a function that constructs a subnetwork with the given parameters """
        if "sub_layers" in params:
            layer_class = params["sub_layers"]
            layer_class = self.get_layer_class(layer_class)
            layer_args = self.get_layer_args(params)
        else:
            layer_class = []
            layer_args = []
            for n in range(params.get("layers_per_block", 3)):
                dicts = {}
                if self.bayesian:
                    layer_class.append(VBLinear)
                    if "prior_prec" in params:
                        dicts["prior_prec"] = params["prior_prec"]
                    if "std_init" in params:
                        dicts["std_init"] = params["std_init"]
                else:
                    layer_class.append(nn.Linear)
                layer_args.append(dicts)
        #if "prior_prec" in params:
        #    layer_args["prior_prec"] = params["prior_prec"]
        #if "std_init" in params:
        #    layer_args["std_init"] = params["std_init"]
        #if "bias" in params:
        #    layer_args["bias"] = params["bias"]
        def func(x_in, x_out):
            subnet = Subnet(
                    params.get("layers_per_block", 3),
                    x_in, x_out,
                    internal_size = params.get("internal_size"),
                    dropout = params.get("dropout", 0.),
                    layer_class = layer_class,
                    layer_args = layer_args,
                    layer_norm = params.get("layer_norm", None),
                    layer_act = params.get("layer_act", "nn.ReLU"),
                    )
            if self.bayesian:
                self.bayesian_layers.extend(
                    layer for layer in subnet.layer_list if isinstance(layer, VBLinear))
            return subnet
        return func

    def get_coupling_block(self, params):
        """ Returns the class and keyword arguments for different coupling block types """
        constructor_fct = self.get_constructor_func(params)
        permute_soft = params.get("permute_soft")
        coupling_type = params.get("coupling_type", "affine")

        if coupling_type == "affine":
            CouplingBlock = fm.AllInOneBlock
            block_kwargs = {
                            "affine_clamping": params.get("clamping", 5.),
                            "subnet_constructor": constructor_fct,
                            "global_affine_init": 0.92,
                            "permute_soft" : permute_soft
                           }

        elif coupling_type == "rational_quadratic":
            CouplingBlock = RationalQuadraticSplineBlock
            block_kwargs = {
                            "num_bins": params.get("num_bins", 10),
                            "subnet_constructor": constructor_fct,
                            "bounds_init":  params.get("bounds_init", 10),
                            "permute_soft" : permute_soft
                            }
        else:
            raise ValueError(f"Unknown Coupling block type {coupling_type}")

        return CouplingBlock, block_kwargs

    def initialize_normalization(self, data, cond):
        """ Calculates the normalization transformation from the training data and stores it. """
        data = torch.clone(data)
        if self.use_norm:
            data /= cond
        data = torch.log(data + self.alpha)
        mean = torch.mean(data, dim=0)
        std = torch.std(data, dim=0)
        self.norm_m = torch.diag(1 / std)
        self.norm_b = - mean/std

        print('num samples out of bounds:', torch.count_nonzero(torch.max(torch.abs(data), dim=1)[0] > self.params.get("bounds_init", 10)).item())

    def define_model_architecture(self, in_dim):
        """ Create a ReversibleGraphNet model based on the settings, using
        SubnetConstructor as the subnet constructor """

        self.in_dim = in_dim
        if self.norm_m is None:
            self.norm_m = torch.eye(in_dim)
            self.norm_b = torch.zeros(in_dim)

        nodes = [ff.InputNode(in_dim, name="inp")]
        cond_node = ff.ConditionNode(self.cond_dim, name="cond")

        #nodes = [ff.InputNode(in_dim, name="inp"), cond_node]

        #nodes.append(ff.Node(
            #[nodes[-1].out0],
            #fm.FixedLinearTransform,
            #{ "M": self.norm_m, "b": self.norm_b },
            #name = "inp_norm"
        #))
        CouplingBlock, block_kwargs = self.get_coupling_block(self.params)
        for i in range(self.params.get("n_blocks", 10)):
            # optional ActNorm (if the scaling is not doine prior to trainqing)
            #nodes.append(ff.Node(
                #[nodes[-1].out0],
                #fm.ActNorm,
                #module_args={},
                #name=f"actnorm_{i}"
            #))
    # then the coupling block
            nodes.append(ff.Node(
                [nodes[-1].out0],
                CouplingBlock,
                block_kwargs,
                conditions=cond_node,
                name=f"block_{i}"
            ))

         
        nodes.append(ff.OutputNode([nodes[-1].out0], name='out'))
        nodes.append(cond_node)

        self.model = ff.GraphINN(nodes)
        self.params_trainable = list(filter(
                lambda p: p.requires_grad, self.model.parameters()))
        n_trainable = sum(p.numel() for p in self.params_trainable)
        print(f"number of parameters: {n_trainable}", flush=True)

    def set_bayesian_std_grad(self, requires_grad):
        for layer in self.bayesian_layers:
            layer.logsig2_w.requires_grad = requires_grad

    def sample_random_state(self):
        return [layer.sample_random_state() for layer in self.bayesian_layers]

    def import_random_state(self, state):
        [layer.import_random_state(s) for layer, s in zip(self.bayesian_layers, state)]

    def get_kl(self):
        return sum(layer.KL() for layer in self.bayesian_layers)

    def enable_map(self):
        for layer in self.bayesian_layers:
            layer.enable_map()

    def disenable_map(self):
        for layer in self.bayesian_layers:
            layer.disenable_map()

    def reset_random(self):
        """ samples a new random state for the Bayesian layers """
        for layer in self.bayesian_layers:
            layer.reset_random()

    def sample(self, num_pts, condition):
        """
            sample from the learned distribution

            Parameters:
            num_pts (int): Number of samples to generate for each given condition
            condition (tensor): Conditions

            Returns:
            tensor[len(condition), num_pts, dims]: Samples 
        """
        z = torch.normal(0, 1,
            size=(num_pts*condition.shape[0], self.in_dim),
            device=next(self.parameters()).device)
        c = condition.repeat(num_pts,1)
        x, _ = self.forward(z, c, rev=True)
        return x.reshape(num_pts, condition.shape[0], self.in_dim).permute(1,0,2)

    def log_prob(self, x, c):
        """
            evaluate conditional log-likelihoods for given samples and conditions

            Parameters:
            x (tensor): Samples
            c (tensor): Conditions

            Returns:
            tensor: Log-likelihoods
        """
        z, log_jac_det = self.forward(x, c, rev=False)
        log_prob = - 0.5*torch.sum(z**2, 1) + log_jac_det - z.shape[1]/2 * math.log(2*math.pi)
        return log_prob