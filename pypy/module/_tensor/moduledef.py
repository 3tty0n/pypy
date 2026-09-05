from pypy.interpreter.mixedmodule import MixedModule


class Module(MixedModule):

    interpleveldefs = {
        'Tensor': 'interp_tensor.W_Tensor',
        '_tensor_flat': 'interp_tensor.tensor_flat',
        'zeros': 'interp_tensor.zeros',
    }

    appleveldefs = {
        'tensor': 'app_tensor.tensor',
    }
