from pypy.interpreter.mixedmodule import MixedModule


class Module(MixedModule):

    interpleveldefs = {
        'Tensor': 'interp_tensor.W_Tensor',
        '_tensor_flat': 'interp_tensor.tensor_flat',
        'zeros': 'interp_tensor.zeros',
        'kernel_count': 'interp_tensor.kernel_count',
        'launch_count': 'interp_tensor.launch_count',
    }

    appleveldefs = {
        'tensor': 'app_tensor.tensor',
    }
