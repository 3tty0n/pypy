from pypy.interpreter.mixedmodule import MixedModule


class Module(MixedModule):

    interpleveldefs = {
        'Tensor': 'interp_tensor.W_Tensor',
        '_tensor_flat': 'interp_tensor.tensor_flat',
        'zeros': 'interp_tensor.zeros',
        'cat': 'interp_tensor.cat',
        'scalar': 'interp_tensor.scalar',
        'kernel_count': 'interp_tensor.kernel_count',
        'kernel_compile_count': 'interp_tensor.kernel_compile_count',
        'launch_count': 'interp_tensor.launch_count',
        'lazy_enabled': 'interp_tensor.lazy_enabled',
        'lazy_stats': 'interp_tensor.lazy_stats',
        'mark_step': 'interp_tensor.mark_step',
        'mem_total': 'interp_tensor.mem_total',
        'live_bytes': 'interp_tensor.live_bytes',
        'alloc_failed': 'interp_tensor.alloc_failed',
        'cpu_fallbacks': 'interp_tensor.cpu_fallbacks',
        'group': 'interp_group.new_group',  # [motion]
    }

    appleveldefs = {
        'tensor': 'app_tensor.tensor',
    }
