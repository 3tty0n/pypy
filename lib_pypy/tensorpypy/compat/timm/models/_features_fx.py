"""timm.models._features_fx: upstream re-exports timm.layers' fx leaf
registration (a no-op here, see timm/layers/_fx.py); the fx feature
extractor classes are not ported."""
from timm.layers import register_notrace_module, register_notrace_function
