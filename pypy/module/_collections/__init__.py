from pypy.interpreter.mixedmodule import MixedModule

class Module(MixedModule):
    """High performance data structures.
- deque:        ordered collection accessible from endpoints only
- defaultdict:  dict subclass with a default value factory
"""

    appleveldefs = {
        'defaultdict': 'app_defaultdict.defaultdict',
        }

    interpleveldefs = {
        'deque' : 'interp_deque.W_Deque',
        'deque_iterator' : 'interp_deque.W_DequeIter',
        'deque_reverse_iterator' : 'interp_deque.W_DequeRevIter',
        '__missing__': 'interp_defaultdict.missing',
        }

    def setup_after_space_initialization(self):
        """NOT_RPYTHON"""
        # must remove the interp-level name '__missing__' after it has
        # been used...  otherwise, some code is not happy about seeing
        # this code object twice
        space = self.space
        space.getattr(self, space.newtext('defaultdict'))  # force importing
        space.delattr(self, space.newtext('__missing__'))

    def startup(self, space):
        # OrderedDict is normally present, but in some cases the line
        # "from __pypy__ import reversed_dict, move_to_end" from
        # _pypy_collections.py raises
        space.appexec([self], """(mod):
            try:
                from _pypy_collections import OrderedDict
                mod.OrderedDict = OrderedDict
            except ImportError:
                pass
        """)
