"""Intermediate representation for offline-generated residual templates.

The objects in this module describe code-independent work completed by the
offline partial evaluator.  They deliberately contain no runtime linker or
JitCode details.
"""

from rpython.flowspace.model import Constant


class TemplateHole(object):
    """A typed value supplied while a concrete code object is linked."""

    def __init__(self, kind, name=None):
        self.kind = kind
        self.name = name or kind

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, self.name)


class PcHole(TemplateHole):
    def __init__(self):
        TemplateHole.__init__(self, "pc")


class OpargHole(TemplateHole):
    def __init__(self):
        TemplateHole.__init__(self, "oparg")


class CodeConstHole(TemplateHole):
    def __init__(self, index):
        TemplateHole.__init__(self, "code-constant", "const[%s]" % index)
        self.index = index


class Continue(object):
    """Continue execution at a late-static or concrete split value."""

    def __init__(self, target, dynamic_values):
        self.target = target
        self.dynamic_values = tuple(dynamic_values)


class Finish(object):
    """Return from the residual program."""

    def __init__(self, values):
        self.values = tuple(values)


class ResidualTemplate(object):
    def __init__(self, key, operations, holes, terminators):
        self.key = key
        self.operations = tuple(operations)
        self.holes = tuple(holes)
        self.terminators = tuple(terminators)


class ResidualTemplateCatalog(object):
    def __init__(self):
        self._templates = {}

    def add(self, template):
        if template.key in self._templates:
            raise ValueError("duplicate residual template %r" %
                             (template.key,))
        self._templates[template.key] = template

    def lookup(self, key):
        return self._templates[key]

    def keys(self):
        return self._templates.keys()


class ResidualTemplateGenerator(object):
    """Normalize concrete residual variants into the template IR.

    This is the first bridge from the existing PE.  A later generator will
    accept symbolic split values and emit typed holes directly.
    """

    def __init__(self, terminal_values=(-1,)):
        self.terminal_values = terminal_values

    def from_residual_graph(self, key, graph, transitions):
        operations = []
        for block in graph.iterblocks():
            operations.extend(block.operations)

        terminators = []
        for transition in transitions:
            next_value = transition.constant_next_value()
            dynamic_values = transition.dynamic_values()
            if next_value in self.terminal_values:
                terminators.append(Finish(dynamic_values))
            else:
                terminators.append(Continue(next_value, dynamic_values))

        return ResidualTemplate(key, operations, (), terminators)
