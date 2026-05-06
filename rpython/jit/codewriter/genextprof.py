"""
Slow-path profiler for GenExtension code generation.

This module instruments the JIT extension code generator to track why operations
go through the slow path instead of fast-path code generation.

Enable with environment variable: PYPY_SLOWPATH_PROFILE=1

Reason codes:
  NO_UNSPEC_METHOD    - No emit_unspecialized_<name> method exists
  NO_SPEC_METHOD      - No emit_specialized_<name> method exists (constant args)
  UNSUPPORTED_SPEC    - Specialized method raised Unsupported exception
  UNSUPPORTED_UNSPEC  - Unspecialized method raised Unsupported exception
  SPEC_RETURNED_NONE  - Specialized method returned None
  UNSPEC_RETURNED_NONE- Unspecialized method returned None
  IS_CALL             - Operation is a call/inline_call
  IS_GUARD            - Operation is a guard
  IS_MEMORY_OP        - Operation is getfield/setfield/getarray/setarray
  HAS_LABEL_ARG       - Operation has label argument (control flow)
  IS_LIVE_OP          - Operation is -live- marker
  NEWFRAME            - Operation requires new frame
  BRIDGEOPT_FACT      - Optimizer fact carried across a trace/bridge boundary
"""

import os
from collections import defaultdict
import atexit

# Enable profiling via environment variable
SLOWPATH_PROFILE_ENABLED = os.environ.get('PYPY_SLOWPATH_PROFILE', '') == '1'
SLOWPATH_PROFILE_VERBOSE = os.environ.get('PYPY_SLOWPATH_PROFILE_VERBOSE', '') == '1'

# Reason codes
REASON_NO_UNSPEC_METHOD = 'NO_UNSPEC_METHOD'
REASON_NO_SPEC_METHOD = 'NO_SPEC_METHOD'
REASON_UNSUPPORTED_SPEC = 'UNSUPPORTED_SPEC'
REASON_UNSUPPORTED_UNSPEC = 'UNSUPPORTED_UNSPEC'
REASON_SPEC_RETURNED_NONE = 'SPEC_RETURNED_NONE'
REASON_UNSPEC_RETURNED_NONE = 'UNSPEC_RETURNED_NONE'
REASON_IS_CALL = 'IS_CALL'
REASON_IS_GUARD = 'IS_GUARD'
REASON_IS_MEMORY_OP = 'IS_MEMORY_OP'
REASON_HAS_LABEL_ARG = 'HAS_LABEL_ARG'
REASON_IS_LIVE_OP = 'IS_LIVE_OP'
REASON_NEWFRAME = 'NEWFRAME'
REASON_BRIDGEOPT_FACT = 'BRIDGEOPT_FACT'
REASON_FAST_PATH = 'FAST_PATH'  # Successfully used fast path


class SlowPathProfiler(object):
    """Tracks slow-path reasons during code generation."""

    _instance = None

    def __init__(self):
        # (opname, reason) -> count at codegen time
        self.codegen_counts = defaultdict(int)
        # (opname, reason) -> runtime execution count
        self.runtime_counts = defaultdict(int)
        # Track unique opcodes seen
        self.all_opcodes = set()
        # Track which opcodes have which methods
        self.opcode_method_info = {}  # opname -> {'has_spec': bool, 'has_unspec': bool}

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
            if SLOWPATH_PROFILE_ENABLED:
                atexit.register(cls._instance.print_report)
        return cls._instance

    def record_codegen(self, opname, reason, count=1):
        """Record a slow-path reason during code generation."""
        if not SLOWPATH_PROFILE_ENABLED:
            return
        self.codegen_counts[(opname, reason)] += count
        self.all_opcodes.add(opname)
        if SLOWPATH_PROFILE_VERBOSE:
            print("[SLOWPATH] codegen: %s -> %s" % (opname, reason))

    def record_method_info(self, opname, has_specialized, has_unspecialized):
        """Record which methods exist for an opcode."""
        if not SLOWPATH_PROFILE_ENABLED:
            return
        self.opcode_method_info[opname] = {
            'has_spec': has_specialized,
            'has_unspec': has_unspecialized
        }

    def increment_runtime(self, opname, reason):
        """Called at runtime to increment execution counter."""
        self.runtime_counts[(opname, reason)] += 1

    def get_runtime_increment_code(self, opname, reason):
        """Generate code that increments runtime counter."""
        if not SLOWPATH_PROFILE_ENABLED:
            return []
        return [
            "from rpython.jit.codewriter.slowpath_profiler import SlowPathProfiler",
            "SlowPathProfiler.get_instance().increment_runtime(%r, %r)" % (opname, reason)
        ]

    def print_report(self):
        """Print the slow-path analysis report."""
        if not self.codegen_counts and not self.runtime_counts:
            return

        print("\n" + "=" * 80)
        print("SLOW-PATH PROFILER REPORT")
        print("=" * 80)

        # Aggregate by reason
        reason_totals = defaultdict(int)
        for (opname, reason), count in self.codegen_counts.items():
            reason_totals[reason] += count

        total_ops = sum(self.codegen_counts.values())
        fast_path_count = reason_totals.get(REASON_FAST_PATH, 0)
        slow_path_count = total_ops - fast_path_count

        print("\n--- SUMMARY ---")
        print("Total operations analyzed: %d" % total_ops)
        print("Fast path operations:      %d (%.1f%%)" % (
            fast_path_count, 100.0 * fast_path_count / total_ops if total_ops else 0))
        print("Slow path operations:      %d (%.1f%%)" % (
            slow_path_count, 100.0 * slow_path_count / total_ops if total_ops else 0))

        print("\n--- SLOW-PATH BREAKDOWN BY REASON ---")
        print("%-25s %10s %8s" % ("Reason", "Count", "Share%"))
        print("-" * 45)
        for reason, count in sorted(reason_totals.items(), key=lambda x: -x[1]):
            if reason == REASON_FAST_PATH:
                continue
            pct = 100.0 * count / slow_path_count if slow_path_count else 0
            print("%-25s %10d %7.1f%%" % (reason, count, pct))

        print("\n--- TOP 30 SLOW-PATH (OPCODE, REASON) PAIRS ---")
        print("%-35s %-25s %10s %8s" % ("Opcode", "Reason", "Count", "Share%"))
        print("-" * 80)

        # Filter out fast path and sort by count
        slow_pairs = [(k, v) for k, v in self.codegen_counts.items()
                      if k[1] != REASON_FAST_PATH]
        slow_pairs.sort(key=lambda x: -x[1])

        for (opname, reason), count in slow_pairs[:30]:
            pct = 100.0 * count / slow_path_count if slow_path_count else 0
            print("%-35s %-25s %10d %7.1f%%" % (opname, reason, count, pct))

        # Report surprising invariants
        self._report_invariants()

        print("\n" + "=" * 80)

    def _report_invariants(self):
        """Report surprising patterns in the data."""
        print("\n--- SURPRISING INVARIANTS ---")

        # Find opcodes that have both fast and slow path hits
        opcode_reasons = defaultdict(set)
        for (opname, reason), count in self.codegen_counts.items():
            if count > 0:
                opcode_reasons[opname].add(reason)

        mixed_opcodes = []
        for opname, reasons in opcode_reasons.items():
            if REASON_FAST_PATH in reasons and len(reasons) > 1:
                mixed_opcodes.append((opname, reasons - {REASON_FAST_PATH}))

        if mixed_opcodes:
            print("\nOpcodes with BOTH fast and slow path (check for missed optimization):")
            for opname, slow_reasons in sorted(mixed_opcodes):
                fast_count = self.codegen_counts.get((opname, REASON_FAST_PATH), 0)
                slow_count = sum(self.codegen_counts.get((opname, r), 0) for r in slow_reasons)
                print("  %s: fast=%d, slow=%d (reasons: %s)" % (
                    opname, fast_count, slow_count, ', '.join(sorted(slow_reasons))))

        # Find opcodes with unspecialized method that still go slow due to UNSUPPORTED
        unsupported_with_method = []
        for opname, info in self.opcode_method_info.items():
            if info.get('has_unspec'):
                count = self.codegen_counts.get((opname, REASON_UNSUPPORTED_UNSPEC), 0)
                if count > 0:
                    unsupported_with_method.append((opname, count))

        if unsupported_with_method:
            print("\nOpcodes with unspecialized method that raised Unsupported:")
            for opname, count in sorted(unsupported_with_method, key=lambda x: -x[1])[:10]:
                print("  %s: %d times" % (opname, count))


def classify_opcode(opname):
    """Classify an opcode into categories for analysis."""
    if opname.startswith('-'):
        if opname == '-live-':
            return REASON_IS_LIVE_OP
        return None

    if 'call' in opname or 'inline_call' in opname:
        return REASON_IS_CALL

    if 'guard' in opname:
        return REASON_IS_GUARD

    if any(x in opname for x in ['getfield', 'setfield', 'getarrayitem', 'setarrayitem',
                                   'getinteriorfield', 'setinteriorfield', 'raw_load', 'raw_store']):
        return REASON_IS_MEMORY_OP

    return None


def get_profiler():
    """Get the singleton profiler instance."""
    return SlowPathProfiler.get_instance()
