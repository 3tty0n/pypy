# MOTION (ERC LIFELINE) evidence, 2026-09-23

Separate from the MLSys evaluation (paper-2026-09-23).  One binary for the
leave check: libpypy-c.so `a99ad86b05e2` (scratch build7, source = pypytensor at the
commit that adds this directory, minus the llama decode work in progress).
The MOTION probe rows carry their own binary column (build6, `85c2bd7f4415`).
Rows are in data.tar.gz (benchmark/paper/archive.sh unpack).

## 1. The leave transition generated from `leave(p, reason)`

Run: `LEAVE_PYPY=... LEAVE_PERF=1 bash benchmark/paper/run_leave.sh`, then
again with `LEAVE_PERF_ONLY=1 LEAVE_SYNC=1`.

### Audit
```
== audit
== pypy/module/_metatensor/rule_leave.py
   imports:    pypy.module._metatensor.motion_iface
   vocabulary: REVOKE, action, drop_member, exclude, group, p, pending, reason, uncommitted
   constants:  none
   pass: the rule speaks only the abstract vocabulary

all rules pass the firewall
== benchmark/motion/rule_leave_leaky.py
   imports:    pypy.module._metatensor.interp_group, pypy.module._metatensor.motion_iface, rpython.jit.metainterp
   vocabulary: REVOKE, action, agg, append, cur, drop_member, group, next, p, parts, pending, reason, resume, snapshot, sub, t, total, version
   constants:  1
   FAIL rule_leave_leaky.py:6: imports from pypy.module._metatensor.interp_group, which is not in the declared vocabulary
   FAIL rule_leave_leaky.py:7: imports from rpython.jit.metainterp, which is not in the declared vocabulary
   FAIL rule_leave_leaky.py:12: mentions .version, a name from the target representation
   FAIL rule_leave_leaky.py:15: mentions .parts, a name from the target representation
   FAIL rule_leave_leaky.py:20: mentions .snapshot, a name from the target representation
   FAIL rule_leave_leaky.py:18: mentions .total, a name from the target representation
   FAIL rule_leave_leaky.py:20: mentions .pending, a name from the target representation
   FAIL rule_leave_leaky.py:20: mentions 'resume', a name from the target representation
   FAIL rule_leave_leaky.py:18: mentions .total, a name from the target representation

AUDIT FAILED: a rule knows where it is going
   contaminated control rejected
776f53862e 2026-09-23 05:13:01 +0900 rule_leave.py: the leave change rule, written behind the firewall
```

### Checker (independent, bitwise; motion = generated, ref = --jit off)
```
== check
ok       h1 motion        commits=200 log=leave:B:preempt  (leave/ref_h1)
ok       h1 motion        commits=200 log=leave:B:preempt  (leave/motion_h1)
same     leave/ref_h1 vs leave/motion_h1
ok       h1 adapter       commits=200 log=leave:B:preempt  (leave/adapter_h1)
same     leave/ref_h1 vs leave/adapter_h1
ok       h1 motion        commits=200 log=leave:B:preempt  (leave/erase_h1)
ok       h1 motion        commits=200 log=leave:B:preempt  (leave/late_h1)
ok       h2 motion        commits=200 log=leave:B:fault|revoke:B  (leave/ref_h2)
ok       h2 motion        commits=200 log=leave:B:fault|revoke:B  (leave/motion_h2)
same     leave/ref_h2 vs leave/motion_h2
ok       h2 adapter       commits=200 log=leave:B:fault|revoke:B  (leave/adapter_h2)
same     leave/ref_h2 vs leave/adapter_h2
MISMATCH h2 motion        log, w_at, w_end  (leave/erase_h2)
MISMATCH h2 motion        w_at, w_end  (leave/late_h2)
ok       h3 motion        commits=200 log=leave:B:fault  (leave/ref_h3)
ok       h3 motion        commits=200 log=leave:B:fault  (leave/motion_h3)
same     leave/ref_h3 vs leave/motion_h3
ok       h3 adapter       commits=200 log=leave:B:fault  (leave/adapter_h3)
same     leave/ref_h3 vs leave/adapter_h3
ok       h3 motion        commits=200 log=leave:B:fault  (leave/erase_h3)
ok       h3 motion        commits=200 log=leave:B:fault  (leave/late_h3)
ok       h2post motion    commits=200 log=leave:B:fault  (leave/ref_h2post)
ok       h2post motion    commits=200 log=leave:B:fault  (leave/motion_h2post)
same     leave/ref_h2post vs leave/motion_h2post
ok       h2post adapter   commits=200 log=leave:B:fault  (leave/adapter_h2post)
same     leave/ref_h2post vs leave/adapter_h2post
ok       h2post motion    commits=200 log=leave:B:fault  (leave/erase_h2post)
ok       h2post motion    commits=200 log=leave:B:fault  (leave/late_h2post)
ok       none motion      commits=200 log=-  (leave/ref_none)
ok       none motion      commits=200 log=-  (leave/motion_none)
same     leave/ref_none vs leave/motion_none
ok       none adapter     commits=200 log=-  (leave/adapter_none)
same     leave/ref_none vs leave/adapter_none
ok       none motion      commits=200 log=-  (leave/erase_none)
ok       none motion      commits=200 log=-  (leave/late_none)
   mutant erase: caught on h2
   mutant late: caught on h2
```
(The leave/ files were written under paper-2026-09-23/ and moved here.)

### Hand-written lines
```
MOTION (rule + everything around it)
     7  change rule                            pypy/module/_metatensor/rule_leave.py
    11  rule vocabulary                        pypy/module/_metatensor/motion_iface.py
   224  domain helpers + annotations           pypy/module/_metatensor/interp_group.py
     1  module registration                    pypy/module/_metatensor/moduledef.py
     1  policy declaration                     benchmark/applevel/leave_probe.py
     2  policy handed to the group             benchmark/applevel/leave_probe.py
   246  total

hand-written adapter
    46  contributor log + dedup + transition   benchmark/motion/adapter_leave.py
     1  policy declaration                     benchmark/applevel/leave_probe.py
    47  total

excluded from both (shared harness): W_Group.arm, W_Group.descr_arm, W_Group.poll, W_Group.__init__ lines marked [harness], Channel
N_rule=246 N_adapter=47
```

### Timing (n=65536, 200 steps, leave at step 100, 10 processes; median [2nd, 9th])
| sync | config | step us | step of the leave us | steady launches |
|---|---|---|---|---|
| 0 | adapter | 10.0 [10.0, 11.0] | 13.1 [12.2, 14.1] | 5.00 |
| 0 | adapter-h2 | 7.5 [7.2, 7.6] | 117.5 [115.9, 125.2] | 5.00 |
| 0 | retained | 1.9 [1.9, 1.9] | 22.9 [21.9, 24.1] | 1.16 |
| 0 | retained-h2 | 2.1 [2.1, 2.1] | 32.9 [32.9, 33.1] | 1.16 |
| 0 | unretained | 1.9 [1.9, 1.9] | 26.9 [26.9, 26.9] | 1.16 |
| 1 | adapter | 37.0 [37.0, 37.9] | 43.9 [42.2, 44.1] | 6.00 |
| 1 | adapter-h2 | 32.7 [31.9, 32.9] | 132.6 [131.1, 135.2] | 6.00 |
| 1 | retained | 26.9 [26.9, 27.2] | 53.9 [52.0, 54.1] | 1.36 |
| 1 | retained-h2 | 19.1 [19.1, 19.9] | 55.0 [53.9, 57.0] | 1.36 |
| 1 | unretained | 26.9 [26.9, 27.1] | 53.9 [52.9, 54.8] | 1.36 |

sync=0 times a step to the end of its (asynchronous) launches, sync=1 to the
device finishing it (one host read per step, on both sides).  retained /
unretained: policy with a revoke / retain for every reason, no leave;
-h2: the fault history, the event column is the step the fault lands in.

## 2. The MOTION probes (axis and blocked-layout change classes)

`bash benchmark/paper/run_motion.sh` with MOTION_CASE=axis|layout,
MOTION_ARMS="derived handwritten direct drain", MOTION_SCOPED=0|1, cold
cache, 10 runs; the retention curve with MOTION_CACHE=warm, 5 runs.
Figures: figures/motion.pdf, figures/motion_retention.pdf.

```
== axis, locals live, cold cache, 200 steps: 10 runs per arm, all pass=True
                        derived                       handwritten                   direct                        drain                         
step latency (us)       12.9 [12.2, 13.1]             21.0 [21.0, 22.9]             31.0 [31.0, 32.9]             40.1 [40.1, 40.1]             
transition step (us)    40.1 [38.9, 42.9]             2034.1 [2013.0, 2126.0]       1094.0 [1068.1, 1159.0]       54.1 [51.0, 58.9]             
retained bytes          9145432 [9145432, 9145432]    12389520 [12389520, 12389520] 16813472 [16813472, 16813472] 8850656 [8850656, 8850656]    
launches / step after   1.16 [1.16, 1.16]             2.12 [2.12, 2.12]             2.16 [2.16, 2.16]             4.08 [4.08, 4.08]             
kernels after           3 [3, 3]                      2 [2, 2]                      2 [2, 2]                      1 [1, 1]                      
derived / handwritten step latency: 0.614x
derived / direct step latency: 0.416x
derived / drain step latency: 0.322x

== axis, scoped, cold cache, 200 steps: 10 runs per arm, all pass=True
                        derived                       handwritten                   direct                        drain                         
step latency (us)       17.9 [17.9, 17.9]             20.0 [20.0, 21.0]             30.5 [30.0, 31.9]             32.2 [31.9, 33.1]             
transition step (us)    48.0 [46.0, 49.1]             45.5 [42.2, 53.2]             55.5 [52.0, 63.9]             55.5 [51.0, 61.0]             
retained bytes          1182744 [1182744, 1182744]    10914840 [10914840, 10914840] 9440712 [9440712, 9440712]    4721880 [4721880, 4721880]    
launches / step after   1.16 [1.16, 1.16]             2.12 [2.12, 2.12]             2.16 [2.16, 2.16]             3.12 [3.12, 3.12]             
kernels after           1 [1, 1]                      1 [1, 1]                      1 [1, 1]                      1 [1, 1]                      
derived / handwritten step latency: 0.895x
derived / direct step latency: 0.587x
derived / drain step latency: 0.556x

== layout, locals live, cold cache, 200 steps: 10 runs per arm, all pass=True
                        derived                       handwritten                   direct                        drain                         
step latency (us)       29.1 [28.8, 29.1]             21.0 [21.0, 21.0]             29.8 [29.1, 30.0]             31.9 [31.9, 31.9]             
transition step (us)    455263.0 [453803.8, 513426.1] 457515.2 [455548.0, 460069.2] 455300.5 [452738.0, 457022.0] 455711.6 [453861.0, 456519.8] 
retained bytes          8454248 [8454248, 8454248]    5308480 [5308480, 5308480]    8454336 [8454336, 8454336]    6488296 [6488296, 6488296]    
launches / step after   1.20 [1.20, 1.20]             2.16 [2.16, 2.16]             2.20 [2.20, 2.20]             4.12 [4.12, 4.12]             
kernels after           4 [4, 4]                      3 [3, 3]                      3 [3, 3]                      2 [2, 2]                      
derived / handwritten step latency: 1.386x
derived / direct step latency: 0.977x
derived / drain step latency: 0.912x

== layout, scoped, cold cache, 200 steps: 10 runs per arm, all pass=True
                        derived                       handwritten                   direct                        drain                         
step latency (us)       17.9 [17.9, 17.9]             20.0 [20.0, 20.0]             28.8 [26.0, 29.1]             31.9 [31.9, 31.9]             
transition step (us)    455851.5 [455540.9, 456615.0] 455007.5 [454217.2, 458050.0] 454635.6 [453145.0, 457530.0] 454102.5 [453234.0, 456998.8] 
retained bytes          1179672 [1179672, 1179672]    2359320 [2359320, 2359320]    5505352 [5505352, 5505352]    2949272 [2949272, 2949272]    
launches / step after   1.20 [1.20, 1.20]             2.16 [2.16, 2.16]             2.20 [2.20, 2.20]             3.16 [3.16, 3.16]             
kernels after           2 [2, 2]                      2 [2, 2]                      2 [2, 2]                      2 [2, 2]                      
derived / handwritten step latency: 0.895x
derived / direct step latency: 0.622x
derived / drain step latency: 0.561x

== axis, retained bytes by loop length (warm cache, medians)
steps   derived       handwritten   direct        drain         derived/handwritten
20      4426792       2362384       5311584       5311584       1.87
40      7081008       5016624       10030256      5901432       1.41
60      5606424       2657296       6786208       7376056       2.11
100     8555600       5606456       3542080       7966168       1.53
200     9145432       12389520      16813472      8850656       0.74
400     15043752      27135320      29789960      9735424       0.55
1000    7375936       71372720      29789960      14749040      0.10

== layout, retained bytes by loop length (warm cache, medians)
steps   derived       handwritten   direct        drain         derived/handwritten
20      3538992       3538976       3342440       4522088       1.00
40      1966096       8060960       5111920       4522112       0.24
60      2752536       4718648       5505144       4915344       0.58
100     5111888       13959200      5505192       5111960       0.37
200     8454248       5308480       8454336       6488296       1.59
400     7274592       43450400      10813736      8847680       0.17
1000    3342360       102432800     9240808       13763080      0.03

intervals: distribution-free sign-test, 97.9% coverage at n=10
```
