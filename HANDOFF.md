# HANDOFF — LIFELINE (ERC StG 2027)

## 1. Where it stands
v2.x. `make final` counts (zero errors, zero overfull hboxes): Part I 5 body pages
(one `\section`/page), Part II 7, CV 4. Clean: framing is a semantics question
("what a program still means once its machine moves"), not an operational
checklist — three prior operational formulations (read-set liveness, immutable
commitment, completion record) died to counterexamples or reduced to known
analysis. Trace-relating compiler correctness [48] positioning; quantum withdrawn
to a named future extension. Team, EUR 1.2M budget, existing-award separation,
baselines and literature (OSR/deoptimisation, migration/checkpointing, KV
transfer, provenance) are in place; success/failure criteria are pre-declared
falsification thresholds. `make final` strips draft notes via `\iffinalversion` in
`erc_lifeline.sty`; `make` restores draft builds. This pass rewrote O1 (Part I
§3), Part II §1's theorem, Part II §6's M1 milestone, and the concept note's O1
paragraph to state the **M1 target** — an effective, transformation-aware
Myhill–Nerode theorem over $\equiv_\Pi$ — with the classical core marked standard,
not claimed [53: Nerode 1958]. Grep confirms no remaining claim of a proved lower
bound, precision bound, or established undecidability.

**r11 pass (generalist readability, this session).** A stateless generalist-panellist
read of Part I + CV (`REVIEW_GPT56SOL_r11_generalist.md`) scored **B**, worked as a
hook (page-1 question sentence), but found §3 read as a specification rather than
"success looks like this" and flagged the artifact paragraph and two defensive
passages as written for a hostile referee. Revised: added one-line plain-language
"what success looks like" glosses before O1 and the M1 target; condensed the
Preliminary-artifact paragraph in Part I §3 (all numbers kept, no claim added or
dropped — verified against the pre-existing wording sentence by sentence); cut the
"Two pre-empted objections" rebuttal paragraph in §5 and the "risks are stated, not
hidden" lead-in (risk content kept, framing plainer); simplified three CV domain-name
phrases (`validity-continuity domain`, `representation-continuity domain`) to
plain WP references. `make final`: Part I 5 pages, Part II 7, CV 4, zero
errors/overfull, unchanged.

**r12 specialist diff-check (this session).** The round-10 specialist
(`REVIEW_GPT56SOL_r12_raw.md`), reading only for claim/qualification/number/scope
drift relative to its own r10 verdict, initially caught one real regression: an
edit that had moved Part II §1's artifact sentence from "conservative for
history-sensitive policies (259 vs. 79 classes)" to "exact at 79 classes" — a
genuine new claim, not a rewording. **Reverted**; Part II §1 is now byte-identical
to its pre-r11 text. A second finding is **not** a regression: the specialist also
flags Part I's "reaches the exact coarsest congruence ... (79 classes ... previously
259, then 156)" / "derived, not hand-supplied" language as stronger than r10's
verdict — but this exact wording, verified word-for-word, was **already present in
Part I before this pass** (this session only condensed its surrounding prose,
preserving every clause and hedge). A targeted test — showing the specialist only
the pre-r11 paragraph text alongside r10 — reproduces the same "tension" finding on
the unedited original, confirming it predates r11 and is not something this pass
introduced. It reflects a standing Part I/Part II framing mismatch that r10 itself
had already flagged (round 10 §1, "Overclaimed"). **Not resolved here**: doing so
would mean picking which of Part I's or Part II's framing of the artifact result is
correct, which is a scientific-scope call outside this pass's readability mandate.
Flagged for the PI (see §5).

**r13 pass (this session): §3 restructure and Part I/Part II alignment.**
Resolved the r12 tension: it was Part II §1 lagging behind the verified
artifact (v4), not Part I overclaiming. Part II §1's artifact sentence is
now brought up to v4's own hedged state — exact 79/9 on the bounded
universe, matching brute-force enumeration; a sufficient, not proved
necessary, exactness condition checked by explicit linear algebra; one
out-of-fragment policy breaks the construction with a real witness; the
fragment argument a labelled sketch — so it no longer contradicts Part I
§3. Acting on r11's one requested change (generalist score **B**, §1
above): Part I §3 now opens each objective (O1/O2/O3) with a 2–3 line
plain-language, falsifiable "Success"/"Failure" statement, with the
formal target given beneath it. No claim, scope, or number was changed in
the process (verified by the r13 specialist re-review, §2b).

## 2. Nine adversarial rounds
| Round | Claim | Verdict | Decisive objection |
|---|---|---|---|
| 1 | 3-way loss decomposition exhausts transitions | Underspecified, "known theory, new terms" | Exhaustiveness unproved/unfalsifiable |
| 2 | Frontier lemma: witness bounded by live read-set | B | False for full-attention KV cache (Θ(n), not O(1)) |
| 3 | Commitment ⇒ frozen state write-immune | B | Commitment ≠ immutability; compaction/requantisation still legal |
| 4 | Retention-gap: $L_0\subseteq L_\Delta$ | B | Inclusion unproved; core is ordinary recovery-edge liveness |
| 5 | Deopt-sufficiency + E/V iff correct continuation | B | "If" direction false — representation/transition/completion obligations missing |
| 6 | Classification discovers meaning under change | B | Kill shot: meaning entirely delegated to $\pi$; collapses to realizability |
| 7 | Reference oracle is independent of policy | B | "Formally repaired, substantively displaced": oracle fixes only $m_0$'s result; $\pi$ still supplies all changed-machine meaning |
| 8 | Minimal-summary iff via distinguishing $C$ | B, unchanged | Kill shot: distinguishability ≠ realizability; decisive example is textbook provenance |
| Math note | 6 theorems formalising minimal-summary | No publishable theorem | 1–3 standard, 4 & 6 wrong as stated, 5 is a routine induction |
| 9 | O1/M1 now names the right theorem (transformation-aware, effective, coarsest compositional congruence, soundness and completeness) | B | Right target, no evidence it is achievable; no text change can move it; remaining gap is research |

## 2b. Artifact review rounds
| Version | Claim | Verdict | Decisive objection |
|---|---|---|---|
| v1 | `congruence.py` computes the coarsest compositional congruence from declared IR/policy/transformation rules | Rejected | Hard-codes the anticipated answer (`state_correct`); no algorithmic derivation from the rules |
| v2 | Rewrite replaces the per-pipeline selector with a uniform field-liveness-style backward pass | Partially fixed | `derive_retained_state()` is syntactically generic but not a semantic quotient derivation; sound, not complete, on the one nontrivial history-sensitive case |
| v3 | Declaration-driven interpreter + worklist, exercised recipe branch, syntactic-vs-semantic gap measured (259 vs. 79 classes, refinable to 156) | Partially accepted — effective transducers and computable compositional obligation mechanism; not the coarsest-congruence algorithm | No necessary/sufficient homomorphism conditions; no coarsest-congruence construction; no soundness/completeness proof — an obligation-propagation experiment, not a congruence-construction algorithm |
| v4 | Declaration-driven derivation computes the exact coarsest congruence: 79/79 for the history-sensitive pipeline, 9/9 for revocation/retention, both sound and complete against independent brute-force enumeration; sufficient (not necessary) exactness condition checked by explicit GF(3) linear algebra; one out-of-fragment policy breaks the construction with a real witness | Partially — "substantial advance": exact 79/9 on the universe genuinely delivered | Remaining: a correct theorem for a non-trivial class — the fragment theorem sketch is not correct as stated, the side-condition check is hard-coded rather than generic, and there is no mechanised (Lean 4) proof |

## 3. What would earn A
> "A proved nontrivial restricted case, a correct transformation-homomorphism
> theorem beyond provenance projection, or an implemented coarsest-congruence
> algorithm with a genuine composition example and a demonstrated boundary
> could be enough."

This is now O1/M1's backbone in Part I §3, Part II §1/§6 and the concept note.

## 4. Calibration
`gpt-5.6-sol` reviews at remote-specialist depth with unlimited time per round; a
Step-1 panel is generalists with ~15 minutes. "The theorem isn't proved in Part I"
(only a target is stated, standard core cited not reproven) exceeds what most
funded StG proposals satisfy at submission. But round 8's "formally yes,
mathematically no" — exact compliance, still unsound without added assumptions —
is Step-2-referee grade and must be taken seriously if this reaches Step 2.

## 5. Only the PI can do these (priority order)
Note: of the three items that would earn an A (§3), the third — implementing
the coarsest-congruence algorithm on a restricted IR with one genuine
composition example — plays to the PI's demonstrated strength (compiler
construction, not theorem-proving) and is the most realistic to produce in
the remaining time.
1. Pin a branch/commit, measure ONE MetaTensor guard-crossing device/layout
   transition (retained descriptor vs. canonical re-record; 10 runs, medians, CI)
   for Part I §4 — no first-party tensor measurement exists yet.
2. Post the four arXiv preprints; IDs to be inserted, all four to be posted,
   replacing every "et al."
3. HHU costed budget: HHU Strategischer Forschungsfonds (SFF), funding line 1
   supports early-career researchers' first external applications; amounts and
   eligibility for an incoming researcher to be confirmed with HHU
   Forschungsförderung (web pages are JS-only). EU/AC working-time %, HHU
   support letter — pending.
4. Decide the fate of the Japanese awards on taking up the ERC post.
5. Confirm doctoral supervisor and TMU start date.
6. Confirm collaborator consent (Leuschel, Kadomoto) in writing.

The next artifact step is **not** more field-level worklist engineering: v3's
own honest scope limit (§2b) is that it propagates obligations over field
names, not over semantic equivalence relations. What is needed is a semantic
(relation-level) construction — computing $\equiv_\Pi$-quotients directly and
proving a transformation induces a homomorphism on them — not another pass
over `reads`/`writes`/`erases` declarations.

## 6. Files
`MATH_NOTE.md` — PI's formalisation attempt (6 theorems, honest status tags).
`MATH_REVIEW_SOL.md` — referee's verdict on it.
`REVIEW_GPT56SOL.md` — index of rounds 1–3. `REVIEW_GPT56SOL_raw.md` (r1) through
`_r9_raw.md` — the nine adversarial rounds, summarised in §2; source of §3 above
is `_r9_raw.md` §7.
`REVIEW_2026-09-19.md` — earlier (Japanese) structural review of v0.9, pre-dating
the adversarial series. `FUNDING_ID_WORKING.md` — internal notes for the Funding
ID section drafted in `03_LIFELINE_Part_II.tex`; not a verified declaration.

## Update — 22 Sep 2026, after the pivot to MOTION / change-as-event (r15)

**Centre changed.** The class theorem is no longer the centre. LIFELINE is now a
meta-compiler that makes architectural change a *computational event*: the
interpreter defines a step for each declared change class and MOTION —
*Generating Executions and the Transitions Between Them*, its original v0.2
identity — derives the transition from that step as it derives everything else.
Deoptimisation is the degenerate case. Retention is conceded as standard once
change is internalised. O1 = MOTION (PI), O2 = meta-tracing correctness extended
to traces continuing on a changed machine (postdoc + Leuschel, small), O3 = cost.

**r15 verdict:** specialist B, generalist "a strong B, close to the boundary".
Both say the programme is now *coherent and implementation-led* and that the PI
is "unusually well placed to build this". **The old deficiency (an unproved
class theorem) is gone.** The new single deficiency is a *demonstration*, in the
specialist's words: "one end-to-end, non-toy MOTION artifact before submission in
which an independently written interpreter change rule generates a transition
across at least two representations", audited to show no hidden source/target
mapping in the change rule, and compared against a manual OSR/deoptimisation
adapter. Kill-shot risk to design against: the change-event step must not
smuggle the adapter in — the audit must show the rule is written without
knowledge of the target representation.

**This is the PI's kind of task.** MetaTensor already has guard recovery across
representations (virtual DAG ↔ materialised; layout change). The demonstration:
(1) write the device/layout change as an interpreter rule in MetaTensor, by
someone who has not seen the transition code (or with a documented firewall);
(2) let the meta-tracer specialise it; (3) audit the rule for target knowledge;
(4) compare the derived transition with the existing hand-written guard-recovery
path on next-token latency and retained bytes, 10 runs, medians, CI. Two to three
weeks. This replaces the theorem work entirely as the pre-submission priority.

Artifact status: v4 exact 79/9 on a bounded universe; v5/v5b checker rejects
vacuous certificates but still trusts certificate-supplied atoms/instance; cited
in Part I only within the referee's v5b wording. No further artifact iteration.

## Update — 22 Sep 2026: §5 item 1 measured, and the MOTION demonstration attempted

**§5 item 1 is done.** A first-party MetaTensor guard-crossing device/layout
measurement now exists, pinned and reproducible: results
`benchmark/results/luchkylilac-rtx3090/paper-2026-09-22`, code at
`benchmark/paper/run_transition.sh` + `benchmark/applevel/transition_probe.py`,
on its own binary (the knob postdates the model sweep, so `build/pypy-c` is
untouched and the rows carry their own hash).

Retained descriptor against canonical re-record, one binary, one emitter, one
cache, one allocator, one program, one bit (`METATENSOR_DRAIN`). The headline
for Part I §4 is *not* the wall-clock ratio — that turned out to be
compile-bound, 1.17–1.18x cold and within noise warm (0.99–1.08x), so quoting
it alone would be quoting a Triton invocation as if it were a policy cost.
The result that survives cache state is structural:

| interior consumer | descriptor kept | drained | ratio |
|---|---|---|---|
| never | 1.04 launches/iter | 4.00 | 3.85x |
| from the first step | 2.04 | 5.00 | 2.45x |
| after a guard fails | 3.04 | 4.50 | 1.48x |

Two findings. Reusing a launched region is what keeps the chain one kernel at
all — it is not merely a transition optimisation. And **crossing the guard
costs the descriptor-keeping arm exactly the difference between the two
regimes and costs the draining arm nothing**: predicted 615 launches for the
halfway run, observed 1215, an excess of exactly 3.00 per post-transition
iteration, which is the fused/unfused gap. The draining arm's halfway run is
1800 against a predicted 1800. After the guard fails, the arm that had been
fusing runs with the profile of the arm that never did, for the rest of the
program. The transition *iteration* itself costs the same in both arms (~70ms,
not removed by a warm cubin cache, so not compilation — leading hypothesis is
one driver-side module load per arm, untested).

So the claim for Part I §4 is not "keep the descriptor". It is that the
descriptor decision does not change what the boundary costs; it changes the
residual program, and **a kept descriptor does not survive a guard**.

**The MOTION demonstration (r15's single deficiency) is attempted and returns
a null, with the blocker named.** `benchmark/motion/layout_rule.py` states the
change class as an equation and nothing else; `benchmark/paper/audit_rules.py`
is the documented firewall, a program rather than a promise, and
`layout_rule_leaky.py` is a deliberately contaminated control it must reject
(it does, on all six counts; the runner refuses to measure otherwise). Ten
fresh processes per arm, sign-test intervals at 97.9% coverage: step latency
41.8us [41.0, 42.0] against 42.0us [41.0, 42.0], retained bytes and
launches/step identical with zero variance.

The two arms are the same program. `head_split` is not among the oopspecs the
tensor pass builds virtuals for, and a gather reaches the emitter only as a
kernel's input addressing, never as a node that can consume a value still
inside a region — so applying the rule to a live value and to a canonicalised
one are indistinguishable.

**What this converts the two-to-three weeks into.** The demonstration needs
one prerequisite before it can succeed: *a gather must be able to consume a
virtual*. The emitter already knows the gather (`GA_HEADSPLIT` in `ttir.py`);
what is missing is a fusible node for it in the pass. Until that exists there
is no derived transition to compare against the hand-written one, and any
comparison that appears to show one is measuring cache state — as an earlier
run of this very probe did (457ms against 0.15ms, which was a cold cache in
one arm and a warm one in the other; with a fresh cache per run both arms pay
452ms).

The audit, the rule, the probe, the 10-run protocol and `live_bytes` are all
in place and reusable, so when the prerequisite lands the measurement is a
re-run rather than a rebuild.
