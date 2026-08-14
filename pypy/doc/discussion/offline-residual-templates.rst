オフライン residual template
==============================

目的
----

特定の Python ``code`` を translation 時に要求せず、meta-tracing が行う
interpreter specialization を可能な限り offline PE に移す。offline PE は
opcode template を生成し、小さな runtime linker は code 固有値だけを埋める。

::

  translation time                 runtime                 meta-tracing
  ----------------                 -------                 ------------
  interpreter graph                任意の PyCode           residual JitCode
          |                              |                        |
      offline PE                   一度だけ decode          型 / guard
          |                              |                        |
  opcode templates  ------------> hole 埋込 + link -------> machine code
   (最適化済み CFG、               (const, pc, edge)
    stack effect、exception exit)

Binding time
------------

``static``
    interpreter 実装と opcode semantics。translation 時に既知。

``hole`` (late static)
    ``PyCode``、``pc``、``oparg``、定数、jump target。translation 時には未知だが
    code object の link 中は固定。

``dynamic``
    frame、stack 値、globals、heap state。meta-tracing に残す。

構成
----

``ResidualTemplateGenerator`` は opcode graph を小さな IR に変換する。
``ResidualTemplate`` は residual operation、型付き hole、``Continue`` や
``Finish`` などの明示的 terminator を持つ。後段が低レベルの tuple allocation
から制御フローを復元する必要はない。

次の lowering は relocatable JitCode stencil を生成する。runtime の work-list
linker は ``(code identity, pc)`` で block を cache し、hole と edge を解決する。
register layout と liveness は offline で正規化しておく。未対応命令や tracing
mode は generic portal へ fallback する。

実装順
------

#. template IR と opcode catalog（今回実装）。
#. symbolic hole と target expression を offline PE に追加。
#. relocatable JitCode stencil へ lower。
#. 小さな stack machine を interpreter tracing なしで link。
#. meta-tracer に接続し、PyPy opcode の一部を移植。
