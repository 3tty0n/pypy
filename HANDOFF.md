# HANDOFF — MetaTensor 側でやること

対象リポジトリ: `~/src/github.com/pypy/tensorpypy`(main, 3581d3a 時点)
目的: ERC 提案 LIFELINE の Step 1 判定を B から A に上げる証拠を作る。
締切: 結果を **2026-10-08 まで**に提案書側へ返す(提出は 10-14)。
詳しい背景と設計の理由: `MOTION_E2E_METATENSOR.md`

## 作るもの

2ワーカーの勾配蓄積で、インタプリタに書いた変化規則 `leave(p, reason)` から、MetaTensor が次の3つを**生成**することを示す。

1. 所属(membership)に対する structural guard
2. 融合で消える寄与者ごとの値(`g_A`, `g_B`)を、未コミットの間だけ resume data に残す判断
3. guard 失敗時に「`g_B` を除いてからコミット」を実行する順序付きの遷移

## 手順

- [x] **プログラム**:ワーカー A, B が `g_A`, `g_B` を計算し、`g = g_A + g_B` を蓄積してから `commit`。2デバイスは CPU 上の2つの文脈でよい(GPU は不要)。
- [x] **policy**:アプリが宣言する。`leave(p, preempt)` なら retain、`leave(p, fault)` なら revoke。
- [x] **変化規則**:`leave(p, reason)` をインタプリタ操作として `pypy/module/_metatensor/interp_tensor.py` 側に書く。
  - 書いてよいもの:無効化する前提と、宣言された policy を未コミットの寄与に適用すること。
  - **書いてはいけないもの**:融合後の表現についての知識、移送元から移送先への状態の対応表、修復手続き。
  - 規則は `optimizeopt/metatensor.py` と resume data のコードを見ずに書く。規則のファイルが何を import しているかと、git の履歴で監査できるようにする。
- [x] **生成**:`rpython/jit/metainterp/optimizeopt/metatensor.py` で、規則が p の寄与を読むことから保持を決め、guard の resume data に残す。retain だけの場合と寄与前の離脱では何も残さないこと。
- [x] **3つの履歴**(`benchmark/applevel/` に probe を1本追加。`recovery_probe.py` と `transition_probe.py` の形式に合わせる):

  | 履歴 | 事象 | 正しい結果 |
  |---|---|---|
  | H1 | `g_B` が和に入った後、B が preempt で離脱 | `g_A + g_B` でコミット |
  | H2 | `g_B` が和に入った後、B が fault で離脱 | `g_A` のみでコミット |
  | H3 | B が寄与する前に離脱 | `g_A` のみ(除くものなし) |

- [x] **参照実行と checker**:同じ履歴を JIT なし(インタプリタのみ)で再生した結果と、出力・コミット回数・revoke の有無を比べる。checker は生成側と別のコードにする(`recovery_probe.py` の pure-Python 参照と同じ方式)。
  - 浮動小数点の和なので、比較の方法を先に決める:「`g_B` を引く」(逆元を使う)か「生き残った寄与から和を作り直す」かを選び、加算順序と許容誤差を記録する(r21 の指摘)。
  - preempt で retain するのは、離脱前に受け取り済みの寄与だけ。fault で revoke するのは未コミットのステップだけ。
- [x] **mutant**:(a) 保持を無効にする(寄与者の区別を消す)と H2 で checker が失敗すること。(b) 順序を逆にする(コミット後に除く)と失敗すること。
- [x] **手書き adapter**:同じ3履歴を「寄与者ログ + 重複除去 + 手書きの遷移」で満たす実装を別に書く。
- [x] **行数**:MOTION 側(規則 + policy 宣言 + annotation + ヘルパー、すべて)と手書き adapter 側を数える。スクリプトで数え、数え方を固定する。
- [ ] **(Step 2 向け、余裕があれば)合成を跨ぐ例**:既存の2つの変換(例:融合とレイアウト変更)の合成を跨いで、同じ遷移が生成・検査できることを示す(r21 の Path to A §4)。
- [~] **(余裕があれば)性能**:保持ありとなしの定常オーバーヘッド、guard 失敗から次のコミットまでの時間と checkpoint 再開との比較。10 run、中央値、95% CI。

## 既存の数値の固定(同時にやる)

- [x] 提案書に載せている数値のブランチ、コミット、計測手順を記録する:1.63×(同一 launch 数で deferred library 比)、1.42×(torch.compile の early exit 比)、3.86×→1.48×(guard を跨ぐ融合の利得)、0.82×(XLA 比の幾何平均)、3× worse tail(torch.compile 比)。

## 提案書側に返すもの

- [ ] コミット ID(実証と既存の数値の両方)
- [ ] H1〜H3 で参照実行と一致したこと、mutant (a)(b) が検出されたことのログ
- [ ] 行数:`N_rule`(MOTION 側)と `N_adapter`(手書き adapter)
- [ ] 規則のファイルが融合・resume data のコードを参照していないことの監査結果
- [ ] (あれば)性能の数値

これらは Part I §3 の「Preliminary artifact」段落の末尾、§4 Device、Part II WP3 に入れる。**数値は実測値だけを使う。**


---

## 結果(2026-09-23、ブランチ `pypytensor`)

### 実証:`leave(p, reason)` から生成された遷移

| 項目 | 結果 | 場所 |
|---|---|---|
| 変化規則 | 7 行。`motion_iface` だけを import する | `pypy/module/_metatensor/rule_leave.py`(`776f53862e`、単独コミット・未修正) |
| ファイアウォール | 規則は別セッションが書いた。渡したのは意味・シグネチャ・読んでよい1ファイル(`motion_iface.py`)だけ。そのセッション自身の記録では、ツール呼び出しは `motion_iface.py` の Read 1回と `rule_leave.py` の Write 1回 | `benchmark/motion/rule_leave.provenance` |
| 監査 | 規則は通過(import は `motion_iface` のみ、識別子 9、定数なし)。汚染した対照 `rule_leave_leaky.py` は9箇所で不合格 | `benchmark/paper/audit_rules.py`、`leave/audit.log` |
| structural guard | 所属は quasi-immutable。group を promote したうえで、各 membership check の後に `guard_not_invalidated` が入る(1 step あたり4個) | `test_group_jit.py` |
| 保持の判断 | 規則を宣言済みの各 reason について記録用の代役で実行し、p の寄与を読むかどうか(read set)から導出する。コミットの guard の resume data は、revoke がある policy では寄与者別の part を **2個**、retain のみでは **0個** 持つ。融合された和(virtual tensor 7個)はどちらにもあるが、寄与者の区別はない。コミット後の guard には `w` しか残らない | `test_group_jit.py::test_resume_data_holds_parts_only_while_the_rule_can_read_them` |
| 定常状態 | 1 step が1 launch(`w - lr*(g_A+g_B)`)で、allocation 0。保持の有無で同じ | JIT test、probe の steady_launches 1.16 |
| 3履歴 | H1 / H2 / H3 と、契約外の H2post・none のすべてで、生成側は独立 checker(bitwise)とも、インタプリタのみの参照実行(フィールドごと)とも一致した。手書き adapter も同じ | `results/.../motion-2026-09-23/summary.md`(`leave/checker.log` は data.tar.gz 内) |
| mutant | (a) erasure(`MOTION_ERASE=1`、part を保持しない)は H2 で値とログが不一致。(b) 順序(`MOTION_ORDER=late`、sync の前に値を取る)は H2 で値が不一致。どちらも検出した | 同上 |
| 浮動小数点 | 「生き残った寄与から和を作り直す」方式。加算は寄与順、入力は二進有限小数(lr=1/8)で、どの和も exact。許容誤差なしの bitwise 比較 | `check_leave.py` の docstring |

**行数**(`benchmark/motion/count_lines.py`、コードのある行だけ。通知チャネルの模擬は両側から除外):

| | 行 |
|---|---|
| MOTION 側 `N_rule` | **246** = 規則 7 + 語彙 11 + ドメインのヘルパーと annotation 224(うち約50行は app-level への公開の boilerplate、約20行は保持の導出、約8行は mutant のスイッチ)+ module 登録 1 + policy 宣言 3 |
| 手書き adapter `N_adapter` | **47** |

→ 総数では MOTION 側が5倍以上多い。変化クラス1つあたりの手書きは、MOTION 側が規則 7 + policy 3 = 10 行、adapter 側が 47 行になる。ただし 224 行のヘルパーが変化クラス間で再利用できる、という主張は2つ目の変化クラスを実装しないと証拠にならない(未実施)。本文にはこの内訳ごと、実測のまま入れる。

**性能**(n=65536、200 steps、leave は step 100、10 run、中央値 [2番目, 9番目])

| 構成 | step µs(dispatch) | step µs(device 完了まで) | steady launches | leave が起きた step の µs(device 完了まで) |
|---|---|---|---|---|
| 生成・保持あり | 1.9 [1.9, 1.9] | 26.9 [26.9, 27.2] | 1.16 | 55.0 [53.9, 57.0](h2) |
| 生成・保持なし(retain のみ) | 1.9 [1.9, 1.9] | 26.9 [26.9, 27.1] | 1.16 | — |
| 手書き adapter | 10.0 [10.0, 11.0] | 37.0 [37.0, 37.9] | 5.00 | 132.6 [131.1, 135.2](h2) |

- 保持の定常コストは、どちらの計り方でも差が出ない(26.9 と 26.9)。
- dispatch は、非同期の launch が出終わるまでの時間。device 完了までは、毎 step の終わりに host 読み(`w.sum().item()`)を1回入れた時間で、両側とも同じ条件。後者で steady launches が 1.16 から 1.40 に増えるのは、その読みの分。
- fault が起きた step は、生成側 55.0 µs に対し adapter 132.6 µs。
- checkpoint からの再開との比較はまだしていない。

結果の場所:`benchmark/results/luchkylilac-rtx3090/motion-2026-09-23/`(summary.md、figures/、data.tar.gz。展開は `benchmark/paper/archive.sh unpack`)。MLSys 用の結果(`paper-2026-09-23/`)とは別のディレクトリ。

### MOTION probe(blocked layout / vector axis、4 arm、10 run)

gather を融合ノードにしたので、blocked layout(`[rows, cols]` と `[heads*rows, cols/heads]` という2つの表現)でも規則を導出できるようになった。変化後の launch/step は derived 1.20、canonical 化してから 2.16、descriptor を保つ手書き adapter 2.20、drain 3.16。step の終わりで tensor を `del` した場合、step 時間は derived 17.9 µs、canonical 化 20.0、adapter 28.8、drain 31.9。dead local をループの back-edge まで生かしたままだと、derived は 29.1 µs で canonical 化(21.0)より遅くなる。このインタプリタは dead local を back-edge まで生かしておくので、step ごとに materialise が起きるためで、両方の形を記録した。図は `figures/motion.pdf`。

### 既存の数値の固定

`benchmark/paper/erc_numbers.py` が、コミット済みの結果ファイルから再計算し、stage・ファイル・行・式・binary・ソースコミットを出力する。

| 数値 | 再計算 | 出典 |
|---|---|---|
| 1.63×(同 launch 数で deferred library 比) | 1.627× | `paper-2026-09-21/lazy.tsv`、bert-base、launch 64.0 vs 64.3。binary `ddc8008cdeaf` ← source `1bdc7e63fa` |
| 1.42×(early exit、torch.compile 比) | **1.427×**(四捨五入すると 1.43×) | `paper-2026-09-19/control.tsv`、varying、p50。同じ binary |
| 3.86× → 1.48×(guard を跨ぐ融合の利得) | 3.855× → 1.481× | `paper-2026-09-22/transition_phases.tsv`(1600/415、1800/1215)。source `3581d3a8a9`、記録は `c8d02109bf` |
| 0.82×(XLA 比の幾何平均) | 0.818× | `paper-2026-09-21/models.tsv`、14 モデル |
| 3× worse tail(torch.compile 比) | 3.063× | `paper-2026-09-21/control.tsv`、stable、p95 |

**要対応:1.42× は切り捨てで、四捨五入なら 1.43×。**提案書側で直すこと。
