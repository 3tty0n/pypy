# MOTION end-to-end 実証を MetaTensor で行う計画(PI 用メモ)

作成: 2026-09-22 / 根拠: gpt-6-sol レビュー r20 §7「Path to A」
(`REVIEW_GPT6SOL_r20.md`)と r15 の specialist 指摘(`HANDOFF.md`)

## 0. なぜこれが必要か

r20 の結論(Step 1 パネル):

> 文章修正 1〜4 だけでは strong B。A にするには新しい証拠が1つだけ要る:
> 実際に動き、独立に検査された **generated transition**。今の artifact は
> 「policy が観測すべきもの」を導出・検査するだけで、MOTION の定義的主張
> (遷移そのものの生成)はまだ示していない。

r20 が示した**最低限の要件**(これを満たせば「With items 1–4, this could move my Step 1 grade to A」):

- 生成されたコードが **3つの履歴**(preemption / fault / 寄与前の離脱)で実際に動く
- 出力と効果を**独立に**検査する
- provenance を消す **erasure mutant が失敗する**
- 入力とコードを保存する(コミット ID)
- 手書き作業を**すべて数える**(変化規則 + annotation 対 手書き adapter)
- **定理も GPU での運用も不要**

r15 の specialist が加えた条件(kill-shot 対策):

- 変化規則は**移送先の表現を知らずに**書かれていること(adapter をこっそり入れていないことを監査で示す)
- 遷移は**少なくとも2つの表現**(融合された表現 ↔ 寄与者別の表現)を跨ぐこと

## 1. なぜ MetaTensor か

1. **1頁目の例と device 軌道が一致する。** r18〜r20 で毎回「1頁目の学習の例は provenance、device 実験は推論」と食い違いを指摘された。2デバイスからの勾配蓄積を MetaTensor で実証すれば、同じ例が device 軌道の実証になる。
2. **情報を消す合成が実物として出る。** MetaTensor の融合は `g_A + g_B` を1 kernel にまとめて寄与者の区別を消す。Step 2 で r20 が求める「必要な情報を消す合成を跨いで遷移が保たれること」が、人工的な変換なしで現れる。
3. **meta-compiler による導出として説明できる。** RPython の guard には resume data(失敗時に状態を再構成する情報)がある。MOTION の保持の判断は次のように言い換えられる。
   > structural guard の resume data に、変化イベントの step が読む値(`g_B`)を残す。融合最適化はそれを畳み込んで消す。
   既存の virtual object と resume data の拡張であり、toy ではない。
4. **比較対象が揃っている。** interpreter-only の参照実行(recovery probe)が既にある。手書き adapter の比較対象は「寄与者ログ + 重複除去」で、作るのは簡単。

## 2. 最小構成(2〜3週間、CPU で可)

### 2.1 プログラム(アプリ側)
- 2ワーカー A, B が各自のシャードで勾配テンソル `g_A`, `g_B` を計算し、1ステップ分 `g = g_A + g_B` を蓄積してからコミット(`commit(step)`)。
- アプリが宣言する policy:
  - `leave(p, preempt)` → `retain`(完了済みの `g_p` は有効)
  - `leave(p, fault)` → `revoke`(未コミットのステップから `g_p` を除く)
- 2つの「デバイス」は CPU 上の2つのデバイス文脈で模擬してよい(r20:GPU での運用は不要)。

### 2.2 変化規則(インタプリタ側、変化クラスごとに1回)
- `leave(p, reason)` をインタプリタの操作(bytecode / builtin)として実装する:
  - p に依存する前提を無効化する
  - reason に対して宣言された policy を、p の**未コミット**の寄与に適用する
- **書いてはいけないもの**(Part I §3 の定義どおり):移送元から移送先への状態の対応表、生成コード向けの修復手続き、融合後の表現についての知識。
- **ファイアウォール**:規則は融合・resume data のコードを見ずに書く(または別の人が書く)。git の履歴と、規則ファイルが import するものの一覧で監査可能にする。

### 2.3 MOTION が生成するもの(meta-tracer / optimizer 側)
1. **structural guard**:トレース中の所属(membership)に対する guard。`leave` はこの guard の失敗として現れる。
2. **保持の判断**:`leave(p, fault)` が p の寄与を読むことを、規則に対する liveness 解析から導き、融合で消える寄与者別の値を**未コミットの間だけ** resume data に残す(virtual として保持し、必要なら materialise できる形)。`retain` しかない場合や寄与前の離脱では何も残さない(余分に保持しないことも結果として示す)。
3. **順序付きの遷移**:guard が失敗したら、「`g_B` を引く → コミット」の順に実行する。コミット後に revoke が起きないという順序を、生成された resume path が守ること。

### 2.4 3つの履歴
| 履歴 | 事象 | 正しい結果 |
|---|---|---|
| H1 preemption | `g_B` が和に入った後、B が preempt で離脱 | `g_A + g_B` を保持してコミット |
| H2 fault | `g_B` が和に入った後、B が fault で離脱 | `g_A` のみでコミット(`g_B` を revoke) |
| H3 寄与前の離脱 | B が寄与する前に離脱 | `g_A` のみ。revoke する対象なし |

(可能なら H2' として「コミット後の fault」を加え、契約の範囲外なので checkpoint fallback に落ちることを示す。)

### 2.5 検査
- **参照実行**:同じ履歴を特殊化せずにインタプリタで再生し(変化イベントの前までを同じにしてから `leave` の step を取る)、結果・効果・provenance を記録する(Part I §3 の ⟦P⟧_ref の定義どおり)。
- **独立 checker**:生成コードの出力と効果(コミットされた値、コミット回数、revoke の有無)を参照実行と比較する。checker は生成器と別のモジュールにする。既存の `artifact/checker.py` の方針を流用してよい。
- **erasure mutant**:保持の判断を無効化する(融合で寄与者別の値を消したままにする)。H2 で checker が不一致を報告することを確認する。
- **追加の mutant**(推奨):順序を逆にする(コミット後に revoke)。checker が検出すること。

### 2.6 手書き作業の計数(annotation audit)
- MOTION 側:変化規則の行数 + アプリの policy 宣言 + annotation + ドメインのヘルパーを**すべて**数える。
- 手書き adapter 側:同じ3履歴を満たす「寄与者ログ + 重複除去 + 遷移手続き」を手で書いた実装の行数。
- 同じカバレッジ(同じ3履歴、同じ checker)で比べる。

### 2.7 性能(Part I では必須ではないが Part II に効く)
- 定常時のオーバーヘッド:保持ありと保持なし(融合のみ)の比較。10 run、中央値、95% CI。
- 遷移のコスト:guard 失敗から次のコミットまでの時間。checkpoint からの再開と比べる。

### 2.8 保存
- MetaTensor のブランチとコミット ID、入力、出力、checker のログ、行数の計測スクリプトを保存する(Part I の既存の draftnote「pin branch/commit and protocol」と一緒に対応)。

## 3. 本文への入れ方(結果が出たら)

**Part I §3**:「Preliminary artifact」段落の末尾にある文
> It derives observation requirements, not yet a runtime transition; the first gate is a generated `leave`/revoke transition, compared with a manual adapter on the same histories.

を、次の型に置き換える(r20 の提案文。**数値は実測値だけを入れる**):

> **End-to-end result.** In MetaTensor, on a two-worker uncommitted gradient
> aggregate, MOTION generated a membership guard, per-contributor retention
> and a `leave`/revoke transition from the interpreter rule. Across the
> preemption, fault and pre-contribution histories, the generated execution
> matched the unspecialised reference; an erasure mutant was rejected. The
> matched manual adapter required [N_adapter] hand-written lines, versus
> [N_rule] for the change rule and annotations.

§3 の頁はほぼ満杯(51〜53 行)。入れるときは preliminary artifact の段落を 2〜3 行に縮める。

**Part I §4 Device**:「The training example (§1) isolates contributor provenance …」の文を、「the end-to-end result (§3) runs this example on the device」を含む形に更新する。

**Part II §3 WP3**:「Measured baseline」の直後に、2.7 の性能数値と「融合という情報を消す合成を跨いで遷移が保たれた」ことを1段落で追加。Step 2 で r20 が求める「composition を跨ぐ」の最小の証拠になる。

**CV**:MetaTensor の output の説明に「end-to-end generated transition」を1句加えるかどうかは、preprint に含めるかどうかで決める。

## 4. 時間の目安(締切 2026-10-14)

| 期間 | 作業 |
|---|---|
| 〜9/27 | 2.1〜2.2(アプリ、変化規則、ファイアウォールの記録) |
| 〜10/4 | 2.3(guard、resume data への保持、順序付き遷移) |
| 〜10/8 | 2.4〜2.6(3履歴、checker、mutant、行数) |
| 〜10/10 | 本文への反映、gpt-6-sol で再レビュー |
| 〜10/14 | 最終チェック(`make final`、頁数、draftnote の除去) |

2.7 の性能は余裕があれば行う。Step 1 には不要で、Step 2(面接前の Part II 評価)で効く。

## 5. リスクと代替

- **resume data への保持が大掛かりになる場合**:保持を virtual のまま resume data に載せる代わりに、「guard 失敗時に materialise する寄与者別バッファ」を MOTION が挿入する形でもよい。r20 の要件は「インタプリタ規則から生成されること」と「独立検査」であり、保持の実装方式は問われていない。
- **間に合わない場合**:Python の制限 IR 上のプロトタイプ(既存の `artifact/` の拡張)で同じ3履歴・mutant・計数を満たす。本文では「restricted-IR prototype of MOTION's derivation」と正直に書く。ただし r15 の「non-toy」には届かないので、A への効果は MetaTensor 版より小さい。
- **やってはいけないこと**:実測していない数値を本文に入れること。placeholder `[N_*]` のまま提出しないこと。
