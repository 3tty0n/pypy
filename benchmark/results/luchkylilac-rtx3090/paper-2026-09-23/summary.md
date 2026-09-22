# MLSys evaluation additions, 2026-09-23

Rows in data.tar.gz (benchmark/paper/archive.sh unpack).  The MOTION (ERC)
results of the same day are in motion-2026-09-23, not here.

## Gathers as fusion nodes: interleaved A/B, 3 rounds

build2 (`ee67b4fb95f8`, before) against build6 (`85c2bd7f4415`, gather node), llama.py forward.

| model | launches/forward | median steady us | new/old (per round) |
|---|---|---|---|
| smollm2-135m | 212.1 -> 182.1 | 3672 -> 3589 | 0.985, 0.970, 0.977 |
| smollm2-360m | 226.1 -> 194.1 | 6062 -> 6006 | 0.990, 0.991, 0.991 |
| qwen2.5-0.5b | 194.1 -> 146.1 | 6518 -> 6378 | 0.978, 0.979, 0.979 |
| smollm2-1.7b | 170.1 -> 146.1 | 14245 -> 14119 | 0.992, 0.991, 0.990 |

## Next-token latency: greedy decode with a key/value cache

`bash benchmark/paper/run_decode.sh`, prompt 63 + 64 generated tokens, 5 generations per process (first cold), 10 processes; median [2nd, 9th] of the per-process medians.  Every run of every system produced the same token stream; ours also matched a full causal forward at every generated position.  Figure: figures/decode.pdf.

| model | system | us/token | launches/token | tokens match |
|---|---|---|---|---|
| distilgpt2 | torch-eager | 2017 [2009, 2161] | - | 10/10 |
| distilgpt2 | ours | 744 [738, 751] | 49.97 | 10/10 |
| distilgpt2 | torch-compile | 852 [814, 928] | - | 10/10 |
| distilgpt2 | torch-compile-ro | 890 [812, 918] | - | 10/10 |
| gpt2 | torch-eager | 3778 [3704, 4166] | - | 10/10 |
| gpt2 | ours | 1175 [1172, 1181] | 92.63 | 10/10 |
| gpt2 | torch-compile | 1471 [1289, 1607] | - | 10/10 |
| gpt2 | torch-compile-ro | 1416 [1265, 1459] | - | 10/10 |
| gpt2-medium | torch-eager | 7237 [7195, 7732] | - | 10/10 |
| gpt2-medium | ours | 2684 [2680, 2685] | 177.98 | 10/10 |
| gpt2-medium | torch-compile | 2460 [2459, 2508] | - | 10/10 |
| gpt2-medium | torch-compile-ro | 2474 [2466, 2511] | - | 10/10 |
