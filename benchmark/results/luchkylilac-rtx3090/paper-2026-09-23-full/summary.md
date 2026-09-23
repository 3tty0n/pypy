# Paper benchmark summary

## Microbenchmarks (median steady_us over rounds)

| variant | k | n | fused (ours) | app-level (ours) | eager (ours) | nojit (ours) | torch.compile | compile-ro | compile-mat | torch eager | JAX/XLA | IREE | Triton | speedup vs compile | interp tax | ours/Triton |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1 | 1000000 | 29.1 | 29.1 | 77.2 | 77.2 | 60.2 | 116.2 | 101.5 | 78.3 | 48.9 | 5678.8 | 30.8 | 2.07x | 1.00x | 0.95x |
| 0 | 4 | 10000 | 13.1 | 19.5 | 39.7 | 39.8 | 53.6 | 113.2 | 102.3 | 60.1 | 44.3 | 437.9 | 19.1 | 4.10x | 1.49x | 0.69x |
| 0 | 4 | 100000 | 13.8 | 21.7 | 43.4 | 43.3 | 45.7 | 112.7 | 119.8 | 59.9 | 46.3 | 934.3 | 19.6 | 3.31x | 1.57x | 0.70x |
| 0 | 4 | 1000000 | 39.8 | 117.4 | 306.4 | 306.8 | 68.4 | 133.8 | 133.9 | 312.5 | 95.1 | 5749.6 | 39.3 | 1.72x | 2.95x | 1.01x |
| 0 | 4 | 10000000 | 329.1 | 1133.4 | 3027.0 | 3027.2 | 646.0 | 1226.2 | 1197.0 | 3045.3 | 946.5 | 51498.0 | 326.9 | 1.96x | 3.44x | 1.01x |
| 0 | 8 | 1000000 | 70.1 | 229.6 | 613.0 | 613.7 | 132.8 | 198.3 | 198.4 | 624.8 | 176.1 | 5886.9 | 70.5 | 1.89x | 3.27x | 1.00x |
| 1 | 4 | 1000000 | 41.6 | 119.7 | 312.2 | 311.0 | 307.8 | 393.6 | 387.9 | 316.8 | 99.6 | 6466.2 | 43.3 | 7.40x | 2.88x | 0.96x |
| 2 | 4 | 1000000 | 42.7 | 119.4 | 311.0 | 311.0 | 72.8 | 138.4 | 138.5 | 326.4 | 173.2 | 9008.1 | 68.7 | 1.71x | 2.80x | 0.62x |
| 3 | 4 | 1000000 | 98.0 | 178.4 | 360.1 | 367.0 | 172.3 | 390.9 | 382.8 | 376.8 | 308.4 | 81575.5 | 122.9 | 1.76x | 1.82x | 0.80x |
| 4 | 4 | 1000000 | 53.4 | 145.1 | 336.3 | 335.9 | 332.2 | 424.5 | 423.3 | 342.4 | 127.5 | 10801.0 | 69.1 | 6.22x | 2.72x | 0.77x |
| 5 | 4 | 1000000 | 53.5 | 147.0 | 335.8 | 335.7 | 332.6 | 426.2 | 414.5 | 342.4 | 125.6 | 10726.4 | 69.4 | 6.22x | 2.75x | 0.77x |
| 6 | 1 | 25600 | 279.4 | 270.0 | 279.0 | 278.6 | 393.6 | 429.4 | 432.7 | 393.5 | 107.3 | 5347.2 | 880.0 | 1.41x | 0.97x | 0.32x |
| 6 | 1 | 256000 | 1015.9 | 1006.0 | 1016.5 | 1016.5 | 1290.7 | 1293.3 | 1292.9 | 1248.3 | 751.0 | 910.7 | 850.3 | 1.27x | 0.99x | 1.19x |
| 7 | 1 | 25600 | 648.4 | 790.3 | 669.7 | 669.5 | 848.5 | 885.3 | 897.3 | 804.0 | 324.6 | 11551.7 | n/a | 1.31x | 1.22x | n/a |
| 7 | 1 | 256000 | 2808.4 | 2942.8 | 2834.7 | 2830.9 | 3000.3 | 3070.7 | 3083.1 | 3058.5 | 1954.2 | 31555.5 | n/a | 1.07x | 1.05x | n/a |
| 8 | 1 | 25600 | 464.3 | 505.0 | 780.2 | 780.4 | 515.4 | 524.5 | 501.4 | 551.9 | 266.9 | 932.9 | n/a | 1.11x | 1.09x | n/a |
| 8 | 1 | 256000 | 22334.3 | 22426.8 | 25184.5 | 25184.6 | 29883.1 | 29726.9 | 27710.5 | 27487.4 | 15462.8 | 38584.8 | n/a | 1.34x | 1.00x | n/a |
| 9 | 1 | 25600 | 757.2 | 756.0 | 764.2 | 765.2 | 859.0 | 875.8 | 891.9 | 806.2 | 255.6 | n/a | n/a | 1.13x | 1.00x | n/a |
| 9 | 1 | 256000 | 573.3 | 555.3 | 602.2 | 569.2 | 680.0 | 699.1 | 694.5 | 591.0 | 329.7 | n/a | n/a | 1.19x | 0.97x | n/a |
| 10 | 1 | 25600 | 3916.2 | 4833.8 | 4836.0 | 4826.5 | 3162.6 | 3126.9 | 3125.9 | 3706.4 | 1498.6 | 10423.8 | n/a | 0.81x | 1.23x | n/a |
| 10 | 1 | 191488 | 94067.1 | 96442.5 | 99362.6 | 99378.7 | 78698.4 | 78704.7 | 75787.2 | 78956.3 | 41978.5 | 136401.1 | n/a | 0.84x | 1.03x | n/a |
| 11 | 1 | 25600 | 3.7 | 5.0 | 56.8 | 57.1 | 63.0 | 127.3 | 122.7 | 57.8 | 49.6 | 56.0 | 18.8 | 17.22x | 1.36x | 0.19x |
| 11 | 1 | 256000 | 15.9 | 14.8 | 143.5 | 141.7 | 171.8 | 209.4 | 228.4 | 156.3 | 51.7 | 184.6 | 20.1 | 10.82x | 0.93x | 0.79x |
| 12 | 1 | 25600 | 98.8 | 90.0 | 98.5 | 98.5 | 138.3 | 164.4 | 180.7 | 137.2 | 48.8 | 1815.6 | 298.8 | 1.40x | 0.91x | 0.33x |
| 12 | 1 | 256000 | 342.9 | 335.4 | 343.2 | 343.1 | 496.1 | 550.7 | 524.8 | 493.5 | 258.2 | 437.2 | 299.8 | 1.45x | 0.98x | 1.14x |
| 13 | 1 | 25600 | 437.5 | 447.8 | 526.9 | 526.8 | 585.3 | 645.9 | 617.8 | 595.3 | 123.3 | 2800.2 | 1274.5 | 1.34x | 1.02x | 0.34x |
| 13 | 1 | 256000 | 4742.5 | 4784.9 | 5423.9 | 5427.8 | 4992.8 | 5057.0 | 5058.7 | 4886.0 | 3734.1 | 5138.4 | 4683.8 | 1.05x | 1.01x | 1.01x |

## Precision sweep (median steady_us)

| dtype | variant | k | n | fused (ours) | app-level (ours) | eager (ours) | nojit (ours) | torch.compile | compile-ro | compile-mat | torch eager | JAX/XLA | IREE | Triton | speedup vs compile | interp tax | ours/Triton |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| float16 | 8 | 1 | 256000 | 708.7 | n/a | 2057.0 | 2059.4 | 885.8 | 891.4 | 779.9 | 1463.7 | 633.8 | 1891.4 | n/a | 1.25x | n/a | n/a |
| float16 | 11 | 1 | 256000 | 22.5 | n/a | 59.5 | 55.6 | 152.8 | 223.9 | 206.7 | 146.7 | 47.9 | 177.8 | 19.0 | 6.80x | n/a | 1.18x |
| float16 | 13 | 1 | 256000 | 211.3 | n/a | 333.9 | 336.8 | 434.1 | 382.7 | 417.1 | 411.9 | 99.4 | 600.8 | 122.1 | 2.05x | n/a | 1.73x |
| float32 | 8 | 1 | 256000 | 1394.1 | n/a | 3914.3 | 3917.8 | 2028.4 | 2022.7 | 1496.1 | 2649.5 | 1260.5 | 7316.7 | n/a | 1.46x | n/a | n/a |
| float32 | 11 | 1 | 256000 | 6.8 | n/a | 41.7 | 38.2 | 159.7 | 245.0 | 243.4 | 147.7 | 42.2 | 82.2 | 19.4 | 23.49x | n/a | 0.35x |
| float32 | 13 | 1 | 256000 | 271.0 | n/a | 587.6 | 583.3 | 550.6 | 614.0 | 512.5 | 617.7 | 215.5 | 1304.0 | 125.3 | 2.03x | n/a | 2.16x |

## Guards / graph breaks (launches per iter, torch graph breaks)

| variant | n | launches/iter (fused) | torch graphs | torch breaks |
|---|---|---|---|---|
| 1 | 1000000 | 1.1 | 1 | 0 |
| 2 | 1000000 | 1.1 | 2 | 1 |
| 3 | 1000000 | 2.0 | 2 | 1 |
| 4 | 1000000 | 1.1 | 1 | 0 |
| 5 | 1000000 | 1.1 | 2 | 1 |

## End-to-end models (median steady_us, ratio to torch.compile)

| model | ours | torch.compile | compile-ro | compile-mat | torch eager | JAX/XLA | IREE | TensorRT | ratio ours/compile | ratio jax/compile | launches/iter (ours) | correctness | tol | pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bert-base | 1846.8 | 2281.6 | 2081.2 | 1857.2 | 4517.2 | 1469.2 | 25976.4 | 2140.5 | 0.81x | 0.64x | 64.3 | 8.10623e-05 | 0.000504128 | pass |
| bert-mini | 428.0 | 872.4 | 481.7 | 408.7 | 1821.9 | 243.8 | 3616.8 | 740.4 | 0.49x | 0.28x | 24.3 | 1.90735e-05 | 0.000445395 | pass |
| deit-tiny | 1038.2 | 2316.6 | 1381.2 | 1315.9 | 3957.1 | 770.4 | n/a | 1123.2 | 0.45x | 0.33x | 76.3 | 1.07288e-05 | 0.000134346 | pass |
| distilgpt2 | 1291.2 | 1359.7 | 1284.0 | 1273.5 | 2825.9 | 998.4 | 17173.1 | 1719.1 | 0.95x | 0.73x | 38.2 | 0.000190735 | 0.0034415 | pass |
| gpt2 | 2096.3 | 2191.3 | 2056.3 | 2008.0 | 5204.3 | 1650.0 | 28731.1 | 2678.4 | 0.96x | 0.75x | 74.2 | 0.000854492 | 0.00663646 | pass |
| gpt2-medium | 4797.6 | 5039.6 | 4781.4 | 4439.6 | 10344.3 | 3896.0 | n/a | 5719.7 | 0.95x | 0.77x | 146.2 | 0.00343323 | 0.00707528 | pass |
| mixer_b16 | 2715.6 | 3164.0 | 3086.7 | 3136.6 | 3024.9 | 2248.2 | 674346.1 | 2498.5 | 0.86x | 0.71x | 63.4 | 3.62396e-05 | 0.000280422 | pass |
| qwen2.5-0.5b | 6403.6 | 7431.4 | 7095.8 | 7101.0 | 13765.6 | 5232.0 | 86360.0 | 7102.4 | 0.86x | 0.70x | 146.1 | 0.000150681 | 0.000508814 | pass |
| resnet18-b1 | 728.5 | 1110.5 | 919.2 | 1035.9 | 1643.3 | 1097.0 | n/a | 997.7 | 0.66x | 0.99x | 39.0 | 0.00963783 | 0.0294047 | pass |
| resnet18-b8 | 2718.6 | 2186.6 | 2123.5 | 2122.7 | 2261.3 | 2278.0 | n/a | 1952.8 | 1.24x | 1.04x | 39.0 | 0.00571394 | 0.0293988 | pass |
| smollm2-1.7b | 14144.7 | 18198.4 | 17882.2 | 17903.1 | 19589.6 | 13528.8 | n/a | 16795.9 | 0.78x | 0.74x | 146.1 | 7.49826e-05 | 0.000587335 | pass |
| smollm2-135m | 3607.6 | 5336.9 | 3809.8 | 3845.1 | 15060.8 | 2662.6 | 19159.5 | 3738.6 | 0.68x | 0.50x | 182.1 | 0.000201941 | 0.000589316 | pass |
| smollm2-360m | 6028.9 | 6819.5 | 6278.3 | 6308.4 | 17248.1 | 4584.1 | 32981.2 | 6396.0 | 0.88x | 0.67x | 194.1 | 4.36306e-05 | 0.00060394 | pass |
| vit-base | 3721.1 | 4651.4 | 4573.6 | 4408.8 | 4477.2 | 3179.7 | 883380.1 | 3778.3 | 0.80x | 0.68x | 76.3 | 9.23872e-06 | 0.000248432 | pass |

## Correctness on five derived inputs (worst case per model; ours is the reference)

| model | system | max maxabsdiff | min argmax match | tol | passed |
|---|---|---|---|---|---|
| bert-base | jax | 0.000101 | 1.000 | 0.000535593 | 5/5 |
| bert-base | torch-compile | 9.3e-05 | 1.000 | 0.000535593 | 5/5 |
| bert-base | torch-eager | 8.01e-05 | 1.000 | 0.000535593 | 5/5 |
| bert-mini | jax | 2.48e-05 | 1.000 | 0.0005167 | 5/5 |
| bert-mini | torch-compile | 2.29e-05 | 1.000 | 0.0005167 | 5/5 |
| bert-mini | torch-eager | 2.67e-05 | 1.000 | 0.0005167 | 5/5 |
| deit-tiny | jax | 1.12e-05 | 1.000 | 0.000133298 | 5/5 |
| deit-tiny | torch-compile | 1.34e-05 | 1.000 | 0.000133298 | 5/5 |
| deit-tiny | torch-eager | 1.32e-05 | 1.000 | 0.000133298 | 5/5 |
| distilgpt2 | jax | 0.000221 | 1.000 | 0.00311393 | 5/5 |
| distilgpt2 | torch-compile | 0.000282 | 1.000 | 0.00311393 | 5/5 |
| distilgpt2 | torch-eager | 0.000282 | 1.000 | 0.00311393 | 5/5 |
| gpt2 | jax | 0.00335 | 1.000 | 0.00648755 | 5/5 |
| gpt2 | torch-compile | 0.00463 | 1.000 | 0.00648755 | 5/5 |
| gpt2 | torch-eager | 0.00389 | 1.000 | 0.00648755 | 5/5 |
| gpt2-medium | jax | 0.00516 | 1.000 | 0.0073956 | 5/5 |
| gpt2-medium | torch-compile | 0.00645 | 1.000 | 0.0073956 | 5/5 |
| gpt2-medium | torch-eager | 0.00439 | 1.000 | 0.0073956 | 5/5 |
| mixer_b16 | jax | 7.06e-05 | 1.000 | 0.000280097 | 5/5 |
| mixer_b16 | torch-compile | 5.91e-05 | 1.000 | 0.000280097 | 5/5 |
| mixer_b16 | torch-eager | 5.15e-05 | 1.000 | 0.000280097 | 5/5 |
| qwen2.5-0.5b | jax | 0.000376 | 1.000 | 0.000553048 | 5/5 |
| qwen2.5-0.5b | torch-compile | 0.000137 | 1.000 | 0.000553048 | 5/5 |
| qwen2.5-0.5b | torch-eager | 0.000149 | 1.000 | 0.000553048 | 5/5 |
| resnet18-b1 | jax | 0.0167 | 1.000 | 0.0299742 | 5/5 |
| resnet18-b1 | torch-compile | 0.00942 | 1.000 | 0.0299742 | 5/5 |
| resnet18-b1 | torch-eager | 0.0158 | 1.000 | 0.0299742 | 5/5 |
| resnet18-b8 | jax | 0.0123 | 1.000 | 0.0299708 | 5/5 |
| resnet18-b8 | torch-compile | 0.0122 | 1.000 | 0.0299708 | 5/5 |
| resnet18-b8 | torch-eager | 0.00725 | 1.000 | 0.0299708 | 5/5 |
| smollm2-1.7b | jax | 0.00027 | 1.000 | 0.000504007 | 5/5 |
| smollm2-1.7b | torch-compile | 0.000765 | 1.000 | 0.000504007 | 2/5 |
| smollm2-1.7b | torch-eager | 0.00107 | 1.000 | 0.000504007 | 3/5 |
| smollm2-135m | jax | 0.000215 | 1.000 | 0.000808265 | 5/5 |
| smollm2-135m | torch-compile | 0.000168 | 1.000 | 0.000808265 | 5/5 |
| smollm2-135m | torch-eager | 0.000147 | 1.000 | 0.000808265 | 5/5 |
| smollm2-360m | jax | 0.000318 | 1.000 | 0.000582302 | 5/5 |
| smollm2-360m | torch-compile | 0.00022 | 1.000 | 0.000582302 | 5/5 |
| smollm2-360m | torch-eager | 0.000219 | 1.000 | 0.000582302 | 5/5 |
| vit-base | jax | 1.26e-05 | 1.000 | 0.000248661 | 5/5 |
| vit-base | torch-compile | 1.17e-05 | 1.000 | 0.000248661 | 5/5 |
| vit-base | torch-eager | 1.07e-05 | 1.000 | 0.000248661 | 5/5 |

## Batch-size sweep (median steady_us per forward; per_seq = steady/B)

| model | batch | ours | torch.compile | compile-ro | torch eager | JAX/XLA | ratio ours/compile | ratio ours/jax | per_seq ours | per_seq compile | per_seq jax | rows identical | failed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bert-base | 1 | 1843.6 | 2296.2 | 2090.2 | 4537.5 | 1465.2 | 0.80x | 1.26x | 1843.6 | 2296.2 | 1465.2 | yes |  |
| bert-base | 2 | 2930.4 | 3088.1 | 2935.1 | 4434.2 | 2374.0 | 0.95x | 1.23x | 1465.2 | 1544.0 | 1187.0 | yes |  |
| bert-base | 4 | 4683.0 | 5123.7 | 5041.0 | 4779.2 | 3802.7 | 0.91x | 1.23x | 1170.7 | 1280.9 | 950.7 | yes |  |
| bert-base | 8 | 7773.6 | 8217.6 | 8139.4 | 8036.3 | 6918.9 | 0.95x | 1.12x | 971.7 | 1027.2 | 864.9 | yes |  |
| bert-base | 16 | 13723.4 | 14365.1 | 14282.6 | 14185.8 | 12818.1 | 0.96x | 1.07x | 857.7 | 897.8 | 801.1 | yes |  |
| bert-base | 32 | 26509.2 | 27633.1 | 27556.8 | 27997.2 | 25556.5 | 0.96x | 1.04x | 828.4 | 863.5 | 798.6 | yes |  |
| distilgpt2 | 1 | 1301.2 | 1364.2 | 1293.5 | 2798.0 | 1002.9 | 0.95x | 1.30x | 1301.2 | 1364.2 | 1002.9 | yes |  |
| distilgpt2 | 2 | 1872.3 | 1755.0 | 1657.6 | 2803.0 | 1547.4 | 1.07x | 1.21x | 936.2 | 877.5 | 773.7 | yes |  |
| distilgpt2 | 4 | 2991.2 | 2808.1 | 2729.9 | 3141.5 | 2606.2 | 1.07x | 1.15x | 747.8 | 702.0 | 651.6 | yes |  |
| distilgpt2 | 8 | 5179.8 | 5095.2 | 5024.5 | 5883.2 | 4791.5 | 1.02x | 1.08x | 647.5 | 636.9 | 598.9 | yes |  |
| distilgpt2 | 16 | 9422.8 | 9149.2 | 9090.6 | 10756.6 | 9030.6 | 1.03x | 1.04x | 588.9 | 571.8 | 564.4 | yes |  |
| distilgpt2 | 32 | 18410.9 | 17958.0 | 17897.0 | 20947.1 | 18067.1 | 1.03x | 1.02x | 575.3 | 561.2 | 564.6 | yes |  |

## Ablations (median steady_us)

| experiment | variant | model | steady_us | launches/iter | note |
|---|---|---|---|---|---|
| budget_mb | 64 | distilgpt2 | 1295.1 | n/a |  |
| budget_mb | 8 | distilgpt2 | 1307.4 | n/a |  |
| flat_block | 256 | distilgpt2 | 1260.2 | n/a |  |
| flat_block | 256 | resnet18 | 637.4 | n/a |  |
| flat_block | 4096 | distilgpt2 | 1292.5 | n/a |  |
| flat_block | 4096 | resnet18 | 700.9 | n/a |  |
| fusion | off | distilgpt2 | 1964.4 | n/a | enable_opts minus tensor |
| fusion | on | distilgpt2 | 1292.5 | n/a |  |
| gather | fused | qwen2.5-0.5b | 6393.8 | 146.1 | checksum=-16811852.000000 |
| gather | fused | smollm2-1.7b | 14185.3 | 146.1 | checksum=9049722.000000 |
| gather | fused | smollm2-135m | 3595.2 | 182.1 | checksum=16329577.000000 |
| gather | fused | smollm2-360m | 6017.8 | 194.1 | checksum=15922315.000000 |
| gather | standalone | qwen2.5-0.5b | 6536.3 | 194.1 | checksum=-16811854.000000 |
| gather | standalone | smollm2-1.7b | 14264.1 | 170.1 | checksum=9049719.000000 |
| gather | standalone | smollm2-135m | 3687.6 | 212.1 | checksum=16329577.000000 |
| gather | standalone | smollm2-360m | 6078.7 | 226.1 | checksum=15922316.000000 |
| max_inputs | mi4 | bert-mini | 466.1 | 34.3 |  |
| max_inputs | mi4 | distilgpt2 | 1320.0 | 45.2 |  |
| max_inputs | mi4 | mixer_b16 | 2742.2 | 76.4 |  |
| max_inputs | mi6 | bert-mini | 423.7 | 24.3 |  |
| max_inputs | mi6 | distilgpt2 | 1290.6 | 38.2 |  |
| max_inputs | mi6 | mixer_b16 | 2694.8 | 63.4 |  |
| max_inputs | mi8 | bert-mini | 431.6 | 24.3 |  |
| max_inputs | mi8 | distilgpt2 | 1292.4 | 38.2 |  |
| max_inputs | mi8 | mixer_b16 | 2695.6 | 63.4 |  |
| precision | float16 | distilgpt2 | 921.8 | n/a |  |
| precision | float16 | smollm2-135m | 2157.8 | n/a |  |
| precision | float32 | distilgpt2 | 1294.2 | n/a |  |
| precision | float32 | smollm2-135m | 3614.9 | n/a |  |
| tf32 | fp32 | resnet18-b1 | 785.8 | n/a |  |
| tf32 | fp32 | resnet18-b8 | 3213.9 | n/a |  |
| tf32 | tf32 | resnet18-b1 | 722.5 | n/a |  |
| tf32 | tf32 | resnet18-b8 | 2735.7 | n/a |  |
| tf32 | torch-fp32 | resnet18-b8 | 3068.0 | n/a |  |

## Deoptimization cost (median over rounds, us per iteration)

| system | pattern | steady_us | first_fail_us | after_fail_us | peak_us | cold_us | launches/it | loops | bridges |
|---|---|---|---|---|---|---|---|---|---|
| ours | never | 37.9 | n/a | n/a | 64 | 4024 | 2.02 | 0 | 0 |
| torch-compile | never | 176.4 | n/a | n/a | 209 | 587595 | n/a | 0 | n/a |
| torch-compile-ro | never | 244.4 | n/a | n/a | 280 | 538822 | n/a | 0 | n/a |
| torch-eager | never | 114.9 | n/a | n/a | 127 | 39838 | n/a | 0 | n/a |
| ours | alternate | 37.6 | 60 | 54 | 281 | 4075 | 2.17 | 0 | 2 |
| torch-compile | alternate | 177.5 | 122103 | 195 | 122103 | 584289 | n/a | 2 | n/a |
| torch-compile-ro | alternate | 263.0 | 126324 | 346 | 126324 | 598603 | n/a | 2 | n/a |
| torch-eager | alternate | 115.4 | 116 | 116 | 148 | 39770 | n/a | 0 | n/a |
| ours | both-hot | 37.2 | n/a | n/a | 65 | 4088 | 2.02 | 0 | 0 |
| torch-compile | both-hot | 188.4 | n/a | n/a | 208 | 587942 | n/a | 0 | n/a |
| torch-compile-ro | both-hot | 249.5 | n/a | n/a | 295 | 541611 | n/a | 0 | n/a |
| torch-eager | both-hot | 110.4 | n/a | n/a | 120 | 36842 | n/a | 0 | n/a |
| ours | fresh | 69.0 | 68 | 70 | 1590 | 4011 | 7.00 | 0 | 63 |
| torch-compile | fresh | 145.2 | 147 | 147 | 165 | 530990 | n/a | 0 | n/a |
| torch-compile-ro | fresh | 166.7 | 169 | 168 | 216 | 591332 | n/a | 0 | n/a |
| torch-eager | fresh | 112.5 | 116 | 114 | 143 | 38822 | n/a | 0 | n/a |
| ours | probe-a | 35.0 | n/a | 52 | 1799 | 3777 | 2.25 | 5 | 3 |
| ours | probe-e | 17.9 | 1345 | 33 | 1533 | 94 | 1.09 | 3 | 4 |

## Compilation overhead (median; break-even vs torch eager, iterations)

| workload | system | compile_ms | first_run_ms | steady_us | break-even |
|---|---|---|---|---|---|
| bert-base | torch-compile | n/a | 2083.3 | 2281.6 | 882 |
| bert-base | torch-compile-ro | n/a | 1947.4 | 2081.2 | 754 |
| bert-base | torch-compile-mat | n/a | 2064.9 | 1857.2 | 735 |
| bert-base | torch-tensorrt | n/a | 7609.6 | 2140.5 | 3155 |
| bert-base | jax | 8224.5 | 7.2 | 1469.2 | 2664 |
| bert-base | iree | 1632.6 | 113.8 | 25976.4 | n/a |
| bert-base | ours | n/a | 422.7 | 1846.8 | 117 |
| bert-mini | torch-compile | n/a | 1331.7 | 872.4 | 1303 |
| bert-mini | torch-compile-ro | n/a | 1305.5 | 481.7 | 904 |
| bert-mini | torch-compile-mat | n/a | 3144.3 | 408.7 | 2158 |
| bert-mini | torch-tensorrt | n/a | 4454.4 | 740.4 | 4031 |
| bert-mini | jax | 7422.7 | 2.9 | 243.8 | 4646 |
| bert-mini | iree | 999.6 | 19.7 | 3616.8 | n/a |
| bert-mini | ours | n/a | 183.4 | 428.0 | 64 |
| deit-tiny | torch-compile | n/a | 1827.0 | 2316.6 | 1034 |
| deit-tiny | torch-compile-ro | n/a | 1834.0 | 1381.2 | 661 |
| deit-tiny | torch-compile-mat | n/a | 3534.9 | 1315.9 | 1289 |
| deit-tiny | torch-tensorrt | n/a | 7250.1 | 1123.2 | 2512 |
| deit-tiny | jax | 6925.6 | 7.9 | 770.4 | 2135 |
| deit-tiny | ours | n/a | 150.5 | 1038.2 | 7 |
| distilgpt2 | torch-compile | n/a | 1512.9 | 1359.7 | 939 |
| distilgpt2 | torch-compile-ro | n/a | 1563.8 | 1284.0 | 925 |
| distilgpt2 | torch-compile-mat | n/a | 2035.7 | 1273.5 | 1223 |
| distilgpt2 | torch-tensorrt | n/a | 15282.6 | 1719.1 | 13684 |
| distilgpt2 | jax | 6842.5 | 4.8 | 998.4 | 3672 |
| distilgpt2 | iree | 873.2 | 76.0 | 17173.1 | n/a |
| distilgpt2 | ours | n/a | 583.3 | 1291.2 | 291 |
| gpt2 | torch-compile | n/a | 2062.5 | 2191.3 | 639 |
| gpt2 | torch-compile-ro | n/a | 2104.4 | 2056.3 | 625 |
| gpt2 | torch-compile-mat | n/a | 3113.8 | 2008.0 | 931 |
| gpt2 | torch-tensorrt | n/a | 27137.9 | 2678.4 | 10689 |
| gpt2 | jax | 7534.6 | 8.7 | 1650.0 | 2084 |
| gpt2 | iree | 1199.3 | 124.5 | 28731.1 | n/a |
| gpt2 | ours | n/a | 647.5 | 2096.3 | 164 |
| gpt2-medium | torch-compile | n/a | 3113.1 | 5039.6 | 561 |
| gpt2-medium | torch-compile-ro | n/a | 3206.3 | 4781.4 | 551 |
| gpt2-medium | torch-compile-mat | n/a | 3316.4 | 4439.6 | 538 |
| gpt2-medium | torch-tensorrt | n/a | 52111.7 | 5719.7 | 11238 |
| gpt2-medium | jax | 7771.0 | 17.1 | 3896.0 | 1186 |
| gpt2-medium | ours | n/a | 1003.0 | 4797.6 | 156 |
| micro v0 k1 n1000000 | torch-compile | n/a | 442.3 | 60.2 | 22447 |
| micro v0 k1 n1000000 | torch-compile-ro | n/a | 502.0 | 116.2 | n/a |
| micro v0 k1 n1000000 | torch-compile-mat | n/a | 612.2 | 101.5 | n/a |
| micro v0 k1 n1000000 | jax | 205.0 | 0.5 | 48.9 | 5769 |
| micro v0 k1 n1000000 | iree | 574.1 | 24.0 | 5678.8 | n/a |
| micro v0 k1 n1000000 | triton | 319.7 | 0.1 | 30.8 | 5964 |
| micro v0 k4 n10000 | torch-compile | n/a | 434.1 | 53.6 | 61313 |
| micro v0 k4 n10000 | torch-compile-ro | n/a | 447.5 | 113.2 | n/a |
| micro v0 k4 n10000 | torch-compile-mat | n/a | 574.9 | 102.3 | n/a |
| micro v0 k4 n10000 | jax | 202.4 | 0.5 | 44.3 | 10531 |
| micro v0 k4 n10000 | iree | 585.3 | 10.0 | 437.9 | n/a |
| micro v0 k4 n10000 | triton | 320.9 | 0.1 | 19.1 | 6927 |
| micro v0 k4 n100000 | torch-compile | n/a | 499.9 | 45.7 | 32563 |
| micro v0 k4 n100000 | torch-compile-ro | n/a | 492.1 | 112.7 | n/a |
| micro v0 k4 n100000 | torch-compile-mat | n/a | 553.2 | 119.8 | n/a |
| micro v0 k4 n100000 | jax | 193.0 | 0.5 | 46.3 | 11417 |
| micro v0 k4 n100000 | iree | 576.4 | 12.1 | 934.3 | n/a |
| micro v0 k4 n100000 | triton | 320.7 | 0.1 | 19.6 | 7017 |
| micro v0 k4 n1000000 | torch-compile | n/a | 446.5 | 68.4 | 1678 |
| micro v0 k4 n1000000 | torch-compile-ro | n/a | 456.5 | 133.8 | 2348 |
| micro v0 k4 n1000000 | torch-compile-mat | n/a | 567.6 | 133.9 | 2973 |
| micro v0 k4 n1000000 | jax | 208.6 | 0.6 | 95.1 | 793 |
| micro v0 k4 n1000000 | iree | 573.5 | 24.5 | 5749.6 | n/a |
| micro v0 k4 n1000000 | triton | 320.7 | 0.1 | 39.3 | 1039 |
| micro v0 k4 n10000000 | torch-compile | n/a | 444.6 | 646.0 | 169 |
| micro v0 k4 n10000000 | torch-compile-ro | n/a | 503.1 | 1226.2 | 255 |
| micro v0 k4 n10000000 | torch-compile-mat | n/a | 617.9 | 1197.0 | 314 |
| micro v0 k4 n10000000 | jax | 222.8 | 1.3 | 946.5 | 89 |
| micro v0 k4 n10000000 | iree | 577.4 | 170.7 | 51498.0 | n/a |
| micro v0 k4 n10000000 | triton | 300.8 | 0.6 | 326.9 | 97 |
| micro v0 k8 n1000000 | torch-compile | n/a | 446.8 | 132.8 | 834 |
| micro v0 k8 n1000000 | torch-compile-ro | n/a | 454.7 | 198.3 | 981 |
| micro v0 k8 n1000000 | torch-compile-mat | n/a | 616.4 | 198.4 | 1360 |
| micro v0 k8 n1000000 | jax | 219.0 | 0.7 | 176.1 | 408 |
| micro v0 k8 n1000000 | iree | 581.8 | 24.2 | 5886.9 | n/a |
| micro v0 k8 n1000000 | triton | 326.9 | 0.2 | 70.5 | 524 |
| micro v1 k4 n1000000 | torch-compile | n/a | 502.4 | 307.8 | 51075 |
| micro v1 k4 n1000000 | torch-compile-ro | n/a | 448.2 | 393.6 | n/a |
| micro v1 k4 n1000000 | torch-compile-mat | n/a | 578.4 | 387.9 | n/a |
| micro v1 k4 n1000000 | jax | 206.0 | 0.6 | 99.6 | 770 |
| micro v1 k4 n1000000 | iree | 575.8 | 24.2 | 6466.2 | n/a |
| micro v1 k4 n1000000 | triton | 320.2 | 0.1 | 43.3 | 1027 |
| micro v10 k1 n191488 | torch-compile | n/a | 1871.9 | 78698.4 | 5764 |
| micro v10 k1 n191488 | torch-compile-ro | n/a | 1982.4 | 78704.7 | 6346 |
| micro v10 k1 n191488 | torch-compile-mat | n/a | 6665.3 | 75787.2 | 1982 |
| micro v10 k1 n191488 | jax | 5056.0 | 70.1 | 41978.5 | 128 |
| micro v10 k1 n191488 | iree | 3774.9 | 207.8 | 136401.1 | n/a |
| micro v10 k1 n25600 | torch-compile | n/a | 1937.7 | 3162.6 | 2979 |
| micro v10 k1 n25600 | torch-compile-ro | n/a | 1875.8 | 3126.9 | 2688 |
| micro v10 k1 n25600 | torch-compile-mat | n/a | 5089.0 | 3125.9 | 8219 |
| micro v10 k1 n25600 | jax | 4752.6 | 6.8 | 1498.6 | 2012 |
| micro v10 k1 n25600 | iree | 1543.9 | 26.4 | 10423.8 | n/a |
| micro v11 k1 n25600 | torch-compile | n/a | 1705.7 | 63.0 | n/a |
| micro v11 k1 n25600 | torch-compile-ro | n/a | 1722.4 | 127.3 | n/a |
| micro v11 k1 n25600 | torch-compile-mat | n/a | 1760.7 | 122.7 | n/a |
| micro v11 k1 n25600 | jax | 98.0 | 0.5 | 49.6 | -10892 |
| micro v11 k1 n25600 | iree | 190.1 | 8.7 | 56.0 | 5846 |
| micro v11 k1 n25600 | triton | 320.5 | 0.1 | 18.8 | 3399 |
| micro v11 k1 n256000 | torch-compile | n/a | 1741.3 | 171.8 | n/a |
| micro v11 k1 n256000 | torch-compile-ro | n/a | 1637.2 | 209.4 | n/a |
| micro v11 k1 n256000 | torch-compile-mat | n/a | 1881.3 | 228.4 | n/a |
| micro v11 k1 n256000 | jax | 84.2 | 0.5 | 51.7 | -1216 |
| micro v11 k1 n256000 | iree | 185.0 | 10.8 | 184.6 | n/a |
| micro v11 k1 n256000 | triton | 319.5 | 0.1 | 20.1 | 790 |
| micro v12 k1 n25600 | torch-compile | n/a | 1767.6 | 138.3 | n/a |
| micro v12 k1 n25600 | torch-compile-ro | n/a | 1684.2 | 164.4 | n/a |
| micro v12 k1 n25600 | torch-compile-mat | n/a | 1839.5 | 180.7 | n/a |
| micro v12 k1 n25600 | jax | 1591.3 | 0.6 | 48.8 | 15588 |
| micro v12 k1 n25600 | iree | 224.1 | 13.8 | 1815.6 | n/a |
| micro v12 k1 n25600 | triton | 321.5 | 0.4 | 298.8 | n/a |
| micro v12 k1 n256000 | torch-compile | n/a | 1759.3 | 496.1 | n/a |
| micro v12 k1 n256000 | torch-compile-ro | n/a | 1803.5 | 550.7 | n/a |
| micro v12 k1 n256000 | torch-compile-mat | n/a | 1891.2 | 524.8 | n/a |
| micro v12 k1 n256000 | jax | 1613.1 | 1.0 | 258.2 | 5848 |
| micro v12 k1 n256000 | iree | 233.1 | 9.9 | 437.2 | 88 |
| micro v12 k1 n256000 | triton | 320.9 | 0.4 | 299.8 | 430 |
| micro v13 k1 n25600 | torch-compile | n/a | 1747.1 | 585.3 | 149893 |
| micro v13 k1 n25600 | torch-compile-ro | n/a | 1677.8 | 645.9 | n/a |
| micro v13 k1 n25600 | torch-compile-mat | n/a | 1986.8 | 617.8 | n/a |
| micro v13 k1 n25600 | jax | 2078.7 | 1.0 | 123.3 | 3867 |
| micro v13 k1 n25600 | iree | 332.6 | 11.2 | 2800.2 | n/a |
| micro v13 k1 n25600 | triton | 325.0 | 1.4 | 1274.5 | n/a |
| micro v13 k1 n256000 | torch-compile | n/a | 1779.6 | 4992.8 | n/a |
| micro v13 k1 n256000 | torch-compile-ro | n/a | 1775.4 | 5057.0 | n/a |
| micro v13 k1 n256000 | torch-compile-mat | n/a | 2052.6 | 5058.7 | n/a |
| micro v13 k1 n256000 | jax | 2004.5 | 4.8 | 3734.1 | 1502 |
| micro v13 k1 n256000 | iree | 646.7 | 19.8 | 5138.4 | n/a |
| micro v13 k1 n256000 | triton | 327.2 | 4.7 | 4683.8 | 261 |
| micro v2 k4 n1000000 | torch-compile | n/a | 569.3 | 72.8 | 2091 |
| micro v2 k4 n1000000 | torch-compile-ro | n/a | 526.1 | 138.4 | 2591 |
| micro v2 k4 n1000000 | torch-compile-mat | n/a | 720.1 | 138.5 | 3625 |
| micro v2 k4 n1000000 | jax | 209.5 | 0.6 | 173.2 | 1116 |
| micro v2 k4 n1000000 | iree | 574.4 | 23.9 | 9008.1 | n/a |
| micro v2 k4 n1000000 | triton | 288.4 | 0.1 | 68.7 | 968 |
| micro v3 k4 n1000000 | torch-compile | n/a | 575.6 | 172.3 | 2643 |
| micro v3 k4 n1000000 | torch-compile-ro | n/a | 587.5 | 390.9 | n/a |
| micro v3 k4 n1000000 | torch-compile-mat | n/a | 903.4 | 382.8 | n/a |
| micro v3 k4 n1000000 | jax | 208.4 | 0.6 | 308.4 | 2541 |
| micro v3 k4 n1000000 | iree | 576.8 | 23.9 | 81575.5 | n/a |
| micro v3 k4 n1000000 | triton | 289.3 | 0.1 | 122.9 | 1001 |
| micro v4 k4 n1000000 | torch-compile | n/a | 502.1 | 332.2 | 45586 |
| micro v4 k4 n1000000 | torch-compile-ro | n/a | 444.5 | 424.5 | n/a |
| micro v4 k4 n1000000 | torch-compile-mat | n/a | 557.1 | 423.3 | n/a |
| micro v4 k4 n1000000 | jax | 204.7 | 0.6 | 127.5 | 773 |
| micro v4 k4 n1000000 | iree | 579.8 | 24.7 | 10801.0 | n/a |
| micro v4 k4 n1000000 | triton | 322.5 | 0.1 | 69.1 | 1037 |
| micro v5 k4 n1000000 | torch-compile | n/a | 526.4 | 332.6 | 50165 |
| micro v5 k4 n1000000 | torch-compile-ro | n/a | 535.1 | 426.2 | n/a |
| micro v5 k4 n1000000 | torch-compile-mat | n/a | 731.8 | 414.5 | n/a |
| micro v5 k4 n1000000 | jax | 206.8 | 0.6 | 125.6 | 788 |
| micro v5 k4 n1000000 | iree | 576.3 | 24.2 | 10726.4 | n/a |
| micro v5 k4 n1000000 | triton | 320.5 | 0.1 | 69.4 | 1040 |
| micro v6 k1 n25600 | torch-compile | n/a | 1721.9 | 393.6 | n/a |
| micro v6 k1 n25600 | torch-compile-ro | n/a | 1752.0 | 429.4 | n/a |
| micro v6 k1 n25600 | torch-compile-mat | n/a | 1983.6 | 432.7 | n/a |
| micro v6 k1 n25600 | jax | 1574.6 | 0.9 | 107.3 | 4687 |
| micro v6 k1 n25600 | iree | 234.4 | 16.4 | 5347.2 | n/a |
| micro v6 k1 n25600 | triton | 321.1 | 0.9 | 880.0 | n/a |
| micro v6 k1 n256000 | torch-compile | n/a | 1723.4 | 1290.7 | n/a |
| micro v6 k1 n256000 | torch-compile-ro | n/a | 1772.6 | 1293.3 | n/a |
| micro v6 k1 n256000 | torch-compile-mat | n/a | 1909.8 | 1292.9 | n/a |
| micro v6 k1 n256000 | jax | 1605.3 | 1.6 | 751.0 | 2732 |
| micro v6 k1 n256000 | iree | 241.7 | 10.3 | 910.7 | 10 |
| micro v6 k1 n256000 | triton | 313.3 | 0.9 | 850.3 | 165 |
| micro v7 k1 n25600 | torch-compile | n/a | 1835.2 | 848.5 | n/a |
| micro v7 k1 n25600 | torch-compile-ro | n/a | 1710.7 | 885.3 | n/a |
| micro v7 k1 n25600 | torch-compile-mat | n/a | 2581.5 | 897.3 | n/a |
| micro v7 k1 n25600 | jax | 2331.0 | 1.6 | 324.6 | 4357 |
| micro v7 k1 n25600 | iree | 331.9 | 28.4 | 11551.7 | n/a |
| micro v7 k1 n256000 | torch-compile | n/a | 1708.0 | 3000.3 | 24824 |
| micro v7 k1 n256000 | torch-compile-ro | n/a | 1873.9 | 3070.7 | n/a |
| micro v7 k1 n256000 | torch-compile-mat | n/a | 2472.2 | 3083.1 | n/a |
| micro v7 k1 n256000 | jax | 2259.8 | 3.8 | 1954.2 | 1813 |
| micro v7 k1 n256000 | iree | 436.1 | 43.5 | 31555.5 | n/a |
| micro v8 k1 n25600 | torch-compile | n/a | 1806.4 | 515.4 | 41772 |
| micro v8 k1 n25600 | torch-compile-ro | n/a | 1811.6 | 524.5 | 55787 |
| micro v8 k1 n25600 | torch-compile-mat | n/a | 2568.4 | 501.4 | 45295 |
| micro v8 k1 n25600 | jax | 1285.4 | 1.3 | 266.9 | 3535 |
| micro v8 k1 n25600 | iree | 497.6 | 9.7 | 932.9 | n/a |
| micro v8 k1 n256000 | torch-compile | n/a | 1759.1 | 29883.1 | n/a |
| micro v8 k1 n256000 | torch-compile-ro | n/a | 1928.1 | 29726.9 | n/a |
| micro v8 k1 n256000 | torch-compile-mat | n/a | 4914.6 | 27710.5 | n/a |
| micro v8 k1 n256000 | jax | 1555.6 | 16.8 | 15462.8 | 106 |
| micro v8 k1 n256000 | iree | 21278.7 | 98.9 | 38584.8 | n/a |
| micro v9 k1 n25600 | torch-compile | n/a | 1858.1 | 859.0 | n/a |
| micro v9 k1 n25600 | torch-compile-ro | n/a | 1893.4 | 875.8 | n/a |
| micro v9 k1 n25600 | torch-compile-mat | n/a | 2668.1 | 891.9 | n/a |
| micro v9 k1 n25600 | jax | 386.6 | 0.8 | 255.6 | 229 |
| micro v9 k1 n256000 | torch-compile | n/a | 1902.9 | 680.0 | n/a |
| micro v9 k1 n256000 | torch-compile-ro | n/a | 1827.0 | 699.1 | n/a |
| micro v9 k1 n256000 | torch-compile-mat | n/a | 2528.5 | 694.5 | n/a |
| micro v9 k1 n256000 | jax | 838.9 | 1.0 | 329.7 | 2082 |
| mixer_b16 | torch-compile | n/a | 1267.5 | 3164.0 | n/a |
| mixer_b16 | torch-compile-ro | n/a | 1212.8 | 3086.7 | n/a |
| mixer_b16 | torch-compile-mat | n/a | 3692.5 | 3136.6 | n/a |
| mixer_b16 | torch-tensorrt | n/a | 4965.6 | 2498.5 | 9173 |
| mixer_b16 | jax | 7376.4 | 6.9 | 2248.2 | 9330 |
| mixer_b16 | iree | 1562.6 | 754.0 | 674346.1 | n/a |
| mixer_b16 | ours | n/a | 186.8 | 2715.6 | 162 |
| qwen2.5-0.5b | torch-compile | n/a | 3390.7 | 7431.4 | 447 |
| qwen2.5-0.5b | torch-compile-ro | n/a | 3627.7 | 7095.8 | 460 |
| qwen2.5-0.5b | torch-compile-mat | n/a | 3763.0 | 7101.0 | 481 |
| qwen2.5-0.5b | torch-tensorrt | n/a | 19935.7 | 7102.4 | 2908 |
| qwen2.5-0.5b | jax | 12825.7 | 18.3 | 5232.0 | 1440 |
| qwen2.5-0.5b | iree | 1866.3 | 480.2 | 86360.0 | n/a |
| qwen2.5-0.5b | ours | n/a | 1938.0 | 6403.6 | 187 |
| resnet18-b1 | torch-compile | n/a | 1076.3 | 1110.5 | 1669 |
| resnet18-b1 | torch-compile-ro | n/a | 1086.6 | 919.2 | 1243 |
| resnet18-b1 | torch-compile-mat | n/a | 4545.7 | 1035.9 | 7176 |
| resnet18-b1 | torch-tensorrt | n/a | 4164.2 | 997.7 | 6161 |
| resnet18-b1 | jax | 655.3 | 4.0 | 1097.0 | 865 |
| resnet18-b1 | ours | n/a | 150.3 | 728.5 | -40 |
| resnet18-b8 | torch-compile | n/a | 1062.8 | 2186.6 | 11590 |
| resnet18-b8 | torch-compile-ro | n/a | 1083.5 | 2123.5 | 6433 |
| resnet18-b8 | torch-compile-mat | n/a | 5119.3 | 2122.7 | 35514 |
| resnet18-b8 | torch-tensorrt | n/a | 4276.8 | 1952.8 | 13225 |
| resnet18-b8 | jax | 948.8 | 4.4 | 2278.0 | n/a |
| resnet18-b8 | ours | n/a | 167.9 | 2718.6 | n/a |
| smollm2-1.7b | torch-compile | n/a | 3173.0 | 18198.4 | 1855 |
| smollm2-1.7b | torch-compile-ro | n/a | 3143.2 | 17882.2 | 1494 |
| smollm2-1.7b | torch-compile-mat | n/a | 3513.8 | 17903.1 | 1732 |
| smollm2-1.7b | torch-tensorrt | n/a | 21861.5 | 16795.9 | 7613 |
| smollm2-1.7b | jax | 7637.9 | 25.4 | 13528.8 | 1167 |
| smollm2-1.7b | ours | n/a | 5688.3 | 14144.7 | 936 |
| smollm2-135m | torch-compile | n/a | 3955.0 | 5336.9 | 348 |
| smollm2-135m | torch-compile-ro | n/a | 4059.0 | 3809.8 | 310 |
| smollm2-135m | torch-compile-mat | n/a | 4657.0 | 3845.1 | 364 |
| smollm2-135m | torch-tensorrt | n/a | 21912.5 | 3738.6 | 1885 |
| smollm2-135m | jax | 10838.6 | 19.6 | 2662.6 | 830 |
| smollm2-135m | iree | 2063.7 | 174.3 | 19159.5 | n/a |
| smollm2-135m | ours | n/a | 468.5 | 3607.6 | -9 |
| smollm2-360m | torch-compile | n/a | 4193.0 | 6819.5 | 351 |
| smollm2-360m | torch-compile-ro | n/a | 4152.0 | 6278.3 | 330 |
| smollm2-360m | torch-compile-mat | n/a | 4215.9 | 6308.4 | 337 |
| smollm2-360m | torch-tensorrt | n/a | 24684.7 | 6396.0 | 2226 |
| smollm2-360m | jax | 10886.7 | 22.5 | 4584.1 | 820 |
| smollm2-360m | iree | 2166.2 | 432.4 | 32981.2 | n/a |
| smollm2-360m | ours | n/a | 978.7 | 6028.9 | 40 |
| vit-base | torch-compile | n/a | 1783.5 | 4651.4 | n/a |
| vit-base | torch-compile-ro | n/a | 1742.9 | 4573.6 | n/a |
| vit-base | torch-compile-mat | n/a | 2125.3 | 4408.8 | 29072 |
| vit-base | torch-tensorrt | n/a | 7493.8 | 3778.3 | 10527 |
| vit-base | jax | 8353.1 | 9.4 | 3179.7 | 6340 |
| vit-base | iree | 1594.1 | 986.1 | 883380.1 | n/a |
| vit-base | ours | n/a | 257.6 | 3721.1 | 160 |

## Warm-up (median over rounds; steady_at/crossover in iterations, 'none' if never reached within N)

| model | system | cache | first (ms) | steady_at | steady_us | crossover vs eager |
|---|---|---|---|---|---|---|
| distilgpt2 | jax | cold | 4.87 | 27 | 1173.5 | 1 |
| distilgpt2 | jax | warm | 57.62 | 15 | 1165.4 | 1 |
| distilgpt2 | ours | cold | 3968.52 | 127 | 1254.1 | none |
| distilgpt2 | ours | warm | 587.56 | 167 | 1261.5 | 274 |
| distilgpt2 | torch-compile | cold | 3956.03 | 89 | 1395.0 | none |
| distilgpt2 | torch-compile | warm | 1567.98 | 54 | 1451.6 | none |
| distilgpt2 | torch-compile-ro | cold | 3387.48 | 135 | 1303.7 | none |
| distilgpt2 | torch-compile-ro | warm | 1579.03 | 64 | 1333.1 | none |
| distilgpt2 | torch-eager | cold | 136.64 | 2 | 3054.0 | none |
| distilgpt2 | torch-eager | warm | 134.20 | 2 | 3056.2 | none |
| gpt2 | jax | cold | 8.55 | 6 | 1946.1 | 1 |
| gpt2 | jax | warm | 56.09 | 60 | 1922.2 | 1 |
| gpt2 | ours | cold | 3923.57 | 215 | 2073.0 | none |
| gpt2 | ours | warm | 643.80 | 8 | 2075.9 | 161 |
| gpt2 | torch-compile | cold | 5770.11 | 35 | 2256.3 | none |
| gpt2 | torch-compile | warm | 2062.83 | 18 | 2334.5 | none |
| gpt2 | torch-compile-ro | cold | 5074.07 | 41 | 2103.9 | none |
| gpt2 | torch-compile-ro | warm | 2094.62 | 19 | 2129.0 | none |
| gpt2 | torch-eager | cold | 139.20 | 225 | 5341.6 | none |
| gpt2 | torch-eager | warm | 134.20 | 184 | 4962.6 | none |

## Warm-up: kernel compiles (ours, median over rounds; cold_compiles = Triton subprocess compiles over the whole trace, first_forward_compiles = at iter 0)

| model | cache | cold_compiles | first_forward_compiles |
|---|---|---|---|
| distilgpt2 | cold | 9 | 7 |
| distilgpt2 | warm | 0 | 0 |
| gpt2 | cold | 9 | 7 |
| gpt2 | warm | 0 | 0 |

## Dynamic sequence length (median steady_us per length, counters per window)

| system | pass | length | median_us | total_us | loops | bridges | kernels | cache_hits | recompiles |
|---|---|---|---|---|---|---|---|---|---|
| ours | first | 32 | 972.2 | 1153.1 | 2 | 4 | 0 | 833 | 0 |
| ours | first | 48 | 1385.0 | 1720.9 | 5 | 0 | 4 | 720 | 0 |
| ours | first | 64 | 1425.0 | 1819.1 | 1 | 3 | 0 | 597 | 0 |
| ours | first | 96 | 1700.9 | 2278.1 | 1 | 0 | 0 | 597 | 0 |
| ours | first | 128 | 2030.1 | 2795.0 | 0 | 1 | 0 | 632 | 0 |
| ours | new | 160 | 2425.9 | 3734.0 | 0 | 0 | 0 | 630 | 0 |
| ours | new | 192 | 2712.0 | 4017.1 | 0 | 0 | 0 | 585 | 0 |
| ours | new | 224 | 3130.0 | 4579.0 | 0 | 0 | 0 | 585 | 0 |
| ours | new | 256 | 3220.1 | 6067.1 | 0 | 0 | 0 | 585 | 0 |
| ours | new | 288 | 3743.9 | 7150.0 | 0 | 0 | 0 | 585 | 0 |
| ours | revisit | 32 | 942.0 | 1138.0 | 0 | 0 | 0 | 630 | 0 |
| ours | revisit | 48 | 1328.0 | 1612.0 | 0 | 0 | 0 | 585 | 0 |
| ours | revisit | 64 | 1390.0 | 1782.2 | 0 | 0 | 0 | 585 | 0 |
| ours | revisit | 96 | 1637.9 | 2219.9 | 0 | 0 | 0 | 585 | 0 |
| ours | revisit | 128 | 1965.0 | 2758.0 | 0 | 0 | 0 | 585 | 0 |
| torch-compile-dynamic | first | 32 | 1076.8 | 1202.8 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-dynamic | first | 48 | 1494.9 | 1633.4 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-dynamic | first | 64 | 1544.5 | 1682.1 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-dynamic | first | 96 | 1717.3 | 1868.8 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-dynamic | first | 128 | 1920.5 | 2082.1 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-dynamic | new | 160 | 2345.7 | 2521.9 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-dynamic | new | 192 | 2579.0 | 2764.0 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-dynamic | new | 224 | 2996.2 | 3191.5 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-dynamic | new | 256 | 3071.1 | 3278.5 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-dynamic | new | 288 | 3726.0 | 3951.3 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-dynamic | revisit | 32 | 1069.8 | 1196.8 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-dynamic | revisit | 48 | 1481.5 | 1615.0 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-dynamic | revisit | 64 | 1528.0 | 1666.8 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-dynamic | revisit | 96 | 1718.8 | 1870.7 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-dynamic | revisit | 128 | 1919.0 | 2079.5 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-static | first | 32 | 1003.8 | 1131.7 | 0 | 0 | 0 | 0 | 1 |
| torch-compile-static | first | 48 | 1370.0 | 1503.5 | 0 | 0 | 0 | 0 | 1 |
| torch-compile-static | first | 64 | 1407.1 | 1548.5 | 0 | 0 | 0 | 0 | 1 |
| torch-compile-static | first | 96 | 1639.1 | 1791.2 | 0 | 0 | 0 | 0 | 1 |
| torch-compile-static | first | 128 | 1791.7 | 1957.2 | 0 | 0 | 0 | 0 | 1 |
| torch-compile-static | new | 160 | 2214.4 | 2387.8 | 0 | 0 | 0 | 0 | 1 |
| torch-compile-static | new | 192 | 2403.3 | 2590.5 | 0 | 0 | 0 | 0 | 1 |
| torch-compile-static | new | 224 | 3372.0 | 3568.0 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-static | new | 256 | 3527.4 | 3737.2 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-static | new | 288 | 4196.4 | 4417.6 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-static | revisit | 32 | 997.6 | 1123.1 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-static | revisit | 48 | 1346.6 | 1479.1 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-static | revisit | 64 | 1380.0 | 1517.5 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-static | revisit | 96 | 1622.0 | 1776.5 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-static | revisit | 128 | 1781.7 | 1943.3 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | first | 32 | 2153.1 | 2275.4 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | first | 48 | 2362.3 | 2490.6 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | first | 64 | 2364.4 | 2497.0 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | first | 96 | 2429.0 | 2577.0 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | first | 128 | 2491.0 | 2647.9 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | new | 160 | 2703.1 | 2875.9 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | new | 192 | 2897.7 | 3080.3 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | new | 224 | 3309.0 | 3502.2 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | new | 256 | 3492.4 | 3699.5 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | new | 288 | 4159.0 | 4381.1 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | revisit | 32 | 2147.9 | 2269.5 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | revisit | 48 | 2337.0 | 2464.6 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | revisit | 64 | 2350.3 | 2483.1 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | revisit | 96 | 2402.5 | 2548.7 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | revisit | 128 | 2463.8 | 2621.1 | 0 | 0 | 0 | 0 | 0 |

## Where the operation DAG lives (median over rounds)

| model | DAG in | steady_us | launches/iter | kernels | compiles | gc_ms | nodes deferred | same argmax |
|---|---|---|---|---|---|---|---|---|
| bert-base | virtual | 1841.5 | 64.3 | 109 | 0 | 11 | 0 | 1 |
| bert-base | deferred | 2828.9 | 64.0 | 112 | 0 | 75 | 174343 | 1 |
| bert-base | eager | 3827.0 | 759.0 | 103 | 0 | 17 | 0 | 1 |
| bert-mini | virtual | 417.4 | 24.3 | 109 | 0 | 9 | 0 | 1 |
| bert-mini | deferred | 1664.3 | 24.0 | 112 | 0 | 53 | 65783 | 1 |
| bert-mini | eager | 1159.8 | 287.0 | 103 | 0 | 14 | 0 | 1 |
| distilgpt2 | virtual | 1302.3 | 38.2 | 110 | 0 | 12 | 0 | 1 |
| distilgpt2 | deferred | 2220.6 | 44.0 | 113 | 0 | 62 | 62103 | 1 |
| distilgpt2 | eager | 1954.2 | 271.0 | 103 | 0 | 14 | 0 | 1 |
| gpt2 | virtual | 2094.0 | 74.2 | 110 | 0 | 14 | 0 | 1 |
| gpt2 | deferred | 3217.6 | 86.0 | 113 | 0 | 91 | 121443 | 1 |
| gpt2 | eager | 3395.1 | 529.0 | 103 | 0 | 13 | 0 | 1 |
| mixer_b16 | virtual | 2703.6 | 63.4 | 109 | 0 | 17 | 0 | 1 |
| mixer_b16 | deferred | 4458.6 | 75.0 | 122 | 0 | 146 | 218273 | 1 |
| mixer_b16 | eager | 5904.8 | 950.0 | 103 | 0 | 22 | 0 | 1 |
| resnet18-b1 | virtual | 700.1 | 39.0 | 117 | 0 | 1 | 0 | 1 |
| resnet18-b1 | deferred | 732.4 | 39.0 | 123 | 0 | 19 | 15183 | 1 |
| resnet18-b1 | eager | 859.5 | 87.0 | 114 | 0 | 15 | 0 | 1 |
| smollm2-135m | virtual | 3605.9 | 182.1 | 110 | 0 | 13 | 0 | 1 |
| smollm2-135m | deferred | 4131.0 | 205.0 | 117 | 0 | 80 | 215513 | 1 |
| smollm2-135m | eager | 5782.3 | 968.0 | 104 | 0 | 14 | 0 | 1 |
| smollm2-360m | virtual | 6025.0 | 194.1 | 110 | 0 | 18 | 0 | 1 |
| smollm2-360m | deferred | 6402.8 | 218.0 | 117 | 0 | 77 | 229773 | 1 |
| smollm2-360m | eager | 8432.1 | 1032.0 | 104 | 0 | 13 | 0 | 1 |
| vit-base | virtual | 3687.0 | 76.3 | 111 | 0 | 14 | 0 | 1 |
| vit-base | deferred | 4486.2 | 88.0 | 116 | 0 | 102 | 165833 | 1 |
| vit-base | eager | 6353.8 | 723.0 | 104 | 0 | 20 | 0 | 1 |

## Where the DAG lives, micro grid (median over rounds)

| variant | k | n | DAG in | steady_us | launches/iter |
|---|---|---|---|---|---|
| 0 | 1 | 10000 | virtual | 4.6 | 1.01 |
| 0 | 1 | 10000 | deferred | 4.7 | 1.00 |
| 0 | 1 | 10000 | eager | 10.6 | 3.00 |
| 0 | 4 | 1000000 | virtual | 117.4 | 4.01 |
| 0 | 4 | 1000000 | deferred | 41.0 | 1.00 |
| 0 | 4 | 1000000 | eager | 307.5 | 12.01 |
| 8 | 1 | 65536 | virtual | 1713.5 | 11.14 |
| 8 | 1 | 65536 | deferred | 1747.3 | 9.00 |
| 8 | 1 | 65536 | eager | 2305.1 | 38.01 |
| 11 | 1 | 256000 | virtual | 14.8 | 2.04 |
| 11 | 1 | 256000 | deferred | 21.5 | 2.00 |
| 11 | 1 | 256000 | eager | 137.8 | 8.01 |
| 13 | 1 | 65536 | virtual | 591.1 | 5.00 |
| 13 | 1 | 65536 | deferred | 591.6 | 5.00 |
| 13 | 1 | 65536 | eager | 792.2 | 10.01 |

## Host-dependent early exit (median over rounds)

| regime | system | total_ms | p50_us | p95_us | max_us | bridges | frame_compiles | exit layers | pass |
|---|---|---|---|---|---|---|---|---|---|
| stable | jax-perlayer | 11563 | 1720.9 | 1828.4 | 2054 | 0 | 0 | 4 | 1 |
| stable | jax-while | 7314 | 1330.6 | 1379.3 | 1420 | 0 | 0 | 4 | 1 |
| stable | ours | 760 | 1071.2 | 2153.9 | 2244 | 1 | 0 | 4 | 1 |
| stable | torch-compile | 1017 | 1400.2 | 1445.8 | 1470 | 0 | 4 | 4 | 1 |
| stable | torch-compile-mat | 1145 | 1312.0 | 1349.9 | 1375 | 0 | 4 | 4 | 1 |
| stable | torch-compile-ro | 1145 | 1424.3 | 1464.1 | 1489 | 0 | 4 | 4 | 1 |
| stable | torch-eager | 477 | 1896.4 | 1942.4 | 1972 | 0 | 0 | 4 | 1 |
| varying | jax-perlayer | 11171 | 1928.6 | 2389.0 | 2647 | 0 | 0 | 2356 | 1 |
| varying | jax-while | 7546 | 1440.8 | 1667.5 | 1735 | 0 | 0 | 2356 | 1 |
| varying | ours | 788 | 1195.9 | 2013.9 | 2515 | 3 | 0 | 2356 | 1 |
| varying | torch-compile | 1098 | 1691.1 | 1971.5 | 1991 | 0 | 4 | 2356 | 1 |
| varying | torch-compile-mat | 1199 | 1524.2 | 1796.2 | 1856 | 0 | 4 | 2356 | 1 |
| varying | torch-compile-ro | 1106 | 1530.2 | 1799.3 | 1845 | 0 | 4 | 2356 | 1 |
| varying | torch-eager | 480 | 2206.3 | 2644.3 | 2673 | 0 | 0 | 2356 | 1 |

## Next-token latency, greedy decode with a KV cache (median over processes)

| model | system | us/token | p90 us | launches/token | tokens match | teacher-forced |
|---|---|---|---|---|---|---|
| distilgpt2 | torch-eager | 2016 | 2043 | n/a | 10/10 | - |
| distilgpt2 | ours | 744 | 772 | 49.96 | 10/10 | 1 |
| distilgpt2 | torch-compile | 818 | 835 | n/a | 10/10 | - |
| distilgpt2 | torch-compile-ro | 912 | 924 | n/a | 10/10 | - |
| gpt2 | torch-eager | 3705 | 3759 | n/a | 10/10 | - |
| gpt2 | ours | 1175 | 2794 | 92.62 | 10/10 | 1 |
| gpt2 | torch-compile | 1460 | 1481 | n/a | 10/10 | - |
| gpt2 | torch-compile-ro | 1427 | 1455 | n/a | 10/10 | - |
| gpt2-medium | torch-eager | 7125 | 7197 | n/a | 10/10 | - |
| gpt2-medium | ours | 2678 | 2711 | 177.98 | 10/10 | 1 |
| gpt2-medium | torch-compile | 2498 | 2512 | n/a | 10/10 | - |
| gpt2-medium | torch-compile-ro | 2470 | 2487 | n/a | 10/10 | - |
| smollm2-135m | torch-eager | 14890 | 15566 | n/a | 10/10 | - |
| smollm2-135m | ours | 2167 | 2212 | 190.38 | 10/10 | 1 |
| smollm2-135m | torch-compile | 5542 | 5754 | n/a | 10/10 | - |
| smollm2-135m | torch-compile-ro | 2844 | 2897 | n/a | 10/10 | - |
| smollm2-360m | torch-eager | 15489 | 16880 | n/a | 10/10 | - |
| smollm2-360m | ours | 3160 | 3200 | 202.57 | 10/10 | 1 |
| smollm2-360m | torch-compile | 5985 | 6163 | n/a | 10/10 | - |
| smollm2-360m | torch-compile-ro | 3955 | 4057 | n/a | 10/10 | - |
| smollm2-1.7b | torch-eager | 11929 | 13301 | n/a | 10/10 | - |
| smollm2-1.7b | ours | 9242 | 9282 | 153.82 | 10/10 | 1 |
| smollm2-1.7b | torch-compile | 9812 | 9939 | n/a | 10/10 | - |
| smollm2-1.7b | torch-compile-ro | 9517 | 9647 | n/a | 10/10 | - |
| qwen2.5-0.5b | torch-eager | 13246 | 13703 | n/a | 10/10 | - |
| qwen2.5-0.5b | ours | 3635 | 3727 | 153.83 | 10/10 | 1 |
| qwen2.5-0.5b | torch-compile | 5399 | 5520 | n/a | 10/10 | - |
| qwen2.5-0.5b | torch-compile-ro | 4085 | 4152 | n/a | 10/10 | - |

## Carrying a fused value across a guard: descriptor kept (direct) against drained (median over rounds)

| cache | shape | arm | before us | at us | after us | total ms | launches | kernels |
|---|---|---|---|---|---|---|---|---|
| cold | 1024x64 | direct | 19.1 | 73522.1 | 71.0 | 2392.7 | 1215 | 5 |
| cold | 1024x64 | drain | 19.1 | 69072.0 | 61.0 | 2817.2 | 1800 | 6 |
| cold | 256x256 | direct | 19.1 | 65545.1 | 134.9 | 2410.0 | 1215 | 5 |
| cold | 256x256 | drain | 19.6 | 65614.0 | 126.1 | 2827.5 | 1800 | 6 |
| cold | 256x64 | direct | 18.1 | 67613.1 | 69.1 | 2387.2 | 1215 | 5 |
| cold | 256x64 | drain | 21.0 | 69222.0 | 60.1 | 2823.5 | 1800 | 6 |
| warm | 1024x64 | direct | 19.1 | 67349.0 | 72.0 | 94.9 | 1215 | 5 |
| warm | 1024x64 | drain | 20.0 | 67410.0 | 61.0 | 97.6 | 1800 | 6 |
| warm | 256x256 | direct | 19.1 | 70486.1 | 135.9 | 110.9 | 1215 | 5 |
| warm | 256x256 | drain | 20.0 | 64395.0 | 126.1 | 107.6 | 1800 | 6 |
| warm | 256x64 | direct | 18.1 | 67832.9 | 69.1 | 94.7 | 1215 | 5 |
| warm | 256x64 | drain | 19.1 | 67825.1 | 60.1 | 97.5 | 1800 | 6 |

