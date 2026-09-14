# Paper benchmark summary

## Microbenchmarks (median steady_us over rounds)

| variant | k | n | fused (ours) | app-level (ours) | eager (ours) | nojit (ours) | torch.compile | compile-ro | compile-mat | torch eager | JAX/XLA | IREE | Triton | speedup vs compile | interp tax | ours/Triton |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1 | 1000000 | 12.0 | 12.5 | 28.1 | 30.1 | 35.0 | 92.8 | 93.8 | 24.5 | 37.8 | n/a | 19.3 | 2.91x | 1.04x | 0.62x |
| 0 | 4 | 10000 | 9.3 | 14.3 | 28.4 | 27.1 | 33.8 | 95.1 | 91.4 | 69.8 | 30.6 | n/a | 18.9 | 3.65x | 1.54x | 0.49x |
| 0 | 4 | 100000 | 9.3 | 15.4 | 31.4 | 28.6 | 34.9 | 93.8 | 90.5 | 71.5 | 31.6 | n/a | 19.4 | 3.75x | 1.65x | 0.48x |
| 0 | 4 | 1000000 | 33.2 | 49.7 | 109.8 | 120.8 | 56.2 | 94.4 | 94.2 | 97.3 | 81.0 | n/a | 29.5 | 1.69x | 1.50x | 1.13x |
| 0 | 4 | 10000000 | 339.2 | 1237.5 | 3289.0 | 3289.3 | 548.6 | 1167.8 | 1169.9 | 3277.4 | 848.5 | n/a | 309.5 | 1.62x | 3.65x | 1.10x |
| 0 | 8 | 1000000 | 64.8 | 95.3 | 241.9 | 241.6 | 110.9 | 138.1 | 139.1 | 194.4 | 153.3 | n/a | 58.2 | 1.71x | 1.47x | 1.11x |
| 1 | 4 | 1000000 | 33.8 | 48.8 | 122.4 | 121.2 | 97.4 | 185.1 | 185.5 | 98.5 | 82.9 | n/a | 30.8 | 2.89x | 1.45x | 1.10x |
| 2 | 4 | 1000000 | 34.3 | 48.8 | 122.4 | 121.3 | 57.4 | 106.0 | 105.3 | 107.9 | 135.0 | n/a | 53.0 | 1.67x | 1.42x | 0.65x |
| 3 | 4 | 1000000 | 65.1 | 86.3 | 147.5 | 147.5 | 127.8 | 311.7 | 311.0 | 131.7 | 245.2 | n/a | 85.0 | 1.96x | 1.33x | 0.77x |
| 4 | 4 | 1000000 | 37.2 | 58.5 | 119.1 | 131.0 | 104.4 | 192.2 | 192.3 | 105.7 | 91.5 | n/a | 38.0 | 2.81x | 1.57x | 0.98x |
| 5 | 4 | 1000000 | 37.2 | 60.2 | 127.9 | 130.2 | 104.6 | 193.3 | 193.3 | 105.7 | 91.3 | n/a | 38.0 | 2.82x | 1.62x | 0.98x |
| 6 | 1 | 25600 | 207.2 | 200.1 | 206.3 | 206.4 | 351.3 | 371.3 | 370.6 | 349.0 | 77.5 | n/a | 684.0 | 1.70x | 0.97x | 0.30x |
| 6 | 1 | 256000 | 763.5 | 750.3 | 763.4 | 763.4 | 1050.7 | 1082.2 | 1082.0 | 1044.7 | 627.1 | n/a | 687.4 | 1.38x | 0.98x | 1.11x |
| 7 | 1 | 25600 | 492.1 | 715.6 | 500.3 | 499.2 | 716.3 | 773.3 | 766.0 | 643.4 | 221.2 | n/a | n/a | 1.46x | 1.45x | n/a |
| 7 | 1 | 256000 | 2029.7 | 2267.0 | 2056.9 | 2039.7 | 2407.0 | 2494.5 | 2477.3 | 2415.8 | 1734.1 | n/a | n/a | 1.19x | 1.12x | n/a |
| 8 | 1 | 25600 | 365.8 | 418.7 | 629.2 | 629.5 | 407.2 | 416.4 | 410.7 | 419.6 | 224.3 | n/a | n/a | 1.11x | 1.14x | n/a |
| 8 | 1 | 256000 | 16168.7 | 16324.9 | 19111.4 | 19107.0 | 21462.6 | 21490.1 | 21485.0 | 20804.4 | 12052.2 | n/a | n/a | 1.33x | 1.01x | n/a |
| 9 | 1 | 25600 | 573.9 | 577.3 | 573.7 | 573.0 | 719.5 | 733.6 | 733.8 | 657.6 | 182.0 | n/a | n/a | 1.25x | 1.01x | n/a |
| 9 | 1 | 256000 | 415.2 | 422.5 | 450.3 | 434.0 | 1344.5 | 1383.7 | 1382.2 | 1275.5 | 255.9 | n/a | n/a | 3.24x | 1.02x | n/a |
| 10 | 1 | 25600 | 3151.7 | 4172.4 | 3787.3 | 3545.4 | 2488.7 | 2487.0 | 2463.0 | 3001.3 | 1063.1 | n/a | n/a | 0.79x | 1.32x | n/a |
| 10 | 1 | 155008 | 49301.9 | 50890.4 | 51693.2 | 51689.1 | 38470.0 | 38437.9 | 37587.5 | 40364.6 | 21882.8 | n/a | n/a | 0.78x | 1.03x | n/a |
| 11 | 1 | 25600 | 3.1 | 17.1 | 41.6 | 41.7 | 54.0 | 113.0 | 113.6 | 67.4 | 33.1 | n/a | 18.5 | 17.29x | 5.46x | 0.17x |
| 11 | 1 | 256000 | 13.8 | 146.2 | 109.9 | 113.2 | 193.6 | 261.2 | 263.3 | 208.0 | 35.9 | n/a | 18.7 | 13.99x | 10.57x | 0.74x |
| 12 | 1 | 25600 | 69.8 | 66.8 | 69.4 | 69.2 | 130.3 | 167.7 | 168.1 | 125.9 | 32.1 | n/a | 228.6 | 1.87x | 0.96x | 0.31x |
| 12 | 1 | 256000 | 256.0 | 250.2 | 258.8 | 258.8 | 458.6 | 489.3 | 489.6 | 447.2 | 214.6 | n/a | 229.8 | 1.79x | 0.98x | 1.11x |
| 13 | 1 | 25600 | 312.4 | 320.4 | 406.3 | 406.0 | 527.5 | 551.4 | 546.5 | 522.4 | 103.6 | n/a | 1002.3 | 1.69x | 1.03x | 0.31x |
| 13 | 1 | 256000 | 3639.6 | 3574.4 | 4133.3 | 4143.8 | 4011.7 | 4061.5 | 4046.6 | 3984.7 | 2904.5 | n/a | 3513.3 | 1.10x | 0.98x | 1.04x |

## Precision sweep (median steady_us)

| dtype | variant | k | n | fused (ours) | app-level (ours) | eager (ours) | nojit (ours) | torch.compile | compile-ro | compile-mat | torch eager | JAX/XLA | IREE | Triton | speedup vs compile | interp tax | ours/Triton |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| float16 | 8 | 1 | 256000 | 798.1 | n/a | 2166.4 | 2169.5 | 923.7 | 941.3 | 839.8 | 1478.1 | 690.9 | n/a | n/a | 1.16x | n/a | n/a |
| float16 | 11 | 1 | 256000 | 24.2 | n/a | 49.6 | 46.8 | 193.4 | 257.8 | 261.9 | 205.8 | 33.2 | n/a | 18.6 | 8.00x | n/a | 1.30x |
| float16 | 13 | 1 | 256000 | 100.1 | n/a | 330.2 | 332.4 | 468.6 | 460.7 | 464.3 | 452.0 | 57.9 | n/a | 129.4 | 4.68x | n/a | 0.77x |
| float32 | 8 | 1 | 256000 | 1709.7 | n/a | 4339.8 | 4345.5 | 1880.1 | 1916.8 | 1514.6 | 2596.2 | 1365.8 | n/a | n/a | 1.10x | n/a | n/a |
| float32 | 11 | 1 | 256000 | 6.9 | n/a | 28.7 | 31.9 | 196.3 | 257.8 | 259.6 | 205.9 | 33.6 | n/a | 18.6 | 28.33x | n/a | 0.37x |
| float32 | 13 | 1 | 256000 | 224.1 | n/a | 401.8 | 403.0 | 556.4 | 582.1 | 514.6 | 604.6 | 155.9 | n/a | 131.8 | 2.48x | n/a | 1.70x |

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
| bert-mini | 403.3 | 868.6 | 434.6 | 400.9 | 1563.9 | 191.8 | n/a | 1030.7 | 0.46x | 0.22x | 24.3 | 1.81198e-05 | 0.001 | pass |
| bert-tiny | 270.6 | 576.5 | 292.8 | 292.2 | 1039.1 | 92.9 | n/a | 634.6 | 0.47x | 0.16x | 14.3 | 2.67029e-05 | 0.001 | pass |
| distilgpt2 | 1146.4 | 1253.8 | 1248.7 | 1137.0 | 2529.8 | 775.8 | n/a | 1627.4 | 0.91x | 0.62x | 38.2 | 0.000259399 | 0.001 | pass |
| mixer_b16 | 2232.3 | 3145.6 | 3122.0 | 3094.1 | 3669.3 | 1733.3 | n/a | 1990.1 | 0.71x | 0.55x | 63.4 | 4.29153e-05 | 0.001 | pass |
| resnet18-b1 | 542.4 | 957.3 | 960.5 | 956.3 | 1325.9 | 1655.9 | n/a | 828.8 | 0.57x | 1.73x | 39.0 | 0.00607491 | 0.02 | pass |
| resnet18-b8 | 2258.2 | 1504.7 | 1537.2 | 1531.6 | 1788.5 | 3435.6 | n/a | 1781.2 | 1.50x | 2.28x | 39.0 | 0.00618362 | 0.02 | pass |
| smollm2-135m | 3052.4 | 5548.5 | 3724.8 | 3730.5 | 13549.0 | 2097.5 | n/a | 3351.2 | 0.55x | 0.38x | 212.1 | 0.000130653 | 0.001 | pass |
| tiny-gpt2 | 276.8 | 425.6 | 229.4 | 295.0 | 2281.9 | 81.4 | n/a | 660.5 | 0.65x | 0.19x | 13.2 | 2.23517e-08 | 0.001 | pass |
| vit-tiny | 924.0 | 2230.5 | 1122.8 | 1020.7 | 3131.3 | 513.8 | n/a | 1668.6 | 0.41x | 0.23x | 76.3 | 1.34706e-05 | 0.001 | pass |

## Batch-size sweep (median steady_us per forward; per_seq = steady/B)

| model | batch | ours | torch.compile | compile-ro | torch eager | JAX/XLA | ratio ours/compile | ratio ours/jax | per_seq ours | per_seq compile | per_seq jax | rows identical | failed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bert-mini | 1 | 401.3 | 870.1 | 434.5 | 1552.4 | 190.2 | 0.46x | 2.11x | 401.3 | 870.1 | 190.2 | yes |  |
| bert-mini | 2 | 541.9 | 867.2 | 520.1 | 1525.0 | 269.4 | 0.62x | 2.01x | 271.0 | 433.6 | 134.7 | yes |  |
| bert-mini | 4 | 716.0 | 871.4 | 662.4 | 1579.3 | 411.3 | 0.82x | 1.74x | 179.0 | 217.9 | 102.8 | yes |  |
| bert-mini | 8 | 1027.9 | 883.8 | 877.3 | 1578.0 | 655.0 | 1.16x | 1.57x | 128.5 | 110.5 | 81.9 | yes |  |
| bert-mini | 16 | 1751.2 | 1454.0 | 1459.9 | 1584.1 | 1170.4 | 1.20x | 1.50x | 109.4 | 90.9 | 73.1 | yes |  |
| bert-mini | 32 | 3030.1 | 2315.2 | 2336.5 | 2317.8 | 2142.7 | 1.31x | 1.41x | 94.7 | 72.3 | 67.0 | yes |  |
| distilgpt2 | 1 | 1146.2 | 1253.7 | 1248.7 | 2611.0 | 778.4 | 0.91x | 1.47x | 1146.2 | 1253.7 | 778.4 | yes |  |
| distilgpt2 | 2 | 1608.9 | 1627.2 | 1628.8 | 2498.0 | 1168.0 | 0.99x | 1.38x | 804.5 | 813.6 | 584.0 | yes |  |
| distilgpt2 | 4 | 2671.5 | 2444.8 | 2469.9 | 2871.3 | 2160.5 | 1.09x | 1.24x | 667.9 | 611.2 | 540.1 | yes |  |
| distilgpt2 | 8 | 4446.8 | 4218.9 | 4198.0 | 4697.1 | 3777.9 | 1.05x | 1.18x | 555.8 | 527.4 | 472.2 | yes |  |
| distilgpt2 | 16 | 7531.7 | 7317.9 | 7350.0 | 7988.1 | 6992.7 | 1.03x | 1.08x | 470.7 | 457.4 | 437.0 | yes |  |
| distilgpt2 | 32 | 14196.0 | 13936.5 | 13934.5 | 16156.4 | 13421.3 | 1.02x | 1.06x | 443.6 | 435.5 | 419.4 | yes |  |

## Ablations (median steady_us)

| experiment | variant | model | steady_us | launches/iter | note |
|---|---|---|---|---|---|
| budget_mb | 64 | distilgpt2 | 1146.3 | n/a |  |
| budget_mb | 8 | distilgpt2 | 1147.6 | n/a |  |
| flat_block | 256 | distilgpt2 | 1117.9 | n/a |  |
| flat_block | 256 | resnet18 | 488.2 | n/a |  |
| flat_block | 4096 | distilgpt2 | 1146.6 | n/a |  |
| flat_block | 4096 | resnet18 | 541.8 | n/a |  |
| fusion | off | distilgpt2 | 1611.3 | n/a | enable_opts minus tensor |
| fusion | on | distilgpt2 | 1146.5 | n/a |  |
| max_inputs | mi4 | bert-mini | 419.2 | 34.3 |  |
| max_inputs | mi4 | distilgpt2 | 1166.4 | 45.2 |  |
| max_inputs | mi4 | mixer_b16 | 2274.8 | 76.4 |  |
| max_inputs | mi6 | bert-mini | 400.0 | 24.3 |  |
| max_inputs | mi6 | distilgpt2 | 1146.5 | 38.2 |  |
| max_inputs | mi6 | mixer_b16 | 2246.3 | 63.4 |  |
| max_inputs | mi8 | bert-mini | 400.2 | 24.3 |  |
| max_inputs | mi8 | distilgpt2 | 1146.4 | 38.2 |  |
| max_inputs | mi8 | mixer_b16 | 2249.7 | 63.4 |  |
| precision | float16 | distilgpt2 | 1204.3 | n/a |  |
| precision | float16 | smollm2-135m | 2266.3 | n/a |  |
| precision | float32 | distilgpt2 | 1146.4 | n/a |  |
| precision | float32 | smollm2-135m | 3037.3 | n/a |  |
| tf32 | fp32 | resnet18-b1 | 689.6 | n/a |  |
| tf32 | fp32 | resnet18-b8 | 2932.6 | n/a |  |
| tf32 | tf32 | resnet18-b1 | 541.4 | n/a |  |
| tf32 | tf32 | resnet18-b8 | 2246.4 | n/a |  |
| tf32 | torch-fp32 | resnet18-b8 | 3864.0 | n/a |  |

## Deoptimization cost (median over rounds, us per iteration)

| system | pattern | steady_us | first_fail_us | after_fail_us | peak_us | cold_us | launches/it | loops | bridges |
|---|---|---|---|---|---|---|---|---|---|
| ours | never | 25.0 | n/a | n/a | 66 | 4645 | 2.02 | 0 | 0 |
| torch-compile | never | 119.4 | n/a | n/a | 136 | 749539 | n/a | 0 | n/a |
| torch-compile-ro | never | 199.1 | n/a | n/a | 225 | 757539 | n/a | 0 | n/a |
| torch-eager | never | 91.1 | n/a | n/a | 104 | 51159 | n/a | 0 | n/a |
| ours | alternate | 25.0 | 55 | 41 | 367 | 4695 | 2.17 | 0 | 2 |
| torch-compile | alternate | 121.6 | 80586 | 138 | 80586 | 745926 | n/a | 2 | n/a |
| torch-compile-ro | alternate | 201.5 | 86501 | 462 | 86501 | 755261 | n/a | 2 | n/a |
| torch-eager | alternate | 91.1 | 93 | 91 | 105 | 51086 | n/a | 0 | n/a |
| ours | both-hot | 25.0 | n/a | n/a | 61 | 4675 | 2.02 | 0 | 0 |
| torch-compile | both-hot | 120.6 | n/a | n/a | 136 | 746813 | n/a | 0 | n/a |
| torch-compile-ro | both-hot | 202.2 | n/a | n/a | 234 | 758734 | n/a | 0 | n/a |
| torch-eager | both-hot | 91.3 | n/a | n/a | 101 | 50979 | n/a | 0 | n/a |
| ours | fresh | 78.6 | 64 | 73 | 2524 | 4658 | 7.00 | 0 | 61 |
| torch-compile | fresh | 113.0 | 116 | 115 | 126 | 742859 | n/a | 0 | n/a |
| torch-compile-ro | fresh | 114.4 | 115 | 116 | 126 | 758432 | n/a | 0 | n/a |
| torch-eager | fresh | 90.8 | 94 | 91 | 107 | 51003 | n/a | 0 | n/a |
| ours | probe-a | 24.1 | n/a | 68 | 2798 | 4322 | 2.25 | 5 | 4 |
| ours | probe-e | 11.0 | 1663 | 33 | 2631 | 152 | 1.09 | 3 | 3 |

## Compilation overhead (median; break-even vs torch eager, iterations)

| workload | system | compile_ms | first_run_ms | steady_us | break-even |
|---|---|---|---|---|---|
| bert-mini | torch-compile | n/a | 1671.3 | 868.6 | 2193 |
| bert-mini | torch-compile-ro | n/a | 1678.6 | 434.6 | 1357 |
| bert-mini | torch-compile-mat | n/a | 1857.1 | 400.9 | 1471 |
| bert-mini | torch-tensorrt | n/a | 7265.4 | 1030.7 | 13351 |
| bert-mini | jax | 5747.6 | 3.2 | 191.8 | 4084 |
| bert-mini | ours | n/a | 243.5 | 403.3 | 84 |
| bert-tiny | torch-compile | n/a | 1502.8 | 576.5 | 2907 |
| bert-tiny | torch-compile-ro | n/a | 1508.9 | 292.8 | 1810 |
| bert-tiny | torch-compile-mat | n/a | 1685.2 | 292.2 | 2045 |
| bert-tiny | torch-tensorrt | n/a | 5934.9 | 634.6 | 14281 |
| bert-tiny | jax | 5745.4 | 2.3 | 92.9 | 5907 |
| bert-tiny | ours | n/a | 175.1 | 270.6 | 22 |
| distilgpt2 | torch-compile | n/a | 1878.9 | 1253.8 | 1320 |
| distilgpt2 | torch-compile-ro | n/a | 1882.7 | 1248.7 | 1318 |
| distilgpt2 | torch-compile-mat | n/a | 2035.7 | 1137.0 | 1322 |
| distilgpt2 | torch-tensorrt | n/a | 26402.5 | 1627.4 | 29043 |
| distilgpt2 | jax | 6085.0 | 5.3 | 775.8 | 3362 |
| distilgpt2 | ours | n/a | 646.5 | 1146.4 | 327 |
| micro v0 k1 n1000000 | torch-compile | n/a | 642.4 | 35.0 | n/a |
| micro v0 k1 n1000000 | torch-compile-ro | n/a | 649.0 | 92.8 | n/a |
| micro v0 k1 n1000000 | torch-compile-mat | n/a | 649.1 | 93.8 | n/a |
| micro v0 k1 n1000000 | jax | 279.4 | 1.0 | 37.8 | n/a |
| micro v0 k1 n1000000 | triton | 490.0 | 0.1 | 19.3 | 84363 |
| micro v0 k4 n10000 | torch-compile | n/a | 641.8 | 33.8 | 16451 |
| micro v0 k4 n10000 | torch-compile-ro | n/a | 645.8 | 95.1 | n/a |
| micro v0 k4 n10000 | torch-compile-mat | n/a | 650.3 | 91.4 | n/a |
| micro v0 k4 n10000 | jax | 271.8 | 1.5 | 30.6 | 5705 |
| micro v0 k4 n10000 | triton | 491.7 | 0.1 | 18.9 | 8683 |
| micro v0 k4 n100000 | torch-compile | n/a | 642.2 | 34.9 | 16165 |
| micro v0 k4 n100000 | torch-compile-ro | n/a | 654.2 | 93.8 | n/a |
| micro v0 k4 n100000 | torch-compile-mat | n/a | 663.9 | 90.5 | n/a |
| micro v0 k4 n100000 | jax | 266.6 | 1.2 | 31.6 | 5434 |
| micro v0 k4 n100000 | triton | 487.6 | 0.1 | 19.4 | 8388 |
| micro v0 k4 n1000000 | torch-compile | n/a | 643.9 | 56.2 | 14449 |
| micro v0 k4 n1000000 | torch-compile-ro | n/a | 651.9 | 94.4 | 206112 |
| micro v0 k4 n1000000 | torch-compile-mat | n/a | 661.1 | 94.2 | 198316 |
| micro v0 k4 n1000000 | jax | 291.1 | 1.4 | 81.0 | 14896 |
| micro v0 k4 n1000000 | triton | 487.9 | 0.1 | 29.5 | 6465 |
| micro v0 k4 n10000000 | torch-compile | n/a | 647.1 | 548.6 | 219 |
| micro v0 k4 n10000000 | torch-compile-ro | n/a | 652.8 | 1167.8 | 285 |
| micro v0 k4 n10000000 | torch-compile-mat | n/a | 653.1 | 1169.9 | 286 |
| micro v0 k4 n10000000 | jax | 305.2 | 1.6 | 848.5 | 105 |
| micro v0 k4 n10000000 | triton | 486.9 | 0.6 | 309.5 | 147 |
| micro v0 k8 n1000000 | torch-compile | n/a | 649.2 | 110.9 | 7184 |
| micro v0 k8 n1000000 | torch-compile-ro | n/a | 660.7 | 138.1 | 10870 |
| micro v0 k8 n1000000 | torch-compile-mat | n/a | 668.1 | 139.1 | 11188 |
| micro v0 k8 n1000000 | jax | 301.6 | 1.3 | 153.3 | 6174 |
| micro v0 k8 n1000000 | triton | 493.1 | 0.2 | 58.2 | 3261 |
| micro v1 k4 n1000000 | torch-compile | n/a | 643.3 | 97.4 | 540012 |
| micro v1 k4 n1000000 | torch-compile-ro | n/a | 650.4 | 185.1 | n/a |
| micro v1 k4 n1000000 | torch-compile-mat | n/a | 661.2 | 185.5 | n/a |
| micro v1 k4 n1000000 | jax | 292.0 | 1.6 | 82.9 | 15671 |
| micro v1 k4 n1000000 | triton | 490.4 | 0.1 | 30.8 | 6516 |
| micro v10 k1 n155008 | torch-compile | n/a | 2841.5 | 38470.0 | 1269 |
| micro v10 k1 n155008 | torch-compile-ro | n/a | 2882.0 | 38437.9 | 1269 |
| micro v10 k1 n155008 | torch-compile-mat | n/a | 2992.2 | 37587.5 | 920 |
| micro v10 k1 n155008 | jax | 6309.2 | 41.4 | 21882.8 | 320 |
| micro v10 k1 n25600 | torch-compile | n/a | 2821.0 | 2488.7 | 4725 |
| micro v10 k1 n25600 | torch-compile-ro | n/a | 2887.4 | 2487.0 | 4839 |
| micro v10 k1 n25600 | torch-compile-mat | n/a | 2928.2 | 2463.0 | 4699 |
| micro v10 k1 n25600 | jax | 5185.7 | 7.0 | 1063.1 | 2473 |
| micro v11 k1 n25600 | torch-compile | n/a | 2481.9 | 54.0 | 171254 |
| micro v11 k1 n25600 | torch-compile-ro | n/a | 2532.0 | 113.0 | n/a |
| micro v11 k1 n25600 | torch-compile-mat | n/a | 2539.6 | 113.6 | n/a |
| micro v11 k1 n25600 | jax | 118.8 | 1.1 | 33.1 | -2280 |
| micro v11 k1 n25600 | triton | 489.0 | 0.1 | 18.5 | 5960 |
| micro v11 k1 n256000 | torch-compile | n/a | 2516.0 | 193.6 | 160267 |
| micro v11 k1 n256000 | torch-compile-ro | n/a | 2518.2 | 261.2 | n/a |
| micro v11 k1 n256000 | torch-compile-mat | n/a | 2516.0 | 263.3 | n/a |
| micro v11 k1 n256000 | jax | 147.7 | 0.6 | 35.9 | -372 |
| micro v11 k1 n256000 | triton | 485.0 | 0.1 | 18.7 | 1442 |
| micro v12 k1 n25600 | torch-compile | n/a | 2591.5 | 130.3 | n/a |
| micro v12 k1 n25600 | torch-compile-ro | n/a | 2628.6 | 167.7 | n/a |
| micro v12 k1 n25600 | torch-compile-mat | n/a | 2675.3 | 168.1 | n/a |
| micro v12 k1 n25600 | jax | 1204.3 | 1.3 | 32.1 | 10005 |
| micro v12 k1 n25600 | triton | 492.5 | 0.3 | 228.6 | n/a |
| micro v12 k1 n256000 | torch-compile | n/a | 2616.2 | 458.6 | n/a |
| micro v12 k1 n256000 | torch-compile-ro | n/a | 2618.8 | 489.3 | n/a |
| micro v12 k1 n256000 | torch-compile-mat | n/a | 2629.8 | 489.6 | n/a |
| micro v12 k1 n256000 | jax | 2482.5 | 1.3 | 214.6 | 9362 |
| micro v12 k1 n256000 | triton | 495.7 | 0.4 | 229.8 | 875 |
| micro v13 k1 n25600 | torch-compile | n/a | 2668.3 | 527.5 | n/a |
| micro v13 k1 n25600 | torch-compile-ro | n/a | 2682.5 | 551.4 | n/a |
| micro v13 k1 n25600 | torch-compile-mat | n/a | 2728.6 | 546.5 | n/a |
| micro v13 k1 n25600 | jax | 1781.6 | 1.4 | 103.6 | 3504 |
| micro v13 k1 n25600 | triton | 490.4 | 1.1 | 1002.3 | n/a |
| micro v13 k1 n256000 | torch-compile | n/a | 2693.2 | 4011.7 | n/a |
| micro v13 k1 n256000 | torch-compile-ro | n/a | 2719.9 | 4061.5 | n/a |
| micro v13 k1 n256000 | torch-compile-mat | n/a | 2738.9 | 4046.6 | n/a |
| micro v13 k1 n256000 | jax | 3140.6 | 4.4 | 2904.5 | 2582 |
| micro v13 k1 n256000 | triton | 497.3 | 3.7 | 3513.3 | 308 |
| micro v2 k4 n1000000 | torch-compile | n/a | 720.8 | 57.4 | 13288 |
| micro v2 k4 n1000000 | torch-compile-ro | n/a | 734.8 | 106.0 | 354557 |
| micro v2 k4 n1000000 | torch-compile-mat | n/a | 746.7 | 105.3 | 261186 |
| micro v2 k4 n1000000 | jax | 294.2 | 1.3 | 135.0 | n/a |
| micro v2 k4 n1000000 | triton | 489.1 | 0.1 | 53.0 | 8010 |
| micro v3 k4 n1000000 | torch-compile | n/a | 691.2 | 127.8 | 167230 |
| micro v3 k4 n1000000 | torch-compile-ro | n/a | 698.2 | 311.7 | n/a |
| micro v3 k4 n1000000 | torch-compile-mat | n/a | 806.2 | 311.0 | n/a |
| micro v3 k4 n1000000 | jax | 294.8 | 1.0 | 245.2 | n/a |
| micro v3 k4 n1000000 | triton | 490.4 | 0.1 | 85.0 | 9451 |
| micro v4 k4 n1000000 | torch-compile | n/a | 640.3 | 104.4 | 438457 |
| micro v4 k4 n1000000 | torch-compile-ro | n/a | 652.9 | 192.2 | n/a |
| micro v4 k4 n1000000 | torch-compile-mat | n/a | 654.5 | 192.3 | n/a |
| micro v4 k4 n1000000 | jax | 293.6 | 1.6 | 91.5 | 17343 |
| micro v4 k4 n1000000 | triton | 496.1 | 0.1 | 38.0 | 6600 |
| micro v5 k4 n1000000 | torch-compile | n/a | 695.4 | 104.6 | 613601 |
| micro v5 k4 n1000000 | torch-compile-ro | n/a | 707.7 | 193.3 | n/a |
| micro v5 k4 n1000000 | torch-compile-mat | n/a | 718.6 | 193.3 | n/a |
| micro v5 k4 n1000000 | jax | 290.7 | 1.1 | 91.3 | 16942 |
| micro v5 k4 n1000000 | triton | 490.7 | 0.1 | 38.0 | 6529 |
| micro v6 k1 n25600 | torch-compile | n/a | 2634.7 | 351.3 | n/a |
| micro v6 k1 n25600 | torch-compile-ro | n/a | 2616.6 | 371.3 | n/a |
| micro v6 k1 n25600 | torch-compile-mat | n/a | 3037.8 | 370.6 | n/a |
| micro v6 k1 n25600 | jax | 1205.0 | 1.0 | 77.5 | 3140 |
| micro v6 k1 n25600 | triton | 492.2 | 0.7 | 684.0 | n/a |
| micro v6 k1 n256000 | torch-compile | n/a | 2646.5 | 1050.7 | n/a |
| micro v6 k1 n256000 | torch-compile-ro | n/a | 2645.0 | 1082.2 | n/a |
| micro v6 k1 n256000 | torch-compile-mat | n/a | 2955.1 | 1082.0 | n/a |
| micro v6 k1 n256000 | jax | 2518.8 | 1.6 | 627.1 | 5251 |
| micro v6 k1 n256000 | triton | 487.6 | 0.7 | 687.4 | 450 |
| micro v7 k1 n25600 | torch-compile | n/a | 2643.7 | 716.3 | n/a |
| micro v7 k1 n25600 | torch-compile-ro | n/a | 2713.7 | 773.3 | n/a |
| micro v7 k1 n25600 | torch-compile-mat | n/a | 2669.3 | 766.0 | n/a |
| micro v7 k1 n25600 | jax | 2767.7 | 2.0 | 221.2 | 5829 |
| micro v7 k1 n256000 | torch-compile | n/a | 2671.4 | 2407.0 | 265092 |
| micro v7 k1 n256000 | torch-compile-ro | n/a | 2741.8 | 2494.5 | n/a |
| micro v7 k1 n256000 | torch-compile-mat | n/a | 2712.4 | 2477.3 | n/a |
| micro v7 k1 n256000 | jax | 3457.3 | 4.1 | 1734.1 | 4577 |
| micro v8 k1 n25600 | torch-compile | n/a | 2662.4 | 407.2 | 186033 |
| micro v8 k1 n25600 | torch-compile-ro | n/a | 2739.3 | 416.4 | 765006 |
| micro v8 k1 n25600 | torch-compile-mat | n/a | 2763.2 | 410.7 | 271612 |
| micro v8 k1 n25600 | jax | 2139.9 | 1.6 | 224.3 | 9157 |
| micro v8 k1 n256000 | torch-compile | n/a | 2710.9 | 21462.6 | n/a |
| micro v8 k1 n256000 | torch-compile-ro | n/a | 2775.0 | 21490.1 | n/a |
| micro v8 k1 n256000 | torch-compile-mat | n/a | 2756.1 | 21485.0 | n/a |
| micro v8 k1 n256000 | jax | 3088.9 | 15.5 | 12052.2 | 310 |
| micro v9 k1 n25600 | torch-compile | n/a | 2667.4 | 719.5 | n/a |
| micro v9 k1 n25600 | torch-compile-ro | n/a | 2751.5 | 733.6 | n/a |
| micro v9 k1 n25600 | torch-compile-mat | n/a | 2763.9 | 733.8 | n/a |
| micro v9 k1 n25600 | jax | 842.6 | 1.7 | 182.0 | 1074 |
| micro v9 k1 n256000 | torch-compile | n/a | 2709.1 | 1344.5 | n/a |
| micro v9 k1 n256000 | torch-compile-ro | n/a | 2808.9 | 1383.7 | n/a |
| micro v9 k1 n256000 | torch-compile-mat | n/a | 3063.0 | 1382.2 | n/a |
| micro v9 k1 n256000 | jax | 921.3 | 1.3 | 255.9 | 516 |
| mixer_b16 | torch-compile | n/a | 1790.5 | 3145.6 | 2953 |
| mixer_b16 | torch-compile-ro | n/a | 1791.2 | 3122.0 | 2827 |
| mixer_b16 | torch-compile-mat | n/a | 2055.3 | 3094.1 | 3149 |
| mixer_b16 | torch-tensorrt | n/a | 8700.4 | 1990.1 | 5036 |
| mixer_b16 | jax | 7008.9 | 8.5 | 1733.3 | 3499 |
| mixer_b16 | ours | n/a | 261.1 | 2232.3 | 12 |
| resnet18-b1 | torch-compile | n/a | 1561.8 | 957.3 | 3562 |
| resnet18-b1 | torch-compile-ro | n/a | 1572.2 | 960.5 | 3622 |
| resnet18-b1 | torch-compile-mat | n/a | 2244.7 | 956.3 | 5400 |
| resnet18-b1 | torch-tensorrt | n/a | 5683.5 | 828.8 | 10933 |
| resnet18-b1 | jax | 630.4 | 4.4 | 1655.9 | n/a |
| resnet18-b1 | ours | n/a | 182.6 | 542.4 | -85 |
| resnet18-b8 | torch-compile | n/a | 1532.7 | 1504.7 | 4572 |
| resnet18-b8 | torch-compile-ro | n/a | 1558.3 | 1537.2 | 5265 |
| resnet18-b8 | torch-compile-mat | n/a | 2821.2 | 1531.6 | 10066 |
| resnet18-b8 | torch-tensorrt | n/a | 5726.0 | 1781.2 | 752164 |
| resnet18-b8 | jax | 1214.7 | 6.2 | 3435.6 | n/a |
| resnet18-b8 | ours | n/a | 195.6 | 2258.2 | n/a |
| smollm2-135m | torch-compile | n/a | 4525.0 | 5548.5 | 469 |
| smollm2-135m | torch-compile-ro | n/a | 4532.4 | 3724.8 | 382 |
| smollm2-135m | torch-compile-mat | n/a | 4708.5 | 3730.5 | 400 |
| smollm2-135m | torch-tensorrt | n/a | 33821.3 | 3351.2 | 3240 |
| smollm2-135m | jax | 7828.7 | 19.3 | 2097.5 | 617 |
| smollm2-135m | ours | n/a | 636.1 | 3052.4 | -13 |
| tiny-gpt2 | torch-compile | n/a | 1506.6 | 425.6 | 396 |
| tiny-gpt2 | torch-compile-ro | n/a | 1520.7 | 229.4 | 365 |
| tiny-gpt2 | torch-compile-mat | n/a | 1536.3 | 295.0 | 385 |
| tiny-gpt2 | torch-tensorrt | n/a | 11623.6 | 660.5 | 6693 |
| tiny-gpt2 | jax | 905.9 | 2.1 | 81.4 | 62 |
| tiny-gpt2 | ours | n/a | 168.8 | 276.8 | -300 |
| vit-tiny | torch-compile | n/a | 2292.6 | 2230.5 | 2277 |
| vit-tiny | torch-compile-ro | n/a | 2306.7 | 1122.8 | 1028 |
| vit-tiny | torch-compile-mat | n/a | 2585.8 | 1020.7 | 1111 |
| vit-tiny | torch-tensorrt | n/a | 12606.6 | 1668.6 | 8454 |
| vit-tiny | jax | 6514.6 | 7.9 | 513.8 | 2400 |
| vit-tiny | ours | n/a | 193.1 | 924.0 | -22 |

## Warm-up (median over rounds; steady_at/crossover in iterations, 'none' if never reached within N)

| model | system | cache | first (ms) | steady_at | steady_us | crossover vs eager |
|---|---|---|---|---|---|---|
| distilgpt2 | jax | cold | 5.01 | 3 | 975.3 | 1 |
| distilgpt2 | jax | warm | 82.49 | 3 | 986.8 | 1 |
| distilgpt2 | ours | cold | 5486.41 | 131 | 1096.0 | none |
| distilgpt2 | ours | warm | 653.26 | 131 | 1099.5 | none |
| distilgpt2 | torch-compile | cold | 5710.96 | 3 | 1338.5 | none |
| distilgpt2 | torch-compile | warm | 1869.48 | 3 | 1328.5 | none |
| distilgpt2 | torch-compile-ro | cold | 4642.88 | 4 | 1276.5 | none |
| distilgpt2 | torch-compile-ro | warm | 1880.15 | 4 | 1277.2 | none |
| distilgpt2 | torch-eager | cold | 195.15 | 4 | 2695.2 | none |
| distilgpt2 | torch-eager | warm | 198.22 | 2 | 2718.2 | none |
| tiny-gpt2 | jax | cold | 2.13 | 64 | 132.6 | 1 |
| tiny-gpt2 | jax | warm | 82.53 | 88 | 134.5 | 1 |
| tiny-gpt2 | ours | cold | 842.05 | none | 165.5 | 1 |
| tiny-gpt2 | ours | warm | 168.98 | none | 165.9 | 1 |
| tiny-gpt2 | torch-compile | cold | 4124.47 | 13 | 358.2 | none |
| tiny-gpt2 | torch-compile | warm | 1511.16 | 29 | 357.4 | none |
| tiny-gpt2 | torch-compile-ro | cold | 3428.31 | 34 | 195.4 | none |
| tiny-gpt2 | torch-compile-ro | warm | 1518.40 | 48 | 198.0 | none |
| tiny-gpt2 | torch-eager | cold | 1379.76 | 3 | 1779.7 | none |
| tiny-gpt2 | torch-eager | warm | 774.17 | 3 | 1765.7 | none |

## Warm-up: kernel compiles (ours, median over rounds; cold_compiles = Triton subprocess compiles over the whole trace, first_forward_compiles = at iter 0)

| model | cache | cold_compiles | first_forward_compiles |
|---|---|---|---|
| distilgpt2 | cold | 9 | 7 |
| distilgpt2 | warm | 0 | 0 |
| tiny-gpt2 | cold | 10 | 1 |
| tiny-gpt2 | warm | 0 | 0 |

## Dynamic sequence length (median steady_us per length, counters per window)

| system | pass | length | median_us | total_us | loops | bridges | kernels | cache_hits | recompiles |
|---|---|---|---|---|---|---|---|---|---|
| ours | 1 | 32 | 950.1 | 1245.5 | 0 | 3 | 0 | 979 | 0 |
| ours | 1 | 48 | 1163.0 | 1608.1 | 5 | 3 | 1 | 1084 | 0 |
| ours | 1 | 64 | 1236.0 | 1824.0 | 2 | 2 | 3 | 1028 | 0 |
| ours | 1 | 96 | 1680.1 | 2581.9 | 1 | 1 | 0 | 912 | 0 |
| ours | 1 | 128 | 1746.0 | 2947.9 | 0 | 0 | 0 | 918 | 0 |
| ours | 2 | 32 | 948.7 | 1239.0 | 0 | 0 | 0 | 900 | 0 |
| ours | 2 | 48 | 1160.4 | 1600.0 | 0 | 0 | 0 | 900 | 0 |
| ours | 2 | 64 | 1234.5 | 1812.0 | 0 | 0 | 0 | 900 | 0 |
| ours | 2 | 96 | 1672.5 | 2567.5 | 0 | 0 | 0 | 900 | 0 |
| ours | 2 | 128 | 1741.5 | 2940.5 | 0 | 0 | 0 | 900 | 0 |
| torch-compile-dynamic | 1 | 32 | 1090.0 | 1238.8 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-dynamic | 1 | 48 | 1289.2 | 1445.5 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-dynamic | 1 | 64 | 1335.2 | 1491.9 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-dynamic | 1 | 96 | 1643.1 | 1822.5 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-dynamic | 1 | 128 | 1751.2 | 1942.1 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-dynamic | 2 | 32 | 1083.9 | 1231.9 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-dynamic | 2 | 48 | 1280.7 | 1435.3 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-dynamic | 2 | 64 | 1331.4 | 1487.7 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-dynamic | 2 | 96 | 1639.2 | 1814.9 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-dynamic | 2 | 128 | 1744.2 | 1932.2 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-static | 1 | 32 | 1015.5 | 1164.1 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-static | 1 | 48 | 1197.9 | 1349.4 | 0 | 0 | 0 | 0 | 1 |
| torch-compile-static | 1 | 64 | 1247.7 | 1406.1 | 0 | 0 | 0 | 0 | 1 |
| torch-compile-static | 1 | 96 | 1550.9 | 1724.9 | 0 | 0 | 0 | 0 | 1 |
| torch-compile-static | 1 | 128 | 1649.3 | 1835.3 | 0 | 0 | 0 | 0 | 1 |
| torch-compile-static | 2 | 32 | 1008.4 | 1155.2 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-static | 2 | 48 | 1191.6 | 1344.2 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-static | 2 | 64 | 1246.2 | 1406.5 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-static | 2 | 96 | 1552.3 | 1720.7 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-static | 2 | 128 | 1639.8 | 1827.4 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | 1 | 32 | 2105.5 | 2251.4 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | 1 | 48 | 2156.4 | 2310.8 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | 1 | 64 | 2165.3 | 2322.6 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | 1 | 96 | 2279.1 | 2451.5 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | 1 | 128 | 2304.4 | 2493.1 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | 2 | 32 | 2103.7 | 2249.2 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | 2 | 48 | 2154.6 | 2304.0 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | 2 | 64 | 2161.2 | 2318.6 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | 2 | 96 | 2271.7 | 2443.2 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | 2 | 128 | 2297.6 | 2482.7 | 0 | 0 | 0 | 0 | 0 |

