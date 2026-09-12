# Paper benchmark summary

## Microbenchmarks (median steady_us over rounds)

| variant | k | n | fused (ours) | app-level (ours) | eager (ours) | nojit (ours) | torch.compile | compile-ro | compile-mat | torch eager | JAX/XLA | IREE | Triton | speedup vs compile | interp tax | ours/Triton |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1 | 1000000 | 29.1 | 29.1 | 77.2 | 77.3 | 45.9 | 118.2 | 112.8 | 78.3 | 45.3 | 5751.0 | 30.7 | 1.58x | 1.00x | 0.95x |
| 0 | 4 | 10000 | 13.1 | 19.0 | 39.7 | 39.6 | 60.3 | 118.2 | 117.3 | 58.4 | 43.9 | 447.9 | 19.3 | 4.60x | 1.45x | 0.68x |
| 0 | 4 | 100000 | 13.8 | 21.0 | 43.2 | 43.1 | 50.7 | 115.1 | 100.5 | 61.3 | 48.0 | 929.4 | 19.7 | 3.68x | 1.52x | 0.70x |
| 0 | 4 | 1000000 | 39.7 | 116.6 | 306.3 | 306.6 | 68.4 | 133.8 | 134.3 | 312.5 | 88.3 | 5807.0 | 38.8 | 1.72x | 2.94x | 1.02x |
| 0 | 4 | 10000000 | 329.2 | 1132.7 | 3027.0 | 3027.2 | 633.9 | 1219.9 | 1209.6 | 3045.7 | 815.4 | 50253.5 | 327.1 | 1.93x | 3.44x | 1.01x |
| 0 | 8 | 1000000 | 70.1 | 229.6 | 612.7 | 613.3 | 132.7 | 198.4 | 198.5 | 624.8 | 169.5 | 6457.2 | 70.5 | 1.89x | 3.28x | 0.99x |
| 1 | 4 | 1000000 | 41.6 | 119.4 | 312.2 | 311.1 | 307.5 | 401.8 | 387.3 | 316.7 | 93.0 | 6462.3 | 43.4 | 7.39x | 2.87x | 0.96x |
| 2 | 4 | 1000000 | 42.6 | 119.1 | 311.0 | 310.9 | 72.7 | 138.5 | 138.2 | 326.1 | 170.2 | 9056.9 | 68.6 | 1.70x | 2.79x | 0.62x |
| 3 | 4 | 1000000 | 98.2 | 177.4 | 360.9 | 367.7 | 181.9 | 353.3 | 372.5 | 377.0 | 304.9 | 81750.1 | 125.1 | 1.85x | 1.81x | 0.78x |
| 4 | 4 | 1000000 | 53.5 | 144.2 | 336.1 | 335.9 | 332.5 | 424.4 | 413.8 | 342.2 | 120.6 | 10871.4 | 69.5 | 6.21x | 2.70x | 0.77x |
| 5 | 4 | 1000000 | 53.6 | 145.8 | 335.9 | 335.8 | 332.7 | 420.4 | 425.3 | 342.2 | 120.6 | 10814.3 | 69.3 | 6.21x | 2.72x | 0.77x |
| 6 | 1 | 25600 | 278.8 | 269.1 | 278.3 | 277.8 | 381.9 | 408.7 | 430.4 | 392.4 | 107.5 | 5364.6 | 887.9 | 1.37x | 0.97x | 0.31x |
| 6 | 1 | 256000 | 1016.6 | 1003.1 | 1017.0 | 1016.3 | 1269.3 | 1263.7 | 1300.8 | 1257.1 | 751.7 | 913.7 | 898.5 | 1.25x | 0.99x | 1.13x |
| 7 | 1 | 25600 | 648.8 | 788.4 | 669.8 | 669.1 | 828.5 | 887.0 | 914.5 | 795.5 | 324.8 | 11535.5 | n/a | 1.28x | 1.22x | n/a |
| 7 | 1 | 256000 | 2808.3 | 2935.0 | 2833.9 | 2830.5 | 2997.4 | 3098.8 | 3061.7 | 3083.7 | 1963.5 | 31542.9 | n/a | 1.07x | 1.05x | n/a |
| 8 | 1 | 25600 | 464.1 | 506.0 | 779.1 | 779.6 | 515.0 | 522.4 | 540.0 | 558.5 | 267.2 | 935.7 | n/a | 1.11x | 1.09x | n/a |
| 8 | 1 | 256000 | 22344.6 | 22391.4 | 25096.7 | 25139.8 | 29776.8 | 29790.0 | 27802.5 | 27394.7 | 15400.0 | 38539.3 | n/a | 1.33x | 1.00x | n/a |
| 9 | 1 | 25600 | 755.5 | 749.9 | 762.5 | 764.8 | 876.0 | 861.3 | 871.0 | 803.4 | 244.2 | n/a | n/a | 1.16x | 0.99x | n/a |
| 9 | 1 | 256000 | 572.6 | 557.0 | 602.5 | 570.5 | 645.1 | 700.5 | 666.1 | 586.4 | 332.0 | n/a | n/a | 1.13x | 0.97x | n/a |
| 10 | 1 | 25600 | 3927.7 | 4811.2 | 4827.6 | 4818.9 | 3186.4 | 3095.8 | 3158.9 | 3708.4 | 1504.6 | 10446.6 | n/a | 0.81x | 1.22x | n/a |
| 10 | 1 | 191488 | 93929.0 | 97669.2 | 99355.9 | 99369.6 | 78678.8 | 78707.1 | 76019.3 | 78949.9 | 41818.5 | 136402.5 | n/a | 0.84x | 1.04x | n/a |
| 11 | 1 | 25600 | 3.6 | 20.1 | 56.8 | 57.1 | 66.8 | 111.6 | 132.8 | 56.9 | 48.5 | 54.8 | 18.5 | 18.46x | 5.55x | 0.20x |
| 11 | 1 | 256000 | 15.9 | 163.1 | 144.3 | 141.4 | 169.7 | 222.3 | 246.4 | 141.7 | 47.8 | 185.1 | 20.0 | 10.65x | 10.24x | 0.80x |
| 12 | 1 | 25600 | 98.8 | 90.3 | 98.5 | 98.5 | 147.3 | 165.8 | 177.1 | 136.7 | 43.2 | 1825.1 | 299.0 | 1.49x | 0.91x | 0.33x |
| 12 | 1 | 256000 | 342.9 | 335.8 | 343.2 | 342.9 | 495.9 | 529.0 | 522.7 | 492.3 | 256.5 | 435.1 | 299.9 | 1.45x | 0.98x | 1.14x |
| 13 | 1 | 25600 | 437.3 | 441.6 | 557.3 | 526.5 | 614.2 | 612.1 | 645.1 | 584.3 | 123.6 | 2837.3 | 1252.8 | 1.40x | 1.01x | 0.35x |
| 13 | 1 | 256000 | 4742.4 | 4775.8 | 5423.4 | 5427.7 | 5027.9 | 5020.1 | 5016.9 | 4868.4 | 3699.0 | 5161.6 | 4679.3 | 1.06x | 1.01x | 1.01x |

## Precision sweep (median steady_us)

| dtype | variant | k | n | fused (ours) | app-level (ours) | eager (ours) | nojit (ours) | torch.compile | compile-ro | compile-mat | torch eager | JAX/XLA | IREE | Triton | speedup vs compile | interp tax | ours/Triton |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| float16 | 8 | 1 | 256000 | 707.6 | n/a | 2050.3 | 2053.0 | 865.0 | 869.9 | 801.7 | 1470.4 | 633.8 | 1918.9 | n/a | 1.22x | n/a | n/a |
| float16 | 11 | 1 | 256000 | 22.4 | n/a | 58.4 | 55.6 | 150.4 | 243.6 | 204.6 | 145.8 | 46.4 | 180.3 | 18.7 | 6.71x | n/a | 1.20x |
| float16 | 13 | 1 | 256000 | 211.5 | n/a | 334.0 | 336.9 | 431.8 | 376.4 | 364.1 | 408.8 | 100.5 | 600.4 | 133.1 | 2.04x | n/a | 1.59x |
| float32 | 8 | 1 | 256000 | 1393.1 | n/a | 3914.8 | 3913.6 | 2024.0 | 2025.8 | 1515.4 | 2651.1 | 1259.9 | 7313.6 | n/a | 1.45x | n/a | n/a |
| float32 | 11 | 1 | 256000 | 7.0 | n/a | 43.2 | 39.9 | 154.9 | 220.8 | 206.3 | 146.3 | 45.1 | 80.9 | 19.5 | 22.09x | n/a | 0.36x |
| float32 | 13 | 1 | 256000 | 265.1 | n/a | 581.2 | 583.2 | 549.3 | 585.6 | 536.2 | 633.5 | 215.7 | 1313.3 | 121.5 | 2.07x | n/a | 2.18x |

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
| bert-mini | 432.5 | 871.5 | 493.1 | 414.8 | 1814.6 | 242.8 | 3639.6 | 663.5 | 0.50x | 0.28x | 24.3 | 1.90735e-05 | 0.001 | pass |
| bert-tiny | 245.3 | 538.8 | 302.1 | 261.7 | 1090.2 | 111.3 | 1354.8 | 439.8 | 0.46x | 0.21x | 14.3 | 3.05176e-05 | 0.001 | pass |
| distilgpt2 | 1284.6 | 1343.0 | 1267.3 | 1263.9 | 2797.0 | 999.6 | 17163.6 | 4209.7 | 0.96x | 0.74x | 38.2 | 0.000175476 | 0.001 | pass |
| mixer_b16 | 2714.0 | 3139.4 | 3083.5 | 3152.1 | 3083.6 | 2246.6 | 673442.5 | 2487.4 | 0.86x | 0.72x | 63.4 | 3.8147e-05 | 0.001 | pass |
| resnet18-b1 | 717.7 | 1109.6 | 920.3 | 1055.2 | 1598.0 | 1098.1 | n/a | 1000.4 | 0.65x | 0.99x | 39.0 | 0.00963783 | 0.02 | pass |
| resnet18-b8 | 2727.0 | 2182.5 | 2134.3 | 2119.7 | 2256.3 | 2278.7 | n/a | 2030.7 | 1.25x | 1.04x | 39.0 | 0.00571394 | 0.02 | pass |
| smollm2-135m | 3690.8 | 5344.0 | 3778.2 | 3826.7 | 15793.4 | 2656.4 | 19167.3 | 3761.3 | 0.69x | 0.50x | 212.1 | 0.000151277 | 0.001 | pass |
| tiny-gpt2 | 201.3 | 421.2 | 209.5 | 257.8 | 1616.5 | 86.6 | 194.2 | 1417.8 | 0.48x | 0.21x | 13.2 | 2.6077e-08 | 0.001 | pass |
| vit-tiny | 1015.2 | 2103.9 | 1415.9 | 1340.3 | 3821.5 | 742.3 | 25055.9 | 1035.3 | 0.48x | 0.35x | 76.3 | 1.07288e-05 | 0.001 | pass |

## Batch-size sweep (median steady_us per forward; per_seq = steady/B)

| model | batch | ours | torch.compile | compile-ro | torch eager | JAX/XLA | ratio ours/compile | ratio ours/jax | per_seq ours | per_seq compile | per_seq jax | rows identical | failed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bert-mini | 1 | 430.4 | 796.0 | 488.7 | 1785.4 | 244.9 | 0.54x | 1.76x | 430.4 | 796.0 | 244.9 | yes |  |
| bert-mini | 2 | n/a | 756.5 | 565.8 | 1753.2 | 342.3 | n/a | n/a | n/a | 378.2 | 171.1 | yes | ours |
| bert-mini | 4 | n/a | 795.4 | 742.3 | 1713.2 | 495.2 | n/a | n/a | n/a | 198.9 | 123.8 | yes | ours |
| bert-mini | 8 | n/a | 1065.8 | 1018.1 | 1847.3 | 830.8 | n/a | n/a | n/a | 133.2 | 103.9 | yes | ours |
| bert-mini | 16 | n/a | 1780.3 | 1741.7 | 1851.9 | 1473.4 | n/a | n/a | n/a | 111.3 | 92.1 | yes | ours |
| bert-mini | 32 | n/a | 3299.8 | 3268.1 | 3484.9 | 2801.6 | n/a | n/a | n/a | 103.1 | 87.5 | yes | ours |
| distilgpt2 | 1 | 1285.0 | 1349.0 | 1269.1 | 2769.7 | 1001.5 | 0.95x | 1.28x | 1285.0 | 1349.0 | 1001.5 | yes |  |
| distilgpt2 | 2 | n/a | 1731.9 | 1658.1 | 2744.8 | 1536.0 | n/a | n/a | n/a | 865.9 | 768.0 | yes | ours |
| distilgpt2 | 4 | n/a | 2765.8 | 2723.9 | 3123.5 | 2595.1 | n/a | n/a | n/a | 691.4 | 648.8 | yes | ours |
| distilgpt2 | 8 | n/a | 5061.6 | 5010.5 | 5852.9 | 4770.8 | n/a | n/a | n/a | 632.7 | 596.4 | yes | ours |
| distilgpt2 | 16 | n/a | 9087.1 | 9062.8 | 10702.5 | 8983.4 | n/a | n/a | n/a | 567.9 | 561.5 | yes | ours |
| distilgpt2 | 32 | n/a | 17883.9 | 17849.7 | 20825.1 | 18006.9 | n/a | n/a | n/a | 558.9 | 562.7 | yes | ours |

## Ablations (median steady_us)

| experiment | variant | model | steady_us | note |
|---|---|---|---|---|
| budget_mb | 64 | distilgpt2 | 1272.6 |  |
| budget_mb | 8 | distilgpt2 | 1266.8 |  |
| flat_block | 256 | distilgpt2 | 1241.0 |  |
| flat_block | 256 | resnet18 | 626.1 |  |
| flat_block | 4096 | distilgpt2 | 1274.7 |  |
| flat_block | 4096 | resnet18 | 686.2 |  |
| fusion | off | distilgpt2 | 1924.8 | enable_opts minus tensor |
| fusion | on | distilgpt2 | 1268.3 |  |
| precision | float16 | distilgpt2 | 914.1 |  |
| precision | float16 | smollm2-135m | 2184.6 |  |
| precision | float32 | distilgpt2 | 1267.9 |  |
| precision | float32 | smollm2-135m | 3671.9 |  |
| tf32 | fp32 | resnet18-b1 | 773.7 |  |
| tf32 | fp32 | resnet18-b8 | 3196.9 |  |
| tf32 | tf32 | resnet18-b1 | 713.7 |  |
| tf32 | tf32 | resnet18-b8 | 2717.1 |  |
| tf32 | torch-fp32 | resnet18-b8 | 3047.1 |  |

## Deoptimization cost (median over rounds, us per iteration)

| system | pattern | steady_us | first_fail_us | after_fail_us | peak_us | cold_us | launches/it | loops | bridges |
|---|---|---|---|---|---|---|---|---|---|
| ours | never | 37.9 | n/a | n/a | 64 | 4912 | 2.02 | 0 | 0 |
| torch-compile | never | 186.7 | n/a | n/a | 299 | 539525 | n/a | 0 | n/a |
| torch-compile-ro | never | 253.4 | n/a | n/a | 314 | 584032 | n/a | 0 | n/a |
| torch-eager | never | 112.8 | n/a | n/a | 124 | 35224 | n/a | 0 | n/a |
| ours | alternate | 39.1 | 73 | 59 | 362 | 5614 | 2.17 | 0 | 2 |
| torch-compile | alternate | 191.6 | 148637 | 207 | 148637 | 559130 | n/a | 2 | n/a |
| torch-compile-ro | alternate | 257.6 | 130683 | 372 | 130683 | 553432 | n/a | 2 | n/a |
| torch-eager | alternate | 112.3 | 121 | 112 | 134 | 40196 | n/a | 0 | n/a |
| ours | both-hot | 38.1 | n/a | n/a | 71 | 4442 | 2.02 | 0 | 0 |
| torch-compile | both-hot | 176.0 | n/a | n/a | 443 | 556736 | n/a | 0 | n/a |
| torch-compile-ro | both-hot | 254.5 | n/a | n/a | 314 | 575124 | n/a | 0 | n/a |
| torch-eager | both-hot | 112.5 | n/a | n/a | 122 | 36859 | n/a | 0 | n/a |
| ours | fresh | 75.6 | 65 | 68 | 2095 | 4290 | 7.00 | 0 | 63 |
| torch-compile | fresh | 165.7 | 166 | 166 | 219 | 565164 | n/a | 0 | n/a |
| torch-compile-ro | fresh | 152.3 | 156 | 155 | 178 | 567384 | n/a | 0 | n/a |
| torch-eager | fresh | 112.3 | 117 | 113 | 123 | 36008 | n/a | 0 | n/a |
| ours | probe-a | 35.0 | n/a | 57 | 1701 | 3746 | 2.25 | 5 | 3 |
| ours | probe-e | 17.9 | 1448 | 33 | 1458 | 90 | 1.09 | 3 | 4 |

## Compilation overhead (median; break-even vs torch eager, iterations)

| workload | system | compile_ms | first_run_ms | steady_us | break-even |
|---|---|---|---|---|---|
| bert-mini | torch-compile | n/a | 1282.6 | 871.5 | 1267 |
| bert-mini | torch-compile-ro | n/a | 1294.0 | 493.1 | 913 |
| bert-mini | torch-compile-mat | n/a | 1414.9 | 414.8 | 948 |
| bert-mini | torch-tensorrt | n/a | 4821.3 | 663.5 | 4112 |
| bert-mini | jax | 7169.9 | 2.9 | 242.8 | 4507 |
| bert-mini | iree | 998.6 | 20.5 | 3639.6 | n/a |
| bert-mini | ours | n/a | 188.9 | 432.5 | 73 |
| bert-tiny | torch-compile | n/a | 1118.5 | 538.8 | 1849 |
| bert-tiny | torch-compile-ro | n/a | 1139.6 | 302.1 | 1321 |
| bert-tiny | torch-compile-mat | n/a | 1234.1 | 261.7 | 1370 |
| bert-tiny | torch-tensorrt | n/a | 4237.8 | 439.8 | 6364 |
| bert-tiny | jax | 5771.4 | 1.7 | 111.3 | 5797 |
| bert-tiny | iree | 838.5 | 14.3 | 1354.8 | n/a |
| bert-tiny | ours | n/a | 108.2 | 245.3 | 11 |
| distilgpt2 | torch-compile | n/a | 1536.6 | 1343.0 | 968 |
| distilgpt2 | torch-compile-ro | n/a | 1528.6 | 1267.3 | 915 |
| distilgpt2 | torch-compile-mat | n/a | 1616.1 | 1263.9 | 970 |
| distilgpt2 | torch-tensorrt | n/a | 2966.6 | 4209.7 | n/a |
| distilgpt2 | jax | 7173.0 | 4.7 | 999.6 | 3922 |
| distilgpt2 | iree | 872.3 | 76.4 | 17163.6 | n/a |
| distilgpt2 | ours | n/a | 553.9 | 1284.6 | 281 |
| micro v0 k1 n1000000 | torch-compile | n/a | 480.5 | 45.9 | 13660 |
| micro v0 k1 n1000000 | torch-compile-ro | n/a | 486.4 | 118.2 | n/a |
| micro v0 k1 n1000000 | torch-compile-mat | n/a | 455.1 | 112.8 | n/a |
| micro v0 k1 n1000000 | torch-tensorrt | n/a | 37.4 | 80.7 | n/a |
| micro v0 k1 n1000000 | jax | 196.4 | 0.5 | 45.3 | 4806 |
| micro v0 k1 n1000000 | iree | 563.4 | 23.7 | 5751.0 | n/a |
| micro v0 k1 n1000000 | triton | 311.8 | 0.1 | 30.7 | 5754 |
| micro v0 k4 n10000 | torch-compile | n/a | 434.4 | 60.3 | n/a |
| micro v0 k4 n10000 | torch-compile-ro | n/a | 504.9 | 118.2 | n/a |
| micro v0 k4 n10000 | torch-compile-mat | n/a | 437.4 | 117.3 | n/a |
| micro v0 k4 n10000 | torch-tensorrt | n/a | 40.7 | 55.5 | 500 |
| micro v0 k4 n10000 | jax | 203.5 | 0.5 | 43.9 | 11361 |
| micro v0 k4 n10000 | iree | 575.6 | 10.3 | 447.9 | n/a |
| micro v0 k4 n10000 | triton | 280.6 | 0.1 | 19.3 | 6164 |
| micro v0 k4 n100000 | torch-compile | n/a | 425.9 | 50.7 | 36783 |
| micro v0 k4 n100000 | torch-compile-ro | n/a | 497.9 | 115.1 | n/a |
| micro v0 k4 n100000 | torch-compile-mat | n/a | 486.6 | 100.5 | n/a |
| micro v0 k4 n100000 | torch-tensorrt | n/a | 40.0 | 55.8 | 451 |
| micro v0 k4 n100000 | jax | 194.4 | 0.5 | 48.0 | 11840 |
| micro v0 k4 n100000 | iree | 571.0 | 11.7 | 929.4 | n/a |
| micro v0 k4 n100000 | triton | 304.2 | 0.1 | 19.7 | 6423 |
| micro v0 k4 n1000000 | torch-compile | n/a | 432.6 | 68.4 | 1624 |
| micro v0 k4 n1000000 | torch-compile-ro | n/a | 430.8 | 133.8 | 2210 |
| micro v0 k4 n1000000 | torch-compile-mat | n/a | 488.8 | 134.3 | 2540 |
| micro v0 k4 n1000000 | torch-tensorrt | n/a | 39.1 | 321.8 | n/a |
| micro v0 k4 n1000000 | jax | 208.4 | 0.6 | 88.3 | 771 |
| micro v0 k4 n1000000 | iree | 567.3 | 23.9 | 5807.0 | n/a |
| micro v0 k4 n1000000 | triton | 282.2 | 0.1 | 38.8 | 900 |
| micro v0 k4 n10000000 | torch-compile | n/a | 485.7 | 633.9 | 186 |
| micro v0 k4 n10000000 | torch-compile-ro | n/a | 498.2 | 1219.9 | 252 |
| micro v0 k4 n10000000 | torch-compile-mat | n/a | 506.8 | 1209.6 | 256 |
| micro v0 k4 n10000000 | torch-tensorrt | n/a | 39.5 | 3045.9 | n/a |
| micro v0 k4 n10000000 | jax | 226.6 | 1.3 | 815.4 | 85 |
| micro v0 k4 n10000000 | iree | 615.5 | 170.8 | 50253.5 | n/a |
| micro v0 k4 n10000000 | triton | 285.8 | 0.6 | 327.1 | 92 |
| micro v0 k8 n1000000 | torch-compile | n/a | 429.4 | 132.7 | 799 |
| micro v0 k8 n1000000 | torch-compile-ro | n/a | 506.0 | 198.4 | 1101 |
| micro v0 k8 n1000000 | torch-compile-mat | n/a | 441.4 | 198.5 | 950 |
| micro v0 k8 n1000000 | torch-tensorrt | n/a | 38.7 | 642.3 | n/a |
| micro v0 k8 n1000000 | jax | 218.4 | 0.7 | 169.5 | 401 |
| micro v0 k8 n1000000 | iree | 591.0 | 24.9 | 6457.2 | n/a |
| micro v0 k8 n1000000 | triton | 312.0 | 0.2 | 70.5 | 497 |
| micro v1 k4 n1000000 | torch-compile | n/a | 423.5 | 307.5 | 42078 |
| micro v1 k4 n1000000 | torch-compile-ro | n/a | 437.4 | 401.8 | n/a |
| micro v1 k4 n1000000 | torch-compile-mat | n/a | 447.4 | 387.3 | n/a |
| micro v1 k4 n1000000 | torch-tensorrt | n/a | 39.0 | 326.0 | n/a |
| micro v1 k4 n1000000 | jax | 210.4 | 0.6 | 93.0 | 782 |
| micro v1 k4 n1000000 | iree | 565.3 | 24.1 | 6462.3 | n/a |
| micro v1 k4 n1000000 | triton | 286.4 | 0.1 | 43.4 | 916 |
| micro v10 k1 n191488 | torch-compile | n/a | 1939.6 | 78678.8 | 5754 |
| micro v10 k1 n191488 | torch-compile-ro | n/a | 2049.2 | 78707.1 | 6876 |
| micro v10 k1 n191488 | torch-compile-mat | n/a | 2009.9 | 76019.3 | 556 |
| micro v10 k1 n191488 | torch-tensorrt | n/a | 2544.7 | 79070.6 | n/a |
| micro v10 k1 n191488 | jax | 4928.3 | 69.0 | 41818.5 | 124 |
| micro v10 k1 n191488 | iree | 3810.8 | 205.2 | 136402.5 | n/a |
| micro v10 k1 n25600 | torch-compile | n/a | 1911.1 | 3186.4 | 3063 |
| micro v10 k1 n25600 | torch-compile-ro | n/a | 1928.6 | 3095.8 | 2638 |
| micro v10 k1 n25600 | torch-compile-mat | n/a | 2039.6 | 3158.9 | 3143 |
| micro v10 k1 n25600 | torch-tensorrt | n/a | 2465.9 | 4091.4 | n/a |
| micro v10 k1 n25600 | jax | 4654.7 | 6.7 | 1504.6 | 1973 |
| micro v10 k1 n25600 | iree | 1533.6 | 26.0 | 10446.6 | n/a |
| micro v11 k1 n25600 | torch-compile | n/a | 1692.6 | 66.8 | n/a |
| micro v11 k1 n25600 | torch-compile-ro | n/a | 1771.6 | 111.6 | n/a |
| micro v11 k1 n25600 | torch-compile-mat | n/a | 1682.4 | 132.8 | n/a |
| micro v11 k1 n25600 | torch-tensorrt | n/a | 2079.3 | 107.8 | n/a |
| micro v11 k1 n25600 | jax | 86.3 | 0.5 | 48.5 | -12022 |
| micro v11 k1 n25600 | iree | 187.0 | 8.7 | 54.8 | 3338 |
| micro v11 k1 n25600 | triton | 309.9 | 0.1 | 18.5 | 3160 |
| micro v11 k1 n256000 | torch-compile | n/a | 1644.2 | 169.7 | n/a |
| micro v11 k1 n256000 | torch-compile-ro | n/a | 1717.5 | 222.3 | n/a |
| micro v11 k1 n256000 | torch-compile-mat | n/a | 1775.9 | 246.4 | n/a |
| micro v11 k1 n256000 | torch-tensorrt | n/a | 2123.9 | 196.3 | n/a |
| micro v11 k1 n256000 | jax | 87.0 | 0.5 | 47.8 | -1203 |
| micro v11 k1 n256000 | iree | 180.9 | 10.2 | 185.1 | n/a |
| micro v11 k1 n256000 | triton | 282.1 | 0.1 | 20.0 | 672 |
| micro v12 k1 n25600 | torch-compile | n/a | 1678.7 | 147.3 | n/a |
| micro v12 k1 n25600 | torch-compile-ro | n/a | 1752.2 | 165.8 | n/a |
| micro v12 k1 n25600 | torch-compile-mat | n/a | 1785.8 | 177.1 | n/a |
| micro v12 k1 n25600 | torch-tensorrt | n/a | 2115.7 | 135.2 | 1226660 |
| micro v12 k1 n25600 | jax | 1612.2 | 0.7 | 43.2 | 14939 |
| micro v12 k1 n25600 | iree | 219.5 | 13.7 | 1825.1 | n/a |
| micro v12 k1 n25600 | triton | 310.6 | 0.4 | 299.0 | n/a |
| micro v12 k1 n256000 | torch-compile | n/a | 1798.4 | 495.9 | n/a |
| micro v12 k1 n256000 | torch-compile-ro | n/a | 1757.9 | 529.0 | n/a |
| micro v12 k1 n256000 | torch-compile-mat | n/a | 1800.2 | 522.7 | n/a |
| micro v12 k1 n256000 | torch-tensorrt | n/a | 2151.4 | 493.6 | n/a |
| micro v12 k1 n256000 | jax | 1610.7 | 0.9 | 256.5 | 5850 |
| micro v12 k1 n256000 | iree | 230.7 | 10.2 | 435.1 | 157 |
| micro v12 k1 n256000 | triton | 309.4 | 0.4 | 299.9 | 405 |
| micro v13 k1 n25600 | torch-compile | n/a | 1739.6 | 614.2 | n/a |
| micro v13 k1 n25600 | torch-compile-ro | n/a | 1720.7 | 612.1 | n/a |
| micro v13 k1 n25600 | torch-compile-mat | n/a | 1757.1 | 645.1 | n/a |
| micro v13 k1 n25600 | torch-tensorrt | n/a | 2093.9 | 582.5 | 1039097 |
| micro v13 k1 n25600 | jax | 2050.2 | 1.0 | 123.6 | 3930 |
| micro v13 k1 n25600 | iree | 336.4 | 11.8 | 2837.3 | n/a |
| micro v13 k1 n25600 | triton | 291.3 | 1.4 | 1252.8 | n/a |
| micro v13 k1 n256000 | torch-compile | n/a | 1818.1 | 5027.9 | n/a |
| micro v13 k1 n256000 | torch-compile-ro | n/a | 1749.7 | 5020.1 | n/a |
| micro v13 k1 n256000 | torch-compile-mat | n/a | 1750.4 | 5016.9 | n/a |
| micro v13 k1 n256000 | torch-tensorrt | n/a | 2168.4 | 4865.6 | 669654 |
| micro v13 k1 n256000 | jax | 1969.3 | 4.7 | 3699.0 | 1461 |
| micro v13 k1 n256000 | iree | 647.1 | 19.1 | 5161.6 | n/a |
| micro v13 k1 n256000 | triton | 293.6 | 4.7 | 4679.3 | 175 |
| micro v2 k4 n1000000 | torch-compile | n/a | 498.6 | 72.7 | 1827 |
| micro v2 k4 n1000000 | torch-compile-ro | n/a | 565.4 | 138.5 | 2823 |
| micro v2 k4 n1000000 | torch-compile-mat | n/a | 516.8 | 138.2 | 2560 |
| micro v2 k4 n1000000 | torch-tensorrt | n/a | 492.0 | 326.1 | 5983079 |
| micro v2 k4 n1000000 | jax | 206.2 | 0.6 | 170.2 | 1098 |
| micro v2 k4 n1000000 | iree | 566.4 | 24.2 | 9056.9 | n/a |
| micro v2 k4 n1000000 | triton | 311.0 | 0.1 | 68.6 | 1070 |
| micro v3 k4 n1000000 | torch-compile | n/a | 558.1 | 181.9 | 2667 |
| micro v3 k4 n1000000 | torch-compile-ro | n/a | 526.0 | 353.3 | 20542 |
| micro v3 k4 n1000000 | torch-compile-mat | n/a | 804.2 | 372.5 | 166940 |
| micro v3 k4 n1000000 | torch-tensorrt | n/a | 41.8 | 384.7 | n/a |
| micro v3 k4 n1000000 | jax | 206.2 | 0.6 | 304.9 | 2342 |
| micro v3 k4 n1000000 | iree | 567.7 | 23.7 | 81750.1 | n/a |
| micro v3 k4 n1000000 | triton | 295.6 | 0.1 | 125.1 | 1023 |
| micro v4 k4 n1000000 | torch-compile | n/a | 439.5 | 332.5 | 41187 |
| micro v4 k4 n1000000 | torch-compile-ro | n/a | 500.9 | 424.4 | n/a |
| micro v4 k4 n1000000 | torch-compile-mat | n/a | 505.9 | 413.8 | n/a |
| micro v4 k4 n1000000 | torch-tensorrt | n/a | 38.5 | 352.1 | n/a |
| micro v4 k4 n1000000 | jax | 206.9 | 0.6 | 120.6 | 766 |
| micro v4 k4 n1000000 | iree | 563.6 | 23.8 | 10871.4 | n/a |
| micro v4 k4 n1000000 | triton | 286.7 | 0.1 | 69.5 | 913 |
| micro v5 k4 n1000000 | torch-compile | n/a | 522.1 | 332.7 | 50936 |
| micro v5 k4 n1000000 | torch-compile-ro | n/a | 584.3 | 420.4 | n/a |
| micro v5 k4 n1000000 | torch-compile-mat | n/a | 533.2 | 425.3 | n/a |
| micro v5 k4 n1000000 | torch-tensorrt | n/a | 42.2 | 351.7 | n/a |
| micro v5 k4 n1000000 | jax | 205.7 | 0.6 | 120.6 | 764 |
| micro v5 k4 n1000000 | iree | 566.4 | 24.2 | 10814.3 | n/a |
| micro v5 k4 n1000000 | triton | 310.0 | 0.1 | 69.3 | 1001 |
| micro v6 k1 n25600 | torch-compile | n/a | 1704.3 | 381.9 | 139763 |
| micro v6 k1 n25600 | torch-compile-ro | n/a | 2018.3 | 408.7 | n/a |
| micro v6 k1 n25600 | torch-compile-mat | n/a | 1953.2 | 430.4 | n/a |
| micro v6 k1 n25600 | torch-tensorrt | n/a | 2083.7 | 381.9 | 174945 |
| micro v6 k1 n25600 | jax | 1605.8 | 1.0 | 107.5 | 4822 |
| micro v6 k1 n25600 | iree | 229.4 | 16.9 | 5364.6 | n/a |
| micro v6 k1 n25600 | triton | 304.4 | 0.9 | 887.9 | n/a |
| micro v6 k1 n256000 | torch-compile | n/a | 1779.7 | 1269.3 | n/a |
| micro v6 k1 n256000 | torch-compile-ro | n/a | 1824.3 | 1263.7 | n/a |
| micro v6 k1 n256000 | torch-compile-mat | n/a | 1938.9 | 1300.8 | n/a |
| micro v6 k1 n256000 | torch-tensorrt | n/a | 2080.8 | 1263.5 | n/a |
| micro v6 k1 n256000 | jax | 1602.2 | 1.6 | 751.7 | 2687 |
| micro v6 k1 n256000 | iree | 240.9 | 9.9 | 913.7 | 15 |
| micro v6 k1 n256000 | triton | 309.7 | 0.9 | 898.5 | 181 |
| micro v7 k1 n25600 | torch-compile | n/a | 1727.0 | 828.5 | n/a |
| micro v7 k1 n25600 | torch-compile-ro | n/a | 1730.9 | 887.0 | n/a |
| micro v7 k1 n25600 | torch-compile-mat | n/a | 1770.2 | 914.5 | n/a |
| micro v7 k1 n25600 | torch-tensorrt | n/a | 2130.6 | 826.4 | n/a |
| micro v7 k1 n25600 | jax | 2231.3 | 1.6 | 324.8 | 4211 |
| micro v7 k1 n25600 | iree | 327.6 | 27.9 | 11535.5 | n/a |
| micro v7 k1 n256000 | torch-compile | n/a | 1805.1 | 2997.4 | 17826 |
| micro v7 k1 n256000 | torch-compile-ro | n/a | 1879.6 | 3098.8 | n/a |
| micro v7 k1 n256000 | torch-compile-mat | n/a | 1883.4 | 3061.7 | 73264 |
| micro v7 k1 n256000 | torch-tensorrt | n/a | 2100.4 | 3091.3 | n/a |
| micro v7 k1 n256000 | jax | 2374.2 | 3.8 | 1963.5 | 1885 |
| micro v7 k1 n256000 | iree | 435.7 | 43.0 | 31542.9 | n/a |
| micro v8 k1 n25600 | torch-compile | n/a | 1839.6 | 515.0 | 35873 |
| micro v8 k1 n25600 | torch-compile-ro | n/a | 1840.7 | 522.4 | 43240 |
| micro v8 k1 n25600 | torch-compile-mat | n/a | 1815.0 | 540.0 | 82742 |
| micro v8 k1 n25600 | torch-tensorrt | n/a | 2271.5 | 564.2 | n/a |
| micro v8 k1 n25600 | jax | 1327.1 | 1.3 | 267.2 | 3609 |
| micro v8 k1 n25600 | iree | 498.0 | 9.7 | 935.7 | n/a |
| micro v8 k1 n256000 | torch-compile | n/a | 1817.7 | 29776.8 | n/a |
| micro v8 k1 n256000 | torch-compile-ro | n/a | 1737.9 | 29790.0 | n/a |
| micro v8 k1 n256000 | torch-compile-mat | n/a | 1903.7 | 27802.5 | n/a |
| micro v8 k1 n256000 | torch-tensorrt | n/a | 2249.7 | 27366.5 | 68668 |
| micro v8 k1 n256000 | jax | 1536.2 | 16.6 | 15400.0 | 103 |
| micro v8 k1 n256000 | iree | 21177.4 | 103.2 | 38539.3 | n/a |
| micro v9 k1 n25600 | torch-compile | n/a | 1788.9 | 876.0 | n/a |
| micro v9 k1 n25600 | torch-compile-ro | n/a | 1744.0 | 861.3 | n/a |
| micro v9 k1 n25600 | torch-compile-mat | n/a | 1707.6 | 871.0 | n/a |
| micro v9 k1 n25600 | torch-tensorrt | n/a | 2159.2 | 945.2 | n/a |
| micro v9 k1 n25600 | jax | 370.7 | 0.8 | 244.2 | 211 |
| micro v9 k1 n256000 | torch-compile | n/a | 1831.2 | 645.1 | n/a |
| micro v9 k1 n256000 | torch-compile-ro | n/a | 1886.9 | 700.5 | n/a |
| micro v9 k1 n256000 | torch-compile-mat | n/a | 2164.9 | 666.1 | n/a |
| micro v9 k1 n256000 | torch-tensorrt | n/a | 2223.4 | 1547.5 | n/a |
| micro v9 k1 n256000 | jax | 844.5 | 1.0 | 332.0 | 2209 |
| mixer_b16 | torch-compile | n/a | 1206.1 | 3139.4 | n/a |
| mixer_b16 | torch-compile-ro | n/a | 1244.1 | 3083.5 | 11100000 |
| mixer_b16 | torch-compile-mat | n/a | 1319.2 | 3152.1 | n/a |
| mixer_b16 | torch-tensorrt | n/a | 5864.3 | 2487.4 | 9611 |
| mixer_b16 | jax | 7444.5 | 7.1 | 2246.6 | 8743 |
| mixer_b16 | iree | 1566.5 | 752.0 | 673442.5 | n/a |
| mixer_b16 | ours | n/a | 185.5 | 2714.0 | 139 |
| resnet18-b1 | torch-compile | n/a | 1049.8 | 1109.6 | 1773 |
| resnet18-b1 | torch-compile-ro | n/a | 1081.7 | 920.3 | 1325 |
| resnet18-b1 | torch-compile-mat | n/a | 1609.6 | 1055.2 | 2627 |
| resnet18-b1 | torch-tensorrt | n/a | 4733.6 | 1000.4 | 7614 |
| resnet18-b1 | jax | 611.6 | 4.0 | 1098.1 | 864 |
| resnet18-b1 | ours | n/a | 130.6 | 717.7 | -60 |
| resnet18-b8 | torch-compile | n/a | 1087.4 | 2182.5 | 12140 |
| resnet18-b8 | torch-compile-ro | n/a | 1009.9 | 2134.3 | 6708 |
| resnet18-b8 | torch-compile-mat | n/a | 1516.8 | 2119.7 | 9702 |
| resnet18-b8 | torch-tensorrt | n/a | 4729.2 | 2030.7 | 20114 |
| resnet18-b8 | jax | 942.2 | 4.3 | 2278.7 | n/a |
| resnet18-b8 | ours | n/a | 170.5 | 2727.0 | n/a |
| smollm2-135m | torch-compile | n/a | 3853.7 | 5344.0 | 317 |
| smollm2-135m | torch-compile-ro | n/a | 4009.0 | 3778.2 | 289 |
| smollm2-135m | torch-compile-mat | n/a | 3950.6 | 3826.7 | 285 |
| smollm2-135m | torch-tensorrt | n/a | 22282.2 | 3761.3 | 1807 |
| smollm2-135m | jax | 10854.0 | 18.4 | 2656.4 | 787 |
| smollm2-135m | iree | 2071.4 | 173.6 | 19167.3 | n/a |
| smollm2-135m | ours | n/a | 492.5 | 3690.8 | -4 |
| tiny-gpt2 | torch-compile | n/a | 1195.3 | 421.2 | 596 |
| tiny-gpt2 | torch-compile-ro | n/a | 1193.5 | 209.5 | 505 |
| tiny-gpt2 | torch-compile-mat | n/a | 1221.6 | 257.8 | 543 |
| tiny-gpt2 | torch-tensorrt | n/a | 1877.1 | 1417.8 | 7015 |
| tiny-gpt2 | jax | 1141.0 | 1.6 | 86.6 | 431 |
| tiny-gpt2 | iree | 385.6 | 11.9 | 194.2 | -60 |
| tiny-gpt2 | ours | n/a | 109.0 | 201.3 | -264 |
| vit-tiny | torch-compile | n/a | 1748.4 | 2103.9 | 941 |
| vit-tiny | torch-compile-ro | n/a | 1710.1 | 1415.9 | 656 |
| vit-tiny | torch-compile-mat | n/a | 1971.1 | 1340.3 | 741 |
| vit-tiny | torch-tensorrt | n/a | 7827.5 | 1035.3 | 2762 |
| vit-tiny | jax | 6407.9 | 7.5 | 742.3 | 2041 |
| vit-tiny | iree | 1688.9 | 38.5 | 25055.9 | n/a |
| vit-tiny | ours | n/a | 144.0 | 1015.2 | 4 |

## Warm-up (median over rounds; steady_at/crossover in iterations, 'none' if never reached within N)

| model | system | cache | first (ms) | steady_at | steady_us | crossover vs eager |
|---|---|---|---|---|---|---|
| distilgpt2 | jax | cold | 4.79 | 18 | 1178.1 | 1 |
| distilgpt2 | jax | warm | 52.09 | 19 | 1193.6 | 1 |
| distilgpt2 | ours | cold | 3768.96 | 167 | 1250.0 | none |
| distilgpt2 | ours | warm | 549.69 | 127 | 1254.1 | 247 |
| distilgpt2 | torch-compile | cold | 3995.27 | 31 | 1407.9 | none |
| distilgpt2 | torch-compile | warm | 1561.74 | 57 | 1398.6 | none |
| distilgpt2 | torch-compile-ro | cold | 3360.16 | 119 | 1308.0 | none |
| distilgpt2 | torch-compile-ro | warm | 1566.40 | 76 | 1320.4 | none |
| distilgpt2 | torch-eager | cold | 131.30 | 22 | 3048.3 | none |
| distilgpt2 | torch-eager | warm | 135.92 | 10 | 3062.4 | none |
| tiny-gpt2 | jax | cold | 1.59 | 46 | 160.1 | 1 |
| tiny-gpt2 | jax | warm | 44.71 | 23 | 163.6 | 1 |
| tiny-gpt2 | ours | cold | 552.29 | none | 140.2 | 1 |
| tiny-gpt2 | ours | warm | 107.82 | none | 135.1 | 1 |
| tiny-gpt2 | torch-compile | cold | 3243.03 | 36 | 330.2 | none |
| tiny-gpt2 | torch-compile | warm | 1193.98 | 8 | 357.0 | none |
| tiny-gpt2 | torch-compile-ro | cold | 2385.26 | 132 | 191.3 | none |
| tiny-gpt2 | torch-compile-ro | warm | 1329.80 | 175 | 209.8 | none |
| tiny-gpt2 | torch-eager | cold | 965.91 | 5 | 1579.3 | none |
| tiny-gpt2 | torch-eager | warm | 525.02 | 50 | 1688.8 | none |

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
| ours | 1 | 32 | 971.0 | 1144.5 | 0 | 3 | 0 | 979 | 0 |
| ours | 1 | 48 | 1375.2 | 1655.6 | 5 | 3 | 1 | 1084 | 0 |
| ours | 1 | 64 | 1442.9 | 1820.0 | 3 | 1 | 3 | 1028 | 0 |
| ours | 1 | 96 | 1695.5 | 2287.9 | 1 | 1 | 0 | 912 | 0 |
| ours | 1 | 128 | 2016.9 | 2759.0 | 0 | 1 | 0 | 918 | 0 |
| ours | 2 | 32 | 939.5 | 1145.0 | 0 | 0 | 0 | 900 | 0 |
| ours | 2 | 48 | 1351.6 | 1621.0 | 0 | 0 | 0 | 900 | 0 |
| ours | 2 | 64 | 1387.5 | 1792.1 | 0 | 0 | 0 | 900 | 0 |
| ours | 2 | 96 | 1640.5 | 2346.9 | 0 | 0 | 0 | 900 | 0 |
| ours | 2 | 128 | 1990.1 | 2796.4 | 0 | 0 | 0 | 900 | 0 |
| torch-compile-dynamic | 1 | 32 | 1172.0 | 1299.0 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-dynamic | 1 | 48 | 1610.4 | 1742.2 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-dynamic | 1 | 64 | 1670.8 | 1809.0 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-dynamic | 1 | 96 | 1838.9 | 1988.8 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-dynamic | 1 | 128 | 2097.7 | 2261.2 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-dynamic | 2 | 32 | 1145.3 | 1274.9 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-dynamic | 2 | 48 | 1576.5 | 1710.1 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-dynamic | 2 | 64 | 1619.2 | 1757.2 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-dynamic | 2 | 96 | 1771.9 | 1923.3 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-dynamic | 2 | 128 | 2016.0 | 2179.7 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-static | 1 | 32 | 1115.4 | 1245.9 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-static | 1 | 48 | 1478.1 | 1613.9 | 0 | 0 | 0 | 0 | 1 |
| torch-compile-static | 1 | 64 | 1530.6 | 1668.8 | 0 | 0 | 0 | 0 | 1 |
| torch-compile-static | 1 | 96 | 1743.2 | 1899.3 | 0 | 0 | 0 | 0 | 1 |
| torch-compile-static | 1 | 128 | 1972.8 | 2147.4 | 0 | 0 | 0 | 0 | 1 |
| torch-compile-static | 2 | 32 | 1107.0 | 1235.1 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-static | 2 | 48 | 1457.8 | 1593.3 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-static | 2 | 64 | 1498.6 | 1644.5 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-static | 2 | 96 | 1729.0 | 1877.2 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-static | 2 | 128 | 1949.5 | 2109.4 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | 1 | 32 | 2337.3 | 2460.2 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | 1 | 48 | 2512.9 | 2641.4 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | 1 | 64 | 2516.2 | 2649.1 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | 1 | 96 | 2583.2 | 2730.8 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | 1 | 128 | 2663.6 | 2824.3 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | 2 | 32 | 2329.8 | 2453.6 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | 2 | 48 | 2501.6 | 2630.4 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | 2 | 64 | 2507.4 | 2640.3 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | 2 | 96 | 2557.2 | 2702.4 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | 2 | 128 | 2650.9 | 2811.8 | 0 | 0 | 0 | 0 | 0 |

