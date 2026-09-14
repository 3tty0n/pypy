# Paper benchmark summary

## Microbenchmarks (median steady_us over rounds)

| variant | k | n | fused (ours) | app-level (ours) | eager (ours) | nojit (ours) | torch.compile | torch eager | JAX/XLA | IREE | Triton | speedup vs compile | interp tax | ours/Triton |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1 | 1000000 | 12.1 | n/a | 27.5 | 30.3 | 34.3 | 24.5 | 35.0 | n/a | 19.1 | 2.84x | n/a | 0.63x |
| 0 | 4 | 10000 | 9.2 | n/a | 28.5 | 27.1 | 34.5 | 69.3 | 37.6 | n/a | 18.8 | 3.76x | n/a | 0.49x |
| 0 | 4 | 100000 | 9.2 | n/a | 31.5 | 28.5 | 34.1 | 71.6 | 36.6 | n/a | 19.3 | 3.70x | n/a | 0.48x |
| 0 | 4 | 1000000 | 32.7 | n/a | 109.2 | 120.7 | 56.2 | 97.3 | 73.7 | n/a | 29.5 | 1.72x | n/a | 1.11x |
| 0 | 4 | 10000000 | 324.4 | n/a | 3289.1 | 3289.0 | 548.0 | 3275.0 | 695.4 | n/a | 309.0 | 1.69x | n/a | 1.05x |
| 0 | 8 | 1000000 | 64.2 | n/a | 242.0 | 241.3 | 110.8 | 194.4 | 143.2 | n/a | 58.2 | 1.73x | n/a | 1.10x |
| 1 | 4 | 1000000 | 33.4 | n/a | 122.3 | 121.2 | 97.4 | 98.5 | 75.6 | n/a | 30.8 | 2.92x | n/a | 1.08x |
| 2 | 4 | 1000000 | 34.4 | n/a | 122.4 | 121.2 | 97.3 | 98.5 | 75.2 | n/a | 30.7 | 2.83x | n/a | 1.12x |
| 3 | 4 | 1000000 | 65.4 | n/a | 144.4 | 148.1 | 127.0 | 131.1 | 217.9 | n/a | 84.2 | 1.94x | n/a | 0.78x |
| 4 | 4 | 1000000 | 36.7 | n/a | 133.5 | 130.1 | 104.4 | 105.7 | 84.1 | n/a | 38.2 | 2.84x | n/a | 0.96x |
| 5 | 4 | 1000000 | 37.6 | n/a | 125.1 | 130.3 | 104.6 | 105.7 | 84.4 | n/a | 38.0 | 2.78x | n/a | 0.99x |
| 6 | 1 | 25600 | 206.2 | n/a | 208.3 | 206.2 | 352.5 | 351.0 | 78.7 | n/a | 684.1 | 1.71x | n/a | 0.30x |
| 6 | 1 | 256000 | 759.9 | n/a | 763.1 | 760.1 | 1051.3 | 1045.5 | 623.2 | n/a | 687.4 | 1.38x | n/a | 1.11x |
| 7 | 1 | 25600 | 490.5 | n/a | 500.5 | 498.8 | 718.9 | 643.0 | 227.5 | n/a | n/a | 1.47x | n/a | n/a |
| 7 | 1 | 256000 | 2043.1 | n/a | 2029.5 | 2049.1 | 2402.9 | 2419.1 | 1726.1 | n/a | n/a | 1.18x | n/a | n/a |
| 8 | 1 | 25600 | 366.1 | n/a | 628.8 | 628.8 | 407.5 | 419.6 | 224.5 | n/a | n/a | 1.11x | n/a | n/a |
| 8 | 1 | 256000 | 16212.3 | n/a | 19106.2 | 19104.4 | 21476.8 | 20820.9 | 12251.9 | n/a | n/a | 1.32x | n/a | n/a |
| 9 | 1 | 25600 | 574.5 | n/a | 573.6 | 574.3 | 721.5 | 657.1 | 184.0 | n/a | n/a | 1.26x | n/a | n/a |
| 9 | 1 | 256000 | 414.3 | n/a | 435.1 | 429.7 | 1347.5 | 1277.0 | 256.1 | n/a | n/a | 3.25x | n/a | n/a |
| 10 | 1 | 25600 | 3170.0 | n/a | 3778.6 | 3513.2 | 2460.6 | 3041.6 | 1070.3 | n/a | n/a | 0.78x | n/a | n/a |
| 10 | 1 | 155008 | 49250.4 | n/a | 51679.8 | 51687.7 | 38463.1 | 40353.7 | 21931.1 | n/a | n/a | 0.78x | n/a | n/a |
| 11 | 1 | 25600 | 3.2 | n/a | 15.6 | 15.8 | 53.0 | 33.6 | 37.5 | n/a | 18.4 | 16.79x | n/a | 0.17x |
| 11 | 1 | 256000 | 13.2 | n/a | 95.1 | 95.5 | 195.0 | 171.7 | 31.7 | n/a | 18.4 | 14.75x | n/a | 0.72x |
| 12 | 1 | 25600 | 69.7 | n/a | 69.7 | 68.8 | 130.4 | 127.2 | 33.8 | n/a | 228.7 | 1.87x | n/a | 0.30x |
| 12 | 1 | 256000 | 259.1 | n/a | 259.3 | 255.3 | 456.0 | 447.8 | 211.2 | n/a | 229.8 | 1.76x | n/a | 1.13x |
| 13 | 1 | 25600 | 315.1 | n/a | 405.8 | 405.9 | 527.4 | 521.9 | 102.1 | n/a | 1002.2 | 1.67x | n/a | 0.31x |
| 13 | 1 | 256000 | 3633.8 | n/a | 4149.5 | 4133.9 | 4068.2 | 3942.5 | 2951.1 | n/a | 3504.8 | 1.12x | n/a | 1.04x |

## Precision sweep (median steady_us)

| dtype | variant | k | n | fused (ours) | app-level (ours) | eager (ours) | nojit (ours) | torch.compile | torch eager | JAX/XLA | IREE | Triton | speedup vs compile | interp tax | ours/Triton |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| float16 | 8 | 1 | 256000 | 801.4 | n/a | 2168.2 | 2167.9 | 922.9 | 1477.0 | 686.7 | n/a | n/a | 1.15x | n/a | n/a |
| float16 | 13 | 1 | 256000 | 106.1 | n/a | 338.0 | 330.6 | 464.7 | 453.0 | 54.1 | n/a | 129.8 | 4.38x | n/a | 0.82x |
| float32 | 8 | 1 | 256000 | 1717.5 | n/a | 4344.1 | 4342.8 | 1895.5 | 2581.1 | 1359.5 | n/a | n/a | 1.10x | n/a | n/a |
| float32 | 13 | 1 | 256000 | 230.8 | n/a | 407.0 | 402.7 | 557.6 | 604.6 | 153.0 | n/a | 133.1 | 2.42x | n/a | 1.73x |

## Guards / graph breaks (launches per iter, torch graph breaks)

| variant | n | launches/iter (fused) | torch graphs | torch breaks |
|---|---|---|---|---|
| 1 | 1000000 | 1.0 | 1 | 0 |
| 2 | 1000000 | 1.1 | 1 | 0 |
| 3 | 1000000 | 2.0 | 2 | 1 |
| 4 | 1000000 | 1.0 | 1 | 0 |
| 5 | 1000000 | 1.1 | 2 | 1 |

## End-to-end models (median steady_us, ratio to torch.compile)

| model | ours | torch.compile | torch eager | JAX/XLA | IREE | TensorRT | ratio ours/compile | ratio jax/compile | correctness |
|---|---|---|---|---|---|---|---|---|---|
| bert-mini | 398.0 | 868.6 | 1586.1 | 194.7 | n/a | n/a | 0.46x | 0.22x | 1.81198e-05 |
| bert-tiny | 253.1 | 575.0 | 1066.2 | 91.5 | n/a | n/a | 0.44x | 0.16x | 2.67029e-05 |
| distilgpt2 | 1140.7 | 1252.7 | 2482.4 | 776.4 | n/a | n/a | 0.91x | 0.62x | 0.000205994 |
| smollm2-135m | 3002.4 | 5536.2 | 13433.6 | 2067.8 | n/a | n/a | 0.54x | 0.37x | 0.000150681 |
| tiny-gpt2 | 249.0 | 418.9 | 2305.4 | 81.2 | n/a | n/a | 0.59x | 0.19x | 2.23517e-08 |

## Ablations (median steady_us)

| experiment | variant | model | steady_us | note |
|---|---|---|---|---|
| budget_mb | 64 | distilgpt2 | 1140.4 |  |
| budget_mb | 8 | distilgpt2 | 1140.6 |  |
| flat_block | 256 | distilgpt2 | 1111.9 |  |
| flat_block | 256 | resnet18 | 483.4 |  |
| flat_block | 4096 | distilgpt2 | 1141.0 |  |
| flat_block | 4096 | resnet18 | 536.3 |  |
| fusion | off | distilgpt2 | 1587.9 | enable_opts minus tensor |
| fusion | on | distilgpt2 | 1140.1 |  |
| precision | float16 | distilgpt2 | 1134.8 |  |
| precision | float16 | smollm2-135m | 2251.3 |  |
| precision | float32 | distilgpt2 | 1140.9 |  |
| precision | float32 | smollm2-135m | 3032.2 |  |
| tf32 | fp32 | resnet18-b1 | 684.6 |  |
| tf32 | fp32 | resnet18-b8 | 2930.9 |  |
| tf32 | tf32 | resnet18-b1 | 537.1 |  |
| tf32 | tf32 | resnet18-b8 | 2254.2 |  |
| tf32 | torch-fp32 | resnet18-b8 | 3835.7 |  |

## Compilation overhead (median; break-even vs torch eager, iterations)

| workload | system | compile_ms | first_run_ms | steady_us | break-even |
|---|---|---|---|---|---|
| bert-mini | torch-compile | n/a | 1669.4 | 868.6 | 2131 |
| bert-mini | jax | 5674.5 | 3.2 | 194.7 | 3980 |
| bert-mini | ours | n/a | 3055.1 | 398.0 | 2453 |
| bert-tiny | torch-compile | n/a | 1514.2 | 575.0 | 2769 |
| bert-tiny | jax | 5743.6 | 2.3 | 91.5 | 5737 |
| bert-tiny | ours | n/a | 162.6 | 253.1 | 11 |
| distilgpt2 | torch-compile | n/a | 1883.4 | 1252.7 | 1374 |
| distilgpt2 | jax | 6064.7 | 5.2 | 776.4 | 3444 |
| distilgpt2 | ours | n/a | 647.5 | 1140.7 | 338 |
| micro v0 k1 n1000000 | torch-compile | n/a | 638.5 | 34.3 | n/a |
| micro v0 k1 n1000000 | jax | 275.2 | 1.1 | 35.0 | n/a |
| micro v0 k1 n1000000 | triton | 488.0 | 0.1 | 19.1 | 80543 |
| micro v0 k4 n10000 | torch-compile | n/a | 641.9 | 34.5 | 17018 |
| micro v0 k4 n10000 | jax | 277.9 | 1.6 | 37.6 | 7267 |
| micro v0 k4 n10000 | triton | 492.8 | 0.1 | 18.8 | 8788 |
| micro v0 k4 n100000 | torch-compile | n/a | 642.8 | 34.1 | 15826 |
| micro v0 k4 n100000 | jax | 276.3 | 1.1 | 36.6 | 6533 |
| micro v0 k4 n100000 | triton | 485.6 | 0.1 | 19.3 | 8349 |
| micro v0 k4 n1000000 | torch-compile | n/a | 643.9 | 56.2 | 14485 |
| micro v0 k4 n1000000 | jax | 294.2 | 1.0 | 73.7 | 10495 |
| micro v0 k4 n1000000 | triton | 486.8 | 0.1 | 29.5 | 6471 |
| micro v0 k4 n10000000 | torch-compile | n/a | 649.2 | 548.0 | 219 |
| micro v0 k4 n10000000 | jax | 309.7 | 2.1 | 695.4 | 101 |
| micro v0 k4 n10000000 | triton | 488.4 | 0.5 | 309.0 | 148 |
| micro v0 k8 n1000000 | torch-compile | n/a | 649.7 | 110.8 | 7190 |
| micro v0 k8 n1000000 | jax | 310.0 | 0.8 | 143.2 | 5117 |
| micro v0 k8 n1000000 | triton | 489.4 | 0.2 | 58.2 | 3236 |
| micro v1 k4 n1000000 | torch-compile | n/a | 637.5 | 97.4 | 515388 |
| micro v1 k4 n1000000 | jax | 286.1 | 1.1 | 75.6 | 10407 |
| micro v1 k4 n1000000 | triton | 491.5 | 0.1 | 30.8 | 6543 |
| micro v10 k1 n155008 | torch-compile | n/a | 2857.8 | 38463.1 | 1275 |
| micro v10 k1 n155008 | jax | 6268.9 | 42.4 | 21931.1 | 318 |
| micro v10 k1 n25600 | torch-compile | n/a | 2805.8 | 2460.6 | 4149 |
| micro v10 k1 n25600 | jax | 4990.2 | 7.1 | 1070.3 | 2335 |
| micro v11 k1 n25600 | torch-compile | n/a | 2434.9 | 53.0 | n/a |
| micro v11 k1 n25600 | jax | 71.0 | 0.8 | 37.5 | n/a |
| micro v11 k1 n25600 | triton | 489.4 | 0.1 | 18.4 | 21046 |
| micro v11 k1 n256000 | torch-compile | n/a | 2467.1 | 195.0 | n/a |
| micro v11 k1 n256000 | jax | 81.8 | 0.9 | 31.7 | -835 |
| micro v11 k1 n256000 | triton | 491.3 | 0.1 | 18.4 | 1905 |
| micro v12 k1 n25600 | torch-compile | n/a | 2605.2 | 130.4 | n/a |
| micro v12 k1 n25600 | jax | 1164.0 | 0.9 | 33.8 | 9539 |
| micro v12 k1 n25600 | triton | 489.9 | 0.3 | 228.7 | n/a |
| micro v12 k1 n256000 | torch-compile | n/a | 2604.7 | 456.0 | n/a |
| micro v12 k1 n256000 | jax | 2472.5 | 1.3 | 211.2 | 9188 |
| micro v12 k1 n256000 | triton | 487.1 | 0.4 | 229.8 | 862 |
| micro v13 k1 n25600 | torch-compile | n/a | 2660.3 | 527.4 | n/a |
| micro v13 k1 n25600 | jax | 1747.8 | 1.5 | 102.1 | 3431 |
| micro v13 k1 n25600 | triton | 498.5 | 1.1 | 1002.2 | n/a |
| micro v13 k1 n256000 | torch-compile | n/a | 2675.3 | 4068.2 | n/a |
| micro v13 k1 n256000 | jax | 3102.5 | 4.3 | 2951.1 | 2790 |
| micro v13 k1 n256000 | triton | 494.3 | 3.7 | 3504.8 | 360 |
| micro v2 k4 n1000000 | torch-compile | n/a | 639.7 | 97.3 | 497705 |
| micro v2 k4 n1000000 | jax | 286.2 | 1.0 | 75.2 | 10253 |
| micro v2 k4 n1000000 | triton | 491.8 | 0.1 | 30.7 | 6544 |
| micro v3 k4 n1000000 | torch-compile | n/a | 684.0 | 127.0 | 154522 |
| micro v3 k4 n1000000 | jax | 293.8 | 1.1 | 217.9 | n/a |
| micro v3 k4 n1000000 | triton | 486.4 | 0.1 | 84.2 | 9340 |
| micro v4 k4 n1000000 | torch-compile | n/a | 641.6 | 104.4 | 467440 |
| micro v4 k4 n1000000 | jax | 291.1 | 0.9 | 84.1 | 11315 |
| micro v4 k4 n1000000 | triton | 490.0 | 0.1 | 38.2 | 6546 |
| micro v5 k4 n1000000 | torch-compile | n/a | 701.3 | 104.6 | 606395 |
| micro v5 k4 n1000000 | jax | 286.5 | 0.8 | 84.4 | 11275 |
| micro v5 k4 n1000000 | triton | 489.4 | 0.1 | 38.0 | 6529 |
| micro v6 k1 n25600 | torch-compile | n/a | 2606.2 | 352.5 | n/a |
| micro v6 k1 n25600 | jax | 1197.8 | 1.3 | 78.7 | 3355 |
| micro v6 k1 n25600 | triton | 490.0 | 0.7 | 684.1 | n/a |
| micro v6 k1 n256000 | torch-compile | n/a | 2627.9 | 1051.3 | n/a |
| micro v6 k1 n256000 | jax | 2496.1 | 1.7 | 623.2 | 5170 |
| micro v6 k1 n256000 | triton | 487.4 | 0.7 | 687.4 | 484 |
| micro v7 k1 n25600 | torch-compile | n/a | 2620.9 | 718.9 | n/a |
| micro v7 k1 n25600 | jax | 2690.7 | 1.9 | 227.5 | 5573 |
| micro v7 k1 n256000 | torch-compile | n/a | 2658.2 | 2402.9 | 143467 |
| micro v7 k1 n256000 | jax | 3504.2 | 4.2 | 1726.1 | 4573 |
| micro v8 k1 n25600 | torch-compile | n/a | 2673.2 | 407.5 | 193695 |
| micro v8 k1 n25600 | jax | 2164.6 | 1.8 | 224.5 | 9339 |
| micro v8 k1 n256000 | torch-compile | n/a | 2737.2 | 21476.8 | n/a |
| micro v8 k1 n256000 | jax | 3051.3 | 15.4 | 12251.9 | 310 |
| micro v9 k1 n25600 | torch-compile | n/a | 2657.6 | 721.5 | n/a |
| micro v9 k1 n25600 | jax | 833.0 | 1.1 | 184.0 | 1073 |
| micro v9 k1 n256000 | torch-compile | n/a | 2708.9 | 1347.5 | n/a |
| micro v9 k1 n256000 | jax | 924.5 | 1.3 | 256.1 | 476 |
| smollm2-135m | torch-compile | n/a | 4512.8 | 5536.2 | 474 |
| smollm2-135m | jax | 7940.3 | 20.1 | 2067.8 | 633 |
| smollm2-135m | ours | n/a | 643.1 | 3002.4 | -12 |
| tiny-gpt2 | torch-compile | n/a | 1525.3 | 418.9 | 400 |
| tiny-gpt2 | jax | 921.3 | 2.1 | 81.2 | 68 |
| tiny-gpt2 | ours | n/a | 169.4 | 249.0 | -293 |

## Dynamic sequence length (median steady_us per length)

| system | length | median_us | loops | bridges | recompiles |
|---|---|---|---|---|---|
| ours | 32 | 947.1 | 170 | 167 | 0 |
| ours | 48 | 1161.1 | 170 | 167 | 0 |
| ours | 64 | 1232.6 | 170 | 167 | 0 |
| ours | 96 | 1676.4 | 170 | 167 | 0 |
| ours | 128 | 1742.9 | 170 | 167 | 0 |
| torch-compile-dynamic | 32 | 1352.8 | 0 | 0 | 0 |
| torch-compile-dynamic | 48 | 1578.0 | 0 | 0 | 0 |
| torch-compile-dynamic | 64 | 1651.2 | 0 | 0 | 0 |
| torch-compile-dynamic | 96 | 1867.2 | 0 | 0 | 0 |
| torch-compile-dynamic | 128 | 1966.5 | 0 | 0 | 0 |
| torch-compile-static | 32 | 1264.8 | 0 | 0 | 4 |
| torch-compile-static | 48 | 1473.6 | 0 | 0 | 4 |
| torch-compile-static | 64 | 1547.2 | 0 | 0 | 4 |
| torch-compile-static | 96 | 1759.5 | 0 | 0 | 4 |
| torch-compile-static | 128 | 1846.7 | 0 | 0 | 4 |
| torch-eager | 32 | 2353.3 | 0 | 0 | 0 |
| torch-eager | 48 | 2395.9 | 0 | 0 | 0 |
| torch-eager | 64 | 2409.8 | 0 | 0 | 0 |
| torch-eager | 96 | 2527.2 | 0 | 0 | 0 |
| torch-eager | 128 | 2570.3 | 0 | 0 | 0 |

