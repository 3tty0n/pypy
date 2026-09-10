# Paper benchmark summary

## Microbenchmarks (median steady_us over rounds)

| variant | k | n | fused (ours) | app-level (ours) | eager (ours) | nojit (ours) | torch.compile | torch eager | JAX/XLA | IREE | Triton | speedup vs compile | interp tax | ours/Triton |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1 | 1000000 | 29.2 | 29.9 | 76.7 | 76.8 | 44.5 | 78.3 | 43.5 | 5785.1 | 30.7 | 1.52x | 1.02x | 0.95x |
| 0 | 4 | 10000 | 11.9 | 18.4 | 39.1 | 39.4 | 58.8 | 57.8 | 43.4 | 455.0 | 18.7 | 4.94x | 1.55x | 0.64x |
| 0 | 4 | 100000 | 12.8 | 19.9 | 44.2 | 43.1 | 42.7 | 54.6 | 45.8 | 918.2 | 19.4 | 3.33x | 1.55x | 0.66x |
| 0 | 4 | 1000000 | 39.2 | 115.8 | 307.5 | 306.6 | 68.4 | 312.3 | 88.2 | 5844.4 | 39.1 | 1.74x | 2.95x | 1.00x |
| 0 | 4 | 10000000 | 315.3 | 1133.6 | 3027.0 | 3027.3 | 637.0 | 3045.6 | 770.6 | 50176.4 | 327.0 | 2.02x | 3.60x | 0.96x |
| 0 | 8 | 1000000 | 67.4 | 231.1 | 612.9 | 613.1 | 132.7 | 624.6 | 160.3 | 5888.7 | 67.0 | 1.97x | 3.43x | 1.01x |
| 1 | 4 | 1000000 | 41.7 | 119.3 | 311.8 | 311.8 | 307.5 | 316.7 | 92.9 | n/a | 43.4 | 7.38x | 2.86x | 0.96x |
| 2 | 4 | 1000000 | 44.1 | 119.2 | 311.8 | 311.6 | 307.5 | 316.7 | 92.9 | n/a | 43.4 | 6.97x | 2.70x | 1.02x |
| 3 | 4 | 1000000 | 103.7 | 198.2 | 357.7 | 394.9 | 164.8 | 376.8 | 280.9 | n/a | 122.8 | 1.59x | 1.91x | 0.84x |
| 4 | 4 | 1000000 | 52.4 | 144.4 | 340.0 | 337.8 | 332.4 | 342.1 | 115.2 | n/a | 67.9 | 6.34x | 2.75x | 0.77x |
| 5 | 4 | 1000000 | 54.0 | 149.3 | 337.1 | 336.4 | 332.7 | 342.2 | 120.3 | n/a | 69.2 | 6.17x | 2.77x | 0.78x |
| 6 | 1 | 25600 | 277.3 | 270.0 | 279.9 | 277.4 | 399.1 | 393.7 | 107.6 | 5392.1 | 875.4 | 1.44x | 0.97x | 0.32x |
| 6 | 1 | 256000 | 1012.2 | 1003.2 | 1016.3 | 1012.2 | 1243.9 | 1254.3 | 758.3 | 911.7 | 883.7 | 1.23x | 0.99x | 1.15x |
| 7 | 1 | 25600 | 646.7 | 791.4 | 668.6 | 668.4 | 859.8 | 792.9 | 324.7 | n/a | n/a | 1.33x | 1.22x | n/a |
| 7 | 1 | 256000 | 2820.3 | 2938.7 | 2817.8 | 2836.5 | 3043.7 | 3093.0 | 1975.4 | n/a | n/a | 1.08x | 1.04x | n/a |
| 8 | 1 | 25600 | 464.3 | 480.9 | 779.9 | 779.8 | 515.1 | 557.9 | 267.2 | n/a | n/a | 1.11x | 1.04x | n/a |
| 8 | 1 | 256000 | 22322.1 | 22375.2 | 25088.5 | 25116.0 | 29781.0 | 27391.0 | 15402.6 | n/a | n/a | 1.33x | 1.00x | n/a |
| 9 | 1 | 25600 | 758.0 | 752.0 | 765.9 | 766.2 | 860.5 | 799.4 | 251.3 | n/a | n/a | 1.14x | 0.99x | n/a |
| 9 | 1 | 256000 | 573.3 | 572.6 | 579.9 | 569.7 | 654.6 | 582.4 | 349.8 | n/a | n/a | 1.14x | 1.00x | n/a |
| 10 | 1 | 25600 | 3908.9 | 5034.3 | 4806.9 | 4801.2 | 3167.6 | 3641.6 | 1510.3 | n/a | n/a | 0.81x | 1.29x | n/a |
| 10 | 1 | 191488 | 94001.0 | 96416.6 | 99281.7 | 99368.5 | 78663.6 | 78936.6 | 41843.8 | n/a | n/a | 0.84x | 1.03x | n/a |
| 10 | 1 | 256000 | n/a | n/a | n/a | n/a | n/a | n/a | 74419.5 | n/a | n/a | n/a | n/a | n/a |
| 11 | 1 | 25600 | 3.5 | 24.6 | 20.7 | 21.1 | 70.5 | 23.3 | 44.4 | 53.3 | 18.0 | 20.20x | 7.05x | 0.19x |
| 11 | 1 | 256000 | 15.3 | 160.4 | 113.4 | 113.4 | 138.9 | 125.9 | 39.0 | 185.3 | 17.7 | 9.08x | 10.49x | 0.86x |
| 12 | 1 | 25600 | 99.1 | 89.8 | 98.9 | 97.9 | 138.0 | 137.2 | 41.2 | 1825.0 | 299.2 | 1.39x | 0.91x | 0.33x |
| 12 | 1 | 256000 | 341.7 | 347.8 | 341.5 | 337.4 | 504.8 | 496.0 | 256.7 | 439.5 | 299.6 | 1.48x | 1.02x | 1.14x |
| 13 | 1 | 25600 | 440.8 | 448.6 | 526.6 | 524.9 | 620.1 | 593.8 | 123.5 | 2813.9 | 1290.5 | 1.41x | 1.02x | 0.34x |
| 13 | 1 | 256000 | 4751.4 | 4763.1 | 5428.4 | 5423.9 | 5001.5 | 4896.1 | 3715.4 | 5152.3 | 4680.3 | 1.05x | 1.00x | 1.02x |

## Precision sweep (median steady_us)

| dtype | variant | k | n | fused (ours) | app-level (ours) | eager (ours) | nojit (ours) | torch.compile | torch eager | JAX/XLA | IREE | Triton | speedup vs compile | interp tax | ours/Triton |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| float16 | 8 | 1 | 256000 | 713.7 | n/a | 2054.9 | 2056.2 | 865.0 | 1463.4 | 630.7 | n/a | n/a | 1.21x | n/a | n/a |
| float16 | 13 | 1 | 256000 | 222.5 | n/a | 350.3 | 342.6 | 365.1 | 413.0 | 96.9 | n/a | 132.2 | 1.64x | n/a | 1.68x |
| float32 | 8 | 1 | 256000 | 1869.9 | n/a | 4384.5 | 4386.2 | 2050.0 | 2665.9 | 1259.0 | n/a | n/a | 1.10x | n/a | n/a |
| float32 | 13 | 1 | 256000 | 317.4 | n/a | 633.7 | 626.2 | 569.9 | 615.2 | 215.1 | n/a | 133.5 | 1.80x | n/a | 2.38x |

## Guards / graph breaks (launches per iter, torch graph breaks)

| variant | n | launches/iter (fused) | torch graphs | torch breaks |
|---|---|---|---|---|
| 1 | 1000000 | 1.1 | 1 | 0 |
| 2 | 1000000 | 1.1 | 1 | 0 |
| 3 | 1000000 | 2.0 | 2 | 1 |
| 4 | 1000000 | 1.0 | 1 | 0 |
| 5 | 1000000 | 1.1 | 2 | 1 |

## End-to-end models (median steady_us, ratio to torch.compile)

| model | ours | torch.compile | torch eager | JAX/XLA | IREE | TensorRT | ratio ours/compile | ratio jax/compile | correctness |
|---|---|---|---|---|---|---|---|---|---|
| bert-mini | 480.7 | 926.2 | 1756.9 | 243.3 | 3632.9 | 667.8 | 0.52x | 0.26x | 2.09808e-05 |
| bert-tiny | 278.0 | 525.9 | 1072.0 | 111.3 | n/a | 440.4 | 0.53x | 0.21x | 2.67029e-05 |
| distilgpt2 | 1338.0 | 1371.8 | 2708.2 | 1001.1 | n/a | 3785.1 | 0.98x | 0.73x | 0.000175476 |
| mixer_b16 | 2693.2 | 3140.5 | 2939.2 | 2240.9 | n/a | 2463.8 | 0.86x | 0.71x | 3.8147e-05 |
| resnet18-b1 | 713.8 | 1069.8 | 1561.9 | 1106.0 | n/a | 1001.8 | 0.67x | 1.03x | 0.00963783 |
| resnet18-b8 | 2721.8 | 2185.9 | 2265.9 | 2274.6 | n/a | 2030.2 | 1.25x | 1.04x | 0.00571394 |
| smollm2-135m | 4066.7 | 5860.8 | 14579.9 | 2647.6 | n/a | 3761.8 | 0.69x | 0.45x | 0.000151277 |
| tiny-gpt2 | 203.0 | 403.5 | 1611.8 | 87.0 | n/a | 1391.8 | 0.50x | 0.22x | 2.6077e-08 |
| vit-tiny | 1331.8 | 2048.4 | 3723.0 | 745.1 | 24988.5 | 1058.9 | 0.65x | 0.36x | 1.04904e-05 |

## Ablations (median steady_us)

| experiment | variant | model | steady_us | note |
|---|---|---|---|---|
| budget_mb | 64 | distilgpt2 | 1354.7 |  |
| budget_mb | 8 | distilgpt2 | 1350.5 |  |
| flat_block | 256 | distilgpt2 | 1318.0 |  |
| flat_block | 256 | resnet18 | 623.6 |  |
| flat_block | 4096 | distilgpt2 | 1343.9 |  |
| flat_block | 4096 | resnet18 | 688.2 |  |
| fusion | off | distilgpt2 | 2020.9 | enable_opts minus tensor |
| fusion | on | distilgpt2 | 1360.3 |  |
| precision | float16 | distilgpt2 | 916.8 |  |
| precision | float16 | smollm2-135m | 2178.6 |  |
| precision | float32 | distilgpt2 | 1362.0 |  |
| precision | float32 | smollm2-135m | 4086.4 |  |
| tf32 | fp32 | resnet18-b1 | 773.2 |  |
| tf32 | fp32 | resnet18-b8 | 3209.4 |  |
| tf32 | tf32 | resnet18-b1 | 713.7 |  |
| tf32 | tf32 | resnet18-b8 | 2735.5 |  |
| tf32 | torch-fp32 | resnet18-b8 | 3060.3 |  |

## Compilation overhead (median; break-even vs torch eager, iterations)

| workload | system | compile_ms | first_run_ms | steady_us | break-even |
|---|---|---|---|---|---|
| bert-mini | torch-compile | n/a | 1286.7 | 926.2 | 1445 |
| bert-mini | torch-tensorrt | n/a | 4905.6 | 667.8 | 4425 |
| bert-mini | jax | 7296.1 | 2.8 | 243.3 | 4765 |
| bert-mini | iree | 998.9 | 20.6 | 3632.9 | n/a |
| bert-mini | ours | n/a | 181.4 | 480.7 | 74 |
| bert-tiny | torch-compile | n/a | 1123.5 | 525.9 | 1879 |
| bert-tiny | torch-tensorrt | n/a | 4121.4 | 440.4 | 6371 |
| bert-tiny | jax | 5794.5 | 1.8 | 111.3 | 5932 |
| bert-tiny | ours | n/a | 108.9 | 278.0 | 14 |
| distilgpt2 | torch-compile | n/a | 1502.3 | 1371.8 | 1024 |
| distilgpt2 | torch-tensorrt | n/a | 2832.4 | 3785.1 | n/a |
| distilgpt2 | jax | 7137.9 | 4.9 | 1001.1 | 4106 |
| distilgpt2 | ours | n/a | 552.8 | 1338.0 | 306 |
| micro v0 k1 n1000000 | torch-compile | n/a | 481.2 | 44.5 | 13224 |
| micro v0 k1 n1000000 | torch-tensorrt | n/a | 38.3 | 80.7 | n/a |
| micro v0 k1 n1000000 | jax | 194.8 | 0.5 | 43.5 | 4627 |
| micro v0 k1 n1000000 | iree | 544.6 | 24.9 | 5785.1 | n/a |
| micro v0 k1 n1000000 | triton | 274.6 | 0.1 | 30.7 | 5043 |
| micro v0 k4 n10000 | torch-compile | n/a | 425.8 | 58.8 | n/a |
| micro v0 k4 n10000 | torch-tensorrt | n/a | 39.7 | 54.9 | 954 |
| micro v0 k4 n10000 | jax | 197.6 | 0.5 | 43.4 | 11181 |
| micro v0 k4 n10000 | iree | 558.3 | 10.0 | 455.0 | n/a |
| micro v0 k4 n10000 | triton | 309.4 | 0.1 | 18.7 | 6964 |
| micro v0 k4 n100000 | torch-compile | n/a | 487.7 | 42.7 | 37884 |
| micro v0 k4 n100000 | torch-tensorrt | n/a | 38.6 | 60.5 | n/a |
| micro v0 k4 n100000 | jax | 184.2 | 0.5 | 45.8 | 16979 |
| micro v0 k4 n100000 | iree | 552.3 | 12.2 | 918.2 | n/a |
| micro v0 k4 n100000 | triton | 283.7 | 0.1 | 19.4 | 7048 |
| micro v0 k4 n1000000 | torch-compile | n/a | 433.3 | 68.4 | 1617 |
| micro v0 k4 n1000000 | torch-tensorrt | n/a | 38.6 | 321.6 | n/a |
| micro v0 k4 n1000000 | jax | 201.9 | 0.6 | 88.2 | 730 |
| micro v0 k4 n1000000 | iree | 553.7 | 25.5 | 5844.4 | n/a |
| micro v0 k4 n1000000 | triton | 307.3 | 0.1 | 39.1 | 983 |
| micro v0 k4 n10000000 | torch-compile | n/a | 426.3 | 637.0 | 161 |
| micro v0 k4 n10000000 | torch-tensorrt | n/a | 42.1 | 3045.6 | 107969 |
| micro v0 k4 n10000000 | jax | 219.5 | 1.3 | 770.6 | 80 |
| micro v0 k4 n10000000 | iree | 557.8 | 173.3 | 50176.4 | n/a |
| micro v0 k4 n10000000 | triton | 282.0 | 0.7 | 327.0 | 90 |
| micro v0 k8 n1000000 | torch-compile | n/a | 426.8 | 132.7 | 789 |
| micro v0 k8 n1000000 | torch-tensorrt | n/a | 40.3 | 642.9 | n/a |
| micro v0 k8 n1000000 | jax | 214.2 | 0.7 | 160.3 | 379 |
| micro v0 k8 n1000000 | iree | 559.0 | 25.5 | 5888.7 | n/a |
| micro v0 k8 n1000000 | triton | 306.8 | 0.2 | 67.0 | 481 |
| micro v1 k4 n1000000 | torch-compile | n/a | 488.2 | 307.5 | 48920 |
| micro v1 k4 n1000000 | torch-tensorrt | n/a | 42.2 | 325.9 | n/a |
| micro v1 k4 n1000000 | jax | 203.3 | 0.6 | 92.9 | 736 |
| micro v1 k4 n1000000 | triton | 289.8 | 0.1 | 43.4 | 917 |
| micro v10 k1 n191488 | torch-compile | n/a | 2037.4 | 78663.6 | 5948 |
| micro v10 k1 n191488 | torch-tensorrt | n/a | 2401.5 | 79030.5 | n/a |
| micro v10 k1 n191488 | jax | 5040.1 | 71.8 | 41843.8 | 127 |
| micro v10 k1 n25600 | torch-compile | n/a | 1810.9 | 3167.6 | 3092 |
| micro v10 k1 n25600 | torch-tensorrt | n/a | 2338.0 | 4085.6 | n/a |
| micro v10 k1 n25600 | jax | 4449.0 | 6.9 | 1510.3 | 1929 |
| micro v10 k1 n256000 | torch-tensorrt | n/a | 2461.9 | 135635.4 | n/a |
| micro v10 k1 n256000 | jax | 4885.1 | 100.2 | 74419.5 | n/a |
| micro v11 k1 n25600 | torch-compile | n/a | 1653.6 | 70.5 | n/a |
| micro v11 k1 n25600 | torch-tensorrt | n/a | 2023.2 | 58.5 | n/a |
| micro v11 k1 n25600 | jax | 54.7 | 0.4 | 44.4 | n/a |
| micro v11 k1 n25600 | iree | 177.4 | 8.8 | 53.3 | n/a |
| micro v11 k1 n25600 | triton | 308.3 | 0.1 | 18.0 | 19730 |
| micro v11 k1 n256000 | torch-compile | n/a | 1734.4 | 138.9 | n/a |
| micro v11 k1 n256000 | torch-tensorrt | n/a | 1990.4 | 152.8 | n/a |
| micro v11 k1 n256000 | jax | 53.6 | 0.4 | 39.0 | -1999 |
| micro v11 k1 n256000 | iree | 174.4 | 11.4 | 185.3 | n/a |
| micro v11 k1 n256000 | triton | 308.8 | 0.1 | 17.7 | 751 |
| micro v12 k1 n25600 | torch-compile | n/a | 1691.1 | 138.0 | n/a |
| micro v12 k1 n25600 | torch-tensorrt | n/a | 2029.6 | 134.9 | 790547 |
| micro v12 k1 n25600 | jax | 1547.6 | 0.6 | 41.2 | 13619 |
| micro v12 k1 n25600 | iree | 214.5 | 14.9 | 1825.0 | n/a |
| micro v12 k1 n25600 | triton | 309.4 | 0.4 | 299.2 | n/a |
| micro v12 k1 n256000 | torch-compile | n/a | 1791.0 | 504.8 | n/a |
| micro v12 k1 n256000 | torch-tensorrt | n/a | 2058.5 | 486.6 | 190499 |
| micro v12 k1 n256000 | jax | 1593.4 | 0.9 | 256.7 | 5560 |
| micro v12 k1 n256000 | iree | 227.4 | 10.5 | 439.5 | -453 |
| micro v12 k1 n256000 | triton | 279.7 | 0.4 | 299.6 | 84 |
| micro v13 k1 n25600 | torch-compile | n/a | 1726.9 | 620.1 | n/a |
| micro v13 k1 n25600 | torch-tensorrt | n/a | 2071.0 | 582.3 | 154794 |
| micro v13 k1 n25600 | jax | 1877.8 | 1.0 | 123.5 | 3406 |
| micro v13 k1 n25600 | iree | 326.6 | 12.3 | 2813.9 | n/a |
| micro v13 k1 n25600 | triton | 313.0 | 1.3 | 1290.5 | n/a |
| micro v13 k1 n256000 | torch-compile | n/a | 1746.0 | 5001.5 | n/a |
| micro v13 k1 n256000 | torch-tensorrt | n/a | 2171.1 | 4846.6 | 37786 |
| micro v13 k1 n256000 | jax | 1946.6 | 4.8 | 3715.4 | 1395 |
| micro v13 k1 n256000 | iree | 637.7 | 20.2 | 5152.3 | n/a |
| micro v13 k1 n256000 | triton | 315.4 | 4.7 | 4680.3 | 74 |
| micro v2 k4 n1000000 | torch-compile | n/a | 481.6 | 307.5 | 48282 |
| micro v2 k4 n1000000 | torch-tensorrt | n/a | 41.7 | 325.9 | n/a |
| micro v2 k4 n1000000 | jax | 202.9 | 0.6 | 92.9 | 736 |
| micro v2 k4 n1000000 | triton | 282.5 | 0.1 | 43.4 | 892 |
| micro v3 k4 n1000000 | torch-compile | n/a | 500.7 | 164.8 | 2178 |
| micro v3 k4 n1000000 | torch-tensorrt | n/a | 39.4 | 384.9 | n/a |
| micro v3 k4 n1000000 | jax | 204.2 | 0.6 | 280.9 | 1728 |
| micro v3 k4 n1000000 | triton | 309.7 | 0.1 | 122.8 | 1066 |
| micro v4 k4 n1000000 | torch-compile | n/a | 417.2 | 332.4 | 38895 |
| micro v4 k4 n1000000 | torch-tensorrt | n/a | 42.1 | 351.5 | n/a |
| micro v4 k4 n1000000 | jax | 204.1 | 0.6 | 115.2 | 741 |
| micro v4 k4 n1000000 | triton | 311.2 | 0.1 | 67.9 | 1002 |
| micro v5 k4 n1000000 | torch-compile | n/a | 499.5 | 332.7 | 48813 |
| micro v5 k4 n1000000 | torch-tensorrt | n/a | 39.5 | 351.5 | n/a |
| micro v5 k4 n1000000 | jax | 205.5 | 0.6 | 120.3 | 761 |
| micro v5 k4 n1000000 | triton | 299.7 | 0.1 | 69.2 | 962 |
| micro v6 k1 n25600 | torch-compile | n/a | 1712.5 | 399.1 | n/a |
| micro v6 k1 n25600 | torch-tensorrt | n/a | 2104.4 | 382.1 | 159474 |
| micro v6 k1 n25600 | jax | 1605.8 | 1.0 | 107.6 | 4735 |
| micro v6 k1 n25600 | iree | 225.4 | 18.0 | 5392.1 | n/a |
| micro v6 k1 n25600 | triton | 308.5 | 0.9 | 875.4 | n/a |
| micro v6 k1 n256000 | torch-compile | n/a | 1675.7 | 1243.9 | 134018 |
| micro v6 k1 n256000 | torch-tensorrt | n/a | 2099.0 | 1254.9 | n/a |
| micro v6 k1 n256000 | jax | 1595.1 | 1.7 | 758.3 | 2649 |
| micro v6 k1 n256000 | iree | 237.7 | 10.7 | 911.7 | -101 |
| micro v6 k1 n256000 | triton | 302.7 | 0.9 | 883.7 | 56 |
| micro v7 k1 n25600 | torch-compile | n/a | 1724.5 | 859.8 | n/a |
| micro v7 k1 n25600 | torch-tensorrt | n/a | 2131.0 | 821.8 | n/a |
| micro v7 k1 n25600 | jax | 2186.2 | 1.5 | 324.7 | 4101 |
| micro v7 k1 n256000 | torch-compile | n/a | 1752.3 | 3043.7 | 29517 |
| micro v7 k1 n256000 | torch-tensorrt | n/a | 2151.8 | 3107.7 | n/a |
| micro v7 k1 n256000 | jax | 2223.7 | 4.0 | 1975.4 | 1727 |
| micro v8 k1 n25600 | torch-compile | n/a | 1731.4 | 515.1 | 33285 |
| micro v8 k1 n25600 | torch-tensorrt | n/a | 2196.9 | 563.8 | n/a |
| micro v8 k1 n25600 | jax | 1256.2 | 1.3 | 267.2 | 3268 |
| micro v8 k1 n256000 | torch-compile | n/a | 1848.4 | 29781.0 | n/a |
| micro v8 k1 n256000 | torch-tensorrt | n/a | 2230.0 | 27364.9 | 72647 |
| micro v8 k1 n256000 | jax | 1547.5 | 16.7 | 15402.6 | 102 |
| micro v9 k1 n25600 | torch-compile | n/a | 1825.4 | 860.5 | n/a |
| micro v9 k1 n25600 | torch-tensorrt | n/a | 2140.4 | 941.0 | n/a |
| micro v9 k1 n25600 | jax | 371.4 | 0.8 | 251.3 | 172 |
| micro v9 k1 n256000 | torch-compile | n/a | 1803.2 | 654.6 | n/a |
| micro v9 k1 n256000 | torch-tensorrt | n/a | 2155.9 | 1536.9 | n/a |
| micro v9 k1 n256000 | jax | 813.9 | 1.0 | 349.8 | 2183 |
| mixer_b16 | torch-compile | n/a | 1174.8 | 3140.5 | n/a |
| mixer_b16 | torch-tensorrt | n/a | 5675.0 | 2463.8 | 11669 |
| mixer_b16 | jax | 7278.4 | 7.0 | 2240.9 | 10250 |
| mixer_b16 | ours | n/a | 184.2 | 2693.2 | 230 |
| resnet18-b1 | torch-compile | n/a | 1037.0 | 1069.8 | 1716 |
| resnet18-b1 | torch-tensorrt | n/a | 4683.0 | 1001.8 | 8017 |
| resnet18-b1 | jax | 603.8 | 3.9 | 1106.0 | 910 |
| resnet18-b1 | ours | n/a | 132.4 | 713.8 | -71 |
| resnet18-b8 | torch-compile | n/a | 998.2 | 2185.9 | 10121 |
| resnet18-b8 | torch-tensorrt | n/a | 4676.9 | 2030.2 | 19043 |
| resnet18-b8 | jax | 941.4 | 4.3 | 2274.6 | n/a |
| resnet18-b8 | ours | n/a | 178.2 | 2721.8 | n/a |
| smollm2-135m | torch-compile | n/a | 3906.8 | 5860.8 | 386 |
| smollm2-135m | torch-tensorrt | n/a | 22309.9 | 3761.8 | 2012 |
| smollm2-135m | jax | 10724.5 | 19.5 | 2647.6 | 855 |
| smollm2-135m | ours | n/a | 495.1 | 4066.7 | -4 |
| tiny-gpt2 | torch-compile | n/a | 1175.2 | 403.5 | 577 |
| tiny-gpt2 | torch-tensorrt | n/a | 1780.9 | 1391.8 | 5920 |
| tiny-gpt2 | jax | 1138.0 | 1.6 | 87.0 | 434 |
| tiny-gpt2 | ours | n/a | 102.4 | 203.0 | -267 |
| vit-tiny | torch-compile | n/a | 1674.5 | 2048.4 | 923 |
| vit-tiny | torch-tensorrt | n/a | 7675.6 | 1058.9 | 2833 |
| vit-tiny | jax | 6527.1 | 7.1 | 745.1 | 2151 |
| vit-tiny | iree | 1663.2 | 39.4 | 24988.5 | n/a |
| vit-tiny | ours | n/a | 141.7 | 1331.8 | 6 |

## Dynamic sequence length (median steady_us per length)

| system | length | median_us | loops | bridges | recompiles |
|---|---|---|---|---|---|
| ours | 32 | 1033.5 | 168 | 167 | 0 |
| ours | 48 | 1431.0 | 168 | 167 | 0 |
| ours | 64 | 1497.5 | 168 | 167 | 0 |
| ours | 96 | 1747.0 | 168 | 167 | 0 |
| ours | 128 | 2067.0 | 168 | 167 | 0 |
| torch-compile-dynamic | 32 | 1259.8 | 0 | 0 | 0 |
| torch-compile-dynamic | 48 | 1685.7 | 0 | 0 | 0 |
| torch-compile-dynamic | 64 | 1739.5 | 0 | 0 | 0 |
| torch-compile-dynamic | 96 | 1886.8 | 0 | 0 | 0 |
| torch-compile-dynamic | 128 | 2141.2 | 0 | 0 | 0 |
| torch-compile-static | 32 | 1169.3 | 0 | 0 | 4 |
| torch-compile-static | 48 | 1552.2 | 0 | 0 | 4 |
| torch-compile-static | 64 | 1592.5 | 0 | 0 | 4 |
| torch-compile-static | 96 | 1827.2 | 0 | 0 | 4 |
| torch-compile-static | 128 | 2056.7 | 0 | 0 | 4 |
| torch-eager | 32 | 2403.7 | 0 | 0 | 0 |
| torch-eager | 48 | 2572.3 | 0 | 0 | 0 |
| torch-eager | 64 | 2575.2 | 0 | 0 | 0 |
| torch-eager | 96 | 2636.9 | 0 | 0 | 0 |
| torch-eager | 128 | 2730.0 | 0 | 0 | 0 |

