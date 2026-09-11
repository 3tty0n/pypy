# Paper benchmark summary

## Microbenchmarks (median steady_us over rounds)

| variant | k | n | fused (ours) | app-level (ours) | eager (ours) | nojit (ours) | torch.compile | torch eager | JAX/XLA | IREE | Triton | speedup vs compile | interp tax | ours/Triton |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1 | 1000000 | 28.7 | n/a | 76.7 | 76.8 | 44.2 | 78.1 | 44.1 | 5820.1 | 30.6 | 1.54x | n/a | 0.94x |
| 0 | 4 | 10000 | 11.9 | n/a | 38.7 | 39.3 | 42.2 | 58.5 | 43.5 | 436.3 | 19.4 | 3.54x | n/a | 0.61x |
| 0 | 4 | 100000 | 12.8 | n/a | 43.3 | 43.0 | 40.0 | 60.8 | 44.2 | 926.5 | 19.0 | 3.13x | n/a | 0.67x |
| 0 | 4 | 1000000 | 39.1 | n/a | 306.9 | 306.5 | 68.4 | 312.3 | 88.1 | 5651.7 | 38.5 | 1.75x | n/a | 1.02x |
| 0 | 4 | 10000000 | 313.7 | n/a | 3026.8 | 3026.9 | 645.5 | 3044.7 | 770.5 | 49698.5 | 327.2 | 2.06x | n/a | 0.96x |
| 0 | 8 | 1000000 | 67.3 | n/a | 612.1 | 612.5 | 132.7 | 624.5 | 160.1 | 5776.2 | 66.8 | 1.97x | n/a | 1.01x |
| 1 | 4 | 1000000 | 41.6 | n/a | 311.5 | 310.7 | 306.9 | 316.7 | 92.8 | n/a | 42.9 | 7.38x | n/a | 0.97x |
| 2 | 4 | 1000000 | 44.0 | n/a | 311.5 | 310.7 | 306.8 | 316.6 | 92.8 | n/a | 43.0 | 6.96x | n/a | 1.03x |
| 3 | 4 | 1000000 | 99.5 | n/a | 357.5 | 365.9 | 166.9 | 376.6 | 272.2 | n/a | 120.2 | 1.68x | n/a | 0.83x |
| 4 | 4 | 1000000 | 53.3 | n/a | 338.8 | 335.6 | 331.6 | 342.2 | 120.3 | n/a | 68.6 | 6.23x | n/a | 0.78x |
| 5 | 4 | 1000000 | 56.1 | n/a | 336.3 | 335.7 | 331.7 | 342.2 | 115.5 | n/a | 68.8 | 5.91x | n/a | 0.82x |
| 6 | 1 | 25600 | 277.2 | n/a | 279.8 | 277.3 | 387.0 | 392.5 | 107.7 | 5150.4 | 881.0 | 1.40x | n/a | 0.31x |
| 6 | 1 | 256000 | 1011.4 | n/a | 1015.7 | 1011.7 | 1234.0 | 1270.9 | 758.4 | 906.2 | 850.0 | 1.22x | n/a | 1.19x |
| 7 | 1 | 25600 | 646.9 | n/a | 668.0 | 667.3 | 840.6 | 800.6 | 331.6 | n/a | n/a | 1.30x | n/a | n/a |
| 7 | 1 | 256000 | 2798.7 | n/a | 2807.2 | 2822.0 | 3045.7 | 3077.3 | 1970.7 | n/a | n/a | 1.09x | n/a | n/a |
| 8 | 1 | 25600 | 464.5 | n/a | 778.7 | 779.2 | 516.3 | 557.3 | 267.1 | n/a | n/a | 1.11x | n/a | n/a |
| 8 | 1 | 256000 | 22300.3 | n/a | 24837.3 | 25048.2 | 29624.8 | 27209.3 | 15363.0 | n/a | n/a | 1.33x | n/a | n/a |
| 9 | 1 | 25600 | 756.8 | n/a | 764.8 | 764.9 | 860.7 | 801.9 | 237.8 | n/a | n/a | 1.14x | n/a | n/a |
| 9 | 1 | 256000 | 572.2 | n/a | 572.8 | 568.2 | 664.0 | 572.8 | 329.9 | n/a | n/a | 1.16x | n/a | n/a |
| 10 | 1 | 25600 | 3905.1 | n/a | 4810.8 | 4813.4 | 3178.4 | 3729.5 | 1527.1 | n/a | n/a | 0.81x | n/a | n/a |
| 10 | 1 | 191488 | 93902.3 | n/a | 99340.3 | 99362.3 | 78668.4 | 78944.0 | 41813.3 | n/a | n/a | 0.84x | n/a | n/a |
| 11 | 1 | 25600 | 3.3 | n/a | 20.5 | 20.9 | 60.0 | 25.7 | 38.9 | 53.0 | 18.6 | 18.16x | n/a | 0.18x |
| 11 | 1 | 256000 | 15.1 | n/a | 113.4 | 113.4 | 160.3 | 122.9 | 43.1 | 183.5 | 18.8 | 10.61x | n/a | 0.81x |
| 12 | 1 | 25600 | 98.8 | n/a | 98.9 | 97.9 | 142.6 | 137.4 | 47.2 | 1871.2 | 299.2 | 1.44x | n/a | 0.33x |
| 12 | 1 | 256000 | 340.9 | n/a | 341.1 | 337.1 | 481.6 | 491.7 | 256.8 | 435.8 | 299.6 | 1.41x | n/a | 1.14x |
| 13 | 1 | 25600 | 441.1 | n/a | 523.7 | 523.8 | 598.0 | 593.0 | 123.7 | 2822.6 | 1232.8 | 1.36x | n/a | 0.36x |
| 13 | 1 | 256000 | 4750.1 | n/a | 5428.2 | 5423.3 | 5024.3 | 4881.9 | 3739.5 | 5158.4 | 4679.0 | 1.06x | n/a | 1.02x |

## Precision sweep (median steady_us)

| dtype | variant | k | n | fused (ours) | app-level (ours) | eager (ours) | nojit (ours) | torch.compile | torch eager | JAX/XLA | IREE | Triton | speedup vs compile | interp tax | ours/Triton |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| float16 | 8 | 1 | 256000 | 711.9 | n/a | 2054.8 | 2053.9 | 860.0 | 1461.9 | 628.8 | n/a | n/a | 1.21x | n/a | n/a |
| float16 | 13 | 1 | 256000 | 221.4 | n/a | 349.1 | 341.7 | 430.0 | 410.7 | 96.4 | n/a | 132.0 | 1.94x | n/a | 1.68x |
| float32 | 8 | 1 | 256000 | 1395.8 | n/a | 3919.6 | 3919.1 | 2017.5 | 2652.5 | 1256.4 | n/a | n/a | 1.45x | n/a | n/a |
| float32 | 13 | 1 | 256000 | 272.5 | n/a | 588.0 | 580.9 | 544.5 | 636.8 | 214.8 | n/a | 131.1 | 2.00x | n/a | 2.08x |

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
| bert-mini | 428.6 | 921.5 | 1804.7 | 242.6 | 3627.1 | 659.5 | 0.47x | 0.26x | 1.90735e-05 |
| bert-tiny | 240.8 | 489.7 | 1090.2 | 111.1 | n/a | 431.9 | 0.49x | 0.23x | 3.05176e-05 |
| distilgpt2 | 1278.7 | 1350.6 | 2761.1 | 1014.6 | n/a | 3739.8 | 0.95x | 0.75x | 0.000175476 |
| mixer_b16 | 2672.0 | 3114.9 | 2876.9 | 2238.2 | n/a | 2461.8 | 0.86x | 0.72x | 3.8147e-05 |
| resnet18-b1 | 708.5 | 960.8 | 1580.7 | 1121.8 | n/a | 989.7 | 0.74x | 1.17x | 0.00963783 |
| resnet18-b8 | 2705.3 | 2188.9 | 2269.6 | 2273.1 | n/a | 2013.1 | 1.24x | 1.04x | 0.00571394 |
| smollm2-135m | 3660.7 | 5206.8 | 15713.2 | 2651.1 | n/a | 3792.3 | 0.70x | 0.51x | 0.000151277 |
| tiny-gpt2 | 194.2 | 409.4 | 1630.6 | 80.8 | n/a | 1384.8 | 0.47x | 0.20x | 2.6077e-08 |
| vit-tiny | 974.8 | 2056.3 | 3826.2 | 746.0 | 24994.8 | 1034.4 | 0.47x | 0.36x | 1.07288e-05 |

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

## Compilation overhead (median; break-even vs torch eager, iterations)

| workload | system | compile_ms | first_run_ms | steady_us | break-even |
|---|---|---|---|---|---|
| bert-mini | torch-compile | n/a | 1302.8 | 921.5 | 1380 |
| bert-mini | torch-tensorrt | n/a | 4856.2 | 659.5 | 4167 |
| bert-mini | jax | 7061.4 | 2.7 | 242.6 | 4468 |
| bert-mini | iree | 1010.4 | 18.6 | 3627.1 | n/a |
| bert-mini | ours | n/a | 194.6 | 428.6 | 80 |
| bert-tiny | torch-compile | n/a | 1117.5 | 489.7 | 1700 |
| bert-tiny | torch-tensorrt | n/a | 4212.0 | 431.9 | 6251 |
| bert-tiny | jax | 5870.0 | 1.8 | 111.1 | 5898 |
| bert-tiny | ours | n/a | 115.7 | 240.8 | 22 |
| distilgpt2 | torch-compile | n/a | 1557.6 | 1350.6 | 1013 |
| distilgpt2 | torch-tensorrt | n/a | 2856.2 | 3739.8 | n/a |
| distilgpt2 | jax | 6912.1 | 4.6 | 1014.6 | 3887 |
| distilgpt2 | ours | n/a | 534.8 | 1278.7 | 274 |
| micro v0 k1 n1000000 | torch-compile | n/a | 481.8 | 44.2 | 13120 |
| micro v0 k1 n1000000 | torch-tensorrt | n/a | 37.9 | 80.4 | n/a |
| micro v0 k1 n1000000 | jax | 200.0 | 0.5 | 44.1 | 4793 |
| micro v0 k1 n1000000 | iree | 555.2 | 23.9 | 5820.1 | n/a |
| micro v0 k1 n1000000 | triton | 307.9 | 0.1 | 30.6 | 5699 |
| micro v0 k4 n10000 | torch-compile | n/a | 469.3 | 42.2 | 26723 |
| micro v0 k4 n10000 | torch-tensorrt | n/a | 40.2 | 59.0 | n/a |
| micro v0 k4 n10000 | jax | 197.6 | 0.5 | 43.5 | 10899 |
| micro v0 k4 n10000 | iree | 568.2 | 9.1 | 436.3 | n/a |
| micro v0 k4 n10000 | triton | 308.1 | 0.1 | 19.4 | 6996 |
| micro v0 k4 n100000 | torch-compile | n/a | 486.1 | 40.0 | 21578 |
| micro v0 k4 n100000 | torch-tensorrt | n/a | 40.6 | 61.1 | n/a |
| micro v0 k4 n100000 | jax | 188.4 | 0.5 | 44.2 | 9144 |
| micro v0 k4 n100000 | iree | 560.9 | 11.0 | 926.5 | n/a |
| micro v0 k4 n100000 | triton | 307.5 | 0.1 | 19.0 | 6461 |
| micro v0 k4 n1000000 | torch-compile | n/a | 420.2 | 68.4 | 1578 |
| micro v0 k4 n1000000 | torch-tensorrt | n/a | 37.7 | 321.4 | n/a |
| micro v0 k4 n1000000 | jax | 202.8 | 0.6 | 88.1 | 749 |
| micro v0 k4 n1000000 | iree | 560.5 | 23.5 | 5651.7 | n/a |
| micro v0 k4 n1000000 | triton | 307.9 | 0.1 | 38.5 | 996 |
| micro v0 k4 n10000000 | torch-compile | n/a | 449.9 | 645.5 | 173 |
| micro v0 k4 n10000000 | torch-tensorrt | n/a | 40.0 | 3045.5 | n/a |
| micro v0 k4 n10000000 | jax | 218.5 | 1.3 | 770.5 | 81 |
| micro v0 k4 n10000000 | iree | 563.6 | 165.9 | 49698.5 | n/a |
| micro v0 k4 n10000000 | triton | 277.0 | 0.6 | 327.2 | 89 |
| micro v0 k8 n1000000 | torch-compile | n/a | 427.3 | 132.7 | 796 |
| micro v0 k8 n1000000 | torch-tensorrt | n/a | 40.8 | 642.4 | n/a |
| micro v0 k8 n1000000 | jax | 214.0 | 0.6 | 160.1 | 386 |
| micro v0 k8 n1000000 | iree | 567.1 | 23.2 | 5776.2 | n/a |
| micro v0 k8 n1000000 | triton | 308.2 | 0.2 | 66.8 | 489 |
| micro v1 k4 n1000000 | torch-compile | n/a | 490.5 | 306.9 | 46213 |
| micro v1 k4 n1000000 | torch-tensorrt | n/a | 37.8 | 325.8 | n/a |
| micro v1 k4 n1000000 | jax | 202.2 | 0.6 | 92.8 | 736 |
| micro v1 k4 n1000000 | triton | 309.3 | 0.1 | 42.9 | 991 |
| micro v10 k1 n191488 | torch-compile | n/a | 1877.9 | 78668.4 | 5450 |
| micro v10 k1 n191488 | torch-tensorrt | n/a | 2455.3 | 79037.6 | n/a |
| micro v10 k1 n191488 | jax | 4927.0 | 71.9 | 41813.3 | 125 |
| micro v10 k1 n25600 | torch-compile | n/a | 1784.2 | 3178.4 | 2696 |
| micro v10 k1 n25600 | torch-tensorrt | n/a | 2381.4 | 4112.4 | n/a |
| micro v10 k1 n25600 | jax | 4229.6 | 6.6 | 1527.1 | 1788 |
| micro v11 k1 n25600 | torch-compile | n/a | 1680.4 | 60.0 | n/a |
| micro v11 k1 n25600 | torch-tensorrt | n/a | 2034.0 | 58.0 | n/a |
| micro v11 k1 n25600 | jax | 54.3 | 0.4 | 38.9 | n/a |
| micro v11 k1 n25600 | iree | 178.7 | 7.9 | 53.0 | n/a |
| micro v11 k1 n25600 | triton | 307.0 | 0.1 | 18.6 | 18615 |
| micro v11 k1 n256000 | torch-compile | n/a | 1711.2 | 160.3 | n/a |
| micro v11 k1 n256000 | torch-tensorrt | n/a | 2022.6 | 151.5 | n/a |
| micro v11 k1 n256000 | jax | 52.6 | 0.4 | 43.1 | -1859 |
| micro v11 k1 n256000 | iree | 175.1 | 9.9 | 183.5 | n/a |
| micro v11 k1 n256000 | triton | 307.0 | 0.1 | 18.8 | 1013 |
| micro v12 k1 n25600 | torch-compile | n/a | 1754.9 | 142.6 | n/a |
| micro v12 k1 n25600 | torch-tensorrt | n/a | 2025.7 | 134.5 | 618692 |
| micro v12 k1 n25600 | jax | 1520.4 | 0.8 | 47.2 | 14495 |
| micro v12 k1 n25600 | iree | 217.0 | 13.6 | 1871.2 | n/a |
| micro v12 k1 n25600 | triton | 308.4 | 0.4 | 299.2 | n/a |
| micro v12 k1 n256000 | torch-compile | n/a | 1767.7 | 481.6 | 152196 |
| micro v12 k1 n256000 | torch-tensorrt | n/a | 2019.1 | 482.9 | 203850 |
| micro v12 k1 n256000 | jax | 1569.6 | 0.9 | 256.8 | 5704 |
| micro v12 k1 n256000 | iree | 229.6 | 9.6 | 435.8 | 161 |
| micro v12 k1 n256000 | triton | 308.5 | 0.4 | 299.6 | 409 |
| micro v13 k1 n25600 | torch-compile | n/a | 1802.6 | 598.0 | n/a |
| micro v13 k1 n25600 | torch-tensorrt | n/a | 2147.5 | 580.5 | 151841 |
| micro v13 k1 n25600 | jax | 1956.8 | 1.0 | 123.7 | 3637 |
| micro v13 k1 n25600 | iree | 331.1 | 11.0 | 2822.6 | n/a |
| micro v13 k1 n25600 | triton | 311.5 | 1.3 | 1232.8 | n/a |
| micro v13 k1 n256000 | torch-compile | n/a | 1826.8 | 5024.3 | n/a |
| micro v13 k1 n256000 | torch-tensorrt | n/a | 2070.4 | 4868.6 | 134859 |
| micro v13 k1 n256000 | jax | 1980.7 | 4.8 | 3739.5 | 1503 |
| micro v13 k1 n256000 | iree | 637.5 | 18.9 | 5158.4 | n/a |
| micro v13 k1 n256000 | triton | 315.2 | 4.7 | 4679.0 | 253 |
| micro v2 k4 n1000000 | torch-compile | n/a | 419.2 | 306.8 | 38874 |
| micro v2 k4 n1000000 | torch-tensorrt | n/a | 37.6 | 325.8 | n/a |
| micro v2 k4 n1000000 | jax | 196.7 | 0.5 | 92.8 | 723 |
| micro v2 k4 n1000000 | triton | 308.9 | 0.1 | 43.0 | 1000 |
| micro v3 k4 n1000000 | torch-compile | n/a | 562.7 | 166.9 | 2503 |
| micro v3 k4 n1000000 | torch-tensorrt | n/a | 38.5 | 384.1 | n/a |
| micro v3 k4 n1000000 | jax | 204.7 | 0.6 | 272.2 | 1605 |
| micro v3 k4 n1000000 | triton | 310.3 | 0.1 | 120.2 | 1063 |
| micro v4 k4 n1000000 | torch-compile | n/a | 430.7 | 331.6 | 36997 |
| micro v4 k4 n1000000 | torch-tensorrt | n/a | 37.3 | 351.7 | n/a |
| micro v4 k4 n1000000 | jax | 197.9 | 0.6 | 120.3 | 725 |
| micro v4 k4 n1000000 | triton | 309.2 | 0.1 | 68.6 | 992 |
| micro v5 k4 n1000000 | torch-compile | n/a | 506.8 | 331.7 | 44932 |
| micro v5 k4 n1000000 | torch-tensorrt | n/a | 37.8 | 351.4 | n/a |
| micro v5 k4 n1000000 | jax | 205.5 | 0.6 | 115.5 | 741 |
| micro v5 k4 n1000000 | triton | 308.4 | 0.1 | 68.8 | 990 |
| micro v6 k1 n25600 | torch-compile | n/a | 1795.2 | 387.0 | 287626 |
| micro v6 k1 n25600 | torch-tensorrt | n/a | 2007.5 | 380.3 | 146864 |
| micro v6 k1 n25600 | jax | 1579.6 | 1.0 | 107.7 | 4762 |
| micro v6 k1 n25600 | iree | 223.8 | 16.8 | 5150.4 | n/a |
| micro v6 k1 n25600 | triton | 308.7 | 0.9 | 881.0 | n/a |
| micro v6 k1 n256000 | torch-compile | n/a | 1779.3 | 1234.0 | 41564 |
| micro v6 k1 n256000 | torch-tensorrt | n/a | 2140.6 | 1271.3 | n/a |
| micro v6 k1 n256000 | jax | 1595.4 | 1.6 | 758.4 | 2634 |
| micro v6 k1 n256000 | iree | 237.3 | 9.8 | 906.2 | 0 |
| micro v6 k1 n256000 | triton | 306.1 | 0.9 | 850.0 | 143 |
| micro v7 k1 n25600 | torch-compile | n/a | 1807.6 | 840.6 | n/a |
| micro v7 k1 n25600 | torch-tensorrt | n/a | 2114.5 | 820.7 | n/a |
| micro v7 k1 n25600 | jax | 2149.9 | 1.6 | 331.6 | 4066 |
| micro v7 k1 n256000 | torch-compile | n/a | 1741.6 | 3045.7 | 47174 |
| micro v7 k1 n256000 | torch-tensorrt | n/a | 2199.7 | 3089.2 | n/a |
| micro v7 k1 n256000 | jax | 2280.0 | 3.8 | 1970.7 | 1834 |
| micro v8 k1 n25600 | torch-compile | n/a | 1785.2 | 516.3 | 36888 |
| micro v8 k1 n25600 | torch-tensorrt | n/a | 2190.8 | 563.5 | n/a |
| micro v8 k1 n25600 | jax | 1262.8 | 1.3 | 267.1 | 3413 |
| micro v8 k1 n256000 | torch-compile | n/a | 1792.2 | 29624.8 | n/a |
| micro v8 k1 n256000 | torch-tensorrt | n/a | 2256.4 | 27305.4 | n/a |
| micro v8 k1 n256000 | jax | 1530.6 | 16.6 | 15363.0 | 105 |
| micro v9 k1 n25600 | torch-compile | n/a | 1789.5 | 860.7 | n/a |
| micro v9 k1 n25600 | torch-tensorrt | n/a | 2136.3 | 942.9 | n/a |
| micro v9 k1 n25600 | jax | 363.0 | 0.8 | 237.8 | 199 |
| micro v9 k1 n256000 | torch-compile | n/a | 1704.2 | 664.0 | n/a |
| micro v9 k1 n256000 | torch-tensorrt | n/a | 2151.9 | 1549.6 | n/a |
| micro v9 k1 n256000 | jax | 802.1 | 1.2 | 329.9 | 2219 |
| mixer_b16 | torch-compile | n/a | 1243.6 | 3114.9 | n/a |
| mixer_b16 | torch-tensorrt | n/a | 5985.3 | 2461.8 | 14117 |
| mixer_b16 | jax | 7200.5 | 7.0 | 2238.2 | 11088 |
| mixer_b16 | ours | n/a | 181.1 | 2672.0 | 272 |
| resnet18-b1 | torch-compile | n/a | 1018.4 | 960.8 | 1336 |
| resnet18-b1 | torch-tensorrt | n/a | 4643.6 | 989.7 | 7535 |
| resnet18-b1 | jax | 587.8 | 4.0 | 1121.8 | 875 |
| resnet18-b1 | ours | n/a | 124.6 | 708.5 | -75 |
| resnet18-b8 | torch-compile | n/a | 1078.6 | 2188.9 | 11048 |
| resnet18-b8 | torch-tensorrt | n/a | 4813.8 | 2013.1 | 18038 |
| resnet18-b8 | jax | 934.8 | 4.4 | 2273.1 | n/a |
| resnet18-b8 | ours | n/a | 153.3 | 2705.3 | n/a |
| smollm2-135m | torch-compile | n/a | 3863.3 | 5206.8 | 316 |
| smollm2-135m | torch-tensorrt | n/a | 21935.5 | 3792.3 | 1794 |
| smollm2-135m | jax | 10616.3 | 17.7 | 2651.1 | 772 |
| smollm2-135m | ours | n/a | 495.4 | 3660.7 | -4 |
| tiny-gpt2 | torch-compile | n/a | 1178.7 | 409.4 | 577 |
| tiny-gpt2 | torch-tensorrt | n/a | 1843.5 | 1384.8 | 5571 |
| tiny-gpt2 | jax | 1142.1 | 1.5 | 80.8 | 432 |
| tiny-gpt2 | ours | n/a | 114.5 | 194.2 | -250 |
| vit-tiny | torch-compile | n/a | 1778.9 | 2056.3 | 938 |
| vit-tiny | torch-tensorrt | n/a | 7641.6 | 1034.4 | 2695 |
| vit-tiny | jax | 6705.4 | 7.7 | 746.0 | 2141 |
| vit-tiny | iree | 1681.5 | 38.6 | 24994.8 | n/a |
| vit-tiny | ours | n/a | 143.5 | 974.8 | 9 |

## Dynamic sequence length (median steady_us per length)

| system | length | median_us | loops | bridges | recompiles |
|---|---|---|---|---|---|
| ours | 32 | 965.3 | 168 | 167 | 0 |
| ours | 48 | 1363.5 | 168 | 167 | 0 |
| ours | 64 | 1420.8 | 168 | 167 | 0 |
| ours | 96 | 1689.5 | 168 | 167 | 0 |
| ours | 128 | 1998.0 | 168 | 167 | 0 |
| torch-compile-dynamic | 32 | 1265.0 | 0 | 0 | 0 |
| torch-compile-dynamic | 48 | 1698.4 | 0 | 0 | 0 |
| torch-compile-dynamic | 64 | 1750.7 | 0 | 0 | 0 |
| torch-compile-dynamic | 96 | 1934.7 | 0 | 0 | 0 |
| torch-compile-dynamic | 128 | 2148.8 | 0 | 0 | 0 |
| torch-compile-static | 32 | 1197.6 | 0 | 0 | 4 |
| torch-compile-static | 48 | 1569.6 | 0 | 0 | 4 |
| torch-compile-static | 64 | 1610.5 | 0 | 0 | 4 |
| torch-compile-static | 96 | 1838.2 | 0 | 0 | 4 |
| torch-compile-static | 128 | 2065.6 | 0 | 0 | 4 |
| torch-eager | 32 | 2447.1 | 0 | 0 | 0 |
| torch-eager | 48 | 2610.0 | 0 | 0 | 0 |
| torch-eager | 64 | 2603.6 | 0 | 0 | 0 |
| torch-eager | 96 | 2681.8 | 0 | 0 | 0 |
| torch-eager | 128 | 2767.8 | 0 | 0 | 0 |

