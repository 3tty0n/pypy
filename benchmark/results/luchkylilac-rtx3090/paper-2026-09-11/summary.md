# Paper benchmark summary

## Microbenchmarks (median steady_us over rounds)

| variant | k | n | fused (ours) | app-level (ours) | eager (ours) | nojit (ours) | torch.compile | compile-ro | compile-mat | torch eager | JAX/XLA | IREE | Triton | speedup vs compile | interp tax | ours/Triton |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1 | 1000000 | 28.7 | n/a | 76.7 | 76.8 | 44.2 | n/a | n/a | 78.1 | 44.1 | 5937.7 | 30.6 | 1.54x | n/a | 0.94x |
| 0 | 4 | 10000 | 11.9 | n/a | 38.7 | 39.3 | 42.2 | n/a | n/a | 58.5 | 43.5 | 732.7 | 19.4 | 3.54x | n/a | 0.61x |
| 0 | 4 | 100000 | 12.8 | n/a | 43.3 | 43.0 | 40.0 | n/a | n/a | 60.8 | 44.2 | 1350.2 | 19.0 | 3.13x | n/a | 0.67x |
| 0 | 4 | 1000000 | 39.1 | n/a | 306.9 | 306.5 | 68.4 | n/a | n/a | 312.3 | 88.1 | 6747.4 | 38.5 | 1.75x | n/a | 1.02x |
| 0 | 4 | 10000000 | 313.7 | n/a | 3026.8 | 3026.9 | 645.5 | n/a | n/a | 3044.7 | 770.5 | 60619.1 | 327.2 | 2.06x | n/a | 0.96x |
| 0 | 8 | 1000000 | 67.3 | n/a | 612.1 | 612.5 | 132.7 | n/a | n/a | 624.5 | 160.1 | 6635.1 | 66.8 | 1.97x | n/a | 1.01x |
| 1 | 4 | 1000000 | 41.6 | n/a | 311.5 | 310.7 | 306.9 | n/a | n/a | 316.7 | 92.8 | 9597.1 | 42.9 | 7.38x | n/a | 0.97x |
| 2 | 4 | 1000000 | 44.0 | n/a | 311.5 | 310.7 | 306.8 | n/a | n/a | 316.6 | 92.8 | 9308.6 | 43.0 | 6.96x | n/a | 1.03x |
| 3 | 4 | 1000000 | 99.5 | n/a | 357.5 | 365.9 | 166.9 | n/a | n/a | 376.6 | 272.2 | 132191.7 | 120.2 | 1.68x | n/a | 0.83x |
| 4 | 4 | 1000000 | 53.3 | n/a | 338.8 | 335.6 | 331.6 | n/a | n/a | 342.2 | 120.3 | 12739.1 | 68.6 | 6.23x | n/a | 0.78x |
| 5 | 4 | 1000000 | 56.1 | n/a | 336.3 | 335.7 | 331.7 | n/a | n/a | 342.2 | 115.5 | 12770.2 | 68.8 | 5.91x | n/a | 0.82x |
| 6 | 1 | 25600 | 277.2 | n/a | 279.8 | 277.3 | 387.0 | n/a | n/a | 392.5 | 107.7 | 11854.1 | 881.0 | 1.40x | n/a | 0.31x |
| 6 | 1 | 256000 | 1011.4 | n/a | 1015.7 | 1011.7 | 1234.0 | n/a | n/a | 1270.9 | 758.4 | 3041.5 | 850.0 | 1.22x | n/a | 1.19x |
| 7 | 1 | 25600 | 646.9 | n/a | 668.0 | 667.3 | 840.6 | n/a | n/a | 800.6 | 331.6 | 22914.2 | n/a | 1.30x | n/a | n/a |
| 7 | 1 | 256000 | 2798.7 | n/a | 2807.2 | 2822.0 | 3045.7 | n/a | n/a | 3077.3 | 1970.7 | 31739.5 | n/a | 1.09x | n/a | n/a |
| 8 | 1 | 25600 | 464.5 | n/a | 778.7 | 779.2 | 516.3 | 525.1 | 539.9 | 557.3 | 267.1 | 889.9 | n/a | 1.11x | n/a | n/a |
| 8 | 1 | 256000 | 22300.3 | n/a | 24837.3 | 25048.2 | 29624.8 | 29544.2 | 27366.9 | 27209.3 | 15363.0 | 53174.5 | n/a | 1.33x | n/a | n/a |
| 9 | 1 | 25600 | 756.8 | n/a | 764.8 | 764.9 | 860.7 | n/a | n/a | 801.9 | 237.8 | n/a | n/a | 1.14x | n/a | n/a |
| 9 | 1 | 256000 | 572.2 | n/a | 572.8 | 568.2 | 664.0 | n/a | n/a | 572.8 | 329.9 | n/a | n/a | 1.16x | n/a | n/a |
| 10 | 1 | 25600 | 3905.1 | n/a | 4810.8 | 4813.4 | 3178.4 | n/a | n/a | 3729.5 | 1527.1 | 10461.8 | n/a | 0.81x | n/a | n/a |
| 10 | 1 | 191488 | 93902.3 | n/a | 99340.3 | 99362.3 | 78668.4 | n/a | n/a | 78944.0 | 41813.3 | n/a | n/a | 0.84x | n/a | n/a |
| 10 | 1 | 256000 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 411364.0 | n/a | n/a | n/a | n/a |
| 11 | 1 | 25600 | 3.3 | n/a | 20.5 | 20.9 | 60.0 | n/a | n/a | 25.7 | 38.9 | 74.9 | 18.6 | 18.16x | n/a | 0.18x |
| 11 | 1 | 256000 | 15.1 | n/a | 113.4 | 113.4 | 160.3 | n/a | n/a | 122.9 | 43.1 | 187.0 | 18.8 | 10.61x | n/a | 0.81x |
| 12 | 1 | 25600 | 98.8 | n/a | 98.9 | 97.9 | 142.6 | n/a | n/a | 137.4 | 47.2 | 2085.8 | 299.2 | 1.44x | n/a | 0.33x |
| 12 | 1 | 256000 | 340.9 | n/a | 341.1 | 337.1 | 481.6 | n/a | n/a | 491.7 | 256.8 | 560.2 | 299.6 | 1.41x | n/a | 1.14x |
| 13 | 1 | 25600 | 441.1 | n/a | 523.7 | 523.8 | 598.0 | 620.4 | 612.4 | 593.0 | 123.7 | 7168.2 | 1232.8 | 1.36x | n/a | 0.36x |
| 13 | 1 | 256000 | 4750.1 | n/a | 5428.2 | 5423.3 | 5024.3 | 5043.8 | 5000.9 | 4881.9 | 3739.5 | 5266.1 | 4679.0 | 1.06x | n/a | 1.02x |

## Precision sweep (median steady_us)

| dtype | variant | k | n | fused (ours) | app-level (ours) | eager (ours) | nojit (ours) | torch.compile | compile-ro | compile-mat | torch eager | JAX/XLA | IREE | Triton | speedup vs compile | interp tax | ours/Triton |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| float16 | 8 | 1 | 256000 | 711.9 | n/a | 2054.8 | 2053.9 | 860.0 | n/a | n/a | 1461.9 | 628.8 | 2163.3 | n/a | 1.21x | n/a | n/a |
| float16 | 13 | 1 | 256000 | 221.4 | n/a | 349.1 | 341.7 | 430.0 | n/a | n/a | 410.7 | 96.4 | 584.1 | 132.0 | 1.94x | n/a | 1.68x |
| float32 | 8 | 1 | 256000 | 1395.8 | n/a | 3919.6 | 3919.1 | 2017.5 | n/a | n/a | 2652.5 | 1256.4 | 10512.4 | n/a | 1.45x | n/a | n/a |
| float32 | 13 | 1 | 256000 | 272.5 | n/a | 588.0 | 580.9 | 544.5 | n/a | n/a | 636.8 | 214.8 | 1398.7 | 131.1 | 2.00x | n/a | 2.08x |

## Guards / graph breaks (launches per iter, torch graph breaks)

| variant | n | launches/iter (fused) | torch graphs | torch breaks |
|---|---|---|---|---|
| 1 | 1000000 | 1.1 | 1 | 0 |
| 2 | 1000000 | 1.1 | 1 | 0 |
| 3 | 1000000 | 2.0 | 2 | 1 |
| 4 | 1000000 | 1.0 | 1 | 0 |
| 5 | 1000000 | 1.1 | 2 | 1 |

## End-to-end models (median steady_us, ratio to torch.compile)

| model | ours | torch.compile | compile-ro | compile-mat | torch eager | JAX/XLA | IREE | TensorRT | ratio ours/compile | ratio jax/compile | correctness |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bert-mini | 428.6 | 921.5 | 481.1 | 413.1 | 1804.7 | 242.6 | 3738.2 | 659.5 | 0.47x | 0.26x | 1.90735e-05 |
| bert-tiny | 240.8 | 489.7 | 295.0 | 262.8 | 1090.2 | 111.1 | 1342.8 | 431.9 | 0.49x | 0.23x | 3.05176e-05 |
| distilgpt2 | 1278.7 | 1350.6 | 1321.6 | 1286.4 | 2761.1 | 1014.6 | 17358.9 | 3739.8 | 0.95x | 0.75x | 0.000175476 |
| mixer_b16 | 2672.0 | 3114.9 | 3049.1 | 3124.0 | 2876.9 | 2238.2 | 1436947.2 | 2461.8 | 0.86x | 0.72x | 3.8147e-05 |
| resnet18-b1 | 708.5 | 960.8 | 916.0 | 1055.7 | 1580.7 | 1121.8 | n/a | 989.7 | 0.74x | 1.17x | 0.00963783 |
| resnet18-b8 | 2705.3 | 2188.9 | 2123.8 | 2135.2 | 2269.6 | 2273.1 | n/a | 2013.1 | 1.24x | 1.04x | 0.00571394 |
| smollm2-135m | 3660.7 | 5206.8 | 3778.4 | 3810.4 | 15713.2 | 2651.1 | 19097.1 | 3792.3 | 0.70x | 0.51x | 0.000151277 |
| tiny-gpt2 | 194.2 | 409.4 | 204.0 | 246.5 | 1630.6 | 80.8 | 190.6 | 1384.8 | 0.47x | 0.20x | 2.6077e-08 |
| vit-tiny | 974.8 | 2056.3 | 1414.5 | 1348.8 | 3826.2 | 746.0 | 36791.8 | 1034.4 | 0.47x | 0.36x | 1.07288e-05 |

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
| bert-mini | torch-compile-ro | n/a | 1293.6 | 481.1 | 914 |
| bert-mini | torch-compile-mat | n/a | 1388.0 | 413.1 | 937 |
| bert-mini | torch-tensorrt | n/a | 4856.2 | 659.5 | 4167 |
| bert-mini | jax | 7061.4 | 2.7 | 242.6 | 4468 |
| bert-mini | iree | 1032.2 | 19.4 | 3738.2 | n/a |
| bert-mini | ours | n/a | 194.6 | 428.6 | 80 |
| bert-tiny | torch-compile | n/a | 1117.5 | 489.7 | 1700 |
| bert-tiny | torch-compile-ro | n/a | 1127.6 | 295.0 | 1296 |
| bert-tiny | torch-compile-mat | n/a | 1245.2 | 262.8 | 1388 |
| bert-tiny | torch-tensorrt | n/a | 4212.0 | 431.9 | 6251 |
| bert-tiny | jax | 5870.0 | 1.8 | 111.1 | 5898 |
| bert-tiny | iree | 862.5 | 13.7 | 1342.8 | n/a |
| bert-tiny | ours | n/a | 115.7 | 240.8 | 22 |
| distilgpt2 | torch-compile | n/a | 1557.6 | 1350.6 | 1013 |
| distilgpt2 | torch-compile-ro | n/a | 1535.0 | 1321.6 | 977 |
| distilgpt2 | torch-compile-mat | n/a | 1630.3 | 1286.4 | 1018 |
| distilgpt2 | torch-tensorrt | n/a | 2856.2 | 3739.8 | n/a |
| distilgpt2 | jax | 6912.1 | 4.6 | 1014.6 | 3887 |
| distilgpt2 | iree | 906.7 | 76.2 | 17358.9 | n/a |
| distilgpt2 | ours | n/a | 534.8 | 1278.7 | 274 |
| micro v0 k1 n1000000 | torch-compile | n/a | 481.8 | 44.2 | 13120 |
| micro v0 k1 n1000000 | torch-tensorrt | n/a | 37.9 | 80.4 | n/a |
| micro v0 k1 n1000000 | jax | 200.0 | 0.5 | 44.1 | 4793 |
| micro v0 k1 n1000000 | iree | 579.5 | 22.9 | 5937.7 | n/a |
| micro v0 k1 n1000000 | triton | 307.9 | 0.1 | 30.6 | 5699 |
| micro v0 k4 n10000 | torch-compile | n/a | 469.3 | 42.2 | 26723 |
| micro v0 k4 n10000 | torch-tensorrt | n/a | 40.2 | 59.0 | n/a |
| micro v0 k4 n10000 | jax | 197.6 | 0.5 | 43.5 | 10899 |
| micro v0 k4 n10000 | iree | 584.9 | 10.1 | 732.7 | n/a |
| micro v0 k4 n10000 | triton | 308.1 | 0.1 | 19.4 | 6996 |
| micro v0 k4 n100000 | torch-compile | n/a | 486.1 | 40.0 | 21578 |
| micro v0 k4 n100000 | torch-tensorrt | n/a | 40.6 | 61.1 | n/a |
| micro v0 k4 n100000 | jax | 188.4 | 0.5 | 44.2 | 9144 |
| micro v0 k4 n100000 | iree | 586.1 | 11.2 | 1350.2 | n/a |
| micro v0 k4 n100000 | triton | 307.5 | 0.1 | 19.0 | 6461 |
| micro v0 k4 n1000000 | torch-compile | n/a | 420.2 | 68.4 | 1578 |
| micro v0 k4 n1000000 | torch-tensorrt | n/a | 37.7 | 321.4 | n/a |
| micro v0 k4 n1000000 | jax | 202.8 | 0.6 | 88.1 | 749 |
| micro v0 k4 n1000000 | iree | 587.4 | 25.3 | 6747.4 | n/a |
| micro v0 k4 n1000000 | triton | 307.9 | 0.1 | 38.5 | 996 |
| micro v0 k4 n10000000 | torch-compile | n/a | 449.9 | 645.5 | 173 |
| micro v0 k4 n10000000 | torch-tensorrt | n/a | 40.0 | 3045.5 | n/a |
| micro v0 k4 n10000000 | jax | 218.5 | 1.3 | 770.5 | 81 |
| micro v0 k4 n10000000 | iree | 611.0 | 190.4 | 60619.1 | n/a |
| micro v0 k4 n10000000 | triton | 277.0 | 0.6 | 327.2 | 89 |
| micro v0 k8 n1000000 | torch-compile | n/a | 427.3 | 132.7 | 796 |
| micro v0 k8 n1000000 | torch-tensorrt | n/a | 40.8 | 642.4 | n/a |
| micro v0 k8 n1000000 | jax | 214.0 | 0.6 | 160.1 | 386 |
| micro v0 k8 n1000000 | iree | 580.5 | 26.6 | 6635.1 | n/a |
| micro v0 k8 n1000000 | triton | 308.2 | 0.2 | 66.8 | 489 |
| micro v1 k4 n1000000 | torch-compile | n/a | 490.5 | 306.9 | 46213 |
| micro v1 k4 n1000000 | torch-tensorrt | n/a | 37.8 | 325.8 | n/a |
| micro v1 k4 n1000000 | jax | 202.2 | 0.6 | 92.8 | 736 |
| micro v1 k4 n1000000 | iree | 602.9 | 27.2 | 9597.1 | n/a |
| micro v1 k4 n1000000 | triton | 309.3 | 0.1 | 42.9 | 991 |
| micro v10 k1 n191488 | torch-compile | n/a | 1877.9 | 78668.4 | 5450 |
| micro v10 k1 n191488 | torch-tensorrt | n/a | 2455.3 | 79037.6 | n/a |
| micro v10 k1 n191488 | jax | 4927.0 | 71.9 | 41813.3 | 125 |
| micro v10 k1 n25600 | torch-compile | n/a | 1784.2 | 3178.4 | 2696 |
| micro v10 k1 n25600 | torch-tensorrt | n/a | 2381.4 | 4112.4 | n/a |
| micro v10 k1 n25600 | jax | 4229.6 | 6.6 | 1527.1 | 1788 |
| micro v10 k1 n25600 | iree | 1577.7 | 26.2 | 10461.8 | n/a |
| micro v10 k1 n256000 | iree | 22024.0 | 533.5 | 411364.0 | n/a |
| micro v11 k1 n25600 | torch-compile | n/a | 1680.4 | 60.0 | n/a |
| micro v11 k1 n25600 | torch-tensorrt | n/a | 2034.0 | 58.0 | n/a |
| micro v11 k1 n25600 | jax | 54.3 | 0.4 | 38.9 | n/a |
| micro v11 k1 n25600 | iree | 179.9 | 11.4 | 74.9 | n/a |
| micro v11 k1 n25600 | triton | 307.0 | 0.1 | 18.6 | 18615 |
| micro v11 k1 n256000 | torch-compile | n/a | 1711.2 | 160.3 | n/a |
| micro v11 k1 n256000 | torch-tensorrt | n/a | 2022.6 | 151.5 | n/a |
| micro v11 k1 n256000 | jax | 52.6 | 0.4 | 43.1 | -1859 |
| micro v11 k1 n256000 | iree | 175.9 | 12.1 | 187.0 | n/a |
| micro v11 k1 n256000 | triton | 307.0 | 0.1 | 18.8 | 1013 |
| micro v12 k1 n25600 | torch-compile | n/a | 1754.9 | 142.6 | n/a |
| micro v12 k1 n25600 | torch-tensorrt | n/a | 2025.7 | 134.5 | 618692 |
| micro v12 k1 n25600 | jax | 1520.4 | 0.8 | 47.2 | 14495 |
| micro v12 k1 n25600 | iree | 231.4 | 15.6 | 2085.8 | n/a |
| micro v12 k1 n25600 | triton | 308.4 | 0.4 | 299.2 | n/a |
| micro v12 k1 n256000 | torch-compile | n/a | 1767.7 | 481.6 | 152196 |
| micro v12 k1 n256000 | torch-tensorrt | n/a | 2019.1 | 482.9 | 203850 |
| micro v12 k1 n256000 | jax | 1569.6 | 0.9 | 256.8 | 5704 |
| micro v12 k1 n256000 | iree | 240.7 | 10.2 | 560.2 | n/a |
| micro v12 k1 n256000 | triton | 308.5 | 0.4 | 299.6 | 409 |
| micro v13 k1 n25600 | torch-compile | n/a | 1802.6 | 598.0 | n/a |
| micro v13 k1 n25600 | torch-compile-ro | n/a | 1725.5 | 620.4 | n/a |
| micro v13 k1 n25600 | torch-compile-mat | n/a | 1800.9 | 612.4 | n/a |
| micro v13 k1 n25600 | torch-tensorrt | n/a | 2147.5 | 580.5 | 151841 |
| micro v13 k1 n25600 | jax | 1956.8 | 1.0 | 123.7 | 3637 |
| micro v13 k1 n25600 | iree | 325.1 | 14.8 | 7168.2 | n/a |
| micro v13 k1 n25600 | triton | 311.5 | 1.3 | 1232.8 | n/a |
| micro v13 k1 n256000 | torch-compile | n/a | 1826.8 | 5024.3 | n/a |
| micro v13 k1 n256000 | torch-compile-ro | n/a | 1843.2 | 5043.8 | n/a |
| micro v13 k1 n256000 | torch-compile-mat | n/a | 1697.5 | 5000.9 | n/a |
| micro v13 k1 n256000 | torch-tensorrt | n/a | 2070.4 | 4868.6 | 134859 |
| micro v13 k1 n256000 | jax | 1980.7 | 4.8 | 3739.5 | 1503 |
| micro v13 k1 n256000 | iree | 636.6 | 19.7 | 5266.1 | n/a |
| micro v13 k1 n256000 | triton | 315.2 | 4.7 | 4679.0 | 253 |
| micro v2 k4 n1000000 | torch-compile | n/a | 419.2 | 306.8 | 38874 |
| micro v2 k4 n1000000 | torch-tensorrt | n/a | 37.6 | 325.8 | n/a |
| micro v2 k4 n1000000 | jax | 196.7 | 0.5 | 92.8 | 723 |
| micro v2 k4 n1000000 | iree | 549.7 | 29.5 | 9308.6 | n/a |
| micro v2 k4 n1000000 | triton | 308.9 | 0.1 | 43.0 | 1000 |
| micro v3 k4 n1000000 | torch-compile | n/a | 562.7 | 166.9 | 2503 |
| micro v3 k4 n1000000 | torch-tensorrt | n/a | 38.5 | 384.1 | n/a |
| micro v3 k4 n1000000 | jax | 204.7 | 0.6 | 272.2 | 1605 |
| micro v3 k4 n1000000 | iree | 607.3 | 31.5 | 132191.7 | n/a |
| micro v3 k4 n1000000 | triton | 310.3 | 0.1 | 120.2 | 1063 |
| micro v4 k4 n1000000 | torch-compile | n/a | 430.7 | 331.6 | 36997 |
| micro v4 k4 n1000000 | torch-tensorrt | n/a | 37.3 | 351.7 | n/a |
| micro v4 k4 n1000000 | jax | 197.9 | 0.6 | 120.3 | 725 |
| micro v4 k4 n1000000 | iree | 611.8 | 25.8 | 12739.1 | n/a |
| micro v4 k4 n1000000 | triton | 309.2 | 0.1 | 68.6 | 992 |
| micro v5 k4 n1000000 | torch-compile | n/a | 506.8 | 331.7 | 44932 |
| micro v5 k4 n1000000 | torch-tensorrt | n/a | 37.8 | 351.4 | n/a |
| micro v5 k4 n1000000 | jax | 205.5 | 0.6 | 115.5 | 741 |
| micro v5 k4 n1000000 | iree | 583.6 | 24.6 | 12770.2 | n/a |
| micro v5 k4 n1000000 | triton | 308.4 | 0.1 | 68.8 | 990 |
| micro v6 k1 n25600 | torch-compile | n/a | 1795.2 | 387.0 | 287626 |
| micro v6 k1 n25600 | torch-tensorrt | n/a | 2007.5 | 380.3 | 146864 |
| micro v6 k1 n25600 | jax | 1579.6 | 1.0 | 107.7 | 4762 |
| micro v6 k1 n25600 | iree | 224.7 | 23.8 | 11854.1 | n/a |
| micro v6 k1 n25600 | triton | 308.7 | 0.9 | 881.0 | n/a |
| micro v6 k1 n256000 | torch-compile | n/a | 1779.3 | 1234.0 | 41564 |
| micro v6 k1 n256000 | torch-tensorrt | n/a | 2140.6 | 1271.3 | n/a |
| micro v6 k1 n256000 | jax | 1595.4 | 1.6 | 758.4 | 2634 |
| micro v6 k1 n256000 | iree | 236.8 | 10.7 | 3041.5 | n/a |
| micro v6 k1 n256000 | triton | 306.1 | 0.9 | 850.0 | 143 |
| micro v7 k1 n25600 | torch-compile | n/a | 1807.6 | 840.6 | n/a |
| micro v7 k1 n25600 | torch-tensorrt | n/a | 2114.5 | 820.7 | n/a |
| micro v7 k1 n25600 | jax | 2149.9 | 1.6 | 331.6 | 4066 |
| micro v7 k1 n25600 | iree | 323.2 | 40.3 | 22914.2 | n/a |
| micro v7 k1 n256000 | torch-compile | n/a | 1741.6 | 3045.7 | 47174 |
| micro v7 k1 n256000 | torch-tensorrt | n/a | 2199.7 | 3089.2 | n/a |
| micro v7 k1 n256000 | jax | 2280.0 | 3.8 | 1970.7 | 1834 |
| micro v7 k1 n256000 | iree | 435.4 | 42.3 | 31739.5 | n/a |
| micro v8 k1 n25600 | torch-compile | n/a | 1785.2 | 516.3 | 36888 |
| micro v8 k1 n25600 | torch-compile-ro | n/a | 1804.8 | 525.1 | 47605 |
| micro v8 k1 n25600 | torch-compile-mat | n/a | 1865.5 | 539.9 | 91442 |
| micro v8 k1 n25600 | torch-tensorrt | n/a | 2190.8 | 563.5 | n/a |
| micro v8 k1 n25600 | jax | 1262.8 | 1.3 | 267.1 | 3413 |
| micro v8 k1 n25600 | iree | 532.9 | 9.9 | 889.9 | n/a |
| micro v8 k1 n256000 | torch-compile | n/a | 1792.2 | 29624.8 | n/a |
| micro v8 k1 n256000 | torch-compile-ro | n/a | 1923.8 | 29544.2 | n/a |
| micro v8 k1 n256000 | torch-compile-mat | n/a | 1885.6 | 27366.9 | n/a |
| micro v8 k1 n256000 | torch-tensorrt | n/a | 2256.4 | 27305.4 | n/a |
| micro v8 k1 n256000 | jax | 1530.6 | 16.6 | 15363.0 | 105 |
| micro v8 k1 n256000 | iree | 20835.2 | 116.0 | 53174.5 | n/a |
| micro v9 k1 n25600 | torch-compile | n/a | 1789.5 | 860.7 | n/a |
| micro v9 k1 n25600 | torch-tensorrt | n/a | 2136.3 | 942.9 | n/a |
| micro v9 k1 n25600 | jax | 363.0 | 0.8 | 237.8 | 199 |
| micro v9 k1 n256000 | torch-compile | n/a | 1704.2 | 664.0 | n/a |
| micro v9 k1 n256000 | torch-tensorrt | n/a | 2151.9 | 1549.6 | n/a |
| micro v9 k1 n256000 | jax | 802.1 | 1.2 | 329.9 | 2219 |
| mixer_b16 | torch-compile | n/a | 1243.6 | 3114.9 | n/a |
| mixer_b16 | torch-compile-ro | n/a | 1306.9 | 3049.1 | n/a |
| mixer_b16 | torch-compile-mat | n/a | 1400.8 | 3124.0 | n/a |
| mixer_b16 | torch-tensorrt | n/a | 5985.3 | 2461.8 | 14117 |
| mixer_b16 | jax | 7200.5 | 7.0 | 2238.2 | 11088 |
| mixer_b16 | iree | 1482.3 | 1660.3 | 1436947.2 | n/a |
| mixer_b16 | ours | n/a | 181.1 | 2672.0 | 272 |
| resnet18-b1 | torch-compile | n/a | 1018.4 | 960.8 | 1336 |
| resnet18-b1 | torch-compile-ro | n/a | 1074.6 | 916.0 | 1330 |
| resnet18-b1 | torch-compile-mat | n/a | 1730.7 | 1055.7 | 2934 |
| resnet18-b1 | torch-tensorrt | n/a | 4643.6 | 989.7 | 7535 |
| resnet18-b1 | jax | 587.8 | 4.0 | 1121.8 | 875 |
| resnet18-b1 | ours | n/a | 124.6 | 708.5 | -75 |
| resnet18-b8 | torch-compile | n/a | 1078.6 | 2188.9 | 11048 |
| resnet18-b8 | torch-compile-ro | n/a | 1080.7 | 2123.8 | 6130 |
| resnet18-b8 | torch-compile-mat | n/a | 1518.5 | 2135.2 | 9907 |
| resnet18-b8 | torch-tensorrt | n/a | 4813.8 | 2013.1 | 18038 |
| resnet18-b8 | jax | 934.8 | 4.4 | 2273.1 | n/a |
| resnet18-b8 | ours | n/a | 153.3 | 2705.3 | n/a |
| smollm2-135m | torch-compile | n/a | 3863.3 | 5206.8 | 316 |
| smollm2-135m | torch-compile-ro | n/a | 4015.2 | 3778.4 | 291 |
| smollm2-135m | torch-compile-mat | n/a | 4321.8 | 3810.4 | 317 |
| smollm2-135m | torch-tensorrt | n/a | 21935.5 | 3792.3 | 1794 |
| smollm2-135m | jax | 10616.3 | 17.7 | 2651.1 | 772 |
| smollm2-135m | iree | 2050.3 | 172.6 | 19097.1 | n/a |
| smollm2-135m | ours | n/a | 495.4 | 3660.7 | -4 |
| tiny-gpt2 | torch-compile | n/a | 1178.7 | 409.4 | 577 |
| tiny-gpt2 | torch-compile-ro | n/a | 1194.8 | 204.0 | 505 |
| tiny-gpt2 | torch-compile-mat | n/a | 1217.1 | 246.5 | 537 |
| tiny-gpt2 | torch-tensorrt | n/a | 1843.5 | 1384.8 | 5571 |
| tiny-gpt2 | jax | 1142.1 | 1.5 | 80.8 | 432 |
| tiny-gpt2 | iree | 400.5 | 11.7 | 190.6 | -43 |
| tiny-gpt2 | ours | n/a | 114.5 | 194.2 | -250 |
| vit-tiny | torch-compile | n/a | 1778.9 | 2056.3 | 938 |
| vit-tiny | torch-compile-ro | n/a | 1796.1 | 1414.5 | 695 |
| vit-tiny | torch-compile-mat | n/a | 1999.8 | 1348.8 | 759 |
| vit-tiny | torch-tensorrt | n/a | 7641.6 | 1034.4 | 2695 |
| vit-tiny | jax | 6705.4 | 7.7 | 746.0 | 2141 |
| vit-tiny | iree | 1697.7 | 71.1 | 36791.8 | n/a |
| vit-tiny | ours | n/a | 143.5 | 974.8 | 9 |

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

