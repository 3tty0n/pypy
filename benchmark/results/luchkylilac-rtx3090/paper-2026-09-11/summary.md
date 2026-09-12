# Paper benchmark summary

## Microbenchmarks (median steady_us over rounds)

| variant | k | n | fused (ours) | app-level (ours) | eager (ours) | nojit (ours) | torch.compile | compile-ro | compile-mat | torch eager | JAX/XLA | IREE | Triton | speedup vs compile | interp tax | ours/Triton |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1 | 1000000 | 29.1 | 29.1 | 77.2 | 77.3 | 50.8 | 118.8 | 119.8 | 78.2 | 48.8 | 5707.4 | 30.7 | 1.75x | 1.00x | 0.95x |
| 0 | 4 | 10000 | 13.1 | 18.9 | 39.9 | 39.7 | 51.9 | 107.4 | 116.8 | 54.8 | 42.8 | 436.9 | 20.2 | 3.96x | 1.44x | 0.65x |
| 0 | 4 | 100000 | 13.8 | 20.7 | 43.3 | 43.2 | 52.2 | 118.6 | 117.5 | 61.4 | 44.6 | 927.8 | 19.5 | 3.79x | 1.50x | 0.71x |
| 0 | 4 | 1000000 | 40.2 | 116.6 | 306.7 | 306.9 | 68.4 | 134.2 | 133.1 | 312.4 | 94.8 | 5759.6 | 39.1 | 1.70x | 2.90x | 1.03x |
| 0 | 4 | 10000000 | 329.1 | 1133.3 | 3027.1 | 3027.4 | 646.0 | 1215.9 | 1196.2 | 3045.8 | 945.0 | 51069.0 | 327.1 | 1.96x | 3.44x | 1.01x |
| 0 | 8 | 1000000 | 70.1 | 229.5 | 612.9 | 613.5 | 132.7 | 198.4 | 198.5 | 624.5 | 175.6 | 5792.7 | 67.3 | 1.89x | 3.27x | 1.04x |
| 1 | 4 | 1000000 | 41.6 | 119.2 | 312.2 | 311.0 | 307.7 | 401.1 | 396.2 | 316.8 | 100.1 | 6582.9 | 43.4 | 7.40x | 2.86x | 0.96x |
| 2 | 4 | 1000000 | 42.6 | 119.1 | 310.9 | 311.0 | 72.8 | 138.4 | 137.1 | 326.0 | 175.7 | 8925.4 | 69.7 | 1.71x | 2.79x | 0.61x |
| 3 | 4 | 1000000 | 98.9 | 177.6 | 361.3 | 366.1 | 176.0 | 394.6 | 340.3 | 377.5 | 317.3 | 81668.8 | 123.4 | 1.78x | 1.80x | 0.80x |
| 4 | 4 | 1000000 | 53.3 | 144.3 | 336.1 | 335.7 | 332.1 | 413.1 | 422.2 | 342.3 | 127.5 | 10801.5 | 69.2 | 6.23x | 2.71x | 0.77x |
| 5 | 4 | 1000000 | 53.5 | 145.8 | 335.7 | 335.7 | 332.6 | 424.4 | 422.5 | 342.2 | 126.9 | 10817.8 | 69.4 | 6.21x | 2.72x | 0.77x |
| 6 | 1 | 25600 | 279.4 | 268.2 | 279.0 | 278.5 | 381.9 | 428.2 | 429.2 | 393.8 | 107.2 | 5383.7 | 894.2 | 1.37x | 0.96x | 0.31x |
| 6 | 1 | 256000 | 1016.3 | 999.7 | 1016.9 | 1016.0 | 1271.2 | 1271.5 | 1307.1 | 1238.4 | 759.8 | 912.4 | 850.2 | 1.25x | 0.98x | 1.20x |
| 7 | 1 | 25600 | 648.4 | 786.4 | 668.9 | 668.8 | 851.5 | 868.9 | 865.7 | 800.6 | 324.8 | 11516.3 | n/a | 1.31x | 1.21x | n/a |
| 7 | 1 | 256000 | 2808.1 | 2934.3 | 2833.5 | 2829.6 | 3078.6 | 3041.4 | 3040.0 | 3066.8 | 1965.9 | 31543.7 | n/a | 1.10x | 1.04x | n/a |
| 8 | 1 | 25600 | 464.3 | 505.9 | 780.2 | 780.4 | 518.9 | 523.4 | 522.6 | 558.4 | 266.9 | 934.8 | n/a | 1.12x | 1.09x | n/a |
| 8 | 1 | 256000 | 22340.7 | 22355.2 | 25126.6 | 25204.9 | 29873.8 | 29700.5 | 27718.5 | 27474.3 | 15450.5 | 38544.8 | n/a | 1.34x | 1.00x | n/a |
| 9 | 1 | 25600 | 754.2 | 749.2 | 763.6 | 764.3 | 856.6 | 867.8 | 867.7 | 797.7 | 243.5 | n/a | n/a | 1.14x | 0.99x | n/a |
| 9 | 1 | 256000 | 573.2 | 556.5 | 599.7 | 568.0 | 652.7 | 699.7 | 656.3 | 584.1 | 333.7 | n/a | n/a | 1.14x | 0.97x | n/a |
| 10 | 1 | 25600 | 3900.5 | 4981.9 | 4825.0 | 4808.4 | 3154.1 | 3098.1 | 3123.5 | 3734.5 | 1503.0 | 10442.1 | n/a | 0.81x | 1.28x | n/a |
| 10 | 1 | 191488 | 94046.7 | 97463.8 | 99361.2 | 99386.4 | 78669.0 | 78689.1 | 75791.9 | 78944.2 | 41818.5 | 136415.3 | n/a | 0.84x | 1.04x | n/a |
| 11 | 1 | 25600 | 3.6 | 20.2 | 56.7 | 57.0 | 56.5 | 123.0 | 112.3 | 57.1 | 48.7 | 56.6 | 18.0 | 15.50x | 5.54x | 0.20x |
| 11 | 1 | 256000 | 16.0 | 163.9 | 143.4 | 141.0 | 180.4 | 225.7 | 244.9 | 151.5 | 54.0 | 185.8 | 20.0 | 11.29x | 10.26x | 0.80x |
| 12 | 1 | 25600 | 98.7 | 90.2 | 98.5 | 98.4 | 138.7 | 177.6 | 177.6 | 137.0 | 42.2 | 1865.6 | 298.9 | 1.41x | 0.91x | 0.33x |
| 12 | 1 | 256000 | 343.3 | 352.4 | 342.6 | 342.5 | 510.8 | 548.8 | 522.4 | 495.4 | 258.4 | 437.1 | 290.3 | 1.49x | 1.03x | 1.18x |
| 13 | 1 | 25600 | 437.3 | 427.4 | 526.7 | 526.7 | 614.2 | 611.5 | 640.0 | 580.7 | 123.3 | 2847.7 | 1290.1 | 1.40x | 0.98x | 0.34x |
| 13 | 1 | 256000 | 4742.2 | 4755.0 | 5423.5 | 5427.6 | 5019.8 | 5003.0 | 5054.9 | 4874.6 | 3700.5 | 5127.8 | 4681.8 | 1.06x | 1.00x | 1.01x |

## Precision sweep (median steady_us)

| dtype | variant | k | n | fused (ours) | app-level (ours) | eager (ours) | nojit (ours) | torch.compile | compile-ro | compile-mat | torch eager | JAX/XLA | IREE | Triton | speedup vs compile | interp tax | ours/Triton |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| float16 | 8 | 1 | 256000 | 709.1 | n/a | 2053.9 | 2059.4 | 867.2 | 891.9 | 807.6 | 1471.4 | 633.3 | 1884.0 | n/a | 1.22x | n/a | n/a |
| float16 | 11 | 1 | 256000 | 22.4 | n/a | 59.4 | 55.6 | 154.6 | 206.8 | 204.1 | 147.6 | 43.6 | 181.5 | 19.0 | 6.90x | n/a | 1.18x |
| float16 | 13 | 1 | 256000 | 211.6 | n/a | 333.9 | 336.7 | 395.3 | 389.1 | 359.8 | 412.5 | 100.6 | 600.4 | 134.5 | 1.87x | n/a | 1.57x |
| float32 | 8 | 1 | 256000 | 1393.7 | n/a | 3913.4 | 3916.9 | 2014.6 | 2030.7 | 1474.0 | 2650.8 | 1258.2 | 7302.9 | n/a | 1.45x | n/a | n/a |
| float32 | 11 | 1 | 256000 | 6.8 | n/a | 42.4 | 37.7 | 151.9 | 213.1 | 210.1 | 144.3 | 42.4 | 80.4 | 18.9 | 22.37x | n/a | 0.36x |
| float32 | 13 | 1 | 256000 | 271.2 | n/a | 588.4 | 583.3 | 561.7 | 617.7 | 512.2 | 635.2 | 215.8 | 1315.0 | 136.3 | 2.07x | n/a | 1.99x |

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
| bert-mini | 429.2 | 861.6 | 480.5 | 423.7 | 1808.4 | 243.3 | 3611.3 | 655.7 | 0.50x | 0.28x | 24.3 | 1.90735e-05 | 0.001 | pass |
| bert-tiny | 253.4 | 539.0 | 299.0 | 269.9 | 1103.2 | 111.3 | 1322.3 | 443.1 | 0.47x | 0.21x | 14.3 | 3.05176e-05 | 0.001 | pass |
| distilgpt2 | 1284.5 | 1352.7 | 1290.0 | 1280.2 | 2779.3 | 1000.6 | 17122.7 | 3722.1 | 0.95x | 0.74x | 38.2 | 0.000190735 | 0.001 | pass |
| mixer_b16 | 2711.7 | 3143.5 | 3062.6 | 3119.0 | 2892.0 | 2249.2 | 673104.7 | 2486.1 | 0.86x | 0.72x | 63.4 | 3.62396e-05 | 0.001 | pass |
| resnet18-b1 | 726.3 | 1037.0 | 918.4 | 1041.3 | 1417.3 | 1097.1 | n/a | 995.3 | 0.70x | 1.06x | 39.0 | 0.00963783 | 0.02 | pass |
| resnet18-b8 | 2715.4 | 2193.9 | 2123.9 | 2126.3 | 2257.0 | 2275.9 | n/a | 2025.2 | 1.24x | 1.04x | 39.0 | 0.00571394 | 0.02 | pass |
| smollm2-135m | 3659.6 | 5251.7 | 3791.2 | 3848.1 | 14730.0 | 2639.4 | 19150.2 | 3741.5 | 0.70x | 0.50x | 212.1 | 0.000201941 | 0.001 | pass |
| tiny-gpt2 | 188.7 | 421.0 | 209.5 | 247.5 | 1634.4 | 87.9 | 192.5 | 1395.6 | 0.45x | 0.21x | 13.2 | 2.6077e-08 | 0.001 | pass |
| vit-tiny | 979.4 | 2085.3 | 1400.8 | 1310.5 | 3739.2 | 737.0 | 25037.3 | 1038.4 | 0.47x | 0.35x | 76.3 | 1.04904e-05 | 0.001 | pass |

## Batch-size sweep (median steady_us per forward; per_seq = steady/B)

| model | batch | ours | torch.compile | compile-ro | torch eager | JAX/XLA | ratio ours/compile | ratio ours/jax | per_seq ours | per_seq compile | per_seq jax | rows identical | failed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bert-mini | 1 | 417.9 | 796.0 | 488.7 | 1785.4 | 244.9 | 0.53x | 1.71x | 417.9 | 796.0 | 244.9 | yes |  |
| bert-mini | 2 | 596.4 | 756.5 | 565.8 | 1753.2 | 342.3 | 0.79x | 1.74x | 298.2 | 378.2 | 171.1 | yes |  |
| bert-mini | 4 | 772.6 | 795.4 | 742.3 | 1713.2 | 495.2 | 0.97x | 1.56x | 193.2 | 198.9 | 123.8 | yes |  |
| bert-mini | 8 | 1154.2 | 1065.8 | 1018.1 | 1847.3 | 830.8 | 1.08x | 1.39x | 144.3 | 133.2 | 103.9 | yes |  |
| bert-mini | 16 | 1955.4 | 1780.3 | 1741.7 | 1851.9 | 1473.4 | 1.10x | 1.33x | 122.2 | 111.3 | 92.1 | yes |  |
| bert-mini | 32 | 3638.8 | 3299.8 | 3268.1 | 3484.9 | 2801.6 | 1.10x | 1.30x | 113.7 | 103.1 | 87.5 | yes |  |
| distilgpt2 | 1 | 1292.8 | 1349.0 | 1269.1 | 2769.7 | 1001.5 | 0.96x | 1.29x | 1292.8 | 1349.0 | 1001.5 | yes |  |
| distilgpt2 | 2 | 1850.2 | 1731.9 | 1658.1 | 2744.8 | 1536.0 | 1.07x | 1.20x | 925.1 | 865.9 | 768.0 | yes |  |
| distilgpt2 | 4 | 2976.0 | 2765.8 | 2723.9 | 3123.5 | 2595.1 | 1.08x | 1.15x | 744.0 | 691.4 | 648.8 | yes |  |
| distilgpt2 | 8 | 5136.9 | 5061.6 | 5010.5 | 5852.9 | 4770.8 | 1.01x | 1.08x | 642.1 | 632.7 | 596.4 | yes |  |
| distilgpt2 | 16 | 9363.5 | 9087.1 | 9062.8 | 10702.5 | 8983.4 | 1.03x | 1.04x | 585.2 | 567.9 | 561.5 | yes |  |
| distilgpt2 | 32 | 18263.2 | 17883.9 | 17849.7 | 20825.1 | 18006.9 | 1.02x | 1.01x | 570.7 | 558.9 | 562.7 | yes |  |

## Ablations (median steady_us)

| experiment | variant | model | steady_us | launches/iter | note |
|---|---|---|---|---|---|
| budget_mb | 64 | distilgpt2 | 1284.9 | n/a |  |
| budget_mb | 8 | distilgpt2 | 1289.1 | n/a |  |
| flat_block | 256 | distilgpt2 | 1260.2 | n/a |  |
| flat_block | 256 | resnet18 | 633.3 | n/a |  |
| flat_block | 4096 | distilgpt2 | 1298.2 | n/a |  |
| flat_block | 4096 | resnet18 | 687.7 | n/a |  |
| fusion | off | distilgpt2 | 1945.4 | n/a | enable_opts minus tensor |
| fusion | on | distilgpt2 | 1290.8 | n/a |  |
| max_inputs | mi4 | bert-mini | 488.3 | 34.3 |  |
| max_inputs | mi4 | distilgpt2 | 1318.7 | 45.2 |  |
| max_inputs | mi4 | mixer_b16 | 2735.2 | 76.4 |  |
| max_inputs | mi6 | bert-mini | 432.8 | 24.3 |  |
| max_inputs | mi6 | distilgpt2 | 1304.9 | 38.2 |  |
| max_inputs | mi6 | mixer_b16 | 2697.4 | 63.4 |  |
| max_inputs | mi8 | bert-mini | 424.5 | 24.3 |  |
| max_inputs | mi8 | distilgpt2 | 1294.0 | 38.2 |  |
| max_inputs | mi8 | mixer_b16 | 2691.1 | 63.4 |  |
| precision | float16 | distilgpt2 | 914.7 | n/a |  |
| precision | float16 | smollm2-135m | 2211.3 | n/a |  |
| precision | float32 | distilgpt2 | 1285.1 | n/a |  |
| precision | float32 | smollm2-135m | 3683.5 | n/a |  |
| tf32 | fp32 | resnet18-b1 | 784.8 | n/a |  |
| tf32 | fp32 | resnet18-b8 | 3204.9 | n/a |  |
| tf32 | tf32 | resnet18-b1 | 720.4 | n/a |  |
| tf32 | tf32 | resnet18-b8 | 2714.9 | n/a |  |
| tf32 | torch-fp32 | resnet18-b8 | 3071.7 | n/a |  |

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
| bert-mini | torch-compile | n/a | 1281.5 | 861.6 | 1254 |
| bert-mini | torch-compile-ro | n/a | 1295.2 | 480.5 | 905 |
| bert-mini | torch-compile-mat | n/a | 4304.7 | 423.7 | 3041 |
| bert-mini | torch-tensorrt | n/a | 5023.4 | 655.7 | 4277 |
| bert-mini | jax | 7243.9 | 2.8 | 243.3 | 4570 |
| bert-mini | iree | 998.3 | 19.3 | 3611.3 | n/a |
| bert-mini | ours | n/a | 189.2 | 429.2 | 69 |
| bert-tiny | torch-compile | n/a | 1117.9 | 539.0 | 1795 |
| bert-tiny | torch-compile-ro | n/a | 1139.1 | 299.0 | 1286 |
| bert-tiny | torch-compile-mat | n/a | 3721.8 | 269.9 | 4340 |
| bert-tiny | torch-tensorrt | n/a | 4190.7 | 443.1 | 6189 |
| bert-tiny | jax | 5741.4 | 1.8 | 111.3 | 5684 |
| bert-tiny | iree | 831.1 | 13.9 | 1322.3 | n/a |
| bert-tiny | ours | n/a | 119.9 | 253.4 | 17 |
| distilgpt2 | torch-compile | n/a | 1535.6 | 1352.7 | 981 |
| distilgpt2 | torch-compile-ro | n/a | 1571.8 | 1290.0 | 964 |
| distilgpt2 | torch-compile-mat | n/a | 2206.0 | 1280.2 | 1381 |
| distilgpt2 | torch-tensorrt | n/a | 2949.7 | 3722.1 | n/a |
| distilgpt2 | jax | 6981.6 | 4.5 | 1000.6 | 3851 |
| distilgpt2 | iree | 869.0 | 75.8 | 17122.7 | n/a |
| distilgpt2 | ours | n/a | 566.0 | 1284.5 | 288 |
| micro v0 k1 n1000000 | torch-compile | n/a | 423.4 | 50.8 | 14128 |
| micro v0 k1 n1000000 | torch-compile-ro | n/a | 430.0 | 118.8 | n/a |
| micro v0 k1 n1000000 | torch-compile-mat | n/a | 669.9 | 119.8 | n/a |
| micro v0 k1 n1000000 | torch-tensorrt | n/a | 40.4 | 80.7 | n/a |
| micro v0 k1 n1000000 | jax | 196.2 | 0.5 | 48.8 | 5450 |
| micro v0 k1 n1000000 | iree | 559.8 | 23.7 | 5707.4 | n/a |
| micro v0 k1 n1000000 | triton | 309.3 | 0.1 | 30.7 | 5748 |
| micro v0 k4 n10000 | torch-compile | n/a | 431.5 | 51.9 | 138306 |
| micro v0 k4 n10000 | torch-compile-ro | n/a | 431.5 | 107.4 | n/a |
| micro v0 k4 n10000 | torch-compile-mat | n/a | 500.8 | 116.8 | n/a |
| micro v0 k4 n10000 | torch-tensorrt | n/a | 39.8 | 59.5 | n/a |
| micro v0 k4 n10000 | jax | 201.0 | 0.5 | 42.8 | 13831 |
| micro v0 k4 n10000 | iree | 572.9 | 10.2 | 436.9 | n/a |
| micro v0 k4 n10000 | triton | 308.0 | 0.1 | 20.2 | 7876 |
| micro v0 k4 n100000 | torch-compile | n/a | 422.7 | 52.2 | 41386 |
| micro v0 k4 n100000 | torch-compile-ro | n/a | 434.2 | 118.6 | n/a |
| micro v0 k4 n100000 | torch-compile-mat | n/a | 616.6 | 117.5 | n/a |
| micro v0 k4 n100000 | torch-tensorrt | n/a | 39.3 | 54.1 | -95 |
| micro v0 k4 n100000 | jax | 186.2 | 0.5 | 44.6 | 8690 |
| micro v0 k4 n100000 | iree | 565.4 | 11.0 | 927.8 | n/a |
| micro v0 k4 n100000 | triton | 309.3 | 0.1 | 19.5 | 6424 |
| micro v0 k4 n1000000 | torch-compile | n/a | 449.7 | 68.4 | 1685 |
| micro v0 k4 n1000000 | torch-compile-ro | n/a | 459.0 | 134.2 | 2359 |
| micro v0 k4 n1000000 | torch-compile-mat | n/a | 567.6 | 133.1 | 2950 |
| micro v0 k4 n1000000 | torch-tensorrt | n/a | 40.4 | 321.7 | n/a |
| micro v0 k4 n1000000 | jax | 205.9 | 0.6 | 94.8 | 771 |
| micro v0 k4 n1000000 | iree | 566.0 | 23.5 | 5759.6 | n/a |
| micro v0 k4 n1000000 | triton | 304.2 | 0.1 | 39.1 | 972 |
| micro v0 k4 n10000000 | torch-compile | n/a | 422.4 | 646.0 | 160 |
| micro v0 k4 n10000000 | torch-compile-ro | n/a | 474.9 | 1215.9 | 239 |
| micro v0 k4 n10000000 | torch-compile-mat | n/a | 604.0 | 1196.2 | 306 |
| micro v0 k4 n10000000 | torch-tensorrt | n/a | 42.8 | 3045.9 | n/a |
| micro v0 k4 n10000000 | jax | 219.3 | 1.3 | 945.0 | 87 |
| micro v0 k4 n10000000 | iree | 568.7 | 166.7 | 51069.0 | n/a |
| micro v0 k4 n10000000 | triton | 309.7 | 0.7 | 327.1 | 100 |
| micro v0 k8 n1000000 | torch-compile | n/a | 498.7 | 132.7 | 939 |
| micro v0 k8 n1000000 | torch-compile-ro | n/a | 429.9 | 198.4 | 922 |
| micro v0 k8 n1000000 | torch-compile-mat | n/a | 598.5 | 198.5 | 1318 |
| micro v0 k8 n1000000 | torch-tensorrt | n/a | 42.1 | 643.0 | n/a |
| micro v0 k8 n1000000 | jax | 215.4 | 0.7 | 175.6 | 398 |
| micro v0 k8 n1000000 | iree | 573.3 | 24.2 | 5792.7 | n/a |
| micro v0 k8 n1000000 | triton | 309.2 | 0.2 | 67.3 | 489 |
| micro v1 k4 n1000000 | torch-compile | n/a | 418.9 | 307.7 | 42034 |
| micro v1 k4 n1000000 | torch-compile-ro | n/a | 425.6 | 401.1 | n/a |
| micro v1 k4 n1000000 | torch-compile-mat | n/a | 573.0 | 396.2 | n/a |
| micro v1 k4 n1000000 | torch-tensorrt | n/a | 38.1 | 326.3 | n/a |
| micro v1 k4 n1000000 | jax | 205.9 | 0.6 | 100.1 | 787 |
| micro v1 k4 n1000000 | iree | 566.3 | 24.3 | 6582.9 | n/a |
| micro v1 k4 n1000000 | triton | 309.3 | 0.1 | 43.4 | 1000 |
| micro v10 k1 n191488 | torch-compile | n/a | 1997.7 | 78669.0 | 5878 |
| micro v10 k1 n191488 | torch-compile-ro | n/a | 1926.8 | 78689.1 | 6065 |
| micro v10 k1 n191488 | torch-compile-mat | n/a | 7318.6 | 75791.9 | 2201 |
| micro v10 k1 n191488 | torch-tensorrt | n/a | 2519.2 | 79067.2 | n/a |
| micro v10 k1 n191488 | jax | 5065.9 | 71.5 | 41818.5 | 128 |
| micro v10 k1 n191488 | iree | 3750.5 | 208.1 | 136415.3 | n/a |
| micro v10 k1 n25600 | torch-compile | n/a | 1867.0 | 3154.1 | 2683 |
| micro v10 k1 n25600 | torch-compile-ro | n/a | 1878.7 | 3098.1 | 2465 |
| micro v10 k1 n25600 | torch-compile-mat | n/a | 5619.8 | 3123.5 | 8691 |
| micro v10 k1 n25600 | torch-tensorrt | n/a | 2408.5 | 4072.8 | n/a |
| micro v10 k1 n25600 | jax | 4183.1 | 6.9 | 1503.0 | 1739 |
| micro v10 k1 n25600 | iree | 1529.9 | 26.0 | 10442.1 | n/a |
| micro v11 k1 n25600 | torch-compile | n/a | 1732.9 | 56.5 | 2704080 |
| micro v11 k1 n25600 | torch-compile-ro | n/a | 1629.3 | 123.0 | n/a |
| micro v11 k1 n25600 | torch-compile-mat | n/a | 1743.1 | 112.3 | n/a |
| micro v11 k1 n25600 | torch-tensorrt | n/a | 2071.7 | 97.6 | n/a |
| micro v11 k1 n25600 | jax | 97.9 | 0.5 | 48.7 | -10377 |
| micro v11 k1 n25600 | iree | 184.1 | 8.4 | 56.6 | 16491 |
| micro v11 k1 n25600 | triton | 309.4 | 0.1 | 18.0 | 3172 |
| micro v11 k1 n256000 | torch-compile | n/a | 1701.1 | 180.4 | n/a |
| micro v11 k1 n256000 | torch-compile-ro | n/a | 1732.6 | 225.7 | n/a |
| micro v11 k1 n256000 | torch-compile-mat | n/a | 1835.1 | 244.9 | n/a |
| micro v11 k1 n256000 | torch-tensorrt | n/a | 2066.3 | 190.5 | n/a |
| micro v11 k1 n256000 | jax | 83.6 | 0.5 | 54.0 | -1278 |
| micro v11 k1 n256000 | iree | 182.8 | 10.2 | 185.8 | n/a |
| micro v11 k1 n256000 | triton | 307.5 | 0.1 | 20.0 | 752 |
| micro v12 k1 n25600 | torch-compile | n/a | 1679.5 | 138.7 | n/a |
| micro v12 k1 n25600 | torch-compile-ro | n/a | 1663.0 | 177.6 | n/a |
| micro v12 k1 n25600 | torch-compile-mat | n/a | 1859.4 | 177.6 | n/a |
| micro v12 k1 n25600 | torch-tensorrt | n/a | 2064.3 | 133.8 | 577215 |
| micro v12 k1 n25600 | jax | 1596.9 | 0.7 | 42.2 | 14595 |
| micro v12 k1 n25600 | iree | 218.4 | 13.7 | 1865.6 | n/a |
| micro v12 k1 n25600 | triton | 310.6 | 0.4 | 298.9 | n/a |
| micro v12 k1 n256000 | torch-compile | n/a | 1667.3 | 510.8 | n/a |
| micro v12 k1 n256000 | torch-compile-ro | n/a | 1822.3 | 548.8 | n/a |
| micro v12 k1 n256000 | torch-compile-mat | n/a | 1947.0 | 522.4 | n/a |
| micro v12 k1 n256000 | torch-tensorrt | n/a | 2079.1 | 484.2 | 164218 |
| micro v12 k1 n256000 | jax | 1587.7 | 0.9 | 258.4 | 5683 |
| micro v12 k1 n256000 | iree | 228.0 | 9.9 | 437.1 | -62 |
| micro v12 k1 n256000 | triton | 286.0 | 0.4 | 290.3 | 219 |
| micro v13 k1 n25600 | torch-compile | n/a | 1703.6 | 614.2 | n/a |
| micro v13 k1 n25600 | torch-compile-ro | n/a | 1769.6 | 611.5 | n/a |
| micro v13 k1 n25600 | torch-compile-mat | n/a | 2057.7 | 640.0 | n/a |
| micro v13 k1 n25600 | torch-tensorrt | n/a | 2130.8 | 582.8 | n/a |
| micro v13 k1 n25600 | jax | 1973.8 | 1.0 | 123.3 | 3783 |
| micro v13 k1 n25600 | iree | 328.4 | 11.3 | 2847.7 | n/a |
| micro v13 k1 n25600 | triton | 288.7 | 1.4 | 1290.1 | n/a |
| micro v13 k1 n256000 | torch-compile | n/a | 1801.2 | 5019.8 | n/a |
| micro v13 k1 n256000 | torch-compile-ro | n/a | 1808.4 | 5003.0 | n/a |
| micro v13 k1 n256000 | torch-compile-mat | n/a | 2022.1 | 5054.9 | n/a |
| micro v13 k1 n256000 | torch-tensorrt | n/a | 2111.7 | 4880.6 | n/a |
| micro v13 k1 n256000 | jax | 2033.2 | 4.8 | 3700.5 | 1504 |
| micro v13 k1 n256000 | iree | 641.0 | 19.5 | 5127.8 | n/a |
| micro v13 k1 n256000 | triton | 315.6 | 4.7 | 4681.8 | 252 |
| micro v2 k4 n1000000 | torch-compile | n/a | 562.4 | 72.8 | 2078 |
| micro v2 k4 n1000000 | torch-compile-ro | n/a | 496.5 | 138.4 | 2453 |
| micro v2 k4 n1000000 | torch-compile-mat | n/a | 720.1 | 137.1 | 3620 |
| micro v2 k4 n1000000 | torch-tensorrt | n/a | 460.4 | 326.0 | 6843650 |
| micro v2 k4 n1000000 | jax | 206.1 | 0.6 | 175.7 | 1134 |
| micro v2 k4 n1000000 | iree | 568.0 | 23.6 | 8925.4 | n/a |
| micro v2 k4 n1000000 | triton | 310.7 | 0.1 | 69.7 | 1071 |
| micro v3 k4 n1000000 | torch-compile | n/a | 499.6 | 176.0 | 2302 |
| micro v3 k4 n1000000 | torch-compile-ro | n/a | 528.1 | 394.6 | n/a |
| micro v3 k4 n1000000 | torch-compile-mat | n/a | 812.5 | 340.3 | 20863 |
| micro v3 k4 n1000000 | torch-tensorrt | n/a | 42.0 | 384.8 | n/a |
| micro v3 k4 n1000000 | jax | 202.2 | 0.6 | 317.3 | 2772 |
| micro v3 k4 n1000000 | iree | 569.3 | 24.3 | 81668.8 | n/a |
| micro v3 k4 n1000000 | triton | 311.6 | 0.1 | 123.4 | 1086 |
| micro v4 k4 n1000000 | torch-compile | n/a | 419.3 | 332.1 | 37849 |
| micro v4 k4 n1000000 | torch-compile-ro | n/a | 487.4 | 413.1 | n/a |
| micro v4 k4 n1000000 | torch-compile-mat | n/a | 669.8 | 422.2 | n/a |
| micro v4 k4 n1000000 | torch-tensorrt | n/a | 42.0 | 351.6 | n/a |
| micro v4 k4 n1000000 | jax | 205.7 | 0.6 | 127.5 | 791 |
| micro v4 k4 n1000000 | iree | 564.3 | 23.9 | 10801.5 | n/a |
| micro v4 k4 n1000000 | triton | 365.7 | 0.1 | 69.2 | 1206 |
| micro v5 k4 n1000000 | torch-compile | n/a | 514.2 | 332.6 | 49425 |
| micro v5 k4 n1000000 | torch-compile-ro | n/a | 579.2 | 424.4 | n/a |
| micro v5 k4 n1000000 | torch-compile-mat | n/a | 716.1 | 422.5 | n/a |
| micro v5 k4 n1000000 | torch-tensorrt | n/a | 41.5 | 351.6 | n/a |
| micro v5 k4 n1000000 | jax | 208.0 | 0.6 | 126.9 | 791 |
| micro v5 k4 n1000000 | iree | 563.9 | 24.3 | 10817.8 | n/a |
| micro v5 k4 n1000000 | triton | 310.0 | 0.1 | 69.4 | 996 |
| micro v6 k1 n25600 | torch-compile | n/a | 1762.2 | 381.9 | 129618 |
| micro v6 k1 n25600 | torch-compile-ro | n/a | 1675.3 | 428.2 | n/a |
| micro v6 k1 n25600 | torch-compile-mat | n/a | 1876.6 | 429.2 | n/a |
| micro v6 k1 n25600 | torch-tensorrt | n/a | 2126.7 | 385.0 | 217212 |
| micro v6 k1 n25600 | jax | 1589.5 | 1.0 | 107.2 | 4755 |
| micro v6 k1 n25600 | iree | 229.1 | 17.1 | 5383.7 | n/a |
| micro v6 k1 n25600 | triton | 309.9 | 0.9 | 894.2 | n/a |
| micro v6 k1 n256000 | torch-compile | n/a | 1735.7 | 1271.2 | n/a |
| micro v6 k1 n256000 | torch-compile-ro | n/a | 1835.0 | 1271.5 | n/a |
| micro v6 k1 n256000 | torch-compile-mat | n/a | 1906.0 | 1307.1 | n/a |
| micro v6 k1 n256000 | torch-tensorrt | n/a | 2155.7 | 1245.0 | n/a |
| micro v6 k1 n256000 | jax | 1607.2 | 1.6 | 759.8 | 2837 |
| micro v6 k1 n256000 | iree | 241.2 | 10.2 | 912.4 | 2 |
| micro v6 k1 n256000 | triton | 309.2 | 0.9 | 850.2 | 153 |
| micro v7 k1 n25600 | torch-compile | n/a | 1800.5 | 851.5 | n/a |
| micro v7 k1 n25600 | torch-compile-ro | n/a | 1712.0 | 868.9 | n/a |
| micro v7 k1 n25600 | torch-compile-mat | n/a | 3158.7 | 865.7 | n/a |
| micro v7 k1 n25600 | torch-tensorrt | n/a | 2131.3 | 828.7 | n/a |
| micro v7 k1 n25600 | jax | 2264.5 | 1.7 | 324.8 | 4252 |
| micro v7 k1 n25600 | iree | 325.3 | 28.4 | 11516.3 | n/a |
| micro v7 k1 n256000 | torch-compile | n/a | 1739.1 | 3078.6 | n/a |
| micro v7 k1 n256000 | torch-compile-ro | n/a | 1801.6 | 3041.4 | 60337 |
| micro v7 k1 n256000 | torch-compile-mat | n/a | 2764.3 | 3040.0 | 93128 |
| micro v7 k1 n256000 | torch-tensorrt | n/a | 2181.2 | 3118.2 | n/a |
| micro v7 k1 n256000 | jax | 2201.4 | 3.8 | 1965.9 | 1759 |
| micro v7 k1 n256000 | iree | 430.4 | 43.6 | 31543.7 | n/a |
| micro v8 k1 n25600 | torch-compile | n/a | 1792.7 | 518.9 | 38244 |
| micro v8 k1 n25600 | torch-compile-ro | n/a | 1748.0 | 523.4 | 41905 |
| micro v8 k1 n25600 | torch-compile-mat | n/a | 2703.9 | 522.6 | 67635 |
| micro v8 k1 n25600 | torch-tensorrt | n/a | 2340.7 | 583.5 | n/a |
| micro v8 k1 n25600 | jax | 1284.4 | 1.3 | 266.9 | 3447 |
| micro v8 k1 n25600 | iree | 491.0 | 9.4 | 934.8 | n/a |
| micro v8 k1 n256000 | torch-compile | n/a | 1847.4 | 29873.8 | n/a |
| micro v8 k1 n256000 | torch-compile-ro | n/a | 1788.8 | 29700.5 | n/a |
| micro v8 k1 n256000 | torch-compile-mat | n/a | 4771.3 | 27718.5 | n/a |
| micro v8 k1 n256000 | torch-tensorrt | n/a | 2287.4 | 27515.3 | n/a |
| micro v8 k1 n256000 | jax | 1562.4 | 16.8 | 15450.5 | 105 |
| micro v8 k1 n256000 | iree | 21130.2 | 97.8 | 38544.8 | n/a |
| micro v9 k1 n25600 | torch-compile | n/a | 1786.4 | 856.6 | n/a |
| micro v9 k1 n25600 | torch-compile-ro | n/a | 1829.2 | 867.8 | n/a |
| micro v9 k1 n25600 | torch-compile-mat | n/a | 2507.8 | 867.7 | n/a |
| micro v9 k1 n25600 | torch-tensorrt | n/a | 2215.7 | 927.4 | n/a |
| micro v9 k1 n25600 | jax | 380.6 | 0.8 | 243.5 | 222 |
| micro v9 k1 n256000 | torch-compile | n/a | 1882.5 | 652.7 | n/a |
| micro v9 k1 n256000 | torch-compile-ro | n/a | 1743.5 | 699.7 | n/a |
| micro v9 k1 n256000 | torch-compile-mat | n/a | 2616.3 | 656.3 | n/a |
| micro v9 k1 n256000 | torch-tensorrt | n/a | 2190.8 | 1550.0 | n/a |
| micro v9 k1 n256000 | jax | 814.5 | 1.2 | 333.7 | 2129 |
| mixer_b16 | torch-compile | n/a | 1249.1 | 3143.5 | n/a |
| mixer_b16 | torch-compile-ro | n/a | 1268.4 | 3062.6 | n/a |
| mixer_b16 | torch-compile-mat | n/a | 7539.2 | 3119.0 | n/a |
| mixer_b16 | torch-tensorrt | n/a | 5828.9 | 2486.1 | 14061 |
| mixer_b16 | jax | 7350.2 | 7.0 | 2249.2 | 11257 |
| mixer_b16 | iree | 1524.5 | 750.1 | 673104.7 | n/a |
| mixer_b16 | ours | n/a | 184.3 | 2711.7 | 349 |
| resnet18-b1 | torch-compile | n/a | 1079.2 | 1037.0 | 2354 |
| resnet18-b1 | torch-compile-ro | n/a | 1083.6 | 918.4 | 1804 |
| resnet18-b1 | torch-compile-mat | n/a | 6952.2 | 1041.3 | 18001 |
| resnet18-b1 | torch-tensorrt | n/a | 4786.8 | 995.3 | 10908 |
| resnet18-b1 | jax | 608.0 | 3.9 | 1097.1 | 1337 |
| resnet18-b1 | ours | n/a | 153.6 | 726.3 | -44 |
| resnet18-b8 | torch-compile | n/a | 1049.6 | 2193.9 | 13678 |
| resnet18-b8 | torch-compile-ro | n/a | 1050.6 | 2123.9 | 6492 |
| resnet18-b8 | torch-compile-mat | n/a | 5545.8 | 2126.3 | 41005 |
| resnet18-b8 | torch-tensorrt | n/a | 4721.6 | 2025.2 | 19565 |
| resnet18-b8 | jax | 954.6 | 4.3 | 2275.9 | n/a |
| resnet18-b8 | ours | n/a | 168.4 | 2715.4 | n/a |
| smollm2-135m | torch-compile | n/a | 3825.6 | 5251.7 | 346 |
| smollm2-135m | torch-compile-ro | n/a | 3944.5 | 3791.2 | 311 |
| smollm2-135m | torch-compile-mat | n/a | 5068.1 | 3848.1 | 416 |
| smollm2-135m | torch-tensorrt | n/a | 22359.3 | 3741.5 | 1985 |
| smollm2-135m | jax | 11097.7 | 19.1 | 2639.4 | 874 |
| smollm2-135m | iree | 2061.3 | 174.8 | 19150.2 | n/a |
| smollm2-135m | ours | n/a | 518.9 | 3659.6 | -2 |
| tiny-gpt2 | torch-compile | n/a | 1171.1 | 421.0 | 561 |
| tiny-gpt2 | torch-compile-ro | n/a | 1188.5 | 209.5 | 490 |
| tiny-gpt2 | torch-compile-mat | n/a | 2422.2 | 247.5 | 1393 |
| tiny-gpt2 | torch-tensorrt | n/a | 1912.3 | 1395.6 | 5956 |
| tiny-gpt2 | jax | 1161.5 | 1.6 | 87.9 | 435 |
| tiny-gpt2 | iree | 382.8 | 11.7 | 192.5 | -66 |
| tiny-gpt2 | ours | n/a | 114.5 | 188.7 | -260 |
| vit-tiny | torch-compile | n/a | 1800.4 | 2085.3 | 1007 |
| vit-tiny | torch-compile-ro | n/a | 1808.1 | 1400.8 | 716 |
| vit-tiny | torch-compile-mat | n/a | 4231.8 | 1310.5 | 1687 |
| vit-tiny | torch-tensorrt | n/a | 7700.9 | 1038.4 | 2802 |
| vit-tiny | jax | 6522.4 | 7.1 | 737.0 | 2130 |
| vit-tiny | iree | 1687.6 | 38.8 | 25037.3 | n/a |
| vit-tiny | ours | n/a | 136.7 | 979.4 | 1 |

## Warm-up (median over rounds; steady_at/crossover in iterations, 'none' if never reached within N)

| model | system | cache | first (ms) | steady_at | steady_us | crossover vs eager |
|---|---|---|---|---|---|---|
| distilgpt2 | jax | cold | 4.71 | 31 | 1175.7 | 1 |
| distilgpt2 | jax | warm | 52.54 | 22 | 1164.3 | 1 |
| distilgpt2 | ours | cold | 3779.15 | 167 | 1254.6 | none |
| distilgpt2 | ours | warm | 565.79 | 127 | 1254.4 | 271 |
| distilgpt2 | torch-compile | cold | 4085.52 | 21 | 1422.5 | none |
| distilgpt2 | torch-compile | warm | 1493.33 | 55 | 1415.1 | none |
| distilgpt2 | torch-compile-ro | cold | 3231.27 | 82 | 1302.8 | none |
| distilgpt2 | torch-compile-ro | warm | 1531.10 | 40 | 1311.8 | none |
| distilgpt2 | torch-eager | cold | 127.09 | 8 | 3018.4 | none |
| distilgpt2 | torch-eager | warm | 135.17 | 2 | 3051.6 | none |
| tiny-gpt2 | jax | cold | 1.56 | 42 | 161.2 | 1 |
| tiny-gpt2 | jax | warm | 42.99 | 15 | 158.1 | 1 |
| tiny-gpt2 | ours | cold | 546.47 | none | 176.1 | 1 |
| tiny-gpt2 | ours | warm | 114.34 | none | 176.4 | 1 |
| tiny-gpt2 | torch-compile | cold | 3107.74 | 30 | 370.4 | none |
| tiny-gpt2 | torch-compile | warm | 1181.06 | 18 | 369.9 | none |
| tiny-gpt2 | torch-compile-ro | cold | 2358.63 | 159 | 190.3 | none |
| tiny-gpt2 | torch-compile-ro | warm | 1183.25 | 136 | 203.4 | none |
| tiny-gpt2 | torch-eager | cold | 969.69 | 2 | 1581.3 | none |
| tiny-gpt2 | torch-eager | warm | 489.95 | 2 | 1579.8 | none |

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
| ours | 1 | 32 | 974.5 | 1165.5 | 0 | 4 | 1 | 978 | 0 |
| ours | 1 | 48 | 1382.9 | 1671.9 | 5 | 2 | 1 | 1084 | 0 |
| ours | 1 | 64 | 1439.0 | 1823.6 | 3 | 1 | 3 | 1028 | 0 |
| ours | 1 | 96 | 1707.4 | 2396.0 | 1 | 1 | 0 | 912 | 0 |
| ours | 1 | 128 | 2038.5 | 2857.9 | 0 | 1 | 0 | 918 | 0 |
| ours | 2 | 32 | 943.5 | 1136.6 | 0 | 0 | 0 | 900 | 0 |
| ours | 2 | 48 | 1334.5 | 1624.4 | 0 | 0 | 0 | 900 | 0 |
| ours | 2 | 64 | 1381.4 | 1802.5 | 0 | 0 | 0 | 900 | 0 |
| ours | 2 | 96 | 1631.0 | 2363.1 | 0 | 0 | 0 | 900 | 0 |
| ours | 2 | 128 | 1960.0 | 2780.1 | 0 | 0 | 0 | 900 | 0 |
| torch-compile-dynamic | 1 | 32 | 1095.3 | 1234.9 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-dynamic | 1 | 48 | 1507.0 | 1652.2 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-dynamic | 1 | 64 | 1554.7 | 1705.0 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-dynamic | 1 | 96 | 1732.6 | 1895.4 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-dynamic | 1 | 128 | 1930.3 | 2106.8 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-dynamic | 2 | 32 | 1071.1 | 1209.8 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-dynamic | 2 | 48 | 1477.8 | 1622.1 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-dynamic | 2 | 64 | 1524.3 | 1672.5 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-dynamic | 2 | 96 | 1669.7 | 1833.1 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-dynamic | 2 | 128 | 1861.1 | 2033.6 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-static | 1 | 32 | 1000.0 | 1126.4 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-static | 1 | 48 | 1356.8 | 1488.3 | 0 | 0 | 0 | 0 | 1 |
| torch-compile-static | 1 | 64 | 1393.2 | 1533.2 | 0 | 0 | 0 | 0 | 1 |
| torch-compile-static | 1 | 96 | 1626.2 | 1774.8 | 0 | 0 | 0 | 0 | 1 |
| torch-compile-static | 1 | 128 | 1787.4 | 1950.2 | 0 | 0 | 0 | 0 | 1 |
| torch-compile-static | 2 | 32 | 998.0 | 1122.9 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-static | 2 | 48 | 1343.3 | 1476.0 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-static | 2 | 64 | 1390.1 | 1525.9 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-static | 2 | 96 | 1618.5 | 1768.9 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-static | 2 | 128 | 1784.3 | 1944.3 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | 1 | 32 | 2167.0 | 2290.9 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | 1 | 48 | 2356.3 | 2483.9 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | 1 | 64 | 2363.0 | 2498.9 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | 1 | 96 | 2427.3 | 2573.1 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | 1 | 128 | 2496.3 | 2653.5 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | 2 | 32 | 2169.3 | 2291.4 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | 2 | 48 | 2347.9 | 2477.4 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | 2 | 64 | 2355.8 | 2489.2 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | 2 | 96 | 2414.7 | 2561.0 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | 2 | 128 | 2481.1 | 2641.6 | 0 | 0 | 0 | 0 | 0 |

