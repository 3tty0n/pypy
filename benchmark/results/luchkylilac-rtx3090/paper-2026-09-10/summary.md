# Paper benchmark summary

## Microbenchmarks (median steady_us over rounds)

| variant | k | n | fused (ours) | app-level (ours) | eager (ours) | nojit (ours) | torch.compile | torch eager | JAX/XLA | IREE | Triton | speedup vs compile | interp tax | ours/Triton |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1 | 1000000 | 29.3 | 30.2 | 76.6 | 76.8 | 60.2 | 78.3 | 43.6 | 9055.6 | 30.7 | 2.06x | 1.03x | 0.95x |
| 0 | 4 | 10000 | 11.9 | 18.5 | 39.0 | 39.3 | 46.3 | 58.4 | 43.3 | 472.8 | 18.3 | 3.89x | 1.55x | 0.65x |
| 0 | 4 | 100000 | 12.8 | 20.0 | 44.2 | 43.1 | 51.5 | 60.2 | 45.8 | 1159.7 | 19.6 | 4.02x | 1.56x | 0.65x |
| 0 | 4 | 1000000 | 38.9 | 115.8 | 307.3 | 306.6 | 68.4 | 312.4 | 88.2 | 6545.4 | 39.1 | 1.76x | 2.98x | 1.00x |
| 0 | 4 | 10000000 | 314.1 | 1133.5 | 3027.2 | 3027.3 | 646.0 | 3045.3 | 770.8 | 60291.1 | 327.1 | 2.06x | 3.61x | 0.96x |
| 0 | 8 | 1000000 | 67.3 | 231.0 | 612.8 | 613.0 | 132.7 | 624.7 | 160.3 | 6063.2 | 67.0 | 1.97x | 3.43x | 1.00x |
| 1 | 4 | 1000000 | 41.6 | 119.5 | 311.7 | 312.7 | 307.5 | 316.7 | 93.2 | 8008.5 | 43.3 | 7.39x | 2.87x | 0.96x |
| 2 | 4 | 1000000 | 44.1 | 119.2 | 311.8 | 311.6 | 307.6 | 316.7 | 93.0 | 8415.0 | 43.4 | 6.97x | 2.70x | 1.02x |
| 3 | 4 | 1000000 | 103.9 | 198.0 | 357.7 | 394.7 | 171.9 | 376.9 | 282.3 | 135128.3 | 123.1 | 1.65x | 1.90x | 0.84x |
| 4 | 4 | 1000000 | 53.4 | 144.1 | 340.5 | 336.3 | 332.4 | 342.2 | 115.4 | 14780.0 | 68.8 | 6.22x | 2.70x | 0.78x |
| 5 | 4 | 1000000 | 53.8 | 149.0 | 337.0 | 336.3 | 332.6 | 342.2 | 117.7 | 14684.5 | 67.8 | 6.18x | 2.77x | 0.79x |
| 6 | 1 | 25600 | 277.3 | 269.4 | 279.6 | 277.3 | 389.1 | 392.6 | 107.5 | 5797.7 | 886.3 | 1.40x | 0.97x | 0.31x |
| 6 | 1 | 256000 | 1012.0 | 1000.7 | 1015.8 | 1012.1 | 1298.1 | 1254.8 | 758.4 | 916.9 | 882.1 | 1.28x | 0.99x | 1.15x |
| 7 | 1 | 25600 | 646.9 | 793.9 | 668.6 | 668.5 | 853.8 | 801.8 | 324.7 | 15221.8 | n/a | 1.32x | 1.23x | n/a |
| 7 | 1 | 256000 | 2801.7 | 2940.0 | 2810.5 | 2825.9 | 3036.3 | 3090.4 | 1982.8 | 54986.4 | n/a | 1.08x | 1.05x | n/a |
| 8 | 1 | 25600 | 464.2 | 480.1 | 779.8 | 779.7 | 513.9 | 557.8 | 267.2 | 1175.9 | n/a | 1.11x | 1.03x | n/a |
| 8 | 1 | 256000 | 22309.4 | 22262.1 | 25076.1 | 25143.9 | 29865.1 | 27451.8 | 15421.0 | 44449.7 | n/a | 1.34x | 1.00x | n/a |
| 9 | 1 | 25600 | 757.6 | 748.1 | 766.0 | 766.0 | 853.0 | 800.6 | 248.4 | n/a | n/a | 1.13x | 0.99x | n/a |
| 9 | 1 | 256000 | 573.6 | 572.1 | 579.8 | 570.0 | 679.0 | 587.2 | 346.3 | n/a | n/a | 1.18x | 1.00x | n/a |
| 10 | 1 | 25600 | 3891.8 | 5024.7 | 4797.6 | 4801.0 | 3160.4 | 3701.2 | 1513.2 | 24146.5 | n/a | 0.81x | 1.29x | n/a |
| 10 | 1 | 191488 | 93998.5 | 96937.8 | 98852.1 | 99376.1 | 78672.3 | 78927.0 | 41824.5 | n/a | n/a | 0.84x | 1.03x | n/a |
| 10 | 1 | 256000 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 430927.4 | n/a | n/a | n/a | n/a |
| 11 | 1 | 25600 | 3.5 | 25.2 | 20.7 | 21.2 | 66.0 | 25.8 | 41.9 | 55.0 | 17.8 | 18.92x | 7.23x | 0.20x |
| 11 | 1 | 256000 | 15.3 | 161.3 | 113.4 | 113.2 | 162.3 | 125.3 | 39.7 | 184.1 | 18.5 | 10.64x | 10.57x | 0.82x |
| 12 | 1 | 25600 | 99.1 | 90.2 | 98.9 | 98.0 | 143.2 | 137.0 | 44.8 | 2640.7 | 298.9 | 1.45x | 0.91x | 0.33x |
| 12 | 1 | 256000 | 343.1 | 352.4 | 342.5 | 338.4 | 483.9 | 493.8 | 256.6 | 424.0 | 299.7 | 1.41x | 1.03x | 1.14x |
| 13 | 1 | 25600 | 441.1 | 448.5 | 557.0 | 525.5 | 616.5 | 593.2 | 123.7 | 4206.0 | 1292.6 | 1.40x | 1.02x | 0.34x |
| 13 | 1 | 256000 | 4769.3 | 4756.2 | 5448.2 | 5444.0 | 5018.1 | 4880.6 | 3733.2 | 5290.0 | 4685.6 | 1.05x | 1.00x | 1.02x |

## Precision sweep (median steady_us)

| dtype | variant | k | n | fused (ours) | app-level (ours) | eager (ours) | nojit (ours) | torch.compile | torch eager | JAX/XLA | IREE | Triton | speedup vs compile | interp tax | ours/Triton |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| float16 | 8 | 1 | 256000 | 712.9 | n/a | 2055.9 | 2057.1 | 863.5 | 1472.6 | 633.5 | 4019.8 | n/a | 1.21x | n/a | n/a |
| float16 | 13 | 1 | 256000 | 222.6 | n/a | 350.0 | 335.6 | 431.7 | 413.4 | 96.6 | 1945.0 | 130.6 | 1.94x | n/a | 1.70x |
| float32 | 8 | 1 | 256000 | 1397.1 | n/a | 3920.0 | 3921.1 | 2029.4 | 2659.3 | 1259.3 | 16394.4 | n/a | 1.45x | n/a | n/a |
| float32 | 13 | 1 | 256000 | 272.9 | n/a | 589.6 | 582.0 | 552.0 | 636.7 | 215.0 | 1734.4 | 132.8 | 2.02x | n/a | 2.06x |

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
| bert-mini | 429.2 | 840.3 | 1763.2 | 244.0 | 3638.6 | 660.9 | 0.51x | 0.29x | 1.90735e-05 |
| bert-tiny | 249.1 | 525.4 | 1086.8 | 112.0 | 1484.2 | 436.0 | 0.47x | 0.21x | 3.05176e-05 |
| distilgpt2 | 1274.6 | 1357.4 | 2699.7 | 1001.2 | 21399.7 | 3706.2 | 0.94x | 0.74x | 0.000175476 |
| mixer_b16 | 2686.8 | 3098.4 | 2934.9 | 2237.3 | 1450791.2 | 2463.2 | 0.87x | 0.72x | 3.8147e-05 |
| resnet18-b1 | 719.3 | 946.3 | 1446.2 | 1097.5 | n/a | 990.5 | 0.76x | 1.16x | 0.00963783 |
| resnet18-b8 | 2721.8 | 2193.0 | 2252.7 | 2276.5 | n/a | 2039.6 | 1.24x | 1.04x | 0.00571394 |
| smollm2-135m | 3676.0 | 5824.3 | 14532.9 | 2669.9 | 19100.9 | 3760.0 | 0.63x | 0.46x | 0.000151277 |
| tiny-gpt2 | 180.4 | 405.4 | 1592.1 | 85.6 | 2239.7 | 1410.9 | 0.44x | 0.21x | 2.6077e-08 |
| vit-tiny | 1017.5 | 2386.6 | 3693.5 | 738.7 | 37783.6 | 1059.7 | 0.43x | 0.31x | 1.07288e-05 |

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
| bert-mini | torch-compile | n/a | 1269.0 | 840.3 | 1281 |
| bert-mini | torch-tensorrt | n/a | 4861.9 | 660.9 | 4332 |
| bert-mini | jax | 6917.9 | 2.7 | 244.0 | 4498 |
| bert-mini | iree | 1043.0 | 20.4 | 3638.6 | n/a |
| bert-mini | ours | n/a | 189.8 | 429.2 | 77 |
| bert-tiny | torch-compile | n/a | 1105.6 | 525.4 | 1787 |
| bert-tiny | torch-tensorrt | n/a | 4080.5 | 436.0 | 6113 |
| bert-tiny | jax | 5771.8 | 1.7 | 112.0 | 5818 |
| bert-tiny | iree | 883.6 | 13.3 | 1484.2 | n/a |
| bert-tiny | ours | n/a | 116.4 | 249.1 | 17 |
| distilgpt2 | torch-compile | n/a | 1511.1 | 1357.4 | 1027 |
| distilgpt2 | torch-tensorrt | n/a | 2799.5 | 3706.2 | n/a |
| distilgpt2 | jax | 7108.2 | 4.8 | 1001.2 | 4110 |
| distilgpt2 | iree | 900.3 | 145.5 | 21399.7 | n/a |
| distilgpt2 | ours | n/a | 558.0 | 1274.6 | 298 |
| micro v0 k1 n1000000 | torch-compile | n/a | 416.1 | 60.2 | 20934 |
| micro v0 k1 n1000000 | torch-tensorrt | n/a | 41.9 | 80.7 | n/a |
| micro v0 k1 n1000000 | jax | 194.4 | 0.5 | 43.6 | 4505 |
| micro v0 k1 n1000000 | iree | 558.0 | 28.6 | 9055.6 | n/a |
| micro v0 k1 n1000000 | triton | 292.9 | 0.1 | 30.7 | 5348 |
| micro v0 k4 n10000 | torch-compile | n/a | 414.3 | 46.3 | 30848 |
| micro v0 k4 n10000 | torch-tensorrt | n/a | 38.2 | 53.7 | -303 |
| micro v0 k4 n10000 | jax | 196.5 | 0.5 | 43.3 | 10376 |
| micro v0 k4 n10000 | iree | 606.9 | 9.7 | 472.8 | n/a |
| micro v0 k4 n10000 | triton | 292.0 | 0.1 | 18.3 | 6287 |
| micro v0 k4 n100000 | torch-compile | n/a | 490.2 | 51.5 | 51834 |
| micro v0 k4 n100000 | torch-tensorrt | n/a | 42.9 | 60.5 | n/a |
| micro v0 k4 n100000 | jax | 189.2 | 0.5 | 45.8 | 10446 |
| micro v0 k4 n100000 | iree | 572.9 | 12.6 | 1159.7 | n/a |
| micro v0 k4 n100000 | triton | 308.3 | 0.1 | 19.6 | 6625 |
| micro v0 k4 n1000000 | torch-compile | n/a | 458.5 | 68.4 | 1728 |
| micro v0 k4 n1000000 | torch-tensorrt | n/a | 38.6 | 321.5 | n/a |
| micro v0 k4 n1000000 | jax | 204.0 | 0.6 | 88.2 | 749 |
| micro v0 k4 n1000000 | iree | 567.6 | 30.1 | 6545.4 | n/a |
| micro v0 k4 n1000000 | triton | 308.3 | 0.1 | 39.1 | 994 |
| micro v0 k4 n10000000 | torch-compile | n/a | 431.0 | 646.0 | 164 |
| micro v0 k4 n10000000 | torch-tensorrt | n/a | 40.7 | 3045.9 | n/a |
| micro v0 k4 n10000000 | jax | 216.5 | 1.3 | 770.8 | 79 |
| micro v0 k4 n10000000 | iree | 580.1 | 189.6 | 60291.1 | n/a |
| micro v0 k4 n10000000 | triton | 306.7 | 0.7 | 327.1 | 99 |
| micro v0 k8 n1000000 | torch-compile | n/a | 433.6 | 132.7 | 802 |
| micro v0 k8 n1000000 | torch-tensorrt | n/a | 42.2 | 642.8 | n/a |
| micro v0 k8 n1000000 | jax | 210.5 | 0.7 | 160.3 | 371 |
| micro v0 k8 n1000000 | iree | 588.5 | 30.3 | 6063.2 | n/a |
| micro v0 k8 n1000000 | triton | 276.5 | 0.1 | 67.0 | 426 |
| micro v1 k4 n1000000 | torch-compile | n/a | 422.3 | 307.5 | 41575 |
| micro v1 k4 n1000000 | torch-tensorrt | n/a | 39.2 | 325.9 | n/a |
| micro v1 k4 n1000000 | jax | 201.5 | 0.6 | 93.2 | 730 |
| micro v1 k4 n1000000 | iree | 563.4 | 27.2 | 8008.5 | n/a |
| micro v1 k4 n1000000 | triton | 309.6 | 0.1 | 43.3 | 990 |
| micro v10 k1 n191488 | torch-compile | n/a | 1975.7 | 78672.3 | 6134 |
| micro v10 k1 n191488 | torch-tensorrt | n/a | 2527.3 | 79040.4 | n/a |
| micro v10 k1 n191488 | jax | 5056.1 | 69.0 | 41824.5 | 127 |
| micro v10 k1 n25600 | torch-compile | n/a | 1825.0 | 3160.4 | 2735 |
| micro v10 k1 n25600 | torch-tensorrt | n/a | 2398.0 | 4053.3 | n/a |
| micro v10 k1 n25600 | jax | 4203.2 | 6.6 | 1513.2 | 1766 |
| micro v10 k1 n25600 | iree | 1622.2 | 39.7 | 24146.5 | n/a |
| micro v10 k1 n256000 | iree | 22412.4 | 321.3 | 430927.4 | n/a |
| micro v11 k1 n25600 | torch-compile | n/a | 1598.8 | 66.0 | n/a |
| micro v11 k1 n25600 | torch-tensorrt | n/a | 1973.0 | 57.7 | n/a |
| micro v11 k1 n25600 | jax | 54.3 | 0.4 | 41.9 | n/a |
| micro v11 k1 n25600 | iree | 172.4 | 11.1 | 55.0 | n/a |
| micro v11 k1 n25600 | triton | 306.3 | 0.1 | 17.8 | 12487 |
| micro v11 k1 n256000 | torch-compile | n/a | 1621.2 | 162.3 | n/a |
| micro v11 k1 n256000 | torch-tensorrt | n/a | 2049.3 | 151.1 | n/a |
| micro v11 k1 n256000 | jax | 53.8 | 0.4 | 39.7 | -2029 |
| micro v11 k1 n256000 | iree | 186.4 | 10.8 | 184.1 | n/a |
| micro v11 k1 n256000 | triton | 307.9 | 0.1 | 18.5 | 750 |
| micro v12 k1 n25600 | torch-compile | n/a | 1742.6 | 143.2 | n/a |
| micro v12 k1 n25600 | torch-tensorrt | n/a | 2028.6 | 134.3 | 657637 |
| micro v12 k1 n25600 | jax | 1588.5 | 0.7 | 44.8 | 14600 |
| micro v12 k1 n25600 | iree | 232.8 | 13.9 | 2640.7 | n/a |
| micro v12 k1 n25600 | triton | 309.6 | 0.4 | 298.9 | n/a |
| micro v12 k1 n256000 | torch-compile | n/a | 1729.3 | 483.9 | 149633 |
| micro v12 k1 n256000 | torch-tensorrt | n/a | 2139.7 | 485.9 | 240046 |
| micro v12 k1 n256000 | jax | 1601.2 | 1.0 | 256.6 | 5667 |
| micro v12 k1 n256000 | iree | 280.8 | 10.4 | 424.0 | 477 |
| micro v12 k1 n256000 | triton | 308.8 | 0.4 | 299.7 | 264 |
| micro v13 k1 n25600 | torch-compile | n/a | 1757.2 | 616.5 | n/a |
| micro v13 k1 n25600 | torch-tensorrt | n/a | 2130.3 | 582.0 | 164938 |
| micro v13 k1 n25600 | jax | 2077.4 | 1.0 | 123.7 | 3835 |
| micro v13 k1 n25600 | iree | 341.0 | 11.2 | 4206.0 | n/a |
| micro v13 k1 n25600 | triton | 280.6 | 1.3 | 1292.6 | n/a |
| micro v13 k1 n256000 | torch-compile | n/a | 1799.8 | 5018.1 | n/a |
| micro v13 k1 n256000 | torch-tensorrt | n/a | 2181.6 | 4865.4 | 124617 |
| micro v13 k1 n256000 | jax | 1979.6 | 4.8 | 3733.2 | 1473 |
| micro v13 k1 n256000 | iree | 656.8 | 19.9 | 5290.0 | n/a |
| micro v13 k1 n256000 | triton | 316.4 | 4.9 | 4685.6 | 136 |
| micro v2 k4 n1000000 | torch-compile | n/a | 481.2 | 307.6 | 49104 |
| micro v2 k4 n1000000 | torch-tensorrt | n/a | 39.3 | 325.9 | n/a |
| micro v2 k4 n1000000 | jax | 205.4 | 0.6 | 93.0 | 753 |
| micro v2 k4 n1000000 | iree | 583.5 | 27.9 | 8415.0 | n/a |
| micro v2 k4 n1000000 | triton | 309.9 | 0.1 | 43.4 | 997 |
| micro v3 k4 n1000000 | torch-compile | n/a | 489.8 | 171.9 | 2212 |
| micro v3 k4 n1000000 | torch-tensorrt | n/a | 42.1 | 384.7 | n/a |
| micro v3 k4 n1000000 | jax | 201.5 | 0.6 | 282.3 | 1752 |
| micro v3 k4 n1000000 | iree | 583.4 | 24.4 | 135128.3 | n/a |
| micro v3 k4 n1000000 | triton | 310.2 | 0.1 | 123.1 | 1080 |
| micro v4 k4 n1000000 | torch-compile | n/a | 486.3 | 332.4 | 45463 |
| micro v4 k4 n1000000 | torch-tensorrt | n/a | 42.2 | 351.5 | n/a |
| micro v4 k4 n1000000 | jax | 202.7 | 0.6 | 115.4 | 726 |
| micro v4 k4 n1000000 | iree | 553.1 | 29.9 | 14780.0 | n/a |
| micro v4 k4 n1000000 | triton | 310.5 | 0.1 | 68.8 | 994 |
| micro v5 k4 n1000000 | torch-compile | n/a | 516.4 | 332.6 | 49603 |
| micro v5 k4 n1000000 | torch-tensorrt | n/a | 38.1 | 351.5 | n/a |
| micro v5 k4 n1000000 | jax | 202.9 | 0.6 | 117.7 | 733 |
| micro v5 k4 n1000000 | iree | 552.2 | 28.9 | 14684.5 | n/a |
| micro v5 k4 n1000000 | triton | 309.1 | 0.1 | 67.8 | 985 |
| micro v6 k1 n25600 | torch-compile | n/a | 1748.5 | 389.1 | 430872 |
| micro v6 k1 n25600 | torch-tensorrt | n/a | 2090.2 | 382.0 | 172737 |
| micro v6 k1 n25600 | jax | 1598.2 | 1.0 | 107.5 | 4707 |
| micro v6 k1 n25600 | iree | 238.0 | 16.0 | 5797.7 | n/a |
| micro v6 k1 n25600 | triton | 306.2 | 0.9 | 886.3 | n/a |
| micro v6 k1 n256000 | torch-compile | n/a | 1719.0 | 1298.1 | n/a |
| micro v6 k1 n256000 | torch-tensorrt | n/a | 2157.3 | 1234.6 | 92917 |
| micro v6 k1 n256000 | jax | 1629.8 | 1.6 | 758.4 | 2723 |
| micro v6 k1 n256000 | iree | 248.6 | 10.3 | 916.9 | -61 |
| micro v6 k1 n256000 | triton | 308.5 | 0.9 | 882.1 | 80 |
| micro v7 k1 n25600 | torch-compile | n/a | 1797.6 | 853.8 | n/a |
| micro v7 k1 n25600 | torch-tensorrt | n/a | 2241.2 | 813.6 | n/a |
| micro v7 k1 n25600 | jax | 2247.4 | 1.6 | 324.7 | 4136 |
| micro v7 k1 n25600 | iree | 342.7 | 27.4 | 15221.8 | n/a |
| micro v7 k1 n256000 | torch-compile | n/a | 1782.5 | 3036.3 | 27649 |
| micro v7 k1 n256000 | torch-tensorrt | n/a | 2153.8 | 3107.8 | n/a |
| micro v7 k1 n256000 | jax | 2243.0 | 3.9 | 1982.8 | 1771 |
| micro v7 k1 n256000 | iree | 427.9 | 60.9 | 54986.4 | n/a |
| micro v8 k1 n25600 | torch-compile | n/a | 1750.1 | 513.9 | 32846 |
| micro v8 k1 n25600 | torch-tensorrt | n/a | 2199.2 | 563.5 | n/a |
| micro v8 k1 n25600 | jax | 1275.7 | 1.3 | 267.2 | 3336 |
| micro v8 k1 n25600 | iree | 490.9 | 9.4 | 1175.9 | n/a |
| micro v8 k1 n256000 | torch-compile | n/a | 1770.4 | 29865.1 | n/a |
| micro v8 k1 n256000 | torch-tensorrt | n/a | 2303.3 | 27411.6 | 48857 |
| micro v8 k1 n256000 | jax | 1548.3 | 16.7 | 15421.0 | 102 |
| micro v8 k1 n256000 | iree | 21094.5 | 78.1 | 44449.7 | n/a |
| micro v9 k1 n25600 | torch-compile | n/a | 1776.1 | 853.0 | n/a |
| micro v9 k1 n25600 | torch-tensorrt | n/a | 2125.6 | 935.5 | n/a |
| micro v9 k1 n25600 | jax | 366.2 | 0.8 | 248.4 | 142 |
| micro v9 k1 n256000 | torch-compile | n/a | 1775.7 | 679.0 | n/a |
| micro v9 k1 n256000 | torch-tensorrt | n/a | 2148.5 | 1559.7 | n/a |
| micro v9 k1 n256000 | jax | 831.0 | 1.0 | 346.3 | 2143 |
| mixer_b16 | torch-compile | n/a | 1174.5 | 3098.4 | n/a |
| mixer_b16 | torch-tensorrt | n/a | 5711.1 | 2463.2 | 11841 |
| mixer_b16 | jax | 7318.6 | 7.0 | 2237.3 | 10321 |
| mixer_b16 | iree | 1522.4 | 1485.8 | 1450791.2 | n/a |
| mixer_b16 | ours | n/a | 188.4 | 2686.8 | 252 |
| resnet18-b1 | torch-compile | n/a | 995.6 | 946.3 | 1611 |
| resnet18-b1 | torch-tensorrt | n/a | 4618.8 | 990.5 | 9718 |
| resnet18-b1 | jax | 607.9 | 3.9 | 1097.5 | 1208 |
| resnet18-b1 | ours | n/a | 132.2 | 719.3 | -80 |
| resnet18-b8 | torch-compile | n/a | 1004.2 | 2193.0 | 13884 |
| resnet18-b8 | torch-tensorrt | n/a | 4697.4 | 2039.6 | 21221 |
| resnet18-b8 | jax | 933.9 | 4.3 | 2276.5 | n/a |
| resnet18-b8 | ours | n/a | 164.8 | 2721.8 | n/a |
| smollm2-135m | torch-compile | n/a | 3899.8 | 5824.3 | 385 |
| smollm2-135m | torch-tensorrt | n/a | 21818.3 | 3760.0 | 1975 |
| smollm2-135m | jax | 10648.4 | 19.0 | 2669.9 | 853 |
| smollm2-135m | iree | 2058.9 | 170.9 | 19100.9 | n/a |
| smollm2-135m | ours | n/a | 510.3 | 3676.0 | -3 |
| tiny-gpt2 | torch-compile | n/a | 1171.4 | 405.4 | 586 |
| tiny-gpt2 | torch-tensorrt | n/a | 1821.5 | 1410.9 | 7423 |
| tiny-gpt2 | jax | 1142.5 | 1.6 | 85.6 | 443 |
| tiny-gpt2 | iree | 395.2 | 11.2 | 2239.7 | n/a |
| tiny-gpt2 | ours | n/a | 104.1 | 180.4 | -264 |
| vit-tiny | torch-compile | n/a | 1775.1 | 2386.6 | 1259 |
| vit-tiny | torch-tensorrt | n/a | 7833.5 | 1059.7 | 2925 |
| vit-tiny | jax | 6343.0 | 7.3 | 738.7 | 2105 |
| vit-tiny | iree | 1716.8 | 74.6 | 37783.6 | n/a |
| vit-tiny | ours | n/a | 151.8 | 1017.5 | 8 |

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

