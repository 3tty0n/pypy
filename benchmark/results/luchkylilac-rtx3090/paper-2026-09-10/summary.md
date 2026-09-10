# Paper benchmark summary

## Microbenchmarks (median steady_us over rounds)

| variant | k | n | fused (ours) | app-level (ours) | eager (ours) | nojit (ours) | torch.compile | torch eager | JAX/XLA | IREE | Triton | speedup vs compile | interp tax | ours/Triton |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1 | 1000000 | 29.0 | 30.1 | 76.6 | 76.8 | 55.2 | 78.3 | 43.3 | 5773.4 | 30.6 | 1.90x | 1.04x | 0.95x |
| 0 | 4 | 10000 | 11.9 | 18.4 | 38.5 | 38.9 | 51.1 | 59.6 | 44.3 | 449.3 | 19.5 | 4.28x | 1.54x | 0.61x |
| 0 | 4 | 100000 | 12.7 | 19.9 | 43.5 | 43.2 | 50.2 | 58.1 | 42.2 | 934.5 | 20.0 | 3.96x | 1.57x | 0.63x |
| 0 | 4 | 1000000 | 39.8 | 115.8 | 307.7 | 306.1 | 68.7 | 312.3 | 88.3 | 5823.0 | 38.1 | 1.73x | 2.91x | 1.04x |
| 0 | 4 | 10000000 | 331.2 | 1133.6 | 3026.5 | 3026.8 | 646.0 | 3044.9 | 767.3 | 50517.6 | 327.1 | 1.95x | 3.42x | 1.01x |
| 0 | 8 | 1000000 | 70.7 | 231.4 | 614.1 | 612.0 | 133.0 | 624.1 | 160.6 | 5803.0 | 70.3 | 1.88x | 3.27x | 1.01x |
| 1 | 4 | 1000000 | 41.7 | 119.1 | 310.0 | 310.8 | 307.6 | 316.6 | 92.9 | n/a | 43.4 | 7.37x | 2.86x | 0.96x |
| 2 | 4 | 1000000 | 44.0 | 119.2 | 311.2 | 310.7 | 307.4 | 316.6 | 93.0 | n/a | 43.4 | 6.98x | 2.71x | 1.01x |
| 3 | 4 | 1000000 | 94.3 | 200.0 | 357.3 | 394.8 | 167.1 | 375.2 | 290.3 | n/a | 122.7 | 1.77x | 2.12x | 0.77x |
| 4 | 4 | 1000000 | 52.2 | 144.4 | 339.6 | 335.8 | 332.0 | 342.2 | 116.9 | n/a | 69.3 | 6.36x | 2.76x | 0.75x |
| 5 | 4 | 1000000 | 55.6 | 149.1 | 335.7 | 335.8 | 332.4 | 342.1 | 115.2 | n/a | 69.3 | 5.98x | 2.68x | 0.80x |
| 6 | 1 | 25600 | 276.9 | 268.0 | 279.5 | 277.1 | 392.4 | 392.8 | 107.7 | 5375.7 | 845.8 | 1.42x | 0.97x | 0.33x |
| 6 | 1 | 256000 | 1030.8 | 996.1 | 1015.7 | 1011.6 | 1283.8 | 1260.2 | 758.3 | 910.6 | 867.4 | 1.25x | 0.97x | 1.19x |
| 7 | 1 | 25600 | 640.2 | 790.5 | 666.9 | 667.1 | 849.5 | 801.2 | 332.7 | n/a | n/a | 1.33x | 1.23x | n/a |
| 7 | 1 | 256000 | 2798.7 | 2927.3 | 2827.5 | 2833.0 | 3077.6 | 3102.1 | 1977.2 | n/a | n/a | 1.10x | 1.05x | n/a |
| 8 | 1 | 25600 | 464.2 | 477.3 | 777.9 | 778.0 | 519.1 | 555.1 | 267.2 | n/a | n/a | 1.12x | 1.03x | n/a |
| 8 | 1 | 256000 | 22334.6 | 22349.4 | 25084.3 | 25097.9 | 29742.8 | 27344.4 | 15336.3 | n/a | n/a | 1.33x | 1.00x | n/a |
| 9 | 1 | 25600 | 755.7 | 746.0 | 765.5 | 765.9 | 865.9 | 798.9 | 263.3 | n/a | n/a | 1.15x | 0.99x | n/a |
| 9 | 1 | 256000 | 544.5 | 568.2 | 582.4 | 572.7 | 685.1 | 587.1 | 350.7 | n/a | n/a | 1.26x | 1.04x | n/a |
| 10 | 1 | 25600 | 4064.9 | 8652.0 | 4818.6 | 4818.5 | 3190.9 | 3716.2 | 1492.9 | n/a | n/a | 0.78x | 2.13x | n/a |
| 10 | 1 | 191488 | 94343.8 | n/a | 99019.2 | 99417.7 | 78479.0 | 78748.5 | n/a | n/a | n/a | 0.83x | n/a | n/a |
| 10 | 1 | 256000 | n/a | n/a | n/a | n/a | n/a | n/a | 74367.7 | n/a | n/a | n/a | n/a | n/a |
| 11 | 1 | 25600 | 3.4 | 24.5 | 20.5 | 20.8 | 72.7 | 26.9 | 42.4 | 54.1 | 18.0 | 21.26x | 7.17x | 0.19x |
| 11 | 1 | 256000 | 14.9 | 159.5 | 113.3 | 113.1 | 155.2 | 127.2 | 44.3 | 185.5 | 18.5 | 10.40x | 10.69x | 0.80x |
| 12 | 1 | 25600 | 98.8 | 91.0 | 98.9 | 97.9 | 144.3 | 139.8 | 45.8 | 1823.3 | 298.8 | 1.46x | 0.92x | 0.33x |
| 12 | 1 | 256000 | 362.5 | 338.5 | 342.8 | 338.6 | 496.5 | 498.0 | 256.4 | 439.6 | 283.4 | 1.37x | 0.93x | 1.28x |
| 13 | 1 | 25600 | 440.8 | 446.8 | 526.2 | 526.2 | 595.6 | 592.7 | 123.7 | 2809.8 | 1302.3 | 1.35x | 1.01x | 0.34x |
| 13 | 1 | 256000 | 4750.7 | 4755.6 | 5428.6 | 5423.0 | 4995.0 | 4882.6 | 3713.4 | 5168.8 | 4679.2 | 1.05x | 1.00x | 1.02x |

## Precision sweep (median steady_us)

| dtype | variant | k | n | fused (ours) | app-level (ours) | eager (ours) | nojit (ours) | torch.compile | torch eager | JAX/XLA | IREE | Triton | speedup vs compile | interp tax | ours/Triton |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| float16 | 8 | 1 | 256000 | 712.6 | n/a | 2055.7 | 2055.7 | 866.5 | 1469.0 | 632.8 | n/a | n/a | 1.22x | n/a | n/a |
| float16 | 13 | 1 | 256000 | 220.8 | n/a | 348.8 | 342.1 | 439.8 | 415.0 | 96.5 | n/a | 134.1 | 1.99x | n/a | 1.65x |
| float32 | 8 | 1 | 256000 | 1870.5 | n/a | 4379.7 | 4381.1 | 2030.8 | 2665.0 | 1260.2 | n/a | n/a | 1.09x | n/a | n/a |
| float32 | 13 | 1 | 256000 | 316.4 | n/a | 632.2 | 625.9 | 549.4 | 640.3 | 215.0 | n/a | 133.9 | 1.74x | n/a | 2.36x |

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
| bert-mini | 672.5 | 778.1 | 1802.2 | 242.1 | 3617.8 | 667.0 | 0.86x | 0.31x | 2.09808e-05 |
| bert-tiny | 392.4 | 532.5 | 1090.0 | 111.4 | n/a | 436.0 | 0.74x | 0.21x | 3.05176e-05 |
| distilgpt2 | 1374.5 | 1323.8 | 2739.8 | 1002.2 | n/a | 3778.7 | 1.04x | 0.76x | 0.000144958 |
| mixer_b16 | 3802.5 | 3127.6 | 3037.3 | 2233.4 | n/a | 2465.0 | 1.22x | 0.71x | 3.71933e-05 |
| resnet18-b1 | 765.6 | 1058.4 | 1568.0 | 1097.0 | n/a | 1010.1 | 0.72x | 1.04x | 0.00524807 |
| resnet18-b8 | 3193.9 | 2186.6 | 2255.2 | 2277.3 | n/a | 2027.1 | 1.46x | 1.04x | 0.00887299 |
| smollm2-135m | 4062.3 | 5199.4 | 15699.0 | 2647.8 | n/a | 3751.1 | 0.78x | 0.51x | 0.000151277 |
| tiny-gpt2 | 195.6 | 418.4 | 1643.7 | 88.6 | n/a | 1393.5 | 0.47x | 0.21x | 2.6077e-08 |
| vit-tiny | 1671.3 | 2147.8 | 3722.4 | 736.8 | 24999.2 | 1062.7 | 0.78x | 0.34x | 1.23978e-05 |

## Ablations (median steady_us)

| experiment | variant | model | steady_us | note |
|---|---|---|---|---|
| budget_mb | 64 | distilgpt2 | 1378.0 |  |
| budget_mb | 8 | distilgpt2 | 1367.5 |  |
| flat_block | 256 | distilgpt2 | 1331.8 |  |
| flat_block | 256 | resnet18 | 680.2 |  |
| flat_block | 4096 | distilgpt2 | 1383.0 |  |
| flat_block | 4096 | resnet18 | 743.1 |  |
| fusion | off | distilgpt2 | 2028.4 | enable_opts minus tensor |
| fusion | on | distilgpt2 | 1388.7 |  |
| precision | float16 | distilgpt2 | 961.1 |  |
| precision | float16 | smollm2-135m | 2215.7 |  |
| precision | float32 | distilgpt2 | 1374.5 |  |
| precision | float32 | smollm2-135m | 4103.3 |  |

## Compilation overhead (median; break-even vs torch eager, iterations)

| workload | system | compile_ms | first_run_ms | steady_us | break-even |
|---|---|---|---|---|---|
| bert-mini | torch-tensorrt | n/a | 4944.6 | 667.0 | n/a |
| bert-mini | jax | 7140.8 | 2.8 | 242.1 | n/a |
| bert-mini | iree | 1005.5 | 20.5 | 3617.8 | n/a |
| bert-tiny | torch-tensorrt | n/a | 4082.3 | 436.0 | n/a |
| bert-tiny | jax | 5942.2 | 1.8 | 111.4 | n/a |
| distilgpt2 | torch-tensorrt | n/a | 2857.0 | 3778.7 | n/a |
| distilgpt2 | jax | 7211.3 | 4.7 | 1002.2 | n/a |
| micro v0 k1 n1000000 | torch-tensorrt | n/a | 39.1 | 80.7 | n/a |
| micro v0 k1 n1000000 | jax | 195.7 | 0.5 | 43.3 | n/a |
| micro v0 k1 n1000000 | iree | 550.1 | 25.8 | 5773.4 | n/a |
| micro v0 k1 n1000000 | triton | 308.5 | 0.1 | 30.6 | n/a |
| micro v0 k4 n10000 | torch-tensorrt | n/a | 41.8 | 57.9 | n/a |
| micro v0 k4 n10000 | jax | 198.8 | 0.5 | 44.3 | n/a |
| micro v0 k4 n10000 | iree | 559.4 | 10.2 | 449.3 | n/a |
| micro v0 k4 n10000 | triton | 309.2 | 0.1 | 19.5 | n/a |
| micro v0 k4 n100000 | torch-tensorrt | n/a | 38.7 | 59.6 | n/a |
| micro v0 k4 n100000 | jax | 183.6 | 0.5 | 42.2 | n/a |
| micro v0 k4 n100000 | iree | 550.3 | 12.1 | 934.5 | n/a |
| micro v0 k4 n100000 | triton | 308.1 | 0.1 | 20.0 | n/a |
| micro v0 k4 n1000000 | torch-tensorrt | n/a | 39.1 | 321.5 | n/a |
| micro v0 k4 n1000000 | jax | 203.0 | 0.6 | 88.3 | n/a |
| micro v0 k4 n1000000 | iree | 553.7 | 25.6 | 5823.0 | n/a |
| micro v0 k4 n1000000 | triton | 307.9 | 0.1 | 38.1 | n/a |
| micro v0 k4 n10000000 | torch-tensorrt | n/a | 41.5 | 3046.1 | n/a |
| micro v0 k4 n10000000 | jax | 211.8 | 1.3 | 767.3 | n/a |
| micro v0 k4 n10000000 | iree | 555.1 | 174.6 | 50517.6 | n/a |
| micro v0 k4 n10000000 | triton | 273.6 | 0.7 | 327.1 | n/a |
| micro v0 k8 n1000000 | torch-tensorrt | n/a | 39.3 | 641.9 | n/a |
| micro v0 k8 n1000000 | jax | 209.9 | 0.7 | 160.6 | n/a |
| micro v0 k8 n1000000 | iree | 560.8 | 25.5 | 5803.0 | n/a |
| micro v0 k8 n1000000 | triton | 310.3 | 0.2 | 70.3 | n/a |
| micro v1 k4 n1000000 | torch-tensorrt | n/a | 39.3 | 325.9 | n/a |
| micro v1 k4 n1000000 | jax | 201.3 | 0.6 | 92.9 | n/a |
| micro v1 k4 n1000000 | triton | 309.8 | 0.1 | 43.4 | n/a |
| micro v10 k1 n25600 | torch-tensorrt | n/a | 2362.7 | 4071.6 | n/a |
| micro v10 k1 n25600 | jax | 4325.4 | 7.0 | 1492.9 | n/a |
| micro v10 k1 n256000 | torch-tensorrt | n/a | 2497.6 | 135811.9 | n/a |
| micro v10 k1 n256000 | jax | 4893.7 | 99.4 | 74367.7 | n/a |
| micro v11 k1 n25600 | torch-tensorrt | n/a | 1955.7 | 58.3 | n/a |
| micro v11 k1 n25600 | jax | 53.6 | 0.4 | 42.4 | n/a |
| micro v11 k1 n25600 | iree | 174.8 | 8.9 | 54.1 | n/a |
| micro v11 k1 n25600 | triton | 308.9 | 0.1 | 18.0 | n/a |
| micro v11 k1 n256000 | torch-tensorrt | n/a | 1991.7 | 152.2 | n/a |
| micro v11 k1 n256000 | jax | 52.3 | 0.4 | 44.3 | n/a |
| micro v11 k1 n256000 | iree | 172.3 | 11.1 | 185.5 | n/a |
| micro v11 k1 n256000 | triton | 306.4 | 0.1 | 18.5 | n/a |
| micro v12 k1 n25600 | torch-tensorrt | n/a | 2010.4 | 134.9 | n/a |
| micro v12 k1 n25600 | jax | 1572.5 | 0.8 | 45.8 | n/a |
| micro v12 k1 n25600 | iree | 213.5 | 14.6 | 1823.3 | n/a |
| micro v12 k1 n25600 | triton | 309.6 | 0.4 | 298.8 | n/a |
| micro v12 k1 n256000 | torch-tensorrt | n/a | 2015.3 | 485.0 | n/a |
| micro v12 k1 n256000 | jax | 1586.5 | 0.9 | 256.4 | n/a |
| micro v12 k1 n256000 | iree | 227.3 | 10.5 | 439.6 | n/a |
| micro v12 k1 n256000 | triton | 309.5 | 0.4 | 283.4 | n/a |
| micro v13 k1 n25600 | torch-tensorrt | n/a | 2107.9 | 582.3 | n/a |
| micro v13 k1 n25600 | jax | 1956.8 | 1.1 | 123.7 | n/a |
| micro v13 k1 n25600 | iree | 326.0 | 11.8 | 2809.8 | n/a |
| micro v13 k1 n25600 | triton | 312.8 | 1.4 | 1302.3 | n/a |
| micro v13 k1 n256000 | torch-tensorrt | n/a | 2115.3 | 4865.4 | n/a |
| micro v13 k1 n256000 | jax | 1971.2 | 4.8 | 3713.4 | n/a |
| micro v13 k1 n256000 | iree | 638.4 | 20.6 | 5168.8 | n/a |
| micro v13 k1 n256000 | triton | 316.0 | 4.7 | 4679.2 | n/a |
| micro v2 k4 n1000000 | torch-tensorrt | n/a | 42.4 | 325.9 | n/a |
| micro v2 k4 n1000000 | jax | 204.2 | 0.6 | 93.0 | n/a |
| micro v2 k4 n1000000 | triton | 309.6 | 0.1 | 43.4 | n/a |
| micro v3 k4 n1000000 | torch-tensorrt | n/a | 42.0 | 384.8 | n/a |
| micro v3 k4 n1000000 | jax | 198.2 | 0.6 | 290.3 | n/a |
| micro v3 k4 n1000000 | triton | 309.1 | 0.1 | 122.7 | n/a |
| micro v4 k4 n1000000 | torch-tensorrt | n/a | 38.6 | 351.5 | n/a |
| micro v4 k4 n1000000 | jax | 202.7 | 0.6 | 116.9 | n/a |
| micro v4 k4 n1000000 | triton | 310.9 | 0.1 | 69.3 | n/a |
| micro v5 k4 n1000000 | torch-tensorrt | n/a | 41.7 | 351.5 | n/a |
| micro v5 k4 n1000000 | jax | 199.2 | 0.6 | 115.2 | n/a |
| micro v5 k4 n1000000 | triton | 309.4 | 0.1 | 69.3 | n/a |
| micro v6 k1 n25600 | torch-tensorrt | n/a | 2046.2 | 382.1 | n/a |
| micro v6 k1 n25600 | jax | 1596.3 | 1.0 | 107.7 | n/a |
| micro v6 k1 n25600 | iree | 227.9 | 18.0 | 5375.7 | n/a |
| micro v6 k1 n25600 | triton | 308.5 | 0.9 | 845.8 | n/a |
| micro v6 k1 n256000 | torch-tensorrt | n/a | 2093.2 | 1260.7 | n/a |
| micro v6 k1 n256000 | jax | 1616.8 | 1.6 | 758.3 | n/a |
| micro v6 k1 n256000 | iree | 238.4 | 10.9 | 910.6 | n/a |
| micro v6 k1 n256000 | triton | 308.6 | 0.9 | 867.4 | n/a |
| micro v7 k1 n25600 | torch-tensorrt | n/a | 2092.9 | 836.7 | n/a |
| micro v7 k1 n25600 | jax | 2183.6 | 1.6 | 332.7 | n/a |
| micro v7 k1 n256000 | torch-tensorrt | n/a | 2166.6 | 3089.8 | n/a |
| micro v7 k1 n256000 | jax | 2182.0 | 3.8 | 1977.2 | n/a |
| micro v8 k1 n25600 | torch-tensorrt | n/a | 2167.5 | 563.6 | n/a |
| micro v8 k1 n25600 | jax | 1339.6 | 1.3 | 267.2 | n/a |
| micro v8 k1 n256000 | torch-tensorrt | n/a | 2284.6 | 27314.8 | n/a |
| micro v8 k1 n256000 | jax | 1537.7 | 16.6 | 15336.3 | n/a |
| micro v9 k1 n25600 | torch-tensorrt | n/a | 2240.1 | 936.7 | n/a |
| micro v9 k1 n25600 | jax | 374.8 | 0.8 | 263.3 | n/a |
| micro v9 k1 n256000 | torch-tensorrt | n/a | 2148.9 | 1565.2 | n/a |
| micro v9 k1 n256000 | jax | 810.4 | 1.2 | 350.7 | n/a |
| mixer_b16 | torch-tensorrt | n/a | 5626.9 | 2465.0 | n/a |
| mixer_b16 | jax | 7306.4 | 6.9 | 2233.4 | n/a |
| resnet18-b1 | torch-tensorrt | n/a | 4629.9 | 1010.1 | n/a |
| resnet18-b1 | jax | 598.7 | 3.9 | 1097.0 | n/a |
| resnet18-b8 | torch-tensorrt | n/a | 4793.9 | 2027.1 | n/a |
| resnet18-b8 | jax | 912.6 | 4.4 | 2277.3 | n/a |
| smollm2-135m | torch-tensorrt | n/a | 22029.7 | 3751.1 | n/a |
| smollm2-135m | jax | 10605.3 | 18.1 | 2647.8 | n/a |
| tiny-gpt2 | torch-tensorrt | n/a | 1776.2 | 1393.5 | n/a |
| tiny-gpt2 | jax | 1147.7 | 1.6 | 88.6 | n/a |
| vit-tiny | torch-tensorrt | n/a | 7839.8 | 1062.7 | n/a |
| vit-tiny | jax | 6408.9 | 7.6 | 736.8 | n/a |
| vit-tiny | iree | 1680.0 | 39.5 | 24999.2 | n/a |

## Dynamic sequence length (median steady_us per length)

| system | length | median_us | loops | bridges | recompiles |
|---|---|---|---|---|---|
| ours | 32 | 1039.8 | 168 | 167 | 0 |
| ours | 48 | 1438.5 | 168 | 167 | 0 |
| ours | 64 | 1502.5 | 168 | 167 | 0 |
| ours | 96 | 1790.2 | 168 | 167 | 0 |
| ours | 128 | 2120.0 | 168 | 167 | 0 |
| torch-compile-dynamic | 32 | 1256.2 | 0 | 0 | 0 |
| torch-compile-dynamic | 48 | 1687.2 | 0 | 0 | 0 |
| torch-compile-dynamic | 64 | 1741.8 | 0 | 0 | 0 |
| torch-compile-dynamic | 96 | 1890.7 | 0 | 0 | 0 |
| torch-compile-dynamic | 128 | 2134.7 | 0 | 0 | 0 |
| torch-compile-static | 32 | 1181.0 | 0 | 0 | 4 |
| torch-compile-static | 48 | 1562.0 | 0 | 0 | 4 |
| torch-compile-static | 64 | 1595.0 | 0 | 0 | 4 |
| torch-compile-static | 96 | 1831.5 | 0 | 0 | 4 |
| torch-compile-static | 128 | 2066.9 | 0 | 0 | 4 |
| torch-eager | 32 | 2480.0 | 0 | 0 | 0 |
| torch-eager | 48 | 2647.2 | 0 | 0 | 0 |
| torch-eager | 64 | 2646.0 | 0 | 0 | 0 |
| torch-eager | 96 | 2702.8 | 0 | 0 | 0 |
| torch-eager | 128 | 2800.5 | 0 | 0 | 0 |

