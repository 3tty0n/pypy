# Paper benchmark summary

## Microbenchmarks (median steady_us over rounds)

| variant | k | n | fused (ours) | app-level (ours) | eager (ours) | nojit (ours) | torch.compile | torch eager | JAX/XLA | IREE | Triton | speedup vs compile | interp tax | ours/Triton |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1 | 1000000 | 29.0 | 30.1 | 76.6 | 76.8 | 55.2 | 78.3 | n/a | n/a | n/a | 1.90x | 1.04x | n/a |
| 0 | 4 | 10000 | 11.9 | 18.4 | 38.5 | 38.9 | 51.1 | 59.6 | n/a | n/a | n/a | 4.28x | 1.54x | n/a |
| 0 | 4 | 100000 | 12.7 | 19.9 | 43.5 | 43.2 | 50.2 | 58.1 | n/a | n/a | n/a | 3.96x | 1.57x | n/a |
| 0 | 4 | 1000000 | 39.8 | 115.8 | 307.7 | 306.1 | 68.7 | 312.3 | n/a | n/a | n/a | 1.73x | 2.91x | n/a |
| 0 | 4 | 10000000 | 331.2 | 1133.6 | 3026.5 | 3026.8 | 646.0 | 3044.9 | n/a | n/a | n/a | 1.95x | 3.42x | n/a |
| 0 | 8 | 1000000 | 70.7 | 231.4 | 614.1 | 612.0 | 133.0 | 624.1 | n/a | n/a | n/a | 1.88x | 3.27x | n/a |
| 1 | 4 | 1000000 | 41.7 | 119.1 | 310.0 | 310.8 | 307.6 | 316.6 | n/a | n/a | n/a | 7.37x | 2.86x | n/a |
| 2 | 4 | 1000000 | 44.0 | 119.2 | 311.2 | 310.7 | 307.4 | 316.6 | n/a | n/a | n/a | 6.98x | 2.71x | n/a |
| 3 | 4 | 1000000 | 94.3 | 200.0 | 357.3 | 394.8 | 167.1 | 375.2 | n/a | n/a | n/a | 1.77x | 2.12x | n/a |
| 4 | 4 | 1000000 | 52.2 | 144.4 | 339.6 | 335.8 | 332.0 | 342.2 | n/a | n/a | n/a | 6.36x | 2.76x | n/a |
| 5 | 4 | 1000000 | 55.6 | 149.1 | 335.7 | 335.8 | 332.4 | 342.1 | n/a | n/a | n/a | 5.98x | 2.68x | n/a |
| 6 | 1 | 25600 | 276.9 | 268.0 | 279.5 | 277.1 | 392.4 | 392.8 | n/a | n/a | n/a | 1.42x | 0.97x | n/a |
| 6 | 1 | 256000 | 1030.8 | 996.1 | 1015.7 | 1011.6 | 1283.8 | 1260.2 | n/a | n/a | n/a | 1.25x | 0.97x | n/a |
| 7 | 1 | 25600 | 640.2 | 790.5 | 666.9 | 667.1 | 849.5 | 801.2 | n/a | n/a | n/a | 1.33x | 1.23x | n/a |
| 7 | 1 | 256000 | 2798.7 | 2927.3 | 2827.5 | 2833.0 | 3077.6 | 3102.1 | n/a | n/a | n/a | 1.10x | 1.05x | n/a |
| 8 | 1 | 25600 | 464.2 | 477.3 | 777.9 | 778.0 | 519.1 | 555.1 | n/a | n/a | n/a | 1.12x | 1.03x | n/a |
| 8 | 1 | 256000 | 22334.6 | 22349.4 | 25084.3 | 25097.9 | 29742.8 | 27344.4 | n/a | n/a | n/a | 1.33x | 1.00x | n/a |
| 9 | 1 | 25600 | 755.7 | 746.0 | 765.5 | 765.9 | 865.9 | 798.9 | n/a | n/a | n/a | 1.15x | 0.99x | n/a |
| 9 | 1 | 256000 | 544.5 | 568.2 | 582.4 | 572.7 | 685.1 | 587.1 | n/a | n/a | n/a | 1.26x | 1.04x | n/a |
| 10 | 1 | 25600 | 4064.9 | 8652.0 | 4818.6 | 4818.5 | 3190.9 | 3716.2 | n/a | n/a | n/a | 0.78x | 2.13x | n/a |
| 10 | 1 | 191488 | 94343.8 | n/a | 99019.2 | 99417.7 | 78479.0 | 78748.5 | n/a | n/a | n/a | 0.83x | n/a | n/a |
| 11 | 1 | 25600 | 3.4 | 24.5 | 20.5 | 20.8 | 72.7 | 26.9 | n/a | n/a | n/a | 21.26x | 7.17x | n/a |
| 11 | 1 | 256000 | 14.9 | 159.5 | 113.3 | 113.1 | 155.2 | 127.2 | n/a | n/a | n/a | 10.40x | 10.69x | n/a |
| 12 | 1 | 25600 | 98.8 | 91.0 | 98.9 | 97.9 | 144.3 | 139.8 | n/a | n/a | n/a | 1.46x | 0.92x | n/a |
| 12 | 1 | 256000 | 362.5 | 338.5 | 342.8 | 338.6 | 496.5 | 498.0 | n/a | n/a | n/a | 1.37x | 0.93x | n/a |
| 13 | 1 | 25600 | 440.8 | 446.8 | 526.2 | 526.2 | 595.6 | 592.7 | n/a | n/a | n/a | 1.35x | 1.01x | n/a |
| 13 | 1 | 256000 | 4750.7 | 4755.6 | 5428.6 | 5423.0 | 4995.0 | 4882.6 | n/a | n/a | n/a | 1.05x | 1.00x | n/a |

## Precision sweep (median steady_us)

| dtype | variant | k | n | fused (ours) | app-level (ours) | eager (ours) | nojit (ours) | torch.compile | torch eager | JAX/XLA | IREE | Triton | speedup vs compile | interp tax | ours/Triton |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| float16 | 8 | 1 | 256000 | 712.6 | n/a | 2055.7 | 2055.7 | 866.5 | 1469.0 | n/a | n/a | n/a | 1.22x | n/a | n/a |
| float16 | 13 | 1 | 256000 | 220.8 | n/a | 348.8 | 342.1 | 439.8 | 415.0 | n/a | n/a | n/a | 1.99x | n/a | n/a |
| float32 | 8 | 1 | 256000 | 1870.5 | n/a | 4379.7 | 4381.1 | 2030.8 | 2665.0 | n/a | n/a | n/a | 1.09x | n/a | n/a |
| float32 | 13 | 1 | 256000 | 316.4 | n/a | 632.2 | 625.9 | 549.4 | 640.3 | n/a | n/a | n/a | 1.74x | n/a | n/a |

## Guards / graph breaks (launches per iter, torch graph breaks)

| variant | n | launches/iter (fused) | torch graphs | torch breaks |
|---|---|---|---|---|
| 1 | 1000000 | 1.1 | 1 | 0 |
| 2 | 1000000 | 1.1 | 1 | 0 |
| 3 | 1000000 | 2.0 | 2 | 1 |
| 4 | 1000000 | 1.0 | 1 | 0 |
| 5 | 1000000 | 1.1 | 2 | 1 |

## End-to-end models (median steady_us, ratio to torch.compile)

| model | ours | torch.compile | torch eager | JAX/XLA | IREE | ratio ours/compile | ratio jax/compile | correctness |
|---|---|---|---|---|---|---|---|---|
| bert-mini | 672.5 | 778.1 | 1802.2 | n/a | n/a | 0.86x | n/a | 2.09808e-05 |
| bert-tiny | 392.4 | 532.5 | 1090.0 | n/a | n/a | 0.74x | n/a | 3.05176e-05 |
| distilgpt2 | 1374.5 | 1323.8 | 2739.8 | n/a | n/a | 1.04x | n/a | 0.000144958 |
| mixer_b16 | 3802.5 | 3127.6 | 3037.3 | n/a | n/a | 1.22x | n/a | 3.71933e-05 |
| resnet18-b1 | 765.6 | 1058.4 | 1568.0 | n/a | n/a | 0.72x | n/a | 0.00524807 |
| resnet18-b8 | 3193.9 | 2186.6 | 2255.2 | n/a | n/a | 1.46x | n/a | 0.00887299 |
| smollm2-135m | 4062.3 | 5199.4 | 15699.0 | n/a | n/a | 0.78x | n/a | 0.000151277 |
| tiny-gpt2 | 195.6 | 418.4 | 1643.7 | n/a | n/a | 0.47x | n/a | 2.6077e-08 |
| vit-tiny | 1671.3 | 2147.8 | 3722.4 | n/a | n/a | 0.78x | n/a | 1.23978e-05 |

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

