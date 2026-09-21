# Paper benchmark summary

## Microbenchmarks (median steady_us over rounds)

| variant | k | n | fused (ours) | app-level (ours) | eager (ours) | nojit (ours) | torch.compile | compile-ro | compile-mat | torch eager | JAX/XLA | IREE | Triton | speedup vs compile | interp tax | ours/Triton |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1 | 1000000 | 29.0 | n/a | 77.3 | 77.3 | 53.9 | 119.7 | 106.5 | 78.7 | 76.5 | 7158.4 | 31.2 | 1.86x | n/a | 0.93x |
| 0 | 4 | 10000 | 13.3 | n/a | 39.3 | 39.7 | 50.1 | 124.3 | 124.1 | 57.4 | 49.2 | 519.8 | 19.7 | 3.78x | n/a | 0.67x |
| 0 | 4 | 100000 | 14.4 | n/a | 43.4 | 43.3 | 53.7 | 121.1 | 104.3 | 69.3 | 65.1 | 1091.6 | 21.1 | 3.74x | n/a | 0.68x |
| 0 | 4 | 1000000 | 39.8 | n/a | 306.4 | 306.7 | 68.5 | 134.0 | 133.2 | 312.7 | 96.1 | 7001.5 | 39.3 | 1.72x | n/a | 1.01x |
| 0 | 4 | 10000000 | 329.3 | n/a | 3027.0 | 3027.3 | 646.6 | 1213.5 | 1197.2 | 3046.0 | 951.5 | 60279.2 | 327.5 | 1.96x | n/a | 1.01x |
| 0 | 8 | 1000000 | 70.1 | n/a | 612.7 | 613.4 | 132.8 | 198.7 | 194.8 | 625.4 | 176.4 | 7143.2 | 70.8 | 1.89x | n/a | 0.99x |
| 1 | 4 | 1000000 | 41.6 | n/a | 312.0 | 311.1 | 308.2 | 396.7 | 401.7 | 317.0 | 101.1 | 8212.4 | 43.7 | 7.40x | n/a | 0.95x |
| 2 | 4 | 1000000 | 42.7 | n/a | 311.0 | 311.0 | 72.7 | 139.5 | 137.8 | 326.3 | 171.3 | 10838.9 | 69.1 | 1.70x | n/a | 0.62x |
| 3 | 4 | 1000000 | 99.9 | n/a | 361.7 | 369.1 | 168.5 | 425.7 | 380.1 | 389.4 | 332.5 | 84201.4 | 126.8 | 1.69x | n/a | 0.79x |
| 4 | 4 | 1000000 | 53.7 | n/a | 336.1 | 335.9 | 334.0 | 427.0 | 422.2 | 342.7 | 138.3 | 13046.9 | 69.5 | 6.22x | n/a | 0.77x |
| 5 | 4 | 1000000 | 53.6 | n/a | 335.8 | 335.8 | 333.0 | 419.0 | 414.6 | 342.8 | 134.4 | 12359.3 | 69.7 | 6.21x | n/a | 0.77x |
| 6 | 1 | 25600 | 280.0 | n/a | 279.6 | 279.1 | 413.5 | 414.3 | 464.9 | 405.9 | 107.6 | 5391.7 | 895.9 | 1.48x | n/a | 0.31x |
| 6 | 1 | 256000 | 1018.1 | n/a | 1019.6 | 1018.4 | 1252.2 | 1338.1 | 1291.6 | 1261.2 | 760.4 | 919.1 | 900.1 | 1.23x | n/a | 1.13x |
| 7 | 1 | 25600 | 661.8 | n/a | 670.6 | 670.4 | 950.5 | 938.1 | 940.5 | 836.6 | 330.2 | 11594.8 | n/a | 1.44x | n/a | n/a |
| 7 | 1 | 256000 | 2824.0 | n/a | 2841.1 | 2833.1 | 3064.2 | 3130.8 | 3147.7 | 3140.9 | 1950.7 | 31660.6 | n/a | 1.09x | n/a | n/a |
| 8 | 1 | 25600 | 464.5 | n/a | 780.1 | 780.7 | 520.4 | 535.1 | 518.8 | 566.3 | 267.4 | 921.2 | n/a | 1.12x | n/a | n/a |
| 8 | 1 | 256000 | 22341.3 | n/a | 25183.4 | 25166.1 | 29879.7 | 29779.7 | 27752.0 | 27472.4 | 15424.6 | 38681.0 | n/a | 1.34x | n/a | n/a |
| 9 | 1 | 25600 | 756.8 | n/a | 774.6 | 765.5 | 862.5 | 867.8 | 847.7 | 839.5 | 266.8 | n/a | n/a | 1.14x | n/a | n/a |
| 9 | 1 | 256000 | 570.3 | n/a | 608.2 | 571.9 | 690.3 | 705.3 | 709.8 | 601.2 | 351.4 | n/a | n/a | 1.21x | n/a | n/a |
| 10 | 1 | 25600 | 4074.7 | n/a | 4901.4 | 4821.2 | 3215.9 | 3100.1 | 3152.9 | 3813.6 | 1523.1 | 10477.7 | n/a | 0.79x | n/a | n/a |
| 10 | 1 | 191488 | 94144.5 | n/a | 99392.7 | 99420.9 | 79062.2 | 79023.0 | 76145.6 | 79246.1 | 42073.8 | 136621.0 | n/a | 0.84x | n/a | n/a |
| 11 | 1 | 25600 | 3.6 | n/a | 56.9 | 57.5 | 67.5 | 129.4 | 123.6 | 56.6 | 49.2 | 62.1 | 19.1 | 18.52x | n/a | 0.19x |
| 11 | 1 | 256000 | 15.9 | n/a | 144.0 | 140.9 | 151.6 | 239.4 | 224.8 | 145.2 | 71.3 | 207.5 | 21.3 | 9.54x | n/a | 0.74x |
| 12 | 1 | 25600 | 98.7 | n/a | 98.6 | 98.6 | 138.3 | 174.2 | 185.2 | 137.1 | 49.1 | 1835.3 | 299.5 | 1.40x | n/a | 0.33x |
| 12 | 1 | 256000 | 344.4 | n/a | 344.0 | 344.0 | 513.3 | 527.7 | 555.0 | 487.9 | 258.4 | 436.9 | 300.3 | 1.49x | n/a | 1.15x |
| 13 | 1 | 25600 | 437.4 | n/a | 528.4 | 526.7 | 617.8 | 640.5 | 630.7 | 588.3 | 123.6 | 2796.8 | 1269.7 | 1.41x | n/a | 0.34x |
| 13 | 1 | 256000 | 4744.5 | n/a | 5425.1 | 5432.9 | 5047.4 | 5070.1 | 5026.9 | 4879.7 | 3718.4 | 5169.3 | 4692.8 | 1.06x | n/a | 1.01x |

## Precision sweep (median steady_us)

| dtype | variant | k | n | fused (ours) | app-level (ours) | eager (ours) | nojit (ours) | torch.compile | compile-ro | compile-mat | torch eager | JAX/XLA | IREE | Triton | speedup vs compile | interp tax | ours/Triton |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| float16 | 8 | 1 | 256000 | 707.9 | n/a | 2059.4 | 2063.0 | 892.8 | 889.1 | 823.8 | 1476.2 | 635.5 | 1908.5 | n/a | 1.26x | n/a | n/a |
| float16 | 11 | 1 | 256000 | 22.4 | n/a | 60.5 | 59.6 | 175.4 | 260.8 | 226.2 | 176.5 | 52.9 | 195.5 | 23.2 | 7.84x | n/a | 0.96x |
| float16 | 13 | 1 | 256000 | 227.3 | n/a | 339.7 | 345.7 | 374.8 | 409.1 | 412.9 | 467.7 | 100.6 | 616.2 | 150.4 | 1.65x | n/a | 1.51x |
| float32 | 8 | 1 | 256000 | 1393.8 | n/a | 3914.3 | 3919.3 | 2050.7 | 2060.6 | 1514.8 | 2681.0 | 1262.7 | 7325.7 | n/a | 1.47x | n/a | n/a |
| float32 | 11 | 1 | 256000 | 7.3 | n/a | 41.6 | 38.6 | 198.1 | 233.3 | 265.5 | 144.0 | 54.7 | 91.9 | 23.8 | 27.31x | n/a | 0.30x |
| float32 | 13 | 1 | 256000 | 271.7 | n/a | 583.7 | 584.9 | 560.3 | 632.5 | 567.7 | 807.4 | 217.2 | 1318.6 | 130.9 | 2.06x | n/a | 2.08x |

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
| bert-base | 1850.1 | 2308.6 | 2118.4 | 1873.3 | 4906.9 | 1465.2 | 26092.9 | 2180.3 | 0.80x | 0.63x | 64.3 | 8.10623e-05 | 0.000504128 | pass |
| bert-mini | 432.4 | 835.5 | 499.9 | 438.3 | 1980.0 | 243.1 | 3667.9 | 748.3 | 0.52x | 0.29x | 24.3 | 1.90735e-05 | 0.000445395 | pass |
| bert-tiny | 288.7 | 528.4 | 326.6 | 267.8 | 1263.0 | 111.2 | 1343.0 | 513.6 | 0.55x | 0.21x | 14.3 | 3.05176e-05 | 0.000510627 | pass |
| deit-tiny | 1029.3 | 2360.3 | 1417.3 | 1299.9 | 3917.7 | 737.8 | n/a | 1116.7 | 0.44x | 0.31x | 76.3 | 1.07288e-05 | 0.000134346 | pass |
| distilgpt2 | 1289.6 | 1382.0 | 1309.7 | 1280.2 | 3014.8 | 1004.2 | 17191.6 | 1724.1 | 0.93x | 0.73x | 38.2 | 0.000190735 | 0.0034415 | pass |
| gpt2 | 2102.7 | 2188.9 | 2061.8 | 2010.9 | 4873.4 | 1642.5 | 28774.3 | 3283.0 | 0.96x | 0.75x | 74.2 | 0.000854492 | 0.00663646 | pass |
| gpt2-medium | 4769.8 | 5021.3 | 4765.1 | 4422.0 | 10227.7 | 3873.2 | n/a | 5703.7 | 0.95x | 0.77x | 146.2 | 0.00343323 | 0.00707528 | pass |
| mixer_b16 | 2712.8 | 3167.8 | 3086.0 | 3137.8 | 2914.8 | 2249.4 | 674484.5 | 2505.9 | 0.86x | 0.71x | 63.4 | 3.62396e-05 | 0.000280422 | pass |
| qwen2.5-0.5b | 6526.5 | 7444.0 | 7105.3 | 7127.7 | 14275.0 | 5231.7 | 86466.8 | 7101.8 | 0.88x | 0.70x | 194.1 | 0.000150681 | 0.000508814 | pass |
| resnet18-b1 | 724.4 | 983.9 | 925.2 | 1061.5 | 1580.4 | 1113.9 | n/a | 998.6 | 0.74x | 1.13x | 39.0 | 0.00963783 | 0.0294047 | pass |
| resnet18-b8 | 2736.9 | 2188.4 | 2134.5 | 2136.1 | 2275.3 | 2286.9 | n/a | 1950.7 | 1.25x | 1.05x | 39.0 | 0.00571394 | 0.0293988 | pass |
| smollm2-1.7b | 14243.3 | 18150.5 | 17852.2 | 17898.6 | 19565.3 | 13536.5 | n/a | 16776.2 | 0.78x | 0.75x | 170.1 | 7.49826e-05 | 0.000587335 | pass |
| smollm2-135m | 3674.6 | 6078.9 | 3785.1 | 3853.2 | 17769.2 | 2666.4 | 19251.2 | 3749.5 | 0.60x | 0.44x | 212.1 | 0.000201941 | 0.000589316 | pass |
| smollm2-360m | 6077.6 | 7688.1 | 6281.5 | 6325.6 | 17882.8 | 4588.3 | 33036.1 | 6488.6 | 0.79x | 0.60x | 226.1 | 4.36306e-05 | 0.00060394 | pass |
| tiny-gpt2 | 249.5 | 459.9 | 208.6 | 297.0 | 1644.3 | 119.1 | 210.7 | 590.9 | 0.54x | 0.26x | 13.2 | 2.6077e-08 | 2.5808e-06 | pass |
| vit-base | 3723.1 | 4660.7 | 4612.8 | 4431.7 | 4474.6 | 3186.5 | 883623.9 | 3798.1 | 0.80x | 0.68x | 76.3 | 9.23872e-06 | 0.000248432 | pass |
| vit-tiny | 1033.5 | 2540.6 | 1438.1 | 1334.3 | 5089.2 | 740.0 | 25163.9 | 1166.0 | 0.41x | 0.29x | 76.3 | 1.04904e-05 | 0.000257036 | pass |

## Correctness on five derived inputs (worst case per model; ours is the reference)

| model | system | max maxabsdiff | min argmax match | tol | passed |
|---|---|---|---|---|---|
| bert-base | jax | 9.16e-05 | 1.000 | 0.000535593 | 5/5 |
| bert-base | torch-compile | 9.3e-05 | 1.000 | 0.000535593 | 5/5 |
| bert-base | torch-eager | 8.01e-05 | 1.000 | 0.000535593 | 5/5 |
| bert-mini | jax | 2.48e-05 | 1.000 | 0.0005167 | 5/5 |
| bert-mini | torch-compile | 2.29e-05 | 1.000 | 0.0005167 | 5/5 |
| bert-mini | torch-eager | 2.67e-05 | 1.000 | 0.0005167 | 5/5 |
| bert-tiny | jax | 3.24e-05 | 1.000 | 0.00047357 | 5/5 |
| bert-tiny | torch-compile | 5.15e-05 | 1.000 | 0.00047357 | 5/5 |
| bert-tiny | torch-eager | 3.91e-05 | 1.000 | 0.00047357 | 5/5 |
| deit-tiny | jax | 1.12e-05 | 1.000 | 0.000133298 | 5/5 |
| deit-tiny | torch-compile | 1.34e-05 | 1.000 | 0.000133298 | 5/5 |
| deit-tiny | torch-eager | 1.32e-05 | 1.000 | 0.000133298 | 5/5 |
| distilgpt2 | jax | 0.000221 | 1.000 | 0.00311393 | 5/5 |
| distilgpt2 | torch-compile | 0.000282 | 1.000 | 0.00311393 | 5/5 |
| distilgpt2 | torch-eager | 0.000282 | 1.000 | 0.00311393 | 5/5 |
| gpt2 | jax | 0.00335 | 1.000 | 0.00648755 | 5/5 |
| gpt2 | torch-compile | 0.00463 | 1.000 | 0.00648755 | 5/5 |
| gpt2 | torch-eager | 0.00389 | 1.000 | 0.00648755 | 5/5 |
| gpt2-medium | jax | 0.00516 | 1.000 | 0.0073956 | 5/5 |
| gpt2-medium | torch-compile | 0.00645 | 1.000 | 0.0073956 | 5/5 |
| gpt2-medium | torch-eager | 0.00439 | 1.000 | 0.0073956 | 5/5 |
| mixer_b16 | jax | 5.63e-05 | 1.000 | 0.000280097 | 5/5 |
| mixer_b16 | torch-compile | 5.91e-05 | 1.000 | 0.000280097 | 5/5 |
| mixer_b16 | torch-eager | 5.15e-05 | 1.000 | 0.000280097 | 5/5 |
| qwen2.5-0.5b | jax | 0.000394 | 1.000 | 0.000553048 | 5/5 |
| qwen2.5-0.5b | torch-compile | 0.000137 | 1.000 | 0.000553048 | 5/5 |
| qwen2.5-0.5b | torch-eager | 0.000149 | 1.000 | 0.000553048 | 5/5 |
| resnet18-b1 | jax | 0.0167 | 1.000 | 0.0299742 | 5/5 |
| resnet18-b1 | torch-compile | 0.00942 | 1.000 | 0.0299742 | 5/5 |
| resnet18-b1 | torch-eager | 0.0158 | 1.000 | 0.0299742 | 5/5 |
| resnet18-b8 | jax | 0.0123 | 1.000 | 0.0299708 | 5/5 |
| resnet18-b8 | torch-compile | 0.0122 | 1.000 | 0.0299708 | 5/5 |
| resnet18-b8 | torch-eager | 0.00725 | 1.000 | 0.0299708 | 5/5 |
| smollm2-1.7b | jax | 0.00027 | 1.000 | 0.000504007 | 5/5 |
| smollm2-1.7b | torch-compile | 0.000765 | 1.000 | 0.000504007 | 2/5 |
| smollm2-1.7b | torch-eager | 0.00107 | 1.000 | 0.000504007 | 3/5 |
| smollm2-135m | jax | 0.000275 | 1.000 | 0.000808265 | 5/5 |
| smollm2-135m | torch-compile | 0.000168 | 1.000 | 0.000808265 | 5/5 |
| smollm2-135m | torch-eager | 0.000147 | 1.000 | 0.000808265 | 5/5 |
| smollm2-360m | jax | 0.000298 | 1.000 | 0.000582302 | 5/5 |
| smollm2-360m | torch-compile | 0.00022 | 1.000 | 0.000582302 | 5/5 |
| smollm2-360m | torch-eager | 0.000219 | 1.000 | 0.000582302 | 5/5 |
| tiny-gpt2 | jax | 3.17e-08 | 1.000 | 2.57908e-06 | 5/5 |
| tiny-gpt2 | torch-compile | 3.73e-08 | 1.000 | 2.57908e-06 | 5/5 |
| tiny-gpt2 | torch-eager | 3.73e-08 | 1.000 | 2.57908e-06 | 5/5 |
| vit-base | jax | 1.19e-05 | 1.000 | 0.000248661 | 5/5 |
| vit-base | torch-compile | 1.17e-05 | 1.000 | 0.000248661 | 5/5 |
| vit-base | torch-eager | 1.07e-05 | 1.000 | 0.000248661 | 5/5 |
| vit-tiny | jax | 1.1e-05 | 1.000 | 0.000258005 | 5/5 |
| vit-tiny | torch-compile | 1.19e-05 | 1.000 | 0.000258005 | 5/5 |
| vit-tiny | torch-eager | 1.41e-05 | 1.000 | 0.000258005 | 5/5 |

## Batch-size sweep (median steady_us per forward; per_seq = steady/B)

| model | batch | ours | torch.compile | compile-ro | torch eager | JAX/XLA | ratio ours/compile | ratio ours/jax | per_seq ours | per_seq compile | per_seq jax | rows identical | failed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bert-base | 1 | 1850.7 | 2290.7 | 2092.7 | 4504.2 | 1469.2 | 0.81x | 1.26x | 1850.7 | 2290.7 | 1469.2 | yes |  |
| bert-base | 2 | 2922.8 | 3047.9 | 2928.2 | 4555.6 | 2369.9 | 0.96x | 1.23x | 1461.4 | 1523.9 | 1184.9 | yes |  |
| bert-base | 4 | 4666.7 | 5103.1 | 5041.5 | 4774.6 | 3809.8 | 0.91x | 1.22x | 1166.7 | 1275.8 | 952.5 | yes |  |
| bert-base | 8 | 7766.7 | 8204.2 | 8137.5 | 8030.7 | 6913.8 | 0.95x | 1.12x | 970.8 | 1025.5 | 864.2 | yes |  |
| bert-base | 16 | 13708.3 | 14350.0 | 14265.4 | 14169.2 | 12817.1 | 0.96x | 1.07x | 856.8 | 896.9 | 801.1 | yes |  |
| bert-base | 32 | 26470.0 | 27694.4 | 27517.2 | 28038.6 | 25538.8 | 0.96x | 1.04x | 827.2 | 865.4 | 798.1 | yes |  |
| distilgpt2 | 1 | 1288.0 | 1352.5 | 1293.4 | 2796.2 | 1002.2 | 0.95x | 1.29x | 1288.0 | 1352.5 | 1002.2 | yes |  |
| distilgpt2 | 2 | 1852.2 | 1743.6 | 1671.5 | 2794.7 | 1555.8 | 1.06x | 1.19x | 926.1 | 871.8 | 777.9 | yes |  |
| distilgpt2 | 4 | 2989.1 | 2788.5 | 2736.7 | 3130.3 | 2619.0 | 1.07x | 1.14x | 747.3 | 697.1 | 654.8 | yes |  |
| distilgpt2 | 8 | 5175.3 | 5084.2 | 5035.5 | 5877.6 | 4790.6 | 1.02x | 1.08x | 646.9 | 635.5 | 598.8 | yes |  |
| distilgpt2 | 16 | 9424.7 | 9142.3 | 9100.1 | 10764.5 | 9032.9 | 1.03x | 1.04x | 589.0 | 571.4 | 564.6 | yes |  |
| distilgpt2 | 32 | 18403.3 | 17965.5 | 17907.6 | 20957.5 | 18098.2 | 1.02x | 1.02x | 575.1 | 561.4 | 565.6 | yes |  |

## Ablations (median steady_us)

| experiment | variant | model | steady_us | launches/iter | note |
|---|---|---|---|---|---|
| budget_mb | 64 | distilgpt2 | 1293.5 | n/a |  |
| budget_mb | 8 | distilgpt2 | 1291.9 | n/a |  |
| flat_block | 256 | distilgpt2 | 1264.1 | n/a |  |
| flat_block | 256 | resnet18 | 631.4 | n/a |  |
| flat_block | 4096 | distilgpt2 | 1298.5 | n/a |  |
| flat_block | 4096 | resnet18 | 689.0 | n/a |  |
| fusion | off | distilgpt2 | 1949.6 | n/a | enable_opts minus tensor |
| fusion | on | distilgpt2 | 1287.3 | n/a |  |
| max_inputs | mi4 | bert-mini | 465.8 | 34.3 |  |
| max_inputs | mi4 | distilgpt2 | 1329.5 | 45.2 |  |
| max_inputs | mi4 | mixer_b16 | 2741.8 | 76.4 |  |
| max_inputs | mi6 | bert-mini | 435.7 | 24.3 |  |
| max_inputs | mi6 | distilgpt2 | 1302.2 | 38.2 |  |
| max_inputs | mi6 | mixer_b16 | 2687.9 | 63.4 |  |
| max_inputs | mi8 | bert-mini | 429.7 | 24.3 |  |
| max_inputs | mi8 | distilgpt2 | 1296.6 | 38.2 |  |
| max_inputs | mi8 | mixer_b16 | 2682.8 | 63.4 |  |
| precision | float16 | distilgpt2 | 927.7 | n/a |  |
| precision | float16 | smollm2-135m | 2197.9 | n/a |  |
| precision | float32 | distilgpt2 | 1289.1 | n/a |  |
| precision | float32 | smollm2-135m | 3657.2 | n/a |  |
| tf32 | fp32 | resnet18-b1 | 791.4 | n/a |  |
| tf32 | fp32 | resnet18-b8 | 3198.1 | n/a |  |
| tf32 | tf32 | resnet18-b1 | 720.7 | n/a |  |
| tf32 | tf32 | resnet18-b8 | 2719.1 | n/a |  |
| tf32 | torch-fp32 | resnet18-b8 | 3068.0 | n/a |  |

## Deoptimization cost (median over rounds, us per iteration)

| system | pattern | steady_us | first_fail_us | after_fail_us | peak_us | cold_us | launches/it | loops | bridges |
|---|---|---|---|---|---|---|---|---|---|
| ours | never | 38.1 | n/a | n/a | 72 | 5400 | 2.02 | 0 | 0 |
| torch-compile | never | 188.1 | n/a | n/a | 513 | 608155 | n/a | 0 | n/a |
| torch-compile-ro | never | 357.5 | n/a | n/a | 376 | 699546 | n/a | 0 | n/a |
| torch-eager | never | 110.4 | n/a | n/a | 122 | 38220 | n/a | 0 | n/a |
| ours | alternate | 38.9 | 61 | 58 | 317 | 5178 | 2.17 | 0 | 2 |
| torch-compile | alternate | 188.7 | 141040 | 216 | 141040 | 685933 | n/a | 2 | n/a |
| torch-compile-ro | alternate | 277.5 | 135891 | 381 | 135891 | 560487 | n/a | 2 | n/a |
| torch-eager | alternate | 122.8 | 125 | 125 | 133 | 44375 | n/a | 0 | n/a |
| ours | both-hot | 37.0 | n/a | n/a | 63 | 4572 | 2.02 | 0 | 0 |
| torch-compile | both-hot | 189.5 | n/a | n/a | 213 | 552810 | n/a | 0 | n/a |
| torch-compile-ro | both-hot | 286.8 | n/a | n/a | 352 | 600158 | n/a | 0 | n/a |
| torch-eager | both-hot | 109.4 | n/a | n/a | 125 | 35670 | n/a | 0 | n/a |
| ours | fresh | 67.0 | 65 | 68 | 1565 | 4134 | 7.00 | 0 | 63 |
| torch-compile | fresh | 163.6 | 168 | 167 | 195 | 560803 | n/a | 0 | n/a |
| torch-compile-ro | fresh | 150.1 | 155 | 152 | 230 | 586619 | n/a | 0 | n/a |
| torch-eager | fresh | 109.7 | 115 | 110 | 167 | 38168 | n/a | 0 | n/a |
| ours | probe-a | 35.0 | n/a | 56 | 2731 | 5111 | 2.25 | 5 | 3 |
| ours | probe-e | 17.9 | 2766 | 33 | 2766 | 128 | 1.09 | 3 | 4 |

## Compilation overhead (median; break-even vs torch eager, iterations)

| workload | system | compile_ms | first_run_ms | steady_us | break-even |
|---|---|---|---|---|---|
| bert-base | torch-compile | n/a | 2169.3 | 2308.6 | 788 |
| bert-base | torch-compile-ro | n/a | 2444.6 | 2118.4 | 833 |
| bert-base | torch-compile-mat | n/a | 2510.3 | 1873.3 | 787 |
| bert-base | torch-tensorrt | n/a | 9949.8 | 2180.3 | 3604 |
| bert-base | jax | 9484.3 | 8.0 | 1465.2 | 2722 |
| bert-base | iree | 1752.9 | 128.2 | 26092.9 | n/a |
| bert-base | ours | n/a | 520.8 | 1850.1 | 130 |
| bert-mini | torch-compile | n/a | 1405.0 | 835.5 | 1135 |
| bert-mini | torch-compile-ro | n/a | 1585.7 | 499.9 | 999 |
| bert-mini | torch-compile-mat | n/a | 2957.7 | 438.3 | 1849 |
| bert-mini | torch-tensorrt | n/a | 4676.4 | 748.3 | 3710 |
| bert-mini | jax | 8051.9 | 5.8 | 243.1 | 4578 |
| bert-mini | iree | 1155.3 | 22.4 | 3667.9 | n/a |
| bert-mini | ours | n/a | 202.1 | 432.4 | 62 |
| bert-tiny | torch-compile | n/a | 1235.6 | 528.4 | 1515 |
| bert-tiny | torch-compile-ro | n/a | 1337.1 | 326.6 | 1297 |
| bert-tiny | torch-compile-mat | n/a | 2642.1 | 267.8 | 2531 |
| bert-tiny | torch-tensorrt | n/a | 4968.1 | 513.6 | 6465 |
| bert-tiny | jax | 6094.2 | 1.8 | 111.2 | 5186 |
| bert-tiny | iree | 891.7 | 15.3 | 1343.0 | n/a |
| bert-tiny | ours | n/a | 142.4 | 288.7 | 20 |
| deit-tiny | torch-compile | n/a | 1839.6 | 2360.3 | 1094 |
| deit-tiny | torch-compile-ro | n/a | 1846.5 | 1417.3 | 684 |
| deit-tiny | torch-compile-mat | n/a | 3671.9 | 1299.9 | 1351 |
| deit-tiny | torch-tensorrt | n/a | 7360.9 | 1116.7 | 2579 |
| deit-tiny | jax | 6551.0 | 7.1 | 737.8 | 2020 |
| deit-tiny | ours | n/a | 147.2 | 1029.3 | 4 |
| distilgpt2 | torch-compile | n/a | 1703.5 | 1382.0 | 955 |
| distilgpt2 | torch-compile-ro | n/a | 1627.8 | 1309.7 | 871 |
| distilgpt2 | torch-compile-mat | n/a | 2191.8 | 1280.2 | 1181 |
| distilgpt2 | torch-tensorrt | n/a | 16834.0 | 1724.1 | 12931 |
| distilgpt2 | jax | 8001.0 | 5.4 | 1004.2 | 3911 |
| distilgpt2 | iree | 923.7 | 83.3 | 17191.6 | n/a |
| distilgpt2 | ours | n/a | 658.4 | 1289.6 | 298 |
| gpt2 | torch-compile | n/a | 2295.6 | 2215.3 | 830 |
| gpt2 | torch-compile-ro | n/a | 2211.4 | 2056.8 | 752 |
| gpt2 | torch-compile-mat | n/a | 3593.2 | 2006.0 | 1232 |
| gpt2 | torch-tensorrt | n/a | 31115.3 | 3283.0 | 20365 |
| gpt2 | jax | 8406.7 | 9.6 | 1644.4 | 2618 |
| gpt2 | iree | 1379.2 | 136.3 | 28784.0 | n/a |
| gpt2 | ours | n/a | 724.9 | 2106.9 | 215 |
| gpt2-medium | torch-compile | n/a | 3133.3 | 5021.3 | 574 |
| gpt2-medium | torch-compile-ro | n/a | 3292.7 | 4765.1 | 576 |
| gpt2-medium | torch-compile-mat | n/a | 3442.0 | 4422.0 | 568 |
| gpt2-medium | torch-tensorrt | n/a | 51434.4 | 5703.7 | 11337 |
| gpt2-medium | jax | 7617.1 | 16.5 | 3873.2 | 1178 |
| gpt2-medium | ours | n/a | 1004.5 | 4769.8 | 157 |
| micro v0 k1 n1000000 | torch-compile | n/a | 542.4 | 53.9 | 20100 |
| micro v0 k1 n1000000 | torch-compile-ro | n/a | 568.6 | 119.7 | n/a |
| micro v0 k1 n1000000 | torch-compile-mat | n/a | 577.6 | 106.5 | n/a |
| micro v0 k1 n1000000 | jax | 251.5 | 0.6 | 76.5 | 93862 |
| micro v0 k1 n1000000 | iree | 719.5 | 29.6 | 7158.4 | n/a |
| micro v0 k1 n1000000 | triton | 360.2 | 0.1 | 31.2 | 6646 |
| micro v0 k4 n10000 | torch-compile | n/a | 516.4 | 50.1 | 66074 |
| micro v0 k4 n10000 | torch-compile-ro | n/a | 528.4 | 124.3 | n/a |
| micro v0 k4 n10000 | torch-compile-mat | n/a | 643.3 | 124.1 | n/a |
| micro v0 k4 n10000 | jax | 234.2 | 0.6 | 49.2 | 24223 |
| micro v0 k4 n10000 | iree | 709.3 | 12.2 | 519.8 | n/a |
| micro v0 k4 n10000 | triton | 302.4 | 0.1 | 19.7 | 7054 |
| micro v0 k4 n100000 | torch-compile | n/a | 456.6 | 53.7 | 26614 |
| micro v0 k4 n100000 | torch-compile-ro | n/a | 482.2 | 121.1 | n/a |
| micro v0 k4 n100000 | torch-compile-mat | n/a | 649.8 | 104.3 | n/a |
| micro v0 k4 n100000 | jax | 234.5 | 0.5 | 65.1 | 46211 |
| micro v0 k4 n100000 | iree | 678.5 | 13.7 | 1091.6 | n/a |
| micro v0 k4 n100000 | triton | 304.5 | 0.1 | 21.1 | 5414 |
| micro v0 k4 n1000000 | torch-compile | n/a | 464.2 | 68.5 | 1732 |
| micro v0 k4 n1000000 | torch-compile-ro | n/a | 571.0 | 134.0 | 2965 |
| micro v0 k4 n1000000 | torch-compile-mat | n/a | 599.2 | 133.2 | 3109 |
| micro v0 k4 n1000000 | jax | 246.5 | 2.2 | 96.1 | 958 |
| micro v0 k4 n1000000 | iree | 659.0 | 28.0 | 7001.5 | n/a |
| micro v0 k4 n1000000 | triton | 322.6 | 0.1 | 39.3 | 1029 |
| micro v0 k4 n10000000 | torch-compile | n/a | 542.2 | 646.6 | 210 |
| micro v0 k4 n10000000 | torch-compile-ro | n/a | 469.8 | 1213.5 | 236 |
| micro v0 k4 n10000000 | torch-compile-mat | n/a | 744.3 | 1197.2 | 382 |
| micro v0 k4 n10000000 | jax | 260.9 | 1.4 | 951.5 | 107 |
| micro v0 k4 n10000000 | iree | 659.2 | 185.1 | 60279.2 | n/a |
| micro v0 k4 n10000000 | triton | 313.3 | 0.7 | 327.5 | 102 |
| micro v0 k8 n1000000 | torch-compile | n/a | 491.3 | 132.8 | 900 |
| micro v0 k8 n1000000 | torch-compile-ro | n/a | 534.2 | 198.7 | 1140 |
| micro v0 k8 n1000000 | torch-compile-mat | n/a | 600.5 | 194.8 | 1284 |
| micro v0 k8 n1000000 | jax | 238.2 | 0.7 | 176.4 | 426 |
| micro v0 k8 n1000000 | iree | 621.6 | 25.6 | 7143.2 | n/a |
| micro v0 k8 n1000000 | triton | 364.4 | 0.2 | 70.8 | 571 |
| micro v1 k4 n1000000 | torch-compile | n/a | 541.3 | 308.2 | 56960 |
| micro v1 k4 n1000000 | torch-compile-ro | n/a | 568.4 | 396.7 | n/a |
| micro v1 k4 n1000000 | torch-compile-mat | n/a | 660.0 | 401.7 | n/a |
| micro v1 k4 n1000000 | jax | 224.2 | 0.6 | 101.1 | 867 |
| micro v1 k4 n1000000 | iree | 666.5 | 29.0 | 8212.4 | n/a |
| micro v1 k4 n1000000 | triton | 312.3 | 0.1 | 43.7 | 1006 |
| micro v10 k1 n191488 | torch-compile | n/a | 2137.8 | 79062.2 | 9409 |
| micro v10 k1 n191488 | torch-compile-ro | n/a | 2466.7 | 79023.0 | 9230 |
| micro v10 k1 n191488 | torch-compile-mat | n/a | 7749.2 | 76145.6 | 2368 |
| micro v10 k1 n191488 | jax | 6041.7 | 74.0 | 42073.8 | 154 |
| micro v10 k1 n191488 | iree | 4189.9 | 210.8 | 136621.0 | n/a |
| micro v10 k1 n25600 | torch-compile | n/a | 2129.5 | 3215.9 | 3032 |
| micro v10 k1 n25600 | torch-compile-ro | n/a | 2153.6 | 3100.1 | 2574 |
| micro v10 k1 n25600 | torch-compile-mat | n/a | 4968.9 | 3152.9 | 7041 |
| micro v10 k1 n25600 | jax | 4922.1 | 7.5 | 1523.1 | 2014 |
| micro v10 k1 n25600 | iree | 1750.5 | 26.2 | 10477.7 | n/a |
| micro v11 k1 n25600 | torch-compile | n/a | 1955.4 | 67.5 | n/a |
| micro v11 k1 n25600 | torch-compile-ro | n/a | 1755.9 | 129.4 | n/a |
| micro v11 k1 n25600 | torch-compile-mat | n/a | 1849.2 | 123.6 | n/a |
| micro v11 k1 n25600 | jax | 97.9 | 0.5 | 49.2 | -12168 |
| micro v11 k1 n25600 | iree | 189.8 | 9.0 | 62.1 | n/a |
| micro v11 k1 n25600 | triton | 310.6 | 0.1 | 19.1 | 3247 |
| micro v11 k1 n256000 | torch-compile | n/a | 1756.9 | 151.6 | n/a |
| micro v11 k1 n256000 | torch-compile-ro | n/a | 1791.6 | 239.4 | n/a |
| micro v11 k1 n256000 | torch-compile-mat | n/a | 1883.8 | 224.8 | n/a |
| micro v11 k1 n256000 | jax | 103.9 | 0.6 | 71.3 | -1431 |
| micro v11 k1 n256000 | iree | 236.5 | 13.8 | 207.5 | n/a |
| micro v11 k1 n256000 | triton | 319.7 | 0.1 | 21.3 | 884 |
| micro v12 k1 n25600 | torch-compile | n/a | 1844.9 | 138.3 | n/a |
| micro v12 k1 n25600 | torch-compile-ro | n/a | 1809.7 | 174.2 | n/a |
| micro v12 k1 n25600 | torch-compile-mat | n/a | 1920.5 | 185.2 | n/a |
| micro v12 k1 n25600 | jax | 1612.7 | 0.7 | 49.1 | 15725 |
| micro v12 k1 n25600 | iree | 232.8 | 15.7 | 1835.3 | n/a |
| micro v12 k1 n25600 | triton | 321.2 | 0.4 | 299.5 | n/a |
| micro v12 k1 n256000 | torch-compile | n/a | 1732.5 | 513.3 | n/a |
| micro v12 k1 n256000 | torch-compile-ro | n/a | 1836.4 | 527.7 | n/a |
| micro v12 k1 n256000 | torch-compile-mat | n/a | 1972.0 | 555.0 | n/a |
| micro v12 k1 n256000 | jax | 1655.5 | 0.9 | 258.4 | 6180 |
| micro v12 k1 n256000 | iree | 233.5 | 10.9 | 436.9 | 123 |
| micro v12 k1 n256000 | triton | 322.2 | 0.4 | 300.3 | 451 |
| micro v13 k1 n25600 | torch-compile | n/a | 1765.7 | 617.8 | n/a |
| micro v13 k1 n25600 | torch-compile-ro | n/a | 1823.5 | 640.5 | n/a |
| micro v13 k1 n25600 | torch-compile-mat | n/a | 2088.1 | 630.7 | n/a |
| micro v13 k1 n25600 | jax | 2213.8 | 1.2 | 123.6 | 4231 |
| micro v13 k1 n25600 | iree | 386.0 | 13.2 | 2796.8 | n/a |
| micro v13 k1 n25600 | triton | 325.8 | 1.4 | 1269.7 | n/a |
| micro v13 k1 n256000 | torch-compile | n/a | 2040.4 | 5047.4 | n/a |
| micro v13 k1 n256000 | torch-compile-ro | n/a | 1881.7 | 5070.1 | n/a |
| micro v13 k1 n256000 | torch-compile-mat | n/a | 2310.8 | 5026.9 | n/a |
| micro v13 k1 n256000 | jax | 2379.2 | 4.9 | 3718.4 | 1788 |
| micro v13 k1 n256000 | iree | 645.6 | 20.6 | 5169.3 | n/a |
| micro v13 k1 n256000 | triton | 315.3 | 5.0 | 4692.8 | 67 |
| micro v2 k4 n1000000 | torch-compile | n/a | 557.1 | 72.7 | 2044 |
| micro v2 k4 n1000000 | torch-compile-ro | n/a | 703.0 | 139.5 | 3556 |
| micro v2 k4 n1000000 | torch-compile-mat | n/a | 781.5 | 137.8 | 3940 |
| micro v2 k4 n1000000 | jax | 242.7 | 0.6 | 171.3 | 1320 |
| micro v2 k4 n1000000 | iree | 693.4 | 28.0 | 10838.9 | n/a |
| micro v2 k4 n1000000 | triton | 320.7 | 0.1 | 69.1 | 1096 |
| micro v3 k4 n1000000 | torch-compile | n/a | 600.3 | 168.5 | 2513 |
| micro v3 k4 n1000000 | torch-compile-ro | n/a | 701.8 | 425.7 | n/a |
| micro v3 k4 n1000000 | torch-compile-mat | n/a | 1038.2 | 380.1 | 107316 |
| micro v3 k4 n1000000 | jax | 215.4 | 0.6 | 332.5 | 2999 |
| micro v3 k4 n1000000 | iree | 617.5 | 25.7 | 84201.4 | n/a |
| micro v3 k4 n1000000 | triton | 374.9 | 0.1 | 126.8 | 1256 |
| micro v4 k4 n1000000 | torch-compile | n/a | 533.2 | 334.0 | 56158 |
| micro v4 k4 n1000000 | torch-compile-ro | n/a | 533.9 | 427.0 | n/a |
| micro v4 k4 n1000000 | torch-compile-mat | n/a | 642.5 | 422.2 | n/a |
| micro v4 k4 n1000000 | jax | 240.8 | 0.6 | 138.3 | 967 |
| micro v4 k4 n1000000 | iree | 688.8 | 29.9 | 13046.9 | n/a |
| micro v4 k4 n1000000 | triton | 373.8 | 0.1 | 69.5 | 1208 |
| micro v5 k4 n1000000 | torch-compile | n/a | 540.8 | 333.0 | 50599 |
| micro v5 k4 n1000000 | torch-compile-ro | n/a | 596.1 | 419.0 | n/a |
| micro v5 k4 n1000000 | torch-compile-mat | n/a | 795.1 | 414.6 | n/a |
| micro v5 k4 n1000000 | jax | 253.5 | 0.6 | 134.4 | 1005 |
| micro v5 k4 n1000000 | iree | 667.0 | 28.0 | 12359.3 | n/a |
| micro v5 k4 n1000000 | triton | 300.6 | 0.1 | 69.7 | 937 |
| micro v6 k1 n25600 | torch-compile | n/a | 2161.5 | 413.5 | n/a |
| micro v6 k1 n25600 | torch-compile-ro | n/a | 1843.3 | 414.3 | n/a |
| micro v6 k1 n25600 | torch-compile-mat | n/a | 2047.6 | 464.9 | n/a |
| micro v6 k1 n25600 | jax | 2042.3 | 1.0 | 107.6 | 5979 |
| micro v6 k1 n25600 | iree | 273.5 | 18.8 | 5391.7 | n/a |
| micro v6 k1 n25600 | triton | 368.0 | 0.9 | 895.9 | n/a |
| micro v6 k1 n256000 | torch-compile | n/a | 1951.6 | 1252.2 | 189557 |
| micro v6 k1 n256000 | torch-compile-ro | n/a | 2276.7 | 1338.1 | n/a |
| micro v6 k1 n256000 | torch-compile-mat | n/a | 2146.3 | 1291.6 | n/a |
| micro v6 k1 n256000 | jax | 1755.9 | 1.6 | 760.4 | 3018 |
| micro v6 k1 n256000 | iree | 291.0 | 11.3 | 919.1 | 164 |
| micro v6 k1 n256000 | triton | 316.0 | 0.9 | 900.1 | 196 |
| micro v7 k1 n25600 | torch-compile | n/a | 2176.1 | 950.5 | n/a |
| micro v7 k1 n25600 | torch-compile-ro | n/a | 2075.9 | 938.1 | n/a |
| micro v7 k1 n25600 | torch-compile-mat | n/a | 2637.3 | 940.5 | n/a |
| micro v7 k1 n25600 | jax | 2760.0 | 1.9 | 330.2 | 4902 |
| micro v7 k1 n25600 | iree | 405.8 | 31.3 | 11594.8 | n/a |
| micro v7 k1 n256000 | torch-compile | n/a | 2068.9 | 3064.2 | 23160 |
| micro v7 k1 n256000 | torch-compile-ro | n/a | 2115.2 | 3130.8 | 180957 |
| micro v7 k1 n256000 | torch-compile-mat | n/a | 2687.3 | 3147.7 | n/a |
| micro v7 k1 n256000 | jax | 2685.9 | 4.1 | 1950.7 | 2014 |
| micro v7 k1 n256000 | iree | 561.3 | 44.6 | 31660.6 | n/a |
| micro v8 k1 n25600 | torch-compile | n/a | 1891.6 | 520.4 | 34249 |
| micro v8 k1 n25600 | torch-compile-ro | n/a | 2055.5 | 535.1 | 55676 |
| micro v8 k1 n25600 | torch-compile-mat | n/a | 3048.3 | 518.8 | 57470 |
| micro v8 k1 n25600 | jax | 1602.1 | 1.4 | 267.4 | 4300 |
| micro v8 k1 n25600 | iree | 617.5 | 11.8 | 921.2 | n/a |
| micro v8 k1 n256000 | torch-compile | n/a | 2091.1 | 29879.7 | n/a |
| micro v8 k1 n256000 | torch-compile-ro | n/a | 1921.4 | 29779.7 | n/a |
| micro v8 k1 n256000 | torch-compile-mat | n/a | 5583.4 | 27752.0 | n/a |
| micro v8 k1 n256000 | jax | 1654.0 | 16.5 | 15424.6 | 109 |
| micro v8 k1 n256000 | iree | 25971.0 | 89.8 | 38681.0 | n/a |
| micro v9 k1 n25600 | torch-compile | n/a | 2027.9 | 862.5 | n/a |
| micro v9 k1 n25600 | torch-compile-ro | n/a | 2224.6 | 867.8 | n/a |
| micro v9 k1 n25600 | torch-compile-mat | n/a | 2474.9 | 847.7 | n/a |
| micro v9 k1 n25600 | jax | 442.9 | 1.0 | 266.8 | 262 |
| micro v9 k1 n256000 | torch-compile | n/a | 1881.1 | 690.3 | n/a |
| micro v9 k1 n256000 | torch-compile-ro | n/a | 1898.4 | 705.3 | n/a |
| micro v9 k1 n256000 | torch-compile-mat | n/a | 3520.4 | 709.8 | n/a |
| micro v9 k1 n256000 | jax | 944.4 | 1.0 | 351.4 | 2472 |
| mixer_b16 | torch-compile | n/a | 1233.5 | 3167.8 | n/a |
| mixer_b16 | torch-compile-ro | n/a | 1378.6 | 3086.0 | n/a |
| mixer_b16 | torch-compile-mat | n/a | 4410.8 | 3137.8 | n/a |
| mixer_b16 | torch-tensorrt | n/a | 5255.2 | 2505.9 | 12530 |
| mixer_b16 | jax | 7681.3 | 7.2 | 2249.4 | 11357 |
| mixer_b16 | iree | 1541.8 | 753.6 | 674484.5 | n/a |
| mixer_b16 | ours | n/a | 209.4 | 2712.8 | 385 |
| qwen2.5-0.5b | torch-compile | n/a | 4103.7 | 7444.0 | 521 |
| qwen2.5-0.5b | torch-compile-ro | n/a | 3987.4 | 7105.3 | 481 |
| qwen2.5-0.5b | torch-compile-mat | n/a | 4228.1 | 7127.7 | 516 |
| qwen2.5-0.5b | torch-tensorrt | n/a | 23271.2 | 7101.8 | 3169 |
| qwen2.5-0.5b | jax | 14314.0 | 20.9 | 5231.7 | 1525 |
| qwen2.5-0.5b | iree | 2136.7 | 585.1 | 86466.8 | n/a |
| qwen2.5-0.5b | ours | n/a | 2244.4 | 6526.5 | 220 |
| resnet18-b1 | torch-compile | n/a | 1091.0 | 983.9 | 1495 |
| resnet18-b1 | torch-compile-ro | n/a | 1087.9 | 925.2 | 1357 |
| resnet18-b1 | torch-compile-mat | n/a | 5610.1 | 1061.5 | 10428 |
| resnet18-b1 | torch-tensorrt | n/a | 5002.3 | 998.6 | 8256 |
| resnet18-b1 | jax | 694.2 | 6.1 | 1113.9 | 1075 |
| resnet18-b1 | ours | n/a | 152.1 | 724.4 | -55 |
| resnet18-b8 | torch-compile | n/a | 1115.1 | 2188.4 | 10003 |
| resnet18-b8 | torch-compile-ro | n/a | 1129.0 | 2134.5 | 6273 |
| resnet18-b8 | torch-compile-mat | n/a | 5688.3 | 2136.1 | 39098 |
| resnet18-b8 | torch-tensorrt | n/a | 5497.5 | 1950.7 | 16179 |
| resnet18-b8 | jax | 985.7 | 4.9 | 2286.9 | n/a |
| resnet18-b8 | ours | n/a | 150.8 | 2736.9 | n/a |
| smollm2-1.7b | torch-compile | n/a | 3410.0 | 18150.5 | 1975 |
| smollm2-1.7b | torch-compile-ro | n/a | 3304.2 | 17852.2 | 1569 |
| smollm2-1.7b | torch-compile-mat | n/a | 3676.9 | 17898.6 | 1837 |
| smollm2-1.7b | torch-tensorrt | n/a | 22434.0 | 16776.2 | 7823 |
| smollm2-1.7b | jax | 7655.7 | 25.6 | 13536.5 | 1172 |
| smollm2-1.7b | ours | n/a | 3676.1 | 14243.3 | 575 |
| smollm2-135m | torch-compile | n/a | 4170.9 | 6078.9 | 309 |
| smollm2-135m | torch-compile-ro | n/a | 4645.6 | 3785.1 | 292 |
| smollm2-135m | torch-compile-mat | n/a | 5534.7 | 3853.2 | 357 |
| smollm2-135m | torch-tensorrt | n/a | 25668.4 | 3749.5 | 1791 |
| smollm2-135m | jax | 12286.0 | 19.2 | 2666.4 | 777 |
| smollm2-135m | iree | 2445.4 | 194.8 | 19251.2 | n/a |
| smollm2-135m | ours | n/a | 651.4 | 3674.6 | 6 |
| smollm2-360m | torch-compile | n/a | 4420.8 | 7688.1 | 384 |
| smollm2-360m | torch-compile-ro | n/a | 4542.6 | 6281.5 | 348 |
| smollm2-360m | torch-compile-mat | n/a | 4945.6 | 6325.6 | 384 |
| smollm2-360m | torch-tensorrt | n/a | 30332.9 | 6488.6 | 2618 |
| smollm2-360m | jax | 13491.0 | 25.5 | 4588.3 | 979 |
| smollm2-360m | iree | 2495.5 | 501.4 | 33036.1 | n/a |
| smollm2-360m | ours | n/a | 991.0 | 6077.6 | 41 |
| tiny-gpt2 | torch-compile | n/a | 1284.4 | 459.9 | 653 |
| tiny-gpt2 | torch-compile-ro | n/a | 1347.0 | 208.6 | 582 |
| tiny-gpt2 | torch-compile-mat | n/a | 2162.3 | 297.0 | 1226 |
| tiny-gpt2 | torch-tensorrt | n/a | 7839.7 | 590.9 | 6957 |
| tiny-gpt2 | jax | 1425.0 | 1.8 | 119.1 | 601 |
| tiny-gpt2 | iree | 482.3 | 14.3 | 210.7 | -10 |
| tiny-gpt2 | ours | n/a | 129.1 | 249.5 | -274 |
| vit-base | torch-compile | n/a | 1841.9 | 4660.7 | n/a |
| vit-base | torch-compile-ro | n/a | 2096.8 | 4612.8 | n/a |
| vit-base | torch-compile-mat | n/a | 2298.5 | 4431.7 | 50014 |
| vit-base | torch-tensorrt | n/a | 8298.5 | 3798.1 | 12041 |
| vit-base | jax | 9377.7 | 11.1 | 3186.5 | 7170 |
| vit-base | iree | 1694.6 | 991.7 | 883623.9 | n/a |
| vit-base | ours | n/a | 295.4 | 3723.1 | 190 |
| vit-tiny | torch-compile | n/a | 1950.5 | 2540.6 | 714 |
| vit-tiny | torch-compile-ro | n/a | 2216.9 | 1438.1 | 571 |
| vit-tiny | torch-compile-mat | n/a | 4195.0 | 1334.3 | 1082 |
| vit-tiny | torch-tensorrt | n/a | 8562.0 | 1166.0 | 2149 |
| vit-tiny | jax | 7740.4 | 7.8 | 740.0 | 1751 |
| vit-tiny | iree | 1771.7 | 39.0 | 25163.9 | n/a |
| vit-tiny | ours | n/a | 139.0 | 1033.5 | 2 |

## Warm-up (median over rounds; steady_at/crossover in iterations, 'none' if never reached within N)

| model | system | cache | first (ms) | steady_at | steady_us | crossover vs eager |
|---|---|---|---|---|---|---|
| distilgpt2 | jax | cold | 5.34 | 3 | 1201.8 | 1 |
| distilgpt2 | jax | warm | 55.34 | 46 | 1186.2 | 1 |
| distilgpt2 | ours | cold | 4512.40 | 193 | 1254.0 | none |
| distilgpt2 | ours | warm | 601.74 | 167 | 1254.6 | 254 |
| distilgpt2 | torch-compile | cold | 4580.52 | 82 | 1419.8 | none |
| distilgpt2 | torch-compile | warm | 1622.07 | 17 | 1482.7 | none |
| distilgpt2 | torch-compile-ro | cold | 3776.42 | 16 | 1308.7 | none |
| distilgpt2 | torch-compile-ro | warm | 1706.04 | 74 | 1334.5 | none |
| distilgpt2 | torch-eager | cold | 154.75 | 31 | 3298.9 | none |
| distilgpt2 | torch-eager | warm | 153.11 | 6 | 3443.1 | none |
| gpt2 | jax | cold | 8.59 | 3 | 1951.3 | 1 |
| gpt2 | jax | warm | 65.32 | 7 | 1997.4 | 1 |
| gpt2 | ours | cold | 3933.60 | 215 | 2076.1 | none |
| gpt2 | ours | warm | 670.82 | 215 | 2076.9 | 141 |
| gpt2 | torch-compile | cold | 6113.28 | 22 | 2282.7 | none |
| gpt2 | torch-compile | warm | 2108.02 | 3 | 2358.1 | none |
| gpt2 | torch-compile-ro | cold | 5610.36 | 61 | 2104.8 | none |
| gpt2 | torch-compile-ro | warm | 2510.74 | 106 | 2263.3 | none |
| gpt2 | torch-eager | cold | 148.26 | 180 | 5099.5 | none |
| gpt2 | torch-eager | warm | 149.45 | 203 | 5291.2 | none |

## Warm-up: kernel compiles (ours, median over rounds; cold_compiles = Triton subprocess compiles over the whole trace, first_forward_compiles = at iter 0)

| model | cache | cold_compiles | first_forward_compiles |
|---|---|---|---|
| distilgpt2 | cold | 9 | 7 |
| distilgpt2 | warm | 0 | 0 |
| gpt2 | cold | 9 | 7 |
| gpt2 | warm | 0 | 0 |

## Dynamic sequence length (median steady_us per length, counters per window)

| system | pass | length | median_us | total_us | loops | bridges | kernels | cache_hits | recompiles |
|---|---|---|---|---|---|---|---|---|---|
| ours | first | 32 | 982.5 | 1237.4 | 2 | 4 | 0 | 833 | 0 |
| ours | first | 48 | 1395.9 | 1822.0 | 5 | 0 | 4 | 720 | 0 |
| ours | first | 64 | 1445.1 | 2004.8 | 1 | 3 | 0 | 597 | 0 |
| ours | first | 96 | 1725.0 | 2522.0 | 1 | 0 | 0 | 597 | 0 |
| ours | first | 128 | 2064.0 | 3207.0 | 0 | 1 | 0 | 632 | 0 |
| ours | new | 160 | 2466.6 | 4140.5 | 0 | 0 | 0 | 630 | 0 |
| ours | new | 192 | 2761.1 | 4627.9 | 0 | 0 | 0 | 585 | 0 |
| ours | new | 224 | 3189.1 | 5426.9 | 0 | 0 | 0 | 585 | 0 |
| ours | new | 256 | 3309.0 | 7236.0 | 0 | 0 | 0 | 585 | 0 |
| ours | new | 288 | 3856.9 | 8222.1 | 0 | 0 | 0 | 585 | 0 |
| ours | revisit | 32 | 957.0 | 1236.9 | 0 | 0 | 0 | 630 | 0 |
| ours | revisit | 48 | 1370.0 | 1745.0 | 0 | 0 | 0 | 585 | 0 |
| ours | revisit | 64 | 1425.0 | 1952.0 | 0 | 0 | 0 | 585 | 0 |
| ours | revisit | 96 | 1673.0 | 2441.0 | 0 | 0 | 0 | 585 | 0 |
| ours | revisit | 128 | 2069.0 | 3088.9 | 0 | 0 | 0 | 585 | 0 |
| torch-compile-dynamic | first | 32 | 1124.6 | 1275.0 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-dynamic | first | 48 | 1551.9 | 1708.8 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-dynamic | first | 64 | 1591.0 | 1755.7 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-dynamic | first | 96 | 1750.0 | 1933.3 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-dynamic | first | 128 | 1947.4 | 2137.7 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-dynamic | new | 160 | 2374.4 | 2584.9 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-dynamic | new | 192 | 2606.9 | 2824.3 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-dynamic | new | 224 | 3032.4 | 3260.6 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-dynamic | new | 256 | 3109.5 | 3350.3 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-dynamic | new | 288 | 3757.7 | 4017.6 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-dynamic | revisit | 32 | 1110.6 | 1260.5 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-dynamic | revisit | 48 | 1518.2 | 1671.7 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-dynamic | revisit | 64 | 1560.0 | 1723.1 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-dynamic | revisit | 96 | 1707.6 | 1884.3 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-dynamic | revisit | 128 | 1881.8 | 2085.4 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-static | first | 32 | 1084.7 | 1241.2 | 0 | 0 | 0 | 0 | 1 |
| torch-compile-static | first | 48 | 1422.4 | 1589.8 | 0 | 0 | 0 | 0 | 1 |
| torch-compile-static | first | 64 | 1476.5 | 1661.3 | 0 | 0 | 0 | 0 | 1 |
| torch-compile-static | first | 96 | 1680.6 | 1870.9 | 0 | 0 | 0 | 0 | 1 |
| torch-compile-static | first | 128 | 1841.3 | 2038.0 | 0 | 0 | 0 | 0 | 1 |
| torch-compile-static | new | 160 | 2284.1 | 2488.2 | 0 | 0 | 0 | 0 | 1 |
| torch-compile-static | new | 192 | 2450.2 | 2665.0 | 0 | 0 | 0 | 0 | 1 |
| torch-compile-static | new | 224 | 3411.1 | 3635.7 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-static | new | 256 | 3576.0 | 3807.3 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-static | new | 288 | 4246.7 | 4488.4 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-static | revisit | 32 | 1076.8 | 1234.2 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-static | revisit | 48 | 1435.8 | 1603.9 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-static | revisit | 64 | 1442.4 | 1617.2 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-static | revisit | 96 | 1678.9 | 1863.2 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-static | revisit | 128 | 1836.3 | 2038.5 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | first | 32 | 2448.3 | 2599.0 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | first | 48 | 2579.2 | 2759.9 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | first | 64 | 2524.1 | 2679.4 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | first | 96 | 2944.7 | 3111.1 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | first | 128 | 2901.3 | 3136.1 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | new | 160 | 2927.9 | 3150.9 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | new | 192 | 2918.2 | 3151.4 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | new | 224 | 3327.8 | 3541.0 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | new | 256 | 3525.5 | 3763.0 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | new | 288 | 4168.5 | 4420.3 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | revisit | 32 | 2384.6 | 2548.4 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | revisit | 48 | 2785.9 | 2926.3 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | revisit | 64 | 2552.7 | 2705.1 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | revisit | 96 | 2570.4 | 2739.4 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | revisit | 128 | 2745.4 | 2939.0 | 0 | 0 | 0 | 0 | 0 |

## Where the operation DAG lives (median over rounds)

| model | DAG in | steady_us | launches/iter | kernels | compiles | gc_ms | nodes deferred | same argmax |
|---|---|---|---|---|---|---|---|---|
| bert-base | virtual | 1851.9 | 64.3 | 109 | 0 | 18 | 0 | 1 |
| bert-base | deferred | 3013.5 | 64.0 | 112 | 0 | 90 | 174343 | 1 |
| bert-base | eager | 3823.0 | 759.0 | 103 | 0 | 26 | 0 | 1 |
| bert-mini | virtual | 423.1 | 24.3 | 109 | 0 | 10 | 0 | 1 |
| bert-mini | deferred | 1563.5 | 24.0 | 112 | 0 | 55 | 65783 | 1 |
| bert-mini | eager | 1153.9 | 287.0 | 103 | 0 | 21 | 0 | 1 |
| distilgpt2 | virtual | 1312.0 | 38.2 | 110 | 0 | 20 | 0 | 1 |
| distilgpt2 | deferred | 1919.2 | 44.0 | 113 | 0 | 77 | 62103 | 1 |
| distilgpt2 | eager | 1962.1 | 271.0 | 103 | 0 | 23 | 0 | 1 |
| gpt2 | virtual | 2098.9 | 74.2 | 110 | 0 | 24 | 0 | 1 |
| gpt2 | deferred | 3496.3 | 86.0 | 113 | 0 | 118 | 121443 | 1 |
| gpt2 | eager | 3382.1 | 529.0 | 103 | 0 | 22 | 0 | 1 |
| mixer_b16 | virtual | 2695.8 | 63.4 | 109 | 0 | 28 | 0 | 1 |
| mixer_b16 | deferred | 4966.0 | 75.0 | 122 | 0 | 189 | 218273 | 1 |
| mixer_b16 | eager | 5890.7 | 950.0 | 103 | 0 | 34 | 0 | 1 |
| resnet18-b1 | virtual | 699.1 | 39.0 | 117 | 0 | 1 | 0 | 1 |
| resnet18-b1 | deferred | 760.6 | 39.0 | 123 | 0 | 31 | 15183 | 1 |
| resnet18-b1 | eager | 891.4 | 87.0 | 114 | 0 | 31 | 0 | 1 |
| smollm2-135m | virtual | 3688.1 | 212.1 | 110 | 0 | 21 | 0 | 1 |
| smollm2-135m | deferred | 4597.8 | 205.0 | 117 | 0 | 119 | 215513 | 1 |
| smollm2-135m | eager | 5906.2 | 968.0 | 104 | 0 | 20 | 0 | 1 |
| smollm2-360m | virtual | 6081.7 | 226.1 | 110 | 0 | 20 | 0 | 1 |
| smollm2-360m | deferred | 6318.6 | 218.0 | 117 | 0 | 101 | 229773 | 1 |
| smollm2-360m | eager | 8400.6 | 1032.0 | 104 | 0 | 17 | 0 | 1 |
| vit-base | virtual | 3705.4 | 76.3 | 111 | 0 | 26 | 0 | 1 |
| vit-base | deferred | 4707.8 | 88.0 | 116 | 0 | 155 | 165833 | 1 |
| vit-base | eager | 6388.2 | 723.0 | 104 | 0 | 35 | 0 | 1 |

## Where the DAG lives, micro grid (median over rounds)

| variant | k | n | DAG in | steady_us | launches/iter |
|---|---|---|---|---|---|
| 0 | 1 | 10000 | virtual | 4.7 | 1.01 |
| 0 | 1 | 10000 | deferred | 4.7 | 1.00 |
| 0 | 1 | 10000 | eager | 10.6 | 3.00 |
| 0 | 4 | 1000000 | virtual | 117.9 | 4.01 |
| 0 | 4 | 1000000 | deferred | 39.3 | 1.00 |
| 0 | 4 | 1000000 | eager | 307.4 | 12.01 |
| 8 | 1 | 65536 | virtual | 1716.8 | 11.14 |
| 8 | 1 | 65536 | deferred | 1745.3 | 9.00 |
| 8 | 1 | 65536 | eager | 2306.4 | 38.01 |
| 11 | 1 | 256000 | virtual | 16.1 | 2.04 |
| 11 | 1 | 256000 | deferred | 25.9 | 2.00 |
| 11 | 1 | 256000 | eager | 141.4 | 8.01 |
| 13 | 1 | 65536 | virtual | 591.5 | 5.00 |
| 13 | 1 | 65536 | deferred | 591.7 | 5.00 |
| 13 | 1 | 65536 | eager | 792.2 | 10.01 |

## Host-dependent early exit (median over rounds)

| regime | system | total_ms | p50_us | p95_us | max_us | bridges | frame_compiles | exit layers | pass |
|---|---|---|---|---|---|---|---|---|---|
| stable | jax-perlayer | 12904 | 1667.3 | 1774.5 | 2143 | 0 | 0 | 4 | 1 |
| stable | jax-while | 9150 | 1420.7 | 1502.8 | 1805 | 0 | 0 | 4 | 1 |
| stable | ours | 897 | 1066.9 | 4536.2 | 4860 | 1 | 0 | 4 | 1 |
| stable | torch-compile | 1124 | 1434.1 | 1481.1 | 1496 | 0 | 4 | 4 | 1 |
| stable | torch-compile-mat | 1289 | 1483.9 | 1545.4 | 1630 | 0 | 4 | 4 | 1 |
| stable | torch-compile-ro | 1271 | 1455.5 | 1494.2 | 1577 | 0 | 4 | 4 | 1 |
| stable | torch-eager | 563 | 2021.1 | 2345.8 | 3789 | 0 | 0 | 4 | 1 |
| varying | jax-perlayer | 12726 | 1967.0 | 2620.0 | 2933 | 0 | 0 | 2356 | 1 |
| varying | jax-while | 8390 | 1505.9 | 1720.2 | 2566 | 0 | 0 | 2356 | 1 |
| varying | ours | 862 | 1215.0 | 2892.0 | 4038 | 3 | 0 | 2356 | 1 |
| varying | torch-compile | 1221 | 1608.4 | 1925.7 | 2116 | 0 | 4 | 2356 | 1 |
| varying | torch-compile-mat | 1184 | 1464.4 | 1781.9 | 1838 | 0 | 4 | 2356 | 1 |
| varying | torch-compile-ro | 1165 | 1555.0 | 1824.6 | 1865 | 0 | 4 | 2356 | 1 |
| varying | torch-eager | 499 | 2277.1 | 2679.8 | 2992 | 0 | 0 | 2356 | 1 |

