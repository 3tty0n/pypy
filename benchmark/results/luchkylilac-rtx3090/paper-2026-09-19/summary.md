# Paper benchmark summary

## Microbenchmarks (median steady_us over rounds)

| variant | k | n | fused (ours) | app-level (ours) | eager (ours) | nojit (ours) | torch.compile | compile-ro | compile-mat | torch eager | JAX/XLA | IREE | Triton | speedup vs compile | interp tax | ours/Triton |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1 | 1000000 | 29.0 | 29.1 | 77.2 | 77.1 | 54.6 | 101.6 | 102.9 | 78.3 | 52.2 | 5786.7 | 30.8 | 1.88x | 1.00x | 0.94x |
| 0 | 4 | 10000 | 13.0 | 19.5 | 39.3 | 39.6 | 51.8 | 106.4 | 109.5 | 59.0 | 41.4 | 439.5 | 19.0 | 3.98x | 1.50x | 0.68x |
| 0 | 4 | 100000 | 13.7 | 20.8 | 43.4 | 43.3 | 45.0 | 109.2 | 118.2 | 60.9 | 44.3 | 975.9 | 20.3 | 3.28x | 1.51x | 0.68x |
| 0 | 4 | 1000000 | 40.5 | 117.5 | 307.0 | 306.6 | 68.4 | 134.0 | 133.0 | 312.4 | 94.8 | 5756.3 | 39.2 | 1.69x | 2.90x | 1.03x |
| 0 | 4 | 10000000 | 329.1 | 1133.4 | 3026.9 | 3027.3 | 645.9 | 1224.0 | 1196.0 | 3045.3 | 946.7 | 51561.5 | 327.0 | 1.96x | 3.44x | 1.01x |
| 0 | 8 | 1000000 | 70.1 | 229.5 | 612.5 | 612.9 | 132.7 | 198.2 | 198.5 | 624.3 | 175.9 | 5969.2 | 66.9 | 1.89x | 3.27x | 1.05x |
| 1 | 4 | 1000000 | 41.6 | 119.5 | 311.5 | 310.9 | 307.8 | 400.0 | 396.5 | 316.8 | 99.3 | 6410.2 | 43.4 | 7.39x | 2.87x | 0.96x |
| 2 | 4 | 1000000 | 42.7 | 119.4 | 310.9 | 310.9 | 72.7 | 138.5 | 134.5 | 326.0 | 177.1 | 8978.1 | 69.1 | 1.70x | 2.80x | 0.62x |
| 3 | 4 | 1000000 | 98.6 | 178.3 | 360.6 | 365.6 | 167.2 | 352.1 | 349.6 | 376.8 | 308.6 | 81591.2 | 122.8 | 1.70x | 1.81x | 0.80x |
| 4 | 4 | 1000000 | 53.4 | 145.0 | 336.0 | 335.7 | 332.3 | 424.4 | 412.2 | 342.3 | 126.6 | 10790.3 | 69.5 | 6.22x | 2.71x | 0.77x |
| 5 | 4 | 1000000 | 53.5 | 146.8 | 335.8 | 335.7 | 332.3 | 424.4 | 421.0 | 342.2 | 127.7 | 10808.9 | 69.7 | 6.21x | 2.74x | 0.77x |
| 6 | 1 | 25600 | 278.6 | 268.1 | 278.1 | 277.7 | 405.9 | 432.7 | 433.1 | 393.7 | 107.2 | 5383.8 | 874.7 | 1.46x | 0.96x | 0.32x |
| 6 | 1 | 256000 | 1015.8 | 999.4 | 1016.8 | 1016.1 | 1252.9 | 1305.7 | 1283.5 | 1249.4 | 752.4 | 910.1 | 850.2 | 1.23x | 0.98x | 1.19x |
| 7 | 1 | 25600 | 647.7 | 780.1 | 668.7 | 668.7 | 839.1 | 880.7 | 876.2 | 804.6 | 324.6 | 11542.1 | n/a | 1.30x | 1.20x | n/a |
| 7 | 1 | 256000 | 2808.2 | 2926.9 | 2834.3 | 2830.5 | 3055.6 | 3095.3 | 3090.9 | 3069.2 | 1959.1 | 31568.2 | n/a | 1.09x | 1.04x | n/a |
| 8 | 1 | 25600 | 464.3 | 505.1 | 779.4 | 779.9 | 513.4 | 522.9 | 511.3 | 555.9 | 266.8 | 922.9 | n/a | 1.11x | 1.09x | n/a |
| 8 | 1 | 256000 | 22328.1 | 22353.9 | 25092.8 | 25099.0 | 29734.5 | 29735.9 | 27770.9 | 27353.3 | 15419.2 | 38532.9 | n/a | 1.33x | 1.00x | n/a |
| 9 | 1 | 25600 | 755.4 | 749.7 | 763.0 | 762.1 | 863.8 | 881.0 | 875.5 | 797.7 | 253.7 | n/a | n/a | 1.14x | 0.99x | n/a |
| 9 | 1 | 256000 | 572.9 | 554.5 | 602.7 | 570.2 | 678.4 | 700.2 | 663.1 | 586.5 | 327.5 | n/a | n/a | 1.18x | 0.97x | n/a |
| 10 | 1 | 25600 | 3885.6 | 4856.6 | 4827.6 | 4817.2 | 3176.8 | 3092.4 | 3161.3 | 3690.6 | 1500.8 | 10437.9 | n/a | 0.82x | 1.25x | n/a |
| 10 | 1 | 191488 | 93971.8 | 96704.7 | 99329.1 | 99373.2 | 78674.0 | 78678.0 | 75790.1 | 78928.6 | 41870.8 | 136367.7 | n/a | 0.84x | 1.03x | n/a |
| 11 | 1 | 25600 | 3.6 | 4.9 | 56.8 | 57.2 | 67.3 | 128.8 | 132.6 | 57.0 | 49.5 | 56.3 | 18.7 | 18.59x | 1.36x | 0.19x |
| 11 | 1 | 256000 | 15.9 | 14.7 | 143.3 | 141.4 | 158.9 | 206.1 | 207.2 | 151.2 | 52.7 | 190.2 | 20.1 | 10.00x | 0.92x | 0.79x |
| 12 | 1 | 25600 | 98.7 | 89.4 | 98.4 | 98.3 | 137.9 | 165.0 | 168.0 | 137.1 | 45.1 | 1832.9 | 298.9 | 1.40x | 0.91x | 0.33x |
| 12 | 1 | 256000 | 343.4 | 333.2 | 342.8 | 342.7 | 493.9 | 536.4 | 519.3 | 494.0 | 258.2 | 436.0 | 299.8 | 1.44x | 0.97x | 1.15x |
| 13 | 1 | 25600 | 437.2 | 448.4 | 526.8 | 526.7 | 597.7 | 620.5 | 622.6 | 594.5 | 123.3 | 2797.3 | 1272.2 | 1.37x | 1.03x | 0.34x |
| 13 | 1 | 256000 | 4742.2 | 4754.6 | 5423.8 | 5427.8 | 5029.4 | 5007.3 | 5024.1 | 4874.5 | 3706.4 | 5130.2 | 4679.4 | 1.06x | 1.00x | 1.01x |

## Precision sweep (median steady_us)

| dtype | variant | k | n | fused (ours) | app-level (ours) | eager (ours) | nojit (ours) | torch.compile | compile-ro | compile-mat | torch eager | JAX/XLA | IREE | Triton | speedup vs compile | interp tax | ours/Triton |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| float16 | 8 | 1 | 256000 | 708.2 | n/a | 2052.7 | 2058.1 | 867.1 | 867.7 | 778.2 | 1450.8 | 635.0 | 1886.7 | n/a | 1.22x | n/a | n/a |
| float16 | 11 | 1 | 256000 | 22.4 | n/a | 59.0 | 55.4 | 144.3 | 241.8 | 209.8 | 145.8 | 41.6 | 181.9 | 19.3 | 6.45x | n/a | 1.16x |
| float16 | 13 | 1 | 256000 | 211.0 | n/a | 333.1 | 335.6 | 434.9 | 375.3 | 360.9 | 409.3 | 99.3 | 598.7 | 131.9 | 2.06x | n/a | 1.60x |
| float32 | 8 | 1 | 256000 | 1393.8 | n/a | 3912.3 | 3916.4 | 2034.2 | 2050.9 | 1497.3 | 2659.2 | 1261.5 | 7307.4 | n/a | 1.46x | n/a | n/a |
| float32 | 11 | 1 | 256000 | 6.8 | n/a | 42.1 | 37.7 | 147.1 | 222.4 | 219.4 | 144.6 | 41.0 | 81.0 | 19.2 | 21.56x | n/a | 0.36x |
| float32 | 13 | 1 | 256000 | 271.2 | n/a | 581.0 | 582.9 | 559.9 | 569.2 | 494.8 | 636.2 | 215.7 | 1313.7 | 120.6 | 2.06x | n/a | 2.25x |

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
| bert-mini | 423.1 | 927.1 | 481.4 | 417.6 | 1803.7 | 243.0 | 3637.7 | 711.6 | 0.46x | 0.26x | 24.3 | 1.90735e-05 | 0.001 | pass |
| bert-tiny | 255.7 | 540.1 | 298.6 | 275.1 | 1101.5 | 110.6 | 1345.1 | 478.5 | 0.47x | 0.20x | 14.3 | 3.05176e-05 | 0.001 | pass |
| distilgpt2 | 1289.3 | 1371.2 | 1306.5 | 1274.0 | 2776.2 | 1001.9 | 17137.2 | 1711.4 | 0.94x | 0.73x | 38.2 | 0.000190735 | 0.001 | pass |
| mixer_b16 | 2703.1 | 3143.0 | 3077.8 | 3105.7 | 2899.7 | 2248.5 | 674112.6 | 2495.8 | 0.86x | 0.72x | 63.4 | 3.62396e-05 | 0.001 | pass |
| resnet18-b1 | 720.0 | 1124.6 | 918.7 | 1038.5 | 1452.0 | 1117.5 | n/a | 987.6 | 0.64x | 0.99x | 39.0 | 0.00963783 | 0.02 | pass |
| resnet18-b8 | 2724.5 | 2194.6 | 2137.6 | 2128.5 | 2265.5 | 2279.8 | n/a | 1949.5 | 1.24x | 1.04x | 39.0 | 0.00571394 | 0.02 | pass |
| smollm2-135m | 3676.9 | 5203.8 | 3798.2 | 3822.8 | 15090.5 | 2656.0 | 19171.0 | 3726.8 | 0.71x | 0.51x | 212.1 | 0.000201941 | 0.001 | pass |
| tiny-gpt2 | 194.2 | 418.5 | 206.6 | 234.3 | 1625.3 | 88.1 | 198.4 | 536.1 | 0.46x | 0.21x | 13.2 | 2.6077e-08 | 0.001 | pass |
| vit-tiny | 1003.9 | 2326.7 | 1402.5 | 1336.1 | 3930.9 | 737.7 | 25019.8 | 1126.0 | 0.43x | 0.32x | 76.3 | 1.04904e-05 | 0.001 | pass |

## Correctness on five derived inputs (worst case per model; ours is the reference)

| model | system | max maxabsdiff | min argmax match | tol | passed |
|---|---|---|---|---|---|
| bert-mini | jax | 2.48e-05 | 1.000 | 0.001 | 5/5 |
| bert-mini | torch-compile | 2.29e-05 | 1.000 | 0.001 | 5/5 |
| bert-mini | torch-eager | 2.67e-05 | 1.000 | 0.001 | 5/5 |
| bert-tiny | jax | 3.43e-05 | 1.000 | 0.001 | 5/5 |
| bert-tiny | torch-compile | 5.15e-05 | 1.000 | 0.001 | 5/5 |
| bert-tiny | torch-eager | 3.91e-05 | 1.000 | 0.001 | 5/5 |
| distilgpt2 | jax | 0.000221 | 1.000 | 0.001 | 5/5 |
| distilgpt2 | torch-compile | 0.000282 | 1.000 | 0.001 | 5/5 |
| distilgpt2 | torch-eager | 0.000282 | 1.000 | 0.001 | 5/5 |
| mixer_b16 | jax | 7.01e-05 | 1.000 | 0.001 | 5/5 |
| mixer_b16 | torch-compile | 5.91e-05 | 1.000 | 0.001 | 5/5 |
| mixer_b16 | torch-eager | 5.15e-05 | 1.000 | 0.001 | 5/5 |
| resnet18-b1 | jax | 0.0167 | 1.000 | 0.02 | 5/5 |
| resnet18-b1 | torch-compile | 0.00942 | 1.000 | 0.02 | 5/5 |
| resnet18-b1 | torch-eager | 0.0158 | 1.000 | 0.02 | 5/5 |
| resnet18-b8 | jax | 0.0123 | 1.000 | 0.02 | 5/5 |
| resnet18-b8 | torch-compile | 0.0122 | 1.000 | 0.02 | 5/5 |
| resnet18-b8 | torch-eager | 0.00725 | 1.000 | 0.02 | 5/5 |
| smollm2-135m | jax | 0.000278 | 1.000 | 0.001 | 5/5 |
| smollm2-135m | torch-compile | 0.000168 | 1.000 | 0.001 | 5/5 |
| smollm2-135m | torch-eager | 0.000147 | 1.000 | 0.001 | 5/5 |
| tiny-gpt2 | jax | 3.17e-08 | 1.000 | 0.001 | 5/5 |
| tiny-gpt2 | torch-compile | 3.73e-08 | 1.000 | 0.001 | 5/5 |
| tiny-gpt2 | torch-eager | 3.73e-08 | 1.000 | 0.001 | 5/5 |
| vit-tiny | jax | 1.05e-05 | 1.000 | 0.001 | 5/5 |
| vit-tiny | torch-compile | 1.19e-05 | 1.000 | 0.001 | 5/5 |
| vit-tiny | torch-eager | 1.41e-05 | 1.000 | 0.001 | 5/5 |

## Batch-size sweep (median steady_us per forward; per_seq = steady/B)

| model | batch | ours | torch.compile | compile-ro | torch eager | JAX/XLA | ratio ours/compile | ratio ours/jax | per_seq ours | per_seq compile | per_seq jax | rows identical | failed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bert-mini | 1 | 422.4 | 904.1 | 479.5 | 1825.9 | 241.7 | 0.47x | 1.75x | 422.4 | 904.1 | 241.7 | yes |  |
| bert-mini | 2 | 622.7 | 764.9 | 592.0 | 1763.1 | 342.5 | 0.81x | 1.82x | 311.3 | 382.4 | 171.2 | yes |  |
| bert-mini | 4 | 801.3 | 786.8 | 748.5 | 1698.1 | 494.7 | 1.02x | 1.62x | 200.3 | 196.7 | 123.7 | yes |  |
| bert-mini | 8 | 1169.4 | 1078.3 | 1028.6 | 1720.6 | 828.4 | 1.08x | 1.41x | 146.2 | 134.8 | 103.6 | yes |  |
| bert-mini | 16 | 1964.7 | 1801.7 | 1734.7 | 1853.0 | 1526.8 | 1.09x | 1.29x | 122.8 | 112.6 | 95.4 | yes |  |
| bert-mini | 32 | 3640.4 | 3315.9 | 3281.5 | 3500.3 | 2803.4 | 1.10x | 1.30x | 113.8 | 103.6 | 87.6 | yes |  |
| distilgpt2 | 1 | 1301.6 | 1342.1 | 1303.1 | 2801.0 | 1000.3 | 0.97x | 1.30x | 1301.6 | 1342.1 | 1000.3 | yes |  |
| distilgpt2 | 2 | 1857.0 | 1751.1 | 1680.5 | 2796.8 | 1538.8 | 1.06x | 1.21x | 928.5 | 875.5 | 769.4 | yes |  |
| distilgpt2 | 4 | 2986.4 | 2774.5 | 2711.5 | 3128.1 | 2597.2 | 1.08x | 1.15x | 746.6 | 693.6 | 649.3 | yes |  |
| distilgpt2 | 8 | 5168.7 | 5078.5 | 5015.8 | 5875.9 | 4785.8 | 1.02x | 1.08x | 646.1 | 634.8 | 598.2 | yes |  |
| distilgpt2 | 16 | 9412.8 | 9138.3 | 9086.0 | 10751.6 | 9015.2 | 1.03x | 1.04x | 588.3 | 571.1 | 563.5 | yes |  |
| distilgpt2 | 32 | 18391.5 | 17952.9 | 17890.8 | 20927.2 | 18050.4 | 1.02x | 1.02x | 574.7 | 561.0 | 564.1 | yes |  |

## Ablations (median steady_us)

| experiment | variant | model | steady_us | launches/iter | note |
|---|---|---|---|---|---|
| budget_mb | 64 | distilgpt2 | 1292.1 | n/a |  |
| budget_mb | 8 | distilgpt2 | 1289.4 | n/a |  |
| flat_block | 256 | distilgpt2 | 1260.1 | n/a |  |
| flat_block | 256 | resnet18 | 632.9 | n/a |  |
| flat_block | 4096 | distilgpt2 | 1297.4 | n/a |  |
| flat_block | 4096 | resnet18 | 695.5 | n/a |  |
| fusion | off | distilgpt2 | 1961.2 | n/a | enable_opts minus tensor |
| fusion | on | distilgpt2 | 1296.7 | n/a |  |
| max_inputs | mi4 | bert-mini | 472.4 | 34.3 |  |
| max_inputs | mi4 | distilgpt2 | 1336.9 | 45.2 |  |
| max_inputs | mi4 | mixer_b16 | 2736.7 | 76.4 |  |
| max_inputs | mi6 | bert-mini | 422.2 | 24.3 |  |
| max_inputs | mi6 | distilgpt2 | 1288.4 | 38.2 |  |
| max_inputs | mi6 | mixer_b16 | 2679.4 | 63.4 |  |
| max_inputs | mi8 | bert-mini | 424.5 | 24.3 |  |
| max_inputs | mi8 | distilgpt2 | 1288.3 | 38.2 |  |
| max_inputs | mi8 | mixer_b16 | 2680.0 | 63.4 |  |
| precision | float16 | distilgpt2 | 916.9 | n/a |  |
| precision | float16 | smollm2-135m | 2186.9 | n/a |  |
| precision | float32 | distilgpt2 | 1283.4 | n/a |  |
| precision | float32 | smollm2-135m | 3663.1 | n/a |  |
| tf32 | fp32 | resnet18-b1 | 790.6 | n/a |  |
| tf32 | fp32 | resnet18-b8 | 3209.1 | n/a |  |
| tf32 | tf32 | resnet18-b1 | 715.7 | n/a |  |
| tf32 | tf32 | resnet18-b8 | 2706.6 | n/a |  |
| tf32 | torch-fp32 | resnet18-b8 | 3065.6 | n/a |  |

## Deoptimization cost (median over rounds, us per iteration)

| system | pattern | steady_us | first_fail_us | after_fail_us | peak_us | cold_us | launches/it | loops | bridges |
|---|---|---|---|---|---|---|---|---|---|
| ours | never | 37.9 | n/a | n/a | 65 | 3933 | 2.02 | 0 | 0 |
| torch-compile | never | 171.7 | n/a | n/a | 205 | 582488 | n/a | 0 | n/a |
| torch-compile-ro | never | 273.0 | n/a | n/a | 330 | 538300 | n/a | 0 | n/a |
| torch-eager | never | 109.4 | n/a | n/a | 126 | 36256 | n/a | 0 | n/a |
| ours | alternate | 37.0 | 63 | 54 | 271 | 3981 | 2.17 | 0 | 2 |
| torch-compile | alternate | 168.8 | 114901 | 190 | 114901 | 514317 | n/a | 2 | n/a |
| torch-compile-ro | alternate | 275.1 | 128650 | 372 | 128650 | 598694 | n/a | 2 | n/a |
| torch-eager | alternate | 113.7 | 115 | 114 | 138 | 39512 | n/a | 0 | n/a |
| ours | both-hot | 37.2 | n/a | n/a | 65 | 4032 | 2.02 | 0 | 0 |
| torch-compile | both-hot | 162.6 | n/a | n/a | 192 | 588985 | n/a | 0 | n/a |
| torch-compile-ro | both-hot | 245.7 | n/a | n/a | 277 | 584396 | n/a | 0 | n/a |
| torch-eager | both-hot | 113.5 | n/a | n/a | 124 | 37414 | n/a | 0 | n/a |
| ours | fresh | 70.1 | 67 | 70 | 1599 | 4167 | 7.00 | 0 | 63 |
| torch-compile | fresh | 145.1 | 147 | 146 | 167 | 525954 | n/a | 0 | n/a |
| torch-compile-ro | fresh | 164.3 | 166 | 167 | 181 | 591417 | n/a | 0 | n/a |
| torch-eager | fresh | 114.0 | 118 | 114 | 138 | 39741 | n/a | 0 | n/a |
| ours | probe-a | 35.0 | n/a | 54 | 1800 | 3803 | 2.25 | 5 | 3 |
| ours | probe-e | 17.9 | 1349 | 33 | 1554 | 90 | 1.09 | 3 | 4 |

## Compilation overhead (median; break-even vs torch eager, iterations)

| workload | system | compile_ms | first_run_ms | steady_us | break-even |
|---|---|---|---|---|---|
| bert-mini | torch-compile | n/a | 1290.8 | 927.1 | 1366 |
| bert-mini | torch-compile-ro | n/a | 1297.1 | 481.4 | 911 |
| bert-mini | torch-compile-mat | n/a | 2784.7 | 417.6 | 1942 |
| bert-mini | torch-tensorrt | n/a | 4288.2 | 711.6 | 3841 |
| bert-mini | jax | 7048.3 | 2.9 | 243.0 | 4458 |
| bert-mini | iree | 998.4 | 20.3 | 3637.7 | n/a |
| bert-mini | ours | n/a | 183.2 | 423.1 | 65 |
| bert-tiny | torch-compile | n/a | 1135.4 | 540.1 | 1834 |
| bert-tiny | torch-compile-ro | n/a | 1140.1 | 298.6 | 1288 |
| bert-tiny | torch-compile-mat | n/a | 1992.5 | 275.1 | 2283 |
| bert-tiny | torch-tensorrt | n/a | 3536.2 | 478.5 | 5506 |
| bert-tiny | jax | 5785.0 | 1.8 | 110.6 | 5733 |
| bert-tiny | iree | 827.5 | 14.4 | 1345.1 | n/a |
| bert-tiny | ours | n/a | 116.4 | 255.7 | 13 |
| distilgpt2 | torch-compile | n/a | 1522.1 | 1371.2 | 991 |
| distilgpt2 | torch-compile-ro | n/a | 1576.9 | 1306.5 | 984 |
| distilgpt2 | torch-compile-mat | n/a | 2188.7 | 1274.0 | 1370 |
| distilgpt2 | torch-tensorrt | n/a | 14734.4 | 1711.4 | 13715 |
| distilgpt2 | jax | 7192.7 | 4.8 | 1001.9 | 3983 |
| distilgpt2 | iree | 882.0 | 76.2 | 17137.2 | n/a |
| distilgpt2 | ours | n/a | 592.3 | 1289.3 | 311 |
| micro v0 k1 n1000000 | torch-compile | n/a | 439.6 | 54.6 | 17078 |
| micro v0 k1 n1000000 | torch-compile-ro | n/a | 491.5 | 101.6 | n/a |
| micro v0 k1 n1000000 | torch-compile-mat | n/a | 607.0 | 102.9 | n/a |
| micro v0 k1 n1000000 | jax | 201.2 | 0.5 | 52.2 | 6384 |
| micro v0 k1 n1000000 | iree | 570.5 | 24.2 | 5786.7 | n/a |
| micro v0 k1 n1000000 | triton | 319.4 | 0.1 | 30.8 | 5983 |
| micro v0 k4 n10000 | torch-compile | n/a | 438.8 | 51.8 | 55254 |
| micro v0 k4 n10000 | torch-compile-ro | n/a | 502.3 | 106.4 | n/a |
| micro v0 k4 n10000 | torch-compile-mat | n/a | 564.2 | 109.5 | n/a |
| micro v0 k4 n10000 | jax | 202.8 | 0.5 | 41.4 | 9374 |
| micro v0 k4 n10000 | iree | 578.9 | 10.4 | 439.5 | n/a |
| micro v0 k4 n10000 | triton | 319.3 | 0.1 | 19.0 | 7021 |
| micro v0 k4 n100000 | torch-compile | n/a | 428.4 | 45.0 | 24476 |
| micro v0 k4 n100000 | torch-compile-ro | n/a | 514.2 | 109.2 | n/a |
| micro v0 k4 n100000 | torch-compile-mat | n/a | 577.2 | 118.2 | n/a |
| micro v0 k4 n100000 | jax | 198.0 | 0.5 | 44.3 | 9513 |
| micro v0 k4 n100000 | iree | 574.8 | 12.1 | 975.9 | n/a |
| micro v0 k4 n100000 | triton | 301.1 | 0.1 | 20.3 | 6437 |
| micro v0 k4 n1000000 | torch-compile | n/a | 443.5 | 68.4 | 1663 |
| micro v0 k4 n1000000 | torch-compile-ro | n/a | 493.8 | 134.0 | 2558 |
| micro v0 k4 n1000000 | torch-compile-mat | n/a | 585.7 | 133.0 | 3056 |
| micro v0 k4 n1000000 | jax | 210.1 | 0.6 | 94.8 | 795 |
| micro v0 k4 n1000000 | iree | 576.8 | 24.8 | 5756.3 | n/a |
| micro v0 k4 n1000000 | triton | 296.1 | 0.1 | 39.2 | 946 |
| micro v0 k4 n10000000 | torch-compile | n/a | 431.9 | 645.9 | 164 |
| micro v0 k4 n10000000 | torch-compile-ro | n/a | 446.5 | 1224.0 | 223 |
| micro v0 k4 n10000000 | torch-compile-mat | n/a | 604.7 | 1196.0 | 306 |
| micro v0 k4 n10000000 | jax | 219.7 | 1.3 | 946.7 | 87 |
| micro v0 k4 n10000000 | iree | 576.1 | 173.0 | 51561.5 | n/a |
| micro v0 k4 n10000000 | triton | 287.2 | 0.6 | 327.0 | 91 |
| micro v0 k8 n1000000 | torch-compile | n/a | 449.9 | 132.7 | 839 |
| micro v0 k8 n1000000 | torch-compile-ro | n/a | 450.3 | 198.2 | 969 |
| micro v0 k8 n1000000 | torch-compile-mat | n/a | 567.0 | 198.5 | 1243 |
| micro v0 k8 n1000000 | jax | 218.9 | 0.7 | 175.9 | 406 |
| micro v0 k8 n1000000 | iree | 580.7 | 24.4 | 5969.2 | n/a |
| micro v0 k8 n1000000 | triton | 294.9 | 0.2 | 66.9 | 462 |
| micro v1 k4 n1000000 | torch-compile | n/a | 439.5 | 307.8 | 44565 |
| micro v1 k4 n1000000 | torch-compile-ro | n/a | 452.6 | 400.0 | n/a |
| micro v1 k4 n1000000 | torch-compile-mat | n/a | 558.4 | 396.5 | n/a |
| micro v1 k4 n1000000 | jax | 207.4 | 0.6 | 99.3 | 787 |
| micro v1 k4 n1000000 | iree | 572.3 | 24.7 | 6410.2 | n/a |
| micro v1 k4 n1000000 | triton | 321.7 | 0.1 | 43.4 | 1042 |
| micro v10 k1 n191488 | torch-compile | n/a | 1975.3 | 78674.0 | 6288 |
| micro v10 k1 n191488 | torch-compile-ro | n/a | 1975.7 | 78678.0 | 6391 |
| micro v10 k1 n191488 | torch-compile-mat | n/a | 6848.5 | 75790.1 | 2063 |
| micro v10 k1 n191488 | jax | 4783.7 | 70.7 | 41870.8 | 121 |
| micro v10 k1 n191488 | iree | 3801.2 | 208.3 | 136367.7 | n/a |
| micro v10 k1 n25600 | torch-compile | n/a | 1861.3 | 3176.8 | 2994 |
| micro v10 k1 n25600 | torch-compile-ro | n/a | 1980.1 | 3092.4 | 2770 |
| micro v10 k1 n25600 | torch-compile-mat | n/a | 5608.8 | 3161.3 | 9985 |
| micro v10 k1 n25600 | jax | 4257.1 | 6.7 | 1500.8 | 1800 |
| micro v10 k1 n25600 | iree | 1533.3 | 25.9 | 10437.9 | n/a |
| micro v11 k1 n25600 | torch-compile | n/a | 1638.8 | 67.3 | n/a |
| micro v11 k1 n25600 | torch-compile-ro | n/a | 1701.7 | 128.8 | n/a |
| micro v11 k1 n25600 | torch-compile-mat | n/a | 1822.0 | 132.6 | n/a |
| micro v11 k1 n25600 | jax | 88.0 | 0.5 | 49.5 | -13709 |
| micro v11 k1 n25600 | iree | 187.7 | 9.1 | 56.3 | 8389 |
| micro v11 k1 n25600 | triton | 319.3 | 0.1 | 18.7 | 3356 |
| micro v11 k1 n256000 | torch-compile | n/a | 1656.8 | 158.9 | n/a |
| micro v11 k1 n256000 | torch-compile-ro | n/a | 1703.1 | 206.1 | n/a |
| micro v11 k1 n256000 | torch-compile-mat | n/a | 1789.7 | 207.2 | n/a |
| micro v11 k1 n256000 | jax | 84.6 | 0.5 | 52.7 | -1273 |
| micro v11 k1 n256000 | iree | 184.4 | 10.2 | 190.2 | n/a |
| micro v11 k1 n256000 | triton | 319.7 | 0.1 | 20.1 | 834 |
| micro v12 k1 n25600 | torch-compile | n/a | 1788.8 | 137.9 | n/a |
| micro v12 k1 n25600 | torch-compile-ro | n/a | 1679.7 | 165.0 | n/a |
| micro v12 k1 n25600 | torch-compile-mat | n/a | 1945.0 | 168.0 | n/a |
| micro v12 k1 n25600 | jax | 1600.4 | 0.6 | 45.1 | 14981 |
| micro v12 k1 n25600 | iree | 218.9 | 14.1 | 1832.9 | n/a |
| micro v12 k1 n25600 | triton | 320.5 | 0.4 | 298.9 | n/a |
| micro v12 k1 n256000 | torch-compile | n/a | 1736.9 | 493.9 | 16921669 |
| micro v12 k1 n256000 | torch-compile-ro | n/a | 1773.1 | 536.4 | n/a |
| micro v12 k1 n256000 | torch-compile-mat | n/a | 1971.1 | 519.3 | n/a |
| micro v12 k1 n256000 | jax | 1609.5 | 0.9 | 258.2 | 5794 |
| micro v12 k1 n256000 | iree | 230.6 | 9.9 | 436.0 | -63 |
| micro v12 k1 n256000 | triton | 288.5 | 0.4 | 299.8 | 230 |
| micro v13 k1 n25600 | torch-compile | n/a | 1832.9 | 597.7 | n/a |
| micro v13 k1 n25600 | torch-compile-ro | n/a | 1846.0 | 620.5 | n/a |
| micro v13 k1 n25600 | torch-compile-mat | n/a | 1934.7 | 622.6 | n/a |
| micro v13 k1 n25600 | jax | 1924.8 | 1.0 | 123.3 | 3549 |
| micro v13 k1 n25600 | iree | 332.6 | 11.4 | 2797.3 | n/a |
| micro v13 k1 n25600 | triton | 324.8 | 1.3 | 1272.2 | n/a |
| micro v13 k1 n256000 | torch-compile | n/a | 1739.1 | 5029.4 | n/a |
| micro v13 k1 n256000 | torch-compile-ro | n/a | 1771.8 | 5007.3 | n/a |
| micro v13 k1 n256000 | torch-compile-mat | n/a | 2053.5 | 5024.1 | n/a |
| micro v13 k1 n256000 | jax | 1963.6 | 4.8 | 3706.4 | 1450 |
| micro v13 k1 n256000 | iree | 647.9 | 19.4 | 5130.2 | n/a |
| micro v13 k1 n256000 | triton | 326.8 | 4.7 | 4679.4 | 289 |
| micro v2 k4 n1000000 | torch-compile | n/a | 552.3 | 72.7 | 2026 |
| micro v2 k4 n1000000 | torch-compile-ro | n/a | 530.8 | 138.5 | 2623 |
| micro v2 k4 n1000000 | torch-compile-mat | n/a | 702.0 | 134.5 | 3463 |
| micro v2 k4 n1000000 | jax | 207.5 | 0.6 | 177.1 | 1136 |
| micro v2 k4 n1000000 | iree | 575.6 | 24.6 | 8978.1 | n/a |
| micro v2 k4 n1000000 | triton | 293.6 | 0.1 | 69.1 | 991 |
| micro v3 k4 n1000000 | torch-compile | n/a | 572.0 | 167.2 | 2555 |
| micro v3 k4 n1000000 | torch-compile-ro | n/a | 543.8 | 352.1 | 20516 |
| micro v3 k4 n1000000 | torch-compile-mat | n/a | 895.8 | 349.6 | 31561 |
| micro v3 k4 n1000000 | jax | 206.2 | 0.6 | 308.6 | 2497 |
| micro v3 k4 n1000000 | iree | 569.6 | 24.0 | 81591.2 | n/a |
| micro v3 k4 n1000000 | triton | 321.9 | 0.1 | 122.8 | 1124 |
| micro v4 k4 n1000000 | torch-compile | n/a | 443.6 | 332.3 | 40467 |
| micro v4 k4 n1000000 | torch-compile-ro | n/a | 442.1 | 424.4 | n/a |
| micro v4 k4 n1000000 | torch-compile-mat | n/a | 584.5 | 412.2 | n/a |
| micro v4 k4 n1000000 | jax | 207.5 | 0.6 | 126.6 | 793 |
| micro v4 k4 n1000000 | iree | 577.9 | 24.6 | 10790.3 | n/a |
| micro v4 k4 n1000000 | triton | 322.3 | 0.1 | 69.5 | 1046 |
| micro v5 k4 n1000000 | torch-compile | n/a | 581.0 | 332.3 | 54893 |
| micro v5 k4 n1000000 | torch-compile-ro | n/a | 575.0 | 424.4 | n/a |
| micro v5 k4 n1000000 | torch-compile-mat | n/a | 735.8 | 421.0 | n/a |
| micro v5 k4 n1000000 | jax | 206.6 | 0.6 | 127.7 | 786 |
| micro v5 k4 n1000000 | iree | 576.7 | 24.9 | 10808.9 | n/a |
| micro v5 k4 n1000000 | triton | 321.4 | 0.1 | 69.7 | 1038 |
| micro v6 k1 n25600 | torch-compile | n/a | 1728.7 | 405.9 | n/a |
| micro v6 k1 n25600 | torch-compile-ro | n/a | 1792.8 | 432.7 | n/a |
| micro v6 k1 n25600 | torch-compile-mat | n/a | 1865.4 | 433.1 | n/a |
| micro v6 k1 n25600 | jax | 1594.4 | 1.0 | 107.2 | 4744 |
| micro v6 k1 n25600 | iree | 230.3 | 16.8 | 5383.8 | n/a |
| micro v6 k1 n25600 | triton | 320.5 | 0.9 | 874.7 | n/a |
| micro v6 k1 n256000 | torch-compile | n/a | 1763.3 | 1252.9 | n/a |
| micro v6 k1 n256000 | torch-compile-ro | n/a | 1711.3 | 1305.7 | n/a |
| micro v6 k1 n256000 | torch-compile-mat | n/a | 1847.1 | 1283.5 | n/a |
| micro v6 k1 n256000 | jax | 1683.8 | 1.6 | 752.4 | 2891 |
| micro v6 k1 n256000 | iree | 242.1 | 10.3 | 910.1 | 10 |
| micro v6 k1 n256000 | triton | 319.3 | 0.9 | 850.2 | 179 |
| micro v7 k1 n25600 | torch-compile | n/a | 1837.1 | 839.1 | n/a |
| micro v7 k1 n25600 | torch-compile-ro | n/a | 1843.3 | 880.7 | n/a |
| micro v7 k1 n25600 | torch-compile-mat | n/a | 2644.7 | 876.2 | n/a |
| micro v7 k1 n25600 | jax | 2253.2 | 1.6 | 324.6 | 4196 |
| micro v7 k1 n25600 | iree | 332.3 | 28.2 | 11542.1 | n/a |
| micro v7 k1 n256000 | torch-compile | n/a | 1749.1 | 3055.6 | 108838 |
| micro v7 k1 n256000 | torch-compile-ro | n/a | 1769.9 | 3095.3 | n/a |
| micro v7 k1 n256000 | torch-compile-mat | n/a | 2686.0 | 3090.9 | n/a |
| micro v7 k1 n256000 | jax | 2229.9 | 3.8 | 1959.1 | 1770 |
| micro v7 k1 n256000 | iree | 433.1 | 43.2 | 31568.2 | n/a |
| micro v8 k1 n25600 | torch-compile | n/a | 1753.7 | 513.4 | 34975 |
| micro v8 k1 n25600 | torch-compile-ro | n/a | 1769.9 | 522.9 | 45470 |
| micro v8 k1 n25600 | torch-compile-mat | n/a | 2616.0 | 511.3 | 52694 |
| micro v8 k1 n25600 | jax | 1294.2 | 1.3 | 266.8 | 3552 |
| micro v8 k1 n25600 | iree | 500.7 | 9.9 | 922.9 | n/a |
| micro v8 k1 n256000 | torch-compile | n/a | 1914.9 | 29734.5 | n/a |
| micro v8 k1 n256000 | torch-compile-ro | n/a | 1784.4 | 29735.9 | n/a |
| micro v8 k1 n256000 | torch-compile-mat | n/a | 4874.4 | 27770.9 | n/a |
| micro v8 k1 n256000 | jax | 1557.0 | 16.6 | 15419.2 | 105 |
| micro v8 k1 n256000 | iree | 21623.8 | 102.5 | 38532.9 | n/a |
| micro v9 k1 n25600 | torch-compile | n/a | 1867.1 | 863.8 | n/a |
| micro v9 k1 n25600 | torch-compile-ro | n/a | 1790.7 | 881.0 | n/a |
| micro v9 k1 n25600 | torch-compile-mat | n/a | 2259.5 | 875.5 | n/a |
| micro v9 k1 n25600 | jax | 382.7 | 0.8 | 253.7 | 219 |
| micro v9 k1 n256000 | torch-compile | n/a | 1817.2 | 678.4 | n/a |
| micro v9 k1 n256000 | torch-compile-ro | n/a | 1821.5 | 700.2 | n/a |
| micro v9 k1 n256000 | torch-compile-mat | n/a | 2680.4 | 663.1 | n/a |
| micro v9 k1 n256000 | jax | 806.9 | 1.0 | 327.5 | 1965 |
| mixer_b16 | torch-compile | n/a | 1246.3 | 3143.0 | n/a |
| mixer_b16 | torch-compile-ro | n/a | 1266.6 | 3077.8 | n/a |
| mixer_b16 | torch-compile-mat | n/a | 5404.9 | 3105.7 | n/a |
| mixer_b16 | torch-tensorrt | n/a | 5045.7 | 2495.8 | 12181 |
| mixer_b16 | jax | 7327.9 | 7.1 | 2248.5 | 11071 |
| mixer_b16 | iree | 1564.8 | 750.8 | 674112.6 | n/a |
| mixer_b16 | ours | n/a | 191.8 | 2703.1 | 337 |
| resnet18-b1 | torch-compile | n/a | 1107.4 | 1124.6 | 2783 |
| resnet18-b1 | torch-compile-ro | n/a | 1107.4 | 918.7 | 1708 |
| resnet18-b1 | torch-compile-mat | n/a | 5062.1 | 1038.5 | 11767 |
| resnet18-b1 | torch-tensorrt | n/a | 4075.8 | 987.6 | 8354 |
| resnet18-b1 | jax | 612.1 | 4.0 | 1117.5 | 1255 |
| resnet18-b1 | ours | n/a | 130.1 | 720.0 | -91 |
| resnet18-b8 | torch-compile | n/a | 1021.4 | 2194.6 | 11647 |
| resnet18-b8 | torch-compile-ro | n/a | 1092.9 | 2137.6 | 7016 |
| resnet18-b8 | torch-compile-mat | n/a | 5815.4 | 2128.5 | 41020 |
| resnet18-b8 | torch-tensorrt | n/a | 4241.5 | 1949.5 | 12803 |
| resnet18-b8 | jax | 941.4 | 4.3 | 2279.8 | n/a |
| resnet18-b8 | ours | n/a | 150.9 | 2724.5 | n/a |
| smollm2-135m | torch-compile | n/a | 4002.1 | 5203.8 | 348 |
| smollm2-135m | torch-compile-ro | n/a | 3856.7 | 3798.2 | 292 |
| smollm2-135m | torch-compile-mat | n/a | 5258.7 | 3822.8 | 417 |
| smollm2-135m | torch-tensorrt | n/a | 22029.8 | 3726.8 | 1889 |
| smollm2-135m | jax | 10820.8 | 19.2 | 2656.0 | 827 |
| smollm2-135m | iree | 2047.6 | 175.0 | 19171.0 | n/a |
| smollm2-135m | ours | n/a | 537.2 | 3676.9 | -2 |
| tiny-gpt2 | torch-compile | n/a | 1206.1 | 418.5 | 588 |
| tiny-gpt2 | torch-compile-ro | n/a | 1186.2 | 206.6 | 487 |
| tiny-gpt2 | torch-compile-mat | n/a | 1958.4 | 234.3 | 1051 |
| tiny-gpt2 | torch-tensorrt | n/a | 6788.8 | 536.1 | 5778 |
| tiny-gpt2 | jax | 1155.3 | 1.6 | 88.1 | 430 |
| tiny-gpt2 | iree | 389.3 | 12.5 | 198.4 | -66 |
| tiny-gpt2 | ours | n/a | 107.2 | 194.2 | -272 |
| vit-tiny | torch-compile | n/a | 1809.5 | 2326.7 | 1044 |
| vit-tiny | torch-compile-ro | n/a | 1804.9 | 1402.5 | 661 |
| vit-tiny | torch-compile-mat | n/a | 5274.4 | 1336.1 | 1981 |
| vit-tiny | torch-tensorrt | n/a | 7128.8 | 1126.0 | 2494 |
| vit-tiny | jax | 6153.7 | 7.3 | 737.7 | 1887 |
| vit-tiny | iree | 1687.3 | 39.0 | 25019.8 | n/a |
| vit-tiny | ours | n/a | 141.7 | 1003.9 | 2 |

## Warm-up (median over rounds; steady_at/crossover in iterations, 'none' if never reached within N)

| model | system | cache | first (ms) | steady_at | steady_us | crossover vs eager |
|---|---|---|---|---|---|---|
| distilgpt2 | jax | cold | 4.92 | 17 | 1166.1 | 1 |
| distilgpt2 | jax | warm | 51.88 | 6 | 1172.8 | 1 |
| distilgpt2 | ours | cold | 3814.56 | 167 | 1251.0 | none |
| distilgpt2 | ours | warm | 595.15 | 127 | 1258.5 | 293 |
| distilgpt2 | torch-compile | cold | 3869.48 | 69 | 1397.8 | none |
| distilgpt2 | torch-compile | warm | 1504.56 | 63 | 1396.4 | none |
| distilgpt2 | torch-compile-ro | cold | 3144.42 | 46 | 1303.1 | none |
| distilgpt2 | torch-compile-ro | warm | 1563.18 | 20 | 1323.5 | none |
| distilgpt2 | torch-eager | cold | 136.29 | 15 | 3034.7 | none |
| distilgpt2 | torch-eager | warm | 127.91 | 2 | 3014.2 | none |
| tiny-gpt2 | jax | cold | 1.54 | 10 | 160.3 | 1 |
| tiny-gpt2 | jax | warm | 42.82 | 26 | 160.4 | 1 |
| tiny-gpt2 | ours | cold | 557.26 | none | 175.5 | 1 |
| tiny-gpt2 | ours | warm | 114.60 | none | 174.0 | 1 |
| tiny-gpt2 | torch-compile | cold | 3200.17 | 12 | 335.2 | none |
| tiny-gpt2 | torch-compile | warm | 1189.61 | 39 | 370.0 | none |
| tiny-gpt2 | torch-compile-ro | cold | 2417.24 | 15 | 191.9 | none |
| tiny-gpt2 | torch-compile-ro | warm | 1202.93 | 75 | 202.8 | none |
| tiny-gpt2 | torch-eager | cold | 989.86 | 3 | 1573.2 | none |
| tiny-gpt2 | torch-eager | warm | 493.50 | 2 | 1570.0 | none |

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
| ours | first | 32 | 971.1 | 1155.7 | 2 | 4 | 0 | 833 | 0 |
| ours | first | 48 | 1400.9 | 1719.0 | 5 | 0 | 4 | 720 | 0 |
| ours | first | 64 | 1426.0 | 1811.0 | 1 | 3 | 0 | 597 | 0 |
| ours | first | 96 | 1700.2 | 2282.2 | 1 | 0 | 0 | 597 | 0 |
| ours | first | 128 | 2028.0 | 2799.1 | 0 | 1 | 0 | 632 | 0 |
| ours | new | 160 | 2429.9 | 3683.6 | 0 | 0 | 0 | 630 | 0 |
| ours | new | 192 | 2718.9 | 4005.2 | 0 | 0 | 0 | 585 | 0 |
| ours | new | 224 | 3133.1 | 4607.9 | 0 | 0 | 0 | 585 | 0 |
| ours | new | 256 | 3224.1 | 6273.1 | 0 | 0 | 0 | 585 | 0 |
| ours | new | 288 | 3758.9 | 7186.8 | 0 | 0 | 0 | 585 | 0 |
| ours | revisit | 32 | 939.5 | 1137.5 | 0 | 0 | 0 | 630 | 0 |
| ours | revisit | 48 | 1323.0 | 1586.9 | 0 | 0 | 0 | 585 | 0 |
| ours | revisit | 64 | 1387.1 | 1750.0 | 0 | 0 | 0 | 585 | 0 |
| ours | revisit | 96 | 1635.8 | 2198.0 | 0 | 0 | 0 | 585 | 0 |
| ours | revisit | 128 | 1943.8 | 2694.8 | 0 | 0 | 0 | 585 | 0 |
| torch-compile-dynamic | first | 32 | 1069.3 | 1195.0 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-dynamic | first | 48 | 1489.4 | 1628.2 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-dynamic | first | 64 | 1535.7 | 1678.3 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-dynamic | first | 96 | 1716.6 | 1868.0 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-dynamic | first | 128 | 1916.2 | 2080.2 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-dynamic | new | 160 | 2347.5 | 2522.3 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-dynamic | new | 192 | 2582.1 | 2768.1 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-dynamic | new | 224 | 3005.0 | 3200.0 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-dynamic | new | 256 | 3076.6 | 3288.2 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-dynamic | new | 288 | 3731.7 | 3952.3 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-dynamic | revisit | 32 | 1043.8 | 1169.7 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-dynamic | revisit | 48 | 1460.3 | 1591.0 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-dynamic | revisit | 64 | 1505.4 | 1643.0 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-dynamic | revisit | 96 | 1652.5 | 1800.3 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-dynamic | revisit | 128 | 1840.8 | 1999.3 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-static | first | 32 | 998.2 | 1123.6 | 0 | 0 | 0 | 0 | 1 |
| torch-compile-static | first | 48 | 1360.4 | 1492.0 | 0 | 0 | 0 | 0 | 1 |
| torch-compile-static | first | 64 | 1393.3 | 1532.8 | 0 | 0 | 0 | 0 | 1 |
| torch-compile-static | first | 96 | 1626.3 | 1778.8 | 0 | 0 | 0 | 0 | 1 |
| torch-compile-static | first | 128 | 1787.9 | 1952.6 | 0 | 0 | 0 | 0 | 1 |
| torch-compile-static | new | 160 | 2225.0 | 2415.9 | 0 | 0 | 0 | 0 | 1 |
| torch-compile-static | new | 192 | 2418.8 | 2620.9 | 0 | 0 | 0 | 0 | 1 |
| torch-compile-static | new | 224 | 3403.2 | 3614.9 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-static | new | 256 | 3565.3 | 3787.3 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-static | new | 288 | 4229.3 | 4468.0 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-static | revisit | 32 | 997.0 | 1121.1 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-static | revisit | 48 | 1351.8 | 1482.7 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-static | revisit | 64 | 1389.7 | 1524.6 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-static | revisit | 96 | 1624.3 | 1773.3 | 0 | 0 | 0 | 0 | 0 |
| torch-compile-static | revisit | 128 | 1783.8 | 1946.9 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | first | 32 | 2150.6 | 2273.1 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | first | 48 | 2334.4 | 2464.6 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | first | 64 | 2356.1 | 2496.2 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | first | 96 | 2415.4 | 2562.8 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | first | 128 | 2479.3 | 2637.2 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | new | 160 | 2708.1 | 2878.6 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | new | 192 | 2891.5 | 3074.1 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | new | 224 | 3304.0 | 3500.7 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | new | 256 | 3492.1 | 3697.6 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | new | 288 | 4155.4 | 4371.2 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | revisit | 32 | 2146.8 | 2269.4 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | revisit | 48 | 2331.7 | 2459.3 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | revisit | 64 | 2343.9 | 2476.2 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | revisit | 96 | 2397.8 | 2541.8 | 0 | 0 | 0 | 0 | 0 |
| torch-eager | revisit | 128 | 2460.2 | 2619.2 | 0 | 0 | 0 | 0 | 0 |

## Where the operation DAG lives (median over rounds)

| model | DAG in | steady_us | launches/iter | kernels | compiles | gc_ms | nodes deferred | same argmax |
|---|---|---|---|---|---|---|---|---|
| bert-mini | virtual | 419.9 | 24.3 | 109 | 0 | 9 | 0 | 1 |
| bert-mini | deferred | 1587.9 | 24.0 | 112 | 0 | 45 | 65783 | 1 |
| bert-mini | eager | 1173.9 | 287.0 | 103 | 0 | 15 | 0 | 1 |
| bert-tiny | virtual | 245.8 | 14.3 | 109 | 0 | 10 | 0 | 1 |
| bert-tiny | deferred | 670.4 | 14.0 | 112 | 0 | 24 | 38643 | 1 |
| bert-tiny | eager | 674.0 | 169.0 | 103 | 0 | 12 | 0 | 1 |
| distilgpt2 | virtual | 1299.9 | 38.2 | 110 | 0 | 11 | 0 | 1 |
| distilgpt2 | deferred | 1786.3 | 44.0 | 113 | 0 | 55 | 62103 | 1 |
| distilgpt2 | eager | 1960.3 | 271.0 | 103 | 0 | 13 | 0 | 1 |
| mixer_b16 | virtual | 2692.4 | 63.4 | 109 | 0 | 15 | 0 | 1 |
| mixer_b16 | deferred | 4465.2 | 75.0 | 122 | 0 | 141 | 218273 | 1 |
| mixer_b16 | eager | 5873.9 | 950.0 | 103 | 0 | 20 | 0 | 1 |
| resnet18-b1 | virtual | 692.5 | 39.0 | 117 | 0 | 1 | 0 | 1 |
| resnet18-b1 | deferred | 731.4 | 39.0 | 123 | 0 | 19 | 15183 | 1 |
| resnet18-b1 | eager | 854.2 | 87.0 | 114 | 0 | 14 | 0 | 1 |
| smollm2-135m | virtual | 3676.3 | 212.1 | 110 | 0 | 13 | 0 | 1 |
| smollm2-135m | deferred | 4279.2 | 205.0 | 117 | 0 | 77 | 215513 | 1 |
| smollm2-135m | eager | 5762.0 | 968.0 | 104 | 0 | 13 | 0 | 1 |
| tiny-gpt2 | virtual | 195.7 | 13.2 | 111 | 0 | 9 | 0 | 1 |
| tiny-gpt2 | deferred | 405.3 | 15.0 | 113 | 0 | 18 | 22543 | 1 |
| tiny-gpt2 | eager | 413.8 | 99.0 | 103 | 0 | 10 | 0 | 1 |
| vit-tiny | virtual | 984.0 | 76.3 | 111 | 0 | 13 | 0 | 1 |
| vit-tiny | deferred | 3662.5 | 88.0 | 116 | 0 | 124 | 165833 | 1 |
| vit-tiny | eager | 2801.1 | 723.0 | 104 | 0 | 18 | 0 | 1 |

## Where the DAG lives, micro grid (median over rounds)

| variant | k | n | DAG in | steady_us | launches/iter |
|---|---|---|---|---|---|
| 0 | 1 | 10000 | virtual | 4.8 | 1.01 |
| 0 | 1 | 10000 | deferred | 4.8 | 1.00 |
| 0 | 1 | 10000 | eager | 10.5 | 3.00 |
| 0 | 4 | 1000000 | virtual | 117.5 | 4.01 |
| 0 | 4 | 1000000 | deferred | 39.3 | 1.00 |
| 0 | 4 | 1000000 | eager | 307.6 | 12.01 |
| 8 | 1 | 65536 | virtual | 1714.3 | 11.14 |
| 8 | 1 | 65536 | deferred | 1746.9 | 9.00 |
| 8 | 1 | 65536 | eager | 2306.3 | 38.01 |
| 11 | 1 | 256000 | virtual | 14.7 | 2.04 |
| 11 | 1 | 256000 | deferred | 21.6 | 2.00 |
| 11 | 1 | 256000 | eager | 137.6 | 8.01 |
| 13 | 1 | 65536 | virtual | 591.3 | 5.00 |
| 13 | 1 | 65536 | deferred | 591.5 | 5.00 |
| 13 | 1 | 65536 | eager | 791.9 | 10.01 |

## Host-dependent early exit (median over rounds)

| regime | system | total_ms | p50_us | p95_us | max_us | bridges | frame_compiles | exit layers | pass |
|---|---|---|---|---|---|---|---|---|---|
| stable | jax-perlayer | 11346 | 1708.0 | 1776.7 | 1809 | 0 | 0 | 4 | 1 |
| stable | jax-while | 7628 | 1329.4 | 1388.3 | 1452 | 0 | 0 | 4 | 1 |
| stable | ours | 745 | 1069.1 | 2082.1 | 2352 | 1 | 0 | 4 | 1 |
| stable | torch-compile | 1080 | 1481.5 | 1514.2 | 1576 | 0 | 4 | 4 | 1 |
| stable | torch-compile-mat | 1120 | 1354.2 | 1387.6 | 1423 | 0 | 4 | 4 | 1 |
| stable | torch-compile-ro | 1052 | 1371.9 | 1405.2 | 1430 | 0 | 4 | 4 | 1 |
| stable | torch-eager | 529 | 2039.7 | 2094.5 | 2184 | 0 | 0 | 4 | 1 |
| varying | jax-perlayer | 11316 | 1861.6 | 2425.7 | 2594 | 0 | 0 | 2356 | 1 |
| varying | jax-while | 7622 | 1429.6 | 1625.8 | 1693 | 0 | 0 | 2356 | 1 |
| varying | ours | 795 | 1179.0 | 1927.9 | 2475 | 3 | 0 | 2356 | 1 |
| varying | torch-compile | 1043 | 1682.5 | 1975.8 | 2033 | 0 | 4 | 2356 | 1 |
| varying | torch-compile-mat | 1193 | 1526.4 | 1788.6 | 1818 | 0 | 4 | 2356 | 1 |
| varying | torch-compile-ro | 1102 | 1607.4 | 1872.8 | 1908 | 0 | 4 | 2356 | 1 |
| varying | torch-eager | 483 | 2202.0 | 2633.3 | 2717 | 0 | 0 | 2356 | 1 |

