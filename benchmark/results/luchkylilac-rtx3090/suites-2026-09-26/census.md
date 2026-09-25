| suite | model | batch | class | missing ops |
|---|---|---|---|---|
| torchbench | Background_Matting | 1 | missing 3+ | cat, reflection_pad2d, tanh, upsample_bilinear2d |
| torchbench | LearningToPaint | 96 | missing 1-2 | avg_pool2d, sigmoid |
| torchbench | Super_SloMo | 6 | missing 3+ | abs, avg_pool2d, cat, grid_sampler_2d, index, leaky_relu, linspace, mse_loss, rsub, select, sigmoid, slice, stack, upsample_bilinear2d |
| torchbench | alexnet | 128 | missing 1-2 | _adaptive_avg_pool2d |
| torchbench | basic_gnn_edgecnn | 1 | missing 3+ | cat, scatter_reduce_, select |
| torchbench | basic_gnn_gcn | 1 | missing 3+ | cat, eq, index, masked_fill_, ne, pow_, repeat, scatter_add_, select |
| torchbench | basic_gnn_gin | 1 | missing 1-2 | scatter_add_, select |
| torchbench | basic_gnn_sage | 1 | missing 3+ | clamp, scatter_add_, select |
| torchbench | cm3leon_generate | 1 | missing 3+ | _log_softmax, any, cat, cumsum, eq, max, nan_to_num, ne, repeat, select, slice, triu |
| torchbench | dcgan | 256 | missing 1-2 | leaky_relu_, sigmoid |
| torchbench | demucs | 8 | missing 3+ | _cudnn_rnn, convolution[conv1d], convolution[transposed+conv1d], glu, set_, slice |
| torchbench | densenet121 | 64 | missing 1-2 | avg_pool2d, cat |
| torchbench | dlrm | 2048 | missing 3+ | _embedding_bag, cat, index, select |
| torchbench | doctr_det_predictor | 1 | missing 3+ | cat, convolution[transposed], sigmoid, upsample_bilinear2d |
| torchbench | doctr_reco_predictor | 1 | missing 3+ | _cudnn_rnn, max, min, set_ |
| torchbench | drq | 1 | missing 3+ | _is_all_true, eq, gt, split, tanh |
| torchbench | functorch_dp_cifar10 | 64 | missing 1-2 | native_group_norm |
| torchbench | lennard_jones | 1000 | missing 1-2 | tanh |
| torchbench | microbench_unbacked_tolist_sum | 1 | missing 1-2 | select |
| torchbench | mnasnet1_0 | 32 | missing 1-2 | convolution[groups] |
| torchbench | mobilenet_v2 | 16 | missing 1-2 | convolution[groups], hardtanh_ |
| torchbench | mobilenet_v3_large | 32 | missing 3+ | convolution[groups], hardsigmoid, hardswish_ |
| torchbench | moco | 32 | missing 3+ | cat, clamp_min, div_, index, linalg_vector_norm, randperm, select, slice, sort, c10d::allgather_, c10d::broadcast_, profiler::_record_function_enter_new, profiler::_record_function_exit |
| torchbench | nanogpt | 1 | missing 3+ | index, pow, split, tanh |
| torchbench | nvidia_deeprecommender | 256 | missing 1-2 | elu |
| torchbench | opacus_cifar10 | 64 | missing 1-2 | native_group_norm |
| torchbench | phlippe_densenet | 128 | missing 1-2 | avg_pool2d, cat |
| torchbench | phlippe_resnet | 128 | covered, layout | - |
| torchbench | pyhpc_equation_of_state | 1048576 | missing 3+ | log, pow, reciprocal |
| torchbench | pyhpc_isoneutral_mixing | 1048576 | missing 3+ | abs, maximum, minimum, pow, rsub, select, slice, tanh |
| torchbench | pyhpc_turbulent_kinetic_energy | 1048576 | missing 3+ | abs, bitwise_and, eq, ge, gt, logical_not, lt, maximum, minimum, reciprocal, rsub, select, slice, where |
| torchbench | pytorch_CycleGAN_and_pix2pix | 1 | missing 3+ | convolution[transposed], reflection_pad2d, tanh |
| torchbench | pytorch_stargan | 16 | missing 3+ | cat, convolution[transposed], repeat, tanh |
| torchbench | pytorch_unet | 1 | missing 3+ | cat, constant_pad_nd, upsample_bilinear2d |
| torchbench | resnet152 | 32 | covered, layout | - |
| torchbench | resnet18 | 8 | covered, layout | - |
| torchbench | resnet50 | 32 | covered, layout | - |
| torchbench | resnext50_32x4d | 8 | missing 1-2 | convolution[groups] |
| torchbench | sam | 32 | missing 3+ | cat, constant_pad_nd, convolution[transposed], cos, cumsum, gt, index, pow, select, sin, slice, stack, unbind, upsample_bilinear2d |
| torchbench | sam_fast | 32 | missing 3+ | cat, constant_pad_nd, convolution[transposed], cos, cumsum, eq, gt, index, pow, select, sin, slice, stack, unbind, upsample_bilinear2d, where, customflash::custom_flash_aligned |
| torchbench | shufflenet_v2_x1_0 | 64 | missing 3+ | cat, convolution[groups], split |
| torchbench | soft_actor_critic | 256 | missing 3+ | _is_all_true, eq, gt, split, tanh |
| torchbench | squeezenet1_1 | 16 | missing 1-2 | cat |
| torchbench | timm_efficientdet | 32 | missing 3+ | cat, constant_pad_nd, convolution[groups], floor_divide, gather, index, max, remainder, select, sigmoid, silu_, slice, stack, topk, unbind, upsample_nearest2d, torchvision::nms |
| torchbench | timm_efficientnet | 64 | missing 3+ | convolution[groups], sigmoid, silu_ |
| torchbench | timm_nfnet | 128 | missing 3+ | avg_pool2d, constant_pad_nd, convolution[groups], sigmoid |
| torchbench | timm_regnet | 32 | missing 1-2 | convolution[groups], sigmoid |
| torchbench | timm_resnest | 32 | missing 1-2 | avg_pool2d, convolution[groups] |
| torchbench | timm_vision_transformer | 32 | missing 3+ | cat, select, unbind |
| torchbench | timm_vision_transformer_large | 32 | missing 3+ | cat, select, unbind |
| torchbench | timm_vovnet | 32 | missing 1-2 | cat |
| torchbench | torch_multimodal_clip | 32 | missing 3+ | _native_multi_head_attention, cat, clamp_min, index, linalg_vector_norm, select, sigmoid |
| torchbench | tts_angular | 64 | missing 3+ | _cudnn_rnn, _cudnn_rnn_flatten_weight, _use_cudnn_rnn_flatten_weight, clamp_min, linalg_vector_norm, select, set_ |
| torchbench | vgg16 | 4 | missing 1-2 | _adaptive_avg_pool2d |
| torchbench | vision_maskrcnn | 1 | missing 3+ | bitwise_and, cat, clamp, constant_pad_nd, convolution[transposed], eq, floor, ge, gt, index, index_put_, log2, lt, max, nonzero, rsub, select, sigmoid, slice, split_with_sizes, stack, topk, unbind, upsample_bilinear2d, upsample_nearest2d, torchvision::nms, torchvision::roi_align |
| torchbench | yolov3 | 8 | missing 3+ | cat, leaky_relu_, sigmoid, sigmoid_, slice, stack, upsample_nearest2d |
| timm | adv_inception_v3 | 128 | missing 1-2 | avg_pool2d, cat |
| timm | beit_base_patch16_224 | 64 | missing 3+ | cat, constant_pad_nd, index, slice, unbind |
| timm | convnextv2_nano.fcmae_ft_in22k_in1k | 128 | missing 3+ | addcmul, as_strided_, convolution[groups], linalg_vector_norm |
| timm | deit_base_distilled_patch16_224 | 64 | missing 3+ | cat, select, unbind |
| timm | deit_tiny_patch16_224.fb_in1k | 128 | missing 3+ | cat, select, unbind |
| timm | dm_nfnet_f0 | 128 | missing 3+ | avg_pool2d, constant_pad_nd, convolution[groups], sigmoid |
| timm | ghostnet_100 | 512 | missing 3+ | cat, convolution[groups], hardsigmoid |
| timm | inception_v3 | 128 | missing 1-2 | avg_pool2d, cat |
| timm | mobilenetv2_100 | 128 | missing 1-2 | convolution[groups], hardtanh_ |
| timm | mobilenetv3_large_100 | 512 | missing 3+ | convolution[groups], hardsigmoid, hardswish_ |
| timm | mobilevit_s | 64 | missing 3+ | cat, convolution[groups], silu_, unbind |
| timm | nfnet_l0 | 128 | missing 3+ | avg_pool2d, convolution[groups], sigmoid, silu_ |
| timm | repvgg_a2 | 128 | covered, layout | - |
| timm | swin_base_patch4_window7_224 | 64 | missing 3+ | constant_pad_nd, index, roll, unbind |
| timm | tf_efficientnet_b0 | 128 | missing 3+ | constant_pad_nd, convolution[groups], sigmoid, silu_ |
| timm | visformer_small | 128 | missing 1-2 | convolution[groups], unbind |
| timm | vit_base_patch14_dinov2.lvd142m | 128 | missing 3+ | cat, select, unbind |
| timm | vit_base_patch16_siglip_256 | 128 | missing 1-2 | select, unbind |
| torchbench | detectron2_fasterrcnn_r_101_c4 |  | load_fail | - |
| torchbench | detectron2_fasterrcnn_r_101_dc5 |  | load_fail | - |
| torchbench | detectron2_fasterrcnn_r_101_fpn |  | load_fail | - |
| torchbench | detectron2_fasterrcnn_r_50_c4 |  | load_fail | - |
| torchbench | detectron2_fasterrcnn_r_50_dc5 |  | load_fail | - |
| torchbench | detectron2_fasterrcnn_r_50_fpn |  | load_fail | - |
| torchbench | detectron2_fcos_r_50_fpn |  | load_fail | - |
| torchbench | detectron2_maskrcnn |  | load_fail | - |
| torchbench | detectron2_maskrcnn_r_101_c4 |  | load_fail | - |
| torchbench | detectron2_maskrcnn_r_101_fpn |  | load_fail | - |
| torchbench | detectron2_maskrcnn_r_50_c4 |  | load_fail | - |
| torchbench | detectron2_maskrcnn_r_50_fpn |  | load_fail | - |
| torchbench | fastNLP_Bert |  | load_fail | - |
| torchbench | functorch_maml_omniglot |  | load_fail | - |
| torchbench | maml | 1 | run_fail | - |
| torchbench | maml_omniglot |  | load_fail | - |
| torchbench | mobilenet_v2_quantized_qat |  | load_fail | - |
| torchbench | modded_nanogpt | 1 | run_fail | - |
| torchbench | resnet50_quantized_qat |  | load_fail | - |
| torchbench | simple_gpt |  | load_fail | - |
| torchbench | simple_gpt_tp_manual |  | load_fail | - |
| torchbench | speech_transformer |  | load_fail | - |
| torchbench | tacotron2 | 64 | missing 3+ | _cudnn_rnn, _cudnn_rnn_flatten_weight, _pack_padded_sequence, _thnn_fused_lstm_cell, _use_cudnn_rnn_flatten_weight, bitwise_not, cat, convolution[conv1d], lt, masked_fill_, max, select, set_, slice, stack, tanh |
| torchbench | BERT_pytorch | 16 | missing 3+ | _log_softmax, eq, gt, masked_fill, repeat, select, slice, std |
| torchbench | hf_Albert | 1 | missing 3+ | gather, pow, tanh |
| torchbench | hf_Bart | 1 | missing 1-2 | all, select |
| torchbench | hf_Bert | 1 | missing 1-2 | gather |
| torchbench | hf_Bert_large | 1 | missing 1-2 | gather |
| torchbench | hf_BigBird | 1 | missing 3+ | any, cat, eq, gather, index, minimum, pow, rsub, scatter_, select, slice, tanh, unsqueeze_ |
| torchbench | hf_DistilBert | 1 | covered, layout | - |
| torchbench | hf_GPT2 | 1 | missing 3+ | all, cat, cumsum, eq, ne, pow, select, slice, split, tanh |
| torchbench | hf_GPT2_large | 1 | missing 3+ | all, cat, cumsum, eq, ne, pow, select, slice, split, tanh |
| torchbench | hf_Longformer | 1 | missing 3+ | any, bitwise_and, clamp, constant_pad_nd, cumsum, div_, eq, flip, ge, gt, index, index_put_, lt, masked_fill, ne, select, slice, tril, where |
| torchbench | hf_Reformer | 1 | missing 3+ | any, cat, eq, gather, index, logsumexp, max, ne, pow, randn, remainder, repeat, scatter_, select, slice, sort, split, where |
| torchbench | hf_Roberta_base | 1 | missing 3+ | cumsum, gather, ne |
| torchbench | hf_T5 | 1 | missing 3+ | abs, gt, le, log, lt, minimum, pow, where |
| torchbench | hf_T5_base | 1 | missing 3+ | abs, gt, le, log, lt, minimum, pow, where |
| torchbench | hf_T5_generate | 1 | missing 3+ | abs, any, bitwise_and, bitwise_not, bitwise_or, cat, eq, gt, isin, le, log, lt, max, minimum, pow, rsub, select, slice, where |
| torchbench | hf_T5_large | 1 | missing 3+ | abs, gt, le, log, lt, minimum, pow, where |
| torchbench | hf_Whisper | 8 | missing 1-2 | convolution[conv1d] |
| torchbench | hf_clip |  | load_fail | - |
| torchbench | hf_distil_whisper | 1 | missing 1-2 | convolution[conv1d] |
| torchbench | llama | 32 | missing 3+ | pow, select, slice, triu, view_as_complex, view_as_real |
| torchbench | llama_v2_7b_16h |  | load_fail | - |
| torchbench | llava |  | load_fail | - |
| torchbench | moondream | 1 | missing 3+ | all, cat, cos, cumsum, eq, ne, pow, select, sin, slice, tanh |
| torchbench | stable_diffusion_text_encoder |  | load_fail | - |
| torchbench | stable_diffusion_unet |  | load_fail | - |
| huggingface | AlbertForMaskedLM | 4 | missing 3+ | _log_softmax, gather, nll_loss_forward, pow, tanh |
| huggingface | AllenaiLongformerBase | 4 | missing 3+ | _log_softmax, any, bitwise_and, clamp, constant_pad_nd, cumsum, div_, eq, flip, ge, gt, index, index_put_, lt, masked_fill, ne, nll_loss_forward, select, slice, tril, where |
| huggingface | BartForCausalLM | 4 | missing 3+ | _log_softmax, all, nll_loss_forward |
| huggingface | BertForMaskedLM | 16 | missing 3+ | _log_softmax, gather, nll_loss_forward |
| huggingface | BlenderbotForCausalLM | 4 | missing 3+ | _log_softmax, all, nll_loss_forward |
| huggingface | BlenderbotForConditionalGeneration | 16 | missing 3+ | _log_softmax, all, nll_loss_forward |
| huggingface | DebertaV2ForMaskedLM | 2 | missing 3+ | _log_softmax, any, bitwise_not, eq, index, masked_fill, nll_loss_forward |
| huggingface | DistilBertForMaskedLM | 128 | missing 3+ | _log_softmax, nll_loss_forward, slice |
| huggingface | DistillGPT2 | 16 | missing 3+ | _log_softmax, all, cat, constant_pad_nd, cumsum, eq, ne, nll_loss_forward, pow, select, slice, split, tanh |
| huggingface | ElectraForCausalLM | 32 | missing 3+ | _log_softmax, constant_pad_nd, gather, nll_loss_forward, slice |
| huggingface | GPT2ForSequenceClassification | 4 | missing 3+ | _log_softmax, all, any, cat, cumsum, eq, index, ne, nll_loss_forward, pow, select, slice, split, tanh |
| huggingface | GPTJForCausalLM |  | load_fail | - |
| huggingface | GPTJForQuestionAnswering |  | load_fail | - |
| huggingface | GPTNeoForCausalLM | 32 | missing 3+ | _log_softmax, all, cat, constant_pad_nd, cumsum, eq, le, ne, nll_loss_forward, pow, select, slice, tanh, where |
| huggingface | GPTNeoForSequenceClassification | 32 | missing 3+ | _log_softmax, all, cat, cumsum, eq, index, le, ne, nll_loss_forward, pow, select, slice, tanh, where |
| huggingface | GoogleFnet | 16 | missing 3+ | _fft_c2c, _log_softmax, nll_loss_forward, pow, select, tanh, view_as_real |
| huggingface | LayoutLMForMaskedLM | 16 | missing 3+ | _log_softmax, any, eq, index, nll_loss_forward, rsub, select, tanh |
| huggingface | M2M100ForConditionalGeneration | 16 | missing 3+ | _log_softmax, all, cumsum, ne, nll_loss_forward |
| huggingface | MBartForCausalLM | 4 | missing 3+ | _log_softmax, all, nll_loss_forward |
| huggingface | MT5ForConditionalGeneration | 16 | missing 3+ | _log_softmax, abs, gt, le, log, lt, minimum, nll_loss_forward, pow, tanh, where |
| huggingface | MegatronBertForCausalLM | 4 | missing 3+ | _log_softmax, all, any, constant_pad_nd, eq, index, nll_loss_forward, slice |
| huggingface | MobileBertForMaskedLM | 128 | missing 3+ | _log_softmax, cat, constant_pad_nd, nll_loss_forward, slice |
| huggingface | OPTForCausalLM | 2 | missing 3+ | _log_softmax, all, constant_pad_nd, cumsum, nll_loss_forward, slice |
| huggingface | PLBartForCausalLM | 8 | missing 3+ | _log_softmax, all, nll_loss_forward |
| huggingface | PegasusForCausalLM | 32 | missing 3+ | _log_softmax, all, nll_loss_forward |
| huggingface | RobertaForCausalLM | 16 | missing 3+ | _log_softmax, constant_pad_nd, cumsum, gather, ne, nll_loss_forward, slice |
| huggingface | T5ForConditionalGeneration | 4 | missing 3+ | _log_softmax, abs, gt, le, log, lt, minimum, nll_loss_forward, pow, where |
| huggingface | T5Small | 4 | missing 3+ | _log_softmax, abs, gt, le, log, lt, minimum, nll_loss_forward, pow, where |
| huggingface | TrOCRForCausalLM | 32 | missing 3+ | _log_softmax, le, nll_loss_forward, where |
| huggingface | XGLMForCausalLM | 8 | missing 3+ | _log_softmax, constant_pad_nd, le, maximum, nll_loss_forward, slice, where |
| huggingface | XLNetLMHeadModel | 8 | missing 3+ | _log_softmax, cat, cos, nll_loss_forward, pow, reciprocal, sin, slice |
| huggingface | YituTechConvBert | 16 | missing 3+ | _log_softmax, all, any, cat, convolution[conv1d], convolution[groups+conv1d], eq, im2col, index, nll_loss_forward |
| huggingface | meta-llama/Llama-3.2-1B |  | load_fail | - |
| huggingface | google/gemma-2-2b |  | load_fail | - |
| huggingface | google/gemma-3-4b-it |  | load_fail | - |
| huggingface | openai/whisper-tiny | 1 | missing 3+ | cat, convolution[conv1d], index, repeat |
| huggingface | Qwen/Qwen3-0.6B | 1 | missing 3+ | cat, cos, pow, sin, slice |
| huggingface | mistralai/Mistral-7B-Instruct-v0.3 |  | load_fail | - |
| huggingface | openai/gpt-oss-20b |  | load_fail | - |

| suite | class | models |
|---|---|---|
| huggingface | load_fail | 7 |
| huggingface | missing 3+ | 32 |
| timm | covered, layout | 1 |
| timm | missing 1-2 | 5 |
| timm | missing 3+ | 12 |
| torchbench | covered, layout | 5 |
| torchbench | load_fail | 25 |
| torchbench | missing 1-2 | 24 |
| torchbench | missing 3+ | 48 |
| torchbench | run_fail | 2 |

| missing op | models needing it |
|---|---|
| cat | 47 |
| select | 41 |
| slice | 36 |
| _log_softmax | 32 |
| nll_loss_forward | 30 |
| pow | 27 |
| eq | 24 |
| index | 23 |
| tanh | 22 |
| constant_pad_nd | 20 |
| convolution[groups] | 19 |
| all | 18 |
| cumsum | 16 |
| gt | 16 |
| where | 16 |
| ne | 15 |
| sigmoid | 14 |
| unbind | 14 |
| lt | 12 |
| any | 11 |
| gather | 11 |
| le | 11 |
| avg_pool2d | 10 |
| abs | 10 |
| minimum | 10 |
| split | 9 |
| log | 8 |
| upsample_bilinear2d | 7 |
| rsub | 7 |
| stack | 7 |
| max | 7 |
| repeat | 6 |
| convolution[conv1d] | 6 |
| convolution[transposed] | 6 |
| bitwise_and | 5 |
| cos | 5 |
| sin | 5 |
| silu_ | 5 |
| clamp | 4 |
| _cudnn_rnn | 4 |
| set_ | 4 |
| linalg_vector_norm | 4 |
| ge | 4 |
| masked_fill | 4 |
| scatter_add_ | 3 |
| hardsigmoid | 3 |
| clamp_min | 3 |
| div_ | 3 |
| reciprocal | 3 |
| maximum | 3 |
| upsample_nearest2d | 3 |
| index_put_ | 3 |
| bitwise_not | 3 |
| reflection_pad2d | 2 |
| _adaptive_avg_pool2d | 2 |
| masked_fill_ | 2 |
| triu | 2 |
| leaky_relu_ | 2 |
| _is_all_true | 2 |
| native_group_norm | 2 |
| hardtanh_ | 2 |
| hardswish_ | 2 |
| sort | 2 |
| remainder | 2 |
| topk | 2 |
| torchvision::nms | 2 |
| _cudnn_rnn_flatten_weight | 2 |
| _use_cudnn_rnn_flatten_weight | 2 |
| scatter_ | 2 |
| flip | 2 |
| tril | 2 |
| view_as_real | 2 |
| grid_sampler_2d | 1 |
| leaky_relu | 1 |
| linspace | 1 |
| mse_loss | 1 |
| scatter_reduce_ | 1 |
| pow_ | 1 |
| nan_to_num | 1 |
| convolution[transposed+conv1d] | 1 |
| glu | 1 |
| _embedding_bag | 1 |
| min | 1 |
| randperm | 1 |
| c10d::allgather_ | 1 |
| c10d::broadcast_ | 1 |
| profiler::_record_function_enter_new | 1 |
| profiler::_record_function_exit | 1 |
| elu | 1 |
| logical_not | 1 |
| customflash::custom_flash_aligned | 1 |
| floor_divide | 1 |
| _native_multi_head_attention | 1 |
| floor | 1 |
| log2 | 1 |
| nonzero | 1 |
| split_with_sizes | 1 |
| torchvision::roi_align | 1 |
| sigmoid_ | 1 |
| addcmul | 1 |
| as_strided_ | 1 |
| roll | 1 |
| _pack_padded_sequence | 1 |
| _thnn_fused_lstm_cell | 1 |
| std | 1 |
| unsqueeze_ | 1 |
| logsumexp | 1 |
| randn | 1 |
| bitwise_or | 1 |
| isin | 1 |
| view_as_complex | 1 |
| _fft_c2c | 1 |
| convolution[groups+conv1d] | 1 |
| im2col | 1 |
