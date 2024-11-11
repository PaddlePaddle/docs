# PyTorch 最新 release 与 Paddle develop API 映射表

本文梳理了 PyTorch 最新发行版（当前 v2.3.0） API 与 PaddlePaddle develop 版本 API 对应关系与差异分析。通过本文档，帮助开发者快速迁移 PyTorch 使用经验，完成模型的开发与调优。

## 贡献代码

欢迎你向我们贡献代码，关于如何编写 API 映射关系，为保证文档格式统一性与可读性，请严格参照 [API 映射关系-格式与模板](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_format_cn.md) 来编写。

## API 映射表目录

| 类别 | 简介 |
| ---- | --- |
| [torch.XX](#id1) | 主要为`torch.XX`类 API |
| [torch.nn.XX](#id2) | 主要为`torch.nn.XX`类 API |
| [torch.nn.functional.XX](#id3) | 主要为`torch.nn.functional.XX`类 API |
| [torch.nn.init.XX](#id4) | 主要为`torch.nn.init.XX`类 API |
| [torch.nn.utils.XX](#id5) | 主要为`torch.nn.utils.XX`类 API |
| [torch.nn.Module.XX](#id15) | 主要为`torch.nn.Module.XX`类 API |
| [torch.Tensor.XX](#id6) | 主要为`torch.Tensor.XX`类 API |
| [torch.autograd.XX](#id20) | 主要为`torch.autograd.XX`类 API |
| [torch.cuda.XX](#id7) | 主要为`torch.cuda.XX`类 API |
| [torch.distributed.XX](#id8) | 主要为`torch.distributed.XX`类 API |
| [torch.distributions.XX](#id9)   | 主要为`torch.distributions.XX`类 API |
| [torch.fft.XX](#id10)   | 主要为`torch.fft.XX`类 API |
| [torch.hub.XX](#id14)   | 主要为`torch.hub.XX`类 API |
| [torch.linalg.XX](#id11)   | 主要为`torch.linalg.XX`类 API |
| [torch.onnx.XX](#id15)   | 主要为`torch.onnx.XX`类 API |
| [torch.profiler.XX](#id21)   | 主要为`torch.profiler.XX`类 API |
| [torch.optim.XX](#id22)   | 主要为`torch.optim.XX`类 API |
| [torch.sparse.XX](#id12)   | 主要为`torch.sparse.XX`类 API |
| [torch 其他](#id13)   | PyTorch 其他 API |
| [fairscale.xx](#id23)   | 第三方库 fairscale API |
| [transformers.xx](#id24)   | 第三方库 transformers API |
| [flash_attn.xx](#id25)   | 第三方库 flash_attn API |

## torch.XX API 映射列表

梳理了`torch.XX`类 API 的 PyTorch-PaddlePaddle API 映射列表。

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ----- | ----------- | ----------------- | ----------- | ------- |
| REFERENCE-MAPPING-TABLE(`torch.`, max_depth=1) |

***持续更新...***

## torch.nn.XX API 映射列表

梳理了`torch.nn.XX`类 API 的 PyTorch-PaddlePaddle API 映射列表。

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ----- | ----------- | ----------------- | ----------- | ------- |
| REFERENCE-MAPPING-TABLE(`torch.nn.`) |

***持续更新...***

## torch.nn.functional.XX API 映射列表
梳理了`torch.nn.functional.XX`类 API 的 PyTorch-PaddlePaddle API 映射列表。

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ----- | ----------- | ----------------- | ----------- | ------- |
| REFERENCE-MAPPING-TABLE(`torch.nn.functional`) |

***持续更新...***

## torch.Tensor.XX API 映射列表

梳理了`torch.Tensor.XX`类 API 的 PyTorch-PaddlePaddle API 映射列表。

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ----- | ----------- | ----------------- | ----------- | ------- |
| REFERENCE-MAPPING-TABLE(`torch.Tensor.`) |

***持续更新...***

## torch.nn.init.XX API 映射列表
梳理了`torch.nn.init.XX`类 API 的 PyTorch-PaddlePaddle API 映射列表。

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ----- | ----------- | ----------------- | ----------- | ------- |
| REFERENCE-MAPPING-TABLE(`torch.nn.init.`) |

***持续更新...***

## torch.nn.utils.XX API 映射列表
梳理了`torch.nn.utils.XX`类 API 的 PyTorch-PaddlePaddle API 映射列表。

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ----- | ----------- | ----------------- | ----------- | ------- |
| REFERENCE-MAPPING-TABLE(`torch.nn.utils.`) |

***持续更新...***

## torch.nn.Module.XX API 映射列表
梳理了`torch.nn.Module.XX`类 API 的 PyTorch-PaddlePaddle API 映射列表。

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ----- | ----------- | ----------------- | ----------- | ------- |
| REFERENCE-MAPPING-TABLE(`torch.nn.Module.`) |

***持续更新...***

## torch.autograd.XX API 映射列表
梳理了`torch.autograd.XX`类 API 的 PyTorch-PaddlePaddle API 映射列表。

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ----- | ----------- | ----------------- | ----------- | ------- |
| REFERENCE-MAPPING-TABLE(`torch.autograd.`) |

***持续更新...***

## torch.cuda.XX API 映射列表
梳理了`torch.cuda.XX`类 API 的 PyTorch-PaddlePaddle API 映射列表。

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ----- | ----------- | ----------------- | ----------- | ------- |
| REFERENCE-MAPPING-TABLE(`torch.cuda.`) |

***持续更新...***

## torch.distributed.XX API 映射列表
梳理了`torch.distributed.XX`类 API 的 PyTorch-PaddlePaddle API 映射列表。

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ----- | ----------- | ----------------- | ----------- | ------- |
| REFERENCE-MAPPING-TABLE(`torch.distributed.`) |

***持续更新...***

## torch.distributions.XX API 映射列表
梳理了`torch.distributions.XX`类 API 的 PyTorch-PaddlePaddle API 映射列表。

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ----- | ----------- | ----------------- | ----------- | ------- |
| REFERENCE-MAPPING-TABLE(`torch.distributions.`) |

***持续更新...***

## torch.fft.XX API 映射列表
梳理了`torch.fft.XX`类 API 的 PyTorch-PaddlePaddle API 映射列表。

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ----- | ----------- | ----------------- | ----------- | ------- |
| REFERENCE-MAPPING-TABLE(`torch.fft.`) |

***持续更新...***

## torch.hub.XX API 映射列表

梳理了`torch.hub.XX`类 API 的 PyTorch-PaddlePaddle API 映射列表。

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ----- | ----------- | ----------------- | ----------- | ------- |
| REFERENCE-MAPPING-TABLE(`torch.hub.`) |


***持续更新...***

## torch.linalg.XX API 映射列表

梳理了`torch.linalg.XX`类 API 的 PyTorch-PaddlePaddle API 映射列表。

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ----- | ----------- | ----------------- | ----------- | ------- |
| REFERENCE-MAPPING-TABLE(`torch.linalg.`) |

***持续更新...***

## torch.onnx.XX API 映射列表

梳理了`torch.onnx.XX`类 API 的 PyTorch-PaddlePaddle API 映射列表。

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ----- | ----------- | ----------------- | ----------- | ------- |
| REFERENCE-MAPPING-TABLE(`torch.onnx.`) |

***持续更新...***

## torch.optim.XX API 映射列表
梳理了`torch.optim.XX`类 API 的 PyTorch-PaddlePaddle API 映射列表。

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ----- | ----------- | ----------------- | ----------- | ------- |
| REFERENCE-MAPPING-TABLE(`torch.optim.`) |

***持续更新...***

## torch.profiler.XX API 映射列表

梳理了`torch.profiler.XX`类 API 的 PyTorch-PaddlePaddle API 映射列表。

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ----- | ----------- | ----------------- | ----------- | ------- |
| REFERENCE-MAPPING-TABLE(`torch.profiler.`) |

***持续更新...***

## torch.sparse.XX API 映射列表

梳理了`torch.sparse.XX`类 API 的 PyTorch-PaddlePaddle API 映射列表。

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ----- | ----------- | ----------------- | ----------- | ------- |
| REFERENCE-MAPPING-TABLE(`torch.sparse.`) |

***持续更新...***

## PyTorch 其他类 API 映射列表

梳理了 PyTorch 其他类 API 的 PyTorch-PaddlePaddle API 映射列表。

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ----- | ----------- | ----------------- | ----------- | ------- |
| REFERENCE-MAPPING-TABLE(`torch.`) |

***持续更新...***

## fairscale.XX API 映射列表

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ----- | ----------- | ----------------- | ----------- | ------- |
| REFERENCE-MAPPING-TABLE(`fairscale.`) |

***持续更新...***

## transformers.XX API 映射列表

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ----- | ----------- | ----------------- | ----------- | ------- |
| REFERENCE-MAPPING-TABLE(`transformers.`) |

***持续更新...***

## flash_attn.XX API 映射列表

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ----- | ----------- | ----------------- | ----------- | ------- |
| REFERENCE-MAPPING-TABLE(`flash_attn.`) |

***持续更新...***

## API 别名映射列表

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ---- | -------------------- | -------------- | ----------- | ---- |
| ALIAS-REFERENCE-ITEM(`torch.Tensor.absolute_`, `torch.Tensor.abs_`) |
| ALIAS-REFERENCE-ITEM(`torch.Tensor.acos`, `torch.Tensor.arccos`) |
| ALIAS-REFERENCE-ITEM(`torch.Tensor.asin`, `torch.Tensor.arcsin`) |
| ALIAS-REFERENCE-ITEM(`torch.Tensor.atan`, `torch.Tensor.arctan`) |
| ALIAS-REFERENCE-ITEM(`torch.Tensor.atan2`, `torch.Tensor.arctan2`) |
| ALIAS-REFERENCE-ITEM(`torch.absolute_`, `torch.abs_`) |
| ALIAS-REFERENCE-ITEM(`torch.adaptive_avg_pool1d`, `torch.nn.functional.adaptive_avg_pool1d`) |
| ALIAS-REFERENCE-ITEM(`torch.autograd.function.Function`, `torch.autograd.Function`) |
| ALIAS-REFERENCE-ITEM(`torch.avg_pool1d`, `torch.nn.functional.avg_pool1d`) |
| ALIAS-REFERENCE-ITEM(`torch.bilinear`, `torch.nn.functional.bilinear`) |
| ALIAS-REFERENCE-ITEM(`torch.conv1d`, `torch.nn.functional.conv1d`) |
| ALIAS-REFERENCE-ITEM(`torch.conv2d`, `torch.nn.functional.conv2d`) |
| ALIAS-REFERENCE-ITEM(`torch.conv3d`, `torch.nn.functional.conv3d`) |
| ALIAS-REFERENCE-ITEM(`torch.conv_transpose1d`, `torch.nn.functional.conv_transpose1d`) |
| ALIAS-REFERENCE-ITEM(`torch.conv_transpose2d`, `torch.nn.functional.conv_transpose2d`) |
| ALIAS-REFERENCE-ITEM(`torch.conv_transpose3d`, `torch.nn.functional.conv_transpose3d`) |
| ALIAS-REFERENCE-ITEM(`torch.cuda.amp.autocast_mode.autocast`, `torch.cuda.amp.autocast`) |
| ALIAS-REFERENCE-ITEM(`torch.distributions.AbsTransform`, `torch.distributions.transforms.AbsTransform`) |
| ALIAS-REFERENCE-ITEM(`torch.distributions.AffineTransform`, `torch.distributions.transforms.AffineTransform`) |
| ALIAS-REFERENCE-ITEM(`torch.distributions.Bernoulli`, `torch.distributions.bernoulli.Bernoulli`) |
| ALIAS-REFERENCE-ITEM(`torch.distributions.Beta`, `torch.distributions.beta.Beta`) |
| ALIAS-REFERENCE-ITEM(`torch.distributions.Categorical`, `torch.distributions.categorical.Categorical`) |
| ALIAS-REFERENCE-ITEM(`torch.distributions.Cauchy`, `torch.distributions.cauchy.Cauchy`) |
| ALIAS-REFERENCE-ITEM(`torch.distributions.ComposeTransform`, `torch.distributions.transforms.ComposeTransform`) |
| ALIAS-REFERENCE-ITEM(`torch.distributions.Dirichlet`, `torch.distributions.dirichlet.Dirichlet`) |
| ALIAS-REFERENCE-ITEM(`torch.distributions.Distribution`, `torch.distributions.distribution.Distribution`) |
| ALIAS-REFERENCE-ITEM(`torch.distributions.ExpTransform`, `torch.distributions.transforms.ExpTransform`) |
| ALIAS-REFERENCE-ITEM(`torch.distributions.ExponentialFamily`, `torch.distributions.exp_family.ExponentialFamily`) |
| ALIAS-REFERENCE-ITEM(`torch.distributions.Geometric`, `torch.distributions.geometric.Geometric`) |
| ALIAS-REFERENCE-ITEM(`torch.distributions.Gumbel`, `torch.distributions.gumbel.Gumbel`) |
| ALIAS-REFERENCE-ITEM(`torch.distributions.Independent`, `torch.distributions.independent.Independent`) |
| ALIAS-REFERENCE-ITEM(`torch.distributions.IndependentTransform`, `torch.distributions.transforms.IndependentTransform`) |
| ALIAS-REFERENCE-ITEM(`torch.distributions.Laplace`, `torch.distributions.laplace.Laplace`) |
| ALIAS-REFERENCE-ITEM(`torch.distributions.LogNormal`, `torch.distributions.log_normal.LogNormal`) |
| ALIAS-REFERENCE-ITEM(`torch.distributions.Multinomial`, `torch.distributions.multinomial.Multinomial`) |
| ALIAS-REFERENCE-ITEM(`torch.distributions.Normal`, `torch.distributions.normal.Normal`) |
| ALIAS-REFERENCE-ITEM(`torch.distributions.PowerTransform`, `torch.distributions.transforms.PowerTransform`) |
| ALIAS-REFERENCE-ITEM(`torch.distributions.ReshapeTransform`, `torch.distributions.transforms.ReshapeTransform`) |
| ALIAS-REFERENCE-ITEM(`torch.distributions.SigmoidTransform`, `torch.distributions.transforms.SigmoidTransform`) |
| ALIAS-REFERENCE-ITEM(`torch.distributions.SoftmaxTransform`, `torch.distributions.transforms.SoftmaxTransform`) |
| ALIAS-REFERENCE-ITEM(`torch.distributions.StackTransform`, `torch.distributions.transforms.StackTransform`) |
| ALIAS-REFERENCE-ITEM(`torch.distributions.StickBreakingTransform`, `torch.distributions.transforms.StickBreakingTransform`) |
| ALIAS-REFERENCE-ITEM(`torch.distributions.TanhTransform`, `torch.distributions.transforms.TanhTransform`) |
| ALIAS-REFERENCE-ITEM(`torch.distributions.Transform`, `torch.distributions.transforms.Transform`) |
| ALIAS-REFERENCE-ITEM(`torch.distributions.TransformedDistribution`, `torch.distributions.transformed_distribution.TransformedDistribution`) |
| ALIAS-REFERENCE-ITEM(`torch.distributions.Uniform`, `torch.distributions.uniform.Uniform`) |
| ALIAS-REFERENCE-ITEM(`torch.greater_equal`, `torch.ge`) |
| ALIAS-REFERENCE-ITEM(`torch.group_norm`, `torch.nn.functional.group_norm`) |
| ALIAS-REFERENCE-ITEM(`torch.hardshrink`, `torch.nn.functional.hardshrink`) |
| ALIAS-REFERENCE-ITEM(`torch.layer_norm`, `torch.nn.functional.layer_norm`) |
| ALIAS-REFERENCE-ITEM(`torch.logsumexp`, `torch.special.logsumexp`) |
| ALIAS-REFERENCE-ITEM(`torch.matrix_exp`, `torch.linalg.matrix_exp`) |
| ALIAS-REFERENCE-ITEM(`torch.nn.NLLLoss2d`, `torch.nn.NLLLoss`) |
| ALIAS-REFERENCE-ITEM(`torch.nn.modules.AvgPool1d`, `torch.nn.AvgPool1d`) |
| ALIAS-REFERENCE-ITEM(`torch.nn.modules.AvgPool2d`, `torch.nn.AvgPool2d`) |
| ALIAS-REFERENCE-ITEM(`torch.nn.modules.AvgPool3d`, `torch.nn.AvgPool3d`) |
| ALIAS-REFERENCE-ITEM(`torch.nn.modules.BatchNorm1d`, `torch.nn.BatchNorm1d`) |
| ALIAS-REFERENCE-ITEM(`torch.nn.modules.BatchNorm2d`, `torch.nn.BatchNorm2d`) |
| ALIAS-REFERENCE-ITEM(`torch.nn.modules.BatchNorm3d`, `torch.nn.BatchNorm3d`) |
| ALIAS-REFERENCE-ITEM(`torch.nn.modules.CosineSimilarity`, `torch.nn.CosineSimilarity`) |
| ALIAS-REFERENCE-ITEM(`torch.nn.modules.Dropout`, `torch.nn.Dropout`) |
| ALIAS-REFERENCE-ITEM(`torch.nn.modules.GroupNorm`, `torch.nn.GroupNorm`) |
| ALIAS-REFERENCE-ITEM(`torch.nn.modules.LSTM`, `torch.nn.LSTM`) |
| ALIAS-REFERENCE-ITEM(`torch.nn.modules.Module`, `torch.nn.Module`) |
| ALIAS-REFERENCE-ITEM(`torch.nn.modules.RNN`, `torch.nn.RNN`) |
| ALIAS-REFERENCE-ITEM(`torch.nn.modules.RNNBase`, `torch.nn.RNNBase`) |
| ALIAS-REFERENCE-ITEM(`torch.nn.modules.RNNCell`, `torch.nn.RNNCell`) |
| ALIAS-REFERENCE-ITEM(`torch.nn.modules.SyncBatchNorm`, `torch.nn.SyncBatchNorm`) |
| ALIAS-REFERENCE-ITEM(`torch.nn.modules.activation.ReLU`, `torch.nn.ReLU`) |
| ALIAS-REFERENCE-ITEM(`torch.nn.modules.batchnorm.BatchNorm1d`, `torch.nn.BatchNorm1d`) |
| ALIAS-REFERENCE-ITEM(`torch.nn.modules.batchnorm.BatchNorm2d`, `torch.nn.BatchNorm2d`) |
| ALIAS-REFERENCE-ITEM(`torch.nn.modules.batchnorm.BatchNorm3d`, `torch.nn.BatchNorm3d`) |
| ALIAS-REFERENCE-ITEM(`torch.nn.modules.batchnorm.SyncBatchNorm`, `torch.nn.SyncBatchNorm`) |
| ALIAS-REFERENCE-ITEM(`torch.nn.modules.conv.Conv2d`, `torch.nn.Conv2d`) |
| ALIAS-REFERENCE-ITEM(`torch.nn.modules.distance.CosineSimilarity`, `torch.nn.CosineSimilarity`) |
| ALIAS-REFERENCE-ITEM(`torch.nn.modules.module.Module`, `torch.nn.Module`) |
| ALIAS-REFERENCE-ITEM(`torch.nn.modules.pooling.AvgPool1d`, `torch.nn.AvgPool1d`) |
| ALIAS-REFERENCE-ITEM(`torch.nn.modules.pooling.AvgPool2d`, `torch.nn.AvgPool2d`) |
| ALIAS-REFERENCE-ITEM(`torch.nn.modules.pooling.AvgPool3d`, `torch.nn.AvgPool3d`) |
| ALIAS-REFERENCE-ITEM(`torch.nn.modules.rnn.LSTM`, `torch.nn.LSTM`) |
| ALIAS-REFERENCE-ITEM(`torch.nn.modules.rnn.RNN`, `torch.nn.RNN`) |
| ALIAS-REFERENCE-ITEM(`torch.nn.modules.rnn.RNNBase`, `torch.nn.RNNBase`) |
| ALIAS-REFERENCE-ITEM(`torch.nn.modules.rnn.RNNCell`, `torch.nn.RNNCell`) |
| ALIAS-REFERENCE-ITEM(`torch.nn.modules.sparse.Embedding`, `torch.nn.Embedding`) |
| ALIAS-REFERENCE-ITEM(`torch.nn.parallel.DataParallel`, `torch.nn.DataParallel`) |
| ALIAS-REFERENCE-ITEM(`torch.nn.parallel.data_parallel.DataParallel`, `torch.nn.DataParallel`) |
| ALIAS-REFERENCE-ITEM(`torch.nn.parallel.distributed.DistributedDataParallel`, `torch.nn.parallel.DistributedDataParallel`) |
| ALIAS-REFERENCE-ITEM(`torch.nn.utils.clip_grad_norm`, `torch.nn.utils.clip_grad_norm_`) |
| ALIAS-REFERENCE-ITEM(`torch.optim.sgd.SGD`, `torch.optim.SGD`) |
| ALIAS-REFERENCE-ITEM(`torch.orgqr`, `torch.linalg.householder_product`) |
| ALIAS-REFERENCE-ITEM(`torch.pairwise_distance`, `torch.nn.functional.pairwise_distance`) |
| ALIAS-REFERENCE-ITEM(`torch.pdist`, `torch.nn.functional.pdist`) |
| ALIAS-REFERENCE-ITEM(`torch.pixel_shuffle`, `torch.nn.functional.pixel_shuffle`) |
| ALIAS-REFERENCE-ITEM(`torch.pixel_unshuffle`, `torch.nn.functional.pixel_unshuffle`) |
| ALIAS-REFERENCE-ITEM(`torch.prelu`, `torch.nn.functional.prelu`) |
| ALIAS-REFERENCE-ITEM(`torch.relu_`, `torch.nn.functional.relu_`) |
| ALIAS-REFERENCE-ITEM(`torch.rrelu_`, `torch.nn.functional.rrelu_`) |
| ALIAS-REFERENCE-ITEM(`torch.tanh`, `torch.nn.functional.tanh`) |
| ALIAS-REFERENCE-ITEM(`torch.threshold`, `torch.nn.functional.threshold`) |
| ALIAS-REFERENCE-ITEM(`torch.torch.Tensor`, `torch.Tensor`) |
| ALIAS-REFERENCE-ITEM(`torch.torch.finfo`, `torch.finfo`) |
| ALIAS-REFERENCE-ITEM(`torch.trapz`, `torch.trapezoid`) |
| ALIAS-REFERENCE-ITEM(`torch.utils.data.dataloader.DataLoader`, `torch.utils.data.DataLoader`) |
| ALIAS-REFERENCE-ITEM(`torch.utils.data.dataset.ConcatDataset`, `torch.utils.data.ConcatDataset`) |
| ALIAS-REFERENCE-ITEM(`torch.utils.data.dataset.Dataset`, `torch.utils.data.Dataset`) |
| ALIAS-REFERENCE-ITEM(`torch.utils.data.sampler.BatchSampler`, `torch.utils.data.BatchSampler`) |
| ALIAS-REFERENCE-ITEM(`torch.utils.data.sampler.RandomSampler`, `torch.utils.data.RandomSampler`) |
| ALIAS-REFERENCE-ITEM(`torch.utils.data.sampler.Sampler`, `torch.utils.data.Sampler`) |
| ALIAS-REFERENCE-ITEM(`torch.utils.data.sampler.SequentialSampler`, `torch.utils.data.SequentialSampler`) |
| ALIAS-REFERENCE-ITEM(`torch.utils.data.sampler.SubsetRandomSampler`, `torch.utils.data.SubsetRandomSampler`) |
| ALIAS-REFERENCE-ITEM(`torch.utils.data.sampler.WeightedRandomSampler`, `torch.utils.data.WeightedRandomSampler`) |

 ## 尚未实现的 API 列表

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ----- | ----------- | ----------------- | ----------- | ------- |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.rename`, https://pytorch.org/docs/stable/named_tensor.html#torch.Tensor.rename, 原型 API) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.rnn.pad_sequence`, https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_sequence.html#torch-nn-utils-rnn-pad-sequence, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.compile`, https://pytorch.org/docs/stable/generated/torch.compile.html#torch-compile, 实现路径不同) |
| NOT-IMPLEMENTED-ITEM(`torch.jit.freeze`, https://pytorch.org/docs/stable/generated/torch.jit.freeze.html#torch-jit-freeze, 实现路径不同) |
| NOT-IMPLEMENTED-ITEM(`torch.export.export`, https://pytorch.org/docs/stable/export.html#torch.export.export, 实现路径不同) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.dequantize`, https://pytorch.org/docs/stable/generated/torch.Tensor.dequantize.html#torch-tensor-dequantize, 实现路径不同) |
| NOT-IMPLEMENTED-ITEM(`torch.xpu.synchronize`, https://pytorch.org/docs/stable/generated/torch.xpu.synchronize.html#torch-xpu-synchronize, 实现路径不同) |
| NOT-IMPLEMENTED-ITEM(`torch.vmap`, https://pytorch.org/docs/stable/generated/torch.vmap.html#torch-vmap, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.resize_`, https://pytorch.org/docs/stable/generated/torch.Tensor.resize_.html#torch-tensor-resize, 废弃 API) |
| NOT-IMPLEMENTED-ITEM(`torch.fx.symbolic_trace`, https://pytorch.org/docs/stable/fx.html#torch.fx.symbolic_trace, 实现路径不同) |
| NOT-IMPLEMENTED-ITEM(`torch.jit.annotate`, https://pytorch.org/docs/stable/generated/torch.jit.annotate.html#torch-jit-annotate, 实现路径不同) |
| NOT-IMPLEMENTED-ITEM(`torch.quantize_per_tensor`, https://pytorch.org/docs/stable/generated/torch.quantize_per_tensor.html#torch-quantize-per-tensor, 实现路径不同) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.to_mkldnn`, https://pytorch.org/docs/stable/generated/torch.Tensor.to_mkldnn.html#torch-tensor-to-mkldnn, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.rnn.pack_padded_sequence`, https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html#torch-nn-utils-rnn-pack-padded-sequence, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.set_`, https://pytorch.org/docs/stable/generated/torch.Tensor.set_.html#torch-tensor-set, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.rnn.pad_packed_sequence`, https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_packed_sequence.html#torch-nn-utils-rnn-pad-packed-sequence, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.utils.model_zoo.load_url`, https://pytorch.org/docs/stable/model_zoo.html#torch.utils.model_zoo.load_url, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.record_stream`, https://pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html#torch-tensor-record-stream, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.xpu.empty_cache`, https://pytorch.org/docs/stable/generated/torch.xpu.empty_cache.html#torch-xpu-empty-cache, 实现路径不同) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.reset_peak_memory_stats`, https://pytorch.org/docs/stable/generated/torch.cuda.reset_peak_memory_stats.html#torch-cuda-reset-peak-memory-stats, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.library.impl`, https://pytorch.org/docs/stable/library.html#torch.library.impl, 实现路径不同) |
| NOT-IMPLEMENTED-ITEM(`torch.BFloat16Storage`, https://pytorch.org/docs/stable/storage.html#torch.BFloat16Storage, 废弃 API) |
| NOT-IMPLEMENTED-ITEM(`torch.BoolStorage`, https://pytorch.org/docs/stable/storage.html#torch.BoolStorage, 废弃 API) |
| NOT-IMPLEMENTED-ITEM(`torch.ByteStorage`, https://pytorch.org/docs/stable/storage.html#torch.ByteStorage, 废弃 API) |
| NOT-IMPLEMENTED-ITEM(`torch.CharStorage`, https://pytorch.org/docs/stable/storage.html#torch.CharStorage, 废弃 API) |
| NOT-IMPLEMENTED-ITEM(`torch.ComplexDoubleStorage`, https://pytorch.org/docs/stable/storage.html#torch.ComplexDoubleStorage, 废弃 API) |
| NOT-IMPLEMENTED-ITEM(`torch.ComplexFloatStorage`, https://pytorch.org/docs/stable/storage.html#torch.ComplexFloatStorage, 废弃 API) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.reduce_op`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.reduce_op, 废弃 API) |
| NOT-IMPLEMENTED-ITEM(`torch.DoubleStorage`, https://pytorch.org/docs/stable/storage.html#torch.DoubleStorage, 废弃 API) |
| NOT-IMPLEMENTED-ITEM(`torch.FloatStorage`, https://pytorch.org/docs/stable/storage.html#torch.FloatStorage, 废弃 API) |
| NOT-IMPLEMENTED-ITEM(`torch.HalfStorage`, https://pytorch.org/docs/stable/storage.html#torch.HalfStorage, 废弃 API) |
| NOT-IMPLEMENTED-ITEM(`torch.IntStorage`, https://pytorch.org/docs/stable/storage.html#torch.IntStorage, 废弃 API) |
| NOT-IMPLEMENTED-ITEM(`torch.LongStorage`, https://pytorch.org/docs/stable/storage.html#torch.LongStorage, 废弃 API) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.stateless.functional_call`, https://pytorch.org/docs/stable/generated/torch.nn.utils.stateless.functional_call.html#torch-nn-utils-stateless-functional-call, 废弃 API) |
| NOT-IMPLEMENTED-ITEM(`torch.QInt32Storage`, https://pytorch.org/docs/stable/storage.html#torch.QInt32Storage, 废弃 API) |
| NOT-IMPLEMENTED-ITEM(`torch.QInt8Storage`, https://pytorch.org/docs/stable/storage.html#torch.QInt8Storage, 废弃 API) |
| NOT-IMPLEMENTED-ITEM(`torch.QUInt2x4Storage`, https://pytorch.org/docs/stable/storage.html#torch.QUInt2x4Storage, 废弃 API) |
| NOT-IMPLEMENTED-ITEM(`torch.QUInt4x2Storage`, https://pytorch.org/docs/stable/storage.html#torch.QUInt4x2Storage, 废弃 API) |
| NOT-IMPLEMENTED-ITEM(`torch.QUInt8Storage`, https://pytorch.org/docs/stable/storage.html#torch.QUInt8Storage, 废弃 API) |
| NOT-IMPLEMENTED-ITEM(`torch.ShortStorage`, https://pytorch.org/docs/stable/storage.html#torch.ShortStorage, 废弃 API) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.storage`, https://pytorch.org/docs/stable/generated/torch.Tensor.storage.html#torch-tensor-storage, 废弃 API) |
| NOT-IMPLEMENTED-ITEM(`torch.TypedStorage`, https://pytorch.org/docs/stable/storage.html#torch.TypedStorage, 废弃 API) |
| NOT-IMPLEMENTED-ITEM(`torch.use_deterministic_algorithms`, https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch-use-deterministic-algorithms, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.parametrize.register_parametrization`, https://pytorch.org/docs/stable/generated/torch.nn.utils.parametrize.register_parametrization.html#torch-nn-utils-parametrize-register-parametrization, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.package.PackageImporter`, https://pytorch.org/docs/stable/package.html#torch.package.PackageImporter, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.EmbeddingBag`, https://pytorch.org/docs/stable/generated/torch.nn.EmbeddingBag.html#torch.nn.EmbeddingBag, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.fx.GraphModule`, https://pytorch.org/docs/stable/fx.html#torch.fx.GraphModule, 实现路径不同) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.share_memory_`, https://pytorch.org/docs/stable/generated/torch.Tensor.share_memory_.html#torch-tensor-share-memory, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.reset_max_memory_allocated`, https://pytorch.org/docs/stable/generated/torch.cuda.reset_max_memory_allocated.html#torch-cuda-reset-max-memory-allocated, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.parametrize.remove_parametrizations`, https://pytorch.org/docs/stable/generated/torch.nn.utils.parametrize.remove_parametrizations.html#torch-nn-utils-parametrize-remove-parametrizations, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.is_shared`, https://pytorch.org/docs/stable/generated/torch.Tensor.is_shared.html#torch-tensor-is-shared, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.storage_offset`, https://pytorch.org/docs/stable/generated/torch.Tensor.storage_offset.html#torch-tensor-storage-offset, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.library.Library`, https://pytorch.org/docs/stable/library.html#torch.library.Library, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.futures.Future`, https://pytorch.org/docs/stable/futures.html#torch.futures.Future, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.jit.Attribute`, https://pytorch.org/docs/stable/generated/torch.jit.Attribute.html#torch.jit.Attribute, 实现路径不同) |
| NOT-IMPLEMENTED-ITEM(`torch.quantize_per_channel`, https://pytorch.org/docs/stable/generated/torch.quantize_per_channel.html#torch-quantize-per-channel, 实现路径不同) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.untyped_storage`, https://pytorch.org/docs/stable/generated/torch.Tensor.untyped_storage.html#torch-tensor-untyped-storage, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.as_subclass`, https://pytorch.org/docs/stable/generated/torch.Tensor.as_subclass.html#torch-tensor-as-subclass, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.q_scale`, https://pytorch.org/docs/stable/generated/torch.Tensor.q_scale.html#torch-tensor-q-scale, 实现路径不同) |
| NOT-IMPLEMENTED-ITEM(`torch.set_float32_matmul_precision`, https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch-set-float32-matmul-precision, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.sparse_dim`, https://pytorch.org/docs/stable/generated/torch.Tensor.sparse_dim.html#torch-tensor-sparse-dim, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.q_zero_point`, https://pytorch.org/docs/stable/generated/torch.Tensor.q_zero_point.html#torch-tensor-q-zero-point, 实现路径不同) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.memory_stats`, https://pytorch.org/docs/stable/generated/torch.cuda.memory_stats.html#torch-cuda-memory-stats, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.pipeline.sync.Pipe`, https://pytorch.org/docs/2.3/pipeline.html#torch.distributed.pipeline.sync.Pipe, 废弃 API) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.set_rng_state`, https://pytorch.org/docs/stable/generated/torch.cuda.set_rng_state.html#torch-cuda-set-rng-state, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.lu_solve`, https://pytorch.org/docs/stable/generated/torch.Tensor.lu_solve.html#torch-tensor-lu-solve, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.linalg.tensorinv`, https://pytorch.org/docs/stable/generated/torch.linalg.tensorinv.html#torch-linalg-tensorinv, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.fsdp.FullStateDictConfig`, https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullStateDictConfig, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.mem_get_info`, https://pytorch.org/docs/stable/generated/torch.cuda.mem_get_info.html#torch-cuda-mem-get-info, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.CUDAGraph`, https://pytorch.org/docs/stable/generated/torch.cuda.CUDAGraph.html#torch.cuda.CUDAGraph, 原型 API) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.remove_spectral_norm`, https://pytorch.org/docs/stable/generated/torch.nn.utils.remove_spectral_norm.html#torch-nn-utils-remove-spectral-norm, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.utils.benchmark.Timer`, https://pytorch.org/docs/stable/benchmark_utils.html#torch.utils.benchmark.Timer, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.utils.mobile_optimizer.optimize_for_mobile`, https://pytorch.org/docs/stable/mobile_optimizer.html#torch.utils.mobile_optimizer.optimize_for_mobile, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.fsdp.MixedPrecision`, https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.MixedPrecision, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.rnn.PackedSequence`, https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.PackedSequence.html#torch.nn.utils.rnn.PackedSequence, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.dense_dim`, https://pytorch.org/docs/stable/generated/torch.Tensor.dense_dim.html#torch-tensor-dense-dim, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.qscheme`, https://pytorch.org/docs/stable/generated/torch.Tensor.qscheme.html#torch-tensor-qscheme, 实现路径不同) |
| NOT-IMPLEMENTED-ITEM(`torch.fx.wrap`, https://pytorch.org/docs/stable/fx.html#torch.fx.wrap, 实现路径不同) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.set_detect_anomaly`, https://pytorch.org/docs/stable/autograd.html#torch.autograd.set_detect_anomaly, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.empty_strided`, https://pytorch.org/docs/stable/generated/torch.empty_strided.html#torch-empty-strided, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.fx.Graph`, https://pytorch.org/docs/stable/fx.html#torch.fx.Graph, 实现路径不同) |
| NOT-IMPLEMENTED-ITEM(`torch.futures.wait_all`, https://pytorch.org/docs/stable/futures.html#torch.futures.wait_all, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.prune.l1_unstructured`, https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.l1_unstructured.html#torch-nn-utils-prune-l1-unstructured, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.ipc_collect`, https://pytorch.org/docs/stable/generated/torch.cuda.ipc_collect.html#torch-cuda-ipc-collect, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.optim.ZeroRedundancyOptimizer`, https://pytorch.org/docs/stable/distributed.optim.html#torch.distributed.optim.ZeroRedundancyOptimizer, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.mps.profiler.start`, https://pytorch.org/docs/stable/generated/torch.mps.profiler.start.html#torch-mps-profiler-start, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.fx.Proxy`, https://pytorch.org/docs/stable/fx.html#torch.fx.Proxy, 实现路径不同) |
| NOT-IMPLEMENTED-ITEM(`torch.mps.profiler.stop`, https://pytorch.org/docs/stable/generated/torch.mps.profiler.stop.html#torch-mps-profiler-stop, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.refine_names`, https://pytorch.org/docs/stable/named_tensor.html#torch.Tensor.refine_names, 原型 API) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.init`, https://pytorch.org/docs/stable/generated/torch.cuda.init.html#torch-cuda-init, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.rpc.TensorPipeRpcBackendOptions`, https://pytorch.org/docs/stable/rpc.html#torch.distributed.rpc.TensorPipeRpcBackendOptions, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.default_stream`, https://pytorch.org/docs/stable/generated/torch.cuda.default_stream.html#torch-cuda-default-stream, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.resolve_conj`, https://pytorch.org/docs/stable/generated/torch.Tensor.resolve_conj.html#torch-tensor-resolve-conj, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.mps.synchronize`, https://pytorch.org/docs/stable/generated/torch.mps.synchronize.html#torch-mps-synchronize, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.skip_init`, https://pytorch.org/docs/stable/generated/torch.nn.utils.skip_init.html#torch-nn-utils-skip-init, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.row_indices`, https://pytorch.org/docs/stable/generated/torch.Tensor.row_indices.html#torch-tensor-row-indices, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.jit.trace_module`, https://pytorch.org/docs/stable/generated/torch.jit.trace_module.html#torch-jit-trace-module, 实现路径不同) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.fsdp.CPUOffload`, https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.CPUOffload, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.quasirandom.SobolEngine`, https://pytorch.org/docs/stable/generated/torch.quasirandom.SobolEngine.html#torch.quasirandom.SobolEngine, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.names`, https://pytorch.org/docs/stable/named_tensor.html#torch.Tensor.names, 原型 API) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.q_per_channel_zero_points`, https://pytorch.org/docs/stable/generated/torch.Tensor.q_per_channel_zero_points.html#torch-tensor-q-per-channel-zero-points, 实现路径不同) |
| NOT-IMPLEMENTED-ITEM(`torch.jit.export`, https://pytorch.org/docs/stable/jit.html#torch.jit.export, 实现路径不同) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.prune.remove`, https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.remove.html#torch-nn-utils-prune-remove, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.rnn.pack_sequence`, https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_sequence.html#torch-nn-utils-rnn-pack-sequence, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.ccol_indices`, https://pytorch.org/docs/stable/generated/torch.Tensor.ccol_indices.html#torch-tensor-ccol-indices, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.is_set_to`, https://pytorch.org/docs/stable/generated/torch.Tensor.is_set_to.html#torch-tensor-is-set-to, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.put_`, https://pytorch.org/docs/stable/generated/torch.Tensor.put_.html#torch-tensor-put, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.q_per_channel_axis`, https://pytorch.org/docs/stable/generated/torch.Tensor.q_per_channel_axis.html#torch-tensor-q-per-channel-axis, 实现路径不同) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.q_per_channel_scales`, https://pytorch.org/docs/stable/generated/torch.Tensor.q_per_channel_scales.html#torch-tensor-q-per-channel-scales, 实现路径不同) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.sign_`, https://pytorch.org/docs/stable/generated/torch.Tensor.sign_.html#torch-tensor-sign, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.hub.get_dir`, https://pytorch.org/docs/stable/hub.html#torch.hub.get_dir, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.hub.set_dir`, https://pytorch.org/docs/stable/hub.html#torch.hub.set_dir, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.is_conj`, https://pytorch.org/docs/stable/generated/torch.Tensor.is_conj.html#torch-tensor-is-conj, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.result_type`, https://pytorch.org/docs/stable/generated/torch.result_type.html#torch-result-type, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.comm.broadcast_coalesced`, https://pytorch.org/docs/stable/generated/torch.cuda.comm.broadcast_coalesced.html#torch-cuda-comm-broadcast-coalesced, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.optim.SparseAdam`, https://pytorch.org/docs/stable/generated/torch.optim.SparseAdam.html#torch.optim.SparseAdam, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.fake_quantize_per_channel_affine`, https://pytorch.org/docs/stable/generated/torch.fake_quantize_per_channel_affine.html#torch-fake-quantize-per-channel-affine, 实现路径不同) |
| NOT-IMPLEMENTED-ITEM(`torch.fake_quantize_per_tensor_affine`, https://pytorch.org/docs/stable/generated/torch.fake_quantize_per_tensor_affine.html#torch-fake-quantize-per-tensor-affine, 实现路径不同) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.to_sparse_csc`, https://pytorch.org/docs/stable/generated/torch.Tensor.to_sparse_csc.html#torch-tensor-to-sparse-csc, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.mps.empty_cache`, https://pytorch.org/docs/stable/generated/torch.mps.empty_cache.html#torch-mps-empty-cache, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.profiler.record_function`, https://pytorch.org/docs/stable/generated/torch.autograd.profiler.record_function.html#torch.autograd.profiler.record_function, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.index_copy`, https://pytorch.org/docs/stable/generated/torch.Tensor.index_copy.html#torch-tensor-index-copy, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.utils.cpp_extension.load_inline`, https://pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.load_inline, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.jit.set_fusion_strategy`, https://pytorch.org/docs/stable/generated/torch.jit.set_fusion_strategy.html#torch-jit-set-fusion-strategy, 实现路径不同) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.TCPStore`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.TCPStore, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.optim.lr_scheduler.SequentialLR`, https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.SequentialLR.html#torch.optim.lr_scheduler.SequentialLR, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.sparse.sampled_addmm`, https://pytorch.org/docs/stable/generated/torch.sparse.sampled_addmm.html#torch-sparse-sampled-addmm, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.linalg.lu_solve`, https://pytorch.org/docs/stable/generated/torch.linalg.lu_solve.html#torch-linalg-lu-solve, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.nested.nested_tensor`, https://pytorch.org/docs/stable/nested.html#torch.nested.nested_tensor, 原型 API) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.align_to`, https://pytorch.org/docs/stable/named_tensor.html#torch.Tensor.align_to, 原型 API) |
| NOT-IMPLEMENTED-ITEM(`torch.promote_types`, https://pytorch.org/docs/stable/generated/torch.promote_types.html#torch-promote-types, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.tensor.parallel.ColwiseParallel`, https://pytorch.org/docs/stable/distributed.tensor.parallel.html#torch.distributed.tensor.parallel.ColwiseParallel, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.to_sparse_bsr`, https://pytorch.org/docs/stable/generated/torch.Tensor.to_sparse_bsr.html#torch-tensor-to-sparse-bsr, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.xpu.device_count`, https://pytorch.org/docs/stable/generated/torch.xpu.device_count.html#torch-xpu-device-count, 实现路径不同) |
| NOT-IMPLEMENTED-ITEM(`torch.fx.Node`, https://pytorch.org/docs/stable/fx.html#torch.fx.Node, 实现路径不同) |
| NOT-IMPLEMENTED-ITEM(`torch.jit.fork`, https://pytorch.org/docs/stable/generated/torch.jit.fork.html#torch-jit-fork, 实现路径不同) |
| NOT-IMPLEMENTED-ITEM(`torch.library.impl_abstract`, https://pytorch.org/docs/stable/library.html#torch.library.impl_abstract, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.linalg.tensorsolve`, https://pytorch.org/docs/stable/generated/torch.linalg.tensorsolve.html#torch-linalg-tensorsolve, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.functional.embedding_bag`, https://pytorch.org/docs/stable/generated/torch.nn.functional.embedding_bag.html#torch-nn-functional-embedding-bag, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.map_`, https://pytorch.org/docs/stable/generated/torch.Tensor.map_.html#torch-tensor-map, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.rename_`, https://pytorch.org/docs/stable/named_tensor.html#torch.Tensor.rename_, 原型 API) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.scatter_reduce_`, https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_reduce_.html#torch-tensor-scatter-reduce, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.set_flush_denormal`, https://pytorch.org/docs/stable/generated/torch.set_flush_denormal.html#torch-set-flush-denormal, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.kaiser_window`, https://pytorch.org/docs/stable/generated/torch.kaiser_window.html#torch-kaiser-window, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.device_mesh.init_device_mesh`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.device_mesh.init_device_mesh, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.fsdp.FullyShardedDataParallel`, https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.tensor.parallel.parallelize_module`, https://pytorch.org/docs/stable/distributed.tensor.parallel.html#torch.distributed.tensor.parallel.parallelize_module, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.tensor.parallel.RowwiseParallel`, https://pytorch.org/docs/stable/distributed.tensor.parallel.html#torch.distributed.tensor.parallel.RowwiseParallel, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.are_deterministic_algorithms_enabled`, https://pytorch.org/docs/stable/generated/torch.are_deterministic_algorithms_enabled.html#torch-are-deterministic-algorithms-enabled, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.is_deterministic_algorithms_warn_only_enabled`, https://pytorch.org/docs/stable/generated/torch.is_deterministic_algorithms_warn_only_enabled.html#torch-is-deterministic-algorithms-warn-only-enabled, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.backends.mps.is_available`, https://pytorch.org/docs/stable/backends.html#torch.backends.mps.is_available, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.fx.Tracer`, https://pytorch.org/docs/stable/fx.html#torch.fx.Tracer, 实现路径不同) |
| NOT-IMPLEMENTED-ITEM(`torch.jit.enable_onednn_fusion`, https://pytorch.org/docs/stable/generated/torch.jit.enable_onednn_fusion.html#torch-jit-enable-onednn-fusion, 实现路径不同) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.comm.reduce_add`, https://pytorch.org/docs/stable/generated/torch.cuda.comm.reduce_add.html#torch-cuda-comm-reduce-add, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.checkpoint.state_dict.get_optimizer_state_dict`, https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.get_optimizer_state_dict, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.parametrizations.orthogonal`, https://pytorch.org/docs/stable/generated/torch.nn.utils.parametrizations.orthogonal.html#torch-nn-utils-parametrizations-orthogonal, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.prune.L1Unstructured`, https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.L1Unstructured.html#torch.nn.utils.prune.L1Unstructured, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.prune.random_unstructured`, https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.random_unstructured.html#torch-nn-utils-prune-random-unstructured, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.special.zeta`, https://pytorch.org/docs/stable/special.html#torch.special.zeta, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.xpu.current_device`, https://pytorch.org/docs/stable/generated/torch.xpu.current_device.html#torch-xpu-current-device, 实现路径不同) |
| NOT-IMPLEMENTED-ITEM(`torch.xpu.get_device_properties`, https://pytorch.org/docs/stable/generated/torch.xpu.get_device_properties.html#torch-xpu-get-device-properties, 实现路径不同) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.functional.multilabel_margin_loss`, https://pytorch.org/docs/stable/generated/torch.nn.functional.multilabel_margin_loss.html#torch-nn-functional-multilabel-margin-loss, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.gradient`, https://pytorch.org/docs/stable/generated/torch.gradient.html#torch-gradient, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.sparse_resize_`, https://pytorch.org/docs/stable/generated/torch.Tensor.sparse_resize_.html#torch-tensor-sparse-resize, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.profiler.profile`, https://pytorch.org/docs/stable/autograd.html#torch.autograd.profiler.profile, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.backends.cuda.enable_math_sdp`, https://pytorch.org/docs/stable/backends.html#torch.backends.cuda.enable_math_sdp, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.backends.cuda.enable_mem_efficient_sdp`, https://pytorch.org/docs/stable/backends.html#torch.backends.cuda.enable_mem_efficient_sdp, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.jit.ScriptModule`, https://pytorch.org/docs/stable/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule, 实现路径不同) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.ExternalStream`, https://pytorch.org/docs/stable/generated/torch.cuda.ExternalStream.html#torch.cuda.ExternalStream, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.memory._record_memory_history`, https://pytorch.org/docs/stable/torch_cuda_memory.html#torch.cuda.memory._record_memory_history, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.memory_summary`, https://pytorch.org/docs/stable/generated/torch.cuda.memory_summary.html#torch-cuda-memory-summary, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.checkpoint.state_dict.get_model_state_dict`, https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.get_model_state_dict, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.checkpoint.state_dict.StateDictOptions`, https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.StateDictOptions, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.optim.lr_scheduler.ChainedScheduler`, https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ChainedScheduler.html#torch.optim.lr_scheduler.ChainedScheduler, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.futures.collect_all`, https://pytorch.org/docs/stable/futures.html#torch.futures.collect_all, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.sparse_compressed_tensor`, https://pytorch.org/docs/stable/generated/torch.sparse_compressed_tensor.html#torch-sparse-compressed-tensor, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.linalg.vecdot`, https://pytorch.org/docs/stable/generated/torch.linalg.vecdot.html#torch-linalg-vecdot, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.mps.current_allocated_memory`, https://pytorch.org/docs/stable/generated/torch.mps.current_allocated_memory.html#torch-mps-current-allocated-memory, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.profiler.tensorboard_trace_handler`, https://pytorch.org/docs/stable/profiler.html#torch.profiler.tensorboard_trace_handler, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.xpu.is_available`, https://pytorch.org/docs/stable/generated/torch.xpu.is_available.html#torch-xpu-is-available, 实现路径不同) |
| NOT-IMPLEMENTED-ITEM(`torch.xpu.set_device`, https://pytorch.org/docs/stable/generated/torch.xpu.set_device.html#torch-xpu-set-device, 实现路径不同) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.rpc.WorkerInfo`, https://pytorch.org/docs/stable/rpc.html#torch.distributed.rpc.WorkerInfo, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.bartlett_window`, https://pytorch.org/docs/stable/generated/torch.bartlett_window.html#torch-bartlett-window, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.signal.windows.kaiser`, https://pytorch.org/docs/stable/generated/torch.signal.windows.kaiser.html#torch-signal-windows-kaiser, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.graph_pool_handle`, https://pytorch.org/docs/stable/generated/torch.cuda.graph_pool_handle.html#torch-cuda-graph-pool-handle, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.library.define`, https://pytorch.org/docs/stable/library.html#torch.library.define, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.monitor.log_event`, https://pytorch.org/docs/stable/monitor.html#torch.monitor.log_event, 原型 API) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.init.sparse_`, https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.sparse_, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.modules.module.register_module_backward_hook`, https://pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_backward_hook.html#torch-nn-modules-module-register-module-backward-hook, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.prune.global_unstructured`, https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.global_unstructured.html#torch-nn-utils-prune-global-unstructured, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.prune.ln_structured`, https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.ln_structured.html#torch-nn-utils-prune-ln-structured, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.special.log_ndtr`, https://pytorch.org/docs/stable/special.html#torch.special.log_ndtr, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.align_as`, https://pytorch.org/docs/stable/named_tensor.html#torch.Tensor.align_as, 原型 API) |
| NOT-IMPLEMENTED-ITEM(`torch.xpu.get_device_name`, https://pytorch.org/docs/stable/generated/torch.xpu.get_device_name.html#torch-xpu-get-device-name, 实现路径不同) |
| NOT-IMPLEMENTED-ITEM(`torch.xpu.manual_seed`, https://pytorch.org/docs/stable/generated/torch.xpu.manual_seed.html#torch-xpu-manual-seed, 实现路径不同) |
| NOT-IMPLEMENTED-ITEM(`torch.distributions.transforms.CatTransform`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.transforms.CatTransform, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.set_warn_always`, https://pytorch.org/docs/stable/generated/torch.set_warn_always.html#torch-set-warn-always, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.backends.cuda.enable_flash_sdp`, https://pytorch.org/docs/stable/backends.html#torch.backends.cuda.enable_flash_sdp, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.backends.cuda.preferred_linalg_library`, https://pytorch.org/docs/stable/backends.html#torch.backends.cuda.preferred_linalg_library, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.UntypedStorage`, https://pytorch.org/docs/stable/storage.html#torch.UntypedStorage, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.fx.Interpreter`, https://pytorch.org/docs/stable/fx.html#module-torch.fx.interpreter, 实现路径不同) |
| NOT-IMPLEMENTED-ITEM(`torch.jit.optimize_for_inference`, https://pytorch.org/docs/stable/generated/torch.jit.optimize_for_inference.html#torch-jit-optimize-for-inference, 实现路径不同) |
| NOT-IMPLEMENTED-ITEM(`torch.jit.wait`, https://pytorch.org/docs/stable/generated/torch.jit.wait.html#torch-jit-wait, 实现路径不同) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.autograd.backward`, https://pytorch.org/docs/stable/rpc.html#torch.distributed.autograd.backward, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributions.transforms.LowerCholeskyTransform`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.transforms.LowerCholeskyTransform, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.overrides.resolve_name`, https://pytorch.org/docs/stable/torch.overrides.html#torch.overrides.resolve_name, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.sparse.log_softmax`, https://pytorch.org/docs/stable/generated/torch.sparse.log_softmax.html#torch-sparse-log-softmax, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.monitor.register_event_handler`, https://pytorch.org/docs/stable/monitor.html#torch.monitor.register_event_handler, 原型 API) |
| NOT-IMPLEMENTED-ITEM(`torch.monitor.Stat`, https://pytorch.org/docs/stable/monitor.html#torch.monitor.Stat, 原型 API) |
| NOT-IMPLEMENTED-ITEM(`torch.monitor.unregister_event_handler`, https://pytorch.org/docs/stable/monitor.html#torch.monitor.unregister_event_handler, 原型 API) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.register_post_accumulate_grad_hook`, https://pytorch.org/docs/stable/generated/torch.Tensor.register_post_accumulate_grad_hook.html#torch-tensor-register-post-accumulate-grad-hook, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.sspaddmm`, https://pytorch.org/docs/stable/generated/torch.Tensor.sspaddmm.html#torch-tensor-sspaddmm, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.sum_to_size`, https://pytorch.org/docs/stable/generated/torch.Tensor.sum_to_size.html#torch-tensor-sum-to-size, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.__config__.parallel_info`, https://pytorch.org/docs/stable/config_mod.html#torch.__config__.parallel_info, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.amp.custom_bwd`, https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.custom_bwd, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.amp.custom_fwd`, https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.custom_fwd, 废弃 API) |
| NOT-IMPLEMENTED-ITEM(`torch.__config__.show`, https://pytorch.org/docs/stable/config_mod.html#torch.__config__.show, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.from_file`, https://pytorch.org/docs/stable/generated/torch.from_file.html#torch-from-file, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.__future__.set_overwrite_module_params_on_conversion`, https://pytorch.org/docs/stable/future_mod.html#torch.__future__.set_overwrite_module_params_on_conversion, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.gradcheck.gradcheck`, https://pytorch.org/docs/stable/generated/torch.autograd.gradcheck.gradcheck.html#torch-autograd-gradcheck-gradcheck, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.backends.cuda.sdp_kernel`, https://pytorch.org/docs/stable/backends.html#torch.backends.cuda.sdp_kernel, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.backends.mkl.is_available`, https://pytorch.org/docs/stable/backends.html#torch.backends.mkl.is_available, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.signal.windows.bartlett`, https://pytorch.org/docs/stable/generated/torch.signal.windows.bartlett.html#torch-signal-windows-bartlett, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.storage_type`, https://pytorch.org/docs/stable/generated/torch.Tensor.storage_type.html#torch-tensor-storage-type, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.can_device_access_peer`, https://pytorch.org/docs/stable/generated/torch.cuda.can_device_access_peer.html#torch-cuda-can-device-access-peer, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.jiterator._create_jit_fn`, https://pytorch.org/docs/stable/generated/torch.cuda.jiterator._create_jit_fn.html#torch-cuda-jiterator-create-jit-fn, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.reset_max_memory_cached`, https://pytorch.org/docs/stable/generated/torch.cuda.reset_max_memory_cached.html#torch-cuda-reset-max-memory-cached, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.set_sync_debug_mode`, https://pytorch.org/docs/stable/generated/torch.cuda.set_sync_debug_mode.html#torch-cuda-set-sync-debug-mode, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.fsdp.FullOptimStateDictConfig`, https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullOptimStateDictConfig, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributions.transforms.CorrCholeskyTransform`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.transforms.CorrCholeskyTransform, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.overrides.get_testing_overrides`, https://pytorch.org/docs/stable/torch.overrides.html#torch.overrides.get_testing_overrides, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.sparse_bsr_tensor`, https://pytorch.org/docs/stable/generated/torch.sparse_bsr_tensor.html#torch-sparse-bsr-tensor, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.sparse_csc_tensor`, https://pytorch.org/docs/stable/generated/torch.sparse_csc_tensor.html#torch-sparse-csc-tensor, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.to_sparse_bsc`, https://pytorch.org/docs/stable/generated/torch.Tensor.to_sparse_bsc.html#torch-tensor-to-sparse-bsc, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.linalg.ldl_factor_ex`, https://pytorch.org/docs/stable/generated/torch.linalg.ldl_factor_ex.html#torch-linalg-ldl-factor-ex, 原型 API) |
| NOT-IMPLEMENTED-ITEM(`torch.monitor.Event`, https://pytorch.org/docs/stable/monitor.html#torch.monitor.Event, 原型 API) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.rnn.unpad_sequence`, https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.unpad_sequence.html#torch-nn-utils-rnn-unpad-sequence, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.xpu.set_rng_state`, https://pytorch.org/docs/stable/generated/torch.xpu.set_rng_state.html#torch-xpu-set-rng-state, 实现路径不同) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.max_memory_cached`, https://pytorch.org/docs/stable/generated/torch.cuda.max_memory_cached.html#torch-cuda-max-memory-cached, 废弃 API) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.get_arch_list`, https://pytorch.org/docs/stable/generated/torch.cuda.get_arch_list.html#torch-cuda-get-arch-list, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.resolve_neg`, https://pytorch.org/docs/stable/generated/torch.Tensor.resolve_neg.html#torch-tensor-resolve-neg, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.compiled_with_cxx11_abi`, https://pytorch.org/docs/stable/generated/torch.compiled_with_cxx11_abi.html#torch-compiled-with-cxx11-abi, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.memory_cached`, https://pytorch.org/docs/stable/generated/torch.cuda.memory_cached.html#torch-cuda-memory-cached, 废弃 API) |
| NOT-IMPLEMENTED-ITEM(`torch.is_warn_always_enabled`, https://pytorch.org/docs/stable/generated/torch.is_warn_always_enabled.html#torch-is-warn-always-enabled, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.detect_anomaly`, https://pytorch.org/docs/stable/autograd.html#torch.autograd.detect_anomaly, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.forward_ad.make_dual`, https://pytorch.org/docs/stable/generated/torch.autograd.forward_ad.make_dual.html#torch-autograd-forward-ad-make-dual, 原型 API) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.nvtx.mark`, https://pytorch.org/docs/stable/generated/torch.cuda.nvtx.mark.html#torch-cuda-nvtx-mark, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.forward_ad.unpack_dual`, https://pytorch.org/docs/stable/generated/torch.autograd.forward_ad.unpack_dual.html#torch-autograd-forward-ad-unpack-dual, 原型 API) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.grad_mode.set_grad_enabled`, https://pytorch.org/docs/stable/generated/torch.autograd.grad_mode.set_grad_enabled.html#torch.autograd.grad_mode.set_grad_enabled, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.graph.save_on_cpu`, https://pytorch.org/docs/stable/autograd.html#torch.autograd.graph.save_on_cpu, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.profiler.load_nvprof`, https://pytorch.org/docs/stable/generated/torch.autograd.profiler.load_nvprof.html#torch-autograd-profiler-load-nvprof, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.profiler.profile.key_averages`, https://pytorch.org/docs/stable/generated/torch.autograd.profiler.profile.key_averages.html#torch-autograd-profiler-profile-key-averages, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.profiler_util.MemRecordsAcc`, https://pytorch.org/docs/stable/generated/torch.autograd.profiler_util.MemRecordsAcc.html#torch.autograd.profiler_util.MemRecordsAcc, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.backends.mps.is_built`, https://pytorch.org/docs/stable/backends.html#torch.backends.mps.is_built, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.backends.nnpack.set_flags`, https://pytorch.org/docs/stable/backends.html#torch.backends.nnpack.set_flags, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.export.ExportedProgram`, https://pytorch.org/docs/stable/export.html#torch.export.ExportedProgram, 实现路径不同) |
| NOT-IMPLEMENTED-ITEM(`torch.export.graph_signature.InputSpec`, https://pytorch.org/docs/stable/export.html#torch.export.graph_signature.InputSpec, 实现路径不同) |
| NOT-IMPLEMENTED-ITEM(`torch.export.load`, https://pytorch.org/docs/stable/export.html#torch.export.load, 实现路径不同) |
| NOT-IMPLEMENTED-ITEM(`torch.fx.replace_pattern`, https://pytorch.org/docs/stable/fx.html#torch.fx.replace_pattern, 实现路径不同) |
| NOT-IMPLEMENTED-ITEM(`torch.fx.Transformer`, https://pytorch.org/docs/stable/fx.html#torch.fx.Transformer, 实现路径不同) |
| NOT-IMPLEMENTED-ITEM(`torch.jit.isinstance`, https://pytorch.org/docs/stable/generated/torch.jit.isinstance.html#torch-jit-isinstance, 实现路径不同) |
| NOT-IMPLEMENTED-ITEM(`torch.jit.script_if_tracing`, https://pytorch.org/docs/stable/generated/torch.jit.script_if_tracing.html#torch-jit-script-if-tracing, 实现路径不同) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.caching_allocator_alloc`, https://pytorch.org/docs/stable/generated/torch.cuda.caching_allocator_alloc.html#torch-cuda-caching-allocator-alloc, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.caching_allocator_delete`, https://pytorch.org/docs/stable/generated/torch.cuda.caching_allocator_delete.html#torch-cuda-caching-allocator-delete, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.get_allocator_backend`, https://pytorch.org/docs/stable/generated/torch.cuda.get_allocator_backend.html#torch-cuda-get-allocator-backend, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.get_sync_debug_mode`, https://pytorch.org/docs/stable/generated/torch.cuda.get_sync_debug_mode.html#torch-cuda-get-sync-debug-mode, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.list_gpu_processes`, https://pytorch.org/docs/stable/generated/torch.cuda.list_gpu_processes.html#torch-cuda-list-gpu-processes, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.memory_snapshot`, https://pytorch.org/docs/stable/generated/torch.cuda.memory_snapshot.html#torch-cuda-memory-snapshot, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.seed`, https://pytorch.org/docs/stable/generated/torch.cuda.seed.html#torch-cuda-seed, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.seed_all`, https://pytorch.org/docs/stable/generated/torch.cuda.seed_all.html#torch-cuda-seed-all, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.utilization`, https://pytorch.org/docs/stable/generated/torch.cuda.utilization.html#torch-cuda-utilization, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.PowerSGDState`, https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.PowerSGDState, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.checkpoint.planner.WriteItem`, https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.planner.WriteItem, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.checkpoint.state_dict.set_model_state_dict`, https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.set_model_state_dict, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.checkpoint.state_dict.set_optimizer_state_dict`, https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.set_optimizer_state_dict, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.FileStore`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.FileStore, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.PrefixStore`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.PrefixStore, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.fsdp.LocalStateDictConfig`, https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.LocalStateDictConfig, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.optim.lr_scheduler.PolynomialLR`, https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.PolynomialLR.html#torch.optim.lr_scheduler.PolynomialLR, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributions.relaxed_bernoulli.RelaxedBernoulli`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.relaxed_bernoulli.RelaxedBernoulli, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.overrides.get_overridable_functions`, https://pytorch.org/docs/stable/torch.overrides.html#torch.overrides.get_overridable_functions, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.overrides.has_torch_function`, https://pytorch.org/docs/stable/torch.overrides.html#torch.overrides.has_torch_function, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.overrides.is_tensor_like`, https://pytorch.org/docs/stable/torch.overrides.html#torch.overrides.is_tensor_like, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.overrides.wrap_torch_function`, https://pytorch.org/docs/stable/torch.overrides.html#torch.overrides.wrap_torch_function, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.sparse_bsc_tensor`, https://pytorch.org/docs/stable/generated/torch.sparse_bsc_tensor.html#torch-sparse-bsc-tensor, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.library.get_ctx`, https://pytorch.org/docs/stable/library.html#torch.library.get_ctx, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.linalg.ldl_factor`, https://pytorch.org/docs/stable/generated/torch.linalg.ldl_factor.html#torch-linalg-ldl-factor, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.linalg.ldl_solve`, https://pytorch.org/docs/stable/generated/torch.linalg.ldl_solve.html#torch-linalg-ldl-solve, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.lobpcg`, https://pytorch.org/docs/stable/generated/torch.lobpcg.html#torch-lobpcg, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.mps.manual_seed`, https://pytorch.org/docs/stable/generated/torch.mps.manual_seed.html#torch-mps-manual-seed, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.prune.identity`, https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.identity.html#torch-nn-utils-prune-identity, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.prune.PruningContainer`, https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.PruningContainer.html#torch.nn.utils.prune.PruningContainer, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.prune.random_structured`, https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.random_structured.html#torch-nn-utils-prune-random-structured, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.prune.RandomStructured`, https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.RandomStructured.html#torch.nn.utils.prune.RandomStructured, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.atan2_`, https://pytorch.org/docs/stable/generated/torch.Tensor.atan2_.html#torch-tensor-atan2, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.chalf`, https://pytorch.org/docs/stable/generated/torch.Tensor.chalf.html#torch-tensor-chalf, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.index_reduce`, https://pytorch.org/docs/stable/generated/torch.Tensor.index_reduce.html#torch-tensor-index-reduce, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.index_reduce_`, https://pytorch.org/docs/stable/generated/torch.Tensor.index_reduce_.html#torch-tensor-index-reduce, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.sgn_`, https://pytorch.org/docs/stable/generated/torch.Tensor.sgn_.html#torch-tensor-sgn, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.utils.cpp_extension.verify_ninja_availability`, https://pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.verify_ninja_availability, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.utils.data._utils.collate.collate`, https://pytorch.org/docs/stable/data.html#torch.utils.data._utils.collate.collate, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.utils.data.StackDataset`, https://pytorch.org/docs/stable/data.html#torch.utils.data.StackDataset, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.utils.swap_tensors`, https://pytorch.org/docs/stable/generated/torch.utils.swap_tensors.html#torch-utils-swap-tensors, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.xpu.get_rng_state`, https://pytorch.org/docs/stable/generated/torch.xpu.get_rng_state.html#torch-xpu-get-rng-state, 实现路径不同) |
| NOT-IMPLEMENTED-ITEM(`torch.xpu.get_rng_state_all`, https://pytorch.org/docs/stable/generated/torch.xpu.get_rng_state_all.html#torch-xpu-get-rng-state-all, 实现路径不同) |
| NOT-IMPLEMENTED-ITEM(`torch.xpu.manual_seed_all`, https://pytorch.org/docs/stable/generated/torch.xpu.manual_seed_all.html#torch-xpu-manual-seed-all, 实现路径不同) |
| NOT-IMPLEMENTED-ITEM(`torch.xpu.set_rng_state_all`, https://pytorch.org/docs/stable/generated/torch.xpu.set_rng_state_all.html#torch-xpu-set-rng-state-all, 实现路径不同) |
| NOT-IMPLEMENTED-ITEM(`torch.utils.cpp_extension.include_paths`, https://pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.include_paths, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.special.entr`, https://pytorch.org/docs/stable/special.html#torch.special.entr, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributions.transforms.CumulativeDistributionTransform`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.transforms.CumulativeDistributionTransform, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributions.transforms.SoftplusTransform`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.transforms.SoftplusTransform, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch._logging.set_logs`, https://pytorch.org/docs/stable/generated/torch._logging.set_logs.html#torch-logging-set-logs, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.cond`, https://pytorch.org/docs/stable/generated/torch.cond.html#torch-cond, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.get_float32_matmul_precision`, https://pytorch.org/docs/stable/generated/torch.get_float32_matmul_precision.html#torch-get-float32-matmul-precision, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.index_reduce`, https://pytorch.org/docs/stable/generated/torch.index_reduce.html#torch-index-reduce, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.is_inference_mode_enabled`, https://pytorch.org/docs/stable/generated/torch.is_inference_mode_enabled.html#torch-is-inference-mode-enabled, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.is_storage`, https://pytorch.org/docs/stable/generated/torch.is_storage.html#torch-is-storage, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.random.fork_rng`, https://pytorch.org/docs/stable/random.html#torch.random.fork_rng, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.Tag`, https://pytorch.org/docs/stable/torch.html#torch.Tag, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.unravel_index`, https://pytorch.org/docs/stable/generated/torch.unravel_index.html#torch-unravel-index, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.__future__.get_overwrite_module_params_on_conversion`, https://pytorch.org/docs/stable/future_mod.html#torch.__future__.get_overwrite_module_params_on_conversion, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.__future__.get_swap_module_params_on_conversion`, https://pytorch.org/docs/stable/future_mod.html#torch.__future__.get_swap_module_params_on_conversion, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.__future__.set_swap_module_params_on_conversion`, https://pytorch.org/docs/stable/future_mod.html#torch.__future__.set_swap_module_params_on_conversion, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.forward_ad.dual_level`, https://pytorch.org/docs/stable/generated/torch.autograd.forward_ad.dual_level.html#torch.autograd.forward_ad.dual_level, 原型 API) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.forward_ad.enter_dual_level`, https://pytorch.org/docs/stable/generated/torch.autograd.forward_ad.enter_dual_level.html#torch-autograd-forward-ad-enter-dual-level, 原型 API) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.forward_ad.exit_dual_level`, https://pytorch.org/docs/stable/generated/torch.autograd.forward_ad.exit_dual_level.html#torch-autograd-forward-ad-exit-dual-level, 原型 API) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.forward_ad.UnpackedDualTensor`, https://pytorch.org/docs/stable/generated/torch.autograd.forward_ad.UnpackedDualTensor.html#torch.autograd.forward_ad.UnpackedDualTensor, 原型 API) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.function.BackwardCFunction`, https://pytorch.org/docs/stable/generated/torch.autograd.function.BackwardCFunction.html#torch.autograd.function.BackwardCFunction, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.function.InplaceFunction`, https://pytorch.org/docs/stable/generated/torch.autograd.function.InplaceFunction.html#torch.autograd.function.InplaceFunction, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.function.NestedIOFunction`, https://pytorch.org/docs/stable/generated/torch.autograd.function.NestedIOFunction.html#torch.autograd.function.NestedIOFunction, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.function.once_differentiable`, https://pytorch.org/docs/stable/generated/torch.autograd.function.once_differentiable.html#torch-autograd-function-once-differentiable, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.Function.vmap`, https://pytorch.org/docs/stable/generated/torch.autograd.Function.vmap.html#torch-autograd-function-vmap, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.functional.hvp`, https://pytorch.org/docs/stable/generated/torch.autograd.functional.hvp.html#torch-autograd-functional-hvp, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.functional.vhp`, https://pytorch.org/docs/stable/generated/torch.autograd.functional.vhp.html#torch-autograd-functional-vhp, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.grad_mode.inference_mode`, https://pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html#torch.autograd.grad_mode.inference_mode, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.grad_mode.set_multithreading_enabled`, https://pytorch.org/docs/stable/generated/torch.autograd.grad_mode.set_multithreading_enabled.html#torch.autograd.grad_mode.set_multithreading_enabled, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.gradcheck.GradcheckError`, https://pytorch.org/docs/stable/generated/torch.autograd.gradcheck.GradcheckError.html#torch-autograd-gradcheck-gradcheckerror, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.gradcheck.gradgradcheck`, https://pytorch.org/docs/stable/generated/torch.autograd.gradcheck.gradgradcheck.html#torch-autograd-gradcheck-gradgradcheck, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.graph.allow_mutation_on_saved_tensors`, https://pytorch.org/docs/stable/autograd.html#torch.autograd.graph.allow_mutation_on_saved_tensors, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.graph.disable_saved_tensors_hooks`, https://pytorch.org/docs/stable/autograd.html#torch.autograd.graph.disable_saved_tensors_hooks, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.graph.get_gradient_edge`, https://pytorch.org/docs/stable/autograd.html#torch.autograd.graph.get_gradient_edge, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.graph.GradientEdge`, https://pytorch.org/docs/stable/autograd.html#torch.autograd.graph.GradientEdge, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.graph.increment_version`, https://pytorch.org/docs/stable/generated/torch.autograd.graph.increment_version.html#torch-autograd-graph-increment-version, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.graph.register_multi_grad_hook`, https://pytorch.org/docs/stable/autograd.html#torch.autograd.graph.register_multi_grad_hook, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.profiler.emit_itt`, https://pytorch.org/docs/stable/autograd.html#torch.autograd.profiler.emit_itt, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.profiler.emit_nvtx`, https://pytorch.org/docs/stable/autograd.html#torch.autograd.profiler.emit_nvtx, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.profiler.EnforceUnique`, https://pytorch.org/docs/stable/generated/torch.autograd.profiler.EnforceUnique.html#torch.autograd.profiler.EnforceUnique, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.profiler.KinetoStepTracker`, https://pytorch.org/docs/stable/generated/torch.autograd.profiler.KinetoStepTracker.html#torch.autograd.profiler.KinetoStepTracker, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.profiler.parse_nvprof_trace`, https://pytorch.org/docs/stable/generated/torch.autograd.profiler.parse_nvprof_trace.html#torch-autograd-profiler-parse-nvprof-trace, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.profiler.profile.total_average`, https://pytorch.org/docs/stable/generated/torch.autograd.profiler.profile.total_average.html#torch-autograd-profiler-profile-total-average, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.profiler_util.Interval`, https://pytorch.org/docs/stable/generated/torch.autograd.profiler_util.Interval.html#torch.autograd.profiler_util.Interval, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.profiler_util.Kernel`, https://pytorch.org/docs/stable/generated/torch.autograd.profiler_util.Kernel.html#torch.autograd.profiler_util.Kernel, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.profiler_util.StringTable`, https://pytorch.org/docs/stable/generated/torch.autograd.profiler_util.StringTable.html#torch.autograd.profiler_util.StringTable, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.backends.cuda.can_use_efficient_attention`, https://pytorch.org/docs/stable/backends.html#torch.backends.cuda.can_use_efficient_attention, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.backends.cuda.cudnn_sdp_enabled`, https://pytorch.org/docs/stable/backends.html#torch.backends.cuda.cudnn_sdp_enabled, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.backends.cuda.enable_cudnn_sdp`, https://pytorch.org/docs/stable/backends.html#torch.backends.cuda.enable_cudnn_sdp, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.backends.cuda.flash_sdp_enabled`, https://pytorch.org/docs/stable/backends.html#torch.backends.cuda.flash_sdp_enabled, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.backends.cuda.math_sdp_enabled`, https://pytorch.org/docs/stable/backends.html#torch.backends.cuda.math_sdp_enabled, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.backends.cuda.mem_efficient_sdp_enabled`, https://pytorch.org/docs/stable/backends.html#torch.backends.cuda.mem_efficient_sdp_enabled, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.backends.cuda.SDPAParams`, https://pytorch.org/docs/stable/backends.html#torch.backends.cuda.SDPAParams, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.backends.mha.get_fastpath_enabled`, https://pytorch.org/docs/stable/backends.html#torch.backends.mha.get_fastpath_enabled, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.backends.mha.set_fastpath_enabled`, https://pytorch.org/docs/stable/backends.html#torch.backends.mha.set_fastpath_enabled, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.backends.mkl.verbose`, https://pytorch.org/docs/stable/backends.html#torch.backends.mkl.verbose, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.backends.mkldnn.is_available`, https://pytorch.org/docs/stable/backends.html#torch.backends.mkldnn.is_available, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.backends.mkldnn.verbose`, https://pytorch.org/docs/stable/backends.html#torch.backends.mkldnn.verbose, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.backends.nnpack.flags`, https://pytorch.org/docs/stable/backends.html#torch.backends.nnpack.flags, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.backends.nnpack.is_available`, https://pytorch.org/docs/stable/backends.html#torch.backends.nnpack.is_available, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.backends.openmp.is_available`, https://pytorch.org/docs/stable/backends.html#torch.backends.openmp.is_available, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.backends.opt_einsum.get_opt_einsum`, https://pytorch.org/docs/stable/backends.html#torch.backends.opt_einsum.get_opt_einsum, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.backends.opt_einsum.is_available`, https://pytorch.org/docs/stable/backends.html#torch.backends.opt_einsum.is_available, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.signal.windows.nuttall`, https://pytorch.org/docs/stable/generated/torch.signal.windows.nuttall.html#torch-signal-windows-nuttall, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.export.dims`, https://pytorch.org/docs/stable/export.html#torch.export.dims, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.export.dynamic_shapes.Dim`, https://pytorch.org/docs/stable/export.html#torch.export.dynamic_shapes.Dim, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.export.dynamic_shapes.dynamic_dim`, https://pytorch.org/docs/stable/export.html#torch.export.dynamic_shapes.dynamic_dim, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.export.ExportBackwardSignature`, https://pytorch.org/docs/stable/export.html#torch.export.ExportBackwardSignature, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.export.ExportGraphSignature`, https://pytorch.org/docs/stable/export.html#torch.export.ExportGraphSignature, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.export.graph_signature.CustomObjArgument`, https://pytorch.org/docs/stable/export.html#torch.export.graph_signature.CustomObjArgument, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.export.graph_signature.ExportGraphSignature`, https://pytorch.org/docs/stable/export.html#torch.export.graph_signature.ExportGraphSignature, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.export.graph_signature.InputKind`, https://pytorch.org/docs/stable/export.html#torch.export.graph_signature.InputKind, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.export.graph_signature.OutputKind`, https://pytorch.org/docs/stable/export.html#torch.export.graph_signature.OutputKind, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.export.graph_signature.OutputSpec`, https://pytorch.org/docs/stable/export.html#torch.export.graph_signature.OutputSpec, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.export.ModuleCallEntry`, https://pytorch.org/docs/stable/export.html#torch.export.ModuleCallEntry, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.export.ModuleCallSignature`, https://pytorch.org/docs/stable/export.html#torch.export.ModuleCallSignature, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.export.register_dataclass`, https://pytorch.org/docs/stable/export.html#torch.export.register_dataclass, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.export.save`, https://pytorch.org/docs/stable/export.html#torch.export.save, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.export.unflatten.FlatArgsAdapter`, https://pytorch.org/docs/stable/export.html#torch.export.unflatten.FlatArgsAdapter, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.export.unflatten.InterpreterModule`, https://pytorch.org/docs/stable/export.html#torch.export.unflatten.InterpreterModule, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.export.unflatten.unflatten`, https://pytorch.org/docs/stable/export.html#torch.export.unflatten.unflatten, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.fx.experimental.symbolic_shapes.canonicalize_bool_expr`, https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.canonicalize_bool_expr.html#torch-fx-experimental-symbolic-shapes-canonicalize-bool-expr, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.fx.experimental.symbolic_shapes.constrain_range`, https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.constrain_range.html#torch-fx-experimental-symbolic-shapes-constrain-range, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.fx.experimental.symbolic_shapes.constrain_unify`, https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.constrain_unify.html#torch-fx-experimental-symbolic-shapes-constrain-unify, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.fx.experimental.symbolic_shapes.definitely_false`, https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.definitely_false.html#torch-fx-experimental-symbolic-shapes-definitely-false, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.fx.experimental.symbolic_shapes.definitely_true`, https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.definitely_true.html#torch-fx-experimental-symbolic-shapes-definitely-true, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.fx.experimental.symbolic_shapes.DimConstraints`, https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.DimConstraints.html#torch.fx.experimental.symbolic_shapes.DimConstraints, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.fx.experimental.symbolic_shapes.DimDynamic`, https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.DimDynamic.html#torch.fx.experimental.symbolic_shapes.DimDynamic, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.fx.experimental.symbolic_shapes.EqualityConstraint`, https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.EqualityConstraint.html#torch.fx.experimental.symbolic_shapes.EqualityConstraint, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.fx.experimental.symbolic_shapes.guard_size_oblivious`, https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.guard_size_oblivious.html#torch-fx-experimental-symbolic-shapes-guard-size-oblivious, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.fx.experimental.symbolic_shapes.has_free_symbols`, https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.has_free_symbols.html#torch-fx-experimental-symbolic-shapes-has-free-symbols, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.fx.experimental.symbolic_shapes.hint_int`, https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.hint_int.html#torch-fx-experimental-symbolic-shapes-hint-int, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.fx.experimental.symbolic_shapes.is_concrete_bool`, https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.is_concrete_bool.html#torch-fx-experimental-symbolic-shapes-is-concrete-bool, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.fx.experimental.symbolic_shapes.is_concrete_int`, https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.is_concrete_int.html#torch-fx-experimental-symbolic-shapes-is-concrete-int, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.fx.experimental.symbolic_shapes.parallel_and`, https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.parallel_and.html#torch-fx-experimental-symbolic-shapes-parallel-and, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.fx.experimental.symbolic_shapes.parallel_or`, https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.parallel_or.html#torch-fx-experimental-symbolic-shapes-parallel-or, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.fx.experimental.symbolic_shapes.RelaxedUnspecConstraint`, https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.RelaxedUnspecConstraint.html#torch.fx.experimental.symbolic_shapes.RelaxedUnspecConstraint, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.fx.experimental.symbolic_shapes.ShapeEnv`, https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.fx.experimental.symbolic_shapes.StatefulSymbolicContext`, https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.StatefulSymbolicContext.html#torch.fx.experimental.symbolic_shapes.StatefulSymbolicContext, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.fx.experimental.symbolic_shapes.StatelessSymbolicContext`, https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.StatelessSymbolicContext.html#torch.fx.experimental.symbolic_shapes.StatelessSymbolicContext, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.fx.experimental.symbolic_shapes.statically_known_true`, https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.statically_known_true.html#torch-fx-experimental-symbolic-shapes-statically-known-true, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.fx.experimental.symbolic_shapes.StrictMinMaxConstraint`, https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.StrictMinMaxConstraint.html#torch.fx.experimental.symbolic_shapes.StrictMinMaxConstraint, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.fx.experimental.symbolic_shapes.SubclassSymbolicContext`, https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.SubclassSymbolicContext.html#torch.fx.experimental.symbolic_shapes.SubclassSymbolicContext, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.fx.experimental.symbolic_shapes.sym_eq`, https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.sym_eq.html#torch-fx-experimental-symbolic-shapes-sym-eq, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.fx.experimental.symbolic_shapes.SymbolicContext`, https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.SymbolicContext.html#torch.fx.experimental.symbolic_shapes.SymbolicContext, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.jit.interface`, https://pytorch.org/docs/stable/generated/torch.jit.interface.html#torch-jit-interface, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.jit.onednn_fusion_enabled`, https://pytorch.org/docs/stable/generated/torch.jit.onednn_fusion_enabled.html#torch-jit-onednn-fusion-enabled, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.jit.ScriptFunction`, https://pytorch.org/docs/stable/generated/torch.jit.ScriptFunction.html#torch.jit.ScriptFunction, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.jit.strict_fusion`, https://pytorch.org/docs/stable/generated/torch.jit.strict_fusion.html#torch.jit.strict_fusion, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.sym_float`, https://pytorch.org/docs/stable/generated/torch.sym_float.html#torch-sym-float, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.sym_int`, https://pytorch.org/docs/stable/generated/torch.sym_int.html#torch-sym-int, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.sym_ite`, https://pytorch.org/docs/stable/generated/torch.sym_ite.html#torch-sym-ite, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.sym_max`, https://pytorch.org/docs/stable/generated/torch.sym_max.html#torch-sym-max, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.sym_min`, https://pytorch.org/docs/stable/generated/torch.sym_min.html#torch-sym-min, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.sym_not`, https://pytorch.org/docs/stable/generated/torch.sym_not.html#torch-sym-not, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.SymBool`, https://pytorch.org/docs/stable/torch.html#torch.SymBool, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.SymFloat`, https://pytorch.org/docs/stable/torch.html#torch.SymFloat, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.SymInt`, https://pytorch.org/docs/stable/torch.html#torch.SymInt, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.cpu.current_device`, https://pytorch.org/docs/stable/generated/torch.cpu.current_device.html#torch-cpu-current-device, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.cpu.current_stream`, https://pytorch.org/docs/stable/generated/torch.cpu.current_stream.html#torch.cpu.current_stream, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.cpu.device_count`, https://pytorch.org/docs/stable/generated/torch.cpu.device_count.html#torch-cpu-device-count, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.cpu.is_available`, https://pytorch.org/docs/stable/generated/torch.cpu.is_available.html#torch-cpu-is-available, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.cpu.set_device`, https://pytorch.org/docs/stable/generated/torch.cpu.set_device.html#torch-cpu-set-device, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.cpu.stream`, https://pytorch.org/docs/stable/generated/torch.cpu.stream.html#torch-cpu-stream, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.cpu.Stream`, https://pytorch.org/docs/stable/generated/torch.cpu.Stream.html#torch.cpu.Stream, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.cpu.StreamContext`, https://pytorch.org/docs/stable/generated/torch.cpu.StreamContext.html#torch.cpu.StreamContext, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.cpu.synchronize`, https://pytorch.org/docs/stable/generated/torch.cpu.synchronize.html#torch-cpu-synchronize, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.change_current_allocator`, https://pytorch.org/docs/stable/generated/torch.cuda.change_current_allocator.html#torch-cuda-change-current-allocator, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.clock_rate`, https://pytorch.org/docs/stable/generated/torch.cuda.clock_rate.html#torch-cuda-clock-rate, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.CUDAPluggableAllocator`, https://pytorch.org/docs/stable/generated/torch.cuda.CUDAPluggableAllocator.html#torch.cuda.CUDAPluggableAllocator, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.current_blas_handle`, https://pytorch.org/docs/stable/generated/torch.cuda.current_blas_handle.html#torch-cuda-current-blas-handle, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.get_gencode_flags`, https://pytorch.org/docs/stable/generated/torch.cuda.get_gencode_flags.html#torch-cuda-get-gencode-flags, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.graph`, https://pytorch.org/docs/stable/cuda.html#module-torch.cuda.graphs, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.jiterator._create_multi_output_jit_fn`, https://pytorch.org/docs/stable/generated/torch.cuda.jiterator._create_multi_output_jit_fn.html#torch-cuda-jiterator-create-multi-output-jit-fn, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.make_graphed_callables`, https://pytorch.org/docs/stable/generated/torch.cuda.make_graphed_callables.html#torch-cuda-make-graphed-callables, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.memory._dump_snapshot`, https://pytorch.org/docs/stable/torch_cuda_memory.html#torch.cuda.memory._dump_snapshot, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.memory._snapshot`, https://pytorch.org/docs/stable/torch_cuda_memory.html#torch.cuda.memory._snapshot, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.power_draw`, https://pytorch.org/docs/stable/generated/torch.cuda.power_draw.html#torch-cuda-power-draw, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.temperature`, https://pytorch.org/docs/stable/generated/torch.cuda.temperature.html#torch-cuda-temperature, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.algorithms.ddp_comm_hooks.debugging_hooks.noop_hook`, https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.debugging_hooks.noop_hook, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.algorithms.ddp_comm_hooks.default_hooks.allreduce_hook`, https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.allreduce_hook, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.algorithms.ddp_comm_hooks.default_hooks.bf16_compress_hook`, https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.bf16_compress_hook, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.algorithms.ddp_comm_hooks.default_hooks.bf16_compress_wrapper`, https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.bf16_compress_wrapper, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_hook`, https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_hook, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_wrapper`, https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_wrapper, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.batched_powerSGD_hook`, https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.batched_powerSGD_hook, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.powerSGD_hook`, https://pytorch.org/docs/stable/distributed.html#module-torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.GradBucket`, https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.GradBucket, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.GradBucket.buffer`, https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.GradBucket.buffer, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.GradBucket.gradients`, https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.GradBucket.gradients, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.GradBucket.index`, https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.GradBucket.index, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.GradBucket.is_last`, https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.GradBucket.is_last, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.GradBucket.parameters`, https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.GradBucket.parameters, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.GradBucket.set_buffer`, https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.GradBucket.set_buffer, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.algorithms.Join`, https://pytorch.org/docs/stable/distributed.algorithms.join.html#torch.distributed.algorithms.Join, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.algorithms.Joinable`, https://pytorch.org/docs/stable/distributed.algorithms.join.html#torch.distributed.algorithms.Joinable, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.algorithms.JoinHook`, https://pytorch.org/docs/stable/distributed.algorithms.join.html#torch.distributed.algorithms.JoinHook, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.autograd.context`, https://pytorch.org/docs/stable/rpc.html#torch.distributed.autograd.context, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.autograd.get_gradients`, https://pytorch.org/docs/stable/rpc.html#torch.distributed.autograd.get_gradients, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.breakpoint`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.breakpoint, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.DistBackendError`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.DistBackendError, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.DistError`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.DistError, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.DistNetworkError`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.DistNetworkError, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.DistStoreError`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.DistStoreError, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.checkpoint.DefaultLoadPlanner`, https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.DefaultLoadPlanner, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.checkpoint.DefaultSavePlanner`, https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.DefaultSavePlanner, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.checkpoint.filesystem.FileSystemReader`, https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.FileSystemReader, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.checkpoint.filesystem.FileSystemWriter`, https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.FileSystemWriter, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader`, https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.checkpoint.format_utils.dcp_to_torch_save`, https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.format_utils.dcp_to_torch_save, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.checkpoint.format_utils.DynamicMetaLoadPlanner`, https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.format_utils.DynamicMetaLoadPlanner, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.checkpoint.format_utils.torch_save_to_dcp`, https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.format_utils.torch_save_to_dcp, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.checkpoint.fsspec.FsspecReader`, https://pytorch.org/docs/2.3/distributed.checkpoint.html#torch.distributed.checkpoint.fsspec.FsspecReader, 废弃 API) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.checkpoint.fsspec.FsspecWriter`, https://pytorch.org/docs/2.3/distributed.checkpoint.html#torch.distributed.checkpoint.fsspec.FsspecWriter, 废弃 API) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.checkpoint.LoadPlan`, https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.LoadPlan, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.checkpoint.LoadPlanner`, https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.LoadPlanner, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.checkpoint.ReadItem`, https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.ReadItem, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.checkpoint.SavePlan`, https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.SavePlan, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.checkpoint.SavePlanner`, https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.SavePlanner, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.checkpoint.state_dict.get_state_dict`, https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.get_state_dict, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.checkpoint.state_dict.set_state_dict`, https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.set_state_dict, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.checkpoint.state_dict_loader.load`, https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict_loader.load, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.checkpoint.state_dict_loader.load_state_dict`, https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict_loader.load_state_dict, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.checkpoint.state_dict_saver.async_save`, https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict_saver.async_save, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.checkpoint.state_dict_saver.save`, https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict_saver.save, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.checkpoint.state_dict_saver.save_state_dict`, https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict_saver.save_state_dict, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.checkpoint.stateful.Stateful`, https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.stateful.Stateful, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.checkpoint.StorageReader`, https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.StorageReader, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.checkpoint.StorageWriter`, https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.StorageWriter, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.is_mpi_available`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.is_mpi_available, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.is_torchelastic_launched`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.is_torchelastic_launched, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.Work`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.Work, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.HashStore`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.HashStore, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.Store`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.Store, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.Store.add`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.Store.add, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.Store.compare_set`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.Store.compare_set, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.Store.delete_key`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.Store.delete_key, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.Store.get`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.Store.get, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.Store.num_keys`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.Store.num_keys, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.Store.set`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.Store.set, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.Store.set_timeout`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.Store.set_timeout, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.Store.wait`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.Store.wait, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.fsdp.BackwardPrefetch`, https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.BackwardPrefetch, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.fsdp.LocalOptimStateDictConfig`, https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.LocalOptimStateDictConfig, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.fsdp.OptimStateDictConfig`, https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.OptimStateDictConfig, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.fsdp.ShardedOptimStateDictConfig`, https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.ShardedOptimStateDictConfig, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.fsdp.ShardedStateDictConfig`, https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.ShardedStateDictConfig, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.fsdp.ShardingStrategy`, https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.ShardingStrategy, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.fsdp.StateDictConfig`, https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.StateDictConfig, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.fsdp.StateDictSettings`, https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.StateDictSettings, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.nn.api.remote_module.RemoteModule`, https://pytorch.org/docs/stable/rpc.html#torch.distributed.nn.api.remote_module.RemoteModule, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.rpc.BackendType`, https://pytorch.org/docs/stable/rpc.html#torch.distributed.rpc.BackendType, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.rpc.PyRRef`, https://pytorch.org/docs/stable/rpc.html#torch.distributed.rpc.PyRRef, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.rpc.RpcBackendOptions`, https://pytorch.org/docs/stable/rpc.html#torch.distributed.rpc.RpcBackendOptions, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.optim.PostLocalSGDOptimizer`, https://pytorch.org/docs/stable/distributed.optim.html#torch.distributed.optim.PostLocalSGDOptimizer, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.pipeline.sync.skip.skippable.pop`, https://pytorch.org/docs/2.3/pipeline.html#torch.distributed.pipeline.sync.skip.skippable.pop, 废弃 API) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.pipeline.sync.skip.skippable.skippable`, https://pytorch.org/docs/2.3/pipeline.html#torch.distributed.pipeline.sync.skip.skippable.skippable, 废弃 API) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.pipeline.sync.skip.skippable.stash`, https://pytorch.org/docs/2.3/pipeline.html#torch.distributed.pipeline.sync.skip.skippable.stash, 废弃 API) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.pipeline.sync.skip.skippable.verify_skippables`, https://pytorch.org/docs/2.3/pipeline.html#torch.distributed.pipeline.sync.skip.skippable.verify_skippables, 废弃 API) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.tensor.parallel.loss_parallel`, https://pytorch.org/docs/stable/distributed.tensor.parallel.html#torch.distributed.tensor.parallel.loss_parallel, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.tensor.parallel.PrepareModuleInput`, https://pytorch.org/docs/stable/distributed.tensor.parallel.html#torch.distributed.tensor.parallel.PrepareModuleInput, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.tensor.parallel.PrepareModuleOutput`, https://pytorch.org/docs/stable/distributed.tensor.parallel.html#torch.distributed.tensor.parallel.PrepareModuleOutput, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.tensor.parallel.SequenceParallel`, https://pytorch.org/docs/stable/distributed.tensor.parallel.html#torch.distributed.tensor.parallel.SequenceParallel, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributions.fishersnedecor.FisherSnedecor`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.fishersnedecor.FisherSnedecor, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributions.half_cauchy.HalfCauchy`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.half_cauchy.HalfCauchy, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributions.half_normal.HalfNormal`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.half_normal.HalfNormal, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributions.inverse_gamma.InverseGamma`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.inverse_gamma.InverseGamma, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributions.kumaraswamy.Kumaraswamy`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.kumaraswamy.Kumaraswamy, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributions.mixture_same_family.MixtureSameFamily`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.mixture_same_family.MixtureSameFamily, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributions.negative_binomial.NegativeBinomial`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.negative_binomial.NegativeBinomial, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributions.pareto.Pareto`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.pareto.Pareto, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributions.relaxed_categorical.RelaxedOneHotCategorical`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.relaxed_categorical.RelaxedOneHotCategorical, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributions.von_mises.VonMises`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.von_mises.VonMises, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributions.weibull.Weibull`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.weibull.Weibull, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributions.wishart.Wishart`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.wishart.Wishart, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.dequantize`, https://pytorch.org/docs/stable/generated/torch.dequantize.html#torch-dequantize, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.quantized_batch_norm`, https://pytorch.org/docs/stable/generated/torch.quantized_batch_norm.html#torch-quantized-batch-norm, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.quantized_max_pool1d`, https://pytorch.org/docs/stable/generated/torch.quantized_max_pool1d.html#torch-quantized-max-pool1d, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.quantized_max_pool2d`, https://pytorch.org/docs/stable/generated/torch.quantized_max_pool2d.html#torch-quantized-max-pool2d, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.overrides.get_ignored_functions`, https://pytorch.org/docs/stable/torch.overrides.html#torch.overrides.get_ignored_functions, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.overrides.handle_torch_function`, https://pytorch.org/docs/stable/torch.overrides.html#torch.overrides.handle_torch_function, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.overrides.is_tensor_method_or_property`, https://pytorch.org/docs/stable/torch.overrides.html#torch.overrides.is_tensor_method_or_property, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.package.Directory`, https://pytorch.org/docs/stable/package.html#torch.package.Directory, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.package.EmptyMatchError`, https://pytorch.org/docs/stable/package.html#torch.package.EmptyMatchError, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.package.PackageExporter`, https://pytorch.org/docs/stable/package.html#torch.package.PackageExporter, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.package.PackagingError`, https://pytorch.org/docs/stable/package.html#torch.package.PackagingError, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.hspmm`, https://pytorch.org/docs/stable/generated/torch.hspmm.html#torch-hspmm, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.smm`, https://pytorch.org/docs/stable/generated/torch.smm.html#torch-smm, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.sparse.as_sparse_gradcheck`, https://pytorch.org/docs/stable/generated/torch.sparse.as_sparse_gradcheck.html#torch-sparse-as-sparse-gradcheck, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.sparse.check_sparse_tensor_invariants`, https://pytorch.org/docs/stable/generated/torch.sparse.check_sparse_tensor_invariants.html#torch.sparse.check_sparse_tensor_invariants, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.sparse.spdiags`, https://pytorch.org/docs/stable/generated/torch.sparse.spdiags.html#torch-sparse-spdiags, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.sspaddmm`, https://pytorch.org/docs/stable/generated/torch.sspaddmm.html#torch-sspaddmm, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.library.fallthrough_kernel`, https://pytorch.org/docs/stable/library.html#torch.library.fallthrough_kernel, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.linalg.solve_ex`, https://pytorch.org/docs/stable/generated/torch.linalg.solve_ex.html#torch-linalg-solve-ex, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.lu_solve`, https://pytorch.org/docs/stable/generated/torch.lu_solve.html#torch-lu-solve, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.monitor.Aggregation`, https://pytorch.org/docs/stable/monitor.html#torch.monitor.Aggregation, 原型 API) |
| NOT-IMPLEMENTED-ITEM(`torch.monitor.data_value_t`, https://pytorch.org/docs/stable/monitor.html#torch.monitor.data_value_t, 原型 API) |
| NOT-IMPLEMENTED-ITEM(`torch.monitor.EventHandlerHandle`, https://pytorch.org/docs/stable/monitor.html#torch.monitor.EventHandlerHandle, 原型 API) |
| NOT-IMPLEMENTED-ITEM(`torch.monitor.TensorboardEventHandler`, https://pytorch.org/docs/stable/monitor.html#torch.monitor.TensorboardEventHandler, 原型 API) |
| NOT-IMPLEMENTED-ITEM(`torch.nested.as_nested_tensor`, https://pytorch.org/docs/stable/nested.html#torch.nested.as_nested_tensor, 原型 API) |
| NOT-IMPLEMENTED-ITEM(`torch.nested.to_padded_tensor`, https://pytorch.org/docs/stable/nested.html#torch.nested.to_padded_tensor, 原型 API) |
| NOT-IMPLEMENTED-ITEM(`torch.mps.driver_allocated_memory`, https://pytorch.org/docs/stable/generated/torch.mps.driver_allocated_memory.html#torch-mps-driver-allocated-memory, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.mps.event.Event`, https://pytorch.org/docs/stable/generated/torch.mps.event.Event.html#torch.mps.event.Event, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.mps.get_rng_state`, https://pytorch.org/docs/stable/generated/torch.mps.get_rng_state.html#torch-mps-get-rng-state, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.mps.profiler.profile`, https://pytorch.org/docs/stable/generated/torch.mps.profiler.profile.html#torch-mps-profiler-profile, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.mps.seed`, https://pytorch.org/docs/stable/generated/torch.mps.seed.html#torch-mps-seed, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.mps.set_per_process_memory_fraction`, https://pytorch.org/docs/stable/generated/torch.mps.set_per_process_memory_fraction.html#torch-mps-set-per-process-memory-fraction, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.mps.set_rng_state`, https://pytorch.org/docs/stable/generated/torch.mps.set_rng_state.html#torch-mps-set-rng-state, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.attention.bias`, https://pytorch.org/docs/stable/nn.attention.bias.html#module-torch.nn.attention.bias, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.attention.sdpa_kernel`, https://pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html#torch-nn-attention-sdpa-kernel, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.attention.SDPBackend`, https://pytorch.org/docs/stable/generated/torch.nn.attention.SDPBackend.html#torch.nn.attention.SDPBackend, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.CircularPad1d`, https://pytorch.org/docs/stable/generated/torch.nn.CircularPad1d.html#torch.nn.CircularPad1d, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.CircularPad2d`, https://pytorch.org/docs/stable/generated/torch.nn.CircularPad2d.html#torch.nn.CircularPad2d, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.functional.lp_pool3d`, https://pytorch.org/docs/stable/generated/torch.nn.functional.lp_pool3d.html#torch-nn-functional-lp-pool3d, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.LPPool3d`, https://pytorch.org/docs/stable/generated/torch.nn.LPPool3d.html#torch.nn.LPPool3d, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.modules.lazy.LazyModuleMixin`, https://pytorch.org/docs/stable/generated/torch.nn.modules.lazy.LazyModuleMixin.html#torch.nn.modules.lazy.LazyModuleMixin, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.modules.module.register_module_buffer_registration_hook`, https://pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_buffer_registration_hook.html#torch-nn-modules-module-register-module-buffer-registration-hook, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.modules.module.register_module_full_backward_hook`, https://pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_full_backward_hook.html#torch-nn-modules-module-register-module-full-backward-hook, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.modules.module.register_module_full_backward_pre_hook`, https://pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_full_backward_pre_hook.html#torch-nn-modules-module-register-module-full-backward-pre-hook, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.modules.module.register_module_module_registration_hook`, https://pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_module_registration_hook.html#torch-nn-modules-module-register-module-module-registration-hook, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.modules.module.register_module_parameter_registration_hook`, https://pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_parameter_registration_hook.html#torch-nn-modules-module-register-module-parameter-registration-hook, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.parametrize.cached`, https://pytorch.org/docs/stable/generated/torch.nn.utils.parametrize.cached.html#torch-nn-utils-parametrize-cached, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.parametrize.ParametrizationList`, https://pytorch.org/docs/stable/generated/torch.nn.utils.parametrize.ParametrizationList.html#torch.nn.utils.parametrize.ParametrizationList, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.prune.BasePruningMethod`, https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.BasePruningMethod.html#torch.nn.utils.prune.BasePruningMethod, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.prune.custom_from_mask`, https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.custom_from_mask.html#torch-nn-utils-prune-custom-from-mask, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.prune.CustomFromMask`, https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.CustomFromMask.html#torch.nn.utils.prune.CustomFromMask, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.prune.Identity`, https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.identity.html#torch-nn-utils-prune-identity, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.prune.is_pruned`, https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.is_pruned.html#torch-nn-utils-prune-is-pruned, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.prune.LnStructured`, https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.LnStructured.html#torch.nn.utils.prune.LnStructured, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.prune.RandomUnstructured`, https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.RandomUnstructured.html#torch.nn.utils.prune.RandomUnstructured, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.rnn.unpack_sequence`, https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.unpack_sequence.html#torch-nn-utils-rnn-unpack-sequence, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.ZeroPad1d`, https://pytorch.org/docs/stable/generated/torch.nn.ZeroPad1d.html#torch.nn.ZeroPad1d, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.ZeroPad3d`, https://pytorch.org/docs/stable/generated/torch.nn.ZeroPad3d.html#torch.nn.ZeroPad3d, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.profiler._KinetoProfile`, https://pytorch.org/docs/stable/profiler.html#torch.profiler._KinetoProfile, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.profiler.itt.is_available`, https://pytorch.org/docs/stable/profiler.html#torch.profiler.itt.is_available, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.profiler.itt.mark`, https://pytorch.org/docs/stable/profiler.html#torch.profiler.itt.mark, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.profiler.itt.range_pop`, https://pytorch.org/docs/stable/profiler.html#torch.profiler.itt.range_pop, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.profiler.itt.range_push`, https://pytorch.org/docs/stable/profiler.html#torch.profiler.itt.range_push, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.special.airy_ai`, https://pytorch.org/docs/stable/special.html#torch.special.airy_ai, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.special.bessel_j0`, https://pytorch.org/docs/stable/special.html#torch.special.bessel_j0, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.special.bessel_j1`, https://pytorch.org/docs/stable/special.html#torch.special.bessel_j1, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.special.scaled_modified_bessel_k0`, https://pytorch.org/docs/stable/special.html#torch.special.scaled_modified_bessel_k0, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.special.scaled_modified_bessel_k1`, https://pytorch.org/docs/stable/special.html#torch.special.scaled_modified_bessel_k1, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.special.spherical_bessel_j0`, https://pytorch.org/docs/stable/special.html#torch.special.spherical_bessel_j0, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.arctan2_`, https://pytorch.org/docs/stable/generated/torch.Tensor.arctan2_.html#torch-tensor-arctan2, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.conj_physical_`, https://pytorch.org/docs/stable/generated/torch.Tensor.conj_physical_.html#torch-tensor-conj-physical, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.is_meta`, https://pytorch.org/docs/stable/generated/torch.Tensor.is_meta.html#torch-tensor-is-meta, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.is_quantized`, https://pytorch.org/docs/stable/generated/torch.Tensor.is_quantized.html#torch-tensor-is-quantized, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.module_load`, https://pytorch.org/docs/stable/generated/torch.Tensor.module_load.html#torch-tensor-module-load, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.nextafter_`, https://pytorch.org/docs/stable/generated/torch.Tensor.nextafter_.html#torch-tensor-nextafter, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.retains_grad`, https://pytorch.org/docs/stable/generated/torch.Tensor.retains_grad.html#torch-tensor-retains-grad, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.smm`, https://pytorch.org/docs/stable/generated/torch.Tensor.smm.html#torch-tensor-smm, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.utils.benchmark.CallgrindStats`, https://pytorch.org/docs/stable/benchmark_utils.html#torch.utils.benchmark.CallgrindStats, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.utils.benchmark.FunctionCounts`, https://pytorch.org/docs/stable/benchmark_utils.html#torch.utils.benchmark.FunctionCounts, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.utils.benchmark.Measurement`, https://pytorch.org/docs/stable/benchmark_utils.html#torch.utils.benchmark.Measurement, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.utils.checkpoint.set_checkpoint_debug_enabled`, https://pytorch.org/docs/stable/checkpoint.html#torch.utils.checkpoint.set_checkpoint_debug_enabled, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.utils.cpp_extension.get_compiler_abi_compatibility_and_version`, https://pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.get_compiler_abi_compatibility_and_version, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.utils.cpp_extension.is_ninja_available`, https://pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.is_ninja_available, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.utils.data.default_convert`, https://pytorch.org/docs/stable/data.html#torch.utils.data.default_convert, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.utils.generate_methods_for_privateuse1_backend`, https://pytorch.org/docs/stable/generated/torch.utils.generate_methods_for_privateuse1_backend.html#torch-utils-generate-methods-for-privateuse1-backend, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.utils.get_cpp_backtrace`, https://pytorch.org/docs/stable/generated/torch.utils.get_cpp_backtrace.html#torch-utils-get-cpp-backtrace, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.utils.rename_privateuse1_backend`, https://pytorch.org/docs/stable/generated/torch.utils.rename_privateuse1_backend.html#torch-utils-rename-privateuse1-backend, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.xpu.current_stream`, https://pytorch.org/docs/stable/generated/torch.xpu.current_stream.html#torch-xpu-current-stream, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.xpu.device`, https://pytorch.org/docs/stable/generated/torch.xpu.device.html#torch.xpu.device, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.xpu.device_of`, https://pytorch.org/docs/stable/generated/torch.xpu.device_of.html#torch.xpu.device_of, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.xpu.Event`, https://pytorch.org/docs/stable/generated/torch.xpu.Event.html#torch.xpu.Event, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.xpu.get_device_capability`, https://pytorch.org/docs/stable/generated/torch.xpu.get_device_capability.html#torch-xpu-get-device-capability, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.xpu.init`, https://pytorch.org/docs/stable/generated/torch.xpu.init.html#torch-xpu-init, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.xpu.initial_seed`, https://pytorch.org/docs/stable/generated/torch.xpu.initial_seed.html#torch-xpu-initial-seed, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.xpu.is_initialized`, https://pytorch.org/docs/stable/generated/torch.xpu.is_initialized.html#torch-xpu-is-initialized, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.xpu.seed`, https://pytorch.org/docs/stable/generated/torch.xpu.seed.html#torch-xpu-seed, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.xpu.seed_all`, https://pytorch.org/docs/stable/generated/torch.xpu.seed_all.html#torch-xpu-seed-all, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.xpu.set_stream`, https://pytorch.org/docs/stable/generated/torch.xpu.set_stream.html#torch-xpu-set-stream, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.xpu.stream`, https://pytorch.org/docs/stable/generated/torch.xpu.stream.html#torch-xpu-stream, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.xpu.Stream`, https://pytorch.org/docs/stable/generated/torch.xpu.Stream.html#torch.xpu.Stream, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.xpu.StreamContext`, https://pytorch.org/docs/stable/generated/torch.xpu.StreamContext.html#torch.xpu.StreamContext, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.geqrf`, https://pytorch.org/docs/stable/generated/torch.geqrf.html#torch-geqrf, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.geqrf`, https://pytorch.org/docs/stable/generated/torch.Tensor.geqrf.html#torch-tensor-geqrf, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributions.one_hot_categorical.OneHotCategorical`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.one_hot_categorical.OneHotCategorical, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributions.constraint_registry.ConstraintRegistry`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.constraint_registry.ConstraintRegistry, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.rpc.functions.async_execution`, https://pytorch.org/docs/stable/rpc.html#torch.distributed.rpc.functions.async_execution, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.sparse_resize_and_clear_`, https://pytorch.org/docs/stable/generated/torch.Tensor.sparse_resize_and_clear_.html#torch-tensor-sparse-resize-and-clear, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.parametrize.is_parametrized`, https://pytorch.org/docs/stable/generated/torch.nn.utils.parametrize.is_parametrized.html#torch-nn-utils-parametrize-is-parametrized, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.profiler.profile.self_cpu_time_total`, https://pytorch.org/docs/stable/generated/torch.autograd.profiler.profile.self_cpu_time_total.html#torch-autograd-profiler-profile-self-cpu-time-total, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.profiler.ProfilerActivity`, https://pytorch.org/docs/stable/profiler.html#torch.profiler.ProfilerActivity, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.profiler.ProfilerAction`, https://pytorch.org/docs/stable/profiler.html#torch.profiler.ProfilerAction, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.resolve_conj`, https://pytorch.org/docs/stable/generated/torch.resolve_conj.html#torch.resolve_conj, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.resolve_neg`, https://pytorch.org/docs/stable/generated/torch.resolve_neg.html#torch-resolve-neg, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.function.FunctionCtx.mark_dirty`, https://pytorch.org/docs/stable/generated/torch.autograd.function.FunctionCtx.mark_dirty.html#torch-autograd-function-functionctx-mark-dirty, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.is_conj`, https://pytorch.org/docs/stable/generated/torch.is_conj.html#torch-is-conj, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.memory_usage`, https://pytorch.org/docs/stable/generated/torch.cuda.memory_usage.html#torch-cuda-memory-usage, 可新增且有相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.layout`, https://pytorch.org/docs/stable/tensor_attributes.html#torch.layout, 可新增且无相关设计) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.is_current_stream_capturing`, https://pytorch.org/docs/stable/generated/torch.cuda.is_current_stream_capturing.html#torch-cuda-is-current-stream-capturing, 可新增且有相关设计) |

## 开发中的 API 列表

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ----- | ----------- | ----------------- | ----------- | ------- |
| IN-DEVELOPMENT-PATTERN(`torch.igamma`, https://pytorch.org/docs/stable/generated/torch.igamma.html#torch-igamma) |
| IN-DEVELOPMENT-PATTERN(`torch.igammac`, https://pytorch.org/docs/stable/generated/torch.igammac.html#torch-igammac) |
| IN-DEVELOPMENT-PATTERN(`torch.nn.parameter.UninitializedParameter`, https://pytorch.org/docs/stable/generated/torch.nn.parameter.UninitializedParameter.html#torch.nn.parameter.UninitializedParameter) |
| IN-DEVELOPMENT-PATTERN(`torch.nn.modules.module.register_module_forward_pre_hook`, https://pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_forward_pre_hook.html#torch-nn-modules-module-register-module-forward-pre-hook) |
| IN-DEVELOPMENT-PATTERN(`torch.nn.modules.module.register_module_forward_hook`, https://pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_forward_hook.html#torch-nn-modules-module-register-module-forward-hook) |
| IN-DEVELOPMENT-PATTERN(`torch.nn.LazyConv1d`, https://pytorch.org/docs/stable/generated/torch.nn.LazyConv1d.html#torch.nn.LazyConv1d) |
| IN-DEVELOPMENT-PATTERN(`torch.nn.LazyConv2d`, https://pytorch.org/docs/stable/generated/torch.nn.LazyConv2d.html#torch.nn.LazyConv2d) |
| IN-DEVELOPMENT-PATTERN(`torch.nn.LazyConv3d`, https://pytorch.org/docs/stable/generated/torch.nn.LazyConv3d.html#torch.nn.LazyConv3d) |
| IN-DEVELOPMENT-PATTERN(`torch.nn.LazyConvTranspose1d`, https://pytorch.org/docs/stable/generated/torch.nn.LazyConvTranspose1d.html#torch.nn.LazyConvTranspose1d) |
| IN-DEVELOPMENT-PATTERN(`torch.nn.LazyConvTranspose2d`, https://pytorch.org/docs/stable/generated/torch.nn.LazyConvTranspose2d.html#torch.nn.LazyConvTranspose2d) |
| IN-DEVELOPMENT-PATTERN(`torch.nn.LazyConvTranspose3d`, https://pytorch.org/docs/stable/generated/torch.nn.LazyConvTranspose3d.html#torch.nn.LazyConvTranspose3d) |
| IN-DEVELOPMENT-PATTERN(`torch.nn.LazyLinear`, https://pytorch.org/docs/stable/generated/torch.nn.LazyLinear.html#torch.nn.LazyLinear) |
| IN-DEVELOPMENT-PATTERN(`torch.Tensor.requires_grad`, https://pytorch.org/docs/stable/generated/torch.Tensor.requires_grad.html#torch-tensor-requires-grad) |
| IN-DEVELOPMENT-PATTERN(`torch.cuda.device_of`, https://pytorch.org/docs/stable/generated/torch.cuda.device_of.html#torch.cuda.device_of) |
| IN-DEVELOPMENT-PATTERN(`torch.cuda.get_rng_state`, https://pytorch.org/docs/stable/generated/torch.cuda.get_rng_state.html#torch-cuda-get-rng-state) |
| IN-DEVELOPMENT-PATTERN(`torch.cuda.set_per_process_memory_fraction`, https://pytorch.org/docs/stable/generated/torch.cuda.set_per_process_memory_fraction.html#torch-cuda-set-per-process-memory-fraction) |
| IN-DEVELOPMENT-PATTERN(`torch.distributed.Backend`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.Backend) |
| IN-DEVELOPMENT-PATTERN(`torch.distributed.is_available`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.is_available) |
| IN-DEVELOPMENT-PATTERN(`torch.distributed.is_nccl_available`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.is_nccl_available) |
| IN-DEVELOPMENT-PATTERN(`torch.distributed.gather_object`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.gather_object) |
| IN-DEVELOPMENT-PATTERN(`torch.distributions.multivariate_normal.MultivariateNormal`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.multivariate_normal.MultivariateNormal) |
| IN-DEVELOPMENT-PATTERN(`torch.jit.script`, https://pytorch.org/docs/stable/generated/torch.jit.script.html#torch-jit-script) |
| IN-DEVELOPMENT-PATTERN(`torch.jit.trace`, https://pytorch.org/docs/stable/generated/torch.jit.trace.html#torch-jit-trace) |
| IN-DEVELOPMENT-PATTERN(`torch.jit.save`, https://pytorch.org/docs/stable/generated/torch.jit.save.html#torch-jit-save) |
| IN-DEVELOPMENT-PATTERN(`torch.jit.ignore`, https://pytorch.org/docs/stable/generated/torch.jit.ignore.html#torch-jit-ignore) |
| IN-DEVELOPMENT-PATTERN(`torch.jit.unused`, https://pytorch.org/docs/stable/generated/torch.jit.unused.html#torch-jit-unused) |
| IN-DEVELOPMENT-PATTERN(`torch.utils.checkpoint.checkpoint`, https://pytorch.org/docs/stable/checkpoint.html#torch.utils.checkpoint.checkpoint) |
| IN-DEVELOPMENT-PATTERN(`torch.utils.checkpoint.checkpoint_sequential`, https://pytorch.org/docs/stable/checkpoint.html#torch.utils.checkpoint.checkpoint_sequential) |
| IN-DEVELOPMENT-PATTERN(`torch.utils.tensorboard.writer.SummaryWriter`, https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter) |
| IN-DEVELOPMENT-PATTERN(`torch.nn.parameter.UninitializedBuffer`, https://pytorch.org/docs/stable/generated/torch.nn.parameter.UninitializedBuffer.html#torch.nn.parameter.UninitializedBuffer) |
| IN-DEVELOPMENT-PATTERN(`torch.optim.Optimizer.zero_grad`, https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html#torch-optim-optimizer-zero-grad) |
| IN-DEVELOPMENT-PATTERN(`torch.distributed.monitored_barrier`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.monitored_barrier) |
| IN-DEVELOPMENT-PATTERN(`torch.autograd.Function.jvp`, https://pytorch.org/docs/stable/generated/torch.autograd.Function.jvp.html#torch-autograd-function-jvp) |
| IN-DEVELOPMENT-PATTERN(`torch.memory_format`, https://pytorch.org/docs/stable/tensor_attributes.html#torch.memory_format) |
| IN-DEVELOPMENT-PATTERN(`torch.set_default_device`, https://pytorch.org/docs/stable/generated/torch.set_default_device.html#torch-set-default-device) |
| IN-DEVELOPMENT-PATTERN(`torch.concatenate`, https://pytorch.org/docs/stable/generated/torch.concatenate.html#torch-concatenate) |
| IN-DEVELOPMENT-PATTERN(`torch._foreach_abs`, https://pytorch.org/docs/stable/generated/torch._foreach_abs.html#torch-foreach-abs) |
| IN-DEVELOPMENT-PATTERN(`torch._foreach_abs_`, https://pytorch.org/docs/stable/generated/torch._foreach_abs_.html#torch-foreach-abs) |
| IN-DEVELOPMENT-PATTERN(`torch._foreach_acos`, https://pytorch.org/docs/stable/generated/torch._foreach_acos.html#torch-foreach-acos) |
| IN-DEVELOPMENT-PATTERN(`torch._foreach_acos_`, https://pytorch.org/docs/stable/generated/torch._foreach_acos_.html#torch-foreach-acos) |
| IN-DEVELOPMENT-PATTERN(`torch._foreach_asin`, https://pytorch.org/docs/stable/generated/torch._foreach_asin.html#torch-foreach-asin) |
| IN-DEVELOPMENT-PATTERN(`torch._foreach_asin_`, https://pytorch.org/docs/stable/generated/torch._foreach_asin_.html#torch-foreach-asin) |
| IN-DEVELOPMENT-PATTERN(`torch._foreach_atan`, https://pytorch.org/docs/stable/generated/torch._foreach_atan.html#torch-foreach-atan) |
| IN-DEVELOPMENT-PATTERN(`torch._foreach_atan_`, https://pytorch.org/docs/stable/generated/torch._foreach_atan_.html#torch-foreach-atan) |
| IN-DEVELOPMENT-PATTERN(`torch._foreach_ceil`, https://pytorch.org/docs/stable/generated/torch._foreach_ceil.html#torch-foreach-ceil) |
| IN-DEVELOPMENT-PATTERN(`torch._foreach_ceil_`, https://pytorch.org/docs/stable/generated/torch._foreach_ceil_.html#torch-foreach-ceil) |
| IN-DEVELOPMENT-PATTERN(`torch._foreach_cos`, https://pytorch.org/docs/stable/generated/torch._foreach_cos.html#torch-foreach-cos) |
| IN-DEVELOPMENT-PATTERN(`torch._foreach_cos_`, https://pytorch.org/docs/stable/generated/torch._foreach_cos_.html#torch-foreach-cos) |
| IN-DEVELOPMENT-PATTERN(`torch._foreach_cosh`, https://pytorch.org/docs/stable/generated/torch._foreach_cosh.html#torch-foreach-cosh) |
| IN-DEVELOPMENT-PATTERN(`torch._foreach_cosh_`, https://pytorch.org/docs/stable/generated/torch._foreach_cosh_.html#torch-foreach-cosh) |
| IN-DEVELOPMENT-PATTERN(`torch._foreach_erf`, https://pytorch.org/docs/stable/generated/torch._foreach_erf.html#torch-foreach-erf) |
| IN-DEVELOPMENT-PATTERN(`torch._foreach_erf_`, https://pytorch.org/docs/stable/generated/torch._foreach_erf_.html#torch-foreach-erf) |
| IN-DEVELOPMENT-PATTERN(`torch._foreach_erfc`, https://pytorch.org/docs/stable/generated/torch._foreach_erfc.html#torch-foreach-erfc) |
| IN-DEVELOPMENT-PATTERN(`torch._foreach_erfc_`, https://pytorch.org/docs/stable/generated/torch._foreach_erfc_.html#torch-foreach-erfc) |
| IN-DEVELOPMENT-PATTERN(`torch._foreach_exp`, https://pytorch.org/docs/stable/generated/torch._foreach_exp.html#torch-foreach-exp) |
| IN-DEVELOPMENT-PATTERN(`torch._foreach_exp_`, https://pytorch.org/docs/stable/generated/torch._foreach_exp_.html#torch-foreach-exp) |
| IN-DEVELOPMENT-PATTERN(`torch._foreach_expm1`, https://pytorch.org/docs/stable/generated/torch._foreach_expm1.html#torch-foreach-expm1) |
| IN-DEVELOPMENT-PATTERN(`torch._foreach_expm1_`, https://pytorch.org/docs/stable/generated/torch._foreach_expm1_.html#torch-foreach-expm1) |
| IN-DEVELOPMENT-PATTERN(`torch._foreach_floor`, https://pytorch.org/docs/stable/generated/torch._foreach_floor.html#torch-foreach-floor) |
| IN-DEVELOPMENT-PATTERN(`torch._foreach_floor_`, https://pytorch.org/docs/stable/generated/torch._foreach_floor_.html#torch-foreach-floor) |
| IN-DEVELOPMENT-PATTERN(`torch._foreach_log`, https://pytorch.org/docs/stable/generated/torch._foreach_log.html#torch-foreach-log) |
| IN-DEVELOPMENT-PATTERN(`torch._foreach_log_`, https://pytorch.org/docs/stable/generated/torch._foreach_log_.html#torch-foreach-log) |
| IN-DEVELOPMENT-PATTERN(`torch._foreach_log10`, https://pytorch.org/docs/stable/generated/torch._foreach_log10.html#torch-foreach-log10) |
| IN-DEVELOPMENT-PATTERN(`torch._foreach_log10_`, https://pytorch.org/docs/stable/generated/torch._foreach_log10_.html#torch-foreach-log10) |
| IN-DEVELOPMENT-PATTERN(`torch._foreach_log1p`, https://pytorch.org/docs/stable/generated/torch._foreach_log1p.html#torch-foreach-log1p) |
| IN-DEVELOPMENT-PATTERN(`torch._foreach_log1p_`, https://pytorch.org/docs/stable/generated/torch._foreach_log1p_.html#torch-foreach-log1p) |
| IN-DEVELOPMENT-PATTERN(`torch._foreach_log2`, https://pytorch.org/docs/stable/generated/torch._foreach_log2.html#torch-foreach-log2) |
| IN-DEVELOPMENT-PATTERN(`torch._foreach_log2_`, https://pytorch.org/docs/stable/generated/torch._foreach_log2_.html#torch-foreach-log2) |
| IN-DEVELOPMENT-PATTERN(`torch._foreach_neg`, https://pytorch.org/docs/stable/generated/torch._foreach_neg.html#torch-foreach-neg) |
| IN-DEVELOPMENT-PATTERN(`torch._foreach_neg_`, https://pytorch.org/docs/stable/generated/torch._foreach_neg_.html#torch-foreach-neg) |
| IN-DEVELOPMENT-PATTERN(`torch._foreach_tan`, https://pytorch.org/docs/stable/generated/torch._foreach_tan.html#torch-foreach-tan) |
| IN-DEVELOPMENT-PATTERN(`torch._foreach_tan_`, https://pytorch.org/docs/stable/generated/torch._foreach_tan_.html#torch-foreach-tan) |
| IN-DEVELOPMENT-PATTERN(`torch._foreach_sin`, https://pytorch.org/docs/stable/generated/torch._foreach_sin.html#torch-foreach-sin) |
| IN-DEVELOPMENT-PATTERN(`torch._foreach_sin_`, https://pytorch.org/docs/stable/generated/torch._foreach_sin_.html#torch-foreach-sin) |
| IN-DEVELOPMENT-PATTERN(`torch._foreach_sinh`, https://pytorch.org/docs/stable/generated/torch._foreach_sinh.html#torch-foreach-sinh) |
| IN-DEVELOPMENT-PATTERN(`torch._foreach_sinh_`, https://pytorch.org/docs/stable/generated/torch._foreach_sinh_.html#torch-foreach-sinh) |
| IN-DEVELOPMENT-PATTERN(`torch._foreach_round`, https://pytorch.org/docs/stable/generated/torch._foreach_round.html#torch-foreach-round) |
| IN-DEVELOPMENT-PATTERN(`torch._foreach_round_`, https://pytorch.org/docs/stable/generated/torch._foreach_round_.html#torch-foreach-round) |
| IN-DEVELOPMENT-PATTERN(`torch._foreach_sqrt`, https://pytorch.org/docs/stable/generated/torch._foreach_sqrt.html#torch-foreach-sqrt) |
| IN-DEVELOPMENT-PATTERN(`torch._foreach_sqrt_`, https://pytorch.org/docs/stable/generated/torch._foreach_sqrt_.html#torch-foreach-sqrt) |
| IN-DEVELOPMENT-PATTERN(`torch._foreach_lgamma`, https://pytorch.org/docs/stable/generated/torch._foreach_lgamma.html#torch-foreach-lgamma) |
| IN-DEVELOPMENT-PATTERN(`torch._foreach_lgamma_`, https://pytorch.org/docs/stable/generated/torch._foreach_lgamma_.html#torch-foreach-lgamma) |
| IN-DEVELOPMENT-PATTERN(`torch._foreach_frac`, https://pytorch.org/docs/stable/generated/torch._foreach_frac.html#torch-foreach-frac) |
| IN-DEVELOPMENT-PATTERN(`torch._foreach_frac_`, https://pytorch.org/docs/stable/generated/torch._foreach_frac_.html#torch-foreach-frac) |
| IN-DEVELOPMENT-PATTERN(`torch._foreach_reciprocal`, https://pytorch.org/docs/stable/generated/torch._foreach_reciprocal.html#torch-foreach-reciprocal) |
| IN-DEVELOPMENT-PATTERN(`torch._foreach_reciprocal_`, https://pytorch.org/docs/stable/generated/torch._foreach_reciprocal_.html#torch-foreach-reciprocal) |
| IN-DEVELOPMENT-PATTERN(`torch._foreach_sigmoid`, https://pytorch.org/docs/stable/generated/torch._foreach_sigmoid.html#torch-foreach-sigmoid) |
| IN-DEVELOPMENT-PATTERN(`torch._foreach_sigmoid_`, https://pytorch.org/docs/stable/generated/torch._foreach_sigmoid_.html#torch-foreach-sigmoid) |
| IN-DEVELOPMENT-PATTERN(`torch._foreach_trunc`, https://pytorch.org/docs/stable/generated/torch._foreach_trunc.html#torch-foreach-trunc) |
| IN-DEVELOPMENT-PATTERN(`torch._foreach_trunc_`, https://pytorch.org/docs/stable/generated/torch._foreach_trunc_.html#torch-foreach-trunc) |
| IN-DEVELOPMENT-PATTERN(`torch._foreach_zero_`, https://pytorch.org/docs/stable/generated/torch._foreach_zero_.html#torch-foreach-zero) |
| IN-DEVELOPMENT-PATTERN(`torch.Tensor.itemsize`, https://pytorch.org/docs/stable/generated/torch.Tensor.itemsize.html#torch-tensor-itemsize) |
| IN-DEVELOPMENT-PATTERN(`torch.Tensor.to_sparse_csr`, https://pytorch.org/docs/stable/generated/torch.Tensor.to_sparse_csr.html#torch-tensor-to-sparse-csr) |
| IN-DEVELOPMENT-PATTERN(`torch.autograd.graph.Node.name`, https://pytorch.org/docs/stable/generated/torch.autograd.graph.Node.name.html#torch-autograd-graph-node-name) |
| IN-DEVELOPMENT-PATTERN(`torch.autograd.graph.Node.metadata`, https://pytorch.org/docs/stable/generated/torch.autograd.graph.Node.metadata.html#torch-autograd-graph-node-metadata) |
| IN-DEVELOPMENT-PATTERN(`torch.autograd.graph.Node.next_functions`, https://pytorch.org/docs/stable/generated/torch.autograd.graph.Node.next_functions.html#torch-autograd-graph-node-next-functions) |
| IN-DEVELOPMENT-PATTERN(`torch.autograd.graph.Node.register_hook`, https://pytorch.org/docs/stable/generated/torch.autograd.graph.Node.register_hook.html#torch-autograd-graph-node-register-hook) |
| IN-DEVELOPMENT-PATTERN(`torch.autograd.graph.Node.register_prehook`, https://pytorch.org/docs/stable/generated/torch.autograd.graph.Node.register_prehook.html#torch-autograd-graph-node-register-prehook) |
| IN-DEVELOPMENT-PATTERN(`torch.cuda.OutOfMemoryError`, https://pytorch.org/docs/stable/generated/torch.cuda.OutOfMemoryError.html#torch-cuda-outofmemoryerror) |
| IN-DEVELOPMENT-PATTERN(`torch.backends.cpu.get_cpu_capability`, https://pytorch.org/docs/stable/backends.html#torch.backends.cpu.get_cpu_capability) |
| IN-DEVELOPMENT-PATTERN(`torch.distributed.P2POp`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.P2POp) |
| IN-DEVELOPMENT-PATTERN(`torch.distributed.is_gloo_available`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.is_gloo_available) |
| IN-DEVELOPMENT-PATTERN(`torch.distributed.get_group_rank`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.get_group_rank) |
| IN-DEVELOPMENT-PATTERN(`torch.distributed.get_global_rank`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.get_global_rank) |
| IN-DEVELOPMENT-PATTERN(`torch.distributed.get_process_group_ranks`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.get_process_group_ranks) |
| IN-DEVELOPMENT-PATTERN(`torch.distributed.batch_isend_irecv`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.batch_isend_irecv) |
| IN-DEVELOPMENT-PATTERN(`torch.distributed.all_gather_into_tensor`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_gather_into_tensor) |
| IN-DEVELOPMENT-PATTERN(`torch.distributed.reduce_scatter_tensor`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.reduce_scatter_tensor) |
| IN-DEVELOPMENT-PATTERN(`torch.distributed.all_to_all_single`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single) |
| IN-DEVELOPMENT-PATTERN(`torch.utils.set_module`, https://pytorch.org/docs/stable/generated/torch.utils.set_module.html#torch-utils-set-module) |
| IN-DEVELOPMENT-PATTERN(`torch.get_default_device`, https://pytorch.org/docs/stable/generated/torch.get_default_device.html#torch-get-default-device) |
| IN-DEVELOPMENT-PATTERN(`torch.nn.utils.fuse_conv_bn_eval`, https://pytorch.org/docs/stable/generated/torch.nn.utils.fuse_conv_bn_eval.html#torch-nn-utils-fuse-conv-bn-eval) |
| IN-DEVELOPMENT-PATTERN(`torch.nn.utils.fuse_conv_bn_weights`, https://pytorch.org/docs/stable/generated/torch.nn.utils.fuse_conv_bn_weights.html#torch-nn-utils-fuse-conv-bn-weights) |
| IN-DEVELOPMENT-PATTERN(`torch.nn.utils.fuse_linear_bn_eval`, https://pytorch.org/docs/stable/generated/torch.nn.utils.fuse_linear_bn_eval.html#torch-nn-utils-fuse-linear-bn-eval) |
| IN-DEVELOPMENT-PATTERN(`torch.nn.utils.fuse_linear_bn_weights`, https://pytorch.org/docs/stable/generated/torch.nn.utils.fuse_linear_bn_weights.html#torch-nn-utils-fuse-linear-bn-weights) |
| IN-DEVELOPMENT-PATTERN(`torch.nn.utils.convert_conv2d_weight_memory_format`, https://pytorch.org/docs/stable/generated/torch.nn.utils.convert_conv2d_weight_memory_format.html#torch-nn-utils-convert-conv2d-weight-memory-format) |
| IN-DEVELOPMENT-PATTERN(`torch.nn.utils.convert_conv3d_weight_memory_format`, https://pytorch.org/docs/stable/generated/torch.nn.utils.convert_conv3d_weight_memory_format.html#torch-nn-utils-convert-conv3d-weight-memory-format) |
| IN-DEVELOPMENT-PATTERN(`torch.backends.cuda.can_use_flash_attention`, https://pytorch.org/docs/stable/backends.html#torch.backends.cuda.can_use_flash_attention) |
| IN-DEVELOPMENT-PATTERN(`torch.distributed.device_mesh.DeviceMesh`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.device_mesh.DeviceMesh) |
| IN-DEVELOPMENT-PATTERN(`torch.cuda.is_initialized`, https://pytorch.org/docs/stable/generated/torch.cuda.is_initialized.html#torch-cuda-is-initialized) |
| IN-DEVELOPMENT-PATTERN(`torch.cuda.comm.scatter`, https://pytorch.org/docs/stable/generated/torch.cuda.comm.scatter.html#torch-cuda-comm-scatter) |
| IN-DEVELOPMENT-PATTERN(`torch.cuda.comm.gather`, https://pytorch.org/docs/stable/generated/torch.cuda.comm.gather.html#torch-cuda-comm-gather) |
| IN-DEVELOPMENT-PATTERN(`torch.distributions.binomial.Binomial`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.binomial.Binomial) |
| IN-DEVELOPMENT-PATTERN(`torch.distributions.continuous_bernoulli.ContinuousBernoulli`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.continuous_bernoulli.ContinuousBernoulli) |
| IN-DEVELOPMENT-PATTERN(`torch.distributions.exponential.Exponential`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.exponential.Exponential) |
| IN-DEVELOPMENT-PATTERN(`torch.nn.LazyBatchNorm1d`, https://pytorch.org/docs/stable/generated/torch.nn.LazyBatchNorm1d.html#torch.nn.LazyBatchNorm1d) |
| IN-DEVELOPMENT-PATTERN(`torch.cuda.StreamContext`, https://pytorch.org/docs/stable/generated/torch.cuda.StreamContext.html#torch.cuda.StreamContext) |
| IN-DEVELOPMENT-PATTERN(`torch.nn.LazyBatchNorm2d`, https://pytorch.org/docs/stable/generated/torch.nn.LazyBatchNorm2d.html#torch.nn.LazyBatchNorm2d) |
| IN-DEVELOPMENT-PATTERN(`torch.nn.LazyBatchNorm3d`, https://pytorch.org/docs/stable/generated/torch.nn.LazyBatchNorm3d.html#torch.nn.LazyBatchNorm3d) |
| IN-DEVELOPMENT-PATTERN(`torch.Tensor.bitwise_right_shift`, https://pytorch.org/docs/stable/generated/torch.Tensor.bitwise_right_shift.html#torch-tensor-bitwise-right-shift) |
| IN-DEVELOPMENT-PATTERN(`torch.nn.LazyInstanceNorm2d`, https://pytorch.org/docs/stable/generated/torch.nn.LazyInstanceNorm2d.html#torch.nn.LazyInstanceNorm2d) |
| IN-DEVELOPMENT-PATTERN(`torch.nn.LazyInstanceNorm1d`, https://pytorch.org/docs/stable/generated/torch.nn.LazyInstanceNorm1d.html#torch.nn.LazyInstanceNorm1d) |
| IN-DEVELOPMENT-PATTERN(`torch.nn.LazyInstanceNorm3d`, https://pytorch.org/docs/stable/generated/torch.nn.LazyInstanceNorm3d.html#torch.nn.LazyInstanceNorm3d) |
| IN-DEVELOPMENT-PATTERN(`torch.set_deterministic_debug_mode`, https://pytorch.org/docs/stable/generated/torch.set_deterministic_debug_mode.html#torch-set-deterministic-debug-mode) |
| IN-DEVELOPMENT-PATTERN(`torch.get_deterministic_debug_mode`, https://pytorch.org/docs/stable/generated/torch.get_deterministic_debug_mode.html#torch-get-deterministic-debug-mode) |
