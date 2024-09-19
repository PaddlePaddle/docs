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
| NOT-IMPLEMENTED-ITEM(`torch.geqrf`, https://pytorch.org/docs/stable/generated/torch.geqrf.html?highlight=geqrf#torch.geqrf) |
| NOT-IMPLEMENTED-ITEM(`torch.gradient`, https://pytorch.org/docs/stable/generated/torch.gradient.html#torch.gradient) |
| NOT-IMPLEMENTED-ITEM(`torch.is_conj`, https://pytorch.org/docs/stable/generated/torch.is_conj.html#torch.is_conj) |
| NOT-IMPLEMENTED-ITEM(`torch.layout`, https://pytorch.org/docs/stable/tensor_attributes.html#torch.layout) |
| NOT-IMPLEMENTED-ITEM(`torch.resolve_conj`, https://pytorch.org/docs/stable/generated/torch.resolve_conj.html#torch.resolve_conj) |
| NOT-IMPLEMENTED-ITEM(`torch.resolve_neg`, https://pytorch.org/docs/stable/generated/torch.resolve_neg.html#torch.resolve_neg) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.functional.multilabel_margin_loss`, https://pytorch.org/docs/stable/generated/torch.nn.functional.multilabel_margin_loss.html#torch.nn.functional.multilabel_margin_loss) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.dequantize`, https://pytorch.org/docs/1.13/generated/torch.Tensor.dequantize.html?highlight=torch+tensor+dequantize#torch.Tensor.dequantize) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.geqrf`, https://pytorch.org/docs/1.13/generated/torch.Tensor.geqrf.html?highlight=torch+tensor+geqrf#torch.Tensor.geqrf) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.is_coalesced`, https://pytorch.org/docs/stable/generated/torch.Tensor.is_coalesced.html#torch.Tensor.is_coalesced) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.is_conj`, https://pytorch.org/docs/stable/generated/torch.Tensor.is_conj.html#torch.Tensor.is_conj) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.resize_`, https://pytorch.org/docs/stable/generated/torch.Tensor.resize_.html?highlight=resize#torch.Tensor.resize_) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.resolve_conj`, https://pytorch.org/docs/stable/generated/torch.Tensor.resolve_conj.html#torch.Tensor.resolve_conj) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.resolve_neg`, https://pytorch.org/docs/stable/generated/torch.Tensor.resolve_neg.html#torch.Tensor.resolve_neg) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.sparse_resize_`, https://pytorch.org/docs/stable/generated/torch.Tensor.sparse_resize_.html#torch.Tensor.sparse_resize_) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.sparse_resize_and_clear_`, https://pytorch.org/docs/stable/generated/torch.Tensor.sparse_resize_and_clear_.html#torch.Tensor.sparse_resize_and_clear_) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.sum_to_size`, https://pytorch.org/docs/stable/generated/torch.Tensor.sum_to_size.html?highlight=sum_to_size#torch.Tensor.sum_to_size) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.parametrize.is_parametrized`, https://pytorch.org/docs/stable/generated/torch.nn.utils.parametrize.is_parametrized.html#torch.nn.utils.parametrize.is_parametrized) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.Module.register_full_backward_hook`, https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_full_backward_hook) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.Module.register_full_backward_pre_hook`, https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_full_backward_pre_hook) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.Module.requires_grad_`, https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.requires_grad_) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.Module.to_empty`, https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.to_empty) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.function.FunctionCtx.mark_dirty`, https://pytorch.org/docs/stable/generated/torch.autograd.function.FunctionCtx.mark_dirty.html#torch.autograd.function.FunctionCtx.mark_dirty) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.profiler.profile.self_cpu_time_total`, https://pytorch.org/docs/stable/generated/torch.autograd.profiler.profile.self_cpu_time_total.html#torch.autograd.profiler.profile.self_cpu_time_total) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.default_stream`, https://pytorch.org/docs/stable/generated/torch.cuda.default_stream.html?highlight=torch+cuda+default_stream#torch.cuda.default_stream) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.get_arch_list`, https://pytorch.org/docs/stable/generated/torch.cuda.get_arch_list.html?highlight=torch+cuda+get_arch_list#torch.cuda.get_arch_list) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.is_current_stream_capturing`, https://pytorch.org/docs/stable/generated/torch.cuda.is_current_stream_capturing.html#torch.cuda.is_current_stream_capturing) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.mem_get_info`, https://pytorch.org/docs/stable/generated/torch.cuda.mem_get_info.html#torch.cuda.mem_get_info) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.memory_usage`, https://pytorch.org/docs/stable/generated/torch.cuda.memory_usage.html#torch.cuda.memory_usage) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.set_rng_state`, https://pytorch.org/docs/stable/generated/torch.cuda.set_rng_state.html#torch.cuda.set_rng_state) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.all_gather_multigpu`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_gather_multigpu) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.all_reduce_multigpu`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_reduce_multigpu) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.broadcast_multigpu`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.broadcast_multigpu) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.reduce_multigpu`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.reduce_multigpu) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.reduce_scatter_multigpu`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.reduce_scatter_multigpu) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.rpc.WorkerInfo`, https://pytorch.org/docs/stable/rpc.html#torch.distributed.rpc.WorkerInfo) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.rpc.functions.async_execution`, https://pytorch.org/docs/stable/rpc.html#torch.distributed.rpc.functions.async_execution) |
| NOT-IMPLEMENTED-ITEM(`torch.distributions.constraint_registry.ConstraintRegistry`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.constraint_registry.ConstraintRegistry) |
| NOT-IMPLEMENTED-ITEM(`torch.distributions.one_hot_categorical.OneHotCategorical`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.one_hot_categorical.OneHotCategorical) |
| NOT-IMPLEMENTED-ITEM(`torch.distributions.transforms.CatTransform`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.transforms.CatTransform) |
| NOT-IMPLEMENTED-ITEM(`torch.distributions.transforms.CumulativeDistributionTransform`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.transforms.CumulativeDistributionTransform) |
| NOT-IMPLEMENTED-ITEM(`torch.distributions.transforms.SoftplusTransform`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.transforms.SoftplusTransform) |
| NOT-IMPLEMENTED-ITEM(`torch.hub.get_dir`, https://pytorch.org/docs/stable/hub.html?highlight=torch+hub+get_dir#torch.hub.get_dir) |
| NOT-IMPLEMENTED-ITEM(`torch.hub.set_dir`, https://pytorch.org/docs/stable/hub.html?highlight=torch+hub+get_dir#torch.hub.set_dir) |
| NOT-IMPLEMENTED-ITEM(`torch.onnx.disable_log`, https://pytorch.org/docs/stable/onnx.html#torch.onnx.disable_log) |
| NOT-IMPLEMENTED-ITEM(`torch.onnx.enable_log`, https://pytorch.org/docs/stable/onnx.html#torch.onnx.enable_log) |
| NOT-IMPLEMENTED-ITEM(`torch.profiler.ProfilerAction`, https://pytorch.org/docs/stable/profiler.html#torch.profiler.ProfilerAction) |
| NOT-IMPLEMENTED-ITEM(`torch.profiler.ProfilerActivity`, https://pytorch.org/docs/stable/profiler.html#torch.profiler.ProfilerActivity) |
| NOT-IMPLEMENTED-ITEM(`torch.sparse.sampled_addmm`, https://pytorch.org/docs/stable/generated/torch.sparse.sampled_addmm.html#torch.sparse.sampled_addmm) |
| NOT-IMPLEMENTED-ITEM(`torch.special.entr`, https://pytorch.org/docs/stable/special.html#torch.special.entr) |
| NOT-IMPLEMENTED-ITEM(`torch.utils.cpp_extension.include_paths`, https://pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.include_paths) |
| NOT-IMPLEMENTED-ITEM(`torch.utils.cpp_extension.load_inline`, https://pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.load_inline) |

## 开发中的 API 列表

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ----- | ----------- | ----------------- | ----------- | ------- |
| IN-DEVELOPMENT-PATTERN(`torch.get_num_threads`, https://pytorch.org/docs/stable/generated/torch.get_num_threads.html#torch-get-num-threads) |
| IN-DEVELOPMENT-PATTERN(`torch.set_num_threads`, https://pytorch.org/docs/stable/generated/torch.set_num_threads.html#torch-set-num-threads) |
| IN-DEVELOPMENT-PATTERN(`torch.get_num_interop_threads`, https://pytorch.org/docs/stable/generated/torch.get_num_interop_threads.html#torch-get-num-interop-threads) |
| IN-DEVELOPMENT-PATTERN(`torch.set_num_interop_threads`, https://pytorch.org/docs/stable/generated/torch.set_num_interop_threads.html#torch-set-num-interop-threads) |
| IN-DEVELOPMENT-PATTERN(`torch.float_power`, https://pytorch.org/docs/stable/generated/torch.float_power.html#torch-float-power) |
| IN-DEVELOPMENT-PATTERN(`torch.igamma`, https://pytorch.org/docs/stable/generated/torch.igamma.html#torch-igamma) |
| IN-DEVELOPMENT-PATTERN(`torch.igammac`, https://pytorch.org/docs/stable/generated/torch.igammac.html#torch-igammac) |
| IN-DEVELOPMENT-PATTERN(`torch.mvlgamma`, https://pytorch.org/docs/stable/generated/torch.mvlgamma.html#torch-mvlgamma) |
| IN-DEVELOPMENT-PATTERN(`torch.isneginf`, https://pytorch.org/docs/stable/generated/torch.isneginf.html#torch-isneginf) |
| IN-DEVELOPMENT-PATTERN(`torch.isreal`, https://pytorch.org/docs/stable/generated/torch.isreal.html#torch-isreal) |
| IN-DEVELOPMENT-PATTERN(`torch.blackman_window`, https://pytorch.org/docs/stable/generated/torch.blackman_window.html#torch-blackman-window) |
| IN-DEVELOPMENT-PATTERN(`torch.hamming_window`, https://pytorch.org/docs/stable/generated/torch.hamming_window.html#torch-hamming-window) |
| IN-DEVELOPMENT-PATTERN(`torch.hann_window`, https://pytorch.org/docs/stable/generated/torch.hann_window.html#torch-hann-window) |
| IN-DEVELOPMENT-PATTERN(`torch.block_diag`, https://pytorch.org/docs/stable/generated/torch.block_diag.html#torch-block-diag) |
| IN-DEVELOPMENT-PATTERN(`torch.cartesian_prod`, https://pytorch.org/docs/stable/generated/torch.cartesian_prod.html#torch-cartesian-prod) |
| IN-DEVELOPMENT-PATTERN(`torch.ormqr`, https://pytorch.org/docs/stable/generated/torch.ormqr.html#torch-ormqr) |
| IN-DEVELOPMENT-PATTERN(`torch._assert`, https://pytorch.org/docs/stable/generated/torch._assert.html#torch-assert) |
| IN-DEVELOPMENT-PATTERN(`torch.nn.parameter.UninitializedParameter`, https://pytorch.org/docs/stable/generated/torch.nn.parameter.UninitializedParameter.html#torch.nn.parameter.UninitializedParameter) |
| IN-DEVELOPMENT-PATTERN(`torch.nn.modules.module.register_module_forward_pre_hook`, https://pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_forward_pre_hook.html#torch-nn-modules-module-register-module-forward-pre-hook) |
| IN-DEVELOPMENT-PATTERN(`torch.nn.modules.module.register_module_forward_hook`, https://pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_forward_hook.html#torch-nn-modules-module-register-module-forward-hook) |
| IN-DEVELOPMENT-PATTERN(`torch.nn.LazyConv1d`, https://pytorch.org/docs/stable/generated/torch.nn.LazyConv1d.html#torch.nn.LazyConv1d) |
| IN-DEVELOPMENT-PATTERN(`torch.nn.LazyConv2d`, https://pytorch.org/docs/stable/generated/torch.nn.LazyConv2d.html#torch.nn.LazyConv2d) |
| IN-DEVELOPMENT-PATTERN(`torch.nn.LazyConv3d`, https://pytorch.org/docs/stable/generated/torch.nn.LazyConv3d.html#torch.nn.LazyConv3d) |
| IN-DEVELOPMENT-PATTERN(`torch.nn.LazyConvTranspose1d`, https://pytorch.org/docs/stable/generated/torch.nn.LazyConvTranspose1d.html#torch.nn.LazyConvTranspose1d) |
| IN-DEVELOPMENT-PATTERN(`torch.nn.LazyConvTranspose2d`, https://pytorch.org/docs/stable/generated/torch.nn.LazyConvTranspose2d.html#torch.nn.LazyConvTranspose2d) |
| IN-DEVELOPMENT-PATTERN(`torch.nn.LazyConvTranspose3d`, https://pytorch.org/docs/stable/generated/torch.nn.LazyConvTranspose3d.html#torch.nn.LazyConvTranspose3d) |
| IN-DEVELOPMENT-PATTERN(`torch.nn.LPPool1d`, https://pytorch.org/docs/stable/generated/torch.nn.LPPool1d.html#torch.nn.LPPool1d) |
| IN-DEVELOPMENT-PATTERN(`torch.nn.LPPool2d`, https://pytorch.org/docs/stable/generated/torch.nn.LPPool2d.html#torch.nn.LPPool2d) |
| IN-DEVELOPMENT-PATTERN(`torch.nn.Softmin`, https://pytorch.org/docs/stable/generated/torch.nn.Softmin.html#torch.nn.Softmin) |
| IN-DEVELOPMENT-PATTERN(`torch.nn.AdaptiveLogSoftmaxWithLoss`, https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveLogSoftmaxWithLoss.html#torch.nn.AdaptiveLogSoftmaxWithLoss) |
| IN-DEVELOPMENT-PATTERN(`torch.nn.LazyLinear`, https://pytorch.org/docs/stable/generated/torch.nn.LazyLinear.html#torch.nn.LazyLinear) |
| IN-DEVELOPMENT-PATTERN(`torch.nn.functional.lp_pool1d`, https://pytorch.org/docs/stable/generated/torch.nn.functional.lp_pool1d.html#torch-nn-functional-lp-pool1d) |
| IN-DEVELOPMENT-PATTERN(`torch.nn.functional.lp_pool2d`, https://pytorch.org/docs/stable/generated/torch.nn.functional.lp_pool2d.html#torch-nn-functional-lp-pool2d) |
| IN-DEVELOPMENT-PATTERN(`torch.nn.functional.threshold_`, https://pytorch.org/docs/stable/generated/torch.nn.functional.threshold_.html#torch-nn-functional-threshold) |
| IN-DEVELOPMENT-PATTERN(`torch.Tensor.requires_grad`, https://pytorch.org/docs/stable/generated/torch.Tensor.requires_grad.html#torch-tensor-requires-grad) |
| IN-DEVELOPMENT-PATTERN(`torch.Tensor.addbmm_`, https://pytorch.org/docs/stable/generated/torch.Tensor.addbmm_.html#torch-tensor-addbmm) |
| IN-DEVELOPMENT-PATTERN(`torch.Tensor.addcdiv_`, https://pytorch.org/docs/stable/generated/torch.Tensor.addcdiv_.html#torch-tensor-addcdiv) |
| IN-DEVELOPMENT-PATTERN(`torch.Tensor.addcmul_`, https://pytorch.org/docs/stable/generated/torch.Tensor.addcmul_.html#torch-tensor-addcmul) |
| IN-DEVELOPMENT-PATTERN(`torch.Tensor.addmv_`, https://pytorch.org/docs/stable/generated/torch.Tensor.addmv_.html#torch-tensor-addmv) |
| IN-DEVELOPMENT-PATTERN(`torch.Tensor.addr_`, https://pytorch.org/docs/stable/generated/torch.Tensor.addr_.html#torch-tensor-addr) |
| IN-DEVELOPMENT-PATTERN(`torch.Tensor.baddbmm_`, https://pytorch.org/docs/stable/generated/torch.Tensor.baddbmm_.html#torch-tensor-baddbmm) |
| IN-DEVELOPMENT-PATTERN(`torch.Tensor.cauchy_`, https://pytorch.org/docs/stable/generated/torch.Tensor.cauchy_.html#torch-tensor-cauchy) |
| IN-DEVELOPMENT-PATTERN(`torch.Tensor.copysign_`, https://pytorch.org/docs/stable/generated/torch.Tensor.copysign_.html#torch-tensor-copysign) |
| IN-DEVELOPMENT-PATTERN(`torch.Tensor.data_ptr`, https://pytorch.org/docs/stable/generated/torch.Tensor.data_ptr.html#torch-tensor-data-ptr) |
| IN-DEVELOPMENT-PATTERN(`torch.Tensor.erfc_`, https://pytorch.org/docs/stable/generated/torch.Tensor.erfc_.html#torch-tensor-erfc) |
| IN-DEVELOPMENT-PATTERN(`torch.Tensor.fix_`, https://pytorch.org/docs/stable/generated/torch.Tensor.fix_.html#torch-tensor-fix) |
| IN-DEVELOPMENT-PATTERN(`torch.Tensor.fmod_`, https://pytorch.org/docs/stable/generated/torch.Tensor.fmod_.html#torch-tensor-fmod) |
| IN-DEVELOPMENT-PATTERN(`torch.Tensor.geometric_`, https://pytorch.org/docs/stable/generated/torch.Tensor.geometric_.html#torch-tensor-geometric) |
| IN-DEVELOPMENT-PATTERN(`torch.Tensor.igamma`, https://pytorch.org/docs/stable/generated/torch.Tensor.igamma.html#torch-tensor-igamma) |
| IN-DEVELOPMENT-PATTERN(`torch.Tensor.igamma_`, https://pytorch.org/docs/stable/generated/torch.Tensor.igamma_.html#torch-tensor-igamma) |
| IN-DEVELOPMENT-PATTERN(`torch.Tensor.igammac`, https://pytorch.org/docs/stable/generated/torch.Tensor.igammac.html#torch-tensor-igammac) |
| IN-DEVELOPMENT-PATTERN(`torch.Tensor.igammac_`, https://pytorch.org/docs/stable/generated/torch.Tensor.igammac_.html#torch-tensor-igammac) |
| IN-DEVELOPMENT-PATTERN(`torch.Tensor.int_repr`, https://pytorch.org/docs/stable/generated/torch.Tensor.int_repr.html#torch-tensor-int-repr) |
| IN-DEVELOPMENT-PATTERN(`torch.Tensor.isposinf`, https://pytorch.org/docs/stable/generated/torch.Tensor.isposinf.html#torch-tensor-isposinf) |
| IN-DEVELOPMENT-PATTERN(`torch.Tensor.isneginf`, https://pytorch.org/docs/stable/generated/torch.Tensor.isneginf.html#torch-tensor-isneginf) |
| IN-DEVELOPMENT-PATTERN(`torch.Tensor.isreal`, https://pytorch.org/docs/stable/generated/torch.Tensor.isreal.html#torch-tensor-isreal) |
| IN-DEVELOPMENT-PATTERN(`torch.Tensor.matrix_exp`, https://pytorch.org/docs/stable/generated/torch.Tensor.matrix_exp.html#torch-tensor-matrix-exp) |
| IN-DEVELOPMENT-PATTERN(`torch.Tensor.mvlgamma`, https://pytorch.org/docs/stable/generated/torch.Tensor.mvlgamma.html#torch-tensor-mvlgamma) |
| IN-DEVELOPMENT-PATTERN(`torch.Tensor.mvlgamma_`, https://pytorch.org/docs/stable/generated/torch.Tensor.mvlgamma_.html#torch-tensor-mvlgamma) |
| IN-DEVELOPMENT-PATTERN(`torch.Tensor.orgqr`, https://pytorch.org/docs/stable/generated/torch.Tensor.orgqr.html#torch-tensor-orgqr) |
| IN-DEVELOPMENT-PATTERN(`torch.Tensor.ormqr`, https://pytorch.org/docs/stable/generated/torch.Tensor.ormqr.html#torch-tensor-ormqr) |
| IN-DEVELOPMENT-PATTERN(`torch.Tensor.random_`, https://pytorch.org/docs/stable/generated/torch.Tensor.random_.html#torch-tensor-random) |
| IN-DEVELOPMENT-PATTERN(`torch.Tensor.sinc_`, https://pytorch.org/docs/stable/generated/torch.Tensor.sinc_.html#torch-tensor-sinc) |
| IN-DEVELOPMENT-PATTERN(`torch.Tensor.stride`, https://pytorch.org/docs/stable/generated/torch.Tensor.stride.html#torch-tensor-stride) |
| IN-DEVELOPMENT-PATTERN(`torch.Tensor.t_`, https://pytorch.org/docs/stable/generated/torch.Tensor.t_.html#torch-tensor-t) |
| IN-DEVELOPMENT-PATTERN(`torch.Tensor.transpose_`, https://pytorch.org/docs/stable/generated/torch.Tensor.transpose_.html#torch-tensor-transpose) |
| IN-DEVELOPMENT-PATTERN(`torch.Tensor.xlogy_`, https://pytorch.org/docs/stable/generated/torch.Tensor.xlogy_.html#torch-tensor-xlogy) |
| IN-DEVELOPMENT-PATTERN(`torch.cuda.device_of`, https://pytorch.org/docs/stable/generated/torch.cuda.device_of.html#torch.cuda.device_of) |
| IN-DEVELOPMENT-PATTERN(`torch.cuda.get_rng_state`, https://pytorch.org/docs/stable/generated/torch.cuda.get_rng_state.html#torch-cuda-get-rng-state) |
| IN-DEVELOPMENT-PATTERN(`torch.cuda.set_per_process_memory_fraction`, https://pytorch.org/docs/stable/generated/torch.cuda.set_per_process_memory_fraction.html#torch-cuda-set-per-process-memory-fraction) |
| IN-DEVELOPMENT-PATTERN(`torch.distributed.rpc.remote`, https://pytorch.org/docs/stable/rpc.html#torch.distributed.rpc.remote) |
| IN-DEVELOPMENT-PATTERN(`torch.distributed.Backend`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.Backend) |
| IN-DEVELOPMENT-PATTERN(`torch.distributed.is_available`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.is_available) |
| IN-DEVELOPMENT-PATTERN(`torch.distributed.is_nccl_available`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.is_nccl_available) |
| IN-DEVELOPMENT-PATTERN(`torch.distributed.gather_object`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.gather_object) |
| IN-DEVELOPMENT-PATTERN(`torch.distributions.chi2.Chi2`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.chi2.Chi2) |
| IN-DEVELOPMENT-PATTERN(`torch.distributions.gamma.Gamma`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.gamma.Gamma) |
| IN-DEVELOPMENT-PATTERN(`torch.distributions.lkj_cholesky.LKJCholesky`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.lkj_cholesky.LKJCholesky) |
| IN-DEVELOPMENT-PATTERN(`torch.distributions.multivariate_normal.MultivariateNormal`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.multivariate_normal.MultivariateNormal) |
| IN-DEVELOPMENT-PATTERN(`torch.distributions.poisson.Poisson`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.poisson.Poisson) |
| IN-DEVELOPMENT-PATTERN(`torch.distributions.studentT.StudentT`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.studentT.StudentT) |
| IN-DEVELOPMENT-PATTERN(`torch.jit.script`, https://pytorch.org/docs/stable/generated/torch.jit.script.html#torch-jit-script) |
| IN-DEVELOPMENT-PATTERN(`torch.jit.trace`, https://pytorch.org/docs/stable/generated/torch.jit.trace.html#torch-jit-trace) |
| IN-DEVELOPMENT-PATTERN(`torch.jit.save`, https://pytorch.org/docs/stable/generated/torch.jit.save.html#torch-jit-save) |
| IN-DEVELOPMENT-PATTERN(`torch.jit.ignore`, https://pytorch.org/docs/stable/generated/torch.jit.ignore.html#torch-jit-ignore) |
| IN-DEVELOPMENT-PATTERN(`torch.jit.unused`, https://pytorch.org/docs/stable/generated/torch.jit.unused.html#torch-jit-unused) |
| IN-DEVELOPMENT-PATTERN(`torch.distributed.optim.DistributedOptimizer`, https://pytorch.org/docs/stable/distributed.optim.html#torch.distributed.optim.DistributedOptimizer) |
| IN-DEVELOPMENT-PATTERN(`torch.utils.checkpoint.checkpoint`, https://pytorch.org/docs/stable/checkpoint.html#torch.utils.checkpoint.checkpoint) |
| IN-DEVELOPMENT-PATTERN(`torch.utils.checkpoint.checkpoint_sequential`, https://pytorch.org/docs/stable/checkpoint.html#torch.utils.checkpoint.checkpoint_sequential) |
| IN-DEVELOPMENT-PATTERN(`torch.utils.tensorboard.writer.SummaryWriter`, https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter) |
| IN-DEVELOPMENT-PATTERN(`torch.Tensor.positive`, https://pytorch.org/docs/stable/generated/torch.Tensor.positive.html#torch-tensor-positive) |
| IN-DEVELOPMENT-PATTERN(`torch.nn.parameter.UninitializedBuffer`, https://pytorch.org/docs/stable/generated/torch.nn.parameter.UninitializedBuffer.html#torch.nn.parameter.UninitializedBuffer) |
| IN-DEVELOPMENT-PATTERN(`torch.optim.Optimizer.zero_grad`, https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html#torch-optim-optimizer-zero-grad) |
| IN-DEVELOPMENT-PATTERN(`torch.optim.RAdam`, https://pytorch.org/docs/stable/generated/torch.optim.RAdam.html#torch.optim.RAdam) |
| IN-DEVELOPMENT-PATTERN(`torch.special.gammaincc`, https://pytorch.org/docs/stable/special.html#torch.special.gammaincc) |
| IN-DEVELOPMENT-PATTERN(`torch.testing.make_tensor`, https://pytorch.org/docs/stable/testing.html#torch.testing.make_tensor) |
| IN-DEVELOPMENT-PATTERN(`torch.special.ndtr`, https://pytorch.org/docs/stable/special.html#torch.special.ndtr) |
| IN-DEVELOPMENT-PATTERN(`torch.optim.NAdam`, https://pytorch.org/docs/stable/generated/torch.optim.NAdam.html#torch.optim.NAdam) |
| IN-DEVELOPMENT-PATTERN(`torch.Tensor.is_inference`, https://pytorch.org/docs/stable/generated/torch.Tensor.is_inference.html#torch-tensor-is-inference) |
| IN-DEVELOPMENT-PATTERN(`torch.special.gammainc`, https://pytorch.org/docs/stable/special.html#torch.special.gammainc) |
| IN-DEVELOPMENT-PATTERN(`torch.distributed.monitored_barrier`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.monitored_barrier) |
| IN-DEVELOPMENT-PATTERN(`torch.frombuffer`, https://pytorch.org/docs/stable/generated/torch.frombuffer.html#torch-frombuffer) |
| IN-DEVELOPMENT-PATTERN(`torch.Tensor.bitwise_left_shift_`, https://pytorch.org/docs/stable/generated/torch.Tensor.bitwise_left_shift_.html#torch-tensor-bitwise-left-shift) |
| IN-DEVELOPMENT-PATTERN(`torch.Tensor.bitwise_right_shift_`, https://pytorch.org/docs/stable/generated/torch.Tensor.bitwise_right_shift_.html#torch-tensor-bitwise-right-shift) |
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
| IN-DEVELOPMENT-PATTERN(`torch.nn.functional.scaled_dot_product_attention`, https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html#torch-nn-functional-scaled-dot-product-attention) |
| IN-DEVELOPMENT-PATTERN(`torch.Tensor.nbytes`, https://pytorch.org/docs/stable/generated/torch.Tensor.nbytes.html#torch-tensor-nbytes) |
| IN-DEVELOPMENT-PATTERN(`torch.Tensor.itemsize`, https://pytorch.org/docs/stable/generated/torch.Tensor.itemsize.html#torch-tensor-itemsize) |
| IN-DEVELOPMENT-PATTERN(`torch.Tensor.dim_order`, https://pytorch.org/docs/stable/generated/torch.Tensor.dim_order.html#torch-tensor-dim-order) |
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
| IN-DEVELOPMENT-PATTERN(`torch.distributions.transforms.PositiveDefiniteTransform`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.transforms.PositiveDefiniteTransform) |
| IN-DEVELOPMENT-PATTERN(`torch.signal.windows.blackman`, https://pytorch.org/docs/stable/generated/torch.signal.windows.blackman.html#torch-signal-windows-blackman) |
| IN-DEVELOPMENT-PATTERN(`torch.signal.windows.cosine`, https://pytorch.org/docs/stable/generated/torch.signal.windows.cosine.html#torch-signal-windows-cosine) |
| IN-DEVELOPMENT-PATTERN(`torch.signal.windows.exponential`, https://pytorch.org/docs/stable/generated/torch.signal.windows.exponential.html#torch-signal-windows-exponential) |
| IN-DEVELOPMENT-PATTERN(`torch.signal.windows.gaussian`, https://pytorch.org/docs/stable/generated/torch.signal.windows.gaussian.html#torch-signal-windows-gaussian) |
| IN-DEVELOPMENT-PATTERN(`torch.signal.windows.general_cosine`, https://pytorch.org/docs/stable/generated/torch.signal.windows.general_cosine.html#torch-signal-windows-general-cosine) |
| IN-DEVELOPMENT-PATTERN(`torch.signal.windows.general_hamming`, https://pytorch.org/docs/stable/generated/torch.signal.windows.general_hamming.html#torch-signal-windows-general-hamming) |
| IN-DEVELOPMENT-PATTERN(`torch.signal.windows.hamming`, https://pytorch.org/docs/stable/generated/torch.signal.windows.hamming.html#torch-signal-windows-hamming) |
| IN-DEVELOPMENT-PATTERN(`torch.signal.windows.hann`, https://pytorch.org/docs/stable/generated/torch.signal.windows.hann.html#torch-signal-windows-hann) |
| IN-DEVELOPMENT-PATTERN(`torch.Tensor.is_sparse_csr`, https://pytorch.org/docs/stable/generated/torch.Tensor.is_sparse_csr.html#torch-tensor-is-sparse-csr) |
| IN-DEVELOPMENT-PATTERN(`torch.Tensor.to_sparse_coo`, https://pytorch.org/docs/stable/generated/torch.Tensor.to_sparse_coo.html#torch-tensor-to-sparse-coo) |
| IN-DEVELOPMENT-PATTERN(`torch.Tensor.crow_indices`, https://pytorch.org/docs/stable/generated/torch.Tensor.crow_indices.html#torch-tensor-crow-indices) |
| IN-DEVELOPMENT-PATTERN(`torch.Tensor.col_indices`, https://pytorch.org/docs/stable/generated/torch.Tensor.col_indices.html#torch-tensor-col-indices) |
| IN-DEVELOPMENT-PATTERN(`torch.utils.set_module`, https://pytorch.org/docs/stable/generated/torch.utils.set_module.html#torch-utils-set-module) |
| IN-DEVELOPMENT-PATTERN(`torch.get_default_device`, https://pytorch.org/docs/stable/generated/torch.get_default_device.html#torch-get-default-device) |
| IN-DEVELOPMENT-PATTERN(`torch.nn.CircularPad3d`, https://pytorch.org/docs/stable/generated/torch.nn.CircularPad3d.html#torch.nn.CircularPad3d) |
| IN-DEVELOPMENT-PATTERN(`torch.nn.utils.fuse_conv_bn_eval`, https://pytorch.org/docs/stable/generated/torch.nn.utils.fuse_conv_bn_eval.html#torch-nn-utils-fuse-conv-bn-eval) |
| IN-DEVELOPMENT-PATTERN(`torch.nn.utils.fuse_conv_bn_weights`, https://pytorch.org/docs/stable/generated/torch.nn.utils.fuse_conv_bn_weights.html#torch-nn-utils-fuse-conv-bn-weights) |
| IN-DEVELOPMENT-PATTERN(`torch.nn.utils.fuse_linear_bn_eval`, https://pytorch.org/docs/stable/generated/torch.nn.utils.fuse_linear_bn_eval.html#torch-nn-utils-fuse-linear-bn-eval) |
| IN-DEVELOPMENT-PATTERN(`torch.nn.utils.fuse_linear_bn_weights`, https://pytorch.org/docs/stable/generated/torch.nn.utils.fuse_linear_bn_weights.html#torch-nn-utils-fuse-linear-bn-weights) |
| IN-DEVELOPMENT-PATTERN(`torch.nn.utils.convert_conv2d_weight_memory_format`, https://pytorch.org/docs/stable/generated/torch.nn.utils.convert_conv2d_weight_memory_format.html#torch-nn-utils-convert-conv2d-weight-memory-format) |
| IN-DEVELOPMENT-PATTERN(`torch.nn.utils.convert_conv3d_weight_memory_format`, https://pytorch.org/docs/stable/generated/torch.nn.utils.convert_conv3d_weight_memory_format.html#torch-nn-utils-convert-conv3d-weight-memory-format) |
| IN-DEVELOPMENT-PATTERN(`torch.nn.utils.parametrizations.weight_norm`, https://pytorch.org/docs/stable/generated/torch.nn.utils.parametrizations.weight_norm.html#torch-nn-utils-parametrizations-weight-norm) |
| IN-DEVELOPMENT-PATTERN(`torch.backends.cuda.can_use_flash_attention`, https://pytorch.org/docs/stable/backends.html#torch.backends.cuda.can_use_flash_attention) |
| IN-DEVELOPMENT-PATTERN(`torch.distributed.device_mesh.DeviceMesh`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.device_mesh.DeviceMesh) |
| IN-DEVELOPMENT-PATTERN(`torch.can_cast`, https://pytorch.org/docs/stable/generated/torch.can_cast.html#torch-can-cast) |
| IN-DEVELOPMENT-PATTERN(`torch.Tensor.copysign`, https://pytorch.org/docs/stable/generated/torch.Tensor.copysign.html#torch-tensor-copysign) |
| IN-DEVELOPMENT-PATTERN(`torch.Tensor.mul_`, https://pytorch.org/docs/stable/generated/torch.Tensor.mul_.html#torch-tensor-mul) |
| IN-DEVELOPMENT-PATTERN(`torch.cuda.is_initialized`, https://pytorch.org/docs/stable/generated/torch.cuda.is_initialized.html#torch-cuda-is-initialized) |
| IN-DEVELOPMENT-PATTERN(`torch.cuda.comm.scatter`, https://pytorch.org/docs/stable/generated/torch.cuda.comm.scatter.html#torch-cuda-comm-scatter) |
| IN-DEVELOPMENT-PATTERN(`torch.cuda.comm.gather`, https://pytorch.org/docs/stable/generated/torch.cuda.comm.gather.html#torch-cuda-comm-gather) |
| IN-DEVELOPMENT-PATTERN(`torch.distributions.binomial.Binomial`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.binomial.Binomial) |
| IN-DEVELOPMENT-PATTERN(`torch.distributions.continuous_bernoulli.ContinuousBernoulli`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.continuous_bernoulli.ContinuousBernoulli) |
| IN-DEVELOPMENT-PATTERN(`torch.distributions.exponential.Exponential`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.exponential.Exponential) |
| IN-DEVELOPMENT-PATTERN(`torch.distributions.constraints.Constraint`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.constraints.Constraint) |
| IN-DEVELOPMENT-PATTERN(`torch.linalg.inv_ex`, https://pytorch.org/docs/stable/generated/torch.linalg.inv_ex.html#torch-linalg-inv-ex) |
| IN-DEVELOPMENT-PATTERN(`torch.nn.LazyBatchNorm1d`, https://pytorch.org/docs/stable/generated/torch.nn.LazyBatchNorm1d.html#torch.nn.LazyBatchNorm1d) |
| IN-DEVELOPMENT-PATTERN(`torch.linalg.cholesky_ex`, https://pytorch.org/docs/stable/generated/torch.linalg.cholesky_ex.html#torch-linalg-cholesky-ex) |
| IN-DEVELOPMENT-PATTERN(`torch.positive`, https://pytorch.org/docs/stable/generated/torch.positive.html#torch-positive) |
| IN-DEVELOPMENT-PATTERN(`torch.cuda.StreamContext`, https://pytorch.org/docs/stable/generated/torch.cuda.StreamContext.html#torch.cuda.StreamContext) |
| IN-DEVELOPMENT-PATTERN(`torch.nn.LazyBatchNorm2d`, https://pytorch.org/docs/stable/generated/torch.nn.LazyBatchNorm2d.html#torch.nn.LazyBatchNorm2d) |
| IN-DEVELOPMENT-PATTERN(`torch.nn.LazyBatchNorm3d`, https://pytorch.org/docs/stable/generated/torch.nn.LazyBatchNorm3d.html#torch.nn.LazyBatchNorm3d) |
| IN-DEVELOPMENT-PATTERN(`torch.Tensor.bitwise_right_shift`, https://pytorch.org/docs/stable/generated/torch.Tensor.bitwise_right_shift.html#torch-tensor-bitwise-right-shift) |
| IN-DEVELOPMENT-PATTERN(`torch.bitwise_right_shift`, https://pytorch.org/docs/stable/generated/torch.bitwise_right_shift.html#torch-bitwise-right-shift) |
| IN-DEVELOPMENT-PATTERN(`torch.nn.LazyInstanceNorm2d`, https://pytorch.org/docs/stable/generated/torch.nn.LazyInstanceNorm2d.html#torch.nn.LazyInstanceNorm2d) |
| IN-DEVELOPMENT-PATTERN(`torch.nn.LazyInstanceNorm1d`, https://pytorch.org/docs/stable/generated/torch.nn.LazyInstanceNorm1d.html#torch.nn.LazyInstanceNorm1d) |
| IN-DEVELOPMENT-PATTERN(`torch.Tensor.bitwise_left_shift`, https://pytorch.org/docs/stable/generated/torch.Tensor.bitwise_left_shift.html#torch-tensor-bitwise-left-shift) |
| IN-DEVELOPMENT-PATTERN(`torch.bitwise_left_shift`, https://pytorch.org/docs/stable/generated/torch.bitwise_left_shift.html#torch-bitwise-left-shift) |
| IN-DEVELOPMENT-PATTERN(`torch.isin`, https://pytorch.org/docs/stable/generated/torch.isin.html#torch-isin) |
| IN-DEVELOPMENT-PATTERN(`torch.nn.LazyInstanceNorm3d`, https://pytorch.org/docs/stable/generated/torch.nn.LazyInstanceNorm3d.html#torch.nn.LazyInstanceNorm3d) |
| IN-DEVELOPMENT-PATTERN(`torch.scatter_reduce`, https://pytorch.org/docs/stable/generated/torch.scatter_reduce.html#torch-scatter-reduce) |
| IN-DEVELOPMENT-PATTERN(`torch.set_deterministic_debug_mode`, https://pytorch.org/docs/stable/generated/torch.set_deterministic_debug_mode.html#torch-set-deterministic-debug-mode) |
| IN-DEVELOPMENT-PATTERN(`torch.get_deterministic_debug_mode`, https://pytorch.org/docs/stable/generated/torch.get_deterministic_debug_mode.html#torch-get-deterministic-debug-mode) |
| IN-DEVELOPMENT-PATTERN(`torch.Tensor.scatter_reduce`, https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_reduce.html#torch-tensor-scatter-reduce) |
