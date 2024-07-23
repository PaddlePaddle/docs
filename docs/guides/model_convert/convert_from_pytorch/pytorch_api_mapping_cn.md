# PyTorch 最新 release 与 Paddle develop API 映射表

本文梳理了 PyTorch 最新发行版（当前 v2.3.0） API 与 PaddlePaddle develop 版本 API 对应关系与差异分析。通过本文档，帮助开发者快速迁移 PyTorch 使用经验，完成模型的开发与调优。

## 贡献代码

欢迎你向我们贡献代码，关于如何编写 API 映射关系，为保证文档格式统一性与可读性，请严格参照 [API 映射关系-格式与模板](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/pytorch_api_mapping_format_cn.md) 来编写。

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
| NOT-IMPLEMENTED-ITEM(`torch.bitwise_left_shift`, https://pytorch.org/docs/stable/generated/torch.bitwise_left_shift.html#torch.bitwise_left_shift) |
| NOT-IMPLEMENTED-ITEM(`torch.bitwise_right_shift`, https://pytorch.org/docs/stable/generated/torch.bitwise_right_shift.html#torch.bitwise_right_shift) |
| NOT-IMPLEMENTED-ITEM(`torch.can_cast`, https://pytorch.org/docs/stable/generated/torch.can_cast.html) |
| NOT-IMPLEMENTED-ITEM(`torch.geqrf`, https://pytorch.org/docs/stable/generated/torch.geqrf.html?highlight=geqrf#torch.geqrf) |
| NOT-IMPLEMENTED-ITEM(`torch.get_deterministic_debug_mode`, https://pytorch.org/docs/stable/generated/torch.get_deterministic_debug_mode.html#torch.get_deterministic_debug_mode) |
| NOT-IMPLEMENTED-ITEM(`torch.gradient`, https://pytorch.org/docs/stable/generated/torch.gradient.html#torch.gradient) |
| NOT-IMPLEMENTED-ITEM(`torch.is_conj`, https://pytorch.org/docs/stable/generated/torch.is_conj.html#torch.is_conj) |
| NOT-IMPLEMENTED-ITEM(`torch.isin`, https://pytorch.org/docs/stable/generated/torch.isin.html#torch.isin) |
| NOT-IMPLEMENTED-ITEM(`torch.layout`, https://pytorch.org/docs/stable/tensor_attributes.html#torch.layout) |
| NOT-IMPLEMENTED-ITEM(`torch.positive`, https://pytorch.org/docs/stable/generated/torch.positive.html#torch.positive) |
| NOT-IMPLEMENTED-ITEM(`torch.resolve_conj`, https://pytorch.org/docs/stable/generated/torch.resolve_conj.html#torch.resolve_conj) |
| NOT-IMPLEMENTED-ITEM(`torch.resolve_neg`, https://pytorch.org/docs/stable/generated/torch.resolve_neg.html#torch.resolve_neg) |
| NOT-IMPLEMENTED-ITEM(`torch.scatter_reduce`, https://pytorch.org/docs/stable/generated/torch.scatter_reduce.html#torch.scatter_reduce) |
| NOT-IMPLEMENTED-ITEM(`torch.set_deterministic_debug_mode`, https://pytorch.org/docs/stable/generated/torch.set_deterministic_debug_mode.html#torch.set_deterministic_debug_mode) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.GLU`, https://pytorch.org/docs/stable/generated/torch.nn.GLU.html#torch.nn.GLU) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.LazyBatchNorm1d`, https://pytorch.org/docs/stable/generated/torch.nn.LazyBatchNorm1d.html#torch.nn.LazyBatchNorm1d) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.LazyBatchNorm2d`, https://pytorch.org/docs/stable/generated/torch.nn.LazyBatchNorm2d.html#torch.nn.LazyBatchNorm2d) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.LazyBatchNorm3d`, https://pytorch.org/docs/stable/generated/torch.nn.LazyBatchNorm3d.html#torch.nn.LazyBatchNorm3d) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.LazyInstanceNorm1d`, https://pytorch.org/docs/stable/generated/torch.nn.LazyInstanceNorm1d.html#torch.nn.LazyInstanceNorm1d) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.LazyInstanceNorm2d`, https://pytorch.org/docs/stable/generated/torch.nn.LazyInstanceNorm2d.html#torch.nn.LazyInstanceNorm2d) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.LazyInstanceNorm3d`, https://pytorch.org/docs/stable/generated/torch.nn.LazyInstanceNorm3d.html#torch.nn.LazyInstanceNorm3d) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.functional.multilabel_margin_loss`, https://pytorch.org/docs/stable/generated/torch.nn.functional.multilabel_margin_loss.html#torch.nn.functional.multilabel_margin_loss) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.bitwise_left_shift`, https://pytorch.org/docs/stable/generated/torch.Tensor.bitwise_left_shift.html#torch.Tensor.bitwise_left_shift) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.bitwise_right_shift`, https://pytorch.org/docs/stable/generated/torch.Tensor.bitwise_right_shift.html#torch.Tensor.bitwise_right_shift) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.copysign`, https://pytorch.org/docs/1.13/generated/torch.Tensor.copysign.html?highlight=torch+tensor+copysign#torch.Tensor.copysign) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.dequantize`, https://pytorch.org/docs/1.13/generated/torch.Tensor.dequantize.html?highlight=torch+tensor+dequantize#torch.Tensor.dequantize) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.geqrf`, https://pytorch.org/docs/1.13/generated/torch.Tensor.geqrf.html?highlight=torch+tensor+geqrf#torch.Tensor.geqrf) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.is_coalesced`, https://pytorch.org/docs/stable/generated/torch.Tensor.is_coalesced.html#torch.Tensor.is_coalesced) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.is_conj`, https://pytorch.org/docs/stable/generated/torch.Tensor.is_conj.html#torch.Tensor.is_conj) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.mul_`, https://pytorch.org/docs/stable/generated/torch.Tensor.mul_.html?highlight=mul_) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.resize_`, https://pytorch.org/docs/stable/generated/torch.Tensor.resize_.html?highlight=resize#torch.Tensor.resize_) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.resolve_conj`, https://pytorch.org/docs/stable/generated/torch.Tensor.resolve_conj.html#torch.Tensor.resolve_conj) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.resolve_neg`, https://pytorch.org/docs/stable/generated/torch.Tensor.resolve_neg.html#torch.Tensor.resolve_neg) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.scatter_reduce`, https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_reduce.html#torch.Tensor.scatter_reduce) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.select_scatter`, https://pytorch.org/docs/stable/generated/torch.Tensor.select_scatter.html#torch.Tensor.select_scatter) |
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
| NOT-IMPLEMENTED-ITEM(`torch.cuda.StreamContext`, https://pytorch.org/docs/stable/generated/torch.cuda.StreamContext.html#torch.cuda.StreamContext) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.comm.gather`, https://pytorch.org/docs/stable/generated/torch.cuda.comm.gather.html#torch-cuda-comm-gather) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.comm.scatter`, https://pytorch.org/docs/stable/generated/torch.cuda.comm.scatter.html#torch-cuda-comm-scatter) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.default_stream`, https://pytorch.org/docs/stable/generated/torch.cuda.default_stream.html?highlight=torch+cuda+default_stream#torch.cuda.default_stream) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.get_arch_list`, https://pytorch.org/docs/stable/generated/torch.cuda.get_arch_list.html?highlight=torch+cuda+get_arch_list#torch.cuda.get_arch_list) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.is_current_stream_capturing`, https://pytorch.org/docs/stable/generated/torch.cuda.is_current_stream_capturing.html#torch.cuda.is_current_stream_capturing) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.is_initialized`, https://pytorch.org/docs/stable/generated/torch.cuda.is_initialized.html?highlight=torch+cuda+is_initialized#torch.cuda.is_initialized) |
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
| NOT-IMPLEMENTED-ITEM(`torch.distributions.binomial.Binomial`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.binomial.Binomial) |
| NOT-IMPLEMENTED-ITEM(`torch.distributions.constraint_registry.ConstraintRegistry`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.constraint_registry.ConstraintRegistry) |
| NOT-IMPLEMENTED-ITEM(`torch.distributions.constraints.Constraint`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.constraints.Constraint) |
| NOT-IMPLEMENTED-ITEM(`torch.distributions.continuous_bernoulli.ContinuousBernoulli`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.continuous_bernoulli.ContinuousBernoulli) |
| NOT-IMPLEMENTED-ITEM(`torch.distributions.exponential.Exponential`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.exponential.Exponential) |
| NOT-IMPLEMENTED-ITEM(`torch.distributions.one_hot_categorical.OneHotCategorical`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.one_hot_categorical.OneHotCategorical) |
| NOT-IMPLEMENTED-ITEM(`torch.distributions.transforms.CatTransform`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.transforms.CatTransform) |
| NOT-IMPLEMENTED-ITEM(`torch.distributions.transforms.CumulativeDistributionTransform`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.transforms.CumulativeDistributionTransform) |
| NOT-IMPLEMENTED-ITEM(`torch.distributions.transforms.SoftplusTransform`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.transforms.SoftplusTransform) |
| NOT-IMPLEMENTED-ITEM(`torch.hub.get_dir`, https://pytorch.org/docs/stable/hub.html?highlight=torch+hub+get_dir#torch.hub.get_dir) |
| NOT-IMPLEMENTED-ITEM(`torch.hub.set_dir`, https://pytorch.org/docs/stable/hub.html?highlight=torch+hub+get_dir#torch.hub.set_dir) |
| NOT-IMPLEMENTED-ITEM(`torch.linalg.cholesky_ex`, https://pytorch.org/docs/stable/generated/torch.linalg.cholesky_ex.html#torch.linalg.cholesky_ex) |
| NOT-IMPLEMENTED-ITEM(`torch.linalg.inv_ex`, https://pytorch.org/docs/stable/generated/torch.linalg.inv_ex.html#torch.linalg.inv_ex) |
| NOT-IMPLEMENTED-ITEM(`torch.onnx.disable_log`, https://pytorch.org/docs/stable/onnx.html#torch.onnx.disable_log) |
| NOT-IMPLEMENTED-ITEM(`torch.onnx.enable_log`, https://pytorch.org/docs/stable/onnx.html#torch.onnx.enable_log) |
| NOT-IMPLEMENTED-ITEM(`torch.profiler.ProfilerAction`, https://pytorch.org/docs/stable/profiler.html#torch.profiler.ProfilerAction) |
| NOT-IMPLEMENTED-ITEM(`torch.profiler.ProfilerActivity`, https://pytorch.org/docs/stable/profiler.html#torch.profiler.ProfilerActivity) |
| NOT-IMPLEMENTED-ITEM(`torch.sparse.sampled_addmm`, https://pytorch.org/docs/stable/generated/torch.sparse.sampled_addmm.html#torch.sparse.sampled_addmm) |
| NOT-IMPLEMENTED-ITEM(`torch.special.entr`, https://pytorch.org/docs/stable/special.html#torch.special.entr) |
| NOT-IMPLEMENTED-ITEM(`torch.utils.cpp_extension.include_paths`, https://pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.include_paths) |
| NOT-IMPLEMENTED-ITEM(`torch.utils.cpp_extension.load_inline`, https://pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.load_inline) |


 ## fairscale.XX API 映射列表

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ----- | ----------- | ----------------- | ----------- | ------- |
|MANUAL_MAINTAINING-ITEM(`fairscale.nn.model_parallel.initialize.get_model_parallel_rank`,https://github.com/facebookresearch/fairscale/blob/164cc0f3170b4a3951dd84dda29c3e1504ac4d6e/fairscale/nn/model_parallel/initialize.py#L155, `paddle.distributed.fleet.base.topology._HYBRID_PARALLEL_GROUP.get_model_parallel_rank`, https://github.com/PaddlePaddle/Paddle/blob/ddac1b431483ddc0f1ee600e799aa31fc0a75961/python/paddle/distributed/fleet/base/topology.py#L463, 无参数 , https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference_third_party/fairscale/fairscale.nn.model_parallel.initialize.get_model_parallel_rank.md) |
|MANUAL_MAINTAINING-ITEM(`fairscale.nn.model_parallel.initialize.get_model_parallel_world_size`,https://github.com/facebookresearch/fairscale/blob/164cc0f3170b4a3951dd84dda29c3e1504ac4d6e/fairscale/nn/model_parallel/initialize.py#L150, `paddle.distributed.fleet.base.topology._HYBRID_PARALLEL_GROUP._mp_degree`,https://github.com/PaddlePaddle/Paddle/blob/ddac1b431483ddc0f1ee600e799aa31fc0a75961/python/paddle/distributed/fleet/base/topology.py#L185, 无参数 , https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference_third_party/fairscale/fairscale.nn.model_parallel.initialize.get_model_parallel_world_size.md) |
|MANUAL_MAINTAINING-ITEM(`fairscale.nn.model_parallel.initialize.initialize_model_parallel`,https://github.com/facebookresearch/fairscale/blob/164cc0f3170b4a3951dd84dda29c3e1504ac4d6e/fairscale/nn/model_parallel/initialize.py#L41, ` `, , 组合替代实现 , https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference_third_party/fairscale/fairscale.nn.model_parallel.initialize.initialize_model_parallel.md) |
|MANUAL_MAINTAINING-ITEM(`fairscale.nn.model_parallel.initialize.model_parallel_is_initialized`, https://github.com/facebookresearch/fairscale/blob/164cc0f3170b4a3951dd84dda29c3e1504ac4d6e/fairscale/nn/model_parallel/initialize.py#L119,` `, , 组合替代实现 , https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference_third_party/fairscale/fairscale.nn.model_parallel.initialize.model_parallel_is_initialized.md) |
|MANUAL_MAINTAINING-ITEM(`fairscale.nn.model_parallel.layers.ColumnParallelLinear`,https://github.com/facebookresearch/fairscale/blob/164cc0f3170b4a3951dd84dda29c3e1504ac4d6e/fairscale/nn/model_parallel/layers.py#L218, `paddle.distributed.meta_parallel.parallel_layers.mp_layers.ColumnParallelLinear`,https://github.com/PaddlePaddle/Paddle/blob/016766cc89fabc10181453ce70b701dd8ed019f6/python/paddle/distributed/fleet/layers/mpu/mp_layers.py#L153, torch 参数更多 , https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference_third_party/fairscale/fairscale.nn.model_parallel.layers.ColumnParallelLinear.md) |
|MANUAL_MAINTAINING-ITEM(`fairscale.nn.model_parallel.layers.ParallelEmbedding`,https://github.com/facebookresearch/fairscale/blob/164cc0f3170b4a3951dd84dda29c3e1504ac4d6e/fairscale/nn/model_parallel/layers.py#L152, `paddle.distributed.meta_parallel.parallel_layers.mp_layers.VocabParallelEmbedding`,https://github.com/PaddlePaddle/Paddle/blob/016766cc89fabc10181453ce70b701dd8ed019f6/python/paddle/distributed/fleet/layers/mpu/mp_layers.py#L37, torch 参数更多 , https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference_third_party/fairscale/fairscale.nn.model_parallel.layers.ParallelEmbedding.md) |
|MANUAL_MAINTAINING-ITEM(`fairscale.nn.model_parallel.layers.RowParallelLinear`,https://github.com/facebookresearch/fairscale/blob/164cc0f3170b4a3951dd84dda29c3e1504ac4d6e/fairscale/nn/model_parallel/layers.py#L299, `paddle.distributed.meta_parallel.parallel_layers.mp_layers.RowParallelLinear`,https://github.com/PaddlePaddle/Paddle/blob/016766cc89fabc10181453ce70b701dd8ed019f6/python/paddle/distributed/fleet/layers/mpu/mp_layers.py#L291,torch 参数更多 , https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference_third_party/fairscale/fairscale.nn.model_parallel.layers.RowParallelLinear.md) |


 ## transformers.XX API 映射列表

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ----- | ----------- | ----------------- | ----------- | ------- |
|MANUAL_MAINTAINING-ITEM(`transformers.AddedToken`,https://github.com/huggingface/transformers/blob/d625294d79341662784495551abdf45e6cb9372f/src/transformers/tokenization_utils_base.py#L84, `paddlenlp.transformers.AddedToken`,https://github.com/PaddlePaddle/PaddleNLP/blob/e336e78c338d2514ee6c937982ce5d8c960b85ff/paddlenlp/transformers/tokenizer_utils_base.py#L48, 仅 paddle 参数更多 , https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference_third_party/transformers/transformers.AddedToken.md) |
|MANUAL_MAINTAINING-ITEM(`transformers.generation.LogitsProcessor`,https://hf-mirror.com/docs/transformers/v4.42.0/en/internal/generation_utils#logitsprocessor, `paddlenlp.generation.LogitsProcessor`,https://github.com/PaddlePaddle/PaddleNLP/blob/e336e78c338d2514ee6c937982ce5d8c960b85ff/paddlenlp/generation/logits_process.py#L26 , 输入参数用法不一致 , https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference_third_party/transformers/transformers.generation.LogitsProcessor.md) |
|MANUAL_MAINTAINING-ITEM(`transformers.generation.LogitsProcessorList`,https://hf-mirror.com/docs/transformers/v4.42.0/en/internal/generation_utils#transformers.LogitsProcessorList, `paddlenlp.generation.LogitsProcessorList`,https://github.com/PaddlePaddle/PaddleNLP/blob/e336e78c338d2514ee6c937982ce5d8c960b85ff/paddlenlp/transformers/bloom/processor.py#L24 , 无参数 , https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference_third_party/transformers/transformers.generation.LogitsProcessorList.md) |
|MANUAL_MAINTAINING-ITEM(`transformers.generation.utils.GenerateOutput`,https://github.com/huggingface/transformers/blob/v4.42.0/src/transformers/generation/utils.py#L345, `paddlenlp.transformers.model_outputs.BaseModelOutput`,https://github.com/PaddlePaddle/PaddleNLP/blob/e336e78c338d2514ee6c937982ce5d8c960b85ff/paddlenlp/transformers/model_outputs.py#L513, 无参数 , https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference_third_party/transformers/transformers.generation.utils.GenerateOutput.md) |
|MANUAL_MAINTAINING-ITEM(`transformers.GenerationConfig`,https://hf-mirror.com/docs/transformers/v4.42.0/en/main_classes/text_generation#transformers.GenerationConfig, `paddlenlp.generation.GenerationConfig`,https://github.com/PaddlePaddle/PaddleNLP/blob/e336e78c338d2514ee6c937982ce5d8c960b85ff/paddlenlp/generation/configuration_utils.py#L62 , torch 参数更多 , https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference_third_party/transformers/transformers.GenerationConfig.md) |
|MANUAL_MAINTAINING-ITEM(`transformers.logging.get_logger`,https://github.com/huggingface/transformers/blob/d625294d79341662784495551abdf45e6cb9372f/src/transformers/utils/logging.py#L147, ` `,  , 组合替代实现 , https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference_third_party/transformers/transformers.logging.get_logger.md) |
|MANUAL_MAINTAINING-ITEM(`transformers.modeling_outputs.BaseModelOutputWithPast`,https://hf-mirror.com/docs/transformers/v4.42.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast, `paddlenlp.transformers.model_outputs.BaseModelOutputWithPast`,https://github.com/PaddlePaddle/PaddleNLP/blob/e336e78c338d2514ee6c937982ce5d8c960b85ff/paddlenlp/transformers/model_outputs.py#L590 , 参数完全一致 , https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference_third_party/transformers/transformers.modeling_outputs.BaseModelOutputWithPast.md) |
|MANUAL_MAINTAINING-ITEM(`transformers.modeling_outputs.CausalLMOutputWithPast`,https://hf-mirror.com/docs/transformers/v4.42.0/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast, `paddlenlp.transformers.model_outputs.CausalLMOutputWithPast`,https://github.com/PaddlePaddle/PaddleNLP/blob/e336e78c338d2514ee6c937982ce5d8c960b85ff/paddlenlp/transformers/model_outputs.py#L874 , 参数完全一致 , https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference_third_party/transformers/transformers.modeling_outputs.CausalLMOutputWithPast.md) |
|MANUAL_MAINTAINING-ITEM(`transformers.PretrainedConfig`,https://hf-mirror.com/docs/transformers/v4.42.0/en/main_classes/configuration#transformers.PretrainedConfig, `paddlenlp.transformers.PretrainedConfig`,https://github.com/PaddlePaddle/PaddleNLP/blob/e336e78c338d2514ee6c937982ce5d8c960b85ff/paddlenlp/generation/configuration_utils.py#L62 , torch 参数更多 , https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference_third_party/transformers/transformers.PretrainedConfig.md) |
|MANUAL_MAINTAINING-ITEM(`transformers.PreTrainedModel.generate`,https://github.com/huggingface/transformers/blob/0fdea8607d7e01eb0e38a1ebeb7feee30a22f0cf/src/transformers/generation/utils.py#L1567, `paddlenlp.transformers.PreTrainedModel.generate`, https://github.com/PaddlePaddle/PaddleNLP/blob/88d4b19bc6865fb28c11d2ce83d07c3b4b8dc423/paddlenlp/generation/utils.py#L604, 返回参数类型不一致 , https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference_third_party/transformers/transformers.PreTrainedModel.generate.md) |
|MANUAL_MAINTAINING-ITEM(`transformers.PreTrainedModel.get_head_mask`,https://hf-mirror.com/docs/transformers/v4.42.0/en/main_classes/model#transformers.modeling_utils.ModuleUtilsMixin.get_head_mask, , , 组合替代实现 , https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference_third_party/transformers/transformers.PreTrainedModel.get_head_mask.md) |
|MANUAL_MAINTAINING-ITEM(`transformers.PreTrainedModel`,https://github.com/huggingface/transformers/blob/d625294d79341662784495551abdf45e6cb9372f/src/transformers/modeling_utils.py#L1231, `paddlenlp.transformers.PreTrainedModel`, https://github.com/PaddlePaddle/PaddleNLP/blob/e336e78c338d2514ee6c937982ce5d8c960b85ff/paddlenlp/transformers/model_utils.py#L843, 参数完全一致 , https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference_third_party/transformers/transformers.PreTrainedModel.md) |
|MANUAL_MAINTAINING-ITEM(`transformers.PreTrainedModel.post_init`,https://hf-mirror.com/docs/transformers/v4.42.0/en/main_classes/model#transformers.PreTrainedModel.post_init, , , 组合替代实现 , https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference_third_party/transformers/transformers.PreTrainedModel.post_init.md) |
|MANUAL_MAINTAINING-ITEM(`transformers.PreTrainedTokenizer.encode`,https://hf-mirror.com/docs/transformers/v4.42.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer.encode, `paddlenlp.transformers.PreTrainedTokenizer.encode`, https://github.com/PaddlePaddle/PaddleNLP/blob/88d4b19bc6865fb28c11d2ce83d07c3b4b8dc423/paddlenlp/transformers/tokenizer_utils_base.py#L2369, 返回参数类型不一致 , https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference_third_party/transformers/transformers.PreTrainedTokenizer.encode.md) |
|MANUAL_MAINTAINING-ITEM(`transformers.PreTrainedTokenizer`,https://hf-mirror.com/docs/transformers/v4.42.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer, `paddlenlp.transformers.PreTrainedTokenizer`, https://github.com/PaddlePaddle/PaddleNLP/blob/e336e78c338d2514ee6c937982ce5d8c960b85ff/paddlenlp/transformers/tokenizer_utils.py#L881, 参数完全一致 , https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference_third_party/transformers/transformers.PreTrainedTokenizer.md) |
|MANUAL_MAINTAINING-ITEM(`transformers.StoppingCriteriaList`,https://github.com/huggingface/transformers/blob/d625294d79341662784495551abdf45e6cb9372f/src/transformers/generation/stopping_criteria.py#L503, `paddlenlp.generation.StoppingCriteriaList`, https://github.com/PaddlePaddle/PaddleNLP/blob/e336e78c338d2514ee6c937982ce5d8c960b85ff/paddlenlp/generation/stopping_criteria.py#L72, 参数完全一致 , https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference_third_party/transformers/transformers.StoppingCriteriaList.md) |


 ## flash_attn.XX API 映射列表

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ----- | ----------- | ----------------- | ----------- | ------- |
|MANUAL_MAINTAINING-ITEM(`flash_attn.flash_attn_interface.flash_attn_func`,https://github.com/Dao-AILab/flash-attention/blob/72e27c6320555a37a83338178caa25a388e46121/flash_attn/flash_attn_interface.py#L808, `paddle.nn.functional.flash_attention.flash_attention`, https://github.com/PaddlePaddle/Paddle/blob/900d27c40ef4567d7ea6342f3f0eedd394885ecb/python/paddle/nn/functional/flash_attention.py#L248, torch 参数更多 , https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference_third_party/flash_attn/flash_attn.flash_attn_interface.flash_attn_func.md) |
|MANUAL_MAINTAINING-ITEM(`flash_attn.flash_attn_interface.flash_attn_unpadded_func`,https://github.com/Dao-AILab/flash-attention/blob/d0787acc16c3667156b51ce5b01bdafc7594ed39/flash_attn/flash_attn_interface.py#L1050 `paddle.nn.functional.flash_attention.flash_attn_unpadded`, https://github.com/PaddlePaddle/Paddle/blob/b32b51b7c21ad62bf794512c849a603c8c0ece44/python/paddle/nn/functional/flash_attention.py#L664, torch 参数更多 , https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference_third_party/flash_attn/flash_attn.flash_attn_interface.flash_attn_unpadded_func.md) |
|MANUAL_MAINTAINING-ITEM(`flash_attn.layers.rotary.apply_rotary_emb_func`,https://github.com/Dao-AILab/flash-attention/blob/d0787acc16c3667156b51ce5b01bdafc7594ed39/flash_attn/layers/rotary.py#L94, ` `,  , 组合替代实现 , https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference_third_party/flash_attn/flash_attn.layers.rotary.apply_rotary_emb_func.md) |
|MANUAL_MAINTAINING-ITEM(`flash_attn.ops.rms_norm.rms_norm`,https://github.com/Dao-AILab/flash-attention/blob/d0787acc16c3667156b51ce5b01bdafc7594ed39/flash_attn/ops/rms_norm.py#L14, `paddle.incubate.nn.functional.fused_rms_norm`, https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/incubate/nn/functional/fused_rms_norm_cn.html, 仅 paddle 参数更多 , https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference_third_party/flash_attn/flash_attn.ops.rms_norm.rms_norm.md) |

***持续更新...***
