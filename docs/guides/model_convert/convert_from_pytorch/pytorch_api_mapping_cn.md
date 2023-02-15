# PyTorch 1.13 与 Paddle 2.4 API 映射表
本文档梳理了 PyTorch（v1.13)常用 API 与 PaddlePaddle 2.4.0 API 对应关系与差异分析。通过本文档，帮助开发者快速迁移 PyTorch 使用经验，完成模型的开发与调优。

## API 映射表目录

| 类别         | 简介 |
| ---------- | ------------------------- |
| [torch.XX](#id1) | 主要为`torch.XX`类 API |
| [torch.nn.XX](#id2) | 主要为`torch.nn.XX`类 API |
| [torch.nn.functional.XX](#id3) | 主要为`torch.nn.functional.XX`类 API |
| [torch.nn.init.XX](#id4) | 主要为`torch.nn.init.XX`类 API |
| [torch.nn.utils.XX](#id5) | 主要为`torch.nn.init.XX`类 API |
| [torch.Tensor.XX](#id6) | 主要为`torch.Tensor.XX`类 API |
| [torch.cuda.XX](#id7) | 主要为`torch.cuda.XX`类 API |
| [torch.distributed.XX](#id8) | 主要为`torch.distributed.XX`类 API |
| [torch.distributions.XX](#id9)    | 主要为`torch.distributions.XX`类 API |
| [torch.fft.XX](#id10)    | 主要为`torch.fft.XX`类 API |
| [torch.linalg.XX](#id11)    | 主要为`torch.linalg.XX`类 API |
| [其他](#id12)    | 其他 API |

## torch.XX API 映射列表
梳理了`torch.XX`类 API 的 PyTorch-PaddlePaddle API 映射列表。

| 序号 | PyTorch API                                                  | PaddlePaddle API                                             | 备注                                                         |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1   | [torch.is\_tensor](https://pytorch.org/docs/stable/generated/torch.is_tensor.html?highlight=is_tensor#torch.is_tensor) | [paddle.is\_tensor](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/is_tensor_cn.html#is-tensor) | 功能一致，仅参数命名不一致，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.is_tensor.md)    |
| 2   | [torch.is\_complex](https://pytorch.org/docs/stable/generated/torch.is_complex.html?highlight=is_complex#torch.is_complex) | [paddle.is\_complex](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/is_complex_cn.html#is-complex) | 功能一致，仅参数命名不一致，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.is_complex.md)    |
| 3   | [torch.is\_floating\_point](https://pytorch.org/docs/stable/generated/torch.is_floating_point.html?highlight=is_floating_point#torch.is_floating_point) | [paddle.is\_floating\_point](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/is_floating_point_cn.html#is-floating-point) | 功能一致，仅参数命名不一致，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.is_floating_point.md)    |
| 4   | [torch.set\_default\_dtype](https://pytorch.org/docs/stable/generated/torch.set_default_dtype.html?highlight=set_default_dtype#torch.set_default_dtype) | [paddle.set\_default\_dtype](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/set_default_dtype_cn.html#set-default-dtype) | 功能一致，参数完全一致    |
| 5   | [torch.get\_default\_dtype](https://pytorch.org/docs/stable/generated/torch.get_default_dtype.html?highlight=get_default_dtype#torch.get_default_dtype) | [paddle.get\_default\_dtype](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/get_default_dtype_cn.html#get-default-dtype) | 功能一致，参数完全一致     |
| 6   | [torch.numel](https://pytorch.org/docs/stable/generated/torch.numel.html?highlight=numel#torch.numel) | [paddle.numel](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/numel_cn.html) | 功能一致，仅参数命名不一致，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.numel.md)    |
| 7   | [torch.real](https://pytorch.org/docs/stable/generated/torch.real.html?highlight=real#torch.real) | [paddle.real](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/real_cn.html#real) | 功能一致，仅参数命名不一致，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.real.md)    |
| 8   | [torch.imag](https://pytorch.org/docs/stable/generated/torch.imag.html?highlight=imag#torch.imag) | [paddle.imag](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/imag_cn.html#imag) | 功能一致，仅参数命名不一致，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.imag.md)    |
| 9   | [torch.chunk](https://pytorch.org/docs/stable/generated/torch.chunk.html?highlight=chunk#torch.chunk) | [paddle.chunk](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/chunk_cn.html#chunk) | 功能一致，仅参数命名不一致，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.chunk.md)    |
| 10   | [torch.movedim](https://pytorch.org/docs/stable/generated/torch.movedim.html?highlight=movedim#torch.movedim) | [paddle.moveaxis](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/moveaxis_cn.html#moveaxis) | 功能一致，仅参数命名不一致，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.movedim.md)    |
| 11   | [torch.reshape](https://pytorch.org/docs/stable/generated/torch.reshape.html?highlight=reshape#torch.reshape) | [paddle.reshape](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/reshape_cn.html#reshape) | 功能一致，仅参数命名不一致，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.reshape.md)    |
| 12   | [torch.split](https://pytorch.org/docs/stable/generated/torch.split.html?highlight=split#torch.split) | [paddle.split](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/split_cn.html#split) | 功能一致，参数不一致，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.split.md)     |
| 13   | [torch.squeeze](https://pytorch.org/docs/stable/generated/torch.squeeze.html?highlight=squeeze#torch.squeeze) | [paddle.squeeze](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/squeeze_cn.html#squeeze) | 功能一致，仅参数命名不一致，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.squeeze.md)    |
| 14   | [torch.t](https://pytorch.org/docs/stable/generated/torch.t.html?highlight=t#torch.t) | [paddle.t](paddlepaddle.org.cn/documentation/docs/zh/api/paddle/t_cn.html#t) | 功能一致，参数完全一致    |
| 15   | [torch.tile](https://pytorch.org/docs/stable/generated/torch.tile.html?highlight=tile#torch.tile) | [paddle.tile](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/tile_cn.html#tile) | 功能一致，仅参数命名不一致，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.tile.md)    |
| 16   | [torch.unbind](https://pytorch.org/docs/stable/generated/torch.unbind.html?highlight=unbind#torch.unbind) | [paddle.unbind](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/unbind_cn.html#unbind) | 功能一致，仅参数命名不一致，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.unbind.md)    |
| 17   | [torch.unsqueeze](https://pytorch.org/docs/stable/generated/torch.unsqueeze.html?highlight=unsqueeze#torch.unsqueeze) | [paddle.unsqueeze](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/unsqueeze_cn.html#unsqueeze) | 功能一致，仅参数命名不一致，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.unsqueeze.md)    |
| 18   | [torch.where](https://pytorch.org/docs/stable/generated/torch.where.html?highlight=where#torch.where) | [paddle.where](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/where_cn.html#where) | 功能一致，参数完全一致    |
| 19   | [torch.manual\_seed](https://pytorch.org/docs/stable/generated/torch.manual_seed.html?highlight=manual_seed#torch.manual_seed) | [paddle.seed](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/seed_cn.html#seed) | 功能一致，参数完全一致                                       |
| 20   | [torch.no\_grad](https://pytorch.org/docs/stable/generated/torch.no_grad.html?highlight=no_grad#torch.no_grad) | [paddle.no\_grad](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/no_grad_cn.html#no-grad) | 功能一致，无参数     |
| 21   | [torch.set\_grad\_enabled](https://pytorch.org/docs/stable/generated/torch.set_grad_enabled.html?highlight=set_grad_enabled#torch.set_grad_enabled) | [paddle.set\_grad\_enabled](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/set_grad_enabled_cn.html#set-grad-enabled) | 功能一致，参数完全一致                                       |
| 22   | [torch.corrcoef](https://pytorch.org/docs/stable/generated/torch.corrcoef.html?highlight=corrcoef#torch.corrcoef) | [paddle.linalg.corrcoef](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/linalg/corrcoef_cn.html#corrcoef) | 功能一致，仅参数命名不一致，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.corrcoef.md)    |
| 23   | [torch.permute](https://pytorch.org/docs/stable/generated/torch.permute.html?highlight=permute#torch.permute) | [paddle.transpose](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/transpose_cn.html#transpose) | 功能一致，仅参数命名不一致，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.permute.md)    |
| 24   | [torch.conj](https://pytorch.org/docs/stable/generated/torch.conj.html?highlight=conj#torch.conj) | [paddle.conj](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/conj_cn.html#conj) | 功能一致，仅参数命名不一致，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.conj.md)    |
| 25   | [torch.argmax](https://pytorch.org/docs/stable/generated/torch.argmax.html?highlight=argmax#torch.argmax) | [paddle.argmax](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/argmax_cn.html#argmax) | 功能一致，仅参数命名不一致，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.argmax.md)    |
| 26   | [torch.argmin](https://pytorch.org/docs/stable/generated/torch.argmin.html?highlight=argmin#torch.argmin) | [paddle.argmin](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/argmin_cn.html#argmin) | 功能一致，仅参数命名不一致，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.argmin.md)    |
| 27   | [torch.all](https://pytorch.org/docs/stable/generated/torch.all.html?highlight=all#torch.all) | [paddle.all](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/all_cn.html#all) | 功能一致，仅参数命名不一致，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.all.md)    |
| 28   | [torch.any](https://pytorch.org/docs/stable/generated/torch.any.html?highlight=any#torch.any) | [paddle.any](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/any_cn.html#any) | 功能一致，仅参数命名不一致，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.any.md)    |
| 29   | [torch.dist](https://pytorch.org/docs/stable/generated/torch.dist.html?highlight=dist#torch.dist) | [paddle.dist](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/dist_cn.html#dist) | 功能一致，仅参数命名不一致，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.dist.md)    |
| 30   | [torch.median](https://pytorch.org/docs/stable/generated/torch.median.html?highlight=median#torch.median) | [paddle.median](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/median_cn.html#median) | 功能一致，仅参数命名不一致，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.median.md)    |
| 31   | [torch.nanmedian](https://pytorch.org/docs/stable/generated/torch.nanmedian.html?highlight=nanmedian#torch.nanmedian) | [paddle.nanmedian](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nanmedian_cn.html#nanmedian) | 功能一致，仅参数命名不一致，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.nanmedian.md)    |
| 32   | [torch.prod](https://pytorch.org/docs/stable/generated/torch.prod.html?highlight=prod#torch.prod) | [paddle.prod](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/prod_cn.html#prod) | 功能一致，仅参数命名不一致，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.prod.md)    |
| 33   | [torch.sum](https://pytorch.org/docs/stable/generated/torch.sum.html?highlight=sum#torch.sum) | [paddle.sum](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/sum_cn.html#sum) | 功能一致，仅参数命名不一致，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.sum.md)    |
| 34   | [torch.unique\_consecutive](https://pytorch.org/docs/stable/generated/torch.unique_consecutive.html?highlight=unique_consecutive#torch.unique_consecutive) | [paddle.unique\_consecutive](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/unique_consecutive_cn.html#unique-consecutive) | 功能一致，仅参数命名不一致，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.unique_consecutive.md)    |
| 35   | [torch.allclose](https://pytorch.org/docs/stable/generated/torch.allclose.html?highlight=allclose#torch.allclose) | [paddle.allclose](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/allclose_cn.html#allclose) | 功能一致，仅参数命名不一致，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.allclose.md)    |
| 36   | [torch.equal](https://pytorch.org/docs/stable/generated/torch.equal.html?highlight=equal#torch.equal) | [paddle.equal\_all](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/equal_all_cn.html#equal-all) | 功能一致，参数不一致，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.equal.md)    |
| 37   | [torch.isclose](https://pytorch.org/docs/stable/generated/torch.isclose.html?highlight=isclose#torch.isclose) | [paddle.isclose](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/isclose_cn.html#isclose) | 功能一致，仅参数命名不一致，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.isclose.md)    |
| 38   | [torch.isfinite](https://pytorch.org/docs/stable/generated/torch.isfinite.html?highlight=isfinite#torch.isfinite) | [paddle.isfinite](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/isfinite_cn.html#isfinite) | 功能一致，仅参数命名不一致，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.isfinite.md)    |
| 39   | [torch.isinf](https://pytorch.org/docs/stable/generated/torch.isinf.html?highlight=isinf#torch.isinf) | [paddle.isinf](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/isinf_cn.html#isinf) | 功能一致，仅参数命名不一致，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.isinf.md)    |
| 40   | [torch.isnan](https://pytorch.org/docs/stable/generated/torch.isnan.html?highlight=isnan#torch.isnan) | [paddle.isnan](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/isnan_cn.html#isnan) | 功能一致，仅参数命名不一致，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.isnan.md)    |
| 41   | [torch.istft](https://pytorch.org/docs/stable/generated/torch.istft.html?highlight=istft#torch.istft) | [paddle.signal.istft](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/signal/istft_cn.html#istft) | 功能一致，仅参数命名不一致，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.istft.md)    |
| 42   | [torch.bincount](https://pytorch.org/docs/stable/generated/torch.bincount.html?highlight=bincount#torch.bincount) | [paddle.bincount](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/bincount_cn.html#bincount) | 功能一致，仅参数命名不一致，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.bincount.md)    |
| 43   | [torch.broadcast\_tensors](https://pytorch.org/docs/stable/generated/torch.broadcast_tensors.html?highlight=broadcast_tensors#torch.broadcast_tensors) | [paddle.broadcast\_tensors](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/broadcast_tensors_cn.html#broadcast-tensors) | 功能一致，参数不一致，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.broadcast_tensors.md)    |
| 44   | [torch.broadcast\_to](https://pytorch.org/docs/stable/generated/torch.broadcast_to.html?highlight=broadcast_to#torch.broadcast_to) | [paddle.broadcast\_to](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/broadcast_to_cn.html#broadcast-to) | 功能一致，仅参数命名不一致，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.broadcast_to.md)    |
| 45   | [torch.diag\_embed](https://pytorch.org/docs/stable/generated/torch.diag_embed.html?highlight=diag_embed#torch.diag_embed) | [paddle.nn.functional.diag\_embed](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/diag_embed_cn.html#diag-embed) | 功能一致，参数完全一致    |
| 46   | [torch.diagflat](https://pytorch.org/docs/stable/generated/torch.diagflat.html?highlight=diagflat#torch.diagflat) | [paddle.diagflat](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/diagflat_cn.html#diagflat) | 功能一致，仅参数命名不一致，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.diagflat.md)    |
| 47   | [torch.diagonal](https://pytorch.org/docs/stable/generated/torch.diagonal.html?highlight=diagonal#torch.diagonal) | [paddle.diagonal](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/diagonal_cn.html#diagonal) | 功能一致，仅参数命名不一致，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.diagonal.md)    |
| 48   | [torch.diff](https://pytorch.org/docs/stable/generated/torch.diff.html?highlight=diff#torch.diff) | [paddle.diff](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/diff_cn.html#diff) | 功能一致，仅参数命名不一致，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.diff.md)    |
| 49   | [torch.flatten](https://pytorch.org/docs/stable/generated/torch.flatten.html?highlight=flatten#torch.flatten) | [paddle.flatten](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/flatten_cn.html#flatten) | 功能一致，仅参数命名不一致，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.flatten.md)    |
| 50   | [torch.flip](https://pytorch.org/docs/stable/generated/torch.flip.html?highlight=flip#torch.flip) | [paddle.flip](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/flip_cn.html#flip) | 功能一致，仅参数命名不一致，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.flip.md)    |
| 51   | [torch.fliplr](https://pytorch.org/docs/stable/generated/torch.fliplr.html?highlight=fliplr#torch.fliplr) | [paddle.flip](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/flip_cn.html#flip) | 功能一致，仅参数命名不一致，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.fliplr.md)    |
| 52   | [torch.rot90](https://pytorch.org/docs/stable/generated/torch.rot90.html?highlight=rot90#torch.rot90) | [paddle.rot90](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/rot90_cn.html#rot90) | 功能一致，仅参数命名不一致，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.rot90.md)    |
| 53   | [torch.roll](https://pytorch.org/docs/stable/generated/torch.roll.html?highlight=roll#torch.roll) | [paddle.roll](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/roll_cn.html#roll) | 功能一致，仅参数命名不一致，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.roll.md)    |
| 54   | [torch.trace](https://pytorch.org/docs/stable/generated/torch.trace.html?highlight=trace#torch.trace) | [paddle.trace](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/trace_cn.html#trace) | 功能一致，仅参数命名不一致，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.trace.md)    |
| 55   | [torch.view\_as\_real](https://pytorch.org/docs/stable/generated/torch.view_as_real.html?highlight=view_as_real#torch.view_as_real) | [paddle.as\_real](paddlepaddle.org.cn/documentation/docs/zh/api/paddle/as_real_cn.html#as-real) | 功能一致，仅参数命名不一致，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.view_as_real.md)    |
| 56   | [torch.view\_as\_complex](https://pytorch.org/docs/stable/generated/torch.view_as_complex.html?highlight=view_as_complex#torch.view_as_complex) | [paddle.as\_complex](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/as_complex_cn.html#as-complex) | 功能一致，仅参数命名不一致，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.view_as_complex.md)    |
| 57   | [torch.det](https://pytorch.org/docs/stable/generated/torch.det.html?highlight=det#torch.det) | [paddle.linalg.det](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/linalg/det_cn.html#det) | 功能一致，仅参数命名不一致，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.det.md)    |
| 58   | [torch.slogdet](https://pytorch.org/docs/stable/generated/torch.slogdet.html?highlight=slogdet#torch.slogdet) | [paddle.linalg.slogdet](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/linalg/slogdet_cn.html#slogdet) | 功能一致，仅参数命名不一致，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.slogdet.md)    |
| 59   | [torch.pinverse](https://pytorch.org/docs/stable/generated/torch.pinverse.html?highlight=pinverse#torch.pinverse) | [paddle.linalg.pinv](paddlepaddle.org.cn/documentation/docs/zh/api/paddle/linalg/pinv_cn.html#pinv) | 功能一致，仅参数命名不一致，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.pinverse.md)    |
| 60    | [torch.is\_grad\_enabled](https://pytorch.org/docs/stable/generated/torch.is_grad_enabled.html?highlight=is_grad_enabled#torch.is_grad_enabled) | [paddle.is\_grad\_enabled](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/is_grad_enabled_cn.html#is-grad-enabled) | 功能一致，无参数   |
| 61   | [torch.cov](https://pytorch.org/docs/stable/generated/torch.cov.html?highlight=cov#torch.cov) | [paddle.linalg.cov](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/linalg/cov_cn.html#cov) | 功能一致，仅参数命名不一致，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.cov.md)    |
| 62   | [torch.moveaxis](https://pytorch.org/docs/stable/generated/torch.moveaxis.html?highlight=moveaxis#torch.moveaxis) | [paddle.moveaxis](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/moveaxis_cn.html#moveaxis) |功能一致，仅参数命名不一致，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.moveaxis.md)    |
| 63   | [torch.sqrt](https://pytorch.org/docs/stable/generated/torch.sqrt.html?highlight=sqrt#torch.sqrt) | [paddle.sqrt](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/sqrt_cn.html#sqrt) | 功能一致，torch 参数更多，torch 多 `out` 参数代表输出 |
| 64    | [torch.tensor](https://pytorch.org/docs/stable/generated/torch.tensor.html?highlight=tensor#torch.tensor) | [paddle.to\_tensor](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/to_tensor_cn.html#to-tensor) | 功能一致，torch 参数更多，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.tensor.md)                              |
| 65    | [torch.zeros\_like](https://pytorch.org/docs/stable/generated/torch.zeros_like.html?highlight=zeros_like#torch.zeros_like) | [paddle.zeros\_like](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/zeros_like_cn.html#zeros-like) | 功能一致，torch 参数更多，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.zeros_like.md)                          |
| 66    | [torch.ones](https://pytorch.org/docs/stable/generated/torch.ones.html?highlight=ones#torch.ones) | [paddle.ones](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/ones_cn.html#ones) | 功能一致，参数不一致，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.ones.md)                                |
| 67    | [torch.ones\_like](https://pytorch.org/docs/stable/generated/torch.zeros_like.html?highlight=zeros_like#torch.zeros_like) | [paddle.ones\_like](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/ones_like_cn.html#ones-like) | 功能一致，torch 参数更多，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.ones_like.md)                           |
| 68   | [torch.empty](https://pytorch.org/docs/stable/generated/torch.empty.html?highlight=empty#torch.empty) | [paddle.empty](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/empty_cn.html#empty) | 功能一致，参数不一致，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.empty.md)                               |
| 69   | [torch.empty\_like](https://pytorch.org/docs/stable/generated/torch.empty_like.html?highlight=empty_like#torch.empty_like) | [paddle.empty\_like](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/empty_like_cn.html#empty-like) | 功能一致，torch 参数更多，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.empty_like.md)                          |
| 70   | [torch.full](https://pytorch.org/docs/stable/generated/torch.full.html?highlight=full#torch.full) | [paddle.full](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/full_cn.html#full) | 功能一致，torch 参数更多，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.full.md)                    |
| 71   | [torch.full\_like](https://pytorch.org/docs/stable/generated/torch.full_like.html?highlight=full_like#torch.full_like) | [paddle.full\_like](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/full_like_cn.html#full-like) | 功能一致，torch 参数更多，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.full_like.md)                           |
| 72   | [torch.arange](https://pytorch.org/docs/stable/generated/torch.arange.html?highlight=arange#torch.arange) | [paddle.arange](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/arange_cn.html#arange) | 功能一致，torch 参数更多，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.arange.md)                   |
| 73   | [torch.range](https://pytorch.org/docs/stable/generated/torch.range.html?highlight=range#torch.range) | [paddle.arange](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/arange_cn.html#arange) | 功能一致，torch 参数更多，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.range.md)                   |
| 74   | [torch.linspace](https://pytorch.org/docs/stable/generated/torch.linspace.html?highlight=linspace#torch.linspace) | [paddle.linspace](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/linspace_cn.html#linspace) | 功能一致，torch 参数更多，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.linspace.md)                |
| 75   | [torch.eye](https://pytorch.org/docs/stable/generated/torch.eye.html?highlight=eye#torch.eye) | [paddle.eye](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/eye_cn.html#eye) | 功能一致，torch 参数更多，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.eye.md)                     |
| 76   | [torch.bernoulli](https://pytorch.org/docs/stable/generated/torch.bernoulli.html?highlight=bernoulli#torch.bernoulli) | [paddle.bernoulli](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/bernoulli_cn.html#bernoulli) | 功能一致，torch 参数更多，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.bernoulli.md)               |
| 77   | [torch.multinomial](https://pytorch.org/docs/stable/generated/torch.multinomial.html?highlight=multinomial#torch.multinomial) | [paddle.multinomial](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/multinomial_cn.html#multinomial) | 功能一致，torch 参数更多，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.multinomial.md)             |
| 78   | [torch.normal](https://pytorch.org/docs/stable/generated/torch.normal.html?highlight=normal#torch.normal) | [paddle.normal](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/normal_cn.html#normal) | 功能一致，torch 参数更多，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.normal.md)                              |
| 79   | [torch.rand](https://pytorch.org/docs/stable/generated/torch.rand.html?highlight=rand#torch.rand) | [paddle.rand](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/rand_cn.html#rand) | 功能一致，torch 参数更多，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.rand.md)                                |
| 80   | [torch.randint](https://pytorch.org/docs/stable/generated/torch.randint.html?highlight=randint#torch.randint) | [paddle.randint](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/randint_cn.html#randint) | 功能一致，torch 参数更多，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.randint.md)                 |
| 81   | [torch.randperm](https://pytorch.org/docs/stable/generated/torch.randperm.html?highlight=randperm#torch.randperm) | [paddle.randperm](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/randperm_cn.html#randperm) | 功能一致，torch 参数更多，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.randperm.md)                |
| 82    | [torch.zeros](https://pytorch.org/docs/stable/generated/torch.zeros.html?highlight=zeros#torch.zeros) | [paddle.zeros](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/zeros_cn.html#zeros) | 功能一致，参数不一致，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.zeros.md)                               |
| 83   | [torch.nansum](https://pytorch.org/docs/stable/generated/torch.nansum.html?highlight=nansum#torch.nansum) | paddle.nansum | 功能一致，仅参数命名不一致，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.nansum.md)    |
| 84   | [torch.abs](https://pytorch.org/docs/stable/generated/torch.abs.html?highlight=abs#torch.abs) | [paddle.abs](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/abs_cn.html#abs) | 功能一致，torch 参数更多，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.abs.md)          |
| 85   | [torch.absolute](https://pytorch.org/docs/stable/generated/torch.absolute.html?highlight=absolute#torch.absolute) | [paddle.abs](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/abs_cn.html#abs) | 功能一致，torch 参数更多，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.absolute.md)          |
| 86   | [torch.acos](https://pytorch.org/docs/stable/generated/torch.acos.html?highlight=torch%20acos#torch.acos) | [paddle.acos](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/acos_cn.html#acos) | 功能一致，torch 参数更多，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.acos.md) |
| 87   | [torch.arccos](https://pytorch.org/docs/stable/generated/torch.arccos.html?highlight=arccos#torch.arccos) | [paddle.acos](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/acos_cn.html#acos) | 功能一致，torch 参数更多，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.arccos.md) |
| 88   | [torch.add](https://pytorch.org/docs/stable/generated/torch.add.html?highlight=add#torch.add) | [padle.add](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/add_cn.html#add) | 功能一致，torch 参数更多，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.add.md) |
| 89   | [torch.asin](https://pytorch.org/docs/stable/generated/torch.asin.html?highlight=asin#torch.asin) | [paddle.asin](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/asin_cn.html#asin) | 功能一致，torch 参数更多，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.asin.md) |
| 90   | [torch.arcsin](https://pytorch.org/docs/stable/generated/torch.arcsin.html?highlight=arcsin#torch.arcsin) | [paddle.asin](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/asin_cn.html#asin) | 功能一致，torch 参数更多，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.arcsin.md) |
| 91   | [torch.atan](https://pytorch.org/docs/stable/generated/torch.atan.html?highlight=atan#torch.atan) | [paddle.atan](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/atan_cn.html#atan) | 功能一致，torch 参数更多，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.atan.md) |
| 92   | [torch.arctan](https://pytorch.org/docs/stable/generated/torch.arctan.html?highlight=arctan#torch.arctan) | [paddle.atan](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/atan_cn.html#atan) | 功能一致，torch 参数更多，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.arctan.md) |
| 93   | [torch.ceil](https://pytorch.org/docs/stable/generated/torch.ceil.html?highlight=ceil#torch.ceil) | [paddle.ceil](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/ceil_cn.html#ceil) | 功能一致，torch 参数更多，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.ceil.md) |
| 94   | [torch.clamp](https://pytorch.org/docs/stable/generated/torch.clamp.html#torch.clamp) | [paddle.clip](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/clip_cn.html#clip) | 功能一致，torch 参数更多，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.clamp.md) |
| 95   | [torch.conj](https://pytorch.org/docs/stable/generated/torch.conj.html?highlight=conj#torch.conj) | [paddle.conj](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/conj_cn.html#conj) | 功能一致，仅参数命名不一致，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.conj.md) |
| 96   | [torch.cos](https://pytorch.org/docs/stable/generated/torch.cos.html?highlight=cos#torch.cos) | [paddle.cos](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/cos_cn.html#cos) | 功能一致，torch 参数更多，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.cos.md) |
| 97   | [torch.cosh](https://pytorch.org/docs/stable/generated/torch.cosh.html?highlight=cosh#torch.cosh) | [paddle.cosh](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/cosh_cn.html#cosh) | 功能一致，torch 参数更多，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.cosh.md) |
| 98   | [torch.div](https://pytorch.org/docs/stable/generated/torch.div.html?highlight=div#torch.div) | [paddle.divide](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/divide_cn.html#divide) | 功能一致，torch 参数更多，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.div.md)                                 |
| 99   | [torch.divide](https://pytorch.org/docs/stable/generated/torch.divide.html?highlight=divide#torch.divide) | [paddle.divide](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/divide_cn.html#divide) | 功能一致，torch 参数更多，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.divide.md)                              |
| 100   | [torch.erf](https://pytorch.org/docs/stable/generated/torch.erf.html?highlight=erf#torch.erf) | [paddle.erf](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/erf_cn.html#erf) | 功能一致，torch 参数更多，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.erf.md) |
| 101   | [torch.exp](https://pytorch.org/docs/stable/generated/torch.exp.html?highlight=exp#torch.exp) | [paddle.exp](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/exp_cn.html#exp) | 功能一致，torch 参数更多，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.exp.md) |
| 102   | [torch.floor](https://pytorch.org/docs/stable/generated/torch.floor.html?highlight=floor#torch.floor) | [paddle.floor](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/floor_cn.html#floor) | 功能一致，torch 参数更多，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.floor.md) |
| 103   | [torch.floor_divide](https://pytorch.org/docs/stable/generated/torch.floor_divide.html?highlight=floor_divide#torch.floor_divide) | [paddle.floor_divide](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/floor_divide_cn.html#floor-divide) | 功能一致，torch 参数更多，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.floor_divide.md) |
| 104   | [torch.fmod](https://pytorch.org/docs/stable/generated/torch.fmod.html?highlight=fmod#torch.fmod) | [paddle.mod](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/mod_cn.html#mod) | 功能一致，参数不一致，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.fmod.md) |
| 105   | [torch.log](https://pytorch.org/docs/stable/generated/torch.log.html?highlight=log#torch.log) | [paddle.log](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/log_cn.html#log) | 功能一致，torch 参数更多，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.log.md) |
| 106   | [torch.log10](https://pytorch.org/docs/stable/generated/torch.log10.html?highlight=log10#torch.log10) | [paddle.log10](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/log10_cn.html#log10) | 功能一致，torch 参数更多，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.log10.md) |
| 107   | [torch.log1p](https://pytorch.org/docs/stable/generated/torch.log1p.html?highlight=log1p#torch.log1p) | [paddle.log1p](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/log1p_cn.html#log1p) | 功能一致，torch 参数更多，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.log1p.md) |
| 108   | [torch.log2](https://pytorch.org/docs/stable/generated/torch.log2.html?highlight=log2#torch.log2) | [paddle.log2](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/log2_cn.html#log2) | 功能一致，torch 参数更多，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.log2.md) |
| 109   | [torch.mul](https://pytorch.org/docs/stable/generated/torch.mul.html?highlight=torch%20mul#torch.mul) | [paddle.multiply](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/multiply_cn.html#multiply) | 功能一致，torch 参数更多，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.mul.md) |
| 110   | [torch.multiply](https://pytorch.org/docs/stable/generated/torch.multiply.html?highlight=multiply#torch.multiply) | [paddle.multiply](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/multiply_cn.html#multiply) | 功能一致，torch 参数更多，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.multiply.md) |
| 111   | [torch.pow](https://pytorch.org/docs/stable/generated/torch.pow.html?highlight=pow#torch.pow) | [paddle.pow](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/pow_cn.html#pow) | 功能一致，torch 参数更多，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.pow.md) |
| 112   | [torch.reciprocal](https://pytorch.org/docs/stable/generated/torch.reciprocal.html?highlight=reciprocal#torch.reciprocal) | [paddle.reciprocal](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/reciprocal_cn.html#reciprocal) | 功能一致，torch 参数更多，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.reciprocal.md) |
| 113   | [torch.remainder](https://pytorch.org/docs/stable/generated/torch.remainder.html?highlight=remainder#torch.remainder) | [paddle.remainder](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/remainder_cn.html#remainder) | 功能一致，torch 参数更多，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.remainder.md) |
| 114   | [torch.round](https://pytorch.org/docs/stable/generated/torch.round.html?highlight=round#torch.round) | [paddle.round](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/round_cn.html#round) | 功能一致，torch 参数更多，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.round.md) |
| 115   | [torch.rsqrt](https://pytorch.org/docs/stable/generated/torch.rsqrt.html?highlight=rsqrt#torch.rsqrt) | [paddle.rsqrt](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/rsqrt_cn.html#rsqrt) | 功能一致，torch 参数更多，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.rsqrt.md) |
| 116   | [torch.sign](https://pytorch.org/docs/stable/generated/torch.sign.html?highlight=sign#torch.sign) | [paddle.sign](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/sign_cn.html#sign) | 功能一致，torch 参数更多，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.sign.md) |
| 117   | [torch.sin](https://pytorch.org/docs/stable/generated/torch.sin.html?highlight=sin#torch.sin) | [paddle.sin](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/sin_cn.html#sin) | 功能一致，torch 参数更多，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.sin.md) |
| 118   | [torch.sinh](https://pytorch.org/docs/stable/generated/torch.sinh.html?highlight=sinh#torch.sinh) | [paddle.sinh](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/sinh_cn.html#sinh) | 功能一致，torch 参数更多，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.sinh.md) |
| 119   | [torch.max](https://pytorch.org/docs/stable/generated/torch.max.html?highlight=max#torch.max) | [paddle.max](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/max_cn.html#max)/[paddle.maximum](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/maximum_cn.html#maximum) | 功能一致，torch 参数更多，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.max.md) |
| 120   | [torch.min](https://pytorch.org/docs/stable/generated/torch.min.html?highlight=min#torch.min) | [paddle.min](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/min_cn.html#min)/[paddle.minimum](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/minimum_cn.html#minimum) | 功能一致，torch 参数更多，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.min.md) |
| 121   | [torch.gather](https://pytorch.org/docs/stable/generated/torch.gather.html?highlight=gather#torch.gather) | [paddle.take_along_axis](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/take_along_axis_cn.html#take-along-axis) | 功能一致，torch 参数更多，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.gather.md)                              |
| 122   | [torch.narrow](https://pytorch.org/docs/stable/generated/torch.narrow.html?highlight=narrow#torch.narrow) | [paddle.slice](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/slice_cn.html#slice) | 功能一致，参数不一致，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.narrow.md)                              |
| 123   | [torch.transpose](https://pytorch.org/docs/stable/generated/torch.transpose.html?highlight=transpose#torch.transpose) | [paddle.transpose](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/transpose_cn.html#transpose) | 功能一致，参数不一致，[差异对比](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/ops/torch.transpose.md)                           |


***持续更新...***

## torch.nn.XX API 映射列表

梳理了`torch.nn.XX`类 API 的 PyTorch-PaddlePaddle API 映射列表。

| 序号 | PyTorch API                                                  | PaddlePaddle API                                             | 备注                                                         |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |


***持续更新...***

## torch.nn.functional.XX API 映射列表
梳理了`torch.nn.functional.XX`类 API 的 PyTorch-PaddlePaddle API 映射列表。

| 序号 | PyTorch API                                                  | PaddlePaddle API                                             | 备注                                                         |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |



***持续更新...***

## torch.Tensor.XX API 映射列表
梳理了`torch.Tensor.XX`类 API 的 PyTorch-PaddlePaddle API 映射列表。

| 序号 | PyTorch API                                                  | PaddlePaddle API                                             | 备注                                                         |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |



***持续更新...***

## torch.nn.init.XX API 映射列表
梳理了`torch.nn.init.XX`类 API 的 PyTorch-PaddlePaddle API 映射列表。

| 序号 | PyTorch API                                                  | PaddlePaddle API                                             | 备注                                                         |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |


***持续更新...***

## torch.nn.utils.XX API 映射列表
梳理了`torch.nn.utils.XX`类 API 的 PyTorch-PaddlePaddle API 映射列表。

| 序号 | PyTorch API                                                  | PaddlePaddle API                                             | 备注                                                         |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |



***持续更新...***

## torch.cuda.XX API 映射列表
梳理了`torch.cuda.XX`类 API 的 PyTorch-PaddlePaddle API 映射列表。

| 序号 | PyTorch API                                                  | PaddlePaddle API                                             | 备注                                                         |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |


***持续更新...***

## torch.distributed.XX API 映射列表
梳理了`torch.distributed.XX`类 API 的 PyTorch-PaddlePaddle API 映射列表。

| 序号 | PyTorch API                                                  | PaddlePaddle API                                             | 备注                                                         |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |


***持续更新...***

## torch.distributions.XX API 映射列表
梳理了`torch.distributions.XX`类 API 的 PyTorch-PaddlePaddle API 映射列表。

| 序号 | PyTorch API                                                  | PaddlePaddle API                                             | 备注                                                         |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |


***持续更新...***

## torch.fft.XX API 映射列表
梳理了`torch.fft.XX`类 API 的 PyTorch-PaddlePaddle API 映射列表。

| 序号 | PyTorch API                                                  | PaddlePaddle API                                             | 备注                                                         |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |


***持续更新...***

## torch.linalg.XX API 映射列表
梳理了`torch.linalg.XX`类 API 的 PyTorch-PaddlePaddle API 映射列表。

| 序号 | PyTorch API                                                  | PaddlePaddle API                                             | 备注                                                         |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |


***持续更新...***

## 其他类 API 映射列表
梳理了其他类 API 的 PyTorch-PaddlePaddle API 映射列表。

| 序号 | PyTorch API                                                  | PaddlePaddle API                                             | 备注                                                         |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |


***持续更新...***
