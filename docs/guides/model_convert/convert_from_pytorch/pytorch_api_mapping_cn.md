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
| [其他](#id13)   | 其他 API |

## torch.XX API 映射列表

梳理了`torch.XX`类 API 的 PyTorch-PaddlePaddle API 映射列表。

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ----- | ----------- | ----------------- | ----------- | ------- |
| REFERENCE-MAPPING-ITEM(`torch.BoolTensor`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.BoolTensor.md) |
| REFERENCE-MAPPING-ITEM(`torch.ByteTensor`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.ByteTensor.md) |
| REFERENCE-MAPPING-ITEM(`torch.DoubleTensor`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.DoubleTensor.md) |
| REFERENCE-MAPPING-ITEM(`torch.FloatTensor`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.FloatTensor.md) |
| REFERENCE-MAPPING-ITEM(`torch.Generator`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.Generator.md) |
| REFERENCE-MAPPING-ITEM(`torch.HalfTensor`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.HalfTensor.md) |
| REFERENCE-MAPPING-ITEM(`torch.IntTensor`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.IntTensor.md) |
| REFERENCE-MAPPING-ITEM(`torch.LongTensor`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.LongTensor.md) |
| REFERENCE-MAPPING-ITEM(`torch.ShortTensor`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.ShortTensor.md) |
| REFERENCE-MAPPING-ITEM(`torch.Size`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.Size.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.Tensor__upper.md) |
| REFERENCE-MAPPING-ITEM(`torch.__version__`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.__version__.md) |
| REFERENCE-MAPPING-ITEM(`torch.abs`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.abs.md) |
| REFERENCE-MAPPING-ITEM(`torch.abs_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.abs_.md) |
| REFERENCE-MAPPING-ITEM(`torch.absolute`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.absolute.md) |
| REFERENCE-MAPPING-ITEM(`torch.acos`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.acos.md) |
| REFERENCE-MAPPING-ITEM(`torch.acosh`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.acosh.md) |
| REFERENCE-MAPPING-ITEM(`torch.add`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.add.md) |
| REFERENCE-MAPPING-ITEM(`torch.addbmm`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.addbmm.md) |
| REFERENCE-MAPPING-ITEM(`torch.addcdiv`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.addcdiv.md) |
| REFERENCE-MAPPING-ITEM(`torch.addcmul`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.addcmul.md) |
| REFERENCE-MAPPING-ITEM(`torch.addmm`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.addmm.md) |
| REFERENCE-MAPPING-ITEM(`torch.addmv`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.addmv.md) |
| REFERENCE-MAPPING-ITEM(`torch.addr`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.addr.md) |
| REFERENCE-MAPPING-ITEM(`torch.adjoint`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.adjoint.md) |
| REFERENCE-MAPPING-ITEM(`torch.all`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.all.md) |
| REFERENCE-MAPPING-ITEM(`torch.allclose`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.allclose.md) |
| REFERENCE-MAPPING-ITEM(`torch.alpha_dropout`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.alpha_dropout.md) |
| REFERENCE-MAPPING-ITEM(`torch.amax`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.amax.md) |
| REFERENCE-MAPPING-ITEM(`torch.amin`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.amin.md) |
| REFERENCE-MAPPING-ITEM(`torch.aminmax`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.aminmax.md) |
| REFERENCE-MAPPING-ITEM(`torch.angle`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.angle.md) |
| REFERENCE-MAPPING-ITEM(`torch.any`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.any.md) |
| REFERENCE-MAPPING-ITEM(`torch.arange`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.arange.md) |
| REFERENCE-MAPPING-ITEM(`torch.arccos`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.arccos.md) |
| REFERENCE-MAPPING-ITEM(`torch.arccosh`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.arccosh.md) |
| REFERENCE-MAPPING-ITEM(`torch.arcsin`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.arcsin.md) |
| REFERENCE-MAPPING-ITEM(`torch.arcsinh`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.arcsinh.md) |
| REFERENCE-MAPPING-ITEM(`torch.arctan`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.arctan.md) |
| REFERENCE-MAPPING-ITEM(`torch.arctan2`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.arctan2.md) |
| REFERENCE-MAPPING-ITEM(`torch.arctanh`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.arctanh.md) |
| REFERENCE-MAPPING-ITEM(`torch.argmax`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.argmax.md) |
| REFERENCE-MAPPING-ITEM(`torch.argmin`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.argmin.md) |
| REFERENCE-MAPPING-ITEM(`torch.argsort`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.argsort.md) |
| REFERENCE-MAPPING-ITEM(`torch.argwhere`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.argwhere.md) |
| REFERENCE-MAPPING-ITEM(`torch.as_strided`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.as_strided.md) |
| REFERENCE-MAPPING-ITEM(`torch.as_tensor`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.as_tensor.md) |
| REFERENCE-MAPPING-ITEM(`torch.asarray`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.asarray.md) |
| REFERENCE-MAPPING-ITEM(`torch.asin`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.asin.md) |
| REFERENCE-MAPPING-ITEM(`torch.asinh`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.asinh.md) |
| REFERENCE-MAPPING-ITEM(`torch.atan`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.atan.md) |
| REFERENCE-MAPPING-ITEM(`torch.atan2`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.atan2.md) |
| REFERENCE-MAPPING-ITEM(`torch.atanh`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.atanh.md) |
| REFERENCE-MAPPING-ITEM(`torch.atleast_1d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.atleast_1d.md) |
| REFERENCE-MAPPING-ITEM(`torch.atleast_2d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.atleast_2d.md) |
| REFERENCE-MAPPING-ITEM(`torch.atleast_3d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.atleast_3d.md) |
| REFERENCE-MAPPING-ITEM(`torch.autocast`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.autocast.md) |
| REFERENCE-MAPPING-ITEM(`torch.baddbmm`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.baddbmm.md) |
| REFERENCE-MAPPING-ITEM(`torch.bernoulli`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.bernoulli.md) |
| REFERENCE-MAPPING-ITEM(`torch.bincount`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.bincount.md) |
| REFERENCE-MAPPING-ITEM(`torch.bitwise_and`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.bitwise_and.md) |
| REFERENCE-MAPPING-ITEM(`torch.bitwise_not`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.bitwise_not.md) |
| REFERENCE-MAPPING-ITEM(`torch.bitwise_or`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.bitwise_or.md) |
| REFERENCE-MAPPING-ITEM(`torch.bitwise_xor`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.bitwise_xor.md) |
| REFERENCE-MAPPING-ITEM(`torch.bmm`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.bmm.md) |
| REFERENCE-MAPPING-ITEM(`torch.broadcast_shapes`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.broadcast_shapes.md) |
| REFERENCE-MAPPING-ITEM(`torch.broadcast_tensors`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.broadcast_tensors.md) |
| REFERENCE-MAPPING-ITEM(`torch.broadcast_to`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.broadcast_to.md) |
| REFERENCE-MAPPING-ITEM(`torch.bucketize`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.bucketize.md) |
| REFERENCE-MAPPING-ITEM(`torch.cat`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.cat.md) |
| REFERENCE-MAPPING-ITEM(`torch.cdist`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.cdist.md) |
| REFERENCE-MAPPING-ITEM(`torch.ceil`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.ceil.md) |
| REFERENCE-MAPPING-ITEM(`torch.celu`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.celu.md) |
| REFERENCE-MAPPING-ITEM(`torch.chain_matmul`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.chain_matmul.md) |
| REFERENCE-MAPPING-ITEM(`torch.cholesky`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.cholesky.md) |
| REFERENCE-MAPPING-ITEM(`torch.cholesky_inverse`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.cholesky_inverse.md) |
| REFERENCE-MAPPING-ITEM(`torch.cholesky_solve`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.cholesky_solve.md) |
| REFERENCE-MAPPING-ITEM(`torch.chunk`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.chunk.md) |
| REFERENCE-MAPPING-ITEM(`torch.clamp`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.clamp.md) |
| REFERENCE-MAPPING-ITEM(`torch.clamp_max`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.clamp_max.md) |
| REFERENCE-MAPPING-ITEM(`torch.clamp_min`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.clamp_min.md) |
| REFERENCE-MAPPING-ITEM(`torch.clip`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.clip.md) |
| REFERENCE-MAPPING-ITEM(`torch.clone`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.clone.md) |
| REFERENCE-MAPPING-ITEM(`torch.column_stack`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.column_stack.md) |
| REFERENCE-MAPPING-ITEM(`torch.combinations`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.combinations.md) |
| REFERENCE-MAPPING-ITEM(`torch.complex`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.complex.md) |
| REFERENCE-MAPPING-ITEM(`torch.concat`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.concat.md) |
| REFERENCE-MAPPING-ITEM(`torch.conj`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.conj.md) |
| REFERENCE-MAPPING-ITEM(`torch.conj_physical`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.conj_physical.md) |
| REFERENCE-MAPPING-ITEM(`torch.copysign`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.copysign.md) |
| REFERENCE-MAPPING-ITEM(`torch.corrcoef`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.corrcoef.md) |
| REFERENCE-MAPPING-ITEM(`torch.cos`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.cos.md) |
| REFERENCE-MAPPING-ITEM(`torch.cosh`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.cosh.md) |
| REFERENCE-MAPPING-ITEM(`torch.cosine_similarity`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.cosine_similarity.md) |
| REFERENCE-MAPPING-ITEM(`torch.count_nonzero`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.count_nonzero.md) |
| REFERENCE-MAPPING-ITEM(`torch.cov`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.cov.md) |
| REFERENCE-MAPPING-ITEM(`torch.cross`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.cross.md) |
| REFERENCE-MAPPING-ITEM(`torch.cummax`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.cummax.md) |
| REFERENCE-MAPPING-ITEM(`torch.cummin`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.cummin.md) |
| REFERENCE-MAPPING-ITEM(`torch.cumprod`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.cumprod.md) |
| REFERENCE-MAPPING-ITEM(`torch.cumsum`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.cumsum.md) |
| REFERENCE-MAPPING-ITEM(`torch.cumulative_trapezoid`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.cumulative_trapezoid.md) |
| REFERENCE-MAPPING-ITEM(`torch.deg2rad`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.deg2rad.md) |
| REFERENCE-MAPPING-ITEM(`torch.det`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.det.md) |
| REFERENCE-MAPPING-ITEM(`torch.device`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.device.md) |
| REFERENCE-MAPPING-ITEM(`torch.diag`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.diag.md) |
| REFERENCE-MAPPING-ITEM(`torch.diag_embed`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.diag_embed.md) |
| REFERENCE-MAPPING-ITEM(`torch.diagflat`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.diagflat.md) |
| REFERENCE-MAPPING-ITEM(`torch.diagonal`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.diagonal.md) |
| REFERENCE-MAPPING-ITEM(`torch.diagonal_scatter`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.diagonal_scatter.md) |
| REFERENCE-MAPPING-ITEM(`torch.diff`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.diff.md) |
| REFERENCE-MAPPING-ITEM(`torch.digamma`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.digamma.md) |
| REFERENCE-MAPPING-ITEM(`torch.dist`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.dist.md) |
| REFERENCE-MAPPING-ITEM(`torch.div`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.div.md) |
| REFERENCE-MAPPING-ITEM(`torch.divide`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.divide.md) |
| REFERENCE-MAPPING-ITEM(`torch.dot`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.dot.md) |
| REFERENCE-MAPPING-ITEM(`torch.dropout`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.dropout.md) |
| REFERENCE-MAPPING-ITEM(`torch.dsplit`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.dsplit.md) |
| REFERENCE-MAPPING-ITEM(`torch.dstack`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.dstack.md) |
| REFERENCE-MAPPING-ITEM(`torch.einsum`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.einsum.md) |
| REFERENCE-MAPPING-ITEM(`torch.empty`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.empty.md) |
| REFERENCE-MAPPING-ITEM(`torch.empty_like`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.empty_like.md) |
| REFERENCE-MAPPING-ITEM(`torch.enable_grad`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.enable_grad.md) |
| REFERENCE-MAPPING-ITEM(`torch.eq`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.eq.md) |
| REFERENCE-MAPPING-ITEM(`torch.equal`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.equal.md) |
| REFERENCE-MAPPING-ITEM(`torch.erf`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.erf.md) |
| REFERENCE-MAPPING-ITEM(`torch.erfc`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.erfc.md) |
| REFERENCE-MAPPING-ITEM(`torch.erfinv`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.erfinv.md) |
| REFERENCE-MAPPING-ITEM(`torch.exp`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.exp.md) |
| REFERENCE-MAPPING-ITEM(`torch.exp2`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.exp2.md) |
| REFERENCE-MAPPING-ITEM(`torch.expm1`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.expm1.md) |
| REFERENCE-MAPPING-ITEM(`torch.eye`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.eye.md) |
| REFERENCE-MAPPING-ITEM(`torch.finfo`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.finfo.md) |
| REFERENCE-MAPPING-ITEM(`torch.fix`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.fix.md) |
| REFERENCE-MAPPING-ITEM(`torch.flatten`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.flatten.md) |
| REFERENCE-MAPPING-ITEM(`torch.flip`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.flip.md) |
| REFERENCE-MAPPING-ITEM(`torch.fliplr`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.fliplr.md) |
| REFERENCE-MAPPING-ITEM(`torch.flipud`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.flipud.md) |
| REFERENCE-MAPPING-ITEM(`torch.floor`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.floor.md) |
| REFERENCE-MAPPING-ITEM(`torch.floor_divide`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.floor_divide.md) |
| REFERENCE-MAPPING-ITEM(`torch.fmax`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.fmax.md) |
| REFERENCE-MAPPING-ITEM(`torch.fmin`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.fmin.md) |
| REFERENCE-MAPPING-ITEM(`torch.fmod`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.fmod.md) |
| REFERENCE-MAPPING-ITEM(`torch.frac`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.frac.md) |
| REFERENCE-MAPPING-ITEM(`torch.frexp`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.frexp.md) |
| REFERENCE-MAPPING-ITEM(`torch.from_dlpack`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.from_dlpack.md) |
| REFERENCE-MAPPING-ITEM(`torch.from_numpy`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.from_numpy.md) |
| REFERENCE-MAPPING-ITEM(`torch.full`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.full.md) |
| REFERENCE-MAPPING-ITEM(`torch.full_like`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.full_like.md) |
| REFERENCE-MAPPING-ITEM(`torch.gather`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.gather.md) |
| REFERENCE-MAPPING-ITEM(`torch.gcd`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.gcd.md) |
| REFERENCE-MAPPING-ITEM(`torch.ge`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.ge.md) |
| REFERENCE-MAPPING-ITEM(`torch.ger`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.ger.md) |
| REFERENCE-MAPPING-ITEM(`torch.get_default_dtype`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.get_default_dtype.md) |
| REFERENCE-MAPPING-ITEM(`torch.get_rng_state`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.get_rng_state.md) |
| REFERENCE-MAPPING-ITEM(`torch.greater`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.greater.md) |
| REFERENCE-MAPPING-ITEM(`torch.gt`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.gt.md) |
| REFERENCE-MAPPING-ITEM(`torch.heaviside`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.heaviside.md) |
| REFERENCE-MAPPING-ITEM(`torch.histc`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.histc.md) |
| REFERENCE-MAPPING-ITEM(`torch.histogram`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.histogram.md) |
| REFERENCE-MAPPING-ITEM(`torch.histogramdd`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.histogramdd.md) |
| REFERENCE-MAPPING-ITEM(`torch.hsplit`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.hsplit.md) |
| REFERENCE-MAPPING-ITEM(`torch.hstack`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.hstack.md) |
| REFERENCE-MAPPING-ITEM(`torch.hypot`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.hypot.md) |
| REFERENCE-MAPPING-ITEM(`torch.i0`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.i0.md) |
| REFERENCE-MAPPING-ITEM(`torch.iinfo`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.iinfo.md) |
| REFERENCE-MAPPING-ITEM(`torch.imag`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.imag.md) |
| REFERENCE-MAPPING-ITEM(`torch.index_add`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.index_add.md) |
| REFERENCE-MAPPING-ITEM(`torch.index_copy`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.index_copy.md) |
| REFERENCE-MAPPING-ITEM(`torch.index_fill`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.index_fill.md) |
| REFERENCE-MAPPING-ITEM(`torch.index_select`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.index_select.md) |
| REFERENCE-MAPPING-ITEM(`torch.inference_mode`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.inference_mode.md) |
| REFERENCE-MAPPING-ITEM(`torch.initial_seed`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.initial_seed.md) |
| REFERENCE-MAPPING-ITEM(`torch.inner`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.inner.md) |
| REFERENCE-MAPPING-ITEM(`torch.inverse`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.inverse.md) |
| REFERENCE-MAPPING-ITEM(`torch.is_complex`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.is_complex.md) |
| REFERENCE-MAPPING-ITEM(`torch.is_floating_point`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.is_floating_point.md) |
| REFERENCE-MAPPING-ITEM(`torch.is_grad_enabled`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.is_grad_enabled.md) |
| REFERENCE-MAPPING-ITEM(`torch.is_nonzero`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.is_nonzero.md) |
| REFERENCE-MAPPING-ITEM(`torch.is_tensor`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.is_tensor.md) |
| REFERENCE-MAPPING-ITEM(`torch.isclose`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.isclose.md) |
| REFERENCE-MAPPING-ITEM(`torch.isfinite`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.isfinite.md) |
| REFERENCE-MAPPING-ITEM(`torch.isinf`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.isinf.md) |
| REFERENCE-MAPPING-ITEM(`torch.isnan`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.isnan.md) |
| REFERENCE-MAPPING-ITEM(`torch.istft`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.istft.md) |
| REFERENCE-MAPPING-ITEM(`torch.kron`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.kron.md) |
| REFERENCE-MAPPING-ITEM(`torch.kthvalue`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.kthvalue.md) |
| REFERENCE-MAPPING-ITEM(`torch.lcm`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.lcm.md) |
| REFERENCE-MAPPING-ITEM(`torch.ldexp`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.ldexp.md) |
| REFERENCE-MAPPING-ITEM(`torch.le`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.le.md) |
| REFERENCE-MAPPING-ITEM(`torch.lerp`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.lerp.md) |
| REFERENCE-MAPPING-ITEM(`torch.less`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.less.md) |
| REFERENCE-MAPPING-ITEM(`torch.less_equal`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.less_equal.md) |
| REFERENCE-MAPPING-ITEM(`torch.lgamma`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.lgamma.md) |
| REFERENCE-MAPPING-ITEM(`torch.linspace`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.linspace.md) |
| REFERENCE-MAPPING-ITEM(`torch.load`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.load.md) |
| REFERENCE-MAPPING-ITEM(`torch.log`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.log.md) |
| REFERENCE-MAPPING-ITEM(`torch.log10`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.log10.md) |
| REFERENCE-MAPPING-ITEM(`torch.log1p`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.log1p.md) |
| REFERENCE-MAPPING-ITEM(`torch.log2`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.log2.md) |
| REFERENCE-MAPPING-ITEM(`torch.logaddexp`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.logaddexp.md) |
| REFERENCE-MAPPING-ITEM(`torch.logaddexp2`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.logaddexp2.md) |
| REFERENCE-MAPPING-ITEM(`torch.logcumsumexp`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.logcumsumexp.md) |
| REFERENCE-MAPPING-ITEM(`torch.logdet`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.logdet.md) |
| REFERENCE-MAPPING-ITEM(`torch.logical_and`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.logical_and.md) |
| REFERENCE-MAPPING-ITEM(`torch.logical_not`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.logical_not.md) |
| REFERENCE-MAPPING-ITEM(`torch.logical_or`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.logical_or.md) |
| REFERENCE-MAPPING-ITEM(`torch.logical_xor`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.logical_xor.md) |
| REFERENCE-MAPPING-ITEM(`torch.logit`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.logit.md) |
| REFERENCE-MAPPING-ITEM(`torch.logspace`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.logspace.md) |
| REFERENCE-MAPPING-ITEM(`torch.lt`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.lt.md) |
| REFERENCE-MAPPING-ITEM(`torch.lu`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.lu.md) |
| REFERENCE-MAPPING-ITEM(`torch.lu_unpack`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.lu_unpack.md) |
| REFERENCE-MAPPING-ITEM(`torch.manual_seed`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.manual_seed.md) |
| REFERENCE-MAPPING-ITEM(`torch.masked_fill`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.masked_fill.md) |
| REFERENCE-MAPPING-ITEM(`torch.masked_select`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.masked_select.md) |
| REFERENCE-MAPPING-ITEM(`torch.matmul`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.matmul.md) |
| REFERENCE-MAPPING-ITEM(`torch.matrix_power`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.matrix_power.md) |
| REFERENCE-MAPPING-ITEM(`torch.max`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.max.md) |
| REFERENCE-MAPPING-ITEM(`torch.max_pool1d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.max_pool1d.md) |
| REFERENCE-MAPPING-ITEM(`torch.max_pool2d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.max_pool2d.md) |
| REFERENCE-MAPPING-ITEM(`torch.max_pool3d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.max_pool3d.md) |
| REFERENCE-MAPPING-ITEM(`torch.maximum`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.maximum.md) |
| REFERENCE-MAPPING-ITEM(`torch.mean`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.mean.md) |
| REFERENCE-MAPPING-ITEM(`torch.median`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.median.md) |
| REFERENCE-MAPPING-ITEM(`torch.meshgrid`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.meshgrid.md) |
| REFERENCE-MAPPING-ITEM(`torch.min`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.min.md) |
| REFERENCE-MAPPING-ITEM(`torch.minimum`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.minimum.md) |
| REFERENCE-MAPPING-ITEM(`torch.mm`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.mm.md) |
| REFERENCE-MAPPING-ITEM(`torch.mode`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.mode.md) |
| REFERENCE-MAPPING-ITEM(`torch.moveaxis`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.moveaxis.md) |
| REFERENCE-MAPPING-ITEM(`torch.movedim`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.movedim.md) |
| REFERENCE-MAPPING-ITEM(`torch.msort`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.msort.md) |
| REFERENCE-MAPPING-ITEM(`torch.mul`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.mul.md) |
| REFERENCE-MAPPING-ITEM(`torch.multinomial`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.multinomial.md) |
| REFERENCE-MAPPING-ITEM(`torch.multiply`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.multiply.md) |
| REFERENCE-MAPPING-ITEM(`torch.mv`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.mv.md) |
| REFERENCE-MAPPING-ITEM(`torch.nan_to_num`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.nan_to_num.md) |
| REFERENCE-MAPPING-ITEM(`torch.nanmean`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.nanmean.md) |
| REFERENCE-MAPPING-ITEM(`torch.nanmedian`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.nanmedian.md) |
| REFERENCE-MAPPING-ITEM(`torch.nanquantile`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.nanquantile.md) |
| REFERENCE-MAPPING-ITEM(`torch.nansum`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.nansum.md) |
| REFERENCE-MAPPING-ITEM(`torch.narrow`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.narrow.md) |
| REFERENCE-MAPPING-ITEM(`torch.narrow_copy`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.narrow_copy.md) |
| REFERENCE-MAPPING-ITEM(`torch.ne`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.ne.md) |
| REFERENCE-MAPPING-ITEM(`torch.neg`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.neg.md) |
| REFERENCE-MAPPING-ITEM(`torch.negative`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.negative.md) |
| REFERENCE-MAPPING-ITEM(`torch.nextafter`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.nextafter.md) |
| REFERENCE-MAPPING-ITEM(`torch.no_grad`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.no_grad.md) |
| REFERENCE-MAPPING-ITEM(`torch.nonzero`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.nonzero.md) |
| REFERENCE-MAPPING-ITEM(`torch.norm`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.norm.md) |
| REFERENCE-MAPPING-ITEM(`torch.normal`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.normal.md) |
| REFERENCE-MAPPING-ITEM(`torch.not_equal`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.not_equal.md) |
| REFERENCE-MAPPING-ITEM(`torch.numel`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.numel.md) |
| REFERENCE-MAPPING-ITEM(`torch.ones`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.ones.md) |
| REFERENCE-MAPPING-ITEM(`torch.ones_like`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.ones_like.md) |
| REFERENCE-MAPPING-ITEM(`torch.outer`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.outer.md) |
| REFERENCE-MAPPING-ITEM(`torch.pca_lowrank`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.pca_lowrank.md) |
| REFERENCE-MAPPING-ITEM(`torch.permute`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.permute.md) |
| REFERENCE-MAPPING-ITEM(`torch.pinverse`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.pinverse.md) |
| REFERENCE-MAPPING-ITEM(`torch.poisson`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.poisson.md) |
| REFERENCE-MAPPING-ITEM(`torch.polar`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.polar.md) |
| REFERENCE-MAPPING-ITEM(`torch.polygamma`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.polygamma.md) |
| REFERENCE-MAPPING-ITEM(`torch.pow`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.pow.md) |
| REFERENCE-MAPPING-ITEM(`torch.prod`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.prod.md) |
| REFERENCE-MAPPING-ITEM(`torch.qr`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.qr.md) |
| REFERENCE-MAPPING-ITEM(`torch.quantile`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.quantile.md) |
| REFERENCE-MAPPING-ITEM(`torch.rad2deg`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.rad2deg.md) |
| REFERENCE-MAPPING-ITEM(`torch.rand`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.rand.md) |
| REFERENCE-MAPPING-ITEM(`torch.rand_like`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.rand_like.md) |
| REFERENCE-MAPPING-ITEM(`torch.randint`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.randint.md) |
| REFERENCE-MAPPING-ITEM(`torch.randint_like`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.randint_like.md) |
| REFERENCE-MAPPING-ITEM(`torch.randn`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.randn.md) |
| REFERENCE-MAPPING-ITEM(`torch.randn_like`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.randn_like.md) |
| REFERENCE-MAPPING-ITEM(`torch.randperm`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.randperm.md) |
| REFERENCE-MAPPING-ITEM(`torch.range`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.range.md) |
| REFERENCE-MAPPING-ITEM(`torch.ravel`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.ravel.md) |
| REFERENCE-MAPPING-ITEM(`torch.real`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.real.md) |
| REFERENCE-MAPPING-ITEM(`torch.reciprocal`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.reciprocal.md) |
| REFERENCE-MAPPING-ITEM(`torch.relu`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.relu.md) |
| REFERENCE-MAPPING-ITEM(`torch.remainder`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.remainder.md) |
| REFERENCE-MAPPING-ITEM(`torch.renorm`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.renorm.md) |
| REFERENCE-MAPPING-ITEM(`torch.repeat_interleave`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.repeat_interleave.md) |
| REFERENCE-MAPPING-ITEM(`torch.reshape`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.reshape.md) |
| REFERENCE-MAPPING-ITEM(`torch.roll`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.roll.md) |
| REFERENCE-MAPPING-ITEM(`torch.rot90`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.rot90.md) |
| REFERENCE-MAPPING-ITEM(`torch.round`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.round.md) |
| REFERENCE-MAPPING-ITEM(`torch.row_stack`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.row_stack.md) |
| REFERENCE-MAPPING-ITEM(`torch.rrelu`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.rrelu.md) |
| REFERENCE-MAPPING-ITEM(`torch.rsqrt`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.rsqrt.md) |
| REFERENCE-MAPPING-ITEM(`torch.save`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.save.md) |
| REFERENCE-MAPPING-ITEM(`torch.scalar_tensor`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.scalar_tensor.md) |
| REFERENCE-MAPPING-ITEM(`torch.scatter`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.scatter.md) |
| REFERENCE-MAPPING-ITEM(`torch.scatter_add`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.scatter_add.md) |
| REFERENCE-MAPPING-ITEM(`torch.searchsorted`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.searchsorted.md) |
| REFERENCE-MAPPING-ITEM(`torch.seed`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.seed.md) |
| REFERENCE-MAPPING-ITEM(`torch.select`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.select.md) |
| REFERENCE-MAPPING-ITEM(`torch.select_scatter`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.select_scatter.md) |
| REFERENCE-MAPPING-ITEM(`torch.selu`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.selu.md) |
| REFERENCE-MAPPING-ITEM(`torch.set_default_dtype`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.set_default_dtype.md) |
| REFERENCE-MAPPING-ITEM(`torch.set_default_tensor_type`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.set_default_tensor_type.md) |
| REFERENCE-MAPPING-ITEM(`torch.set_grad_enabled`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.set_grad_enabled.md) |
| REFERENCE-MAPPING-ITEM(`torch.set_printoptions`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.set_printoptions.md) |
| REFERENCE-MAPPING-ITEM(`torch.set_rng_state`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.set_rng_state.md) |
| REFERENCE-MAPPING-ITEM(`torch.sgn`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.sgn.md) |
| REFERENCE-MAPPING-ITEM(`torch.sigmoid`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.sigmoid.md) |
| REFERENCE-MAPPING-ITEM(`torch.sign`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.sign.md) |
| REFERENCE-MAPPING-ITEM(`torch.signbit`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.signbit.md) |
| REFERENCE-MAPPING-ITEM(`torch.sin`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.sin.md) |
| REFERENCE-MAPPING-ITEM(`torch.sinc`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.sinc.md) |
| REFERENCE-MAPPING-ITEM(`torch.sinh`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.sinh.md) |
| REFERENCE-MAPPING-ITEM(`torch.slice_scatter`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.slice_scatter.md) |
| REFERENCE-MAPPING-ITEM(`torch.slogdet`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.slogdet.md) |
| REFERENCE-MAPPING-ITEM(`torch.softmax`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.softmax.md) |
| REFERENCE-MAPPING-ITEM(`torch.sort`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.sort.md) |
| REFERENCE-MAPPING-ITEM(`torch.sparse_coo_tensor`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.sparse_coo_tensor.md) |
| REFERENCE-MAPPING-ITEM(`torch.sparse_csr_tensor`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.sparse_csr_tensor.md) |
| REFERENCE-MAPPING-ITEM(`torch.split`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.split.md) |
| REFERENCE-MAPPING-ITEM(`torch.sqrt`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.sqrt.md) |
| REFERENCE-MAPPING-ITEM(`torch.square`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.square.md) |
| REFERENCE-MAPPING-ITEM(`torch.squeeze`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.squeeze.md) |
| REFERENCE-MAPPING-ITEM(`torch.stack`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.stack.md) |
| REFERENCE-MAPPING-ITEM(`torch.std`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.std.md) |
| REFERENCE-MAPPING-ITEM(`torch.std_mean`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.std_mean.md) |
| REFERENCE-MAPPING-ITEM(`torch.stft`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.stft.md) |
| REFERENCE-MAPPING-ITEM(`torch.sub`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.sub.md) |
| REFERENCE-MAPPING-ITEM(`torch.subtract`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.subtract.md) |
| REFERENCE-MAPPING-ITEM(`torch.sum`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.sum.md) |
| REFERENCE-MAPPING-ITEM(`torch.svd`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.svd.md) |
| REFERENCE-MAPPING-ITEM(`torch.svd_lowrank`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.svd_lowrank.md) |
| REFERENCE-MAPPING-ITEM(`torch.swapaxes`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.swapaxes.md) |
| REFERENCE-MAPPING-ITEM(`torch.swapdims`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.swapdims.md) |
| REFERENCE-MAPPING-ITEM(`torch.symeig`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.symeig.md) |
| REFERENCE-MAPPING-ITEM(`torch.t`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.t.md) |
| REFERENCE-MAPPING-ITEM(`torch.take`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.take.md) |
| REFERENCE-MAPPING-ITEM(`torch.take_along_dim`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.take_along_dim.md) |
| REFERENCE-MAPPING-ITEM(`torch.tan`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.tan.md) |
| REFERENCE-MAPPING-ITEM(`torch.tensor`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.tensor.md) |
| REFERENCE-MAPPING-ITEM(`torch.tensor_split`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.tensor_split.md) |
| REFERENCE-MAPPING-ITEM(`torch.tensordot`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.tensordot.md) |
| REFERENCE-MAPPING-ITEM(`torch.tile`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.tile.md) |
| REFERENCE-MAPPING-ITEM(`torch.topk`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.topk.md) |
| REFERENCE-MAPPING-ITEM(`torch.trace`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.trace.md) |
| REFERENCE-MAPPING-ITEM(`torch.transpose`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.transpose.md) |
| REFERENCE-MAPPING-ITEM(`torch.trapezoid`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.trapezoid.md) |
| REFERENCE-MAPPING-ITEM(`torch.triangular_solve`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.triangular_solve.md) |
| REFERENCE-MAPPING-ITEM(`torch.tril`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.tril.md) |
| REFERENCE-MAPPING-ITEM(`torch.tril_indices`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.tril_indices.md) |
| REFERENCE-MAPPING-ITEM(`torch.triu`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.triu.md) |
| REFERENCE-MAPPING-ITEM(`torch.triu_indices`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.triu_indices.md) |
| REFERENCE-MAPPING-ITEM(`torch.true_divide`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.true_divide.md) |
| REFERENCE-MAPPING-ITEM(`torch.trunc`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.trunc.md) |
| REFERENCE-MAPPING-ITEM(`torch.unbind`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.unbind.md) |
| REFERENCE-MAPPING-ITEM(`torch.unflatten`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.unflatten.md) |
| REFERENCE-MAPPING-ITEM(`torch.unique`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.unique.md) |
| REFERENCE-MAPPING-ITEM(`torch.unique_consecutive`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.unique_consecutive.md) |
| REFERENCE-MAPPING-ITEM(`torch.unsqueeze`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.unsqueeze.md) |
| REFERENCE-MAPPING-ITEM(`torch.vander`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.vander.md) |
| REFERENCE-MAPPING-ITEM(`torch.var`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.var.md) |
| REFERENCE-MAPPING-ITEM(`torch.var_mean`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.var_mean.md) |
| REFERENCE-MAPPING-ITEM(`torch.vdot`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.vdot.md) |
| REFERENCE-MAPPING-ITEM(`torch.view_as_complex`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.view_as_complex.md) |
| REFERENCE-MAPPING-ITEM(`torch.view_as_real`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.view_as_real.md) |
| REFERENCE-MAPPING-ITEM(`torch.vsplit`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.vsplit.md) |
| REFERENCE-MAPPING-ITEM(`torch.vstack`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.vstack.md) |
| REFERENCE-MAPPING-ITEM(`torch.where`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.where.md) |
| REFERENCE-MAPPING-ITEM(`torch.xlogy`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.xlogy.md) |
| REFERENCE-MAPPING-ITEM(`torch.zeros`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.zeros.md) |
| REFERENCE-MAPPING-ITEM(`torch.zeros_like`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.zeros_like.md) |
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

***持续更新...***

## torch.nn.XX API 映射列表

梳理了`torch.nn.XX`类 API 的 PyTorch-PaddlePaddle API 映射列表。

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ----- | ----------- | ----------------- | ----------- | ------- |
| REFERENCE-MAPPING-ITEM(`torch.nn.AdaptiveAvgPool1d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.AdaptiveAvgPool1d.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.AdaptiveAvgPool2d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.AdaptiveAvgPool2d.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.AdaptiveAvgPool3d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.AdaptiveAvgPool3d.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.AdaptiveMaxPool1d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.AdaptiveMaxPool1d.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.AdaptiveMaxPool2d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.AdaptiveMaxPool2d.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.AdaptiveMaxPool3d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.AdaptiveMaxPool3d.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.AlphaDropout`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.AlphaDropout.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.AvgPool1d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.AvgPool1d.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.AvgPool2d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.AvgPool2d.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.AvgPool3d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.AvgPool3d.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.BCELoss`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.BCELoss.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.BCEWithLogitsLoss`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.BCEWithLogitsLoss.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.BatchNorm1d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.BatchNorm1d.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.BatchNorm2d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.BatchNorm2d.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.BatchNorm3d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.BatchNorm3d.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.Bilinear`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.Bilinear.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.CELU`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.CELU.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.CTCLoss`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.CTCLoss.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.ChannelShuffle`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.ChannelShuffle.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.ConstantPad1d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.ConstantPad1d.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.ConstantPad2d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.ConstantPad2d.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.ConstantPad3d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.ConstantPad3d.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.Conv1d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.Conv1d.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.Conv2d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.Conv2d.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.Conv3d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.Conv3d.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.ConvTranspose1d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.ConvTranspose1d.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.ConvTranspose2d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.ConvTranspose2d.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.ConvTranspose3d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.ConvTranspose3d.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.CosineEmbeddingLoss`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.CosineEmbeddingLoss.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.CosineSimilarity`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.CosineSimilarity.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.CrossEntropyLoss`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.CrossEntropyLoss.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.DataParallel`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.DataParallel.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.Dropout`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.Dropout.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.Dropout1d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.Dropout1d.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.Dropout2d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.Dropout2d.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.Dropout3d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.Dropout3d.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.ELU`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.ELU.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.Embedding`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.Embedding.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.Flatten`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.Flatten.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.Fold`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.Fold.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.FractionalMaxPool2d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.FractionalMaxPool2d.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.FractionalMaxPool3d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.FractionalMaxPool3d.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.GELU`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.GELU.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.GRU`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.GRU.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.GRUCell`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.GRUCell.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.GaussianNLLLoss`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.GaussianNLLLoss.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.GroupNorm`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.GroupNorm.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.Hardshrink`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.Hardshrink.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.Hardsigmoid`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.Hardsigmoid.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.Hardswish`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.Hardswish.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.Hardtanh`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.Hardtanh.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.HingeEmbeddingLoss`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.HingeEmbeddingLoss.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.HuberLoss`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.HuberLoss.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.Identity`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.Identity.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.InstanceNorm1d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.InstanceNorm1d.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.InstanceNorm2d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.InstanceNorm2d.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.InstanceNorm3d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.InstanceNorm3d.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.KLDivLoss`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.KLDivLoss.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.L1Loss`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.L1Loss.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.LSTM`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.LSTM.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.LSTMCell`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.LSTMCell.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.LayerNorm`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.LayerNorm.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.LeakyReLU`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.LeakyReLU.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.Linear`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.Linear.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.LocalResponseNorm`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.LocalResponseNorm.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.LogSigmoid`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.LogSigmoid.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.LogSoftmax`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.LogSoftmax.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.MSELoss`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.MSELoss.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.MarginRankingLoss`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.MarginRankingLoss.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.MaxPool1d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.MaxPool1d.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.MaxPool2d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.MaxPool2d.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.MaxPool3d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.MaxPool3d.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.MaxUnpool1d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.MaxUnpool1d.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.MaxUnpool2d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.MaxUnpool2d.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.MaxUnpool3d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.MaxUnpool3d.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.Mish`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.Mish.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.Module`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.Module.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.ModuleDict`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.ModuleDict.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.ModuleList`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.ModuleList.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.MultiLabelMarginLoss`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.MultiLabelMarginLoss.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.MultiLabelSoftMarginLoss`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.MultiLabelSoftMarginLoss.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.MultiMarginLoss`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.MultiMarginLoss.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.MultiheadAttention`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.MultiheadAttention.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.NLLLoss`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.NLLLoss.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.PReLU`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.PReLU.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.PairwiseDistance`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.PairwiseDistance.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.Parameter`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.Parameter.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.ParameterList`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.ParameterList.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.PixelShuffle`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.PixelShuffle.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.PixelUnshuffle`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.PixelUnshuffle.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.PoissonNLLLoss`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.PoissonNLLLoss.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.RNN`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.RNN.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.RNNBase`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.RNNBase.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.RNNCell`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.RNNCell.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.RReLU`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.RReLU.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.ReLU`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.ReLU.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.ReLU6`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.ReLU6.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.ReflectionPad1d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.ReflectionPad1d.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.ReflectionPad2d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.ReflectionPad2d.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.ReflectionPad3d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.ReflectionPad3d.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.ReplicationPad1d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.ReplicationPad1d.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.ReplicationPad2d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.ReplicationPad2d.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.ReplicationPad3d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.ReplicationPad3d.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.SELU`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.SELU.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.Sequential`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.Sequential.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.SiLU`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.SiLU.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.Sigmoid`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.Sigmoid.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.SmoothL1Loss`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.SmoothL1Loss.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.SoftMarginLoss`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.SoftMarginLoss.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.Softmax`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.Softmax.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.Softmax2d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.Softmax2d.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.Softplus`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.Softplus.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.Softshrink`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.Softshrink.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.Softsign`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.Softsign.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.SyncBatchNorm`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.SyncBatchNorm.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.SyncBatchNorm.convert_sync_batchnorm`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.SyncBatchNorm.convert_sync_batchnorm.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.Tanh`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.Tanh.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.Tanhshrink`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.Tanhshrink.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.Threshold`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.Threshold.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.Transformer`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.Transformer.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.TransformerDecoder`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.TransformerDecoder.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.TransformerDecoderLayer`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.TransformerDecoderLayer.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.TransformerEncoder`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.TransformerEncoder.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.TransformerEncoderLayer`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.TransformerEncoderLayer.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.TripletMarginLoss`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.TripletMarginLoss.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.TripletMarginWithDistanceLoss`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.TripletMarginWithDistanceLoss.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.Unflatten`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.Unflatten.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.Unfold`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.Unfold.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.Upsample`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.Upsample.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.UpsamplingBilinear2d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.UpsamplingBilinear2d.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.UpsamplingNearest2d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.UpsamplingNearest2d.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.ZeroPad2d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.ZeroPad2d.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.modules.batchnorm._BatchNorm`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.nn.modules.batchnorm._BatchNorm.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.modules.utils._ntuple`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.modules.utils._ntuple.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.modules.utils._pair`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.modules.utils._pair.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.parallel.DistributedDataParallel`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.parallel.DistributedDataParallel.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.parameter.Parameter`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/utils/torch.nn.parameter.Parameter.md) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.GLU`, https://pytorch.org/docs/stable/generated/torch.nn.GLU.html#torch.nn.GLU) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.LazyBatchNorm1d`, https://pytorch.org/docs/stable/generated/torch.nn.LazyBatchNorm1d.html#torch.nn.LazyBatchNorm1d) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.LazyBatchNorm2d`, https://pytorch.org/docs/stable/generated/torch.nn.LazyBatchNorm2d.html#torch.nn.LazyBatchNorm2d) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.LazyBatchNorm3d`, https://pytorch.org/docs/stable/generated/torch.nn.LazyBatchNorm3d.html#torch.nn.LazyBatchNorm3d) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.LazyInstanceNorm1d`, https://pytorch.org/docs/stable/generated/torch.nn.LazyInstanceNorm1d.html#torch.nn.LazyInstanceNorm1d) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.LazyInstanceNorm2d`, https://pytorch.org/docs/stable/generated/torch.nn.LazyInstanceNorm2d.html#torch.nn.LazyInstanceNorm2d) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.LazyInstanceNorm3d`, https://pytorch.org/docs/stable/generated/torch.nn.LazyInstanceNorm3d.html#torch.nn.LazyInstanceNorm3d) |

***持续更新...***

## torch.nn.functional.XX API 映射列表
梳理了`torch.nn.functional.XX`类 API 的 PyTorch-PaddlePaddle API 映射列表。

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ----- | ----------- | ----------------- | ----------- | ------- |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional._Reduction.get_enum`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional._Reduction.get_enum.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.adaptive_avg_pool1d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.adaptive_avg_pool1d.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.adaptive_avg_pool2d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.adaptive_avg_pool2d.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.adaptive_avg_pool3d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.adaptive_avg_pool3d.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.adaptive_max_pool1d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.adaptive_max_pool1d.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.adaptive_max_pool2d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.adaptive_max_pool2d.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.adaptive_max_pool3d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.adaptive_max_pool3d.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.affine_grid`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.affine_grid.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.alpha_dropout`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.alpha_dropout.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.avg_pool1d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.avg_pool1d.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.avg_pool2d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.avg_pool2d.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.avg_pool3d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.avg_pool3d.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.batch_norm`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.batch_norm.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.bilinear`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.bilinear.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.binary_cross_entropy`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.binary_cross_entropy.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.binary_cross_entropy_with_logits`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.binary_cross_entropy_with_logits.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.celu`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.celu.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.conv1d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.conv1d.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.conv2d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.conv2d.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.conv3d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.conv3d.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.conv_transpose1d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.conv_transpose1d.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.conv_transpose2d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.conv_transpose2d.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.conv_transpose3d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.conv_transpose3d.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.cosine_embedding_loss`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.cosine_embedding_loss.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.cosine_similarity`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.cosine_similarity.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.cross_entropy`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.cross_entropy.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.ctc_loss`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.ctc_loss.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.dropout`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.dropout.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.dropout1d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.dropout1d.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.dropout2d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.dropout2d.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.dropout3d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.dropout3d.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.elu`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.elu.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.elu_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.elu_.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.embedding`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.embedding.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.fold`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.fold.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.fractional_max_pool2d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.fractional_max_pool2d.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.fractional_max_pool3d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.fractional_max_pool3d.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.gaussian_nll_loss`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.gaussian_nll_loss.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.gelu`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.gelu.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.glu`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.glu.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.grid_sample`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.grid_sample.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.group_norm`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.group_norm.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.gumbel_softmax`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.gumbel_softmax.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.hardshrink`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.hardshrink.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.hardsigmoid`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.hardsigmoid.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.hardswish`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.hardswish.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.hardtanh`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.hardtanh.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.hardtanh_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.hardtanh_.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.hinge_embedding_loss`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.hinge_embedding_loss.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.huber_loss`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.huber_loss.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.instance_norm`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.instance_norm.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.interpolate`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.interpolate.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.kl_div`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.kl_div.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.l1_loss`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.l1_loss.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.layer_norm`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.layer_norm.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.leaky_relu`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.leaky_relu.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.leaky_relu_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.leaky_relu_.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.linear`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.linear.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.local_response_norm`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.local_response_norm.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.log_softmax`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.log_softmax.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.logsigmoid`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.log_sigmoid.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.margin_ranking_loss`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.margin_ranking_loss.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.max_pool1d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.max_pool1d.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.max_pool2d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.max_pool2d.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.max_pool3d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.max_pool3d.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.max_unpool1d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.max_unpool1d.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.max_unpool2d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.max_unpool2d.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.max_unpool3d`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.max_unpool3d.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.mish`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.mish.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.mse_loss`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.mse_loss.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.multi_margin_loss`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.multi_margin_loss.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.multilabel_soft_margin_loss`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.multilabel_soft_margin_loss.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.nll_loss`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.nll_loss.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.normalize`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.normalize.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.one_hot`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.one_hot.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.pad`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.pad.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.pairwise_distance`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.pairwise_distance.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.pdist`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.pdist.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.pixel_shuffle`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.pixel_shuffle.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.pixel_unshuffle`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.pixel_unshuffle.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.poisson_nll_loss`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.poisson_nll_loss.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.prelu`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.prelu.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.relu`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.relu.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.relu6`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.relu6.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.relu_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.relu_.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.rrelu`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.rrelu.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.rrelu_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.rrelu_.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.selu`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.selu.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.sigmoid`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.sigmoid.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.silu`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.silu.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.smooth_l1_loss`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.smooth_l1_loss.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.soft_margin_loss`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.soft_margin_loss.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.softmax`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.softmax.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.softmin`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.softmin.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.softplus`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.softplus.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.softshrink`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.softshrink.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.softsign`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.softsign.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.tanh`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.tanh.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.tanhshrink`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.tanhshrink.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.threshold`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.threshold.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.triplet_margin_loss`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.triplet_margin_loss.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.triplet_margin_with_distance_loss`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.triplet_margin_with_distance_loss.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.unfold`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.unfold.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.upsample`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.upsample.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.upsample_bilinear`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.upsample_bilinear.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.functional.upsample_nearest`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.upsample_nearest.md) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.functional.multilabel_margin_loss`, https://pytorch.org/docs/stable/generated/torch.nn.functional.multilabel_margin_loss.html#torch.nn.functional.multilabel_margin_loss) |


***持续更新...***

## torch.Tensor.XX API 映射列表

梳理了`torch.Tensor.XX`类 API 的 PyTorch-PaddlePaddle API 映射列表。

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ----- | ----------- | ----------------- | ----------- | ------- |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.H`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.H.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.T`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.T__upper.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.abs`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.abs.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.abs_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.abs_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.absolute`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.absolute.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.acos_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.acos_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.acosh`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.acosh.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.acosh_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.acosh_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.add`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.add.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.add_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.add_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.addbmm`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.addbmm.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.addcdiv`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.addcdiv.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.addcmul`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.addcmul.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.addmm`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.addmm.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.addmm_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.addmm_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.addmv`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.addmv.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.addr`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.addr.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.adjoint`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.adjoint.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.all`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.all.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.allclose`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.allclose.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.amax`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.amax.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.amin`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.amin.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.aminmax`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.aminmax.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.angle`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.angle.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.any`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.any.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.apply_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.apply_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.arccos`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.arccos.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.arccos_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.arccos_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.arccosh`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.arccosh.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.arccosh_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.arccosh_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.arcsin`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.arcsin.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.arcsin_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.arcsin_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.arcsinh`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.arcsinh.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.arcsinh_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.arcsinh_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.arctan`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.arctan.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.arctan2`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.arctan2.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.arctan_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.arctan_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.arctanh`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.arctanh.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.arctanh_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.arctanh_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.argmax`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.argmax.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.argmin`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.argmin.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.argsort`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.argsort.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.argwhere`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.argwhere.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.as_strided`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.as_strided.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.asin_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.asin_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.asinh`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.asinh.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.asinh_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.asinh_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.atan_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.atan_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.atanh`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.atanh.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.atanh_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.atanh_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.backward`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.backward.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.baddbmm`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.baddbmm.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.bernoulli`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.bernoulli.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.bernoulli_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.bernoulli_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.bfloat16`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.bfloat16.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.bincount`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.bincount.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.bitwise_and`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.bitwise_and.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.bitwise_and_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.bitwise_and_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.bitwise_not`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.bitwise_not.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.bitwise_not_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.bitwise_not_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.bitwise_or`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.bitwise_or.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.bitwise_or_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.bitwise_or_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.bitwise_xor`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.bitwise_xor.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.bitwise_xor_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.bitwise_xor_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.bmm`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.bmm.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.bool`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.bool.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.broadcast_to`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.broadcast_to.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.byte`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.byte.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.cdouble`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.cdouble.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.ceil`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.ceil.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.ceil_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.ceil_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.cfloat`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.cfloat.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.char`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.char.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.cholesky`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.cholesky.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.cholesky_inverse`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.cholesky_inverse.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.cholesky_solve`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.cholesky_solve.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.chunk`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.chunk.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.clamp`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.clamp.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.clamp_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.clamp_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.clip`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.clip.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.clip_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.clip_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.clone`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.clone.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.coalesce`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.coalesce.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.conj`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.conj.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.conj_physical`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.conj_physical.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.contiguous`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.contiguous.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.copy_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.copy_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.corrcoef`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.corrcoef.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.cos`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.cos.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.cos_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.cos_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.cosh`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.cosh.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.cosh_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.cosh_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.count_nonzero`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.count_nonzero.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.cov`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.cov.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.cpu`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.cpu.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.cross`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.cross.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.cuda`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.cuda.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.cummax`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.cummax.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.cummin`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.cummin.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.cumprod`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.cumprod.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.cumprod_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.cumprod_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.cumsum`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.cumsum.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.cumsum_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.cumsum_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.deg2rad`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.deg2rad.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.det`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.det.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.detach`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.detach.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.detach_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.detach_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.device`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.device.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.diag`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.diag.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.diag_embed`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.diag_embed.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.diagflat`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.diagflat.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.diagonal`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.diagonal.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.diagonal_scatter`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.diagonal_scatter.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.diff`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.diff.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.digamma`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.digamma.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.digamma_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.digamma_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.dim`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.dim.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.dist`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.dist.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.div`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.div.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.div_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.div_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.divide`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.divide.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.divide_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.divide_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.dot`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.dot.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.double`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.double.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.dsplit`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.dsplit.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.dtype`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tenosr.dtype.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.element_size`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.element_size.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.eq`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.eq.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.eq_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.eq_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.equal`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.equal.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.erf`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.erf.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.erf_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.erf_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.erfc`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.erfc.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.erfinv`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.erfinv.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.erfinv\_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.erfinv_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.erfinv_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.erfinv_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.exp`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.exp.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.exp\_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.exp_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.exp_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.exp_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.expand`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.expand.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.expand_as`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.expand_as.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.expm1`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.expm1.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.expm1_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.expm1_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.exponential_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.exponential_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.fill\_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.fill_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.fill_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.fill_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.fill_diagonal_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.fill_diagonal_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.fix`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.fix.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.flatten`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.flatten.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.flip`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.flip.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.fliplr`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.fliplr.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.flipud`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.flipud.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.float`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.float.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.float_power`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.float_power.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.float_power_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.float_power_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.floor`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.floor.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.floor_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.floor_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.floor_divide`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.floor_divide.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.floor_divide_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.floor_divide_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.fmax`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.fmax.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.fmin`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.fmin.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.fmod`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.fmod.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.frac`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.frac.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.frac_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.frac_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.frexp`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.frexp.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.gather`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.gather.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.gcd`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.gcd.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.gcd_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.gcd_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.ge`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.ge.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.ge_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.ge_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.ger`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.ger.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.get_device`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.get_device.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.grad`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.grad.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.greater`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.greater.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.greater_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.greater_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.greater_equal`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.greater_equal.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.greater_equal_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.greater_equal_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.gt`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.gt.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.gt_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.gt_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.half`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.half.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.hardshrink`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.hardshrink.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.heaviside`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.heaviside.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.histc`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.histc.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.histogram`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.histogram.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.hsplit`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.hsplit.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.hypot`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.hypot.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.hypot_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.hypot_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.i0`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.i0.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.i0_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.i0_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.imag`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.imag.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.index_add`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.index_add.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.index_add_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.index_add_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.index_copy_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.index_copy_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.index_fill`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.index_fill.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.index_fill_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.index_fill_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.index_put`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.index_put.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.index_put_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.index_put_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.index_select`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.index_select.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.indices`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.indices.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.inner`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.inner.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.int`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.int.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.inverse`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.inverse.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.is_complex`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.is_complex.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.is_contiguous`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.is_contiguous.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.is_cuda`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.is_cuda.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.is_floating_point`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.is_floating_point.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.is_leaf`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.is_leaf.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.is_pinned`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.is_pinned.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.is_signed`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.is_signed.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.is_sparse`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.is_sparse.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.isclose`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.isclose.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.isfinite`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.isfinite.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.isinf`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.isinf.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.isnan`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.isnan.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.istft`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.istft.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.item`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.item.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.kthvalue`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.kthvalue.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.lcm`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.lcm.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.lcm_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.lcm_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.ldexp`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.ldexp.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.ldexp_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.ldexp_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.le`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.le.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.le_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.le_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.lerp`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.lerp.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.lerp_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.lerp_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.less`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.less.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.less_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.less_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.less_equal`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.less_equal.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.less_equal_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.less_equal_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.lgamma`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.lgamma.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.lgamma_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.lgamma_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.log`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.log.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.log10`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.log10.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.log10_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.log10_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.log1p`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.log1p.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.log1p_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.log1p_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.log2`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.log2.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.log2_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.log2_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.log_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.log_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.logaddexp`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.logaddexp.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.logaddexp2`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.logaddexp2.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.logcumsumexp`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.logcumsumexp.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.logdet`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.logdet.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.logical_and`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.logical_and.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.logical_and_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.logical_and_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.logical_not`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.logical_not.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.logical_not_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.logical_not_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.logical_or`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.logical_or.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.logical_or_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.logical_or_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.logical_xor`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.logical_xor.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.logical_xor_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.logical_xor_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.logit`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.logit.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.logit_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.logit_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.logsumexp`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.logsumexp.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.long`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.long.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.lstsq`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.lstsq.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.lt`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.lt.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.lt_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.lt_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.lu`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.lu.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.mH`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.mH.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.mT`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.mT.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.masked_fill`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.masked_fill.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.masked_fill_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.masked_fill_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.masked_scatter`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.masked_scatter.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.masked_scatter_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.masked_scatter_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.masked_select`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.masked_select.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.matmul`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.matmul.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.matrix_power`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.matrix_power.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.maximum`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.maximum.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.mean`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.mean.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.median`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.median.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.minimum`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.minimum.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.mm`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.mm.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.mode`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.mode.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.moveaxis`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.moveaxis.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.movedim`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.movedim.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.msort`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.msort.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.mul`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.mul.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.multinomial`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.multinomial.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.multiply`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.multiply.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.multiply_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.multiply_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.mv`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.mv.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.nan_to_num`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.nan_to_num.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.nan_to_num_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.nan_to_num_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.nanmean`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.nanmean.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.nanmedian`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.nanmedian.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.nanquantile`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.nanquantile.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.nansum`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.nansum.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.narrow`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.narrow.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.narrow_copy`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.narrow_copy.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.ndim`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.ndim.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.ndimension`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.ndimension.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.ne`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.ne.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.ne_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.ne_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.neg`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.neg.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.neg_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.neg_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.negative`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.negative.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.negative_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.negative_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.nelement`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.nelement.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.new_empty`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.new_empty.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.new_full`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.new_full.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.new_ones`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.new_ones.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.new_tensor`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.new_tensor.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.new_zeros`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.new_zeros.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.nextafter`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.nextafter.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.nonzero`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.nonzero.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.norm`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.norm.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.normal_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.normal_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.not_equal`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.not_equal.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.not_equal_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.not_equal_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.numel`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.numel.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.numpy`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.numpy.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.outer`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.outer.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.permute`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.permute.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.pin_memory`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.pin_memory.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.pinverse`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.pinverse.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.polygamma`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.polygamma.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.polygamma_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.polygamma_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.pow`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.pow.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.pow_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.pow_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.prod`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.prod.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.qr`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.qr.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.quantile`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.quantile.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.rad2deg`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.rad2deg.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.ravel`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.ravel.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.real`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.real.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.reciprocal`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.reciprocal.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.reciprocal_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.reciprocal_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.register_hook`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.register_hook.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.remainder`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.remainder.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.remainder_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.remainder_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.renorm`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.renorm.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.renorm_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.renorm_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.repeat`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.repeat.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.repeat_interleave`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.repeat_interleave.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.requires_grad_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.requires_grad_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.reshape`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.reshape.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.reshape_as`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.reshape_as.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.resize_as_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.resize_as_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.retain_grad`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.retain_grad.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.roll`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.roll.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.rot90`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.rot90.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.round`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.round.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.round_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.round_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.rsqrt`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.rsqrt.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.rsqrt_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.rsqrt_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.scatter`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.scatter.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.scatter_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.scatter_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.scatter_add`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.scatter_add.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.scatter_add_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.scatter_add_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.select`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.select.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.sgn`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.sgn.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.shape`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.shape.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.short`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.short.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.sigmoid`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.sigmoid.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.sigmoid_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.sigmoid_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.sign`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.sign.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.signbit`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.signbit.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.sin`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.sin.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.sin_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.sin_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.sinc`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.sinc.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.sinh`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.sinh.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.sinh_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.sinh_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.size`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.size.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.slice_scatter`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.slice_scatter.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.slogdet`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.slogdet.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.softmax`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.softmax.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.sort`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.sort.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.split`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.split.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.sparse_mask`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.sparse_mask.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.sqrt`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.sqrt.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.sqrt_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.sqrt_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.square`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.square.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.square_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.square_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.squeeze`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.squeeze.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.squeeze_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.squeeze_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.std`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.std.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.stft`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.stft.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.sub`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.sub.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.sub_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.sub_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.subtract`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.subtract.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.subtract_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.subtract_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.sum`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.sum.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.svd`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.svd.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.swapaxes`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.swapaxes.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.swapdims`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.swapdims.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.symeig`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.symeig.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.t`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.t.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.take`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.take.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.take_along_dim`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.take_along_dim.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.tan`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.tan.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.tan_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.tan_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.tanh`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.tanh.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.tanh_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.tanh_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.tensor_split`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.tensor_split.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.tile`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.tile.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.to`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.to.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.to_dense`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.to_dense.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.to_sparse`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.to_sparse.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.tolist`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.tolist.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.topk`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.topk.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.trace`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.trace.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.transpose`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.transpose.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.triangular_solve`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.triangular_solve.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.tril`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.tril.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.tril_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.tril_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.triu`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.triu.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.triu_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.triu_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.true_divide`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.true_divide.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.true_divide_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.true_divide_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.trunc`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.trunc.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.trunc_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.trunc_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.type`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.type.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.type_as`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.type_as.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.unbind`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.unbind.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.unflatten`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.unflatten.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.unfold`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.unfold.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.uniform_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.uniform_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.unique`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.unique.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.unique_consecutive`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.unique_consecutive.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.unsqueeze`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.unsqueeze.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.unsqueeze_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.unsqueeze_.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.values`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.values.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.var`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.var.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.vdot`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.vdot.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.view`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.view.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.view_as`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.view_as.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.vsplit`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.vsplit.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.where`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.where.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.xlogy`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.xlogy.md) |
| REFERENCE-MAPPING-ITEM(`torch.Tensor.zero_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.zero_.md) |
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

***持续更新...***

## torch.nn.init.XX API 映射列表
梳理了`torch.nn.init.XX`类 API 的 PyTorch-PaddlePaddle API 映射列表。

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ----- | ----------- | ----------------- | ----------- | ------- |
| REFERENCE-MAPPING-ITEM(`torch.nn.init.calculate_gain`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/init/torch.nn.init.calculate_gain.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.init.constant_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/init/torch.nn.init.constant_.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.init.dirac_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/init/torch.nn.init.dirac_.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.init.eye_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/init/torch.nn.init.eye_.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.init.kaiming_normal_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/init/torch.nn.init.kaiming_normal_.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.init.kaiming_uniform_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/init/torch.nn.init.kaiming_uniform_.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.init.normal_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/init/torch.nn.init.normal_.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.init.ones_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/init/torch.nn.init.ones_.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.init.orthogonal_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/init/torch.nn.init.orthogonal_.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.init.trunc_normal_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/init/torch.nn.init.trunc_normal_.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.init.uniform_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/init/torch.nn.init.uniform_.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.init.xavier_normal_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/init/torch.nn.init.xavier_normal_.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.init.xavier_uniform_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/init/torch.nn.init.xavier_uniform_.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.init.zeros_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/init/torch.nn.init.zeros_.md) |

***持续更新...***

## torch.nn.utils.XX API 映射列表
梳理了`torch.nn.utils.XX`类 API 的 PyTorch-PaddlePaddle API 映射列表。

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ----- | ----------- | ----------------- | ----------- | ------- |
| REFERENCE-MAPPING-ITEM(`torch.nn.utils.clip_grad_norm_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/utils/torch.nn.utils.clip_grad_norm_.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.utils.clip_grad_value_`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/utils/torch.nn.utils.clip_grad_value_.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.utils.parameters_to_vector`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.utils.parameters_to_vector.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.utils.parametrizations.spectral_norm`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/utils/torch.nn.utils.parametrizations.spectral_norm.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.utils.remove_weight_norm`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/utils/torch.nn.utils.remove_weight_norm.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.utils.spectral_norm`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/utils/torch.nn.utils.spectral_norm.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.utils.vector_to_parameters`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.utils.vector_to_parameters.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.utils.weight_norm`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/utils/torch.nn.utils.weight_norm.md) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.parametrize.is_parametrized`, https://pytorch.org/docs/stable/generated/torch.nn.utils.parametrize.is_parametrized.html#torch.nn.utils.parametrize.is_parametrized) |


***持续更新...***

## torch.nn.Module.XX API 映射列表
梳理了`torch.nn.Module.XX`类 API 的 PyTorch-PaddlePaddle API 映射列表。

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ----- | ----------- | ----------------- | ----------- | ------- |
| REFERENCE-MAPPING-ITEM(`torch.nn.Module.add_module`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.Module.add_module.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.Module.apply`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.Module.apply.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.Module.bfloat16`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.Module.bfloat16.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.Module.buffers`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.Module.buffers.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.Module.children`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.Module.children.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.Module.cpu`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.Module.cpu.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.Module.cuda`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.Module.cuda.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.Module.double`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.Module.double.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.Module.eval`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.Module.eval.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.Module.float`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.Module.float.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.Module.get_buffer`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.Module.get_buffer.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.Module.get_parameter`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.Module.get_parameter.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.Module.get_submodule`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.Module.get_submodule.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.Module.half`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.Module.half.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.Module.load_state_dict`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.Module.load_state_dict.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.Module.modules`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.nn.Module.modules.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.Module.named_buffers`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.Module.named_buffers.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.Module.named_children`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.Module.named_children.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.Module.named_modules`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.Module.named_modules.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.Module.named_parameters`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.Module.named_parameters.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.Module.parameters`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.Module.parameters.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.Module.register_buffer`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.Module.register_buffer.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.Module.register_forward_hook`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.Module.register_forward_hook.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.Module.register_forward_pre_hook`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.Module.register_forward_pre_hook.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.Module.register_module`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.Module.register_module.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.Module.register_parameter`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.Module.register_parameter.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.Module.state_dict`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.Module.state_dict.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.Module.to`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.Module.to.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.Module.train`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.Module.train.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.Module.type`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.Module.type.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.Module.xpu`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.Module.xpu.md) |
| REFERENCE-MAPPING-ITEM(`torch.nn.Module.zero_grad`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.Module.zero_grad.md) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.Module.register_full_backward_hook`, https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_full_backward_hook) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.Module.register_full_backward_pre_hook`, https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_full_backward_pre_hook) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.Module.requires_grad_`, https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.requires_grad_) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.Module.to_empty`, https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.to_empty) |


***持续更新...***

## torch.autograd.XX API 映射列表
梳理了`torch.autograd.XX`类 API 的 PyTorch-PaddlePaddle API 映射列表。

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ----- | ----------- | ----------------- | ----------- | ------- |
| REFERENCE-MAPPING-ITEM(`torch.autograd.Function`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.autograd.Function.md) |
| REFERENCE-MAPPING-ITEM(`torch.autograd.Function.backward`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.autograd.Function.backward.md) |
| REFERENCE-MAPPING-ITEM(`torch.autograd.Function.forward`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.autograd.Function.forward.md) |
| REFERENCE-MAPPING-ITEM(`torch.autograd.backward`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.autograd.backward.md) |
| REFERENCE-MAPPING-ITEM(`torch.autograd.enable_grad`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.autograd.enable_grad.md) |
| REFERENCE-MAPPING-ITEM(`torch.autograd.function.FunctionCtx`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.autograd.function.FunctionCtx.md) |
| REFERENCE-MAPPING-ITEM(`torch.autograd.function.FunctionCtx.mark_non_differentiable`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.autograd.function.FunctionCtx.mark_non_differentiable.md) |
| REFERENCE-MAPPING-ITEM(`torch.autograd.function.FunctionCtx.save_for_backward`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.autograd.function.FunctionCtx.save_for_backward.md) |
| REFERENCE-MAPPING-ITEM(`torch.autograd.function.FunctionCtx.set_materialize_grads`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.autograd.function.FunctionCtx.set_materialize_grads.md) |
| REFERENCE-MAPPING-ITEM(`torch.autograd.functional.hessian`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.autograd.functional.hessian.md) |
| REFERENCE-MAPPING-ITEM(`torch.autograd.functional.jacobian`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.autograd.functional.jacobian.md) |
| REFERENCE-MAPPING-ITEM(`torch.autograd.functional.jvp`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.autograd.functional.jvp.md) |
| REFERENCE-MAPPING-ITEM(`torch.autograd.functional.vjp`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.autograd.functional.vjp.md) |
| REFERENCE-MAPPING-ITEM(`torch.autograd.grad`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.autograd.grad.md) |
| REFERENCE-MAPPING-ITEM(`torch.autograd.graph.saved_tensors_hooks`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.autograd.graph.saved_tensors_hooks.md) |
| REFERENCE-MAPPING-ITEM(`torch.autograd.profiler.profile.export_chrome_trace`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.autograd.profiler.profile.export_chrome_trace.md) |
| REFERENCE-MAPPING-ITEM(`torch.autograd.set_grad_enabled`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.autograd.set_grad_enabled.md) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.function.FunctionCtx.mark_dirty`, https://pytorch.org/docs/stable/generated/torch.autograd.function.FunctionCtx.mark_dirty.html#torch.autograd.function.FunctionCtx.mark_dirty) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.profiler.profile.self_cpu_time_total`, https://pytorch.org/docs/stable/generated/torch.autograd.profiler.profile.self_cpu_time_total.html#torch.autograd.profiler.profile.self_cpu_time_total) |

***持续更新...***

## torch.cuda.XX API 映射列表
梳理了`torch.cuda.XX`类 API 的 PyTorch-PaddlePaddle API 映射列表。

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ----- | ----------- | ----------------- | ----------- | ------- |
| REFERENCE-MAPPING-ITEM(`torch.cuda.BoolTensor`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/cuda/torch.cuda.BoolTensor.md) |
| REFERENCE-MAPPING-ITEM(`torch.cuda.ByteTensor`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/cuda/torch.cuda.ByteTensor.md) |
| REFERENCE-MAPPING-ITEM(`torch.cuda.DoubleTensor`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/cuda/torch.cuda.DoubleTensor.md) |
| REFERENCE-MAPPING-ITEM(`torch.cuda.Event`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/cuda/torch.cuda.Event.md) |
| REFERENCE-MAPPING-ITEM(`torch.cuda.FloatTensor`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/cuda/torch.cuda.FloatTensor.md) |
| REFERENCE-MAPPING-ITEM(`torch.cuda.HalfTensor`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/cuda/torch.cuda.HalfTensor.md) |
| REFERENCE-MAPPING-ITEM(`torch.cuda.IntTensor`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/cuda/torch.cuda.IntTensor.md) |
| REFERENCE-MAPPING-ITEM(`torch.cuda.LongTensor`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/cuda/torch.cuda.LongTensor.md) |
| REFERENCE-MAPPING-ITEM(`torch.cuda.ShortTensor`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/cuda/torch.cuda.ShortTensor.md) |
| REFERENCE-MAPPING-ITEM(`torch.cuda.Stream`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/cuda/torch.cuda.Stream__upper.md) |
| REFERENCE-MAPPING-ITEM(`torch.cuda.amp.GradScaler`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/cuda/torch.cuda.amp.GradScaler.md) |
| REFERENCE-MAPPING-ITEM(`torch.cuda.amp.autocast`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/cuda/torch.cuda.amp.autocast.md) |
| REFERENCE-MAPPING-ITEM(`torch.cuda.comm.broadcast`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/cuda/torch.cuda.comm.broadcast.md) |
| REFERENCE-MAPPING-ITEM(`torch.cuda.current_device`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/cuda/torch.cuda.current_device.md) |
| REFERENCE-MAPPING-ITEM(`torch.cuda.current_stream`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/cuda/torch.cuda.current_stream.md) |
| REFERENCE-MAPPING-ITEM(`torch.cuda.device`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/cuda/torch.cuda.device.md) |
| REFERENCE-MAPPING-ITEM(`torch.cuda.device_count`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/cuda/torch.cuda.device_count.md) |
| REFERENCE-MAPPING-ITEM(`torch.cuda.empty_cache`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/cuda/torch.cuda.empty_cache.md) |
| REFERENCE-MAPPING-ITEM(`torch.cuda.get_device_capability`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/cuda/torch.cuda.get_device_capability.md) |
| REFERENCE-MAPPING-ITEM(`torch.cuda.get_device_name`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/cuda/torch.cuda.get_device_name.md) |
| REFERENCE-MAPPING-ITEM(`torch.cuda.get_device_properties`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/cuda/torch.cuda.get_device_properties.md) |
| REFERENCE-MAPPING-ITEM(`torch.cuda.get_rng_state_all`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/cuda/torch.cuda.get_rng_state_all.md) |
| REFERENCE-MAPPING-ITEM(`torch.cuda.initial_seed`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/cuda/torch.cuda.initial_seed.md) |
| REFERENCE-MAPPING-ITEM(`torch.cuda.is_available`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/cuda/torch.cuda.is_available.md) |
| REFERENCE-MAPPING-ITEM(`torch.cuda.manual_seed`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/cuda/torch.cuda.manual_seed.md) |
| REFERENCE-MAPPING-ITEM(`torch.cuda.manual_seed_all`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/cuda/torch.cuda.manual_seed_all.md) |
| REFERENCE-MAPPING-ITEM(`torch.cuda.max_memory_allocated`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/cuda/torch.cuda.max_memory_allocated.md) |
| REFERENCE-MAPPING-ITEM(`torch.cuda.max_memory_reserved`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/cuda/torch.cuda.max_memory_reserved.md) |
| REFERENCE-MAPPING-ITEM(`torch.cuda.memory_allocated`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/cuda/torch.cuda.memory_allocated.md) |
| REFERENCE-MAPPING-ITEM(`torch.cuda.memory_reserved`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/cuda/torch.cuda.memory_reserved.md) |
| REFERENCE-MAPPING-ITEM(`torch.cuda.nvtx.range_pop`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/cuda/torch.cuda.nvtx.range_pop.md) |
| REFERENCE-MAPPING-ITEM(`torch.cuda.nvtx.range_push`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/cuda/torch.cuda.nvtx.range_push.md) |
| REFERENCE-MAPPING-ITEM(`torch.cuda.set_device`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/cuda/torch.cuda.set_device.md) |
| REFERENCE-MAPPING-ITEM(`torch.cuda.set_rng_state_all`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/cuda/torch.cuda.set_rng_state_all.md) |
| REFERENCE-MAPPING-ITEM(`torch.cuda.set_stream`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/cuda/torch.cuda.set_stream.md) |
| REFERENCE-MAPPING-ITEM(`torch.cuda.stream`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/cuda/torch.cuda.stream.md) |
| REFERENCE-MAPPING-ITEM(`torch.cuda.synchronize`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/cuda/torch.cuda.synchronize.md) |
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

***持续更新...***

## torch.distributed.XX API 映射列表
梳理了`torch.distributed.XX`类 API 的 PyTorch-PaddlePaddle API 映射列表。

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ----- | ----------- | ----------------- | ----------- | ------- |
| REFERENCE-MAPPING-ITEM(`torch.distributed.ReduceOp`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/distributed/torch.distributed.ReduceOp.md) |
| REFERENCE-MAPPING-ITEM(`torch.distributed.ReduceOp.MAX`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.distributed.ReduceOp.MAX.md) |
| REFERENCE-MAPPING-ITEM(`torch.distributed.ReduceOp.MIN`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.distributed.ReduceOp.MIN.md) |
| REFERENCE-MAPPING-ITEM(`torch.distributed.ReduceOp.PRODUCT`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.distributed.ReduceOp.PRODUCT.md) |
| REFERENCE-MAPPING-ITEM(`torch.distributed.ReduceOp.SUM`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.distributed.ReduceOp.SUM.md) |
| REFERENCE-MAPPING-ITEM(`torch.distributed.all_gather`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/distributed/torch.distributed.all_gather.md) |
| REFERENCE-MAPPING-ITEM(`torch.distributed.all_gather_object`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/distributed/torch.distributed.all_gather_object.md) |
| REFERENCE-MAPPING-ITEM(`torch.distributed.all_reduce`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/distributed/torch.distributed.all_reduce.md) |
| REFERENCE-MAPPING-ITEM(`torch.distributed.all_to_all`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/distributed/torch.distributed.all_to_all.md) |
| REFERENCE-MAPPING-ITEM(`torch.distributed.barrier`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/distributed/torch.distributed.barrier.md) |
| REFERENCE-MAPPING-ITEM(`torch.distributed.broadcast`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/distributed/torch.distributed.broadcast.md) |
| REFERENCE-MAPPING-ITEM(`torch.distributed.broadcast_object_list`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/distributed/torch.distributed.broadcast_object_list.md) |
| REFERENCE-MAPPING-ITEM(`torch.distributed.gather`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/distributed/torch.distributed.gather.md) |
| REFERENCE-MAPPING-ITEM(`torch.distributed.get_backend`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/distributed/torch.distributed.get_backend.md) |
| REFERENCE-MAPPING-ITEM(`torch.distributed.get_rank`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/distributed/torch.distributed.get_rank.md) |
| REFERENCE-MAPPING-ITEM(`torch.distributed.get_world_size`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/distributed/torch.distributed.get_world_size.md) |
| REFERENCE-MAPPING-ITEM(`torch.distributed.init_process_group`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/distributed/torch.distributed.init_process_group.md) |
| REFERENCE-MAPPING-ITEM(`torch.distributed.irecv`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/distributed/torch.distributed.irecv.md) |
| REFERENCE-MAPPING-ITEM(`torch.distributed.isend`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/distributed/torch.distributed.isend.md) |
| REFERENCE-MAPPING-ITEM(`torch.distributed.new_group`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/distributed/torch.distributed.new_group.md) |
| REFERENCE-MAPPING-ITEM(`torch.distributed.recv`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/distributed/torch.distributed.recv.md) |
| REFERENCE-MAPPING-ITEM(`torch.distributed.reduce`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/distributed/torch.distributed.reduce.md) |
| REFERENCE-MAPPING-ITEM(`torch.distributed.reduce_scatter`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/distributed/torch.distributed.reduce_scatter.md) |
| REFERENCE-MAPPING-ITEM(`torch.distributed.rpc.get_worker_info`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/distributed/torch.distributed.rpc.get_worker_info.md) |
| REFERENCE-MAPPING-ITEM(`torch.distributed.rpc.init_rpc`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/distributed/torch.distributed.rpc.init_rpc.md) |
| REFERENCE-MAPPING-ITEM(`torch.distributed.rpc.rpc_async`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/distributed/torch.distributed.rpc.rpc_async.md) |
| REFERENCE-MAPPING-ITEM(`torch.distributed.rpc.rpc_sync`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/distributed/torch.distributed.rpc.rpc_sync.md) |
| REFERENCE-MAPPING-ITEM(`torch.distributed.rpc.shutdown`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/distributed/torch.distributed.rpc.shutdown.md) |
| REFERENCE-MAPPING-ITEM(`torch.distributed.scatter`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/distributed/torch.distributed.scatter.md) |
| REFERENCE-MAPPING-ITEM(`torch.distributed.scatter_object_list`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/distributed/torch.distributed.scatter_object_list.md) |
| REFERENCE-MAPPING-ITEM(`torch.distributed.send`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/distributed/torch.distributed.send.md) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.all_gather_multigpu`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_gather_multigpu) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.all_reduce_multigpu`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_reduce_multigpu) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.broadcast_multigpu`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.broadcast_multigpu) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.reduce_multigpu`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.reduce_multigpu) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.reduce_scatter_multigpu`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.reduce_scatter_multigpu) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.rpc.WorkerInfo`, https://pytorch.org/docs/stable/rpc.html#torch.distributed.rpc.WorkerInfo) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.rpc.functions.async_execution`, https://pytorch.org/docs/stable/rpc.html#torch.distributed.rpc.functions.async_execution) |


***持续更新...***

## torch.distributions.XX API 映射列表
梳理了`torch.distributions.XX`类 API 的 PyTorch-PaddlePaddle API 映射列表。

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ----- | ----------- | ----------------- | ----------- | ------- |
| REFERENCE-MAPPING-ITEM(`torch.distributions.Distribution.log_prob`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/distributions/torch.distributions.Distribution.log_prob.md) |
| REFERENCE-MAPPING-ITEM(`torch.distributions.Distribution.rsample`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/distributions/torch.distributions.Distribution.rsample.md) |
| REFERENCE-MAPPING-ITEM(`torch.distributions.Distribution.sample`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/distributions/torch.distributions.Distribution.sample.md) |
| REFERENCE-MAPPING-ITEM(`torch.distributions.bernoulli.Bernoulli`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/distributions/torch.distributions.bernoulli.Bernoulli.md) |
| REFERENCE-MAPPING-ITEM(`torch.distributions.beta.Beta`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/distributions/torch.distributions.beta.Beta.md) |
| REFERENCE-MAPPING-ITEM(`torch.distributions.categorical.Categorical`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/distributions/torch.distributions.categorical.Categorical.md) |
| REFERENCE-MAPPING-ITEM(`torch.distributions.cauchy.Cauchy`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/distributions/torch.distributions.cauchy.Cauchy.md) |
| REFERENCE-MAPPING-ITEM(`torch.distributions.dirichlet.Dirichlet`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/distributions/torch.distributions.dirichlet.Dirichlet.md) |
| REFERENCE-MAPPING-ITEM(`torch.distributions.distribution.Distribution`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/distributions/torch.distributions.distribution.Distribution.md) |
| REFERENCE-MAPPING-ITEM(`torch.distributions.exp_family.ExponentialFamily`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/distributions/torch.distributions.exp_family.ExponentialFamily.md) |
| REFERENCE-MAPPING-ITEM(`torch.distributions.geometric.Geometric`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/distributions/torch.distributions.geometric.Geometric.md) |
| REFERENCE-MAPPING-ITEM(`torch.distributions.gumbel.Gumbel`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/distributions/torch.distributions.gumbel.Gumbel.md) |
| REFERENCE-MAPPING-ITEM(`torch.distributions.independent.Independent`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/distributions/torch.distributions.independent.Independent.md) |
| REFERENCE-MAPPING-ITEM(`torch.distributions.kl.kl_divergence`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/distributions/torch.distributions.kl.kl_divergence.md) |
| REFERENCE-MAPPING-ITEM(`torch.distributions.kl.register_kl`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/distributions/torch.distributions.kl.register_kl.md) |
| REFERENCE-MAPPING-ITEM(`torch.distributions.laplace.Laplace`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/distributions/torch.distributions.laplace.Laplace.md) |
| REFERENCE-MAPPING-ITEM(`torch.distributions.log_normal.LogNormal`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/distributions/torch.distributions.log_normal.LogNormal.md) |
| REFERENCE-MAPPING-ITEM(`torch.distributions.multinomial.Multinomial`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/distributions/torch.distributions.multinomial.Multinomial.md) |
| REFERENCE-MAPPING-ITEM(`torch.distributions.normal.Normal`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/distributions/torch.distributions.normal.Normal.md) |
| REFERENCE-MAPPING-ITEM(`torch.distributions.transformed_distribution.TransformedDistribution`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/distributions/torch.distributions.transformed_distribution.TransformedDistribution.md) |
| REFERENCE-MAPPING-ITEM(`torch.distributions.transforms.AbsTransform`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/distributions/torch.distributions.transforms.AbsTransform.md) |
| REFERENCE-MAPPING-ITEM(`torch.distributions.transforms.AffineTransform`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/distributions/torch.distributions.transforms.AffineTransform.md) |
| REFERENCE-MAPPING-ITEM(`torch.distributions.transforms.ComposeTransform`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/distributions/torch.distributions.transforms.ComposeTransform.md) |
| REFERENCE-MAPPING-ITEM(`torch.distributions.transforms.ExpTransform`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/distributions/torch.distributions.transforms.ExpTransform.md) |
| REFERENCE-MAPPING-ITEM(`torch.distributions.transforms.IndependentTransform`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/distributions/torch.distributions.transforms.IndependentTransform.md) |
| REFERENCE-MAPPING-ITEM(`torch.distributions.transforms.PowerTransform`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/distributions/torch.distributions.transforms.PowerTransform.md) |
| REFERENCE-MAPPING-ITEM(`torch.distributions.transforms.ReshapeTransform`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/distributions/torch.distributions.transforms.ReshapeTransform.md) |
| REFERENCE-MAPPING-ITEM(`torch.distributions.transforms.SigmoidTransform`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/distributions/torch.distributions.transforms.SigmoidTransform.md) |
| REFERENCE-MAPPING-ITEM(`torch.distributions.transforms.SoftmaxTransform`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/distributions/torch.distributions.transforms.SoftmaxTransform.md) |
| REFERENCE-MAPPING-ITEM(`torch.distributions.transforms.StackTransform`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/distributions/torch.distributions.transforms.StackTransform.md) |
| REFERENCE-MAPPING-ITEM(`torch.distributions.transforms.StickBreakingTransform`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/distributions/torch.distributions.transforms.StickBreakingTransform.md) |
| REFERENCE-MAPPING-ITEM(`torch.distributions.transforms.TanhTransform`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/distributions/torch.distributions.transforms.TanhTransform.md) |
| REFERENCE-MAPPING-ITEM(`torch.distributions.transforms.Transform`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/distributions/torch.distributions.transforms.Transform.md) |
| REFERENCE-MAPPING-ITEM(`torch.distributions.uniform.Uniform`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/distributions/torch.distributions.uniform.Uniform.md) |
| NOT-IMPLEMENTED-ITEM(`torch.distributions.binomial.Binomial`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.binomial.Binomial) |
| NOT-IMPLEMENTED-ITEM(`torch.distributions.constraint_registry.ConstraintRegistry`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.constraint_registry.ConstraintRegistry) |
| NOT-IMPLEMENTED-ITEM(`torch.distributions.constraints.Constraint`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.constraints.Constraint) |
| NOT-IMPLEMENTED-ITEM(`torch.distributions.continuous_bernoulli.ContinuousBernoulli`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.continuous_bernoulli.ContinuousBernoulli) |
| NOT-IMPLEMENTED-ITEM(`torch.distributions.exponential.Exponential`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.exponential.Exponential) |
| NOT-IMPLEMENTED-ITEM(`torch.distributions.one_hot_categorical.OneHotCategorical`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.one_hot_categorical.OneHotCategorical) |
| NOT-IMPLEMENTED-ITEM(`torch.distributions.transforms.CatTransform`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.transforms.CatTransform) |
| NOT-IMPLEMENTED-ITEM(`torch.distributions.transforms.CumulativeDistributionTransform`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.transforms.CumulativeDistributionTransform) |
| NOT-IMPLEMENTED-ITEM(`torch.distributions.transforms.SoftplusTransform`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.transforms.SoftplusTransform) |


***持续更新...***

## torch.fft.XX API 映射列表
梳理了`torch.fft.XX`类 API 的 PyTorch-PaddlePaddle API 映射列表。

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ----- | ----------- | ----------------- | ----------- | ------- |
| REFERENCE-MAPPING-ITEM(`torch.fft.fft`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/fft/torch.fft.fft.md) |
| REFERENCE-MAPPING-ITEM(`torch.fft.fft2`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/fft/torch.fft.fft2.md) |
| REFERENCE-MAPPING-ITEM(`torch.fft.fftfreq`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/fft/torch.fft.fftfreq.md) |
| REFERENCE-MAPPING-ITEM(`torch.fft.fftn`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/fft/torch.fft.fftn.md) |
| REFERENCE-MAPPING-ITEM(`torch.fft.fftshift`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/fft/torch.fft.fftshift.md) |
| REFERENCE-MAPPING-ITEM(`torch.fft.hfft`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/fft/torch.fft.hfft.md) |
| REFERENCE-MAPPING-ITEM(`torch.fft.hfft2`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/fft/torch.fft.hfft2.md) |
| REFERENCE-MAPPING-ITEM(`torch.fft.hfftn`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/fft/torch.fft.hfftn.md) |
| REFERENCE-MAPPING-ITEM(`torch.fft.ifft`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/fft/torch.fft.ifft.md) |
| REFERENCE-MAPPING-ITEM(`torch.fft.ifft2`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/fft/torch.fft.ifft2.md) |
| REFERENCE-MAPPING-ITEM(`torch.fft.ifftn`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/fft/torch.fft.ifftn.md) |
| REFERENCE-MAPPING-ITEM(`torch.fft.ifftshift`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/fft/torch.fft.ifftshift.md) |
| REFERENCE-MAPPING-ITEM(`torch.fft.ihfft`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/fft/torch.fft.ihfft.md) |
| REFERENCE-MAPPING-ITEM(`torch.fft.ihfft2`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/fft/torch.fft.ihfft2.md) |
| REFERENCE-MAPPING-ITEM(`torch.fft.ihfftn`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/fft/torch.fft.ihfftn.md) |
| REFERENCE-MAPPING-ITEM(`torch.fft.irfft`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/fft/torch.fft.irfft.md) |
| REFERENCE-MAPPING-ITEM(`torch.fft.irfft2`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/fft/torch.fft.irfft2.md) |
| REFERENCE-MAPPING-ITEM(`torch.fft.irfftn`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/fft/torch.fft.irfftn.md) |
| REFERENCE-MAPPING-ITEM(`torch.fft.rfft`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/fft/torch.fft.rfft.md) |
| REFERENCE-MAPPING-ITEM(`torch.fft.rfft2`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/fft/torch.fft.rfft2.md) |
| REFERENCE-MAPPING-ITEM(`torch.fft.rfftfreq`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/fft/torch.fft.rfftfreq.md) |
| REFERENCE-MAPPING-ITEM(`torch.fft.rfftn`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/fft/torch.fft.rfftn.md) |


***持续更新...***

## torch.hub.XX API 映射列表

梳理了`torch.hub.XX`类 API 的 PyTorch-PaddlePaddle API 映射列表。

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ----- | ----------- | ----------------- | ----------- | ------- |
| REFERENCE-MAPPING-ITEM(`torch.hub.download_url_to_file`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/hub/torch.hub.download_url_to_file.md) |
| REFERENCE-MAPPING-ITEM(`torch.hub.help`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/hub/torch.hub.help.md) |
| REFERENCE-MAPPING-ITEM(`torch.hub.list`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/hub/torch.hub.list.md) |
| REFERENCE-MAPPING-ITEM(`torch.hub.load`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/hub/torch.hub.load.md) |
| NOT-IMPLEMENTED-ITEM(`torch.hub.get_dir`, https://pytorch.org/docs/stable/hub.html?highlight=torch+hub+get_dir#torch.hub.get_dir) |
| NOT-IMPLEMENTED-ITEM(`torch.hub.set_dir`, https://pytorch.org/docs/stable/hub.html?highlight=torch+hub+get_dir#torch.hub.set_dir) |


***持续更新...***

## torch.linalg.XX API 映射列表

梳理了`torch.linalg.XX`类 API 的 PyTorch-PaddlePaddle API 映射列表。

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ----- | ----------- | ----------------- | ----------- | ------- |
| REFERENCE-MAPPING-ITEM(`torch.linalg.cholesky`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/linalg/torch.linalg.cholesky.md) |
| REFERENCE-MAPPING-ITEM(`torch.linalg.cond`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/linalg/torch.linalg.cond.md) |
| REFERENCE-MAPPING-ITEM(`torch.linalg.cross`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/linalg/torch.linalg.cross.md) |
| REFERENCE-MAPPING-ITEM(`torch.linalg.det`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/linalg/torch.linalg.det.md) |
| REFERENCE-MAPPING-ITEM(`torch.linalg.diagonal`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/linalg/torch.linalg.diagonal.md) |
| REFERENCE-MAPPING-ITEM(`torch.linalg.eig`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/linalg/torch.linalg.eig.md) |
| REFERENCE-MAPPING-ITEM(`torch.linalg.eigh`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/linalg/torch.linalg.eigh.md) |
| REFERENCE-MAPPING-ITEM(`torch.linalg.eigvals`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/linalg/torch.linalg.eigvals.md) |
| REFERENCE-MAPPING-ITEM(`torch.linalg.eigvalsh`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/linalg/torch.linalg.eigvalsh.md) |
| REFERENCE-MAPPING-ITEM(`torch.linalg.householder_product`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/linalg/torch.linalg.householder_product.md) |
| REFERENCE-MAPPING-ITEM(`torch.linalg.inv`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/linalg/torch.linalg.inv.md) |
| REFERENCE-MAPPING-ITEM(`torch.linalg.lstsq`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/linalg/torch.linalg.lstsq.md) |
| REFERENCE-MAPPING-ITEM(`torch.linalg.lu`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/linalg/torch.linalg.lu.md) |
| REFERENCE-MAPPING-ITEM(`torch.linalg.lu_factor`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/linalg/torch.linalg.lu_factor.md) |
| REFERENCE-MAPPING-ITEM(`torch.linalg.lu_factor_ex`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/linalg/torch.linalg.lu_factor_ex.md) |
| REFERENCE-MAPPING-ITEM(`torch.linalg.matmul`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/linalg/torch.linalg.matmul.md) |
| REFERENCE-MAPPING-ITEM(`torch.linalg.matrix_exp`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/linalg/torch.linalg.matrix_exp.md) |
| REFERENCE-MAPPING-ITEM(`torch.linalg.matrix_norm`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/linalg/torch.linalg.matrix_norm.md) |
| REFERENCE-MAPPING-ITEM(`torch.linalg.matrix_power`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/linalg/torch.linalg.matrix_power.md) |
| REFERENCE-MAPPING-ITEM(`torch.linalg.matrix_rank`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/linalg/torch.linalg.matrix_rank.md) |
| REFERENCE-MAPPING-ITEM(`torch.linalg.multi_dot`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/linalg/torch.linalg.multi_dot.md) |
| REFERENCE-MAPPING-ITEM(`torch.linalg.norm`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/linalg/torch.linalg.norm.md) |
| REFERENCE-MAPPING-ITEM(`torch.linalg.pinv`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/linalg/torch.linalg.pinv.md) |
| REFERENCE-MAPPING-ITEM(`torch.linalg.qr`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/linalg/torch.linalg.qr.md) |
| REFERENCE-MAPPING-ITEM(`torch.linalg.slogdet`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/linalg/torch.linalg.slogdet.md) |
| REFERENCE-MAPPING-ITEM(`torch.linalg.solve`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/linalg/torch.linalg.solve.md) |
| REFERENCE-MAPPING-ITEM(`torch.linalg.solve_triangular`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/linalg/torch.linalg.solve_triangular.md) |
| REFERENCE-MAPPING-ITEM(`torch.linalg.svd`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/linalg/torch.linalg.svd.md) |
| REFERENCE-MAPPING-ITEM(`torch.linalg.svdvals`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/linalg/torch.linalg.svdvals.md) |
| REFERENCE-MAPPING-ITEM(`torch.linalg.vander`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/linalg/torch.linalg.vander.md) |
| REFERENCE-MAPPING-ITEM(`torch.linalg.vector_norm`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/linalg/torch.linalg.vector_norm.md) |
| NOT-IMPLEMENTED-ITEM(`torch.linalg.cholesky_ex`, https://pytorch.org/docs/stable/generated/torch.linalg.cholesky_ex.html#torch.linalg.cholesky_ex) |
| NOT-IMPLEMENTED-ITEM(`torch.linalg.inv_ex`, https://pytorch.org/docs/stable/generated/torch.linalg.inv_ex.html#torch.linalg.inv_ex) |

***持续更新...***

## torch.onnx.XX API 映射列表

梳理了`torch.onnx.XX`类 API 的 PyTorch-PaddlePaddle API 映射列表。

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ----- | ----------- | ----------------- | ----------- | ------- |
| REFERENCE-MAPPING-ITEM(`torch.onnx.export`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.onnx.export.md) |
| NOT-IMPLEMENTED-ITEM(`torch.onnx.disable_log`, https://pytorch.org/docs/stable/onnx.html#torch.onnx.disable_log) |
| NOT-IMPLEMENTED-ITEM(`torch.onnx.enable_log`, https://pytorch.org/docs/stable/onnx.html#torch.onnx.enable_log) |

***持续更新...***

## torch.optim.XX API 映射列表
梳理了`torch.optim.XX`类 API 的 PyTorch-PaddlePaddle API 映射列表。

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ----- | ----------- | ----------------- | ----------- | ------- |
| REFERENCE-MAPPING-ITEM(`torch.optim.ASGD`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/optimizer/torch.optim.ASGD.md) |
| REFERENCE-MAPPING-ITEM(`torch.optim.Adadelta`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/optimizer/torch.optim.Adadelta.md) |
| REFERENCE-MAPPING-ITEM(`torch.optim.Adagrad`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/optimizer/torch.optim.Adagrad.md) |
| REFERENCE-MAPPING-ITEM(`torch.optim.Adam`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/optimizer/torch.optim.Adam.md) |
| REFERENCE-MAPPING-ITEM(`torch.optim.AdamW`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/optimizer/torch.optim.AdamW.md) |
| REFERENCE-MAPPING-ITEM(`torch.optim.Adamax`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/optimizer/torch.optim.Adamax.md) |
| REFERENCE-MAPPING-ITEM(`torch.optim.LBFGS`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/optimizer/torch.optim.LBFGS.md) |
| REFERENCE-MAPPING-ITEM(`torch.optim.Optimizer`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/optimizer/torch.optim.Optimizer.md) |
| REFERENCE-MAPPING-ITEM(`torch.optim.Optimizer.add_param_group`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.optim.Optimizer.add_param_group.md) |
| REFERENCE-MAPPING-ITEM(`torch.optim.Optimizer.load_state_dict`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/optimizer/torch.optim.Optimizer.load_state_dict.md) |
| REFERENCE-MAPPING-ITEM(`torch.optim.Optimizer.state_dict`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/optimizer/torch.optim.Optimizer.state_dict.md) |
| REFERENCE-MAPPING-ITEM(`torch.optim.Optimizer.step`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/optimizer/torch.optim.Optimizer.step.md) |
| REFERENCE-MAPPING-ITEM(`torch.optim.RMSprop`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/optimizer/torch.optim.RMSprop.md) |
| REFERENCE-MAPPING-ITEM(`torch.optim.Rprop`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/optimizer/torch.optim.Rprop.md) |
| REFERENCE-MAPPING-ITEM(`torch.optim.SGD`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/optimizer/torch.optim.SGD.md) |
| REFERENCE-MAPPING-ITEM(`torch.optim.lr_scheduler.ConstantLR`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/optimizer/torch.optim.lr_scheduler.ConstantLR.md) |
| REFERENCE-MAPPING-ITEM(`torch.optim.lr_scheduler.CosineAnnealingLR`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/optimizer/torch.optim.lr_scheduler.CosineAnnealingLR.md) |
| REFERENCE-MAPPING-ITEM(`torch.optim.lr_scheduler.CosineAnnealingWarmRestarts`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/optimizer/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.md) |
| REFERENCE-MAPPING-ITEM(`torch.optim.lr_scheduler.CyclicLR`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/optimizer/torch.optim.lr_scheduler.CyclicLR.md) |
| REFERENCE-MAPPING-ITEM(`torch.optim.lr_scheduler.ExponentialLR`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/optimizer/torch.optim.lr_scheduler.ExponentialLR.md) |
| REFERENCE-MAPPING-ITEM(`torch.optim.lr_scheduler.LambdaLR`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/optimizer/torch.optim.lr_scheduler.LambdaLR.md) |
| REFERENCE-MAPPING-ITEM(`torch.optim.lr_scheduler.LinearLR`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/optimizer/torch.optim.lr_scheduler.LinearLR.md) |
| REFERENCE-MAPPING-ITEM(`torch.optim.lr_scheduler.MultiStepLR`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/optimizer/torch.optim.lr_scheduler.MultiStepLR.md) |
| REFERENCE-MAPPING-ITEM(`torch.optim.lr_scheduler.MultiplicativeLR`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/optimizer/torch.optim.lr_scheduler.MultiplicativeLR.md) |
| REFERENCE-MAPPING-ITEM(`torch.optim.lr_scheduler.OneCycleLR`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/optimizer/torch.optim.lr_scheduler.OneCycleLR.md) |
| REFERENCE-MAPPING-ITEM(`torch.optim.lr_scheduler.ReduceLROnPlateau`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/optimizer/torch.optim.lr_scheduler.ReduceLROnPlateau.md) |
| REFERENCE-MAPPING-ITEM(`torch.optim.lr_scheduler.StepLR`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/optimizer/torch.optim.lr_scheduler.StepLR.md) |

***持续更新...***

## torch.profiler.XX API 映射列表

梳理了`torch.profiler.XX`类 API 的 PyTorch-PaddlePaddle API 映射列表。

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ----- | ----------- | ----------------- | ----------- | ------- |
| REFERENCE-MAPPING-ITEM(`torch.profiler.profile`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.profiler.profile.md) |
| REFERENCE-MAPPING-ITEM(`torch.profiler.schedule`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.profiler.schedule.md) |
| NOT-IMPLEMENTED-ITEM(`torch.profiler.ProfilerAction`, https://pytorch.org/docs/stable/profiler.html#torch.profiler.ProfilerAction) |
| NOT-IMPLEMENTED-ITEM(`torch.profiler.ProfilerActivity`, https://pytorch.org/docs/stable/profiler.html#torch.profiler.ProfilerActivity) |

***持续更新...***

## torch.sparse.XX API 映射列表

梳理了`torch.sparse.XX`类 API 的 PyTorch-PaddlePaddle API 映射列表。

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ----- | ----------- | ----------------- | ----------- | ------- |
| REFERENCE-MAPPING-ITEM(`torch.sparse.addmm`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/sparse/torch.sparse.addmm.md) |
| REFERENCE-MAPPING-ITEM(`torch.sparse.mm`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/sparse/torch.sparse.mm.md) |
| REFERENCE-MAPPING-ITEM(`torch.sparse.softmax`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/sparse/torch.sparse.softmax.md) |
| REFERENCE-MAPPING-ITEM(`torch.sparse.sum`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/sparse/torch.sparse.sum.md) |
| NOT-IMPLEMENTED-ITEM(`torch.sparse.sampled_addmm`, https://pytorch.org/docs/stable/generated/torch.sparse.sampled_addmm.html#torch.sparse.sampled_addmm) |

***持续更新...***

## 其他类 API 映射列表

梳理了其他类 API 的 PyTorch-PaddlePaddle API 映射列表。

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ----- | ----------- | ----------------- | ----------- | ------- |
| REFERENCE-MAPPING-ITEM(`torch.backends.cuda.is_built`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.backends.cuda.is_built.md) |
| REFERENCE-MAPPING-ITEM(`torch.backends.cudnn.is_available`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.backends.cudnn.is_available.md) |
| REFERENCE-MAPPING-ITEM(`torch.backends.cudnn.version`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.backends.cudnn.version.md) |
| REFERENCE-MAPPING-ITEM(`torch.cpu.amp.autocast`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.cpu.amp.autocast.md) |
| REFERENCE-MAPPING-ITEM(`torch.jit.load`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.jit.load.md) |
| REFERENCE-MAPPING-ITEM(`torch.multiprocessing.spawn`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.multiprocessing.spawn.md) |
| REFERENCE-MAPPING-ITEM(`torch.random.get_rng_state`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.random.get_rng_state.md) |
| REFERENCE-MAPPING-ITEM(`torch.random.initial_seed`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.random.initial_seed.md) |
| REFERENCE-MAPPING-ITEM(`torch.random.manual_seed`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.random.manual_seed.md) |
| REFERENCE-MAPPING-ITEM(`torch.random.seed`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.random.seed.md) |
| REFERENCE-MAPPING-ITEM(`torch.random.set_rng_state`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.random.set_rng_state.md) |
| REFERENCE-MAPPING-ITEM(`torch.special.digamma`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.special.digamma.md) |
| REFERENCE-MAPPING-ITEM(`torch.special.erf`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.special.erf.md) |
| REFERENCE-MAPPING-ITEM(`torch.special.erfc`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.special.erfc.md) |
| REFERENCE-MAPPING-ITEM(`torch.special.erfcx`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.special.erfcx.md) |
| REFERENCE-MAPPING-ITEM(`torch.special.erfinv`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.special.erfinv.md) |
| REFERENCE-MAPPING-ITEM(`torch.special.exp2`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.special.exp2.md) |
| REFERENCE-MAPPING-ITEM(`torch.special.expit`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.special.expit.md) |
| REFERENCE-MAPPING-ITEM(`torch.special.expm1`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.special.expm1.md) |
| REFERENCE-MAPPING-ITEM(`torch.special.gammaln`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.special.gammaln.md) |
| REFERENCE-MAPPING-ITEM(`torch.special.i0`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.special.i0.md) |
| REFERENCE-MAPPING-ITEM(`torch.special.i0e`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.special.i0e.md) |
| REFERENCE-MAPPING-ITEM(`torch.special.i1`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.special.i1.md) |
| REFERENCE-MAPPING-ITEM(`torch.special.i1e`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.special.i1e.md) |
| REFERENCE-MAPPING-ITEM(`torch.special.log1p`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.special.log1p.md) |
| REFERENCE-MAPPING-ITEM(`torch.special.log_softmax`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.special.log_softmax.md) |
| REFERENCE-MAPPING-ITEM(`torch.special.logit`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.special.logit.md) |
| REFERENCE-MAPPING-ITEM(`torch.special.logsumexp`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.special.logsumexp.md) |
| REFERENCE-MAPPING-ITEM(`torch.special.multigammaln`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.special.multigammaln.md) |
| REFERENCE-MAPPING-ITEM(`torch.special.ndtri`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.special.ndtri.md) |
| REFERENCE-MAPPING-ITEM(`torch.special.polygamma`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.special.polygamma.md) |
| REFERENCE-MAPPING-ITEM(`torch.special.psi`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.special.psi.md) |
| REFERENCE-MAPPING-ITEM(`torch.special.round`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.special.round.md) |
| REFERENCE-MAPPING-ITEM(`torch.special.sinc`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.special.sinc.md) |
| REFERENCE-MAPPING-ITEM(`torch.special.softmax`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.special.softmax.md) |
| REFERENCE-MAPPING-ITEM(`torch.special.xlog1py`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.special.xlog1py.md) |
| REFERENCE-MAPPING-ITEM(`torch.special.xlogy`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.special.xlogy.md) |
| REFERENCE-MAPPING-ITEM(`torch.testing.assert_allclose`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.testing.assert_allclose.md) |
| REFERENCE-MAPPING-ITEM(`torch.testing.assert_close`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/others/torch.testing.assert_close.md) |
| REFERENCE-MAPPING-ITEM(`torch.utils.cpp_extension.BuildExtension`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/utils/torch.utils.cpp_extension.BuildExtension.md) |
| REFERENCE-MAPPING-ITEM(`torch.utils.cpp_extension.BuildExtension.with_options`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/utils/torch.utils.cpp_extension.BuildExtension.with_options.md) |
| REFERENCE-MAPPING-ITEM(`torch.utils.cpp_extension.CUDAExtension`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/utils/torch.utils.cpp_extension.CUDAExtension.md) |
| REFERENCE-MAPPING-ITEM(`torch.utils.cpp_extension.CUDA_HOME`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/utils/torch.utils.cpp_extension.CUDA_HOME.md) |
| REFERENCE-MAPPING-ITEM(`torch.utils.cpp_extension.CppExtension`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/utils/torch.utils.cpp_extension.CppExtension.md) |
| REFERENCE-MAPPING-ITEM(`torch.utils.cpp_extension.load`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/utils/torch.utils.cpp_extension.load.md) |
| REFERENCE-MAPPING-ITEM(`torch.utils.data.BatchSampler`, https://github.com/PaddlePaddle/X2Paddle/tree/develop/docs/pytorch_project_convertor/API_docs/utils/torch.utils.data.BatchSampler.md) |
| REFERENCE-MAPPING-ITEM(`torch.utils.data.ChainDataset`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/utils/torch.utils.data.ChainDataset.md) |
| REFERENCE-MAPPING-ITEM(`torch.utils.data.ConcatDataset`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/utils/torch.utils.data.ConcatDataset.md) |
| REFERENCE-MAPPING-ITEM(`torch.utils.data.DataLoader`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/utils/torch.utils.data.DataLoader.md) |
| REFERENCE-MAPPING-ITEM(`torch.utils.data.Dataset`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/utils/torch.utils.data.Dataset.md) |
| REFERENCE-MAPPING-ITEM(`torch.utils.data.DistributedSampler`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/utils/torch.utils.data.DistributedSampler.md) |
| REFERENCE-MAPPING-ITEM(`torch.utils.data.IterableDataset`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/utils/torch.utils.data.IterableDataset.md) |
| REFERENCE-MAPPING-ITEM(`torch.utils.data.RandomSampler`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/utils/torch.utils.data.RandomSampler.md) |
| REFERENCE-MAPPING-ITEM(`torch.utils.data.Sampler`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/utils/torch.utils.data.Sampler.md) |
| REFERENCE-MAPPING-ITEM(`torch.utils.data.SequentialSampler`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.utils.data.SequentialSampler.md) |
| REFERENCE-MAPPING-ITEM(`torch.utils.data.Subset`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/utils/torch.utils.data.Subset.md) |
| REFERENCE-MAPPING-ITEM(`torch.utils.data.SubsetRandomSampler`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/utils/torch.utils.data.SubsetRandomSampler.md) |
| REFERENCE-MAPPING-ITEM(`torch.utils.data.TensorDataset`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/utils/torch.utils.data.TensorDataset.md) |
| REFERENCE-MAPPING-ITEM(`torch.utils.data.WeightedRandomSampler`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/utils/torch.utils.data.WeightedRandomSampler.md) |
| REFERENCE-MAPPING-ITEM(`torch.utils.data._utils.collate.default_collate`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/utils/torch.utils.data._utils.collate.default_collate.md) |
| REFERENCE-MAPPING-ITEM(`torch.utils.data.dataloader.default_collate`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/utils/torch.utils.data.dataloader.default_collate.md) |
| REFERENCE-MAPPING-ITEM(`torch.utils.data.default_collate`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/utils/torch.utils.data.default_collate.md) |
| REFERENCE-MAPPING-ITEM(`torch.utils.data.distributed.DistributedSampler`, https://github.com/PaddlePaddle/X2Paddle/tree/develop/docs/pytorch_project_convertor/API_docs/utils/torch.utils.data.distributed.DistributedSampler.md) |
| REFERENCE-MAPPING-ITEM(`torch.utils.data.get_worker_info`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/utils/torch.utils.data.get_worker_info.md) |
| REFERENCE-MAPPING-ITEM(`torch.utils.data.random_split`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/utils/torch.utils.data.random_split.md) |
| REFERENCE-MAPPING-ITEM(`torch.utils.dlpack.from_dlpack`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/utils/torch.utils.dlpack.from_dlpack.md) |
| REFERENCE-MAPPING-ITEM(`torch.utils.dlpack.to_dlpack`, https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/utils/torch.utils.dlpack.to_dlpack.md) |
| NOT-IMPLEMENTED-ITEM(`torch.special.entr`, https://pytorch.org/docs/stable/special.html#torch.special.entr) |
| NOT-IMPLEMENTED-ITEM(`torch.utils.cpp_extension.include_paths`, https://pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.include_paths) |
| NOT-IMPLEMENTED-ITEM(`torch.utils.cpp_extension.load_inline`, https://pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.load_inline) |

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
