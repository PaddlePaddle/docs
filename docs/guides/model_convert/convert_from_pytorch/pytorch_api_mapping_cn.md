# PyTorch 最新 release 与 Paddle develop API 映射表

本文梳理了 PyTorch 最新发行版（当前 v2.8.0） API 与 PaddlePaddle develop 版本 API 对应关系与差异分析。通过本文档，帮助开发者快速迁移 PyTorch 使用经验，完成模型的开发与调优。

## 贡献代码

欢迎你向我们贡献代码，关于如何编写 API 映射关系，为保证文档格式统一性与可读性，请严格参照 [API 映射关系-格式与模板](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_format_cn.md) 来编写。

## API 映射表目录

|类别|简介|
|-|-|
|参数与 API 名均一致|此类 API 功能和使用方法一致，只需将 ``torch.`` 替换为 ``paddle.``|
|参数一致但 API 名不一致|此类 API 功能相同且参数一致，但 API 名不同|
|仅参数名不一致|​  此类 API 功能相同，但部分参数名称不同|
|paddle 参数更多|此类 API 在 PaddlePaddle 中提供了更多可选参数|
|参数默认值不一致|此类 API 功能相同，但某些参数的默认值不同|
|torch 参数更多|​此类 API 在 PyTorch 中提供了更多参数|
|输入参数用法不一致|此类 API 对输入参数的处理方式不同|
|输入参数类型不一致|此类 API 要求的输入数据类型不同|
|返回参数类型不一致|​此类 API 返回值的类型或结构不同|
|组合替代实现|此类功能在 PaddlePaddle 中没有直接对应的单一 API，需要通过多个 PaddlePaddle API 组合来实现|
|可删除|此类 PyTorch API 在 PaddlePaddle 中可以直接删除|
|功能缺失|此类 PyTorch API 的功能在 PaddlePaddle 中暂时没有等效实现|

## 参数与 API 名均一致
**分类依据​​**
此类 API 功能和使用方法在 PyTorch 和 PaddlePaddle 中完全一致，只需将 ``torch.`` 替换为 ``paddle.``

**转写示例**
```python
# PyTorch 写法
x = torch.eye(5)
torch.einsum('ii->i', x)
model = torch.nn.Softplus(beta=0.5, threshold=15)

# Paddle 写法
x = paddle.eye(5)
paddle.einsum('ii->i', x)
model = paddle.nn.Softplus(beta=0.5, threshold=15)
```

| 序号 | Pytorch 最新 release | Paddle develop | 备注 |
|------|-------------------|---------------|------|
| 1 | [torch.Tensor.bfloat16](https://pytorch.org/docs/stable/generated/torch.Tensor.bfloat16.html#torch.Tensor.bfloat16) | paddle.Tensor.bfloat16 | - |
| 2 | [torch.Tensor.bool](https://pytorch.org/docs/stable/generated/torch.Tensor.bool.html#torch.Tensor.bool) | paddle.Tensor.bool | - |
| 3 | [torch.Tensor.byte](https://pytorch.org/docs/stable/generated/torch.Tensor.byte.html#torch.Tensor.byte) | paddle.Tensor.byte | - |
| 4 | [torch.Tensor.char](https://pytorch.org/docs/stable/generated/torch.Tensor.char.html#torch.Tensor.char) | paddle.Tensor.char | - |
| 5 | [torch.Tensor.double](https://pytorch.org/docs/stable/generated/torch.Tensor.double.html#torch-Tensor-double) | paddle.Tensor.double | - |
| 6 | [torch.Tensor.float](https://pytorch.org/docs/stable/generated/torch.Tensor.float.html?highlight=float#torch.Tensor.float) | paddle.Tensor.float | - |
| 7 | [torch.Tensor.half](https://pytorch.org/docs/stable/generated/torch.Tensor.half.html#torch.Tensor.half) | paddle.Tensor.half | - |
| 8 | [torch.Tensor.int](https://pytorch.org/docs/stable/generated/torch.Tensor.int.html?highlight=int#torch.Tensor.int) | paddle.Tensor.int | - |
| 9 | [torch.Tensor.long](https://pytorch.org/docs/stable/generated/torch.Tensor.long.html#torch.Tensor.long) | paddle.Tensor.long | - |
| 10 | [torch.Tensor.short](https://pytorch.org/docs/stable/generated/torch.Tensor.short.html#torch.Tensor.short) | paddle.Tensor.short | - |
| 11 | [torch.Tensor.cfloat](https://pytorch.org/docs/stable/generated/torch.Tensor.cfloat.html?highlight=torch+tensor+cfloat#torch.Tensor.cfloat) | paddle.Tensor.cfloat | - |
| 12 | [torch.Tensor.cdouble](https://pytorch.org/docs/stable/generated/torch.Tensor.cdouble.html?highlight=torch+tensor+cdouble#torch.Tensor.cdouble) | paddle.Tensor.cdouble | - |
| 13 | [torch.nn.init.calculate_gain](https://pytorch.org/docs/stable/nn.init.html?highlight=gain#torch.nn.init.calculate_gain) | paddle.nn.init.calculate_gain | - |
| 14 | [torch.nn.init.constant_](https://pytorch.org/docs/stable/nn.init.html?highlight=constant_#torch.nn.init.constant_) | paddle.nn.init.constant_ | - |
| 15 | [torch.nn.init.dirac_](https://pytorch.org/docs/stable/nn.init.html?highlight=dirac_#torch.nn.init.dirac_) | paddle.nn.init.dirac_ | - |
| 16 | [torch.nn.init.eye_](https://pytorch.org/docs/stable/nn.init.html?highlight=eye_#torch.nn.init.eye_) | paddle.nn.init.eye_ | - |
| 17 | [torch.nn.init.kaiming_normal_](https://pytorch.org/docs/stable/nn.init.html?highlight=kaiming_normal_#torch.nn.init.kaiming_normal_) | paddle.nn.init.kaiming_normal_ | - |
| 18 | [torch.nn.init.kaiming_uniform_](https://pytorch.org/docs/stable/nn.init.html?highlight=kaiming_uniform_#torch.nn.init.kaiming_uniform_) | paddle.nn.init.kaiming_uniform_ | - |
| 19 | [torch.nn.init.normal_](https://pytorch.org/docs/stable/nn.init.html?highlight=normal_#torch.nn.init.normal_) | paddle.nn.init.normal_ | - |
| 20 | torch.nn.init.ones | paddle.nn.init.ones | - |
| 21 | [torch.nn.init.orthogonal_](https://pytorch.org/docs/stable/nn.init.html?highlight=orthogonal_#torch.nn.init.orthogonal_) | paddle.nn.init.orthogonal_ | - |
| 22 | [torch.nn.init.trunc_normal_](https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.trunc_normal_) | paddle.nn.init.trunc_normal_ | - |
| 23 | [torch.nn.init.uniform_](https://pytorch.org/docs/stable/nn.init.html?highlight=uniform_#torch.nn.init.uniform_) | paddle.nn.init.uniform_ | - |
| 24 | [torch.nn.init.xavier_normal_](https://pytorch.org/docs/stable/nn.init.html?highlight=xavier_normal_#torch.nn.init.xavier_normal_) | paddle.nn.init.xavier_normal_ | - |
| 25 | [torch.nn.init.xavier_uniform_](https://pytorch.org/docs/stable/nn.init.html?highlight=xavier_uniform_#torch.nn.init.xavier_uniform_) | paddle.nn.init.xavier_uniform_ | - |
| 26 | [torch.nn.init.zeros_](https://pytorch.org/docs/stable/nn.init.html?highlight=zeros_#torch.nn.init.zeros_) | paddle.nn.init.zeros_ | - |
| 27 | [torch.nn.Conv1d](https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html?highlight=conv1d#torch.nn.Conv1d) | paddle.nn.Conv1d | - |
| 28 | [torch.nn.Conv2d](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html?highlight=conv2d#torch.nn.Conv2d) | paddle.nn.Conv2d | - |
| 29 | [torch.nn.Conv3d](https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html?highlight=conv3d#torch.nn.Conv3d) | paddle.nn.Conv3d | - |
| 30 | [torch.nn.Embedding](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html?highlight=embedding#torch.nn.Embedding) | [paddle.nn.Embedding](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/Embedding_cn.html#embedding) | - |
| 31 | [torch.complex](https://pytorch.org/docs/stable/generated/torch.complex.html) | [paddle.complex](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/complex_cn.html#complex) | - |
| 32 | [torch.polar](https://pytorch.org/docs/stable/generated/torch.polar.html#torch.polar) | [paddle.polar](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/polar_cn.html) | - |
| 33 | [torch.cat](https://pytorch.org/docs/stable/generated/torch.cat.html?highlight=cat#torch.cat) | paddle.cat | - |
| 34 | [torch.stack](https://pytorch.org/docs/stable/generated/torch.stack.html#torch.stack) | [paddle.stack](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/stack_cn.html) | - |
| 35 | [torch.swapaxes](https://pytorch.org/docs/stable/generated/torch.swapaxes.html#torch.swapaxes) | paddle.swapaxes | - |
| 36 | [torch.swapdims](https://pytorch.org/docs/stable/generated/torch.swapdims.html#torch.swapdims) | paddle.swapdims | - |
| 37 | [torch.where](https://pytorch.org/docs/stable/generated/torch.where.html) | [paddle.where](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/where_cn.html) | - |
| 38 | [torch.clamp](https://pytorch.org/docs/stable/generated/torch.clamp.html#torch-clamp) | paddle.clamp | - |
| 39 | torch.clip | [paddle.clip](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/clip_cn.html#clip) | - |
| 40 | [torch.cos](https://pytorch.org/docs/stable/generated/torch.cos.html#torch-cos) | [paddle.cos](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/cos_cn.html#cos) | - |
| 41 | [torch.floor](https://pytorch.org/docs/stable/generated/torch.floor.html?highlight=torch+floor#torch.floor) | [paddle.floor](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/floor_cn.html#floor) | - |
| 42 | [torch.log](https://pytorch.org/docs/stable/generated/torch.log.html?highlight=log#torch.log) | [paddle.log](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/log_cn.html#log) | - |
| 43 | [torch.mul](https://pytorch.org/docs/stable/generated/torch.mul.html?highlight=torch+mul#torch.mul) | paddle.mul | - |
| 44 | torch.multiply | [paddle.multiply](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/multiply_cn.html) | - |
| 45 | [torch.pow](https://pytorch.org/docs/stable/generated/torch.pow.html?highlight=pow#torch.pow) | [paddle.pow](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/pow_cn.html) | - |
| 46 | [torch.rsqrt](https://pytorch.org/docs/stable/generated/torch.rsqrt.html?highlight=rsqrt#torch.rsqrt) | [paddle.rsqrt](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/rsqrt_cn.html#rsqrt) | - |
| 47 | [torch.sign](https://pytorch.org/docs/stable/generated/torch.sign.html?highlight=sign#torch.sign) | [paddle.sign](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/sign_cn.html#sign) | - |
| 48 | [torch.sin](https://pytorch.org/docs/stable/generated/torch.sin.html?highlight=sin#torch.sin) | [paddle.sin](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/sin_cn.html#sin) | - |
| 49 | [torch.eq](https://pytorch.org/docs/stable/generated/torch.eq.html) | paddle.eq | - |
| 50 | [torch.gt](https://pytorch.org/docs/stable/generated/torch.gt.html) | paddle.gt | - |
| 51 | [torch.view_as_real](https://pytorch.org/docs/stable/generated/torch.view_as_real.html?highlight=view_as_real#torch.view_as_real) | paddle.view_as_real | - |
| 52 | [torch.view_as_complex](https://pytorch.org/docs/stable/generated/torch.view_as_complex.html?highlight=view_as_complex#torch.view_as_complex) | paddle.view_as_complex | - |
| 53 | [torch.ger](https://pytorch.org/docs/stable/generated/torch.ger.html?highlight=ger#torch.ger) | paddle.ger | - |
| 54 | [torch.Tensor.mul_](https://pytorch.org/docs/stable/generated/torch.Tensor.mul_.html) | paddle.Tensor.mul_ | - |
| 55 | [torch.Tensor.swapaxes](https://pytorch.org/docs/stable/generated/torch.Tensor.swapaxes.html#torch.Tensor.swapaxes) | paddle.Tensor.swapaxes | - |
| 56 | [torch.Tensor.swapdims](https://pytorch.org/docs/stable/generated/torch.Tensor.swapdims.html#torch.Tensor.swapdims) | paddle.Tensor.swapdims | - |
| 57 | [torch.autograd.Function](https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function) | paddle.autograd.Function | - |
| 58 | [torch.take_along_dim](https://pytorch.org/docs/stable/generated/torch.take_along_dim.html?highlight=torch+take_along_dim#torch.take_along_dim) | paddle.take_along_dim | - |
| 59 | [torch.Tensor.take_along_dim](https://pytorch.org/docs/stable/generated/torch.Tensor.take_along_dim.html?highlight=torch+tensor+take_along_dim#torch.Tensor.take_along_dim) | paddle.Tensor.take_along_dim | - |
| 60 | [torch.special.logsumexp](https://pytorch.org/docs/stable/special.html#torch.special.logsumexp) | paddle.special.logsumexp | - |
| 61 | [torch.argwhere](https://pytorch.org/docs/stable/generated/torch.argwhere.html#torch.argwhere) | paddle.argwhere | - |
| 62 | torch.concatenate | paddle.concatenate | - |
| 63 | torch.is_autocast_enabled | paddle.is_autocast_enabled | - |
| 64 | torch.get_autocast_gpu_dtype | paddle.get_autocast_gpu_dtype | - |
| 65 | [torch.cumsum](https://pytorch.org/docs/stable/generated/torch.cumsum.html?highlight=cumsum#torch.cumsum) | [paddle.cumsum](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/cumsum_cn.html#cumsum) | - |
| 66 | [torch.diff](https://pytorch.org/docs/stable/generated/torch.diff.html?highlight=diff#torch.diff) | [paddle.diff](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/diff_cn.html#diff) | - |
| 67 | [torch.nn.functional.dropout1d](https://pytorch.org/docs/stable/generated/torch.nn.functional.dropout1d.html#torch.nn.functional.dropout1d) | paddle.nn.functional.dropout1d | - |
| 68 | [torch.nn.parameter.Parameter](https://pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html?highlight=torch%20nn%20parameter#torch.nn.parameter.Parameter) | paddle.nn.parameter.Parameter | - |
| 69 | [torch.add](https://pytorch.org/docs/stable/generated/torch.add.html?highlight=add#torch.add) | [paddle.add](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/add_cn.html#add) | - |
| 70 | [torch.div](https://pytorch.org/docs/stable/generated/torch.div.html#torch.div) | paddle.div | - |
| 71 | torch.divide | [paddle.divide](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/divide_cn.html) | - |
| 72 | [torch.true_divide](https://pytorch.org/docs/stable/generated/torch.true_divide.html) | paddle.true_divide | - |
| 73 | [torch.Tensor.add](https://pytorch.org/docs/stable/generated/torch.Tensor.add.html#torch.Tensor.add) | [paddle.Tensor.add](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#add-y-name-none) | - |
| 74 | [torch.Tensor.add_](https://pytorch.org/docs/stable/generated/torch.Tensor.add_.html#torch.Tensor.add_) | [paddle.Tensor.add_](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#id3) | - |
| 75 | [torch.Tensor.div](https://pytorch.org/docs/stable/generated/torch.Tensor.div.html#torch.Tensor.div) | paddle.Tensor.div | - |
| 76 | [torch.Tensor.div_](https://pytorch.org/docs/stable/generated/torch.Tensor.div_.html) | paddle.Tensor.div_ | - |
| 77 | torch.Tensor.divide | [paddle.Tensor.divide](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#divide-y-name-none) | - |
| 78 | torch.Tensor.divide_ | paddle.Tensor.divide_ | - |
| 79 | [torch.Tensor.true_divide](https://pytorch.org/docs/stable/generated/torch.Tensor.true_divide.html#torch.Tensor.true_divide) | paddle.Tensor.true_divide | - |
| 80 | [torch.range](https://pytorch.org/docs/stable/generated/torch.range.html?highlight=range#torch.range) | paddle.range | - |
| 81 | [torch.arange](https://pytorch.org/docs/stable/generated/torch.arange.html?highlight=arange#torch.arange) | [paddle.arange](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/arange_cn.html) | - |
| 82 | [torch.randn](https://pytorch.org/docs/stable/generated/torch.randn.html?highlight=randn#torch.randn) | [paddle.randn](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/randn_cn.html#randn) | - |
| 83 | [torch.zeros](https://pytorch.org/docs/stable/generated/torch.zeros.html?highlight=zeros#torch.zeros) | [paddle.zeros](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/zeros_cn.html) | - |
| 84 | [torch.ones](https://pytorch.org/docs/stable/generated/torch.ones.html?highlight=ones#torch.ones) | [paddle.ones](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/ones_cn.html) | - |
| 85 | [torch.full](https://pytorch.org/docs/stable/generated/torch.full.html?highlight=ful#torch.full) | [paddle.full](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/full_cn.html) | - |
| 86 | [torch.empty](https://pytorch.org/docs/stable/generated/torch.empty.html?highlight=empty#torch.empty) | [paddle.empty](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/empty_cn.html) | - |
| 87 | [torch.zeros_like](https://pytorch.org/docs/stable/generated/torch.zeros_like.html?highlight=zeros_like#torch.zeros_like) | [paddle.zeros_like](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/zeros_like_cn.html) | - |
| 88 | [torch.ones_like](https://pytorch.org/docs/stable/generated/torch.ones_like.html?highlight=ones_like#torch.ones_like) | [paddle.ones_like](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/ones_like_cn.html) | - |
| 89 | [torch.full_like](https://pytorch.org/docs/stable/generated/torch.full_like.html?highlight=full_like#torch.full_like) | [paddle.full_like](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/full_like_cn.html#full-like) | - |
| 90 | [torch.empty_like](https://pytorch.org/docs/stable/generated/torch.empty_like.html?highlight=empty_like#torch.empty_like) | [paddle.empty_like](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/empty_like_cn.html) | - |
| 91 | [torch.Tensor.new_zeros](https://pytorch.org/docs/stable/generated/torch.Tensor.new_zeros.html#torch-tensor-new-zeros) | paddle.Tensor.new_zeros | - |
| 92 | [torch.Tensor.new_ones](https://pytorch.org/docs/stable/generated/torch.Tensor.new_ones.html#torch-tensor-new-ones) | paddle.Tensor.new_ones | - |
| 93 | [torch.Tensor.new_full](https://pytorch.org/docs/stable/generated/torch.Tensor.new_full.html#torch-tensor-new-full) | paddle.Tensor.new_full | - |
| 94 | [torch.Tensor.new_empty](https://pytorch.org/docs/stable/generated/torch.Tensor.new_empty.html#torch-tensor-new-empty) | paddle.Tensor.new_empty | - |
| 95 | [torch.eye](https://pytorch.org/docs/stable/generated/torch.eye.html?highlight=eye#torch.eye) | [paddle.eye](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/eye_cn.html) | - |
| 96 | [torch.permute](https://pytorch.org/docs/stable/generated/torch.permute.html?highlight=permute#torch.permute) | paddle.permute | - |
| 97 | [torch.Tensor.permute](https://pytorch.org/docs/stable/generated/torch.Tensor.permute.html) | paddle.Tensor.permute | - |
| 98 | [torch.repeat_interleave](https://pytorch.org/docs/stable/generated/torch.repeat_interleave.html#torch-repeat-interleave) | [paddle.repeat_interleave](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/repeat_interleave_cn.html#repeat-interleave) | - |
| 99 | [torch.Tensor.repeat_interleave](https://pytorch.org/docs/stable/generated/torch.Tensor.repeat_interleave.html#torch.Tensor.repeat_interleave) | [paddle.Tensor.repeat_interleave](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#repeat-interleave-repeats-axis-none-name-none) | - |
| 100 | [torch.Tensor.repeat](https://pytorch.org/docs/stable/generated/torch.Tensor.repeat.html) | paddle.Tensor.repeat | - |
| 101 | [torch.maximum](https://pytorch.org/docs/stable/generated/torch.maximum.html#torch.maximum) | [paddle.maximum](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/maximum_cn.html) | - |
| 102 | [torch.minimum](https://pytorch.org/docs/stable/generated/torch.minimum.html#torch.minimum) | [paddle.minimum](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/minimum_cn.html) | - |
| 103 | [torch.topk](https://pytorch.org/docs/stable/generated/torch.topk.html?highlight=topk#torch.topk) | [paddle.topk](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/topk_cn.html#paddle.topk) | - |
| 104 | [torch.sqrt](https://pytorch.org/docs/stable/generated/torch.sqrt.html?highlight=sqrt#torch.sqrt) | [paddle.sqrt](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/sqrt_cn.html#sqrt) | - |
| 105 | [torch.amin](https://pytorch.org/docs/stable/generated/torch.amin.html) | [paddle.amin](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/amin_cn.html#amin) | - |
| 106 | [torch.amax](https://pytorch.org/docs/stable/generated/torch.amax.html) | [paddle.amax](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/amax_cn.html#amax) | - |
| 107 | [torch.as_tensor](https://pytorch.org/docs/stable/generated/torch.as_tensor.html#torch.as_tensor) | paddle.as_tensor | - |
| 108 | [torch.tensor](https://pytorch.org/docs/stable/generated/torch.tensor.html?highlight=tensor#torch.tensor) | paddle.tensor | - |
| 109 | [torch.Tensor.copy_](https://pytorch.org/docs/stable/generated/torch.Tensor.copy_.html#torch.Tensor.copy_) | paddle.Tensor.copy_ | - |
| 110 | [torch.Tensor.norm](https://pytorch.org/docs/stable/generated/torch.Tensor.norm.html#torch.Tensor.norm) | [paddle.Tensor.norm](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#norm-p-fro-axis-none-keepdim-false-name-none) | - |
| 111 | [torch.Tensor](https://pytorch.org/docs/stable/tensors.html) | paddle.Tensor | - |
| 112 | [torch.FloatTensor](https://pytorch.org/docs/stable/tensors.html) | paddle.FloatTensor | - |
| 113 | [torch.DoubleTensor](https://pytorch.org/docs/stable/tensors.html) | paddle.DoubleTensor | - |
| 114 | [torch.HalfTensor](https://pytorch.org/docs/stable/tensors.html) | paddle.HalfTensor | - |
| 115 | torch.BFloat16Tensor | paddle.BFloat16Tensor | - |
| 116 | [torch.ByteTensor](https://pytorch.org/docs/stable/tensors.html) | paddle.ByteTensor | - |
| 117 | torch.CharTensor | paddle.CharTensor | - |
| 118 | [torch.ShortTensor](https://pytorch.org/docs/stable/tensors.html) | paddle.ShortTensor | - |
| 119 | [torch.IntTensor](https://pytorch.org/docs/stable/tensors.html) | paddle.IntTensor | - |
| 120 | [torch.LongTensor](https://pytorch.org/docs/stable/tensors.html) | paddle.LongTensor | - |
| 121 | [torch.BoolTensor](https://pytorch.org/docs/stable/tensors.html) | paddle.BoolTensor | - |
| 122 | [torch.norm](https://pytorch.org/docs/stable/generated/torch.norm.html) | paddle.norm | - |
| 123 | [torch.linalg.norm](https://pytorch.org/docs/stable/generated/torch.linalg.norm.html#torch.linalg.norm) | [paddle.linalg.norm](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/linalg/norm_cn.html#norm) | - |
| 124 | [torch.multinomial](https://pytorch.org/docs/stable/generated/torch.multinomial.html#torch.multinomial) | [paddle.multinomial](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/multinomial_cn.html) | - |
| 125 | [torch.var](https://pytorch.org/docs/stable/generated/torch.var.html) | [paddle.var](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/var_cn.html#var) | - |
| 126 | [torch.rand_like](https://pytorch.org/docs/stable/generated/torch.rand_like.html#torch.rand_like) | paddle.rand_like | - |
| 127 | [torch.mean](https://pytorch.org/docs/stable/generated/torch.mean.html) | [paddle.mean](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/mean_cn.html#mean) | - |
| 128 | [torch.Tensor.mean](https://pytorch.org/docs/stable/generated/torch.Tensor.mean.html) | [paddle.Tensor.mean](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#mean-axis-none-keepdim-false-name-none) | - |
| 129 | [torch.msort](https://pytorch.org/docs/stable/generated/torch.msort.html#torch.msort) | paddle.msort | - |
| 130 | [torch.Tensor.msort](https://pytorch.org/docs/stable/generated/torch.Tensor.msort.html#torch.Tensor.msort) | paddle.Tensor.msort | - |
| 131 | [torch.Tensor.ravel](https://pytorch.org/docs/stable/generated/torch.Tensor.ravel.html#torch.Tensor.ravel) | paddle.Tensor.ravel | - |
| 132 | [torch.ravel](https://pytorch.org/docs/stable/generated/torch.ravel.html?highlight=ravel#torch.ravel) | paddle.ravel | - |
| 133 | [torch.Tensor.scatter_add](https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_add.html#torch.Tensor.scatter_add) | paddle.Tensor.scatter_add | - |
| 134 | [torch.scatter_add](https://pytorch.org/docs/stable/generated/torch.scatter_add.html#torch.scatter_add) | paddle.scatter_add | - |
| 135 | [torch.Tensor.scatter_add_](https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_add_.html#torch.Tensor.scatter_add_) | paddle.Tensor.scatter_add_ | - |
| 136 | [torch.Tensor.tril](https://pytorch.org/docs/stable/generated/torch.Tensor.tril.html#torch.Tensor.tril) | paddle.Tensor.tril | - |
| 137 | [torch.tril](https://pytorch.org/docs/stable/generated/torch.tril.html?highlight=tril#torch.tril) | [paddle.tril](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/tril_cn.html) | - |
| 138 | [torch.Tensor.triu](https://pytorch.org/docs/stable/generated/torch.Tensor.triu.html#torch.Tensor.triu) | paddle.Tensor.triu | - |
| 139 | [torch.triu](https://pytorch.org/docs/stable/generated/torch.triu.html?highlight=triu#torch.triu) | [paddle.triu](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/triu_cn.html) | - |
| 140 | [torch.bmm](https://pytorch.org/docs/stable/generated/torch.bmm.html?highlight=bmm#torch.bmm) | [paddle.bmm](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/bmm_cn.html) | - |
| 141 | [torch.Tensor.bmm](https://pytorch.org/docs/stable/generated/torch.Tensor.bmm.html) | [paddle.Tensor.bmm](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#bmm-y-name-none) | - |
| 142 | [torch.nn.GELU](https://pytorch.org/docs/stable/generated/torch.nn.GELU.html#torch.nn.GELU) | [paddle.nn.GELU](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/GELU_cn.html) | - |
| 143 | [torch.broadcast_shapes](https://pytorch.org/docs/stable/generated/torch.broadcast_shapes.html#torch.broadcast_shapes) | paddle.broadcast_shapes | - |
| 144 | [torch.Tensor.scatter_reduce](https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_reduce.html#torch-tensor-scatter-reduce) | paddle.Tensor.scatter_reduce | - |
| 145 | [torch.scatter_reduce](https://pytorch.org/docs/stable/generated/torch.scatter_reduce.html#torch-scatter-reduce) | paddle.scatter_reduce | - |
| 146 | [torch.nn.functional.silu](https://pytorch.org/docs/stable/generated/torch.nn.functional.silu.html) | [paddle.nn.functional.silu](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/functional/silu_cn.html#silu) | - |
| 147 | [torch.Tensor.softmax](https://pytorch.org/docs/stable/generated/torch.Tensor.softmax.html?highlight=softmax#torch.Tensor.softmax) | paddle.Tensor.softmax | - |
| 148 | [torch.special.softmax](https://pytorch.org/docs/stable/special.html#torch.special.softmax) | paddle.special.softmax | - |
| 149 | [torch.softmax](https://pytorch.org/docs/stable/generated/torch.softmax.html) | paddle.softmax | - |
| 150 | [torch.Tensor.clamp](https://pytorch.org/docs/stable/generated/torch.Tensor.clamp.html?highlight=clamp#torch.Tensor.clamp) | paddle.Tensor.clamp | - |
| 151 | [torch.Tensor.itemsize](https://pytorch.org/docs/stable/generated/torch.Tensor.itemsize.html) | paddle.Tensor.itemsize | - |
| 152 | [torch.get_default_dtype](https://pytorch.org/docs/stable/generated/torch.get_default_dtype.html#torch-get-default-dtype) | [paddle.get_default_dtype](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/get_default_dtype_cn.html#get-default-dtype) | - |
| 153 | [torch.einsum](https://pytorch.org/docs/stable/generated/torch.einsum.html#torch.einsum) | [paddle.einsum](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/einsum_cn.html) | - |
| 154 | [torch.nn.Identity](https://pytorch.org/docs/stable/generated/torch.nn.Identity.html#identity) | [paddle.nn.Identity](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/Identity_cn.html#cn-api-paddle-nn-layer-common-identity) | - |
| 155 | [torch.Tensor.ndim](https://pytorch.org/docs/stable/generated/torch.Tensor.ndim.html) | [paddle.Tensor.ndim](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#ndim) | - |
| 156 | [torch.Tensor.T](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.T) | [paddle.Tensor.T](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#tensor) | - |
| 157 | [torch.Tensor.abs](https://pytorch.org/docs/stable/generated/torch.Tensor.abs.html#torch.Tensor.abs) | [paddle.Tensor.abs](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#abs-name-none) | - |
| 158 | [torch.Tensor.cos](https://pytorch.org/docs/stable/generated/torch.Tensor.cos.html?highlight=cos#torch.Tensor.cos) | [paddle.Tensor.cos](paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Tensor_cn.html#cos-name-none) | - |
| 159 | [torch.Tensor.detach](https://pytorch.org/docs/stable/generated/torch.Tensor.detach.html?highlight=detach#torch.Tensor.detach) | [paddle.Tensor.detach](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#detach) | - |
| 160 | [torch.Tensor.dim](https://pytorch.org/docs/stable/generated/torch.Tensor.dim.html?highlight=dim#torch.Tensor.dim) | [paddle.Tensor.dim](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#dim) | - |
| 161 | [torch.Tensor.fill_](https://pytorch.org/docs/stable/generated/torch.Tensor.fill_.html?highlight=fill_#torch.Tensor.fill_) | [paddle.Tensor.fill_](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#fill-x-value-name-none) | - |
| 162 | [torch.Tensor.isnan](https://pytorch.org/docs/stable/generated/torch.Tensor.isnan.html) | [paddle.Tensor.isnan](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#isnan-name-none) | - |
| 163 | [torch.Tensor.item](https://pytorch.org/docs/stable/generated/torch.Tensor.item.html#torch-tensor-item) | [paddle.Tensor.item](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#item-args) | - |
| 164 | [torch.Tensor.log](https://pytorch.org/docs/stable/generated/torch.Tensor.log.html) | [paddle.Tensor.log](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#log-name-none) | - |
| 165 | [torch.Tensor.masked_scatter](https://pytorch.org/docs/stable/generated/torch.Tensor.masked_scatter.html?highlight=masked_scatter#torch.Tensor.masked_scatter) | [paddle.Tensor.masked_scatter](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Tensor_cn.html#masked-scatter-mask-value-name-non) | - |
| 166 | [torch.Tensor.masked_fill_](https://pytorch.org/docs/stable/generated/torch.Tensor.masked_fill_.html?highlight=masked_fill_#torch.Tensor.masked_fill_) | paddle.Tensor.masked_fill_ | - |
| 167 | [torch.Tensor.masked_fill](https://pytorch.org/docs/stable/generated/torch.Tensor.masked_fill.html?highlight=masked_fill#torch.Tensor.masked_fill) | [paddle.Tensor.masked_fill](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Tensor_cn.html#masked-fill-mask-value-name-non) | - |
| 168 | [torch.Tensor.nonzero](https://pytorch.org/docs/stable/generated/torch.Tensor.nonzero.html?highlight=nonzero#torch.Tensor.nonzero) | [paddle.Tensor.nonzero](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#nonzero-as-tuple-false) | - |
| 169 | [torch.Tensor.normal_](https://pytorch.org/docs/stable/generated/torch.Tensor.normal_.html#torch-tensor-normal) | paddle.Tensor.normal_ | - |
| 170 | [torch.Tensor.sigmoid](https://pytorch.org/docs/stable/generated/torch.Tensor.sigmoid) | paddle.Tensor.sigmoid | - |
| 171 | [torch.Tensor.sin](https://pytorch.org/docs/stable/generated/torch.Tensor.sin) | [paddle.Tensor.sin](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#sin-name-none) | - |
| 172 | [torch.Tensor.square](https://pytorch.org/docs/stable/generated/torch.Tensor.square.html#torch-tensor-square) | [paddle.Tensor.square](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#square-name-none) | - |
| 173 | [torch.Tensor.tolist](https://pytorch.org/docs/stable/generated/torch.Tensor.tolist.html#torch.Tensor.tolist) | [paddle.Tensor.tolist](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#tolist) | - |
| 174 | [torch.Tensor.zero_](https://pytorch.org/docs/stable/generated/torch.Tensor.zero_.html#torch.Tensor.zero_) | [paddle.Tensor.zero_](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#zero-x-name-none) | - |
| 175 | [torch.distributed.get_rank](https://pytorch.org/docs/stable/distributed.html#torch.distributed.get_rank) | [paddle.distributed.get_rank](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/get_rank_cn.html) | - |
| 176 | [torch.distributed.get_world_size](https://pytorch.org/docs/stable/distributed.html#torch.distributed.get_world_size) | [paddle.distributed.get_world_size](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/get_world_size_cn.html) | - |
| 177 | [torch.special.softmax](https://pytorch.org/docs/stable/special.html#torch.special.softmax) | paddle.special.softmax | - |
| 178 | [torch.Tensor.shape](https://pytorch.org/docs/stable/generated/torch.Tensor.shape.html) | [paddle.Tensor.shape](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#shape) | - |
| 179 | [torch.float32](https://github.com/pytorch/pytorch/tree/main/torch) | [paddle.float32](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/framework/dtype.pyi#L28) | - |
| 180 | torch.long | paddle.long | - |
| 181 | [torch.int32](https://github.com/pytorch/pytorch/tree/main/torch) | [paddle.int32](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/framework/dtype.pyi#L25) | - |
| 182 | [torch.bfloat16](https://github.com/pytorch/pytorch/tree/main/torch) | [paddle.bfloat16](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/framework/dtype.pyi#L31) | - |
| 183 | [torch.int64](https://github.com/pytorch/pytorch/tree/main/torch) | [paddle.int64](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/framework/dtype.pyi#L26) | - |
| 184 | [torch.bool](https://github.com/pytorch/pytorch/tree/main/torch) | [paddle.bool](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/framework/dtype.pyi#L36) | - |
| 185 | [torch.uint8](https://github.com/pytorch/pytorch/tree/main/torch) | [paddle.uint8](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/framework/dtype.pyi#L22) | - |
| 186 | [torch.Tensor.abs_](https://pytorch.org/docs/stable/generated/torch.Tensor.abs_.html) | paddle.Tensor.abs_ | - |
| 187 | [torch.Tensor.acos](https://pytorch.org/docs/stable/generated/torch.Tensor.acos.html) | paddle.Tensor.acos | - |
| 188 | [torch.Tensor.acos_](https://pytorch.org/docs/stable/generated/torch.Tensor.acos_.html) | paddle.Tensor.acos_ | - |
| 189 | [torch.Tensor.acosh](https://pytorch.org/docs/stable/generated/torch.Tensor.acosh.html?highlight=acosh#torch.Tensor.acosh) | paddle.Tensor.acosh | - |
| 190 | [torch.Tensor.acosh_](https://pytorch.org/docs/stable/generated/torch.Tensor.acosh_.html) | paddle.Tensor.acosh_ | - |
| 191 | [torch.Tensor.angle](https://pytorch.org/docs/stable/generated/torch.Tensor.angle.html) | [paddle.Tensor.angle](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#angle-name-none) | - |
| 192 | [torch.Tensor.apply_](https://pytorch.org/docs/stable/generated/torch.Tensor.apply_.html) | paddle.Tensor.apply_ | - |
| 193 | [torch.Tensor.asin](https://pytorch.org/docs/stable/generated/torch.Tensor.asin.html) | [paddle.Tensor.asin](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#asin-name-none) | - |
| 194 | [torch.Tensor.asin_](https://pytorch.org/docs/stable/generated/torch.Tensor.asin_.html) | paddle.Tensor.asin_ | - |
| 195 | [torch.Tensor.asinh](https://pytorch.org/docs/stable/generated/torch.Tensor.asinh) | paddle.Tensor.asinh | - |
| 196 | [torch.Tensor.asinh_](https://pytorch.org/docs/stable/generated/torch.Tensor.asinh_) | paddle.Tensor.asinh_ | - |
| 197 | [torch.Tensor.atan](https://pytorch.org/docs/stable/generated/torch.Tensor.atan.html) | [paddle.Tensor.atan](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#atan-name-none) | - |
| 198 | [torch.Tensor.atan_](https://pytorch.org/docs/stable/generated/torch.Tensor.atan_.html) | paddle.Tensor.atan_ | - |
| 199 | [torch.Tensor.atanh](https://pytorch.org/docs/stable/generated/torch.Tensor.atanh.html#torch.Tensor.atanh) | paddle.Tensor.atanh | - |
| 200 | [torch.Tensor.atanh_](https://pytorch.org/docs/stable/generated/torch.Tensor.atanh_.html#torch.Tensor.atanh_) | paddle.Tensor.atanh_ | - |
| 201 | [torch.Tensor.bincount](https://pytorch.org/docs/stable/generated/torch.Tensor.bincount.html) | [paddle.Tensor.bincount](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#bincount-weights-none-minlength-0) | - |
| 202 | [torch.Tensor.bitwise_not](https://pytorch.org/docs/stable/generated/torch.Tensor.bitwise_not.html) | paddle.Tensor.bitwise_not | - |
| 203 | [torch.Tensor.bitwise_not_](https://pytorch.org/docs/stable/generated/torch.Tensor.bitwise_not_.html) | paddle.Tensor.bitwise_not_ | - |
| 204 | [torch.Tensor.ceil](https://pytorch.org/docs/stable/generated/torch.Tensor.ceil.html) | [paddle.Tensor.ceil](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#ceil-name-none) | - |
| 205 | [torch.Tensor.ceil_](https://pytorch.org/docs/stable/generated/torch.Tensor.ceil_.html) | [paddle.Tensor.ceil_](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#id7) | - |
| 206 | [torch.Tensor.cholesky](https://pytorch.org/docs/stable/generated/torch.Tensor.cholesky.html) | [paddle.Tensor.cholesky](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#cholesky-upper-false-name-none) | - |
| 207 | [torch.Tensor.cholesky_inverse](https://pytorch.org/docs/stable/generated/torch.cholesky_inverse.html#torch.cholesky_inverse) | [paddle.Tensor.cholesky_inverse](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html) | - |
| 208 | [torch.Tensor.clip](https://pytorch.org/docs/stable/generated/torch.Tensor.clip.html?highlight=clip#torch.Tensor.clip) | [paddle.Tensor.clip](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#clip-min-none-max-none-name-none) | - |
| 209 | [torch.Tensor.clip_](https://pytorch.org/docs/stable/generated/torch.Tensor.clip_.html?highlight=clip_#torch.Tensor.clip_) | [paddle.Tensor.clip_](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#id6) | - |
| 210 | [torch.Tensor.coalesce](https://pytorch.org/docs/stable/generated/torch.Tensor.coalesce.html#torch-tensor-coalesce) | [paddle.Tensor.coalesce](https://www.paddlepaddle.org.cn/documentation/docs/en/develop/api/paddle/Tensor/coalesce_en.html) | - |
| 211 | [torch.Tensor.conj](https://pytorch.org/docs/stable/generated/torch.Tensor.conj.html?highlight=conj#torch.Tensor.conj) | [paddle.Tensor.conj](paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Tensor_cn.html#conj-name-none) | - |
| 212 | [torch.Tensor.cos_](https://pytorch.org/docs/stable/generated/torch.Tensor.cos_.html) | paddle.Tensor.cos_ | - |
| 213 | [torch.Tensor.cosh](https://pytorch.org/docs/stable/generated/torch.Tensor.cosh.html?highlight=cosh#torch.Tensor.cosh) | [paddle.Tensor.cosh](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#cosh-name-none) | - |
| 214 | [torch.Tensor.cosh_](https://pytorch.org/docs/stable/generated/torch.Tensor.cosh_.html) | paddle.Tensor.cosh_ | - |
| 215 | [torch.Tensor.cumprod](https://pytorch.org/docs/stable/generated/torch.Tensor.cumprod.html?highlight=cumprod#torch.Tensor.cumprod) | [paddle.Tensor.cumprod](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/cumprod_cn.html#cumprod) | - |
| 216 | [torch.Tensor.cumprod_](https://pytorch.org/docs/stable/generated/torch.Tensor.cumprod_.html) | paddle.Tensor.cumprod_ | - |
| 217 | [torch.Tensor.data_ptr](https://pytorch.org/docs/stable/generated/torch.Tensor.data_ptr.html) | [paddle.Tensor.data_ptr](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html) | - |
| 218 | [torch.Tensor.deg2rad](https://pytorch.org/docs/stable/generated/torch.Tensor.deg2rad.html?highlight=deg2rad#torch.Tensor.deg2rad) | [paddle.Tensor.deg2rad](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#deg2rad-x-name-none) | - |
| 219 | [torch.Tensor.dense_dim](https://pytorch.org/docs/stable/generated/torch.Tensor.dense_dim.html#torch.Tensor.dense_dim) | [paddle.Tensor.dense_dim](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html) | - |
| 220 | [torch.Tensor.detach_](https://pytorch.org/docs/stable/generated/torch.Tensor.detach_.html) | paddle.Tensor.detach_ | - |
| 221 | [torch.Tensor.diag_embed](https://pytorch.org/docs/stable/generated/torch.Tensor.diag_embed.html) | paddle.Tensor.diag_embed | - |
| 222 | [torch.Tensor.diagflat](https://pytorch.org/docs/stable/generated/torch.Tensor.diagflat.html?highlight=diagflat#torch.Tensor.diagflat) | paddle.Tensor.diagflat | - |
| 223 | [torch.Tensor.digamma](https://pytorch.org/docs/stable/generated/torch.Tensor.digamma.html?highlight=digamma#torch.Tensor.digamma) | [paddle.Tensor.digamma](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#digamma-name-none) | - |
| 224 | [torch.Tensor.digamma_](https://pytorch.org/docs/stable/generated/torch.Tensor.digamma_.html) | paddle.Tensor.digamma_ | - |
| 225 | [torch.Tensor.dtype](https://pytorch.org/docs/stable/generated/torch.Tensor.type.html#torch-tensor-type) | [paddle.Tensor.dtype](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#dtype) | - |
| 226 | [torch.Tensor.erf](https://pytorch.org/docs/stable/generated/torch.Tensor.erf.html?highlight=erf#torch.Tensor.erf) | [paddle.Tensor.erf](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#erf-name-none) | - |
| 227 | [torch.Tensor.erfinv](https://pytorch.org/docs/stable/generated/torch.Tensor.erfinv.html?highlight=erfinv#torch.Tensor.erfinv) | [paddle.Tensor.erfinv](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#erfinv-x-name-none) | - |
| 228 | [torch.Tensor.erfinv_](https://pytorch.org/docs/stable/generated/torch.Tensor.erfinv_.html?highlight=erfinv_#torch.Tensor.erfinv_) | paddle.Tensor.erfinv_ | - |
| 229 | [torch.Tensor.exp](https://pytorch.org/docs/stable/generated/torch.Tensor.exp.html?highlight=exp#torch.Tensor.exp) | [paddle.Tensor.exp](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#exp-name-none) | - |
| 230 | [torch.Tensor.exp_](https://pytorch.org/docs/stable/generated/torch.Tensor.exp_.html?highlight=exp_#torch.Tensor.exp_) | [paddle.Tensor.exp_](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#id7) | - |
| 231 | [torch.Tensor.expm1](https://pytorch.org/docs/stable/generated/torch.Tensor.expm1.html#torch.Tensor.expm1) | paddle.Tensor.expm1 | - |
| 232 | [torch.Tensor.floor](https://pytorch.org/docs/stable/generated/torch.Tensor.floor.html?highlight=floor#torch.Tensor.floor) | [paddle.Tensor.floor](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/floor_cn.html#floor) | - |
| 233 | [torch.Tensor.floor_](https://pytorch.org/docs/stable/generated/torch.Tensor.floor_.html?highlight=floor_#torch.Tensor.floor_) | [paddle.Tensor.floor_](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#id10) | - |
| 234 | [torch.Tensor.frac](https://pytorch.org/docs/stable/generated/torch.Tensor.frac.html?highlight=frac#torch.Tensor.frac) | [paddle.Tensor.frac](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#frac-name-none) | - |
| 235 | [torch.Tensor.frac_](https://pytorch.org/docs/stable/generated/torch.Tensor.frac_.html) | paddle.Tensor.frac_ | - |
| 236 | [torch.Tensor.frexp](https://pytorch.org/docs/stable/generated/torch.Tensor.frexp.html#torch-tensor-frexp) | [paddle.Tensor.frexp](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#frexp-x) | - |
| 237 | [torch.Tensor.grad](https://pytorch.org/docs/stable/generated/torch.Tensor.grad.html) | [paddle.Tensor.grad](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#grad) | - |
| 238 | [torch.Tensor.i0](https://pytorch.org/docs/stable/generated/torch.Tensor.i0.html?highlight=i0#torch.Tensor.i0) | [paddle.Tensor.i0](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#i0-x-name-none) | - |
| 239 | [torch.Tensor.i0_](https://pytorch.org/docs/stable/generated/torch.Tensor.i0_.html) | paddle.Tensor.i0_ | - |
| 240 | [torch.Tensor.indices](https://pytorch.org/docs/stable/generated/torch.Tensor.indices.html#torch.Tensor.indices) | [paddle.Tensor.indices](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/sparse/Overview_cn.html) | - |
| 241 | [torch.Tensor.inverse](https://pytorch.org/docs/stable/generated/torch.Tensor.inverse.html) | paddle.Tensor.inverse | - |
| 242 | [torch.Tensor.is_complex](https://pytorch.org/docs/stable/generated/torch.Tensor.is_complex.html) | [paddle.Tensor.is_complex](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#is-complex) | - |
| 243 | [torch.Tensor.is_floating_point](https://pytorch.org/docs/stable/generated/torch.Tensor.is_floating_point.html) | [paddle.Tensor.is_floating_point](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#is-floating-point-x) | - |
| 244 | [torch.Tensor.is_leaf](https://pytorch.org/docs/stable/generated/torch.Tensor.is_leaf.html) | [paddle.Tensor.is_leaf](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#is-leaf) | - |
| 245 | [torch.Tensor.isfinite](https://pytorch.org/docs/stable/generated/torch.Tensor.isfinite.html) | [paddle.Tensor.isfinite](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#isfinite-name-none) | - |
| 246 | [torch.Tensor.isinf](https://pytorch.org/docs/stable/generated/torch.Tensor.isinf.html) | [paddle.Tensor.isinf](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#isinf-name-none) | - |
| 247 | [torch.Tensor.isneginf](https://pytorch.org/docs/stable/generated/torch.Tensor.isneginf.html#torch.Tensor.isneginf) | [paddle.Tensor.isneginf](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#isneginf-name-none) | - |
| 248 | [torch.Tensor.isposinf](https://pytorch.org/docs/stable/generated/torch.Tensor.isposinf.html#torch.Tensor.isposinf) | [paddle.Tensor.isposinf](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#isposinf-name-none) | - |
| 249 | [torch.Tensor.isreal](https://pytorch.org/docs/stable/generated/torch.Tensor.isreal.html#torch.Tensor.isreal) | [paddle.Tensor.isreal](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#isreal-name-none) | - |
| 250 | [torch.Tensor.istft](https://pytorch.org/docs/stable/generated/torch.Tensor.istft.html#torch.Tensor.istft) | paddle.Tensor.istft | - |
| 251 | [torch.Tensor.lgamma](https://pytorch.org/docs/stable/generated/torch.lgamma.html#torch.lgamma) | [paddle.Tensor.lgamma](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/lgamma_cn.html) | - |
| 252 | [torch.Tensor.lgamma_](https://pytorch.org/docs/stable/generated/torch.Tensor.lgamma_.html) | paddle.Tensor.lgamma_ | - |
| 253 | [torch.Tensor.log10](https://pytorch.org/docs/stable/generated/torch.Tensor.log10.html#torch.Tensor.log10) | [paddle.Tensor.log10](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#log10-name-none) | - |
| 254 | [torch.Tensor.log10_](https://pytorch.org/docs/stable/generated/torch.Tensor.log10_.html) | [paddle.Tensor.log10_](e) | - |
| 255 | [torch.Tensor.log1p](https://pytorch.org/docs/stable/generated/torch.Tensor.log1p.html#torch.Tensor.log1p) | [paddle.Tensor.log1p](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#log1p-name-none) | - |
| 256 | [torch.Tensor.log1p_](https://pytorch.org/docs/stable/generated/torch.Tensor.log1p_.html) | paddle.Tensor.log1p_ | - |
| 257 | [torch.Tensor.log2](https://pytorch.org/docs/stable/generated/torch.Tensor.log2.html#torch.Tensor.log2) | [paddle.Tensor.log2](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#log2-name-none) | - |
| 258 | [torch.Tensor.log2_](https://pytorch.org/docs/stable/generated/torch.Tensor.log2_.html) | paddle.Tensor.log2_ | - |
| 259 | [torch.Tensor.log_](https://pytorch.org/docs/stable/generated/torch.Tensor.log_.html) | paddle.Tensor.log_ | - |
| 260 | [torch.Tensor.logit](https://pytorch.org/docs/stable/generated/torch.Tensor.logit.html) | [paddle.Tensor.logit](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#logit-eps-none-name-none) | - |
| 261 | [torch.Tensor.logit_](https://pytorch.org/docs/stable/generated/torch.Tensor.logit_.html) | [paddle.Tensor.logit_](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/logit_cn.html) | - |
| 262 | [torch.Tensor.lu](https://pytorch.org/docs/stable/generated/torch.Tensor.lu.html) | paddle.Tensor.lu | - |
| 263 | [torch.Tensor.mT](https://pytorch.org/docs/stable/tensors.html?#torch.Tensor.mT) | [paddle.Tensor.mT](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/base/dygraph/math_op_patch.py#L208) | - |
| 264 | [torch.Tensor.masked_scatter_](https://pytorch.org/docs/stable/generated/torch.Tensor.masked_scatter_.html?highlight=masked_scatter#torch.Tensor.masked_scatter_) | paddle.Tensor.masked_scatter_ | - |
| 265 | [torch.Tensor.masked_select](https://pytorch.org/docs/stable/generated/torch.Tensor.masked_select.html?highlight=masked_select#torch.Tensor.masked_select) | [paddle.Tensor.masked_select](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#masked-select-mask-name-none) | - |
| 266 | [torch.Tensor.matrix_power](https://pytorch.org/docs/stable/generated/torch.Tensor.matrix_power.html) | [paddle.Tensor.matrix_power](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#matrix-power-x-n-name-none) | - |
| 267 | [torch.Tensor.mm](https://pytorch.org/docs/stable/generated/torch.Tensor.mm.html) | [paddle.Tensor.mm](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#mm-mat2-name-none) | - |
| 268 | [torch.Tensor.moveaxis](https://pytorch.org/docs/stable/generated/torch.Tensor.moveaxis.html) | [paddle.Tensor.moveaxis](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/moveaxis_cn.html) | - |
| 269 | [torch.Tensor.mv](https://pytorch.org/docs/stable/generated/torch.Tensor.mv.html) | [paddle.Tensor.mv](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#mv-vec-name-none) | - |
| 270 | [torch.Tensor.nan_to_num](https://pytorch.org/docs/stable/generated/torch.Tensor.nan_to_num.html#torch.Tensor.nan_to_num) | [paddle.Tensor.nan_to_num](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#nan-to-num) | - |
| 271 | [torch.Tensor.nan_to_num_](https://pytorch.org/docs/stable/generated/torch.Tensor.nan_to_num_.html#torch.Tensor.nan_to_num_) | paddle.Tensor.nan_to_num_ | - |
| 272 | [torch.Tensor.ndimension](https://pytorch.org/docs/stable/generated/torch.Tensor.ndimension.html?highlight=ndimension#torch.Tensor.ndimension) | [paddle.Tensor.ndimension](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#ndimension) | - |
| 273 | [torch.Tensor.neg](https://pytorch.org/docs/stable/generated/torch.Tensor.neg.html?highlight=neg#torch.Tensor.neg) | [paddle.Tensor.neg](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#neg-name-none) | - |
| 274 | [torch.Tensor.neg_](https://pytorch.org/docs/stable/generated/torch.Tensor.neg_.html) | paddle.Tensor.neg_ | - |
| 275 | [torch.Tensor.pin_memory](https://pytorch.org/docs/stable/generated/torch.Tensor.pin_memory.html?highlight=pin_mem#torch.Tensor.pin_memory) | [paddle.Tensor.pin_memory](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#pin-memory-y-name-none) | - |
| 276 | [torch.Tensor.polygamma](https://pytorch.org/docs/stable/generated/torch.Tensor.polygamma.html?highlight=tensor+polygamma#torch.Tensor.polygamma) | [paddle.Tensor.polygamma](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/polygamma_cn.html#polygamma) | - |
| 277 | [torch.Tensor.polygamma_](https://pytorch.org/docs/stable/generated/torch.Tensor.polygamma_.html) | paddle.Tensor.polygamma_ | - |
| 278 | [torch.Tensor.rad2deg](https://pytorch.org/docs/stable/generated/torch.Tensor.rad2deg.html#torch-tensor-rad2deg) | [paddle.Tensor.rad2deg](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#rad2deg-x-name-none) | - |
| 279 | [torch.Tensor.reciprocal](https://pytorch.org/docs/stable/generated/torch.Tensor.reciprocal.html?highlight=torch+tensor+reciprocal#torch.Tensor.reciprocal) | paddle.Tensor.reciprocal | - |
| 280 | [torch.Tensor.reciprocal_](https://pytorch.org/docs/stable/generated/torch.Tensor.reciprocal_.html?highlight=torch+tensor+reciprocal_#torch.Tensor.reciprocal_) | [paddle.Tensor.reciprocal_](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#id11) | - |
| 281 | [torch.Tensor.register_hook](https://pytorch.org/docs/stable/generated/torch.Tensor.register_hook.html#torch-tensor-register-hook) | [paddle.Tensor.register_hook](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#register-hook-hook) | - |
| 282 | [torch.Tensor.rsqrt](https://pytorch.org/docs/stable/generated/torch.Tensor.rsqrt) | [paddle.Tensor.rsqrt](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#rsqrt-name-none) | - |
| 283 | [torch.Tensor.rsqrt_](https://pytorch.org/docs/stable/generated/torch.Tensor.rsqrt_) | [paddle.Tensor.rsqrt_](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#id15) | - |
| 284 | [torch.Tensor.sgn](https://pytorch.org/docs/stable/generated/torch.Tensor.sgn.html#torch.Tensor.sgn) | [paddle.Tensor.sgn](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#sgn-name-none) | - |
| 285 | [torch.Tensor.sigmoid_](https://pytorch.org/docs/stable/generated/torch.Tensor.sigmoid_) | paddle.Tensor.sigmoid_ | - |
| 286 | [torch.Tensor.sign](https://pytorch.org/docs/stable/generated/torch.Tensor.sign) | [paddle.Tensor.sign](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#sign-name-none) | - |
| 287 | [torch.Tensor.signbit](https://pytorch.org/docs/stable/generated/torch.Tensor.signbit.html#torch-signbit) | [paddle.Tensor.signbit](https://github.com/PaddlePaddle/Paddle/blob/9ce3a54f456011c664c70fbcd318f2e1af0a7d81/python/paddle/tensor/math.py#L7175) | - |
| 288 | [torch.Tensor.sin_](https://pytorch.org/docs/stable/generated/torch.Tensor.sin_.html) | paddle.Tensor.sin_ | - |
| 289 | [torch.Tensor.sinc](https://pytorch.org/docs/stable/generated/torch.Tensor.sinc.html#torch.Tensor.sinc) | [paddle.Tensor.sinc](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/sinc_cn.html#sinc) | - |
| 290 | [torch.Tensor.sinc_](https://pytorch.org/docs/stable/generated/torch.Tensor.sinc_.html#torch-tensor-sinc) | [paddle.Tensor.sinc_](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/sinc__cn.html#sinc) | - |
| 291 | [torch.Tensor.sinh](https://pytorch.org/docs/stable/generated/torch.Tensor.sinh.html?highlight=torch+tensor+sinh#torch.Tensor.sinh) | [paddle.Tensor.sinh](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#sinh-name-none) | - |
| 292 | [torch.Tensor.sinh_](https://pytorch.org/docs/stable/generated/torch.Tensor.sinh_.html) | paddle.Tensor.sinh_ | - |
| 293 | [torch.Tensor.sparse_dim](https://pytorch.org/docs/stable/generated/torch.Tensor.sparse_dim.html#torch.Tensor.sparse_dim) | [paddle.Tensor.sparse_dim](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html) | - |
| 294 | [torch.Tensor.sqrt](https://pytorch.org/docs/stable/generated/torch.sqrt.html) | [paddle.Tensor.sqrt](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#sqrt-name-none) | - |
| 295 | [torch.Tensor.sqrt_](https://pytorch.org/docs/stable/generated/torch.Tensor.sqrt_.html?highlight=torch+tensor+sqrt_#torch.Tensor.sqrt_) | [paddle.Tensor.sqrt_](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#id18) | - |
| 296 | [torch.Tensor.t](https://pytorch.org/docs/stable/generated/torch.Tensor.t.html#torch.Tensor.t) | [paddle.Tensor.t](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#t-name-none) | - |
| 297 | [torch.Tensor.t_](https://pytorch.org/docs/stable/generated/torch.Tensor.t_.html#torch.Tensor.t_) | paddle.Tensor.t_ | - |
| 298 | [torch.Tensor.tan](https://pytorch.org/docs/stable/generated/torch.Tensor.tan.html#torch.Tensor.tan) | paddle.Tensor.tan | - |
| 299 | [torch.Tensor.tan_](https://pytorch.org/docs/stable/generated/torch.Tensor.tan_.html) | paddle.Tensor.tan_ | - |
| 300 | [torch.Tensor.tanh](https://pytorch.org/docs/stable/generated/torch.Tensor.tanh.html#torch.Tensor.tanh) | [paddle.Tensor.tanh](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#tanh-name-none) | - |
| 301 | [torch.Tensor.tanh_](https://pytorch.org/docs/stable/generated/torch.Tensor.tanh_.html#torch.Tensor.tanh_) | [paddle.Tensor.tanh_](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#id22) | - |
| 302 | [torch.Tensor.to_dense](https://pytorch.org/docs/stable/generated/torch.Tensor.to_dense.html#torch-tensor-to-dense) | [paddle.Tensor.to_dense](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor/to_dense_en.html#to-dense) | - |
| 303 | [torch.Tensor.tril_](https://pytorch.org/docs/stable/generated/torch.Tensor.tril_.html#torch.Tensor.tril_) | paddle.Tensor.tril_ | - |
| 304 | [torch.Tensor.triu_](https://pytorch.org/docs/stable/generated/torch.Tensor.triu_.html#torch.Tensor.triu_) | paddle.Tensor.triu_ | - |
| 305 | [torch.Tensor.trunc](https://pytorch.org/docs/stable/generated/torch.Tensor.trunc.html#torch.Tensor.trunc) | [paddle.Tensor.trunc](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#trunc-name-none) | - |
| 306 | [torch.Tensor.trunc_](https://pytorch.org/docs/stable/generated/torch.Tensor.trunc_.html) | paddle.Tensor.trunc_ | - |
| 307 | [torch.Tensor.values](https://pytorch.org/docs/stable/generated/torch.Tensor.values.html?highlight=torch+tensor+values#torch.Tensor.values) | [paddle.Tensor.values](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/sparse/Overview_cn.html) | - |
| 308 | torch.__version__ | paddle.__version__ | - |
| 309 | torch.__version__.split | [paddle.__version__.split](https://github.com/PaddlePaddle/Paddle/tree/develop) | - |
| 310 | [torch.diag_embed](https://pytorch.org/docs/stable/generated/torch.diag_embed.html) | [paddle.diag_embed](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/diag_embed_cn.html) | - |
| 311 | [torch.distributed.ReduceOp.MAX](https://pytorch.org/docs/stable/distributed.html#torch.distributed.ReduceOp) | [paddle.distributed.ReduceOp.MAX](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/ReduceOp_cn.html#reduceop) | - |
| 312 | [torch.distributed.ReduceOp.MIN](https://pytorch.org/docs/stable/distributed.html#torch.distributed.ReduceOp) | [paddle.distributed.ReduceOp.MIN](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/ReduceOp_cn.html#reduceop) | - |
| 313 | [torch.distributed.ReduceOp.SUM](https://pytorch.org/docs/stable/distributed.html#torch.distributed.ReduceOp) | [paddle.distributed.ReduceOp.SUM](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/ReduceOp_cn.html#reduceop) | - |
| 314 | [torch.distributed.batch_isend_irecv](https://pytorch.org/docs/stable/distributed.html#torch.distributed.batch_isend_irecv) | [paddle.distributed.batch_isend_irecv](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/distributed/communication/batch_isend_irecv.py#L134) | - |
| 315 | [torch.distributed.get_backend](https://pytorch.org/docs/stable/distributed.html#torch.distributed.get_backend) | [paddle.distributed.get_backend](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/get_backend_cn.html#get-backend) | - |
| 316 | [torch.distributed.is_available](https://pytorch.org/docs/stable/distributed.html#torch.distributed.is_available) | [paddle.distributed.is_available](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/is_available_cn.html#cn-api-paddle-distributed-is-available) | - |
| 317 | [torch.distributed.is_initialized](https://pytorch.org/docs/stable/distributed.html#torch.distributed.is_initialized) | [paddle.distributed.is_initialized](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/is_initialized_cn.html#is-initialized) | - |
| 318 | [torch.e](https://github.com/pytorch/pytorch/blob/main/torch/__init__.py#L1815) | [paddle.e](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/__init__.py#L787) | - |
| 319 | [torch.enable_grad](https://pytorch.org/docs/stable/generated/torch.enable_grad.html?highlight=enable_grad#torch.enable_grad) | [paddle.enable_grad](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/enable_grad.html#enable-grad) | - |
| 320 | [torch.inf](https://github.com/pytorch/pytorch/blob/main/torch/__init__.py#L1815) | [paddle.inf](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/__init__.py#L784) | - |
| 321 | [torch.is_grad_enabled](https://pytorch.org/docs/stable/generated/torch.is_grad_enabled.html?highlight=torch+is_grad_enabled#torch.is_grad_enabled) | [paddle.is_grad_enabled](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/is_grad_enabled_cn.html#is-grad-enabled) | - |
| 322 | [torch.nan](https://github.com/pytorch/pytorch/blob/main/torch/__init__.py#L1815) | [paddle.nan](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/__init__.py#L785) | - |
| 323 | [torch.newaxis](https://github.com/pytorch/pytorch/blob/main/torch/__init__.py#L1814) | [paddle.newaxis](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/__init__.py#L783) | - |
| 324 | [torch.nn.LogSigmoid](https://pytorch.org/docs/stable/generated/torch.nn.LogSigmoid.html) | [paddle.nn.LogSigmoid](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/LogSigmoid_cn.html#logsigmoid) | - |
| 325 | [torch.nn.Sigmoid](https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html) | [paddle.nn.Sigmoid](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Sigmoid_cn.html#sigmoid) | - |
| 326 | [torch.nn.Softplus](https://pytorch.org/docs/stable/generated/torch.nn.Softplus.html) | [paddle.nn.Softplus](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Softplus_cn.html) | - |
| 327 | [torch.nn.Softsign](https://pytorch.org/docs/stable/generated/torch.nn.Softsign.html) | [paddle.nn.Softsign](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Softsign_cn.html) | - |
| 328 | [torch.nn.Tanh](https://pytorch.org/docs/stable/generated/torch.nn.Tanh.html) | [paddle.nn.Tanh](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Tanh_cn.html) | - |
| 329 | [torch.nn.Tanhshrink](https://pytorch.org/docs/stable/generated/torch.nn.Tanhshrink.html) | [paddle.nn.Tanhshrink](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Tanhshrink_cn.html) | - |
| 330 | [torch.nn.TransformerDecoder](https://pytorch.org/docs/stable/generated/torch.nn.TransformerDecoder.html#transformerdecoder) | [paddle.nn.TransformerDecoder](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/TransformerDecoder_cn.html) | - |
| 331 | [torch.nn.TripletMarginWithDistanceLoss](https://pytorch.org/docs/stable/generated/torch.nn.TripletMarginWithDistanceLoss.html) | [paddle.nn.TripletMarginWithDistanceLoss](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/TripletMarginWithDistanceLoss_cn.html#tripletmarginwithdistanceloss) | - |
| 332 | [torch.nn.utils.parameters_to_vector](https://pytorch.org/docs/stable/generated/torch.nn.utils.parameters_to_vector.html#torch-nn-utils-parameters-to-vector) | [paddle.nn.utils.parameters_to_vector](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/utils/parameters_to_vector_cn.html#parameters-to-vector) | - |
| 333 | [torch.nn.utils.vector_to_parameters](https://pytorch.org/docs/stable/generated/torch.nn.utils.vector_to_parameters.html#torch-nn-utils-vector-to-parameters) | [paddle.nn.utils.vector_to_parameters](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/utils/vector_to_parameters_cn.html#vector-to-parameters) | - |
| 334 | [torch.pi](https://github.com/pytorch/pytorch) | [paddle.pi](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/__init__.py#L786) | - |
| 335 | [torch.set_default_dtype](https://pytorch.org/docs/stable/generated/torch.set_default_dtype.html) | [paddle.set_default_dtype](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/set_default_dtype_cn.html) | - |
| 336 | [torch.t](https://pytorch.org/docs/stable/generated/torch.t.html) | [paddle.t](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/t_cn.html) | - |
| 337 | [torch.utils.cpp_extension.BuildExtension](https://pytorch.org/docs/stable/cpp_extension.html?highlight=cpp_extension#torch.utils.cpp_extension.BuildExtension) | paddle.utils.cpp_extension.BuildExtension | - |
| 338 | torch.utils.cpp_extension.BuildExtension.with_options | paddle.utils.cpp_extension.BuildExtension.with_options | - |
| 339 | [torch.is_grad_enabled](https://pytorch.org/docs/stable/generated/torch.is_grad_enabled.html?highlight=torch+is_grad_enabled#torch.is_grad_enabled) | [paddle.is_grad_enabled](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/is_grad_enabled_cn.html#is-grad-enabled) | - |
| 340 | [torch.nn.Conv2d](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html?highlight=conv2d#torch.nn.Conv2d) | paddle.nn.Conv2d | - |
| 341 | [torch.nn.init.calculate_gain](https://pytorch.org/docs/stable/nn.init.html?highlight=gain#torch.nn.init.calculate_gain) | paddle.nn.init.calculate_gain | - |
| 342 | [torch.nn.init.ones_](https://pytorch.org/docs/stable/nn.init.html?highlight=ones_#torch.nn.init.ones_) | paddle.nn.init.ones_ | - |
| 343 | [torch.nn.init.uniform_](https://pytorch.org/docs/stable/nn.init.html?highlight=uniform_#torch.nn.init.uniform_) | paddle.nn.init.uniform_ | - |
| 344 | [torch.nn.init.zeros_](https://pytorch.org/docs/stable/nn.init.html?highlight=zeros_#torch.nn.init.zeros_) | paddle.nn.init.zeros_ | - |
| 345 | [torch.Tensor.div](https://pytorch.org/docs/stable/generated/torch.Tensor.div.html#torch.Tensor.div) | paddle.Tensor.div | - |
| 346 | [torch.Tensor.element_size](https://pytorch.org/docs/stable/generated/torch.Tensor.element_size.html?highlight=element_size#torch.Tensor.element_size) | [paddle.Tensor.element_size](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#element-size) | - |
| 347 | [torch.Tensor.is_floating_point](https://pytorch.org/docs/stable/generated/torch.Tensor.is_floating_point.html) | [paddle.Tensor.is_floating_point](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#is-floating-point-x) | - |
| 348 | [torch.Tensor.neg](https://pytorch.org/docs/stable/generated/torch.Tensor.neg.html?highlight=neg#torch.Tensor.neg) | [paddle.Tensor.neg](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#neg-name-none) | - |
| 349 | [torch.Tensor.pin_memory](https://pytorch.org/docs/stable/generated/torch.Tensor.pin_memory.html?highlight=pin_mem#torch.Tensor.pin_memory) | [paddle.Tensor.pin_memory](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#pin-memory-y-name-none) | - |
| 350 | [torch.Tensor.view_as](https://pytorch.org/docs/stable/generated/torch.Tensor.view_as.html?highlight=view_as#torch.Tensor.view_as) | [paddle.Tensor.view_as](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#view-as-x-other-name-none) | - |
| 351 | [torch.distributed.is_available](https://pytorch.org/docs/stable/distributed.html#torch.distributed.is_available) | [paddle.distributed.is_available](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/is_available_cn.html#cn-api-paddle-distributed-is-available) | - |
| 352 | [torch.distributed.is_initialized](https://pytorch.org/docs/stable/distributed.html#torch.distributed.is_initialized) | [paddle.distributed.is_initialized](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/is_initialized_cn.html#is-initialized) | - |
| 353 | [torch.set_default_dtype](https://pytorch.org/docs/stable/generated/torch.set_default_dtype.html) | [paddle.set_default_dtype](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/set_default_dtype_cn.html) | - |
| 354 | torch.dtype | paddle.dtype | - |
| 355 | [torch.Tensor.data_ptr](https://pytorch.org/docs/stable/generated/torch.Tensor.data_ptr.html) | [paddle.Tensor.data_ptr](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html) | - |
| 356 | [torch.matmul](https://pytorch.org/docs/stable/generated/torch.matmul.html?highlight=matmul#torch.matmul) | [paddle.matmul](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/matmul_cn.html) | - |
| 357 | torch.linalg.matmul | paddle.linalg.matmul | - |
| 358 | torch.multiply | [paddle.multiply](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/multiply_cn.html) | - |
| 359 | [torch.Tensor.matmul](https://pytorch.org/docs/stable/generated/torch.Tensor.matmul.html) | [paddle.Tensor.matmul](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#matmul-y-transpose-x-false-transpose-y-false-name-none) | - |
| 360 | torch.Tensor.multiply | [paddle.Tensor.multiply](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#multiply-y-axis-1-name-none) | - |
| 361 | [torch.amax](https://pytorch.org/docs/stable/generated/torch.amax.html) | [paddle.amax](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/amax_cn.html#amax) | - |
| 362 | [torch.amin](https://pytorch.org/docs/stable/generated/torch.amin.html) | [paddle.amin](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/amin_cn.html#amin) | - |
| 363 | [torch.Tensor.amax](https://pytorch.org/docs/stable/generated/torch.Tensor.amax.html) | [paddle.Tensor.amax](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#amax-axis-none-keepdim-false-name-none) | - |
| 364 | [torch.Tensor.amin](https://pytorch.org/docs/stable/generated/torch.Tensor.amin.html) | [paddle.Tensor.amin](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#amin-axis-none-keepdim-false-name-none) | - |
| 365 | [torch.Tensor.log2](https://pytorch.org/docs/stable/generated/torch.Tensor.log2.html#torch.Tensor.log2) | [paddle.Tensor.log2](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#log2-name-none) | - |
| 366 | [torch.log2](https://pytorch.org/docs/stable/generated/torch.log2.html?highlight=log2#torch.log2) | [paddle.log2](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/log2_cn.html#log2) | - |
| 367 | [torch.broadcast_to](https://pytorch.org/docs/stable/generated/torch.broadcast_to.html?highlight=broadcast_to#torch.broadcast_to) | [paddle.broadcast_to](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/broadcast_to_cn.html#broadcast-to) | - |
| 368 | [torch.nn.functional.embedding](https://pytorch.org/docs/stable/generated/torch.nn.functional.embedding.html) | [paddle.nn.functional.embedding](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/embedding_cn.html#embedding) | - |
| 369 | [torch.no_grad](https://pytorch.org/docs/stable/generated/torch.no_grad.html) | [paddle.no_grad](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/no_grad_cn.html) | - |
| 370 | [torch.ones_like](https://pytorch.org/docs/stable/generated/torch.ones_like.html?highlight=ones_like#torch.ones_like) | [paddle.ones_like](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/ones_like_cn.html) | - |
| 371 | [torch.reshape](https://pytorch.org/docs/stable/generated/torch.reshape.html?highlight=reshape#torch.reshape) | [paddle.reshape](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/reshape_cn.html#reshape) | - |
| 372 | [torch.take_along_dim](https://pytorch.org/docs/stable/generated/torch.take_along_dim.html?highlight=torch+take_along_dim#torch.take_along_dim) | paddle.take_along_dim | - |
| 373 | [torch.Tensor.bitwise_or_](https://pytorch.org/docs/stable/generated/torch.Tensor.bitwise_or_.html) | paddle.Tensor.bitwise_or_ | - |
| 374 | [torch.Tensor.view](https://pytorch.org/docs/stable/generated/torch.Tensor.view.html?highlight=view#torch.Tensor.view) | [paddle.Tensor.view](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#view-x-shape-or-dtype-name-none) | - |
| 375 | [torch.unique_consecutive](https://pytorch.org/docs/stable/generated/torch.unique_consecutive.html?highlight=unique_consecutive#torch.unique_consecutive) | [paddle.unique_consecutive](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/unique_consecutive_cn.html#unique-consecutive) | - |
| 376 | [torch.eye](https://pytorch.org/docs/stable/generated/torch.eye.html?highlight=eye#torch.eye) | [paddle.eye](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/eye_cn.html) | - |
| 377 | [torch.full_like](https://pytorch.org/docs/stable/generated/torch.full_like.html?highlight=full_like#torch.full_like) | [paddle.full_like](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/full_like_cn.html#full-like) | - |
| 378 | [torch.Tensor.cumsum](https://pytorch.org/docs/stable/generated/torch.Tensor.cumsum.html?highlight=cumsum#torch.Tensor.cumsum) | [paddle.Tensor.cumsum](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#cumsum-axis-none-dtype-none-name-none) | - |
| 379 | [torch.Tensor.expand](https://pytorch.org/docs/stable/generated/torch.Tensor.expand.html?highlight=expand#torch.Tensor.expand) | [paddle.Tensor.expand](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#expand-shape-name-none) | - |
| 380 | torch.clip | [paddle.clip](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/clip_cn.html#clip) | - |
| 381 | [torch.isfinite](https://pytorch.org/docs/stable/generated/torch.isfinite.html?highlight=isfinite#torch.isfinite) | [paddle.isfinite](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/isfinite_cn.html#isfinite) | - |
| 382 | [torch.isinf](https://pytorch.org/docs/stable/generated/torch.isinf.html?highlight=isinf#torch.isinf) | [paddle.isinf](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/isinf_cn.html#isinf) | - |
| 383 | [torch.isnan](https://pytorch.org/docs/stable/generated/torch.isnan.html?highlight=isnan#torch.isnan) | [paddle.isnan](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/isnan_cn.html#isnan) | - |
| 384 | [torch.flatten](https://pytorch.org/docs/stable/generated/torch.flatten.html?highlight=flatten#torch.flatten) | [paddle.flatten](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/flatten_cn.html#flatten) | - |
| 385 | [torch.Tensor.flatten](https://pytorch.org/docs/stable/generated/torch.Tensor.flatten.html?highlight=flatten#torch.Tensor.flatten) | [paddle.Tensor.flatten](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#flatten-start-axis-0-stop-axis-1-name-none) | - |
| 386 | [torch.roll](https://pytorch.org/docs/stable/generated/torch.roll.html?highlight=roll#torch.roll) | [paddle.roll](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/roll_cn.html#roll) | - |
| 387 | [torch.Tensor.sum](https://pytorch.org/docs/stable/generated/torch.Tensor.sum.html) | [paddle.Tensor.sum](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#sum-axis-none-dtype-none-keepdim-false-name-none) | - |
| 388 | [torch.sum](https://pytorch.org/docs/stable/generated/torch.sum.html?highlight=sum#torch.sum) | [paddle.sum](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/sum_cn.html#sum) | - |
| 389 | [torch.repeat_interleave](https://pytorch.org/docs/stable/generated/torch.repeat_interleave.html#torch-repeat-interleave) | [paddle.repeat_interleave](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/repeat_interleave_cn.html#repeat-interleave) | - |
| 390 | [torch.Tensor.repeat_interleave](https://pytorch.org/docs/stable/generated/torch.Tensor.repeat_interleave.html#torch.Tensor.repeat_interleave) | [paddle.Tensor.repeat_interleave](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#repeat-interleave-repeats-axis-none-name-none) | - |
| 391 | [torch.var](https://pytorch.org/docs/stable/generated/torch.var.html) | [paddle.var](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/var_cn.html#var) | - |
| 392 | [torch.prod](https://pytorch.org/docs/stable/generated/torch.prod.html?highlight=prod#torch.prod) | [paddle.prod](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/prod_cn.html#prod) | - |
| 393 | [torch.finfo](https://pytorch.org/docs/stable/type_info.html#torch-finfo) | [paddle.finfo](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/finfo_cn.html) | - |
| 394 | [torch.is_complex](https://pytorch.org/docs/stable/generated/torch.is_complex.html?highlight=is_complex#torch.is_complex) | [paddle.is_complex](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/is_complex_cn.html#is-complex) | - |
| 395 | torch.concat | [paddle.concat](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/concat_cn.html#concat) | - |
| 396 | [torch.empty_like](https://pytorch.org/docs/stable/generated/torch.empty_like.html?highlight=empty_like#torch.empty_like) | [paddle.empty_like](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/empty_like_cn.html) | - |
| 397 | [torch.full](https://pytorch.org/docs/stable/generated/torch.full.html?highlight=ful#torch.full) | [paddle.full](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/full_cn.html) | - |
| 398 | [torch.nonzero](https://pytorch.org/docs/stable/generated/torch.nonzero.html#torch.nonzero) | [paddle.nonzero](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nonzero_cn.html#nonzero) | - |
| 399 | [torch.Tensor.pow](https://pytorch.org/docs/stable/generated/torch.Tensor.pow.html?highlight=pow#torch.Tensor.pow) | paddle.Tensor.pow | - |
| 400 | [torch.Tensor.prod](https://pytorch.org/docs/stable/generated/torch.prod.html#torch.prod) | [paddle.Tensor.prod](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/prod_cn.html) | - |
| 401 | [torch.Tensor.reshape](https://pytorch.org/docs/stable/generated/torch.Tensor.reshape.html) | [paddle.Tensor.reshape](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#reshape-shape-name-none) | - |
| 402 | [torch.zeros_like](https://pytorch.org/docs/stable/generated/torch.zeros_like.html?highlight=zeros_like#torch.zeros_like) | [paddle.zeros_like](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/zeros_like_cn.html) | - |
| 403 | [torch.argsort](https://pytorch.org/docs/stable/generated/torch.argsort.html#torch.argsort) | [paddle.argsort](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/argsort_cn.html#argsort) | - |
| 404 | [torch.Tensor.argsort](https://pytorch.org/docs/stable/generated/torch.Tensor.argsort.html) | [paddle.Tensor.argsort](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#argsort-axis-1-descending-false-name-none) | - |
| 405 | [torch.Tensor.squeeze](https://pytorch.org/docs/stable/generated/torch.Tensor.squeeze.html#torch.Tensor.squeeze) | [paddle.Tensor.squeeze](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#squeeze-axis-none-name-none) | - |
| 406 | [torch.chunk](https://pytorch.org/docs/stable/generated/torch.chunk.html?highlight=chunk#torch.chunk) | [paddle.chunk](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/chunk_cn.html#chunk) | - |
| 407 | [torch.Tensor.chunk](https://pytorch.org/docs/stable/generated/torch.Tensor.chunk.html?highlight=chunk#torch.Tensor.chunk) | [paddle.Tensor.chunk](paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Tensor_cn.html#chunk-chunks-axis-0-name-none) | - |
| 408 | [torch.any](https://pytorch.org/docs/stable/generated/torch.any.html?highlight=any#torch.any) | [paddle.any](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/any_cn.html#any) | - |
| 409 | [torch.unbind](https://pytorch.org/docs/stable/generated/torch.unbind.html?highlight=unbind#torch.unbind) | [paddle.unbind](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/unbind_cn.html#unbind) | - |
| 410 | torch.Tensor.unbindtorch.Tensor.expand_as | paddle.Tensor.unbindpaddle.Tensor.expand_as | - |
| 411 | torch.logsumexp | [paddle.logsumexp](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/logsumexp_cn.html) | - |
| 412 | [torch.Tensor.logsumexp](https://pytorch.org/docs/stable/generated/torch.Tensor.logsumexp.html) | [paddle.Tensor.logsumexp](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#logsumexp-axis-none-keepdim-false-name-none) | - |
| 413 | [torch.argmax](https://pytorch.org/docs/stable/generated/torch.argmax.html?highlight=argmax#torch.argmax) | [paddle.argmax](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/argmax_cn.html#argmax) | - |
| 414 | [torch.Tensor.argmax](https://pytorch.org/docs/stable/generated/torch.Tensor.argmax.html) | [paddle.Tensor.argmax](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#argmax-axis-none-keepdim-false-dtype-int64-name-none) | - |
| 415 | [torch.argmin](https://pytorch.org/docs/stable/generated/torch.argmin.html?highlight=argmin#torch.argmin) | [paddle.argmin](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/argmin_cn.html#argmin) | - |
| 416 | [torch.Tensor.argmin](https://pytorch.org/docs/stable/generated/torch.Tensor.argmin.html) | [paddle.Tensor.argmin](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#argmin-axis-none-keepdim-false-dtype-int64-name-none) | - |
| 417 | [torch.all](https://pytorch.org/docs/stable/generated/torch.all.html?highlight=all#torch.all) | [paddle.all](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/all_cn.html#all) | - |
| 418 | [torch.Tensor.all](https://pytorch.org/docs/stable/generated/torch.Tensor.all.html?highlight=torch+tensor+all#torch.Tensor.all) | [paddle.Tensor.all](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#all-axis-none-keepdim-false-name-none) | - |
| 419 | [torch.Tensor.any](https://pytorch.org/docs/stable/generated/torch.Tensor.any.html?highlight=torch+tensor+any#torch.Tensor.any) | [paddle.Tensor.any](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#any-axis-none-keepdim-false-name-none) | - |
| 420 | [torch.logical_not](https://pytorch.org/docs/stable/generated/torch.logical_not.html?highlight=logical_not#torch.logical_not) | [paddle.logical_not](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/logical_not_cn.html#logical-not) | - |
| 421 | [torch.Tensor.logical_not](https://pytorch.org/docs/stable/generated/torch.Tensor.logical_not.html) | [paddle.Tensor.logical_not](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#logical-not-out-none-name-none) | - |
| 422 | [torch.logical_and](https://pytorch.org/docs/stable/generated/torch.logical_and.html?highlight=logical_and#torch.logical_and) | [paddle.logical_and](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/logical_and_cn.html#logical-and) | - |
| 423 | [torch.Tensor.logical_and](https://pytorch.org/docs/stable/generated/torch.Tensor.logical_and.html) | [paddle.Tensor.logical_and](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#logical-and-y-out-none-name-none) | - |
| 424 | [torch.logical_or](https://pytorch.org/docs/stable/generated/torch.logical_or.html?highlight=logical_or#torch.logical_or) | [paddle.logical_or](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/logical_or_cn.html#logical-or) | - |
| 425 | [torch.Tensor.logical_or](https://pytorch.org/docs/stable/generated/torch.Tensor.logical_or.html) | [paddle.Tensor.logical_or](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#logical-or-y-out-none-name-none) | - |
| 426 | [torch.logical_xor](https://pytorch.org/docs/stable/generated/torch.logical_xor.html?highlight=torch+logical_xor#torch.logical_xor) | [paddle.logical_xor](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/logical_xor_cn.html) | - |
| 427 | [torch.Tensor.logical_xor](https://pytorch.org/docs/stable/generated/torch.Tensor.logical_xor.html) | paddle.Tensor.logical_xor | - |
| 428 | [torch.index_select](https://www.paddlepaddle.org.cn/documentation/docs/stable/develop/api/paddle/index_select_cn.html#index-select) | [paddle.index_select](https://www.paddlepaddle.org.cn/documentation/docs/stable/develop/api/paddle/index_select_cn.html#index-select) | - |
| 429 | [torch.Tensor.index_select](https://pytorch.org/docs/stable/generated/torch.Tensor.index_select.html) | [paddle.Tensor.index_select](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#index-select-index-axis-0-name-none) | - |
| 430 | [torch.dot](https://pytorch.org/docs/stable/generated/torch.dot.html?highlight=dot#torch.dot) | [paddle.dot](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/dot_cn.html#dot) | - |
| 431 | [torch.Tensor.dot](https://pytorch.org/docs/stable/generated/torch.Tensor.dot.html?highlight=dot#torch.Tensor.dot) | [paddle.Tensor.dot](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#dot-y-name-none) | - |
| 432 | [torch.bfloat16](https://github.com/pytorch/pytorch/tree/main/torch) | [paddle.bfloat16](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/framework/dtype.pyi#L31) | - |
| 433 | [torch.bool](https://github.com/pytorch/pytorch/tree/main/torch) | [paddle.bool](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/framework/dtype.pyi#L36) | - |
| 434 | [torch.complex128](https://github.com/pytorch/pytorch/tree/main/torch) | [paddle.complex128](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/framework/dtype.pyi#L34) | - |
| 435 | [torch.complex64](https://github.com/pytorch/pytorch/tree/main/torch) | [paddle.complex64](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/framework/dtype.pyi#L33) | - |
| 436 | [torch.float64](https://github.com/pytorch/pytorch/tree/main/torch) | [paddle.float64](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/framework/dtype.pyi#L29) | - |
| 437 | [torch.float16](https://github.com/pytorch/pytorch/tree/main/torch) | [paddle.float16](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/framework/dtype.pyi#L30) | - |
| 438 | [torch.float32](https://github.com/pytorch/pytorch/tree/main/torch) | [paddle.float32](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/framework/dtype.pyi#L28) | - |
| 439 | [torch.int16](https://github.com/pytorch/pytorch/tree/main/torch) | [paddle.int16](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/framework/dtype.pyi#L24) | - |
| 440 | [torch.int32](https://github.com/pytorch/pytorch/tree/main/torch) | [paddle.int32](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/framework/dtype.pyi#L25) | - |
| 441 | [torch.int64](https://github.com/pytorch/pytorch/tree/main/torch) | [paddle.int64](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/framework/dtype.pyi#L26) | - |
| 442 | [torch.int8](https://github.com/pytorch/pytorch/tree/main/torch) | [paddle.int8](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/framework/dtype.pyi#L23) | - |
| 443 | [torch.ravel](https://pytorch.org/docs/stable/generated/torch.ravel.html?highlight=ravel#torch.ravel) | paddle.ravel | - |
| 444 | [torch.Tensor.narrow](https://pytorch.org/docs/stable/generated/torch.Tensor.narrow.html#torch.Tensor.narrow) | paddle.Tensor.narrow | - |
| 445 | [torch.narrow](https://pytorch.org/docs/stable/generated/torch.narrow.html?highlight=narrow#torch.narrow) | paddle.narrow | - |
| 446 | [torch.Tensor.type_as](https://pytorch.org/docs/stable/generated/torch.Tensor.type_as.html) | paddle.Tensor.type_as | - |
| 447 | [torch.nn.Sequential](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html#torch.nn.Sequential) | [paddle.nn.Sequential](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/Sequential_cn.html) | - |
| 448 | [torch.transpose](https://pytorch.org/docs/stable/generated/torch.transpose.html?highlight=transpose#torch.transpose) | [paddle.transpose](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/transpose_cn.html#transpose) | - |
| 449 | [torch.Tensor.transpose](https://pytorch.org/docs/stable/generated/torch.Tensor.transpose.html) | [paddle.Tensor.transpose](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#transpose-perm-name-none) | - |
| 450 | [torch.unsqueeze](https://pytorch.org/docs/stable/generated/torch.unsqueeze.html?highlight=unsqueeze#torch.unsqueeze) | [paddle.unsqueeze](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/unsqueeze_cn.html#unsqueeze) | - |
| 451 | [torch.Tensor.unsqueeze](https://pytorch.org/docs/stable/generated/torch.Tensor.unsqueeze.html#torch.Tensor.unsqueeze) | [paddle.Tensor.unsqueeze](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#unsqueeze-axis-name-none) | - |
| 452 | torch.sigmoid | paddle.sigmoid | - |
| 453 | [torch.Tensor.topk](https://pytorch.org/docs/stable/generated/torch.Tensor.topk.html#torch.Tensor.topk) | [paddle.Tensor.topk](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#topk-k-axis-none-largest-true-sorted-true-name-none) | - |
| 454 | [torch.outer](https://pytorch.org/docs/stable/generated/torch.outer.html#torch.outer) | [paddle.outer](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/outer_cn.html) | - |
| 455 | [torch.nn.functional.sigmoid](https://pytorch.org/docs/stable/generated/torch.nn.functional.sigmoid.html?highlight=sigmoid#torch.nn.functional.sigmoid) | [paddle.nn.functional.sigmoid](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/functional/sigmoid_cn.html) | - |
| 456 | [torch.Tensor.requires_grad](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.requires_grad.html#torch-tensor-requires-grad) | paddle.Tensor.requires_grad | - |
| 457 | [torch.Tensor.data](https://pytorch.org/docs/stable/tensors.html#torch-tensor) | [paddle.Tensor.data](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#data) | - |
| 458 | [torch.is_tensor](https://pytorch.org/docs/stable/generated/torch.is_tensor.html?highlight=is_tensor#torch.is_tensor) | [paddle.is_tensor](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/is_tensor_cn.html#is-tensor) | - |
| 459 | [torch.Tensor.element_size](https://pytorch.org/docs/stable/generated/torch.Tensor.element_size.html?highlight=element_size#torch.Tensor.element_size) | [paddle.Tensor.element_size](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#element-size) | - |
| 460 | [torch.Tensor.cuda](https://pytorch.org/docs/stable/generated/torch.Tensor.cuda.html#torch.Tensor.cuda) | [paddle.Tensor.cuda](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#cuda-device-id-none-blocking-false) | - |
| 461 | [torch.Tensor.view_as](https://pytorch.org/docs/stable/generated/torch.Tensor.view_as.html?highlight=view_as#torch.Tensor.view_as) | [paddle.Tensor.view_as](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#view-as-x-other-name-none) | - |

## 参数一致但 API 名不一致
**分类依据**
此类 API 两者完全一致，只有 API 名称不同，只需用户将 Pytorch API 名称替换为 Paddle API 名称即可。

**转写示例**
```python
## Pytorch 写法
m = torch.nn.AdaptiveAvgPool1d(5)
y = x.to_sparse(1)

## Paddle 写法
m = paddle.nn.AdaptiveAvgPool1D(5)
y = x.to_sparse_coo(1)
```

| 序号 | Pytorch 最新 release | Paddle develop | 备注 |
|------|-------------------|---------------|------|
| 1 | [torch.Tensor.clamp](https://pytorch.org/docs/stable/generated/torch.Tensor.clamp.html?highlight=clamp#torch.Tensor.clamp) | [paddle.Tensor.clip](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#clip-min-none-max-none-name-none) | - |
| 2 | [torch.Tensor.clamp_](https://pytorch.org/docs/stable/generated/torch.Tensor.clamp_.html?highlight=clamp_#torch.Tensor.clamp_) | [paddle.Tensor.clip_](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#id6) | - |
| 3 | [torch.Tensor.col_indices](https://pytorch.org/docs/stable/generated/torch.Tensor.col_indices.html) | paddle.Tensor.cols | - |
| 4 | [torch.Tensor.conj_physical](https://pytorch.org/docs/stable/generated/torch.Tensor.conj_physical.html#torch.Tensor.conj_physical) | [paddle.Tensor.conj](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#conj-name-none) | - |
| 5 | [torch.Tensor.crow_indices](https://pytorch.org/docs/stable/generated/torch.Tensor.crow_indices.html) | paddle.Tensor.crows | - |
| 6 | [torch.Tensor.det](https://pytorch.org/docs/stable/generated/torch.Tensor.det.html?highlight=det#torch.Tensor.det) | [paddle.linalg.det](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/linalg/det_cn.html#det) | - |
| 7 | [torch.Tensor.device](https://pytorch.org/docs/stable/generated/torch.Tensor.device.html) | [paddle.Tensor.place](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#place) | - |
| 8 | [torch.Tensor.erf_](https://pytorch.org/docs/stable/generated/torch.Tensor.erf_.html) | paddle.erf_ | - |
| 9 | [torch.Tensor.expm1_](https://pytorch.org/docs/stable/generated/torch.Tensor.expm1_.html) | paddle.expm1_ | - |
| 10 | [torch.Tensor.fix](https://pytorch.org/docs/stable/generated/torch.Tensor.fix.html?highlight=fix#torch.Tensor.fix) | [paddle.Tensor.trunc](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#trunc-name-none) | - |
| 11 | [torch.Tensor.fix_](https://pytorch.org/docs/stable/generated/torch.Tensor.fix_.html) | paddle.Tensor.trunc_ | - |
| 12 | [torch.Tensor.get_device](https://pytorch.org/docs/stable/generated/torch.Tensor.get_device.html?highlight=torch+tensor+get_device#torch.Tensor.get_device) | paddle.Tensor.place.gpu_device_id | - |
| 13 | [torch.Tensor.is_inference](https://pytorch.org/docs/stable/generated/torch.Tensor.is_inference.html) | paddle.Tensor.stop_gradient | - |
| 14 | [torch.Tensor.itemsize](https://pytorch.org/docs/stable/generated/torch.Tensor.itemsize.html) | [paddle.Tensor.element_size](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#element-size) | - |
| 15 | [torch.Tensor.matrix_exp](https://pytorch.org/docs/stable/generated/torch.Tensor.matrix_exp.html#torch-tensor-matrix-exp) | [paddle.linalg.matrix_exp](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/linalg/matrix_exp_cn.html) | - |
| 16 | [torch.Tensor.movedim](https://pytorch.org/docs/stable/generated/torch.Tensor.movedim.html) | [paddle.Tensor.moveaxis](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/moveaxis_cn.html) | - |
| 17 | [torch.Tensor.mvlgamma](https://pytorch.org/docs/stable/generated/torch.Tensor.mvlgamma.html#torch-tensor-mvlgamma) | [paddle.Tensor.multigammaln](https://github.com/PaddlePaddle/Paddle/blob/be090bd0bc9ac7a8595296c316b3a6ed3dc60ba6/python/paddle/tensor/math.py#L5099) | - |
| 18 | [torch.Tensor.mvlgamma_](https://pytorch.org/docs/stable/generated/torch.Tensor.mvlgamma_.html#torch-tensor-mvlgamma) | [paddle.Tensor.multigammaln_](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/multigammaln__cn.html#multigammaln) | - |
| 19 | [torch.Tensor.negative](https://pytorch.org/docs/stable/generated/torch.negative.html#torch.negative) | [paddle.Tensor.neg](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/neg_cn.html#cn-api-paddle-neg) | - |
| 20 | [torch.Tensor.negative_](https://pytorch.org/docs/stable/generated/torch.Tensor.negative_.html) | paddle.Tensor.neg_ | - |
| 21 | [torch.Tensor.nelement](https://pytorch.org/docs/stable/generated/torch.Tensor.nelement.html?highlight=nelement#torch.Tensor.nelement) | [paddle.Tensor.size](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/fluid/layers/size_cn.html#cn-api-fluid-layers-size) | - |
| 22 | [torch.Tensor.numel](https://pytorch.org/docs/stable/generated/torch.numel.html?highlight=numel#torch.numel) | [paddle.Tensor.size](https://www.paddlepaddle.org.cn/documentation/docs/guides/beginner/tensor_cn.html#tensor-shape) | - |
| 23 | [torch.Tensor.positive](https://pytorch.org/docs/stable/generated/torch.Tensor.positive.html#torch.Tensor.positive) | [paddle.positive](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/positive_cn.html#positive) | - |
| 24 | [torch.Tensor.retain_grad](https://pytorch.org/docs/stable/generated/torch.Tensor.retain_grad.html) | [paddle.Tensor.retain_grads](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Overview_cn.html#paddle) | - |
| 25 | [torch.Tensor.sparse_mask](https://pytorch.org/docs/stable/generated/torch.Tensor.sparse_mask.html) | [paddle.sparse.mask_as](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/sparse/mask_as_cn.html) | - |
| 26 | [torch.Tensor.square_](https://pytorch.org/docs/stable/generated/torch.Tensor.square_.html) | paddle.square_ | - |
| 27 | [torch.Tensor.to_sparse](https://pytorch.org/docs/stable/generated/torch.Tensor.to_sparse.html#torch.Tensor.to_sparse) | paddle.Tensor.to_sparse_coo | - |
| 28 | [torch.autograd.Function.forward](https://pytorch.org/docs/stable/generated/torch.autograd.Function.forward.html#torch.autograd.Function.forward) | [paddle.autograd.PyLayer.forward](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/autograd/PyLayer_cn.html#forward-ctx-args-kwargs) | - |
| 29 | [torch.autograd.enable_grad](https://pytorch.org/docs/stable/generated/torch.enable_grad.html#enable-grad) | [paddle.enable_grad](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/enable_grad.html#enable-grad) | - |
| 30 | torch.autograd.function.FunctionCtx | [paddle.autograd.PyLayerContext](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/autograd/PyLayerContext_cn.html#pylayercontext) | - |
| 31 | [torch.autograd.function.FunctionCtx.save_for_backward](https://pytorch.org/docs/stable/generated/torch.autograd.function.FunctionCtx.save_for_backward.html#torch.autograd.function.FunctionCtx.save_for_backward) | [paddle.autograd.PyLayerContext.save_for_backward](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/autograd/PyLayerContext_cn.html#save-for-backward-tensors) | - |
| 32 | [torch.autograd.function.FunctionCtx.set_materialize_grads](https://pytorch.org/docs/stable/generated/torch.autograd.function.FunctionCtx.set_materialize_grads.html#torch.autograd.function.FunctionCtx.set_materialize_grads) | [paddle.autograd.PyLayerContext.set_materialize_grads](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/autograd/PyLayerContext_cn.html#set-materialize-grads-self-value) | - |
| 33 | [torch.autograd.grad_mode.set_grad_enabled](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.set_grad_enabled.html#torch.autograd.grad_mode.set_grad_enabled) | [paddle.set_grad_enabled](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/set_grad_enabled_cn.html) | - |
| 34 | [torch.autograd.graph.saved_tensors_hooks](https://pytorch.org/docs/stable/autograd.html?highlight=saved_tensors_hooks#torch.autograd.graph.saved_tensors_hooks) | [paddle.autograd.saved_tensors_hooks](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/autograd/saved_tensors_hooks_cn.html) | - |
| 35 | [torch.backends.cuda.is_built](https://pytorch.org/docs/stable/backends.html?highlight=torch+backends+cudnn+is_available#torch.backends.cuda.is_built) | [paddle.device.is_compiled_with_cuda](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/device/is_compiled_with_cuda_cn.html#is-compiled-with-cuda) | - |
| 36 | [torch.backends.cudnn.version](https://pytorch.org/docs/stable/generated/torch.backends.cudnn.version.html) | [paddle.device.get_cudnn_version](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/device/get_cudnn_version_cn.html#get-cudnn-version) | - |
| 37 | [torch.cpu.current_device](https://pytorch.org/docs/stable/generated/torch.cpu.current_device.html) | [paddle.get_device](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/device/get_device_cn.html#get-device) | - |
| 38 | [torch.cuda.Event](https://pytorch.org/docs/stable/generated/torch.cuda.Event.html#torch.cuda.Event) | [paddle.device.cuda.Event](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/device/cuda/Event_cn.html) | - |
| 39 | [torch.cuda.StreamContext](https://pytorch.org/docs/stable/generated/torch.cuda.StreamContext.html#torch.cuda.StreamContext) | [paddle.device.stream_guard](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/device/stream_guard_cn.html#stream-guard) | - |
| 40 | [torch.cuda.current_device](https://pytorch.org/docs/stable/generated/torch.cuda.current_device.html#torch.cuda.current_device) | [paddle.device.get_device](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/device/get_device_cn.html#get-device) | - |
| 41 | [torch.cuda.device_count](https://pytorch.org/docs/stable/generated/torch.cuda.device_count.html#torch.cuda.device_count) | [paddle.device.cuda.device_count](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/device/cuda/device_count_cn.html) | - |
| 42 | [torch.cuda.empty_cache](https://pytorch.org/docs/stable/generated/torch.cuda.empty_cache.html#torch.cuda.empty_cache) | [paddle.device.cuda.empty_cache](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/device/cuda/empty_cache_cn.html) | - |
| 43 | [torch.cuda.get_device_capability](https://pytorch.org/docs/stable/generated/torch.cuda.get_device_capability.html#torch.cuda.get_device_capability) | [paddle.device.cuda.get_device_capability](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/device/cuda/get_device_capability_cn.html) | - |
| 44 | [torch.cuda.get_device_name](https://pytorch.org/docs/stable/generated/torch.cuda.get_device_name.html) | [paddle.device.cuda.get_device_name](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/device/cuda/get_device_name_cn.html) | - |
| 45 | [torch.cuda.is_bf16_supported](https://pytorch.org/docs/stable/cuda.html) | [paddle.amp.is_bfloat16_supported](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/amp/is_bfloat16_supported_cn.html#is-bfloat16-supported) | - |
| 46 | [torch.cuda.is_initialized](https://docs.pytorch.org/docs/stable/generated/torch.cuda.is_initialized.html#torch-cuda-is-initialized) | [paddle.is_compiled_with_cuda](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/base/framework.py#L980) | - |
| 47 | [torch.cuda.manual_seed_all](https://pytorch.org/docs/2.0/generated/torch.cuda.manual_seed_all.html#torch.cuda.manual_seed_all) | [paddle.seed](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/seed_cn.html) | - |
| 48 | [torch.cuda.max_memory_allocated](https://pytorch.org/docs/stable/generated/torch.cuda.max_memory_allocated.html#torch.cuda.max_memory_allocated) | [paddle.device.cuda.max_memory_allocated](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/device/cuda/max_memory_allocated_cn.html) | - |
| 49 | [torch.cuda.max_memory_reserved](https://pytorch.org/docs/stable/generated/torch.cuda.max_memory_reserved.html#torch.cuda.max_memory_reserved) | [paddle.device.cuda.max_memory_reserved](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/device/cuda/max_memory_reserved_cn.html) | - |
| 50 | [torch.cuda.memory_allocated](https://pytorch.org/docs/stable/generated/torch.cuda.memory_allocated.html#torch.cuda.memory_allocated) | [paddle.device.cuda.memory_allocated](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/device/cuda/memory_allocated_cn.html) | - |
| 51 | [torch.cuda.memory_reserved](https://pytorch.org/docs/stable/generated/torch.cuda.memory_reserved.html#torch.cuda.memory_reserved) | [paddle.device.cuda.memory_reserved](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/device/cuda/memory_reserved_cn.html) | - |
| 52 | [torch.cuda.nvtx.range_pop](https://pytorch.org/docs/stable/generated/torch.cuda.nvtx.range_pop.html#torch.cuda.nvtx.range_pop) | [paddle.framework.core.nvprof_nvtx_pop](https://github.com/PaddlePaddle/Paddle/blob/645dfb4040a15712cea9ccfed4dcb0655aeeb0ea/paddle/fluid/pybind/pybind.cc#L2468) | - |
| 53 | [torch.cuda.reset_max_memory_allocated](https://pytorch.org/docs/stable/generated/torch.cuda.reset_max_memory_allocated.html#torch.cuda.reset_max_memory_allocated) | [paddle.device.cuda.reset_max_memory_allocated](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/device/cuda/reset_max_memory_allocated_cn.html) | - |
| 54 | [torch.cuda.reset_max_memory_cached](https://docs.pytorch.org/docs/stable/generated/torch.cuda.reset_max_memory_cached.html#torch-cuda-reset-max-memory-cached) | [paddle.device.cuda.reset_max_memory_reserved](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/device/cuda/reset_max_memory_reserved_cn.html) | - |
| 55 | [torch.cuda.set_stream](https://pytorch.org/docs/stable/generated/torch.cuda.set_stream.html#torch.cuda.set_stream) | [paddle.device.set_stream](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/device/set_stream_cn.html#set-stream) | - |
| 56 | [torch.cuda.stream](https://pytorch.org/docs/stable/generated/torch.cuda.stream.html) | [paddle.device.stream_guard](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/device/stream_guard_cn.html#stream-guard) | - |
| 57 | [torch.distributed.ReduceOp.PRODUCT](https://pytorch.org/docs/stable/distributed.html#torch.distributed.ReduceOp) | [paddle.distributed.ReduceOp.PROD](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/ReduceOp_cn.html#reduceop) | - |
| 58 | [torch.distributed.is_nccl_available](https://pytorch.org/docs/stable/distributed.html#torch.distributed.is_nccl_available) | [paddle.core.is_compiled_with_nccl](https://github.com/PaddlePaddle/Paddle/blob/61de6003525166856157b6220205fe53df638376/python/paddle/jit/sot/utils/paddle_api_config.py#L159) | - |
| 59 | [torch.distributions.constraints.Constraint](https://pytorch.org/docs/stable/distributions.html#module-torch.distributions.constraints) | [paddle.distribution.constraint.Constraint](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/distribution/constraint.py) | - |
| 60 | [torch.distributions.distribution.Distribution.log_prob](https://pytorch.org/docs/stable/distributions.html#torch.distributions.distribution.Distribution.log_prob) | [paddle.distribution.Distribution.log_prob](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distribution/Distribution_cn.html#log-prob-value) | - |
| 61 | [torch.distributions.kl.kl_divergence](https://pytorch.org/docs/stable/distributions.html?highlight=torch+distributions+kl+kl_divergence#torch.distributions.kl.kl_divergence) | [paddle.distribution.kl_divergence](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distribution/kl_divergence_cn.html) | - |
| 62 | [torch.ge](https://pytorch.org/docs/stable/generated/torch.ge.html) | [paddle.greater_equal](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/greater_equal_cn.html#greater-equal) | - |
| 63 | [torch.get_default_device](https://pytorch.org/docs/stable/generated/torch.get_default_device.html#torch-get-default-device) | [paddle.device.get_device](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/device/get_device_cn.html#get-device) | - |
| 64 | torch.is_inference | paddle.Tensor.stop_gradient | - |
| 65 | [torch.manual_seed](https://pytorch.org/docs/stable/generated/torch.manual_seed.html#torch-manual-seed) | [paddle.seed](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/seed_cn.html) | - |
| 66 | [torch.nn.AdaptiveAvgPool1d](https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool1d.html) | [paddle.nn.AdaptiveAvgPool1D](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/AdaptiveAvgPool1D_cn.html#adaptiveavgpool1d) | - |
| 67 | [torch.nn.HuberLoss](https://pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html#torch.nn.HuberLoss) | [paddle.nn.SmoothL1Loss](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/SmoothL1Loss_cn.html) | - |
| 68 | [torch.nn.Module.apply](https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=torch+nn+module+apply#torch.nn.Module.apply) | [paddle.nn.Layer.apply](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/Layer_cn.html) | - |
| 69 | [torch.nn.Module.children](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.children) | [paddle.nn.Layer.children](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/Layer_cn.html#children) | - |
| 70 | [torch.nn.Module.eval](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.eval) | [paddle.nn.Layer.eval](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/Layer_cn.html#eval) | - |
| 71 | [torch.nn.Module.named_children](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.named_children) | [paddle.nn.Layer.named_children](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/Layer_cn.html#named-children) | - |
| 72 | [torch.nn.Module.train](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.train) | [paddle.nn.Layer.train](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/Layer_cn.html#train) | - |
| 73 | [torch.nn.init.calculate_gain](https://pytorch.org/docs/stable/nn.init.html?highlight=gain#torch.nn.init.calculate_gain) | [paddle.nn.initializer.calculate_gain](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/initializer/calculate_gain_cn.html) | - |
| 74 | [torch.numel](https://pytorch.org/docs/stable/generated/torch.numel.html?highlight=numel#torch.numel) | [paddle.Tensor.size](https://www.paddlepaddle.org.cn/documentation/docs/guides/beginner/tensor_cn.html#tensor-shape) | - |
| 75 | [torch.optim.Optimizer.add_param_group](https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.add_param_group.html?highlight=torch+optim+optimizer+add_param_group#torch.optim.Optimizer.add_param_group) | paddle.optimizer.Optimizer._add_param_group | - |
| 76 | [torch.optim.Optimizer.load_state_dict](https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.load_state_dict.html#torch.optim.Optimizer.load_state_dict) | [paddle.optimizer.Optimizer.load_state_dict](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/optimizer/Optimizer_cn.html) | - |
| 77 | [torch.optim.Optimizer.state_dict](https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.state_dict.html?highlight=torch+optim+optimizer+state_dict#torch.optim.Optimizer.state_dict) | [paddle.optimizer.Optimizer.state_dict](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/optimizer/Optimizer_cn.html) | - |
| 78 | torch.utils.cpp_extension.CUDA_HOME | paddle.utils.cpp_extension.cpp_extension.CUDA_HOME | - |
| 79 | [torch.utils.data.ChainDataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.ChainDataset) | [paddle.io.ChainDataset](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/io/ChainDataset_cn.html) | - |
| 80 | [torch.utils.data.ConcatDataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.ConcatDataset) | [paddle.io.ConcatDataset](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/io/ConcatDataset_cn.html) | - |
| 81 | [torch.utils.data.Dataset](https://pytorch.org/docs/stable/data.html?highlight=torch%20utils%20data%20dataset#torch.utils.data.Dataset) | [paddle.io.Dataset](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/io/Dataset_cn.html#dataset) | - |
| 82 | [torch.utils.data.IterableDataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset) | [paddle.io.IterableDataset](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/io/IterableDataset_cn.html#iterabledataset) | - |
| 83 | [torch.utils.data.RandomSampler](https://pytorch.org/docs/stable/data.html#torch.utils.data.RandomSampler) | [paddle.io.RandomSampler](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/io/RandomSampler_cn.html#paddle.io.RandomSampler) | - |
| 84 | [torch.utils.data.Sampler](https://pytorch.org/docs/stable/data.html#torch.utils.data.Sampler) | [paddle.io.Sampler](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/io/Sampler_cn.html) | - |
| 85 | [torch.utils.data.SequentialSampler](https://pytorch.org/docs/stable/generated/torch.utils.data.SequentialSampler.html) | [paddle.io.SequenceSampler](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/io/SequenceSampler_cn.html#sequencesampler) | - |
| 86 | [torch.utils.data.Subset](https://pytorch.org/docs/stable/data.html#torch.utils.data.Subset) | [paddle.io.Subset](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/io/Subset_cn.html) | - |
| 87 | [torch.utils.data.WeightedRandomSampler](https://pytorch.org/docs/stable/data.html#torch.utils.data.WeightedRandomSampler) | [paddle.io.WeightedRandomSampler](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/io/WeightedRandomSampler_cn.html#paddle.io.WeightedRandomSampler) | - |
| 88 | [torch.utils.data.get_worker_info](https://pytorch.org/docs/stable/data.html#torch.utils.data.get_worker_info) | [paddle.io.get_worker_info](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/io/get_worker_info_cn.html#get-worker-info) | - |
| 89 | [torch.utils.data.random_split](https://pytorch.org/docs/stable/data.html?highlight=torch+utils+data+random_split#torch.utils.data.random_split) | [paddle.io.random_split](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/io/random_split_cn.html) | - |
| 90 | [torchvision.ops.RoIPool](https://pytorch.org/vision/main/generated/torchvision.ops.RoIPool.html) | [paddle.vision.ops.RoIPool](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/vision/ops/RoIPool_cn.html) | - |
| 91 | [torchvision.transforms.Compose](https://pytorch.org/vision/main/generated/torchvision.transforms.Compose.html) | [paddle.vision.transforms.Compose](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/Compose_cn.html) | - |
| 92 | [torchvision.transforms.InterpolationMode.BICUBIC](https://pytorch.org/vision/stable/index.html) | 'bicubic' | - |
| 93 | [torchvision.transforms.InterpolationMode.BILINEAR](https://pytorch.org/vision/stable/index.html) | 'bilinear' | - |
| 94 | [torchvision.transforms.InterpolationMode.BOX](https://pytorch.org/vision/stable/index.html) | 'box' | - |
| 95 | [torchvision.transforms.InterpolationMode.HAMMING](https://pytorch.org/vision/stable/index.html) | 'hamming' | - |
| 96 | [torchvision.transforms.InterpolationMode.LANCZOS](https://pytorch.org/vision/stable/index.html) | 'lanczos' | - |
| 97 | [torchvision.transforms.InterpolationMode.NEAREST](https://pytorch.org/vision/stable/index.html) | 'nearest' | - |
| 98 | [torchvision.transforms.InterpolationMode.NEAREST_EXACT](https://pytorch.org/vision/stable/index.html) | 'nearest_exact' | - |
| 99 | [torchvision.transforms.functional.adjust_brightness](https://pytorch.org/vision/main/generated/torchvision.transforms.functional.adjust_brightness.html) | [paddle.vision.transforms.adjust_brightness](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/adjust_brightness_cn.html) | - |
| 100 | [torchvision.transforms.functional.adjust_contrast](https://pytorch.org/vision/main/generated/torchvision.transforms.functional.adjust_contrast.html) | [paddle.vision.transforms.adjust_contrast](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/adjust_contrast_cn.html) | - |
| 101 | [torchvision.transforms.functional.adjust_hue](https://pytorch.org/vision/main/generated/torchvision.transforms.functional.adjust_hue.html) | [paddle.vision.transforms.adjust_hue](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/adjust_hue_cn.html) | - |
| 102 | [torchvision.transforms.functional.center_crop](https://pytorch.org/vision/main/generated/torchvision.transforms.functional.center_crop.html) | [paddle.vision.transforms.center_crop](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/center_crop_cn.html) | - |
| 103 | [torchvision.transforms.functional.crop](https://pytorch.org/vision/main/generated/torchvision.transforms.functional.crop.html) | [paddle.vision.transforms.crop](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/crop_cn.html) | - |
| 104 | [torchvision.transforms.functional.erase](https://pytorch.org/vision/main/generated/torchvision.transforms.functional.erase.html?highlight=erase#torchvision.transforms.functional.erase) | [paddle.vision.transforms.erase](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/erase_cn.html) | - |
| 105 | [torchvision.transforms.functional.hflip](https://pytorch.org/vision/main/generated/torchvision.transforms.functional.hflip.html) | [paddle.vision.transforms.hflip](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/hflip_cn.html) | - |
| 106 | [torchvision.transforms.functional.pad](https://pytorch.org/vision/main/generated/torchvision.transforms.functional.pad.html) | [paddle.vision.transforms.pad](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/vision/transforms/pad_cn.html) | - |
| 107 | [torchvision.transforms.functional.to_grayscale](https://pytorch.org/vision/main/generated/torchvision.transforms.functional.to_grayscale.html?highlight=to_grayscale#torchvision.transforms.functional.to_grayscale) | [paddle.vision.transforms.to_grayscale](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/to_grayscale_cn.html#to-grayscale) | - |
| 108 | [torchvision.transforms.functional.vflip](https://pytorch.org/vision/main/generated/torchvision.transforms.functional.vflip.html) | [paddle.vision.transforms.vflip](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/vflip_cn.html) | - |

## 仅参数名不一致
**分类依据**
此类 API 功能相同，但部分参数名称不同

| 序号 | Pytorch 最新 release | Paddle develop | 备注 |
|------|-------------------|---------------|------|
新增中......

## paddle 参数更多
**分类依据**
此类 API 在 PaddlePaddle 中提供了更多可选参数

| 序号 | Pytorch 最新 release | Paddle develop | 备注 |
|------|-------------------|---------------|------|
新增中......

## 参数默认值不一致
**分类依据**
此类 API 功能相同，但某些参数的默认值不同

| 序号 | Pytorch 最新 release | Paddle develop | 备注 |
|------|-------------------|---------------|------|
新增中......

## torch 参数更多
**分类依据**
​此类 API 在 PyTorch 中提供了更多参数

| 序号 | Pytorch 最新 release | Paddle develop | 备注 |
|------|-------------------|---------------|------|
新增中......

## 输入参数用法不一致
**分类依据**
此类 API 对输入参数的处理方式不同

| 序号 | Pytorch 最新 release | Paddle develop | 备注 |
|------|-------------------|---------------|------|
新增中......

## 输入参数类型不一致
**分类依据**
此类 API 要求的输入数据类型不同

| 序号 | Pytorch 最新 release | Paddle develop | 备注 |
|------|-------------------|---------------|------|
新增中......

## 返回参数类型不一致
**分类依据**
​此类 API 返回值的类型或结构不同

| 序号 | Pytorch 最新 release | Paddle develop | 备注 |
|------|-------------------|---------------|------|
新增中......

## 组合替代实现
**分类依据**
此类功能在 PaddlePaddle 中没有直接对应的单一 API，需要通过多个 PaddlePaddle API 组合来实现

| 序号 | Pytorch 最新 release | Paddle develop | 备注 |
|------|-------------------|---------------|------|
新增中......

## 可删除
**分类依据**
此类 PyTorch API 在 PaddlePaddle 中可以直接删除

| 序号 | Pytorch 最新 release | Paddle develop | 备注 |
|------|-------------------|---------------|------|
新增中......

## 功能缺失
**分类依据**
此类 PyTorch API 的功能在 PaddlePaddle 中暂时没有等效实现

| 序号 | Pytorch 最新 release | Paddle develop | 备注 |
|------|-------------------|---------------|------|
新增中......
