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
##### 分类依据
此类 API 功能和使用方法在 PyTorch 和 PaddlePaddle 中完全一致，只需将 ``torch.`` 替换为 ``paddle.``

##### 转写示例
### [torch.Tensor.t](https://pytorch.org/docs/stable/generated/torch.Tensor.t.html#torch.Tensor.t)

```python
torch.Tensor.t()
```

### [paddle.Tensor.t](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#t-name-none)

```python
paddle.Tensor.t()
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
此类 API 功能和使用方法在 PyTorch 和 PaddlePaddle 中完全一致,但是 API 名称不同，需要用户进行 API 名称替换。

##### 转写示例
### [torch.nn.BatchNorm1d](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html?highlight=torch%20nn%20batchnorm1d#torch.nn.BatchNorm1d)

```python
torch.nn.BatchNorm1d(num_features,
                     eps=1e-05,
                     momentum=0.1,
                     affine=True,
                     track_running_stats=True,
                     device=None,
                     dtype=None)
```

### [paddle.nn.BatchNorm1D](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/BatchNorm1D_cn.html#batchnorm1d)

```python
paddle.nn.BatchNorm1D(num_features,
                     eps=1e-05,
                     momentum=0.1,
                     affine=True,
                     track_running_stats=True,
                     device=None,
                     dtype=None)
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


## <span id="id1">torch.XX API 映射列表</span>

梳理了`torch.XX`类 API 的 PyTorch-PaddlePaddle API 映射列表。

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ----- | ----------- | ----------------- | ----------- | ------- |
| REFERENCE-MAPPING-TABLE(`torch.`, max_depth=1) |

***持续更新...***

## <span id="id2">torch.nn.XX API 映射列表</span>

梳理了`torch.nn.XX`类 API 的 PyTorch-PaddlePaddle API 映射列表。

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ----- | ----------- | ----------------- | ----------- | ------- |
| REFERENCE-MAPPING-TABLE(`torch.nn.`) |

***持续更新...***

## <span id="id3">torch.nn.functional.XX API 映射列表</span>
梳理了`torch.nn.functional.XX`类 API 的 PyTorch-PaddlePaddle API 映射列表。

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ----- | ----------- | ----------------- | ----------- | ------- |
| REFERENCE-MAPPING-TABLE(`torch.nn.functional`) |

***持续更新...***

## <span id="id4">torch.Tensor.XX API 映射列表</span>

梳理了`torch.Tensor.XX`类 API 的 PyTorch-PaddlePaddle API 映射列表。

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ----- | ----------- | ----------------- | ----------- | ------- |
| REFERENCE-MAPPING-TABLE(`torch.Tensor.`) |

***持续更新...***

## <span id="id5">torch.nn.init.XX API 映射列表</span>

梳理了`torch.nn.init.XX`类 API 的 PyTorch-PaddlePaddle API 映射列表。

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ----- | ----------- | ----------------- | ----------- | ------- |
| REFERENCE-MAPPING-TABLE(`torch.nn.init.`) |

***持续更新...***

## <span id="id6">torch.nn.utils.XX API 映射列表</span>

梳理了`torch.nn.utils.XX`类 API 的 PyTorch-PaddlePaddle API 映射列表。

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ----- | ----------- | ----------------- | ----------- | ------- |
| REFERENCE-MAPPING-TABLE(`torch.nn.utils.`) |

***持续更新...***

## <span id="id7">torch.nn.Module.XX API 映射列表</span>

梳理了`torch.nn.Module.XX`类 API 的 PyTorch-PaddlePaddle API 映射列表。

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ----- | ----------- | ----------------- | ----------- | ------- |
| REFERENCE-MAPPING-TABLE(`torch.nn.Module.`) |

***持续更新...***

## <span id="id8">torch.autograd.XX API 映射列表</span>
梳理了`torch.autograd.XX`类 API 的 PyTorch-PaddlePaddle API 映射列表。

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ----- | ----------- | ----------------- | ----------- | ------- |
| REFERENCE-MAPPING-TABLE(`torch.autograd.`) |

***持续更新...***

## <span id="id9">torch.cuda.XX API 映射列表</span>
梳理了`torch.cuda.XX`类 API 的 PyTorch-PaddlePaddle API 映射列表。

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ----- | ----------- | ----------------- | ----------- | ------- |
| REFERENCE-MAPPING-TABLE(`torch.cuda.`) |

***持续更新...***

## <span id="id10">torch.distributed.XX API 映射列表</span>
梳理了`torch.distributed.XX`类 API 的 PyTorch-PaddlePaddle API 映射列表。

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ----- | ----------- | ----------------- | ----------- | ------- |
| REFERENCE-MAPPING-TABLE(`torch.distributed.`) |

***持续更新...***

## <span id="id11">torch.distributions.XX API 映射列表</span>
梳理了`torch.distributions.XX`类 API 的 PyTorch-PaddlePaddle API 映射列表。

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ----- | ----------- | ----------------- | ----------- | ------- |
| REFERENCE-MAPPING-TABLE(`torch.distributions.`) |

***持续更新...***

## <span id="id12">torch.fft.XX API 映射列表</span>
梳理了`torch.fft.XX`类 API 的 PyTorch-PaddlePaddle API 映射列表。

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ----- | ----------- | ----------------- | ----------- | ------- |
| REFERENCE-MAPPING-TABLE(`torch.fft.`) |

***持续更新...***

## <span id="id13">torch.hub.XX API 映射列表</span>

梳理了`torch.hub.XX`类 API 的 PyTorch-PaddlePaddle API 映射列表。

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ----- | ----------- | ----------------- | ----------- | ------- |
| REFERENCE-MAPPING-TABLE(`torch.hub.`) |


***持续更新...***

## <span id="id14">torch.linalg.XX API 映射列表</span>

梳理了`torch.linalg.XX`类 API 的 PyTorch-PaddlePaddle API 映射列表。

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ----- | ----------- | ----------------- | ----------- | ------- |
| REFERENCE-MAPPING-TABLE(`torch.linalg.`) |

***持续更新...***

## <span id="id15">torch.onnx.XX API 映射列表</span>

梳理了`torch.onnx.XX`类 API 的 PyTorch-PaddlePaddle API 映射列表。

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ----- | ----------- | ----------------- | ----------- | ------- |
| REFERENCE-MAPPING-TABLE(`torch.onnx.`) |

***持续更新...***

## <span id="id16">torch.optim.XX API 映射列表</span>
梳理了`torch.optim.XX`类 API 的 PyTorch-PaddlePaddle API 映射列表。

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ----- | ----------- | ----------------- | ----------- | ------- |
| REFERENCE-MAPPING-TABLE(`torch.optim.`) |

***持续更新...***

## <span id="id17">torch.profiler.XX API 映射列表</span>

梳理了`torch.profiler.XX`类 API 的 PyTorch-PaddlePaddle API 映射列表。

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ----- | ----------- | ----------------- | ----------- | ------- |
| REFERENCE-MAPPING-TABLE(`torch.profiler.`) |

***持续更新...***

## <span id="id18">torch.sparse.XX API 映射列表</span>

梳理了`torch.sparse.XX`类 API 的 PyTorch-PaddlePaddle API 映射列表。

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ----- | ----------- | ----------------- | ----------- | ------- |
| REFERENCE-MAPPING-TABLE(`torch.sparse.`) |

***持续更新...***

## <span id="id19">PyTorch 其他类 API 映射列表</span>

梳理了 PyTorch 其他类 API 的 PyTorch-PaddlePaddle API 映射列表。

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ----- | ----------- | ----------------- | ----------- | ------- |
| REFERENCE-MAPPING-TABLE(`torch.`) |

***持续更新...***

## <span id="id20">fairscale.XX API 映射列表</span>

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ----- | ----------- | ----------------- | ----------- | ------- |
| REFERENCE-MAPPING-TABLE(`fairscale.`) |

***持续更新...***

## <span id="id21">transformers.XX API 映射列表</span>

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ----- | ----------- | ----------------- | ----------- | ------- |
| REFERENCE-MAPPING-TABLE(`transformers.`) |

***持续更新...***

## <span id="id22">flash_attn.XX API 映射列表</span>

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ----- | ----------- | ----------------- | ----------- | ------- |
| REFERENCE-MAPPING-TABLE(`flash_attn.`) |

***持续更新...***

## <span id="id23">torchvision.XX API 映射列表</span>

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ----- | ----------- | ----------------- | ----------- | ------- |
| REFERENCE-MAPPING-TABLE(`torchvision.`) |

***持续更新...***

## <span id="id24">API 别名映射列表</span>

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ---- | -------------------- | -------------- | ----------- | ---- |
| ALIAS-REFERENCE-ITEM(`torch.Tensor.absolute`, `torch.Tensor.abs`) |
| ALIAS-REFERENCE-ITEM(`torch.Tensor.absolute_`, `torch.Tensor.abs_`) |
| ALIAS-REFERENCE-ITEM(`torch.Tensor.arccos`, `torch.Tensor.acos`) |
| ALIAS-REFERENCE-ITEM(`torch.Tensor.arccos_`, `torch.Tensor.acos_`) |
| ALIAS-REFERENCE-ITEM(`torch.Tensor.arccosh`, `torch.Tensor.acosh`) |
| ALIAS-REFERENCE-ITEM(`torch.Tensor.arccosh_`, `torch.Tensor.acosh_`) |
| ALIAS-REFERENCE-ITEM(`torch.Tensor.arcsin`, `torch.Tensor.asin`) |
| ALIAS-REFERENCE-ITEM(`torch.Tensor.arcsin_`, `torch.Tensor.asin_`) |
| ALIAS-REFERENCE-ITEM(`torch.Tensor.arcsinh`, `torch.Tensor.asinh`) |
| ALIAS-REFERENCE-ITEM(`torch.Tensor.arcsinh_`, `torch.Tensor.asinh_`) |
| ALIAS-REFERENCE-ITEM(`torch.Tensor.arctan`, `torch.Tensor.atan`) |
| ALIAS-REFERENCE-ITEM(`torch.Tensor.arctan2`, `torch.Tensor.atan2`) |
| ALIAS-REFERENCE-ITEM(`torch.Tensor.arctan_`, `torch.Tensor.atan_`) |
| ALIAS-REFERENCE-ITEM(`torch.Tensor.arctanh`, `torch.Tensor.atanh`) |
| ALIAS-REFERENCE-ITEM(`torch.Tensor.arctanh_`, `torch.Tensor.atanh_`) |
| ALIAS-REFERENCE-ITEM(`torch.Tensor.divide`, `torch.Tensor.div`) |
| ALIAS-REFERENCE-ITEM(`torch.Tensor.divide_`, `torch.Tensor.div_`) |
| ALIAS-REFERENCE-ITEM(`torch.Tensor.greater`, `torch.Tensor.gt`) |
| ALIAS-REFERENCE-ITEM(`torch.Tensor.greater_`, `torch.Tensor.gt_`) |
| ALIAS-REFERENCE-ITEM(`torch.Tensor.greater_equal`, `torch.Tensor.ge`) |
| ALIAS-REFERENCE-ITEM(`torch.Tensor.greater_equal_`, `torch.Tensor.ge_`) |
| ALIAS-REFERENCE-ITEM(`torch.Tensor.less`, `torch.Tensor.lt`) |
| ALIAS-REFERENCE-ITEM(`torch.Tensor.less_`, `torch.Tensor.lt_`) |
| ALIAS-REFERENCE-ITEM(`torch.Tensor.less_equal`, `torch.Tensor.le`) |
| ALIAS-REFERENCE-ITEM(`torch.Tensor.less_equal_`, `torch.Tensor.le_`) |
| ALIAS-REFERENCE-ITEM(`torch.Tensor.multiply`, `torch.Tensor.mul`) |
| ALIAS-REFERENCE-ITEM(`torch.Tensor.multiply_`, `torch.Tensor.mul_`) |
| ALIAS-REFERENCE-ITEM(`torch.Tensor.not_equal`, `torch.Tensor.ne`) |
| ALIAS-REFERENCE-ITEM(`torch.Tensor.not_equal_`, `torch.Tensor.ne_`) |
| ALIAS-REFERENCE-ITEM(`torch.Tensor.subtract`, `torch.Tensor.sub`) |
| ALIAS-REFERENCE-ITEM(`torch.Tensor.subtract_`, `torch.Tensor.sub_`) |
| ALIAS-REFERENCE-ITEM(`torch.absolute`, `torch.abs`) |
| ALIAS-REFERENCE-ITEM(`torch.absolute_`, `torch.abs_`) |
| ALIAS-REFERENCE-ITEM(`torch.adaptive_avg_pool1d`, `torch.nn.functional.adaptive_avg_pool1d`) |
| ALIAS-REFERENCE-ITEM(`torch.amp.autocast_mode.autocast`, `torch.amp.autocast`) |
| ALIAS-REFERENCE-ITEM(`torch.arccos`, `torch.acos`) |
| ALIAS-REFERENCE-ITEM(`torch.arccosh`, `torch.acosh`) |
| ALIAS-REFERENCE-ITEM(`torch.arcsin`, `torch.asin`) |
| ALIAS-REFERENCE-ITEM(`torch.arcsinh`, `torch.asinh`) |
| ALIAS-REFERENCE-ITEM(`torch.arctan`, `torch.atan`) |
| ALIAS-REFERENCE-ITEM(`torch.arctan2`, `torch.atan2`) |
| ALIAS-REFERENCE-ITEM(`torch.arctanh`, `torch.atanh`) |
| ALIAS-REFERENCE-ITEM(`torch.autograd.function.Function`, `torch.autograd.Function`) |
| ALIAS-REFERENCE-ITEM(`torch.autograd.set_grad_enabled`, `torch.autograd.grad_mode.set_grad_enabled`) |
| ALIAS-REFERENCE-ITEM(`torch.avg_pool1d`, `torch.nn.functional.avg_pool1d`) |
| ALIAS-REFERENCE-ITEM(`torch.bilinear`, `torch.nn.functional.bilinear`) |
| ALIAS-REFERENCE-ITEM(`torch.channel_shuffle`, `torch.nn.functional.channel_shuffle`) |
| ALIAS-REFERENCE-ITEM(`torch.clip`, `torch.clamp`) |
| ALIAS-REFERENCE-ITEM(`torch.concat`, `torch.cat`) |
| ALIAS-REFERENCE-ITEM(`torch.concatenate`, `torch.cat`) |
| ALIAS-REFERENCE-ITEM(`torch.conv1d`, `torch.nn.functional.conv1d`) |
| ALIAS-REFERENCE-ITEM(`torch.conv2d`, `torch.nn.functional.conv2d`) |
| ALIAS-REFERENCE-ITEM(`torch.conv3d`, `torch.nn.functional.conv3d`) |
| ALIAS-REFERENCE-ITEM(`torch.conv_transpose1d`, `torch.nn.functional.conv_transpose1d`) |
| ALIAS-REFERENCE-ITEM(`torch.conv_transpose2d`, `torch.nn.functional.conv_transpose2d`) |
| ALIAS-REFERENCE-ITEM(`torch.conv_transpose3d`, `torch.nn.functional.conv_transpose3d`) |
| ALIAS-REFERENCE-ITEM(`torch.cosine_similarity`, `torch.nn.functional.cosine_similarity`) |
| ALIAS-REFERENCE-ITEM(`torch.cuda.amp.autocast_mode.autocast`, `torch.cuda.amp.autocast`) |
| ALIAS-REFERENCE-ITEM(`torch.digamma`, `torch.special.digamma`) |
| ALIAS-REFERENCE-ITEM(`torch.distributions.AbsTransform`, `torch.distributions.transforms.AbsTransform`) |
| ALIAS-REFERENCE-ITEM(`torch.distributions.AffineTransform`, `torch.distributions.transforms.AffineTransform`) |
| ALIAS-REFERENCE-ITEM(`torch.distributions.Bernoulli`, `torch.distributions.bernoulli.Bernoulli`) |
| ALIAS-REFERENCE-ITEM(`torch.distributions.Beta`, `torch.distributions.beta.Beta`) |
| ALIAS-REFERENCE-ITEM(`torch.distributions.Binomial`, `torch.distributions.binomial.Binomial`) |
| ALIAS-REFERENCE-ITEM(`torch.distributions.Categorical`, `torch.distributions.categorical.Categorical`) |
| ALIAS-REFERENCE-ITEM(`torch.distributions.Cauchy`, `torch.distributions.cauchy.Cauchy`) |
| ALIAS-REFERENCE-ITEM(`torch.distributions.ComposeTransform`, `torch.distributions.transforms.ComposeTransform`) |
| ALIAS-REFERENCE-ITEM(`torch.distributions.ContinuousBernoulli`, `torch.distributions.continuous_bernoulli.ContinuousBernoulli`) |
| ALIAS-REFERENCE-ITEM(`torch.distributions.Dirichlet`, `torch.distributions.dirichlet.Dirichlet`) |
| ALIAS-REFERENCE-ITEM(`torch.distributions.Distribution`, `torch.distributions.distribution.Distribution`) |
| ALIAS-REFERENCE-ITEM(`torch.distributions.ExpTransform`, `torch.distributions.transforms.ExpTransform`) |
| ALIAS-REFERENCE-ITEM(`torch.distributions.Exponential`, `torch.distributions.exponential.Exponential`) |
| ALIAS-REFERENCE-ITEM(`torch.distributions.ExponentialFamily`, `torch.distributions.exp_family.ExponentialFamily`) |
| ALIAS-REFERENCE-ITEM(`torch.distributions.Geometric`, `torch.distributions.geometric.Geometric`) |
| ALIAS-REFERENCE-ITEM(`torch.distributions.Gumbel`, `torch.distributions.gumbel.Gumbel`) |
| ALIAS-REFERENCE-ITEM(`torch.distributions.Independent`, `torch.distributions.independent.Independent`) |
| ALIAS-REFERENCE-ITEM(`torch.distributions.IndependentTransform`, `torch.distributions.transforms.IndependentTransform`) |
| ALIAS-REFERENCE-ITEM(`torch.distributions.Laplace`, `torch.distributions.laplace.Laplace`) |
| ALIAS-REFERENCE-ITEM(`torch.distributions.LogNormal`, `torch.distributions.log_normal.LogNormal`) |
| ALIAS-REFERENCE-ITEM(`torch.distributions.Multinomial`, `torch.distributions.multinomial.Multinomial`) |
| ALIAS-REFERENCE-ITEM(`torch.distributions.MultivariateNormal`, `torch.distributions.multivariate_normal.MultivariateNormal`) |
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
| ALIAS-REFERENCE-ITEM(`torch.divide`, `torch.div`) |
| ALIAS-REFERENCE-ITEM(`torch.erf`, `torch.special.erf`) |
| ALIAS-REFERENCE-ITEM(`torch.erfc`, `torch.special.erfc`) |
| ALIAS-REFERENCE-ITEM(`torch.erfinv`, `torch.special.erfinv`) |
| ALIAS-REFERENCE-ITEM(`torch.exp2`, `torch.special.exp2`) |
| ALIAS-REFERENCE-ITEM(`torch.expm1`, `torch.special.expm1`) |
| ALIAS-REFERENCE-ITEM(`torch.greater`, `torch.gt`) |
| ALIAS-REFERENCE-ITEM(`torch.greater_equal`, `torch.ge`) |
| ALIAS-REFERENCE-ITEM(`torch.group_norm`, `torch.nn.functional.group_norm`) |
| ALIAS-REFERENCE-ITEM(`torch.hardshrink`, `torch.nn.functional.hardshrink`) |
| ALIAS-REFERENCE-ITEM(`torch.i0`, `torch.special.i0`) |
| ALIAS-REFERENCE-ITEM(`torch.igamma`, `torch.special.gammainc`) |
| ALIAS-REFERENCE-ITEM(`torch.igammac`, `torch.special.gammaincc`) |
| ALIAS-REFERENCE-ITEM(`torch.layer_norm`, `torch.nn.functional.layer_norm`) |
| ALIAS-REFERENCE-ITEM(`torch.less`, `torch.lt`) |
| ALIAS-REFERENCE-ITEM(`torch.less_equal`, `torch.le`) |
| ALIAS-REFERENCE-ITEM(`torch.linalg.matmul`, `torch.matmul`) |
| ALIAS-REFERENCE-ITEM(`torch.logit`, `torch.special.logit`) |
| ALIAS-REFERENCE-ITEM(`torch.logsumexp`, `torch.special.logsumexp`) |
| ALIAS-REFERENCE-ITEM(`torch.matrix_exp`, `torch.linalg.matrix_exp`) |
| ALIAS-REFERENCE-ITEM(`torch.matrix_power`, `torch.linalg.matrix_power`) |
| ALIAS-REFERENCE-ITEM(`torch.multiply`, `torch.mul`) |
| ALIAS-REFERENCE-ITEM(`torch.nn.NLLLoss2d`, `torch.nn.NLLLoss`) |
| ALIAS-REFERENCE-ITEM(`torch.nn.Parameter`, `torch.nn.parameter.Parameter`) |
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
| ALIAS-REFERENCE-ITEM(`torch.nn.modules.Linear`, `torch.nn.Linear`) |
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
| ALIAS-REFERENCE-ITEM(`torch.nn.modules.linear.Linear`, `torch.nn.Linear`) |
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
| ALIAS-REFERENCE-ITEM(`torch.nn.utils.parametrizations.spectral_norm`, `torch.nn.utils.spectral_norm`) |
| ALIAS-REFERENCE-ITEM(`torch.nn.utils.spectral_norm.SpectralNorm.apply`, `torch.nn.utils.spectral_norm`) |
| ALIAS-REFERENCE-ITEM(`torch.not_equal`, `torch.ne`) |
| ALIAS-REFERENCE-ITEM(`torch.optim.sgd.SGD`, `torch.optim.SGD`) |
| ALIAS-REFERENCE-ITEM(`torch.orgqr`, `torch.linalg.householder_product`) |
| ALIAS-REFERENCE-ITEM(`torch.pairwise_distance`, `torch.nn.functional.pairwise_distance`) |
| ALIAS-REFERENCE-ITEM(`torch.pdist`, `torch.nn.functional.pdist`) |
| ALIAS-REFERENCE-ITEM(`torch.pixel_shuffle`, `torch.nn.functional.pixel_shuffle`) |
| ALIAS-REFERENCE-ITEM(`torch.pixel_unshuffle`, `torch.nn.functional.pixel_unshuffle`) |
| ALIAS-REFERENCE-ITEM(`torch.polygamma`, `torch.special.polygamma`) |
| ALIAS-REFERENCE-ITEM(`torch.prelu`, `torch.nn.functional.prelu`) |
| ALIAS-REFERENCE-ITEM(`torch.random.get_rng_state`, `torch.get_rng_state`) |
| ALIAS-REFERENCE-ITEM(`torch.random.initial_seed`, `torch.initial_seed`) |
| ALIAS-REFERENCE-ITEM(`torch.random.manual_seed`, `torch.manual_seed`) |
| ALIAS-REFERENCE-ITEM(`torch.random.seed`, `torch.seed`) |
| ALIAS-REFERENCE-ITEM(`torch.random.set_rng_state`, `torch.set_rng_state`) |
| ALIAS-REFERENCE-ITEM(`torch.relu_`, `torch.nn.functional.relu_`) |
| ALIAS-REFERENCE-ITEM(`torch.rrelu_`, `torch.nn.functional.rrelu_`) |
| ALIAS-REFERENCE-ITEM(`torch.set_grad_enabled`, `torch.autograd.grad_mode.set_grad_enabled`) |
| ALIAS-REFERENCE-ITEM(`torch.sigmoid`, `torch.nn.functional.sigmoid`) |
| ALIAS-REFERENCE-ITEM(`torch.sinc`, `torch.special.sinc`) |
| ALIAS-REFERENCE-ITEM(`torch.subtract`, `torch.sub`) |
| ALIAS-REFERENCE-ITEM(`torch.tanh`, `torch.nn.functional.tanh`) |
| ALIAS-REFERENCE-ITEM(`torch.threshold`, `torch.nn.functional.threshold`) |
| ALIAS-REFERENCE-ITEM(`torch.threshold_`, `torch.nn.functional.threshold_`) |
| ALIAS-REFERENCE-ITEM(`torch.torch.Tensor`, `torch.Tensor`) |
| ALIAS-REFERENCE-ITEM(`torch.torch.finfo`, `torch.finfo`) |
| ALIAS-REFERENCE-ITEM(`torch.torch.tril`, `torch.tril`) |
| ALIAS-REFERENCE-ITEM(`torch.trapz`, `torch.trapezoid`) |
| ALIAS-REFERENCE-ITEM(`torch.utils.data.DistributedSampler`, `torch.utils.data.distributed.DistributedSampler`) |
| ALIAS-REFERENCE-ITEM(`torch.utils.data._utils.collate.default_collate`, `torch.utils.data.default_collate`) |
| ALIAS-REFERENCE-ITEM(`torch.utils.data.dataloader.DataLoader`, `torch.utils.data.DataLoader`) |
| ALIAS-REFERENCE-ITEM(`torch.utils.data.dataloader.default_collate`, `torch.utils.data.default_collate`) |
| ALIAS-REFERENCE-ITEM(`torch.utils.data.dataset.ConcatDataset`, `torch.utils.data.ConcatDataset`) |
| ALIAS-REFERENCE-ITEM(`torch.utils.data.dataset.Dataset`, `torch.utils.data.Dataset`) |
| ALIAS-REFERENCE-ITEM(`torch.utils.data.sampler.BatchSampler`, `torch.utils.data.BatchSampler`) |
| ALIAS-REFERENCE-ITEM(`torch.utils.data.sampler.RandomSampler`, `torch.utils.data.RandomSampler`) |
| ALIAS-REFERENCE-ITEM(`torch.utils.data.sampler.Sampler`, `torch.utils.data.Sampler`) |
| ALIAS-REFERENCE-ITEM(`torch.utils.data.sampler.SequentialSampler`, `torch.utils.data.SequentialSampler`) |
| ALIAS-REFERENCE-ITEM(`torch.utils.data.sampler.SubsetRandomSampler`, `torch.utils.data.SubsetRandomSampler`) |
| ALIAS-REFERENCE-ITEM(`torch.utils.data.sampler.WeightedRandomSampler`, `torch.utils.data.WeightedRandomSampler`) |
| ALIAS-REFERENCE-ITEM(`torch.utils.model_zoo.load_url`, `torch.hub.load_state_dict_from_url`) |
| ALIAS-REFERENCE-ITEM(`torch.xlogy`, `torch.special.xlogy`) |
| ALIAS-REFERENCE-ITEM(`torch.cuda.reset_max_memory_reserved`, `torch.cuda.reset_max_memory_cached`) |
| ALIAS-REFERENCE-ITEM(`torch.cuda.reset_peak_memory_stats`, `torch.cuda.reset_max_memory_allocated`) |
| ALIAS-REFERENCE-ITEM(`torch.cdouble`, `torch.complex128`) |
| ALIAS-REFERENCE-ITEM(`torch.cfloat`, `torch.complex64`) |
| ALIAS-REFERENCE-ITEM(`torch.float`, `torch.float32`) |
| ALIAS-REFERENCE-ITEM(`torch.double`, `torch.float64`) |
| ALIAS-REFERENCE-ITEM(`torch.half`, `torch.float16`) |
| ALIAS-REFERENCE-ITEM(`torch.short`, `torch.int16`) |
| ALIAS-REFERENCE-ITEM(`torch.int`, `torch.int32`) |
| ALIAS-REFERENCE-ITEM(`torch.distributions.Distribution.log_prob`, `torch.distributions.distribution.Distribution.log_prob`) |
| ALIAS-REFERENCE-ITEM(`torch.distributions.Distribution.rsample`, `torch.distributions.distribution.Distribution.rsample`) |
| ALIAS-REFERENCE-ITEM(`torch.distributions.Distribution.sample`, `torch.distributions.distribution.Distribution.sample`) |

 ## <span id="id25">功能缺失的 API 列表</span>

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ----- | ----------- | ----------------- | ----------- | ------- |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.rename`, https://pytorch.org/docs/stable/named_tensor.html#torch.Tensor.rename, 实验阶段不稳定 API ，无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.rnn.pad_sequence`, https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_sequence.html#torch-nn-utils-rnn-pad-sequence, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.compile`, https://pytorch.org/docs/stable/generated/torch.compile.html#torch-compile, 有对应相近功能但设计差异大无法映射，一般无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.jit.freeze`, https://pytorch.org/docs/stable/generated/torch.jit.freeze.html#torch-jit-freeze, 有对应相近功能但设计差异大无法映射，一般无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.export.export`, https://pytorch.org/docs/stable/export.html#torch.export.export, 有对应相近功能但设计差异大无法映射，一般无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.dequantize`, https://pytorch.org/docs/stable/generated/torch.Tensor.dequantize.html#torch-tensor-dequantize, 有对应相近功能但设计差异大无法映射，一般无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.xpu.synchronize`, https://pytorch.org/docs/stable/generated/torch.xpu.synchronize.html#torch-xpu-synchronize, 有对应相近功能但设计差异大无法映射，一般无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.vmap`, https://pytorch.org/docs/stable/generated/torch.vmap.html#torch-vmap, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.fx.symbolic_trace`, https://pytorch.org/docs/stable/fx.html#torch.fx.symbolic_trace, 有对应相近功能但设计差异大无法映射，一般无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.jit.annotate`, https://pytorch.org/docs/stable/generated/torch.jit.annotate.html#torch-jit-annotate, 有对应相近功能但设计差异大无法映射，一般无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.quantize_per_tensor`, https://pytorch.org/docs/stable/generated/torch.quantize_per_tensor.html#torch-quantize-per-tensor, 有对应相近功能但设计差异大无法映射，一般无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.to_mkldnn`, https://pytorch.org/docs/stable/generated/torch.Tensor.to_mkldnn.html#torch-tensor-to-mkldnn, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.rnn.pack_padded_sequence`, https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html#torch-nn-utils-rnn-pack-padded-sequence, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.rnn.pad_packed_sequence`, https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_packed_sequence.html#torch-nn-utils-rnn-pad-packed-sequence, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.record_stream`, https://pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html#torch-tensor-record-stream, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.xpu.empty_cache`, https://pytorch.org/docs/stable/generated/torch.xpu.empty_cache.html#torch-xpu-empty-cache, 有对应相近功能但设计差异大无法映射，一般无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.library.impl`, https://pytorch.org/docs/stable/library.html#torch.library.impl, 有对应相近功能但设计差异大无法映射，一般无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.BFloat16Storage`, https://pytorch.org/docs/stable/storage.html#torch.BFloat16Storage, 废弃 API ，无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.BoolStorage`, https://pytorch.org/docs/stable/storage.html#torch.BoolStorage, 废弃 API ，无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.ByteStorage`, https://pytorch.org/docs/stable/storage.html#torch.ByteStorage, 废弃 API ，无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.CharStorage`, https://pytorch.org/docs/stable/storage.html#torch.CharStorage, 废弃 API ，无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.ComplexDoubleStorage`, https://pytorch.org/docs/stable/storage.html#torch.ComplexDoubleStorage, 废弃 API ，无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.ComplexFloatStorage`, https://pytorch.org/docs/stable/storage.html#torch.ComplexFloatStorage, 废弃 API ，无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.reduce_op`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.reduce_op, 废弃 API ，无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.DoubleStorage`, https://pytorch.org/docs/stable/storage.html#torch.DoubleStorage, 废弃 API ，无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.FloatStorage`, https://pytorch.org/docs/stable/storage.html#torch.FloatStorage, 废弃 API ，无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.HalfStorage`, https://pytorch.org/docs/stable/storage.html#torch.HalfStorage, 废弃 API ，无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.IntStorage`, https://pytorch.org/docs/stable/storage.html#torch.IntStorage, 废弃 API ，无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.LongStorage`, https://pytorch.org/docs/stable/storage.html#torch.LongStorage, 废弃 API ，无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.stateless.functional_call`, https://pytorch.org/docs/stable/generated/torch.nn.utils.stateless.functional_call.html#torch-nn-utils-stateless-functional-call, 废弃 API ，无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.QInt32Storage`, https://pytorch.org/docs/stable/storage.html#torch.QInt32Storage, 废弃 API ，无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.QInt8Storage`, https://pytorch.org/docs/stable/storage.html#torch.QInt8Storage, 废弃 API ，无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.QUInt2x4Storage`, https://pytorch.org/docs/stable/storage.html#torch.QUInt2x4Storage, 废弃 API ，无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.QUInt4x2Storage`, https://pytorch.org/docs/stable/storage.html#torch.QUInt4x2Storage, 废弃 API ，无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.QUInt8Storage`, https://pytorch.org/docs/stable/storage.html#torch.QUInt8Storage, 废弃 API ，无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.ShortStorage`, https://pytorch.org/docs/stable/storage.html#torch.ShortStorage, 废弃 API ，无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.storage`, https://pytorch.org/docs/stable/generated/torch.Tensor.storage.html#torch-tensor-storage, 废弃 API ，无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.TypedStorage`, https://pytorch.org/docs/stable/storage.html#torch.TypedStorage, 废弃 API ，无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.use_deterministic_algorithms`, https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch-use-deterministic-algorithms, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.parametrize.register_parametrization`, https://pytorch.org/docs/stable/generated/torch.nn.utils.parametrize.register_parametrization.html#torch-nn-utils-parametrize-register-parametrization, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.package.PackageImporter`, https://pytorch.org/docs/stable/package.html#torch.package.PackageImporter, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.EmbeddingBag`, https://pytorch.org/docs/stable/generated/torch.nn.EmbeddingBag.html#torch.nn.EmbeddingBag, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.fx.GraphModule`, https://pytorch.org/docs/stable/fx.html#torch.fx.GraphModule, 有对应相近功能但设计差异大无法映射，一般无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.share_memory_`, https://pytorch.org/docs/stable/generated/torch.Tensor.share_memory_.html#torch-tensor-share-memory, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.parametrize.remove_parametrizations`, https://pytorch.org/docs/stable/generated/torch.nn.utils.parametrize.remove_parametrizations.html#torch-nn-utils-parametrize-remove-parametrizations, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.is_shared`, https://pytorch.org/docs/stable/generated/torch.Tensor.is_shared.html#torch-tensor-is-shared, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.storage_offset`, https://pytorch.org/docs/stable/generated/torch.Tensor.storage_offset.html#torch-tensor-storage-offset, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.library.Library`, https://pytorch.org/docs/stable/library.html#torch.library.Library, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.futures.Future`, https://pytorch.org/docs/stable/futures.html#torch.futures.Future, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.jit.Attribute`, https://pytorch.org/docs/stable/generated/torch.jit.Attribute.html#torch.jit.Attribute, 有对应相近功能但设计差异大无法映射，一般无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.quantize_per_channel`, https://pytorch.org/docs/stable/generated/torch.quantize_per_channel.html#torch-quantize-per-channel, 有对应相近功能但设计差异大无法映射，一般无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.untyped_storage`, https://pytorch.org/docs/stable/generated/torch.Tensor.untyped_storage.html#torch-tensor-untyped-storage, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.as_subclass`, https://pytorch.org/docs/stable/generated/torch.Tensor.as_subclass.html#torch-tensor-as-subclass, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.q_scale`, https://pytorch.org/docs/stable/generated/torch.Tensor.q_scale.html#torch-tensor-q-scale, 有对应相近功能但设计差异大无法映射，一般无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.set_float32_matmul_precision`, https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch-set-float32-matmul-precision, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.q_zero_point`, https://pytorch.org/docs/stable/generated/torch.Tensor.q_zero_point.html#torch-tensor-q-zero-point, 有对应相近功能但设计差异大无法映射，一般无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.memory_stats`, https://pytorch.org/docs/stable/generated/torch.cuda.memory_stats.html#torch-cuda-memory-stats, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.pipeline.sync.Pipe`, https://pytorch.org/docs/2.3/pipeline.html#torch.distributed.pipeline.sync.Pipe, 废弃 API ，无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.set_rng_state`, https://pytorch.org/docs/stable/generated/torch.cuda.set_rng_state.html#torch-cuda-set-rng-state, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.linalg.tensorinv`, https://pytorch.org/docs/stable/generated/torch.linalg.tensorinv.html#torch-linalg-tensorinv, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.fsdp.FullStateDictConfig`, https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullStateDictConfig, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.mem_get_info`, https://pytorch.org/docs/stable/generated/torch.cuda.mem_get_info.html#torch-cuda-mem-get-info, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.CUDAGraph`, https://pytorch.org/docs/stable/generated/torch.cuda.CUDAGraph.html#torch.cuda.CUDAGraph, 实验阶段不稳定 API ，无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.remove_spectral_norm`, https://pytorch.org/docs/stable/generated/torch.nn.utils.remove_spectral_norm.html#torch-nn-utils-remove-spectral-norm, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.utils.benchmark.Timer`, https://pytorch.org/docs/stable/benchmark_utils.html#torch.utils.benchmark.Timer, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.utils.mobile_optimizer.optimize_for_mobile`, https://pytorch.org/docs/stable/mobile_optimizer.html#torch.utils.mobile_optimizer.optimize_for_mobile, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.fsdp.MixedPrecision`, https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.MixedPrecision, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.rnn.PackedSequence`, https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.PackedSequence.html#torch.nn.utils.rnn.PackedSequence, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.qscheme`, https://pytorch.org/docs/stable/generated/torch.Tensor.qscheme.html#torch-tensor-qscheme, 有对应相近功能但设计差异大无法映射，一般无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.fx.wrap`, https://pytorch.org/docs/stable/fx.html#torch.fx.wrap, 有对应相近功能但设计差异大无法映射，一般无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.set_detect_anomaly`, https://pytorch.org/docs/stable/autograd.html#torch.autograd.set_detect_anomaly, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.empty_strided`, https://pytorch.org/docs/stable/generated/torch.empty_strided.html#torch-empty-strided, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.fx.Graph`, https://pytorch.org/docs/stable/fx.html#torch.fx.Graph, 有对应相近功能但设计差异大无法映射，一般无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.futures.wait_all`, https://pytorch.org/docs/stable/futures.html#torch.futures.wait_all, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.prune.l1_unstructured`, https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.l1_unstructured.html#torch-nn-utils-prune-l1-unstructured, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.ipc_collect`, https://pytorch.org/docs/stable/generated/torch.cuda.ipc_collect.html#torch-cuda-ipc-collect, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.optim.ZeroRedundancyOptimizer`, https://pytorch.org/docs/stable/distributed.optim.html#torch.distributed.optim.ZeroRedundancyOptimizer, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.mps.profiler.start`, https://pytorch.org/docs/stable/generated/torch.mps.profiler.start.html#torch-mps-profiler-start, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.fx.Proxy`, https://pytorch.org/docs/stable/fx.html#torch.fx.Proxy, 有对应相近功能但设计差异大无法映射，一般无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.mps.profiler.stop`, https://pytorch.org/docs/stable/generated/torch.mps.profiler.stop.html#torch-mps-profiler-stop, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.refine_names`, https://pytorch.org/docs/stable/named_tensor.html#torch.Tensor.refine_names, 实验阶段不稳定 API ，无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.init`, https://pytorch.org/docs/stable/generated/torch.cuda.init.html#torch-cuda-init, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.rpc.TensorPipeRpcBackendOptions`, https://pytorch.org/docs/stable/rpc.html#torch.distributed.rpc.TensorPipeRpcBackendOptions, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.default_stream`, https://pytorch.org/docs/stable/generated/torch.cuda.default_stream.html#torch-cuda-default-stream, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.resolve_conj`, https://pytorch.org/docs/stable/generated/torch.Tensor.resolve_conj.html#torch-tensor-resolve-conj, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.mps.synchronize`, https://pytorch.org/docs/stable/generated/torch.mps.synchronize.html#torch-mps-synchronize, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.skip_init`, https://pytorch.org/docs/stable/generated/torch.nn.utils.skip_init.html#torch-nn-utils-skip-init, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.row_indices`, https://pytorch.org/docs/stable/generated/torch.Tensor.row_indices.html#torch-tensor-row-indices, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.jit.trace_module`, https://pytorch.org/docs/stable/generated/torch.jit.trace_module.html#torch-jit-trace-module, 有对应相近功能但设计差异大无法映射，一般无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.fsdp.CPUOffload`, https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.CPUOffload, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.quasirandom.SobolEngine`, https://pytorch.org/docs/stable/generated/torch.quasirandom.SobolEngine.html#torch.quasirandom.SobolEngine, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.names`, https://pytorch.org/docs/stable/named_tensor.html#torch.Tensor.names, 实验阶段不稳定 API ，无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.q_per_channel_zero_points`, https://pytorch.org/docs/stable/generated/torch.Tensor.q_per_channel_zero_points.html#torch-tensor-q-per-channel-zero-points, 有对应相近功能但设计差异大无法映射，一般无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.jit.export`, https://pytorch.org/docs/stable/jit.html#torch.jit.export, 有对应相近功能但设计差异大无法映射，一般无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.prune.remove`, https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.remove.html#torch-nn-utils-prune-remove, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.rnn.pack_sequence`, https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_sequence.html#torch-nn-utils-rnn-pack-sequence, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.ccol_indices`, https://pytorch.org/docs/stable/generated/torch.Tensor.ccol_indices.html#torch-tensor-ccol-indices, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.is_set_to`, https://pytorch.org/docs/stable/generated/torch.Tensor.is_set_to.html#torch-tensor-is-set-to, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.put_`, https://pytorch.org/docs/stable/generated/torch.Tensor.put_.html#torch-tensor-put, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.q_per_channel_axis`, https://pytorch.org/docs/stable/generated/torch.Tensor.q_per_channel_axis.html#torch-tensor-q-per-channel-axis, 有对应相近功能但设计差异大无法映射，一般无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.q_per_channel_scales`, https://pytorch.org/docs/stable/generated/torch.Tensor.q_per_channel_scales.html#torch-tensor-q-per-channel-scales, 有对应相近功能但设计差异大无法映射，一般无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.sign_`, https://pytorch.org/docs/stable/generated/torch.Tensor.sign_.html#torch-tensor-sign, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.hub.get_dir`, https://pytorch.org/docs/stable/hub.html#torch.hub.get_dir, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.hub.set_dir`, https://pytorch.org/docs/stable/hub.html#torch.hub.set_dir, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.is_conj`, https://pytorch.org/docs/stable/generated/torch.Tensor.is_conj.html#torch-tensor-is-conj, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.result_type`, https://pytorch.org/docs/stable/generated/torch.result_type.html#torch-result-type, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.comm.broadcast_coalesced`, https://pytorch.org/docs/stable/generated/torch.cuda.comm.broadcast_coalesced.html#torch-cuda-comm-broadcast-coalesced, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.optim.SparseAdam`, https://pytorch.org/docs/stable/generated/torch.optim.SparseAdam.html#torch.optim.SparseAdam, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.fake_quantize_per_channel_affine`, https://pytorch.org/docs/stable/generated/torch.fake_quantize_per_channel_affine.html#torch-fake-quantize-per-channel-affine, 有对应相近功能但设计差异大无法映射，一般无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.fake_quantize_per_tensor_affine`, https://pytorch.org/docs/stable/generated/torch.fake_quantize_per_tensor_affine.html#torch-fake-quantize-per-tensor-affine, 有对应相近功能但设计差异大无法映射，一般无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.to_sparse_csc`, https://pytorch.org/docs/stable/generated/torch.Tensor.to_sparse_csc.html#torch-tensor-to-sparse-csc, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.mps.empty_cache`, https://pytorch.org/docs/stable/generated/torch.mps.empty_cache.html#torch-mps-empty-cache, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.profiler.record_function`, https://pytorch.org/docs/stable/generated/torch.autograd.profiler.record_function.html#torch.autograd.profiler.record_function, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.index_copy`, https://pytorch.org/docs/stable/generated/torch.Tensor.index_copy.html#torch-tensor-index-copy, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.utils.cpp_extension.load_inline`, https://pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.load_inline, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.jit.set_fusion_strategy`, https://pytorch.org/docs/stable/generated/torch.jit.set_fusion_strategy.html#torch-jit-set-fusion-strategy, 有对应相近功能但设计差异大无法映射，一般无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.TCPStore`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.TCPStore, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.optim.lr_scheduler.SequentialLR`, https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.SequentialLR.html#torch.optim.lr_scheduler.SequentialLR, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.sparse.sampled_addmm`, https://pytorch.org/docs/stable/generated/torch.sparse.sampled_addmm.html#torch-sparse-sampled-addmm, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.nested.nested_tensor`, https://pytorch.org/docs/stable/nested.html#torch.nested.nested_tensor, 实验阶段不稳定 API ，无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.align_to`, https://pytorch.org/docs/stable/named_tensor.html#torch.Tensor.align_to, 实验阶段不稳定 API ，无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.promote_types`, https://pytorch.org/docs/stable/generated/torch.promote_types.html#torch-promote-types, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.tensor.parallel.ColwiseParallel`, https://pytorch.org/docs/stable/distributed.tensor.parallel.html#torch.distributed.tensor.parallel.ColwiseParallel, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.to_sparse_bsr`, https://pytorch.org/docs/stable/generated/torch.Tensor.to_sparse_bsr.html#torch-tensor-to-sparse-bsr, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.xpu.device_count`, https://pytorch.org/docs/stable/generated/torch.xpu.device_count.html#torch-xpu-device-count, 有对应相近功能但设计差异大无法映射，一般无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.fx.Node`, https://pytorch.org/docs/stable/fx.html#torch.fx.Node, 有对应相近功能但设计差异大无法映射，一般无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.jit.fork`, https://pytorch.org/docs/stable/generated/torch.jit.fork.html#torch-jit-fork, 有对应相近功能但设计差异大无法映射，一般无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.library.impl_abstract`, https://pytorch.org/docs/stable/library.html#torch.library.impl_abstract, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.linalg.tensorsolve`, https://pytorch.org/docs/stable/generated/torch.linalg.tensorsolve.html#torch-linalg-tensorsolve, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.functional.embedding_bag`, https://pytorch.org/docs/stable/generated/torch.nn.functional.embedding_bag.html#torch-nn-functional-embedding-bag, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.map_`, https://pytorch.org/docs/stable/generated/torch.Tensor.map_.html#torch-tensor-map, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.rename_`, https://pytorch.org/docs/stable/named_tensor.html#torch.Tensor.rename_, 实验阶段不稳定 API ，无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.scatter_reduce_`, https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_reduce_.html#torch-tensor-scatter-reduce, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.set_flush_denormal`, https://pytorch.org/docs/stable/generated/torch.set_flush_denormal.html#torch-set-flush-denormal, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.kaiser_window`, https://pytorch.org/docs/stable/generated/torch.kaiser_window.html#torch-kaiser-window, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.device_mesh.init_device_mesh`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.device_mesh.init_device_mesh, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.fsdp.FullyShardedDataParallel`, https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.tensor.parallel.parallelize_module`, https://pytorch.org/docs/stable/distributed.tensor.parallel.html#torch.distributed.tensor.parallel.parallelize_module, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.tensor.parallel.RowwiseParallel`, https://pytorch.org/docs/stable/distributed.tensor.parallel.html#torch.distributed.tensor.parallel.RowwiseParallel, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.are_deterministic_algorithms_enabled`, https://pytorch.org/docs/stable/generated/torch.are_deterministic_algorithms_enabled.html#torch-are-deterministic-algorithms-enabled, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.is_deterministic_algorithms_warn_only_enabled`, https://pytorch.org/docs/stable/generated/torch.is_deterministic_algorithms_warn_only_enabled.html#torch-is-deterministic-algorithms-warn-only-enabled, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.backends.mps.is_available`, https://pytorch.org/docs/stable/backends.html#torch.backends.mps.is_available, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.fx.Tracer`, https://pytorch.org/docs/stable/fx.html#torch.fx.Tracer, 有对应相近功能但设计差异大无法映射，一般无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.jit.enable_onednn_fusion`, https://pytorch.org/docs/stable/generated/torch.jit.enable_onednn_fusion.html#torch-jit-enable-onednn-fusion, 有对应相近功能但设计差异大无法映射，一般无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.comm.reduce_add`, https://pytorch.org/docs/stable/generated/torch.cuda.comm.reduce_add.html#torch-cuda-comm-reduce-add, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.checkpoint.state_dict.get_optimizer_state_dict`, https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.get_optimizer_state_dict, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.parametrizations.orthogonal`, https://pytorch.org/docs/stable/generated/torch.nn.utils.parametrizations.orthogonal.html#torch-nn-utils-parametrizations-orthogonal, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.prune.L1Unstructured`, https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.L1Unstructured.html#torch.nn.utils.prune.L1Unstructured, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.prune.random_unstructured`, https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.random_unstructured.html#torch-nn-utils-prune-random-unstructured, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.special.zeta`, https://pytorch.org/docs/stable/special.html#torch.special.zeta, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.xpu.current_device`, https://pytorch.org/docs/stable/generated/torch.xpu.current_device.html#torch-xpu-current-device, 有对应相近功能但设计差异大无法映射，一般无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.xpu.get_device_properties`, https://pytorch.org/docs/stable/generated/torch.xpu.get_device_properties.html#torch-xpu-get-device-properties, 有对应相近功能但设计差异大无法映射，一般无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.gradient`, https://pytorch.org/docs/stable/generated/torch.gradient.html#torch-gradient, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.sparse_resize_`, https://pytorch.org/docs/stable/generated/torch.Tensor.sparse_resize_.html#torch-tensor-sparse-resize, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.profiler.profile`, https://pytorch.org/docs/stable/autograd.html#torch.autograd.profiler.profile, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.backends.cuda.enable_math_sdp`, https://pytorch.org/docs/stable/backends.html#torch.backends.cuda.enable_math_sdp, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.backends.cuda.enable_mem_efficient_sdp`, https://pytorch.org/docs/stable/backends.html#torch.backends.cuda.enable_mem_efficient_sdp, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.jit.ScriptModule`, https://pytorch.org/docs/stable/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule, 有对应相近功能但设计差异大无法映射，一般无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.ExternalStream`, https://pytorch.org/docs/stable/generated/torch.cuda.ExternalStream.html#torch.cuda.ExternalStream, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.memory._record_memory_history`, https://pytorch.org/docs/stable/torch_cuda_memory.html#torch.cuda.memory._record_memory_history, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.memory_summary`, https://pytorch.org/docs/stable/generated/torch.cuda.memory_summary.html#torch-cuda-memory-summary, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.checkpoint.state_dict.get_model_state_dict`, https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.get_model_state_dict, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.checkpoint.state_dict.StateDictOptions`, https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.StateDictOptions, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.optim.lr_scheduler.ChainedScheduler`, https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ChainedScheduler.html#torch.optim.lr_scheduler.ChainedScheduler, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.futures.collect_all`, https://pytorch.org/docs/stable/futures.html#torch.futures.collect_all, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.sparse_compressed_tensor`, https://pytorch.org/docs/stable/generated/torch.sparse_compressed_tensor.html#torch-sparse-compressed-tensor, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.mps.current_allocated_memory`, https://pytorch.org/docs/stable/generated/torch.mps.current_allocated_memory.html#torch-mps-current-allocated-memory, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.profiler.tensorboard_trace_handler`, https://pytorch.org/docs/stable/profiler.html#torch.profiler.tensorboard_trace_handler, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.xpu.is_available`, https://pytorch.org/docs/stable/generated/torch.xpu.is_available.html#torch-xpu-is-available, 有对应相近功能但设计差异大无法映射，一般无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.xpu.set_device`, https://pytorch.org/docs/stable/generated/torch.xpu.set_device.html#torch-xpu-set-device, 有对应相近功能但设计差异大无法映射，一般无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.rpc.WorkerInfo`, https://pytorch.org/docs/stable/rpc.html#torch.distributed.rpc.WorkerInfo, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.bartlett_window`, https://pytorch.org/docs/stable/generated/torch.bartlett_window.html#torch-bartlett-window, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.signal.windows.kaiser`, https://pytorch.org/docs/stable/generated/torch.signal.windows.kaiser.html#torch-signal-windows-kaiser, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.graph_pool_handle`, https://pytorch.org/docs/stable/generated/torch.cuda.graph_pool_handle.html#torch-cuda-graph-pool-handle, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.library.define`, https://pytorch.org/docs/stable/library.html#torch.library.define, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.monitor.log_event`, https://pytorch.org/docs/stable/monitor.html#torch.monitor.log_event, 实验阶段不稳定 API ，无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.init.sparse_`, https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.sparse_, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.modules.module.register_module_backward_hook`, https://pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_backward_hook.html#torch-nn-modules-module-register-module-backward-hook, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.prune.global_unstructured`, https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.global_unstructured.html#torch-nn-utils-prune-global-unstructured, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.prune.ln_structured`, https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.ln_structured.html#torch-nn-utils-prune-ln-structured, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.special.log_ndtr`, https://pytorch.org/docs/stable/special.html#torch.special.log_ndtr, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.align_as`, https://pytorch.org/docs/stable/named_tensor.html#torch.Tensor.align_as, 实验阶段不稳定 API ，无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.xpu.get_device_name`, https://pytorch.org/docs/stable/generated/torch.xpu.get_device_name.html#torch-xpu-get-device-name, 有对应相近功能但设计差异大无法映射，一般无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.xpu.manual_seed`, https://pytorch.org/docs/stable/generated/torch.xpu.manual_seed.html#torch-xpu-manual-seed, 有对应相近功能但设计差异大无法映射，一般无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.distributions.transforms.CatTransform`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.transforms.CatTransform, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.set_warn_always`, https://pytorch.org/docs/stable/generated/torch.set_warn_always.html#torch-set-warn-always, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.backends.cuda.enable_flash_sdp`, https://pytorch.org/docs/stable/backends.html#torch.backends.cuda.enable_flash_sdp, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.backends.cuda.preferred_linalg_library`, https://pytorch.org/docs/stable/backends.html#torch.backends.cuda.preferred_linalg_library, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.UntypedStorage`, https://pytorch.org/docs/stable/storage.html#torch.UntypedStorage, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.fx.Interpreter`, https://pytorch.org/docs/stable/fx.html#module-torch.fx.interpreter, 有对应相近功能但设计差异大无法映射，一般无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.jit.optimize_for_inference`, https://pytorch.org/docs/stable/generated/torch.jit.optimize_for_inference.html#torch-jit-optimize-for-inference, 有对应相近功能但设计差异大无法映射，一般无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.jit.wait`, https://pytorch.org/docs/stable/generated/torch.jit.wait.html#torch-jit-wait, 有对应相近功能但设计差异大无法映射，一般无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.autograd.backward`, https://pytorch.org/docs/stable/rpc.html#torch.distributed.autograd.backward, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributions.transforms.LowerCholeskyTransform`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.transforms.LowerCholeskyTransform, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.overrides.resolve_name`, https://pytorch.org/docs/stable/torch.overrides.html#torch.overrides.resolve_name, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.sparse.log_softmax`, https://pytorch.org/docs/stable/generated/torch.sparse.log_softmax.html#torch-sparse-log-softmax, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.monitor.register_event_handler`, https://pytorch.org/docs/stable/monitor.html#torch.monitor.register_event_handler, 实验阶段不稳定 API ，无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.monitor.Stat`, https://pytorch.org/docs/stable/monitor.html#torch.monitor.Stat, 实验阶段不稳定 API ，无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.monitor.unregister_event_handler`, https://pytorch.org/docs/stable/monitor.html#torch.monitor.unregister_event_handler, 实验阶段不稳定 API ，无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.register_post_accumulate_grad_hook`, https://pytorch.org/docs/stable/generated/torch.Tensor.register_post_accumulate_grad_hook.html#torch-tensor-register-post-accumulate-grad-hook, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.sspaddmm`, https://pytorch.org/docs/stable/generated/torch.Tensor.sspaddmm.html#torch-tensor-sspaddmm, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.sum_to_size`, https://pytorch.org/docs/stable/generated/torch.Tensor.sum_to_size.html#torch-tensor-sum-to-size, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.__config__.parallel_info`, https://pytorch.org/docs/stable/config_mod.html#torch.__config__.parallel_info, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.amp.custom_bwd`, https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.custom_bwd, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.amp.custom_fwd`, https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.custom_fwd, 废弃 API ，无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.__config__.show`, https://pytorch.org/docs/stable/config_mod.html#torch.__config__.show, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.from_file`, https://pytorch.org/docs/stable/generated/torch.from_file.html#torch-from-file, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.__future__.set_overwrite_module_params_on_conversion`, https://pytorch.org/docs/stable/future_mod.html#torch.__future__.set_overwrite_module_params_on_conversion, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.gradcheck.gradcheck`, https://pytorch.org/docs/stable/generated/torch.autograd.gradcheck.gradcheck.html#torch-autograd-gradcheck-gradcheck, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.backends.cuda.sdp_kernel`, https://pytorch.org/docs/stable/backends.html#torch.backends.cuda.sdp_kernel, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.backends.mkl.is_available`, https://pytorch.org/docs/stable/backends.html#torch.backends.mkl.is_available, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.signal.windows.bartlett`, https://pytorch.org/docs/stable/generated/torch.signal.windows.bartlett.html#torch-signal-windows-bartlett, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.storage_type`, https://pytorch.org/docs/stable/generated/torch.Tensor.storage_type.html#torch-tensor-storage-type, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.can_device_access_peer`, https://pytorch.org/docs/stable/generated/torch.cuda.can_device_access_peer.html#torch-cuda-can-device-access-peer, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.jiterator._create_jit_fn`, https://pytorch.org/docs/stable/generated/torch.cuda.jiterator._create_jit_fn.html#torch-cuda-jiterator-create-jit-fn, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.set_sync_debug_mode`, https://pytorch.org/docs/stable/generated/torch.cuda.set_sync_debug_mode.html#torch-cuda-set-sync-debug-mode, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.fsdp.FullOptimStateDictConfig`, https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullOptimStateDictConfig, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributions.transforms.CorrCholeskyTransform`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.transforms.CorrCholeskyTransform, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.overrides.get_testing_overrides`, https://pytorch.org/docs/stable/torch.overrides.html#torch.overrides.get_testing_overrides, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.sparse_bsr_tensor`, https://pytorch.org/docs/stable/generated/torch.sparse_bsr_tensor.html#torch-sparse-bsr-tensor, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.sparse_csc_tensor`, https://pytorch.org/docs/stable/generated/torch.sparse_csc_tensor.html#torch-sparse-csc-tensor, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.to_sparse_bsc`, https://pytorch.org/docs/stable/generated/torch.Tensor.to_sparse_bsc.html#torch-tensor-to-sparse-bsc, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.linalg.ldl_factor_ex`, https://pytorch.org/docs/stable/generated/torch.linalg.ldl_factor_ex.html#torch-linalg-ldl-factor-ex, 实验阶段不稳定 API ，无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.monitor.Event`, https://pytorch.org/docs/stable/monitor.html#torch.monitor.Event, 实验阶段不稳定 API ，无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.rnn.unpad_sequence`, https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.unpad_sequence.html#torch-nn-utils-rnn-unpad-sequence, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.xpu.set_rng_state`, https://pytorch.org/docs/stable/generated/torch.xpu.set_rng_state.html#torch-xpu-set-rng-state, 有对应相近功能但设计差异大无法映射，一般无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.max_memory_cached`, https://pytorch.org/docs/stable/generated/torch.cuda.max_memory_cached.html#torch-cuda-max-memory-cached, 废弃 API ，无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.get_arch_list`, https://pytorch.org/docs/stable/generated/torch.cuda.get_arch_list.html#torch-cuda-get-arch-list, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.resolve_neg`, https://pytorch.org/docs/stable/generated/torch.Tensor.resolve_neg.html#torch-tensor-resolve-neg, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.compiled_with_cxx11_abi`, https://pytorch.org/docs/stable/generated/torch.compiled_with_cxx11_abi.html#torch-compiled-with-cxx11-abi, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.memory_cached`, https://pytorch.org/docs/stable/generated/torch.cuda.memory_cached.html#torch-cuda-memory-cached, 废弃 API ，无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.is_warn_always_enabled`, https://pytorch.org/docs/stable/generated/torch.is_warn_always_enabled.html#torch-is-warn-always-enabled, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.detect_anomaly`, https://pytorch.org/docs/stable/autograd.html#torch.autograd.detect_anomaly, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.forward_ad.make_dual`, https://pytorch.org/docs/stable/generated/torch.autograd.forward_ad.make_dual.html#torch-autograd-forward-ad-make-dual, 实验阶段不稳定 API ，无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.nvtx.mark`, https://pytorch.org/docs/stable/generated/torch.cuda.nvtx.mark.html#torch-cuda-nvtx-mark, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.forward_ad.unpack_dual`, https://pytorch.org/docs/stable/generated/torch.autograd.forward_ad.unpack_dual.html#torch-autograd-forward-ad-unpack-dual, 实验阶段不稳定 API ，无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.graph.save_on_cpu`, https://pytorch.org/docs/stable/autograd.html#torch.autograd.graph.save_on_cpu, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.profiler.load_nvprof`, https://pytorch.org/docs/stable/generated/torch.autograd.profiler.load_nvprof.html#torch-autograd-profiler-load-nvprof, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.profiler.profile.key_averages`, https://pytorch.org/docs/stable/generated/torch.autograd.profiler.profile.key_averages.html#torch-autograd-profiler-profile-key-averages, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.profiler_util.MemRecordsAcc`, https://pytorch.org/docs/stable/generated/torch.autograd.profiler_util.MemRecordsAcc.html#torch.autograd.profiler_util.MemRecordsAcc, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.backends.mps.is_built`, https://pytorch.org/docs/stable/backends.html#torch.backends.mps.is_built, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.backends.nnpack.set_flags`, https://pytorch.org/docs/stable/backends.html#torch.backends.nnpack.set_flags, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.export.ExportedProgram`, https://pytorch.org/docs/stable/export.html#torch.export.ExportedProgram, 有对应相近功能但设计差异大无法映射，一般无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.export.graph_signature.InputSpec`, https://pytorch.org/docs/stable/export.html#torch.export.graph_signature.InputSpec, 有对应相近功能但设计差异大无法映射，一般无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.export.load`, https://pytorch.org/docs/stable/export.html#torch.export.load, 有对应相近功能但设计差异大无法映射，一般无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.fx.replace_pattern`, https://pytorch.org/docs/stable/fx.html#torch.fx.replace_pattern, 有对应相近功能但设计差异大无法映射，一般无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.fx.Transformer`, https://pytorch.org/docs/stable/fx.html#torch.fx.Transformer, 有对应相近功能但设计差异大无法映射，一般无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.jit.isinstance`, https://pytorch.org/docs/stable/generated/torch.jit.isinstance.html#torch-jit-isinstance, 有对应相近功能但设计差异大无法映射，一般无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.jit.script_if_tracing`, https://pytorch.org/docs/stable/generated/torch.jit.script_if_tracing.html#torch-jit-script-if-tracing, 有对应相近功能但设计差异大无法映射，一般无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.caching_allocator_alloc`, https://pytorch.org/docs/stable/generated/torch.cuda.caching_allocator_alloc.html#torch-cuda-caching-allocator-alloc, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.caching_allocator_delete`, https://pytorch.org/docs/stable/generated/torch.cuda.caching_allocator_delete.html#torch-cuda-caching-allocator-delete, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.get_allocator_backend`, https://pytorch.org/docs/stable/generated/torch.cuda.get_allocator_backend.html#torch-cuda-get-allocator-backend, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.get_sync_debug_mode`, https://pytorch.org/docs/stable/generated/torch.cuda.get_sync_debug_mode.html#torch-cuda-get-sync-debug-mode, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.list_gpu_processes`, https://pytorch.org/docs/stable/generated/torch.cuda.list_gpu_processes.html#torch-cuda-list-gpu-processes, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.memory_snapshot`, https://pytorch.org/docs/stable/generated/torch.cuda.memory_snapshot.html#torch-cuda-memory-snapshot, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.seed`, https://pytorch.org/docs/stable/generated/torch.cuda.seed.html#torch-cuda-seed, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.seed_all`, https://pytorch.org/docs/stable/generated/torch.cuda.seed_all.html#torch-cuda-seed-all, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.utilization`, https://pytorch.org/docs/stable/generated/torch.cuda.utilization.html#torch-cuda-utilization, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.PowerSGDState`, https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.PowerSGDState, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.checkpoint.planner.WriteItem`, https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.planner.WriteItem, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.checkpoint.state_dict.set_model_state_dict`, https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.set_model_state_dict, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.checkpoint.state_dict.set_optimizer_state_dict`, https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.set_optimizer_state_dict, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.FileStore`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.FileStore, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.PrefixStore`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.PrefixStore, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.fsdp.LocalStateDictConfig`, https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.LocalStateDictConfig, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.optim.lr_scheduler.PolynomialLR`, https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.PolynomialLR.html#torch.optim.lr_scheduler.PolynomialLR, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributions.relaxed_bernoulli.RelaxedBernoulli`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.relaxed_bernoulli.RelaxedBernoulli, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.overrides.get_overridable_functions`, https://pytorch.org/docs/stable/torch.overrides.html#torch.overrides.get_overridable_functions, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.overrides.has_torch_function`, https://pytorch.org/docs/stable/torch.overrides.html#torch.overrides.has_torch_function, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.overrides.is_tensor_like`, https://pytorch.org/docs/stable/torch.overrides.html#torch.overrides.is_tensor_like, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.overrides.wrap_torch_function`, https://pytorch.org/docs/stable/torch.overrides.html#torch.overrides.wrap_torch_function, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.sparse_bsc_tensor`, https://pytorch.org/docs/stable/generated/torch.sparse_bsc_tensor.html#torch-sparse-bsc-tensor, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.library.get_ctx`, https://pytorch.org/docs/stable/library.html#torch.library.get_ctx, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.linalg.ldl_factor`, https://pytorch.org/docs/stable/generated/torch.linalg.ldl_factor.html#torch-linalg-ldl-factor, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.linalg.ldl_solve`, https://pytorch.org/docs/stable/generated/torch.linalg.ldl_solve.html#torch-linalg-ldl-solve, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.lobpcg`, https://pytorch.org/docs/stable/generated/torch.lobpcg.html#torch-lobpcg, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.mps.manual_seed`, https://pytorch.org/docs/stable/generated/torch.mps.manual_seed.html#torch-mps-manual-seed, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.prune.identity`, https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.identity.html#torch-nn-utils-prune-identity, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.prune.PruningContainer`, https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.PruningContainer.html#torch.nn.utils.prune.PruningContainer, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.prune.random_structured`, https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.random_structured.html#torch-nn-utils-prune-random-structured, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.prune.RandomStructured`, https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.RandomStructured.html#torch.nn.utils.prune.RandomStructured, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.atan2_`, https://pytorch.org/docs/stable/generated/torch.Tensor.atan2_.html#torch-tensor-atan2, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.chalf`, https://pytorch.org/docs/stable/generated/torch.Tensor.chalf.html#torch-tensor-chalf, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.index_reduce`, https://pytorch.org/docs/stable/generated/torch.Tensor.index_reduce.html#torch-tensor-index-reduce, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.index_reduce_`, https://pytorch.org/docs/stable/generated/torch.Tensor.index_reduce_.html#torch-tensor-index-reduce, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.sgn_`, https://pytorch.org/docs/stable/generated/torch.Tensor.sgn_.html#torch-tensor-sgn, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.utils.cpp_extension.verify_ninja_availability`, https://pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.verify_ninja_availability, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.utils.data._utils.collate.collate`, https://pytorch.org/docs/stable/data.html#torch.utils.data._utils.collate.collate, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.utils.data.StackDataset`, https://pytorch.org/docs/stable/data.html#torch.utils.data.StackDataset, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.utils.swap_tensors`, https://pytorch.org/docs/stable/generated/torch.utils.swap_tensors.html#torch-utils-swap-tensors, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.xpu.get_rng_state`, https://pytorch.org/docs/stable/generated/torch.xpu.get_rng_state.html#torch-xpu-get-rng-state, 有对应相近功能但设计差异大无法映射，一般无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.xpu.get_rng_state_all`, https://pytorch.org/docs/stable/generated/torch.xpu.get_rng_state_all.html#torch-xpu-get-rng-state-all, 有对应相近功能但设计差异大无法映射，一般无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.xpu.manual_seed_all`, https://pytorch.org/docs/stable/generated/torch.xpu.manual_seed_all.html#torch-xpu-manual-seed-all, 有对应相近功能但设计差异大无法映射，一般无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.xpu.set_rng_state_all`, https://pytorch.org/docs/stable/generated/torch.xpu.set_rng_state_all.html#torch-xpu-set-rng-state-all, 有对应相近功能但设计差异大无法映射，一般无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.utils.cpp_extension.include_paths`, https://pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.include_paths, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.special.entr`, https://pytorch.org/docs/stable/special.html#torch.special.entr, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributions.transforms.CumulativeDistributionTransform`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.transforms.CumulativeDistributionTransform, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributions.transforms.SoftplusTransform`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.transforms.SoftplusTransform, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch._logging.set_logs`, https://pytorch.org/docs/stable/generated/torch._logging.set_logs.html#torch-logging-set-logs, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.cond`, https://pytorch.org/docs/stable/generated/torch.cond.html#torch-cond, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.get_float32_matmul_precision`, https://pytorch.org/docs/stable/generated/torch.get_float32_matmul_precision.html#torch-get-float32-matmul-precision, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.index_reduce`, https://pytorch.org/docs/stable/generated/torch.index_reduce.html#torch-index-reduce, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.is_inference_mode_enabled`, https://pytorch.org/docs/stable/generated/torch.is_inference_mode_enabled.html#torch-is-inference-mode-enabled, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.is_storage`, https://pytorch.org/docs/stable/generated/torch.is_storage.html#torch-is-storage, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.random.fork_rng`, https://pytorch.org/docs/stable/random.html#torch.random.fork_rng, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.Tag`, https://pytorch.org/docs/stable/torch.html#torch.Tag, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.unravel_index`, https://pytorch.org/docs/stable/generated/torch.unravel_index.html#torch-unravel-index, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.__future__.get_overwrite_module_params_on_conversion`, https://pytorch.org/docs/stable/future_mod.html#torch.__future__.get_overwrite_module_params_on_conversion, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.__future__.get_swap_module_params_on_conversion`, https://pytorch.org/docs/stable/future_mod.html#torch.__future__.get_swap_module_params_on_conversion, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.__future__.set_swap_module_params_on_conversion`, https://pytorch.org/docs/stable/future_mod.html#torch.__future__.set_swap_module_params_on_conversion, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.forward_ad.dual_level`, https://pytorch.org/docs/stable/generated/torch.autograd.forward_ad.dual_level.html#torch.autograd.forward_ad.dual_level, 实验阶段不稳定 API ，无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.forward_ad.enter_dual_level`, https://pytorch.org/docs/stable/generated/torch.autograd.forward_ad.enter_dual_level.html#torch-autograd-forward-ad-enter-dual-level, 实验阶段不稳定 API ，无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.forward_ad.exit_dual_level`, https://pytorch.org/docs/stable/generated/torch.autograd.forward_ad.exit_dual_level.html#torch-autograd-forward-ad-exit-dual-level, 实验阶段不稳定 API ，无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.forward_ad.UnpackedDualTensor`, https://pytorch.org/docs/stable/generated/torch.autograd.forward_ad.UnpackedDualTensor.html#torch.autograd.forward_ad.UnpackedDualTensor, 实验阶段不稳定 API ，无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.function.BackwardCFunction`, https://pytorch.org/docs/stable/generated/torch.autograd.function.BackwardCFunction.html#torch.autograd.function.BackwardCFunction, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.function.InplaceFunction`, https://pytorch.org/docs/stable/generated/torch.autograd.function.InplaceFunction.html#torch.autograd.function.InplaceFunction, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.function.NestedIOFunction`, https://pytorch.org/docs/stable/generated/torch.autograd.function.NestedIOFunction.html#torch.autograd.function.NestedIOFunction, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.function.once_differentiable`, https://pytorch.org/docs/stable/generated/torch.autograd.function.once_differentiable.html#torch-autograd-function-once-differentiable, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.Function.vmap`, https://pytorch.org/docs/stable/generated/torch.autograd.Function.vmap.html#torch-autograd-function-vmap, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.functional.hvp`, https://pytorch.org/docs/stable/generated/torch.autograd.functional.hvp.html#torch-autograd-functional-hvp, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.functional.vhp`, https://pytorch.org/docs/stable/generated/torch.autograd.functional.vhp.html#torch-autograd-functional-vhp, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.grad_mode.inference_mode`, https://pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html#torch.autograd.grad_mode.inference_mode, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.grad_mode.set_multithreading_enabled`, https://pytorch.org/docs/stable/generated/torch.autograd.grad_mode.set_multithreading_enabled.html#torch.autograd.grad_mode.set_multithreading_enabled, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.gradcheck.GradcheckError`, https://pytorch.org/docs/stable/generated/torch.autograd.gradcheck.GradcheckError.html#torch-autograd-gradcheck-gradcheckerror, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.gradcheck.gradgradcheck`, https://pytorch.org/docs/stable/generated/torch.autograd.gradcheck.gradgradcheck.html#torch-autograd-gradcheck-gradgradcheck, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.graph.allow_mutation_on_saved_tensors`, https://pytorch.org/docs/stable/autograd.html#torch.autograd.graph.allow_mutation_on_saved_tensors, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.graph.disable_saved_tensors_hooks`, https://pytorch.org/docs/stable/autograd.html#torch.autograd.graph.disable_saved_tensors_hooks, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.graph.get_gradient_edge`, https://pytorch.org/docs/stable/autograd.html#torch.autograd.graph.get_gradient_edge, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.graph.GradientEdge`, https://pytorch.org/docs/stable/autograd.html#torch.autograd.graph.GradientEdge, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.graph.increment_version`, https://pytorch.org/docs/stable/generated/torch.autograd.graph.increment_version.html#torch-autograd-graph-increment-version, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.graph.register_multi_grad_hook`, https://pytorch.org/docs/stable/autograd.html#torch.autograd.graph.register_multi_grad_hook, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.profiler.emit_itt`, https://pytorch.org/docs/stable/autograd.html#torch.autograd.profiler.emit_itt, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.profiler.emit_nvtx`, https://pytorch.org/docs/stable/autograd.html#torch.autograd.profiler.emit_nvtx, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.profiler.EnforceUnique`, https://pytorch.org/docs/stable/generated/torch.autograd.profiler.EnforceUnique.html#torch.autograd.profiler.EnforceUnique, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.profiler.KinetoStepTracker`, https://pytorch.org/docs/stable/generated/torch.autograd.profiler.KinetoStepTracker.html#torch.autograd.profiler.KinetoStepTracker, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.profiler.parse_nvprof_trace`, https://pytorch.org/docs/stable/generated/torch.autograd.profiler.parse_nvprof_trace.html#torch-autograd-profiler-parse-nvprof-trace, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.profiler.profile.total_average`, https://pytorch.org/docs/stable/generated/torch.autograd.profiler.profile.total_average.html#torch-autograd-profiler-profile-total-average, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.profiler_util.Interval`, https://pytorch.org/docs/stable/generated/torch.autograd.profiler_util.Interval.html#torch.autograd.profiler_util.Interval, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.profiler_util.Kernel`, https://pytorch.org/docs/stable/generated/torch.autograd.profiler_util.Kernel.html#torch.autograd.profiler_util.Kernel, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.profiler_util.StringTable`, https://pytorch.org/docs/stable/generated/torch.autograd.profiler_util.StringTable.html#torch.autograd.profiler_util.StringTable, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.backends.cuda.can_use_efficient_attention`, https://pytorch.org/docs/stable/backends.html#torch.backends.cuda.can_use_efficient_attention, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.backends.cuda.cudnn_sdp_enabled`, https://pytorch.org/docs/stable/backends.html#torch.backends.cuda.cudnn_sdp_enabled, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.backends.cuda.enable_cudnn_sdp`, https://pytorch.org/docs/stable/backends.html#torch.backends.cuda.enable_cudnn_sdp, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.backends.cuda.flash_sdp_enabled`, https://pytorch.org/docs/stable/backends.html#torch.backends.cuda.flash_sdp_enabled, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.backends.cuda.math_sdp_enabled`, https://pytorch.org/docs/stable/backends.html#torch.backends.cuda.math_sdp_enabled, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.backends.cuda.mem_efficient_sdp_enabled`, https://pytorch.org/docs/stable/backends.html#torch.backends.cuda.mem_efficient_sdp_enabled, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.backends.cuda.SDPAParams`, https://pytorch.org/docs/stable/backends.html#torch.backends.cuda.SDPAParams, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.backends.mha.get_fastpath_enabled`, https://pytorch.org/docs/stable/backends.html#torch.backends.mha.get_fastpath_enabled, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.backends.mha.set_fastpath_enabled`, https://pytorch.org/docs/stable/backends.html#torch.backends.mha.set_fastpath_enabled, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.backends.mkl.verbose`, https://pytorch.org/docs/stable/backends.html#torch.backends.mkl.verbose, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.backends.mkldnn.is_available`, https://pytorch.org/docs/stable/backends.html#torch.backends.mkldnn.is_available, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.backends.mkldnn.verbose`, https://pytorch.org/docs/stable/backends.html#torch.backends.mkldnn.verbose, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.backends.nnpack.flags`, https://pytorch.org/docs/stable/backends.html#torch.backends.nnpack.flags, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.backends.nnpack.is_available`, https://pytorch.org/docs/stable/backends.html#torch.backends.nnpack.is_available, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.backends.openmp.is_available`, https://pytorch.org/docs/stable/backends.html#torch.backends.openmp.is_available, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.backends.opt_einsum.get_opt_einsum`, https://pytorch.org/docs/stable/backends.html#torch.backends.opt_einsum.get_opt_einsum, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.backends.opt_einsum.is_available`, https://pytorch.org/docs/stable/backends.html#torch.backends.opt_einsum.is_available, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.signal.windows.nuttall`, https://pytorch.org/docs/stable/generated/torch.signal.windows.nuttall.html#torch-signal-windows-nuttall, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.export.dims`, https://pytorch.org/docs/stable/export.html#torch.export.dims, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.export.dynamic_shapes.Dim`, https://pytorch.org/docs/stable/export.html#torch.export.dynamic_shapes.Dim, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.export.dynamic_shapes.dynamic_dim`, https://pytorch.org/docs/stable/export.html#torch.export.dynamic_shapes.dynamic_dim, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.export.ExportBackwardSignature`, https://pytorch.org/docs/stable/export.html#torch.export.ExportBackwardSignature, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.export.ExportGraphSignature`, https://pytorch.org/docs/stable/export.html#torch.export.ExportGraphSignature, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.export.graph_signature.CustomObjArgument`, https://pytorch.org/docs/stable/export.html#torch.export.graph_signature.CustomObjArgument, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.export.graph_signature.ExportGraphSignature`, https://pytorch.org/docs/stable/export.html#torch.export.graph_signature.ExportGraphSignature, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.export.graph_signature.InputKind`, https://pytorch.org/docs/stable/export.html#torch.export.graph_signature.InputKind, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.export.graph_signature.OutputKind`, https://pytorch.org/docs/stable/export.html#torch.export.graph_signature.OutputKind, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.export.graph_signature.OutputSpec`, https://pytorch.org/docs/stable/export.html#torch.export.graph_signature.OutputSpec, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.export.ModuleCallEntry`, https://pytorch.org/docs/stable/export.html#torch.export.ModuleCallEntry, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.export.ModuleCallSignature`, https://pytorch.org/docs/stable/export.html#torch.export.ModuleCallSignature, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.export.register_dataclass`, https://pytorch.org/docs/stable/export.html#torch.export.register_dataclass, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.export.save`, https://pytorch.org/docs/stable/export.html#torch.export.save, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.export.unflatten.FlatArgsAdapter`, https://pytorch.org/docs/stable/export.html#torch.export.unflatten.FlatArgsAdapter, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.export.unflatten.InterpreterModule`, https://pytorch.org/docs/stable/export.html#torch.export.unflatten.InterpreterModule, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.export.unflatten.unflatten`, https://pytorch.org/docs/stable/export.html#torch.export.unflatten.unflatten, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.fx.experimental.symbolic_shapes.canonicalize_bool_expr`, https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.canonicalize_bool_expr.html#torch-fx-experimental-symbolic-shapes-canonicalize-bool-expr, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.fx.experimental.symbolic_shapes.constrain_range`, https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.constrain_range.html#torch-fx-experimental-symbolic-shapes-constrain-range, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.fx.experimental.symbolic_shapes.constrain_unify`, https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.constrain_unify.html#torch-fx-experimental-symbolic-shapes-constrain-unify, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.fx.experimental.symbolic_shapes.definitely_false`, https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.definitely_false.html#torch-fx-experimental-symbolic-shapes-definitely-false, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.fx.experimental.symbolic_shapes.definitely_true`, https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.definitely_true.html#torch-fx-experimental-symbolic-shapes-definitely-true, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.fx.experimental.symbolic_shapes.DimConstraints`, https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.DimConstraints.html#torch.fx.experimental.symbolic_shapes.DimConstraints, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.fx.experimental.symbolic_shapes.DimDynamic`, https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.DimDynamic.html#torch.fx.experimental.symbolic_shapes.DimDynamic, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.fx.experimental.symbolic_shapes.EqualityConstraint`, https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.EqualityConstraint.html#torch.fx.experimental.symbolic_shapes.EqualityConstraint, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.fx.experimental.symbolic_shapes.guard_size_oblivious`, https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.guard_size_oblivious.html#torch-fx-experimental-symbolic-shapes-guard-size-oblivious, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.fx.experimental.symbolic_shapes.has_free_symbols`, https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.has_free_symbols.html#torch-fx-experimental-symbolic-shapes-has-free-symbols, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.fx.experimental.symbolic_shapes.hint_int`, https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.hint_int.html#torch-fx-experimental-symbolic-shapes-hint-int, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.fx.experimental.symbolic_shapes.is_concrete_bool`, https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.is_concrete_bool.html#torch-fx-experimental-symbolic-shapes-is-concrete-bool, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.fx.experimental.symbolic_shapes.is_concrete_int`, https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.is_concrete_int.html#torch-fx-experimental-symbolic-shapes-is-concrete-int, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.fx.experimental.symbolic_shapes.parallel_and`, https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.parallel_and.html#torch-fx-experimental-symbolic-shapes-parallel-and, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.fx.experimental.symbolic_shapes.parallel_or`, https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.parallel_or.html#torch-fx-experimental-symbolic-shapes-parallel-or, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.fx.experimental.symbolic_shapes.RelaxedUnspecConstraint`, https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.RelaxedUnspecConstraint.html#torch.fx.experimental.symbolic_shapes.RelaxedUnspecConstraint, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.fx.experimental.symbolic_shapes.ShapeEnv`, https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.fx.experimental.symbolic_shapes.StatefulSymbolicContext`, https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.StatefulSymbolicContext.html#torch.fx.experimental.symbolic_shapes.StatefulSymbolicContext, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.fx.experimental.symbolic_shapes.StatelessSymbolicContext`, https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.StatelessSymbolicContext.html#torch.fx.experimental.symbolic_shapes.StatelessSymbolicContext, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.fx.experimental.symbolic_shapes.statically_known_true`, https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.statically_known_true.html#torch-fx-experimental-symbolic-shapes-statically-known-true, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.fx.experimental.symbolic_shapes.StrictMinMaxConstraint`, https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.StrictMinMaxConstraint.html#torch.fx.experimental.symbolic_shapes.StrictMinMaxConstraint, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.fx.experimental.symbolic_shapes.SubclassSymbolicContext`, https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.SubclassSymbolicContext.html#torch.fx.experimental.symbolic_shapes.SubclassSymbolicContext, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.fx.experimental.symbolic_shapes.sym_eq`, https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.sym_eq.html#torch-fx-experimental-symbolic-shapes-sym-eq, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.fx.experimental.symbolic_shapes.SymbolicContext`, https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.SymbolicContext.html#torch.fx.experimental.symbolic_shapes.SymbolicContext, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.jit.interface`, https://pytorch.org/docs/stable/generated/torch.jit.interface.html#torch-jit-interface, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.jit.onednn_fusion_enabled`, https://pytorch.org/docs/stable/generated/torch.jit.onednn_fusion_enabled.html#torch-jit-onednn-fusion-enabled, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.jit.ScriptFunction`, https://pytorch.org/docs/stable/generated/torch.jit.ScriptFunction.html#torch.jit.ScriptFunction, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.jit.strict_fusion`, https://pytorch.org/docs/stable/generated/torch.jit.strict_fusion.html#torch.jit.strict_fusion, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.sym_float`, https://pytorch.org/docs/stable/generated/torch.sym_float.html#torch-sym-float, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.sym_int`, https://pytorch.org/docs/stable/generated/torch.sym_int.html#torch-sym-int, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.sym_ite`, https://pytorch.org/docs/stable/generated/torch.sym_ite.html#torch-sym-ite, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.sym_max`, https://pytorch.org/docs/stable/generated/torch.sym_max.html#torch-sym-max, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.sym_min`, https://pytorch.org/docs/stable/generated/torch.sym_min.html#torch-sym-min, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.sym_not`, https://pytorch.org/docs/stable/generated/torch.sym_not.html#torch-sym-not, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.SymBool`, https://pytorch.org/docs/stable/torch.html#torch.SymBool, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.SymFloat`, https://pytorch.org/docs/stable/torch.html#torch.SymFloat, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.SymInt`, https://pytorch.org/docs/stable/torch.html#torch.SymInt, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.cpu.current_stream`, https://pytorch.org/docs/stable/generated/torch.cpu.current_stream.html#torch.cpu.current_stream, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.cpu.device_count`, https://pytorch.org/docs/stable/generated/torch.cpu.device_count.html#torch-cpu-device-count, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.cpu.is_available`, https://pytorch.org/docs/stable/generated/torch.cpu.is_available.html#torch-cpu-is-available, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.cpu.stream`, https://pytorch.org/docs/stable/generated/torch.cpu.stream.html#torch-cpu-stream, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.cpu.Stream`, https://pytorch.org/docs/stable/generated/torch.cpu.Stream.html#torch.cpu.Stream, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.cpu.StreamContext`, https://pytorch.org/docs/stable/generated/torch.cpu.StreamContext.html#torch.cpu.StreamContext, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.cpu.synchronize`, https://pytorch.org/docs/stable/generated/torch.cpu.synchronize.html#torch-cpu-synchronize, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.change_current_allocator`, https://pytorch.org/docs/stable/generated/torch.cuda.change_current_allocator.html#torch-cuda-change-current-allocator, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.clock_rate`, https://pytorch.org/docs/stable/generated/torch.cuda.clock_rate.html#torch-cuda-clock-rate, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.CUDAPluggableAllocator`, https://pytorch.org/docs/stable/generated/torch.cuda.CUDAPluggableAllocator.html#torch.cuda.CUDAPluggableAllocator, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.current_blas_handle`, https://pytorch.org/docs/stable/generated/torch.cuda.current_blas_handle.html#torch-cuda-current-blas-handle, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.get_gencode_flags`, https://pytorch.org/docs/stable/generated/torch.cuda.get_gencode_flags.html#torch-cuda-get-gencode-flags, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.graph`, https://pytorch.org/docs/stable/cuda.html#module-torch.cuda.graphs, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.jiterator._create_multi_output_jit_fn`, https://pytorch.org/docs/stable/generated/torch.cuda.jiterator._create_multi_output_jit_fn.html#torch-cuda-jiterator-create-multi-output-jit-fn, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.make_graphed_callables`, https://pytorch.org/docs/stable/generated/torch.cuda.make_graphed_callables.html#torch-cuda-make-graphed-callables, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.memory._dump_snapshot`, https://pytorch.org/docs/stable/torch_cuda_memory.html#torch.cuda.memory._dump_snapshot, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.memory._snapshot`, https://pytorch.org/docs/stable/torch_cuda_memory.html#torch.cuda.memory._snapshot, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.power_draw`, https://pytorch.org/docs/stable/generated/torch.cuda.power_draw.html#torch-cuda-power-draw, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.temperature`, https://pytorch.org/docs/stable/generated/torch.cuda.temperature.html#torch-cuda-temperature, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.algorithms.ddp_comm_hooks.debugging_hooks.noop_hook`, https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.debugging_hooks.noop_hook, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.algorithms.ddp_comm_hooks.default_hooks.allreduce_hook`, https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.allreduce_hook, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.algorithms.ddp_comm_hooks.default_hooks.bf16_compress_hook`, https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.bf16_compress_hook, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.algorithms.ddp_comm_hooks.default_hooks.bf16_compress_wrapper`, https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.bf16_compress_wrapper, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_hook`, https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_hook, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_wrapper`, https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_wrapper, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.batched_powerSGD_hook`, https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.batched_powerSGD_hook, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.powerSGD_hook`, https://pytorch.org/docs/stable/distributed.html#module-torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.GradBucket`, https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.GradBucket, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.GradBucket.buffer`, https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.GradBucket.buffer, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.GradBucket.gradients`, https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.GradBucket.gradients, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.GradBucket.index`, https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.GradBucket.index, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.GradBucket.is_last`, https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.GradBucket.is_last, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.GradBucket.parameters`, https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.GradBucket.parameters, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.GradBucket.set_buffer`, https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.GradBucket.set_buffer, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.algorithms.Join`, https://pytorch.org/docs/stable/distributed.algorithms.join.html#torch.distributed.algorithms.Join, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.algorithms.Joinable`, https://pytorch.org/docs/stable/distributed.algorithms.join.html#torch.distributed.algorithms.Joinable, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.algorithms.JoinHook`, https://pytorch.org/docs/stable/distributed.algorithms.join.html#torch.distributed.algorithms.JoinHook, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.autograd.context`, https://pytorch.org/docs/stable/rpc.html#torch.distributed.autograd.context, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.autograd.get_gradients`, https://pytorch.org/docs/stable/rpc.html#torch.distributed.autograd.get_gradients, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.breakpoint`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.breakpoint, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.DistBackendError`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.DistBackendError, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.DistError`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.DistError, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.DistNetworkError`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.DistNetworkError, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.DistStoreError`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.DistStoreError, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.checkpoint.DefaultLoadPlanner`, https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.DefaultLoadPlanner, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.checkpoint.DefaultSavePlanner`, https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.DefaultSavePlanner, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.checkpoint.filesystem.FileSystemReader`, https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.FileSystemReader, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.checkpoint.filesystem.FileSystemWriter`, https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.FileSystemWriter, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader`, https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.checkpoint.format_utils.dcp_to_torch_save`, https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.format_utils.dcp_to_torch_save, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.checkpoint.format_utils.DynamicMetaLoadPlanner`, https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.format_utils.DynamicMetaLoadPlanner, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.checkpoint.format_utils.torch_save_to_dcp`, https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.format_utils.torch_save_to_dcp, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.checkpoint.fsspec.FsspecReader`, https://pytorch.org/docs/2.3/distributed.checkpoint.html#torch.distributed.checkpoint.fsspec.FsspecReader, 废弃 API ，无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.checkpoint.fsspec.FsspecWriter`, https://pytorch.org/docs/2.3/distributed.checkpoint.html#torch.distributed.checkpoint.fsspec.FsspecWriter, 废弃 API ，无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.checkpoint.LoadPlan`, https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.LoadPlan, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.checkpoint.LoadPlanner`, https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.LoadPlanner, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.checkpoint.ReadItem`, https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.ReadItem, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.checkpoint.SavePlan`, https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.SavePlan, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.checkpoint.SavePlanner`, https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.SavePlanner, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.checkpoint.state_dict.get_state_dict`, https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.get_state_dict, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.checkpoint.state_dict.set_state_dict`, https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.set_state_dict, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.checkpoint.state_dict_loader.load`, https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict_loader.load, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.checkpoint.state_dict_loader.load_state_dict`, https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict_loader.load_state_dict, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.checkpoint.state_dict_saver.async_save`, https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict_saver.async_save, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.checkpoint.state_dict_saver.save`, https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict_saver.save, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.checkpoint.state_dict_saver.save_state_dict`, https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict_saver.save_state_dict, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.checkpoint.stateful.Stateful`, https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.stateful.Stateful, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.checkpoint.StorageReader`, https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.StorageReader, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.checkpoint.StorageWriter`, https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.StorageWriter, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.is_mpi_available`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.is_mpi_available, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.is_torchelastic_launched`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.is_torchelastic_launched, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.Work`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.Work, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.HashStore`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.HashStore, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.Store`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.Store, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.Store.add`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.Store.add, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.Store.compare_set`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.Store.compare_set, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.Store.delete_key`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.Store.delete_key, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.Store.get`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.Store.get, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.Store.num_keys`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.Store.num_keys, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.Store.set`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.Store.set, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.Store.set_timeout`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.Store.set_timeout, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.Store.wait`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.Store.wait, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.fsdp.BackwardPrefetch`, https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.BackwardPrefetch, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.fsdp.LocalOptimStateDictConfig`, https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.LocalOptimStateDictConfig, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.fsdp.OptimStateDictConfig`, https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.OptimStateDictConfig, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.fsdp.ShardedOptimStateDictConfig`, https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.ShardedOptimStateDictConfig, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.fsdp.ShardedStateDictConfig`, https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.ShardedStateDictConfig, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.fsdp.ShardingStrategy`, https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.ShardingStrategy, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.fsdp.StateDictConfig`, https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.StateDictConfig, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.fsdp.StateDictSettings`, https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.StateDictSettings, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.nn.api.remote_module.RemoteModule`, https://pytorch.org/docs/stable/rpc.html#torch.distributed.nn.api.remote_module.RemoteModule, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.rpc.BackendType`, https://pytorch.org/docs/stable/rpc.html#torch.distributed.rpc.BackendType, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.rpc.PyRRef`, https://pytorch.org/docs/stable/rpc.html#torch.distributed.rpc.PyRRef, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.rpc.RpcBackendOptions`, https://pytorch.org/docs/stable/rpc.html#torch.distributed.rpc.RpcBackendOptions, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.optim.PostLocalSGDOptimizer`, https://pytorch.org/docs/stable/distributed.optim.html#torch.distributed.optim.PostLocalSGDOptimizer, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.pipeline.sync.skip.skippable.pop`, https://pytorch.org/docs/2.3/pipeline.html#torch.distributed.pipeline.sync.skip.skippable.pop, 废弃 API ，无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.pipeline.sync.skip.skippable.skippable`, https://pytorch.org/docs/2.3/pipeline.html#torch.distributed.pipeline.sync.skip.skippable.skippable, 废弃 API ，无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.pipeline.sync.skip.skippable.stash`, https://pytorch.org/docs/2.3/pipeline.html#torch.distributed.pipeline.sync.skip.skippable.stash, 废弃 API ，无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.pipeline.sync.skip.skippable.verify_skippables`, https://pytorch.org/docs/2.3/pipeline.html#torch.distributed.pipeline.sync.skip.skippable.verify_skippables, 废弃 API ，无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.tensor.parallel.loss_parallel`, https://pytorch.org/docs/stable/distributed.tensor.parallel.html#torch.distributed.tensor.parallel.loss_parallel, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.tensor.parallel.PrepareModuleInput`, https://pytorch.org/docs/stable/distributed.tensor.parallel.html#torch.distributed.tensor.parallel.PrepareModuleInput, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.tensor.parallel.PrepareModuleOutput`, https://pytorch.org/docs/stable/distributed.tensor.parallel.html#torch.distributed.tensor.parallel.PrepareModuleOutput, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.tensor.parallel.SequenceParallel`, https://pytorch.org/docs/stable/distributed.tensor.parallel.html#torch.distributed.tensor.parallel.SequenceParallel, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributions.fishersnedecor.FisherSnedecor`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.fishersnedecor.FisherSnedecor, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributions.half_cauchy.HalfCauchy`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.half_cauchy.HalfCauchy, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributions.half_normal.HalfNormal`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.half_normal.HalfNormal, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributions.inverse_gamma.InverseGamma`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.inverse_gamma.InverseGamma, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributions.kumaraswamy.Kumaraswamy`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.kumaraswamy.Kumaraswamy, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributions.mixture_same_family.MixtureSameFamily`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.mixture_same_family.MixtureSameFamily, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributions.negative_binomial.NegativeBinomial`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.negative_binomial.NegativeBinomial, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributions.pareto.Pareto`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.pareto.Pareto, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributions.relaxed_categorical.RelaxedOneHotCategorical`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.relaxed_categorical.RelaxedOneHotCategorical, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributions.von_mises.VonMises`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.von_mises.VonMises, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributions.weibull.Weibull`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.weibull.Weibull, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributions.wishart.Wishart`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.wishart.Wishart, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.dequantize`, https://pytorch.org/docs/stable/generated/torch.dequantize.html#torch-dequantize, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.quantized_batch_norm`, https://pytorch.org/docs/stable/generated/torch.quantized_batch_norm.html#torch-quantized-batch-norm, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.quantized_max_pool1d`, https://pytorch.org/docs/stable/generated/torch.quantized_max_pool1d.html#torch-quantized-max-pool1d, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.quantized_max_pool2d`, https://pytorch.org/docs/stable/generated/torch.quantized_max_pool2d.html#torch-quantized-max-pool2d, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.overrides.get_ignored_functions`, https://pytorch.org/docs/stable/torch.overrides.html#torch.overrides.get_ignored_functions, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.overrides.handle_torch_function`, https://pytorch.org/docs/stable/torch.overrides.html#torch.overrides.handle_torch_function, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.overrides.is_tensor_method_or_property`, https://pytorch.org/docs/stable/torch.overrides.html#torch.overrides.is_tensor_method_or_property, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.package.Directory`, https://pytorch.org/docs/stable/package.html#torch.package.Directory, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.package.EmptyMatchError`, https://pytorch.org/docs/stable/package.html#torch.package.EmptyMatchError, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.package.PackageExporter`, https://pytorch.org/docs/stable/package.html#torch.package.PackageExporter, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.package.PackagingError`, https://pytorch.org/docs/stable/package.html#torch.package.PackagingError, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.hspmm`, https://pytorch.org/docs/stable/generated/torch.hspmm.html#torch-hspmm, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.smm`, https://pytorch.org/docs/stable/generated/torch.smm.html#torch-smm, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.sparse.as_sparse_gradcheck`, https://pytorch.org/docs/stable/generated/torch.sparse.as_sparse_gradcheck.html#torch-sparse-as-sparse-gradcheck, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.sparse.check_sparse_tensor_invariants`, https://pytorch.org/docs/stable/generated/torch.sparse.check_sparse_tensor_invariants.html#torch.sparse.check_sparse_tensor_invariants, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.sparse.spdiags`, https://pytorch.org/docs/stable/generated/torch.sparse.spdiags.html#torch-sparse-spdiags, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.sspaddmm`, https://pytorch.org/docs/stable/generated/torch.sspaddmm.html#torch-sspaddmm, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.library.fallthrough_kernel`, https://pytorch.org/docs/stable/library.html#torch.library.fallthrough_kernel, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.linalg.solve_ex`, https://pytorch.org/docs/stable/generated/torch.linalg.solve_ex.html#torch-linalg-solve-ex, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.monitor.Aggregation`, https://pytorch.org/docs/stable/monitor.html#torch.monitor.Aggregation, 实验阶段不稳定 API ，无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.monitor.data_value_t`, https://pytorch.org/docs/stable/monitor.html#torch.monitor.data_value_t, 实验阶段不稳定 API ，无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.monitor.EventHandlerHandle`, https://pytorch.org/docs/stable/monitor.html#torch.monitor.EventHandlerHandle, 实验阶段不稳定 API ，无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.monitor.TensorboardEventHandler`, https://pytorch.org/docs/stable/monitor.html#torch.monitor.TensorboardEventHandler, 实验阶段不稳定 API ，无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.nested.as_nested_tensor`, https://pytorch.org/docs/stable/nested.html#torch.nested.as_nested_tensor, 实验阶段不稳定 API ，无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.nested.to_padded_tensor`, https://pytorch.org/docs/stable/nested.html#torch.nested.to_padded_tensor, 实验阶段不稳定 API ，无需新增) |
| NOT-IMPLEMENTED-ITEM(`torch.mps.driver_allocated_memory`, https://pytorch.org/docs/stable/generated/torch.mps.driver_allocated_memory.html#torch-mps-driver-allocated-memory, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.mps.event.Event`, https://pytorch.org/docs/stable/generated/torch.mps.event.Event.html#torch.mps.event.Event, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.mps.get_rng_state`, https://pytorch.org/docs/stable/generated/torch.mps.get_rng_state.html#torch-mps-get-rng-state, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.mps.profiler.profile`, https://pytorch.org/docs/stable/generated/torch.mps.profiler.profile.html#torch-mps-profiler-profile, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.mps.seed`, https://pytorch.org/docs/stable/generated/torch.mps.seed.html#torch-mps-seed, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.mps.set_per_process_memory_fraction`, https://pytorch.org/docs/stable/generated/torch.mps.set_per_process_memory_fraction.html#torch-mps-set-per-process-memory-fraction, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.mps.set_rng_state`, https://pytorch.org/docs/stable/generated/torch.mps.set_rng_state.html#torch-mps-set-rng-state, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.attention.bias`, https://pytorch.org/docs/stable/nn.attention.bias.html#module-torch.nn.attention.bias, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.attention.sdpa_kernel`, https://pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html#torch-nn-attention-sdpa-kernel, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.attention.SDPBackend`, https://pytorch.org/docs/stable/generated/torch.nn.attention.SDPBackend.html#torch.nn.attention.SDPBackend, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.CircularPad1d`, https://pytorch.org/docs/stable/generated/torch.nn.CircularPad1d.html#torch.nn.CircularPad1d, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.CircularPad2d`, https://pytorch.org/docs/stable/generated/torch.nn.CircularPad2d.html#torch.nn.CircularPad2d, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.functional.lp_pool3d`, https://pytorch.org/docs/stable/generated/torch.nn.functional.lp_pool3d.html#torch-nn-functional-lp-pool3d, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.LPPool3d`, https://pytorch.org/docs/stable/generated/torch.nn.LPPool3d.html#torch.nn.LPPool3d, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.modules.lazy.LazyModuleMixin`, https://pytorch.org/docs/stable/generated/torch.nn.modules.lazy.LazyModuleMixin.html#torch.nn.modules.lazy.LazyModuleMixin, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.modules.module.register_module_buffer_registration_hook`, https://pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_buffer_registration_hook.html#torch-nn-modules-module-register-module-buffer-registration-hook, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.modules.module.register_module_full_backward_hook`, https://pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_full_backward_hook.html#torch-nn-modules-module-register-module-full-backward-hook, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.modules.module.register_module_full_backward_pre_hook`, https://pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_full_backward_pre_hook.html#torch-nn-modules-module-register-module-full-backward-pre-hook, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.modules.module.register_module_module_registration_hook`, https://pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_module_registration_hook.html#torch-nn-modules-module-register-module-module-registration-hook, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.modules.module.register_module_parameter_registration_hook`, https://pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_parameter_registration_hook.html#torch-nn-modules-module-register-module-parameter-registration-hook, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.parametrize.cached`, https://pytorch.org/docs/stable/generated/torch.nn.utils.parametrize.cached.html#torch-nn-utils-parametrize-cached, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.parametrize.ParametrizationList`, https://pytorch.org/docs/stable/generated/torch.nn.utils.parametrize.ParametrizationList.html#torch.nn.utils.parametrize.ParametrizationList, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.prune.BasePruningMethod`, https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.BasePruningMethod.html#torch.nn.utils.prune.BasePruningMethod, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.prune.custom_from_mask`, https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.custom_from_mask.html#torch-nn-utils-prune-custom-from-mask, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.prune.CustomFromMask`, https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.CustomFromMask.html#torch.nn.utils.prune.CustomFromMask, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.prune.Identity`, https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.identity.html#torch-nn-utils-prune-identity, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.prune.is_pruned`, https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.is_pruned.html#torch-nn-utils-prune-is-pruned, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.prune.LnStructured`, https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.LnStructured.html#torch.nn.utils.prune.LnStructured, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.prune.RandomUnstructured`, https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.RandomUnstructured.html#torch.nn.utils.prune.RandomUnstructured, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.rnn.unpack_sequence`, https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.unpack_sequence.html#torch-nn-utils-rnn-unpack-sequence, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.ZeroPad1d`, https://pytorch.org/docs/stable/generated/torch.nn.ZeroPad1d.html#torch.nn.ZeroPad1d, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.ZeroPad3d`, https://pytorch.org/docs/stable/generated/torch.nn.ZeroPad3d.html#torch.nn.ZeroPad3d, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.profiler._KinetoProfile`, https://pytorch.org/docs/stable/profiler.html#torch.profiler._KinetoProfile, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.profiler.itt.is_available`, https://pytorch.org/docs/stable/profiler.html#torch.profiler.itt.is_available, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.profiler.itt.mark`, https://pytorch.org/docs/stable/profiler.html#torch.profiler.itt.mark, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.profiler.itt.range_pop`, https://pytorch.org/docs/stable/profiler.html#torch.profiler.itt.range_pop, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.profiler.itt.range_push`, https://pytorch.org/docs/stable/profiler.html#torch.profiler.itt.range_push, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.special.airy_ai`, https://pytorch.org/docs/stable/special.html#torch.special.airy_ai, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.special.bessel_j0`, https://pytorch.org/docs/stable/special.html#torch.special.bessel_j0, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.special.bessel_j1`, https://pytorch.org/docs/stable/special.html#torch.special.bessel_j1, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.special.scaled_modified_bessel_k0`, https://pytorch.org/docs/stable/special.html#torch.special.scaled_modified_bessel_k0, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.special.scaled_modified_bessel_k1`, https://pytorch.org/docs/stable/special.html#torch.special.scaled_modified_bessel_k1, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.special.spherical_bessel_j0`, https://pytorch.org/docs/stable/special.html#torch.special.spherical_bessel_j0, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.arctan2_`, https://pytorch.org/docs/stable/generated/torch.Tensor.arctan2_.html#torch-tensor-arctan2, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.conj_physical_`, https://pytorch.org/docs/stable/generated/torch.Tensor.conj_physical_.html#torch-tensor-conj-physical, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.is_meta`, https://pytorch.org/docs/stable/generated/torch.Tensor.is_meta.html#torch-tensor-is-meta, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.is_quantized`, https://pytorch.org/docs/stable/generated/torch.Tensor.is_quantized.html#torch-tensor-is-quantized, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.module_load`, https://pytorch.org/docs/stable/generated/torch.Tensor.module_load.html#torch-tensor-module-load, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.nextafter_`, https://pytorch.org/docs/stable/generated/torch.Tensor.nextafter_.html#torch-tensor-nextafter, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.retains_grad`, https://pytorch.org/docs/stable/generated/torch.Tensor.retains_grad.html#torch-tensor-retains-grad, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.smm`, https://pytorch.org/docs/stable/generated/torch.Tensor.smm.html#torch-tensor-smm, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.utils.benchmark.CallgrindStats`, https://pytorch.org/docs/stable/benchmark_utils.html#torch.utils.benchmark.CallgrindStats, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.utils.benchmark.FunctionCounts`, https://pytorch.org/docs/stable/benchmark_utils.html#torch.utils.benchmark.FunctionCounts, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.utils.benchmark.Measurement`, https://pytorch.org/docs/stable/benchmark_utils.html#torch.utils.benchmark.Measurement, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.utils.checkpoint.set_checkpoint_debug_enabled`, https://pytorch.org/docs/stable/checkpoint.html#torch.utils.checkpoint.set_checkpoint_debug_enabled, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.utils.cpp_extension.get_compiler_abi_compatibility_and_version`, https://pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.get_compiler_abi_compatibility_and_version, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.utils.cpp_extension.is_ninja_available`, https://pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.is_ninja_available, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.utils.data.default_convert`, https://pytorch.org/docs/stable/data.html#torch.utils.data.default_convert, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.utils.generate_methods_for_privateuse1_backend`, https://pytorch.org/docs/stable/generated/torch.utils.generate_methods_for_privateuse1_backend.html#torch-utils-generate-methods-for-privateuse1-backend, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.utils.get_cpp_backtrace`, https://pytorch.org/docs/stable/generated/torch.utils.get_cpp_backtrace.html#torch-utils-get-cpp-backtrace, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.utils.rename_privateuse1_backend`, https://pytorch.org/docs/stable/generated/torch.utils.rename_privateuse1_backend.html#torch-utils-rename-privateuse1-backend, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.xpu.current_stream`, https://pytorch.org/docs/stable/generated/torch.xpu.current_stream.html#torch-xpu-current-stream, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.xpu.device`, https://pytorch.org/docs/stable/generated/torch.xpu.device.html#torch.xpu.device, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.xpu.device_of`, https://pytorch.org/docs/stable/generated/torch.xpu.device_of.html#torch.xpu.device_of, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.xpu.Event`, https://pytorch.org/docs/stable/generated/torch.xpu.Event.html#torch.xpu.Event, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.xpu.get_device_capability`, https://pytorch.org/docs/stable/generated/torch.xpu.get_device_capability.html#torch-xpu-get-device-capability, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.xpu.init`, https://pytorch.org/docs/stable/generated/torch.xpu.init.html#torch-xpu-init, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.xpu.initial_seed`, https://pytorch.org/docs/stable/generated/torch.xpu.initial_seed.html#torch-xpu-initial-seed, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.xpu.is_initialized`, https://pytorch.org/docs/stable/generated/torch.xpu.is_initialized.html#torch-xpu-is-initialized, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.xpu.seed`, https://pytorch.org/docs/stable/generated/torch.xpu.seed.html#torch-xpu-seed, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.xpu.seed_all`, https://pytorch.org/docs/stable/generated/torch.xpu.seed_all.html#torch-xpu-seed-all, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.xpu.set_stream`, https://pytorch.org/docs/stable/generated/torch.xpu.set_stream.html#torch-xpu-set-stream, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.xpu.stream`, https://pytorch.org/docs/stable/generated/torch.xpu.stream.html#torch-xpu-stream, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.xpu.Stream`, https://pytorch.org/docs/stable/generated/torch.xpu.Stream.html#torch.xpu.Stream, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.xpu.StreamContext`, https://pytorch.org/docs/stable/generated/torch.xpu.StreamContext.html#torch.xpu.StreamContext, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.geqrf`, https://pytorch.org/docs/stable/generated/torch.geqrf.html#torch-geqrf, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.geqrf`, https://pytorch.org/docs/stable/generated/torch.Tensor.geqrf.html#torch-tensor-geqrf, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributions.one_hot_categorical.OneHotCategorical`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.one_hot_categorical.OneHotCategorical, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributions.constraint_registry.ConstraintRegistry`, https://pytorch.org/docs/stable/distributions.html#torch.distributions.constraint_registry.ConstraintRegistry, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.rpc.functions.async_execution`, https://pytorch.org/docs/stable/rpc.html#torch.distributed.rpc.functions.async_execution, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.Tensor.sparse_resize_and_clear_`, https://pytorch.org/docs/stable/generated/torch.Tensor.sparse_resize_and_clear_.html#torch-tensor-sparse-resize-and-clear, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.parametrize.is_parametrized`, https://pytorch.org/docs/stable/generated/torch.nn.utils.parametrize.is_parametrized.html#torch-nn-utils-parametrize-is-parametrized, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.profiler.profile.self_cpu_time_total`, https://pytorch.org/docs/stable/generated/torch.autograd.profiler.profile.self_cpu_time_total.html#torch-autograd-profiler-profile-self-cpu-time-total, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.profiler.ProfilerActivity`, https://pytorch.org/docs/stable/profiler.html#torch.profiler.ProfilerActivity, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.profiler.ProfilerAction`, https://pytorch.org/docs/stable/profiler.html#torch.profiler.ProfilerAction, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.resolve_conj`, https://pytorch.org/docs/stable/generated/torch.resolve_conj.html#torch.resolve_conj, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.resolve_neg`, https://pytorch.org/docs/stable/generated/torch.resolve_neg.html#torch-resolve-neg, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.function.FunctionCtx.mark_dirty`, https://pytorch.org/docs/stable/generated/torch.autograd.function.FunctionCtx.mark_dirty.html#torch-autograd-function-functionctx-mark-dirty, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.is_conj`, https://pytorch.org/docs/stable/generated/torch.is_conj.html#torch-is-conj, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.memory_usage`, https://pytorch.org/docs/stable/generated/torch.cuda.memory_usage.html#torch-cuda-memory-usage, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.layout`, https://pytorch.org/docs/stable/tensor_attributes.html#torch.layout, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.is_current_stream_capturing`, https://pytorch.org/docs/stable/generated/torch.cuda.is_current_stream_capturing.html#torch-cuda-is-current-stream-capturing, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.device_of`, https://pytorch.org/docs/stable/generated/torch.cuda.device_of.html, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.parameter.UninitializedParameter`, https://pytorch.org/docs/stable/generated/torch.nn.parameter.UninitializedParameter.html, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.gather_object`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.gather_object, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.jit.trace`, https://pytorch.org/docs/stable/generated/torch.jit.trace.html#torch-jit-trace, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.jit.unused`, https://pytorch.org/docs/stable/generated/torch.jit.unused.html#torch-jit-unused, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.utils.checkpoint.checkpoint_sequential`, https://pytorch.org/docs/stable/checkpoint.html#torch.utils.checkpoint.checkpoint_sequential, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.parameter.UninitializedBuffer`, https://pytorch.org/docs/stable/generated/torch.nn.parameter.UninitializedBuffer.html#torch.nn.parameter.UninitializedBuffer, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.memory_format`, https://pytorch.org/docs/stable/tensor_attributes.html#torch.memory_format, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.is_gloo_available`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.is_gloo_available, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.get_group_rank`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.get_group_rank, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.get_global_rank`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.get_process_group_ranks, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.set_deterministic_debug_mode`, https://pytorch.org/docs/stable/generated/torch.set_deterministic_debug_mode.html#torch-set-deterministic-debug-mode, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.get_deterministic_debug_mode`, https://pytorch.org/docs/stable/generated/torch.get_deterministic_debug_mode.html#torch-get-deterministic-debug-mode, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.graph.Node.name`, https://pytorch.org/docs/stable/generated/torch.autograd.graph.Node.name.html#torch-autograd-graph-node-name, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.graph.Node.metadata`, https://pytorch.org/docs/stable/generated/torch.autograd.graph.Node.metadata.html#torch-autograd-graph-node-metadata, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.graph.Node.next_functions`, https://pytorch.org/docs/stable/generated/torch.autograd.graph.Node.next_functions.html#torch-autograd-graph-node-next-functions, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.graph.Node.register_hook`, https://pytorch.org/docs/stable/generated/torch.autograd.graph.Node.register_hook.html#torch-autograd-graph-node-register-hook, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.graph.Node.register_prehook`, https://pytorch.org/docs/stable/generated/torch.autograd.graph.Node.register_prehook.html#torch-autograd-graph-node-register-prehook, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.OutOfMemoryError`, https://pytorch.org/docs/stable/generated/torch.cuda.OutOfMemoryError.html#torch-cuda-outofmemoryerror, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.backends.cpu.get_cpu_capability`, https://pytorch.org/docs/stable/backends.html#torch.backends.cpu.get_cpu_capability, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.get_process_group_ranks`, https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.get_process_group_ranks, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.fuse_conv_bn_eval`, https://pytorch.org/docs/stable/generated/torch.nn.utils.fuse_conv_bn_eval.html#torch-nn-utils-fuse-conv-bn-eval, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.fuse_conv_bn_weights`, https://pytorch.org/docs/stable/generated/torch.nn.utils.fuse_conv_bn_weights.html#torch-nn-utils-fuse-conv-bn-weights, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.fuse_linear_bn_eval`, https://pytorch.org/docs/stable/generated/torch.nn.utils.fuse_linear_bn_eval.html#torch-nn-utils-fuse-linear-bn-eval, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.fuse_linear_bn_weights`, https://pytorch.org/docs/stable/generated/torch.nn.utils.fuse_linear_bn_weights.html#torch-nn-utils-fuse-linear-bn-weights, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.functional.conv_tbc`, https://pytorch.org/docs/, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.functional.celu_`, https://pytorch.org/docs, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.functional.selu_`, https://pytorch.org/docs, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.convert_conv2d_weight_memory_format`, https://pytorch.org/docs/stable/generated/torch.nn.utils.convert_conv2d_weight_memory_format.html#torch-nn-utils-convert-conv2d-weight-memory-format, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.nn.utils.convert_conv3d_weight_memory_format`, https://pytorch.org/docs/stable/generated/torch.nn.utils.convert_conv3d_weight_memory_format.html#torch-nn-utils-convert-conv3d-weight-memory-format, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.utils.tensorboard.writer.SummaryWriter`, https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter, 可新增，但框架底层无相关设计，成本高) |
| NOT-IMPLEMENTED-ITEM(`torch.backends.cuda.can_use_flash_attention`, https://pytorch.org/docs/stable/backends.html#torch.backends.cuda.can_use_flash_attention, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.distributed.device_mesh.DeviceMesh`, https://pytorch.org/docs/stable/distributed.html#torch.distributed.device_mesh.DeviceMesh, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.comm.scatter`, https://pytorch.org/docs/stable/generated/torch.cuda.comm.scatter.html#torch-cuda-comm-scatter, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.cuda.comm.gather`, https://pytorch.org/docs/stable/generated/torch.cuda.comm.gather.html#torch-cuda-comm-gather, 可新增，且框架底层有相关设计，成本低) |
| NOT-IMPLEMENTED-ITEM(`torch.autograd.Function.jvp`, https://pytorch.org/docs/stable/generated/torch.autograd.Function.jvp.html#torch-autograd-function-jvp, 可新增，且框架底层有相关设计，成本低) |


## <span id="id26">映射关系开发中的 API 列表</span>

| 序号 | Pytorch 最新 release | Paddle develop | 映射关系分类 | 备注 |
| ----- | ----------- | ----------------- | ----------- | ------- |
