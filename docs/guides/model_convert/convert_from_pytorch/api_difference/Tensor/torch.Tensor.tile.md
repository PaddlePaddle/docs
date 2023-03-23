## 分类名[仅参数名不一致]torch.Tensor.tile

### [torch.Tensor.tile](https://pytorch.org/docs/1.13/generated/torch.Tensor.tile.html#torch.Tensor.tile)

```
torch.Tensor.tile(*reps)
```

### [paddle.Tensor.tile](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Tensor_cn.html#tile-repeat-times-name-none)

```
paddle.Tensor.tile(repeat_times, name=None)
```

两者功能一致，仅参数名不一致，具体如下：

### 参数映射

| Pytorch | PaddlePaddle | 备注                                                |
| ------- | ------------ | :-------------------------------------------------- |
| reps    | repeat_times | 指定输入 tensor每个维度的复制次数，仅参数名不一致。 |

### 差异

Pytorch的reps的参数，仅支持tensor

PaddlePaddle的repeat_times，支持list/tuple/tensor；如果 `repeat_times` 的类型是 list 或 tuple，它的元素可以是整数或者数据类型为 int32 的 1-D Tensor。如果 `repeat_times` 的类型是 Tensor，则是数据类型为 int32 的 1-D Tensor。