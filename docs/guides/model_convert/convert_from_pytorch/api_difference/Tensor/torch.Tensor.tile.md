## [参数用法不一致]torch.Tensor.tile

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

| Pytorch | PaddlePaddle | 备注                                                         |
| ------- | ------------ | :----------------------------------------------------------- |
| *reps   | repeat_times | 维度复制次数，Pytorch参数*reps为可变参，Paddle参数repeat_times为list/tuple/tensor的形式。 |

转写示例

#### ***reps:维度复制次数**

```
# Pytorch 写法
x = torch.tensor([1, 2, 3])
x.tile((2,))
# Paddle 写法
y= paddle.to_tensor([1, 2, 3], dtype='int32')
y.tile([2,])
```

