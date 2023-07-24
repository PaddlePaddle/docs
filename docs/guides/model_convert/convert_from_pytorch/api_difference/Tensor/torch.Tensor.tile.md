## [ 参数不一致 ] torch.Tensor.tile

### [torch.Tensor.tile](https://pytorch.org/docs/stable/generated/torch.Tensor.tile.html#torch.Tensor.tile)

```
torch.Tensor.tile(*reps)
```

### [paddle.Tensor.tile](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Tensor_cn.html#tile-repeat-times-name-none)

```
paddle.Tensor.tile(repeat_times, name=None)
```

两者功能一致，但 pytorch 的 `reps` 和 paddle 的 `repeat_times` 参数用法不一致，具体如下：

### 参数映射

| Pytorch | PaddlePaddle | 备注                                                         |
| ------- | ------------ | :----------------------------------------------------------- |
| *reps   | repeat_times | 维度复制次数， Pytorch 参数 reps 既可以是可变参数，也可以是 list/tuple/tensor 的形式， Paddle 参数 repeat_times 为 list/tuple/tensor 的形式。 |

转写示例

#### ***reps: 维度复制次数**

```
# Pytorch 写法
x = torch.tensor([1, 2, 3])
x.tile(2,3)
# Paddle 写法
y= paddle.to_tensor([1, 2, 3], dtype='int32')
y.tile([2,3])
```
