## [ 参数不一致 ]torch.Tensor.repeat

### [torch.Tensor.repeat](https://pytorch.org/docs/stable/generated/torch.Tensor.repeat.html)

```python
torch.Tensor.repeat(*sizes)
```

### [paddle.Tensor.tile](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Tensor_cn.html#tile-repeat-times-name-none)

```python
paddle.Tensor.tile(repeat_times, name=None)
```

Pytorch 的 `*sizes` 相比于 Paddle 的 `repeat_times` 额外支持可变参数的用法，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| *sizes | repeat_times | torch 支持可变参数或 list/tuple，paddle 仅支持 list/tuple。对于可变参数的用法，需要进行转写。 |

### 转写示例
#### *sizes: 各个维度重复的次数，可变参数用法
```python
# pytorch
x = torch.randn(2, 3, 5)
y = x.repeat(4, 2, 1)

# paddle
x = paddle.randn([2, 3, 5])
y = x.tile((4, 2, 1))
```
