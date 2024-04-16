## [torch 参数更多]torch.Tensor.add

### [torch.Tensor.add](https://pytorch.org/docs/stable/generated/torch.Tensor.add.html#torch.Tensor.add)

```python
torch.Tensor.add(other, *, alpha=1)
```

### [paddle.Tensor.add](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#add-y-name-none)

```python
paddle.Tensor.add(y, name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                     |
| ------- | ------------ | -------------------------------------------------------- |
| other   | y            | 输入的 Tensor，仅参数名不一致。                          |
| alpha   | -            | 表示 other 的乘数，Paddle 无此参数，需要转写。 |

### 转写示例

#### alpha：other 的乘数

```python
# PyTorch 写法
y = torch.tensor([3, 5]).add(torch.tensor([2, 3]), alpha=2)

# Paddle 写法
y = paddle.to_tensor([3, 5]).add(2 * paddle.to_tensor([2, 3]))
```
