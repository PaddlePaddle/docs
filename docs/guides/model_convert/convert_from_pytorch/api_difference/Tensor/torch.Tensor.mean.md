## [ 参数不一致 ]torch.Tensor.mean

### [torch.Tensor.mean](https://pytorch.org/docs/stable/generated/torch.Tensor.mean.html)

```python
torch.Tensor.mean(dim=None, keepdim=False, *, dtype=None)
```

### [paddle.Tensor.mean](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#mean-axis-none-keepdim-false-name-none)

```python
paddle.Tensor.mean(axis=None, keepdim=False, name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注 |
| ------- | ------------ | -- |
| dim     | axis         | 指定对 x 进行计算的轴。 |
| keepdim | keepdim      | 是否在输出 Tensor 中保留减小的维度。 |
| dtype   | -       | 输出 Tensor 的类型，Paddle 无此参数, 需要转写。  |

### 转写示例

#### dtype：输出数据类型
```python
# PyTorch 写法
x.mean(dtype=torch.float32)

# Paddle 写法
x.mean().astype(paddle.float32)
```
