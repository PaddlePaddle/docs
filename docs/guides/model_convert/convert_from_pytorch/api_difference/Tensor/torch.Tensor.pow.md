## [ 仅参数名不一致 ] torch.Tensor.pow

### [torch.Tensor.pow](https://pytorch.org/docs/stable/generated/torch.Tensor.pow.html?highlight=pow#torch.Tensor.pow)

```python
torch.Tensor.pow(input, exponent, *, out=None)
```

### [paddle.Tensor.pow](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/outer_cn.html=)

```python
paddle.pow(x, y, name=None)
```

两者功能一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                               |
| ------- | ------------ | -------------------------------------------------- |
| input   | x            | 一个 N 维 Tensor 或者标量 Tensor，仅参数名不一致。 |
| vec2    | y            | 一个 N 维 Tensor 或者标量 Tensor，仅参数名不一致。 |
| -      | name      | 一般无需设置，默认值为 None。 |
| out           | -            | 表示输出的 Tensor，PaddlePaddle 无此参数，需要进行转写。                              |

### 转写示例

#### out：指定输出
```python
# Pytorch 写法
torch.Tensor.pow(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]), out = y) # 同 y = torch.Tensor.pow(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))
# Paddle 写法
y = paddle.pow(paddle.to_tensor([[1, 2], [3, 4]]), paddle.to_tensor([[1, 1], [4, 4]]))
```
