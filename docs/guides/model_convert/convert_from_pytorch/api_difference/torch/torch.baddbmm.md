## [ torch 参数更多]torch.baddbmm

### [torch.baddbmm](https://pytorch.org/docs/stable/generated/torch.baddbmm.html?highlight=baddbmm#torch.baddbmm)

```python
torch.baddbmm(input, batch1, batch2, beta=1, alpha=1, out=None)
```

### [paddle.baddbmm](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/baddbmm_cn.html)

```python
paddle.baddbmm(input, x, y, alpha=1.0, beta=1.0, name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注 |
| ------- | ------- | ------- |
| input | input | 表示输入的 Tensor 。 |
| batch1 | x | 表示输入的第一个 Tensor ，仅参数名不一致。 |
| batch2 | y | 表示输入的第二个 Tensor ，仅参数名不一致。 |
| beta | beta | 表示乘以 input 的标量。 |
| alpha | alpha | 表示乘以 batch1 * batch2 的标量。 |
| out | - | 表示输出的 Tensor ， Paddle 无此参数，需要转写。 |

### 转写示例

#### out: 输出的 Tensor

```python
# PyTorch 写法
torch.baddbmm(input, batch1, batch2, beta, alpha, out=output)

# Paddle 写法
paddle.assign(paddle.baddbmm(input, x, y, beta, alpha), output)
```
