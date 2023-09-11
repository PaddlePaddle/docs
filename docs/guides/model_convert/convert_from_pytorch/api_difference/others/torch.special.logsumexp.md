## [ torch 参数更多 ]torch.special.logsumexp
### [torch.special.logsumexp](https://pytorch.org/docs/stable/special.html#torch.special.logsumexp)

```python
torch.special.logsumexp(input, dim, keepdim=False, *, out=None)
```

### [paddle.logsumexp](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/logsumexp_cn.html)

```python
paddle.logsumexp(x, axis=None, keepdim=False, name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input          |  x           | 输入的 Tensor.仅参数名不一致。  |
| dim          |  axis           | 指定对输入进行计算的轴。仅参数名不一致。  |
| keepdim          |  keepdim           | 是否在输出 Tensor 中保留减小的维度。  |
| out         | -         | 表示输出的 Tensor ， Paddle 无此参数，需要转写。 |

### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.special.logsumexp(input, dim=1, out=y)

# Paddle 写法
paddle.assign(paddle.logsumexp(input, axis=1), y)
```
