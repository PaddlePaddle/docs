## [ torch参数更多 ]torch.Tensor.logsumexp

同torch.Tensor

### [torch.Tensor.logsumexp](https://pytorch.org/docs/stable/generated/torch.logsumexp.html)

```python
torch.Tensor.logsumexp(input, 
                       dim, 
                       keepdim=False, 
                       *, 
                       out=None)
```

### [paddle.Tensor.logsumexp](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/logsumexp_cn.html#logsumexp)

```python
paddle.Tensor.logsumexp(x, 
                        axis=None, 
                        keepdim=False, 
                        name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                     |
| ------- | ------------ | -------------------------------------------------------- |
| input   | x            | 输入的多维 Tensor ，仅参数名不同。                       |
| dim     | axis         | 指定进行运算的维度，仅参数名不同。                       |
| keepdim | keepdim      | 是否在输出 Tensor 中保留减小的维度。                     |
| out     | -            | 表示输出的 Tensor，PaddlePaddle 无此参数，需要进行转写。 |

### 转写示例

#### out:指定输出

```python
# Pytorch 写法
torch.logsumexp(torch.randn(3, 3), 1, False, out = x)

# Paddle 写法
paddle.logsumexp(paddle.randn([3, 3]), 1, False)
```