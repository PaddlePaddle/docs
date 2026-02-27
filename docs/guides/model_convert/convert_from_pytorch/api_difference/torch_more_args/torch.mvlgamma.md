## [ torch 参数更多 ]torch.mvlgamma
### [torch.mvlgamma](https://docs.pytorch.org/docs/stable/generated/torch.mvlgamma.html#torch.mvlgamma)
```python
torch.mvlgamma(input, p, *, out=None)
```

### [paddle.multigammaln](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/multigammaln_cn.html#paddle.multigammaln)
```python
paddle.multigammaln(x, p, name=None)
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                            |
| ------- | ------------ | ----------------------------------------------- |
| input   | x            | 输入的 Tensor，仅参数名不一致。                 |
| p       | p            | 维度的数量。                                    |
| out     | -            | 表示输出的 Tensor， Paddle 无此参数，需要转写。 |

### 转写示例
#### out：指定输出
```python
# PyTorch 写法
torch.mvlgamma(x, p, out=y)

# Paddle 写法
paddle.assign(paddle.multigammaln(x, p), y)
```
