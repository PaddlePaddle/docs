## [ 参数完全一致 ]torch.Tensor.mvlgamma_

### [torch.Tensor.mvlgamma_](https://pytorch.org/docs/stable/generated/torch.Tensor.mvlgamma_.html#torch-tensor-mvlgamma)

```python
torch.Tensor.mvlgamma_(p)
```

### [paddle.Tensor.multigammaln_](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/multigammaln__cn.html#multigammaln)

```python
paddle.Tensor.multigammaln_(p, name=None)
```

两者功能一致，参数完全一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                 |
| ------- | ------------ | ---------------------------------------------------- |
| p       | p            | 多元伽马函数积分空间的维度。                         |

### 转写示例

```python
# PyTorch 写法
y = x.mvlgamma_(p)

# Paddle 写法
y = x.multigammaln_(p)
```
