## [ 仅 API 调用方式不一致 ]torch.Tensor.mvlgamma_

### [torch.Tensor.mvlgamma_](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.mvlgamma_)

```python
torch.Tensor.mvlgamma_(p)
```

### [paddle.Tensor.multigammaln_](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor/multigammaln__cn.html#paddle/Tensor/multigammaln__cn#cn-api-paddle-Tensor-multigammaln_)

```python
paddle.Tensor.multigammaln_(p, name=None)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
result = x.mvlgamma_(2)

# Paddle 写法
result = x.multigammaln_(2)
```
