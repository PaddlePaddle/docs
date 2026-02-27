## [ 仅 API 调用方式不一致 ]torch.Tensor.mvlgamma

### [torch.Tensor.mvlgamma](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.mvlgamma.html#torch.Tensor.mvlgamma)

```python
torch.Tensor.mvlgamma(p)
```

### [paddle.Tensor.multigammaln](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor/multigammaln_cn.html#paddle/Tensor/multigammaln_cn#cn-api-paddle-Tensor-multigammaln)

```python
paddle.Tensor.multigammaln(p, name=None)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
result = x.mvlgamma(2)

# Paddle 写法
result = x.multigammaln(2)
```
