## [ 仅 API 调用方式不一致 ]torch.distributions.kl.kl_divergence

### [torch.distributions.kl.kl\_divergence](https://docs.pytorch.org/docs/stable/distributions.html#torch.distributions.kl.kl_divergence)

```python
torch.distributions.kl.kl_divergence(p, q)
```

### [paddle.distribution.kl\_divergence](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distribution/kl_divergence_cn.html#paddle.distribution.kl_divergence)

```python
paddle.distribution.kl_divergence(p, q)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
result = torch.distributions.kl.kl_divergence(m, n)

# Paddle 写法
result = paddle.distribution.kl_divergence(m, n)
```
