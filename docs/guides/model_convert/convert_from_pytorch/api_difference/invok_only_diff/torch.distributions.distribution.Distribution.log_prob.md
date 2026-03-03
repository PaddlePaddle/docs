## [ 仅 API 调用方式不一致 ]torch.distributions.distribution.Distribution.log_prob

### [torch.distributions.distribution.Distribution.log\_prob](https://docs.pytorch.org/docs/stable/distributions.html#torch.distributions.distribution.Distribution.log_prob)

```python
torch.distributions.distribution.Distribution.log_prob(value)
```

### [paddle.distribution.Distribution.log_prob](https://github.com/PaddlePaddle/Paddle/blob/e6b0ae75d5a7a47100e0e13f34a6ddd61c14b450/python/paddle/distribution/distribution.py#L122-L124)

```python
paddle.distribution.Distribution.log_prob(value)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
uniform = torch.distributions.Uniform(0.0, 1.0)
result = uniform.log_prob(torch.tensor(0.3))

# Paddle 写法
uniform = paddle.distribution.Uniform(0.0, 1.0)
result = uniform.log_prob(paddle.to_tensor(0.3))
```
