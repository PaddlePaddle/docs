## [ 仅 API 调用方式不一致 ]torch.distributions.distribution.Distribution.log_prob

### [torch.distributions.distribution.Distribution.log_prob](https://pytorch.org/docs/stable/generated/torch.distributions.distribution.Distribution.html#torch.distributions.distribution.Distribution.log_prob)

```python
torch.distributions.distribution.Distribution.log_prob(value)
```

### [paddle.distribution.Distribution.log_prob](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distribution/Distribution/log_prob_cn.html#paddle/distribution/Distribution/log_prob_cn#cn-api-paddle-distribution-Distribution-log_prob)

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
