## [ 仅 API 调用方式不一致 ]torch.distributions.constraints.Constraint

### [torch.distributions.constraints.Constraint](https://docs.pytorch.org/docs/stable/distributions.html#torch.distributions.constraints.Constraint)

```python
torch.distributions.constraints.Constraint(*args, **kwargs)
```

### [paddle.distribution.constraint.Constraint](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/distribution/constraint.py#L24-L31)

```python
paddle.distribution.constraint.Constraint(*args, **kwargs)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
result = torch.distributions.constraints.Constraint().check(1)

# Paddle 写法
result = paddle.distribution.constraint.Constraint().check(1)
```
