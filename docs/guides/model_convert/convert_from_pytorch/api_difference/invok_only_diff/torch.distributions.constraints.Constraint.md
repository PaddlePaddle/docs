## [ 仅 API 调用方式不一致 ]torch.distributions.constraints.Constraint

### [torch.distributions.constraints.Constraint](https://pytorch.org/docs/stable/generated/torch.distributions.constraints.Constraint.html#torch.distributions.constraints.Constraint)

```python
torch.distributions.constraints.Constraint(*args, **kwargs)
```

### [paddle.distribution.constraint.Constraint](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distribution/constraint/Constraint_cn.html#paddle/distribution/constraint/Constraint_cn#cn-api-paddle-distribution-constraint-Constraint)

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
