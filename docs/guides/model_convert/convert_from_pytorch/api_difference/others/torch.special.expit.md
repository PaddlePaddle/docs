## [组合替代实现]torch.special.expit

### [torch.special.expit](https://pytorch.org/docs/stable/special.html#torch.special.expit)

```python
torch.special.expit(input, *, out=None)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# Pytorch 写法
y = torch.special.expit(x)

# Paddle 写法
y = 1 / (1 + 1 / paddle.exp(x))
```
