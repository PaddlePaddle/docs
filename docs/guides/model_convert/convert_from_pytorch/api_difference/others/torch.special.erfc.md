## [组合替代实现]torch.special.erfc

### [torch.special.erfc](https://pytorch.org/docs/stable/special.html#torch.special.erfc)

```python
torch.special.erfc(input, *, out=None)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
y = torch.special.erfc(x)

# Paddle 写法
y = 1 - paddle.erf(x)
```
