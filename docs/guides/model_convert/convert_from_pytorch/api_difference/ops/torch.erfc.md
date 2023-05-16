## [ 组合替代实现 ]torch.erfc

### [torch.erfc](https://pytorch.org/docs/stable/generated/torch.erfc.html#torch.erfc)

```python
torch.erfc(input, *, out=None)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# Pytorch 写法
y = torch.erfc(a)

# Paddle 写法
y = 1 - paddle.erf(a)
```
