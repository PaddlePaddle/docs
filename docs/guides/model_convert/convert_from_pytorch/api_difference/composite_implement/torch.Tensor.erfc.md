## [ 组合替代实现 ]torch.Tensor.erfc

### [torch.Tensor.erfc](https://pytorch.org/docs/stable/generated/torch.Tensor.erfc.html)

```python
torch.Tensor.erfc()
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
y = x.erfc()

# Paddle 写法
y = 1 - x.erf()
```
