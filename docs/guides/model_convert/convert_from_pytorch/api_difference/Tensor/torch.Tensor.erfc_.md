## [ 组合替代实现 ]torch.Tensor.erfc_

### [torch.Tensor.erfc_](https://pytorch.org/docs/stable/generated/torch.Tensor.erfc_.html)

```python
torch.Tensor.erfc_()
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
x.erfc_()

# Paddle 写法
paddle.assign(1 - x.erf(), x)
```
