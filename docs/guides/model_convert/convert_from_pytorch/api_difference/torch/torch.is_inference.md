## [ 组合替代实现 ]torch.is_inference

### [torch.is_inference]()

```python
torch.is_inference(input)
```

Paddle 无此 API，需要组合是实现。 `is_inference` 会强制关闭梯度记录。并且不能在中途设置梯度，`Paddle` 为近似实现。

### 转写示例

```python
# PyTorch 写法
torch.is_inference(x)

# Paddle 写法
not x.stop_gradient
```