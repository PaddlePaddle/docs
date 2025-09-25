## [ 无参数 ]torch.Tensor.is_inference

### [torch.Tensor.is_inference](https://pytorch.org/docs/stable/generated/torch.Tensor.is_inference.html)

```python
torch.Tensor.is_inference()
```

### [paddle.Tensor.stop_gradient]()

```python
paddle.Tensor.stop_gradient
```

两者功能一致，无参数。 `is_inference` 会强制关闭梯度记录。并且不能在中途设置梯度，而 `stop_gradient` 仅为停止计算该算子梯度，可在中途重新设为 `True` ，`Paddle` 为近似实现。

### 转写示例

```python
# PyTorch 写法
x.is_inference()

# Paddle 写法
x.stop_gradient
```
