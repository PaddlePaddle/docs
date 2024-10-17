## [ 无参数 ]torch.is_inference

### [torch.is_inference]()

```python
torch.is_inference(input)
```

两者功能一致，无参数。 `is_inference` 会强制关闭梯度记录。并且不能在中途设置梯度，而 `stop_gradient` 仅为停止计算该算子梯度，可在中途重新设为 `True` ，`Paddle` 为近似实现。

### 转写示例

```python
# PyTorch 写法
torch.is_inference(x)

# Paddle 写法
not x.stop_gradient
```
