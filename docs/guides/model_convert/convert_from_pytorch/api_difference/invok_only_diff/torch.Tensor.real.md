## [ 仅 API 调用方式不一致 ]torch.Tensor.real

### [torch.Tensor.real](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.real.html#torch.Tensor.real)

```python
torch.Tensor.real
```

### [paddle.Tensor.real](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor__upper_cn.html#real-name-none)

```python
paddle.Tensor.real(name=None)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
result = src.real

# Paddle 写法
result = src.real()
```
