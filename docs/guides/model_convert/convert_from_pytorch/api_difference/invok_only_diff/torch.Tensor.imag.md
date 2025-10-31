## [ 仅 API 调用方式不一致 ]torch.Tensor.imag

### [torch.Tensor.imag](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.imag)

```python
torch.Tensor.imag
```

### [paddle.Tensor.imag](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor/imag_cn.html#paddle/Tensor/imag_cn#cn-api-paddle-Tensor-imag)

```python
paddle.Tensor.imag(name=None)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
result = src.imag

# Paddle 写法
result = src.imag()
```
