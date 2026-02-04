## [ 仅 API 调用方式不一致 ]torch.Tensor.movedim

### [torch.Tensor.movedim](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.movedim.html#torch.Tensor.movedim)

```python
torch.Tensor.movedim(source, destination)
```

### [paddle.Tensor.moveaxis](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor/moveaxis_cn.html#paddle/Tensor/moveaxis_cn#cn-api-paddle-Tensor-moveaxis)

```python
paddle.Tensor.moveaxis(source, destination, name=None)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
result = x.movedim(1, 0)

# Paddle 写法
result = x.moveaxis(1, 0)
```
