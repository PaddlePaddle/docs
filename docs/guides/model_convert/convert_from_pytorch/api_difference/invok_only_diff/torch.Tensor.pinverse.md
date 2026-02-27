## [ 仅 API 调用方式不一致 ]torch.Tensor.pinverse

### [torch.Tensor.pinverse](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.pinverse.html#torch.Tensor.pinverse)

```python
torch.Tensor.pinverse()
```

### [paddle.Tensor.pinv](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor/pinv_cn.html#paddle/Tensor/pinv_cn#cn-api-paddle-Tensor-pinv)

```python
paddle.Tensor.pinv(rcond=1e-15, hermitian=False, name=None)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
result = a.pinverse()

# Paddle 写法
result = a.pinv()
```
