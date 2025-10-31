## [ 仅 API 调用方式不一致 ]torch.backends.cuda.is_built

### [torch.backends.cuda.is_built](https://pytorch.org/docs/stable/generated/torch.backends.cuda.html#torch.backends.cuda.is_built)

```python
torch.backends.cuda.is_built()
```

### [paddle.device.is_compiled_with_cuda](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/device/is_compiled_with_cuda_cn.html#paddle/device/is_compiled_with_cuda_cn#cn-api-paddle-device-is_compiled_with_cuda)

```python
paddle.device.is_compiled_with_cuda()
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
result = torch.backends.cuda.is_built()

# Paddle 写法
result = paddle.device.is_compiled_with_cuda()
```
