## [ 仅 API 调用方式不一致 ]torch.cuda.is_bf16_supported

### [torch.cuda.is_bf16_supported](https://pytorch.org/docs/stable/generated/torch.cuda.is_bf16_supported.html#torch.cuda.is_bf16_supported)

```python
torch.cuda.is_bf16_supported(including_emulation=True)
```

### [paddle.amp.is_bfloat16_supported](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/amp/is_bfloat16_supported_cn.html#paddle/amp/is_bfloat16_supported_cn#cn-api-paddle-amp-is_bfloat16_supported)

```python
paddle.amp.is_bfloat16_supported(device=None)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
result = torch.cuda.is_bf16_supported()

# Paddle 写法
result = paddle.amp.is_bfloat16_supported()
```
