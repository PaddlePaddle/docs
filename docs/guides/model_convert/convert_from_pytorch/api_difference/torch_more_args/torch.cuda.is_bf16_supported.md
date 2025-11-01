## [ torch 参数更多  ]torch.cuda.is_bf16_supported

### [torch.cuda.is_bf16_supported](https://pytorch.org/docs/stable/generated/torch.cuda.is_bf16_supported.html#torch.cuda.is_bf16_supported)

```python
torch.cuda.is_bf16_supported(including_emulation=True)
```

### [paddle.amp.is_bfloat16_supported](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/amp/is_bfloat16_supported_cn.html#paddle/amp/is_bfloat16_supported_cn#cn-api-paddle-amp-is_bfloat16_supported)

```python
paddle.amp.is_bfloat16_supported(device=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch    | PaddlePaddle | 备注 |
| ---------- | ------------ | -- |
| including_emulation | - | 是否包含软件模拟支持，暂无转写方式。 |
| - | device | 查询的设备类型, PyTorch 无此参数，Paddle 保持默认即可。|
