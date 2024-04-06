## [参数完全一致]torch.cuda.stream

### [torch.cuda.stream](https://pytorch.org/docs/stable/generated/torch.cuda.stream.html)

```python
torch.cuda.stream(stream)
```

### [paddle.device.cuda.stream_guard](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/device/cuda/stream_guard_cn.html)

```python
paddle.device.cuda.stream_guard(stream)
```

功能一致，参数完全一致，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| stream        | stream            | 指定的 CUDA stream。如果为 None，则不进行 stream 流切换。Paddle 的该 API 目前仅支持动态图模式。                                    |
