## [torch 参数更多]torch.Tensor.cuda

### [torch.Tensor.cuda](https://pytorch.org/docs/stable/generated/torch.Tensor.cuda.html#torch.Tensor.cuda)

```python
torch.Tensor.cuda(device=None, non_blocking=False, memory_format=torch.preserve_format)
```

### [paddle.Tensor.cuda](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#cuda-device-id-none-blocking-false)

```python
paddle.Tensor.cuda(device_id=None, blocking=False)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                                     |
| ------------- | ------------ | ------------------------------------------------------------------------ |
| device        | device_id    | 目标 GPU 设备，仅参数名不一致。                                          |
| non_blocking  | blocking     | 是否同步或异步拷贝，PyTorch 和 Paddle 取值相反，需要转写。                                           |
| memory_format | -            | 表示内存格式， Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
