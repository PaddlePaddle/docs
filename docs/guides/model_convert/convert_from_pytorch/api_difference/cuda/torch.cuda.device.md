## [参数不一致]torch.cuda.device

### [torch.cuda.device](https://pytorch.org/docs/1.13/generated/torch.cuda.device.html)

```python
torch.cuda.device(device)
```

### [paddle.CUDAPlace](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/CUDAPlace_cn.html#cudaplace)

```python
paddle.CUDAPlace(id)
```

Pytorch 的 `device` 与 Paddle 的 `id` 用法不同，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| device | id | torch 的 device 参数类型为 torch.device 或 int 。paddle 的 id 为 int。 torch 参数为 int 时无需转写， 参数为 torch.device 时， paddle 暂无转写方式。|
