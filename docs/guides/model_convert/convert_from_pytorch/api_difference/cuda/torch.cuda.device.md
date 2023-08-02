## [参数不一致]torch.cuda.device

### [torch.cuda.device](https://pytorch.org/docs/stable/generated/torch.cuda.device.html#torch.cuda.device)

```python
torch.cuda.device(device)
```

### [paddle.CUDAPlace](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/CUDAPlace_cn.html)

```python
paddle.CUDAPlace(id)
```

其中 Pytorch 与 Paddle 的参数支持类型不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                                             |
| ------- | ------------ | -------------------------------------------------------------------------------- |
| device  | id           | GPU 的设备 ID, Pytorch 支持 torch.device 和 int，Paddle 支持 int，需要转写。 |

### 转写示例

#### device: 获取 device 参数，对其取 device.index 值

```python
# Pytorch 写法
torch.cuda.device(torch.device('cuda'))

# Paddle 写法
paddle.CUDAPlace(0)

# 增加 index
# Pytorch 写法
torch.cuda.device(torch.device('cuda', index=index))

# Paddle 写法
paddle.CUDAPlace(index)
```
