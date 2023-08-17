## [参数不一致]torch.nn.Module.xpu

### [torch.nn.Module.xpu](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.xpu)

```python
torch.nn.Module.xpu(device=None)
```

### [paddle.nn.Layer.to](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/Layer_cn.html#to-device-none-dtype-none-blocking-none)

```python
paddle.nn.Layer.to(device="xpu")
```

两者参数用法不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                       |
| ------- | ------------ | ---------------------------------------------------------- |
| device  | device       | PyTorch 为设备编号，Paddle 为 xpu:设备编号，需要转写。 |

### 转写示例

#### device 参数：设备

```python
# PyTorch 写法:
module = torch.nn.Module()
module.xpu(0)

# Paddle 写法:
module = paddle.nn.Layer()
module.to("xpu:0")
```
