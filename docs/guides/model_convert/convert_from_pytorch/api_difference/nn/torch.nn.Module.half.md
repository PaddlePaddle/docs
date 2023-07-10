## [参数不一致]torch.nn.Module.half

### [torch.nn.Module.half](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.half)

```python
torch.nn.Module.half()
```

### [paddle.nn.Layer.to](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Layer_cn.html#to-device-none-dtype-none-blocking-none)

```python
paddle.nn.Layer.to(dtype="float16")
```

两者参数用法不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                              |
| ------- | ------------ | ------------------------------------------------- |
| -       | dtype        | 转换的数据类型，Paddle 为 float16，需要进行转写。 |

### 转写示例

#### dtype 参数：转换的数据类型

```python
# PyTorch 写法:
module = torch.nn.Module()
module.float()

# Paddle 写法:
module = paddle.nn.Layer()
module.to(dtype="float16")
```
