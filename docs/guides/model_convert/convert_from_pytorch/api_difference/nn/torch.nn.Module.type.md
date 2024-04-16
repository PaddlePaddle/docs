## [参数不一致]torch.nn.Module.type

### [torch.nn.Module.type](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.type)

```python
torch.nn.Module.type(dst_type)
```

### [paddle.nn.Layer.astype](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/Layer_cn.html#astype-dtype-none)

```python
paddle.nn.Layer.astype(dtype=None)
```

两者参数用法不一致，具体如下：

### 参数映射

| PyTorch  | PaddlePaddle | 备注                                                                                    |
| -------- | ------------ | --------------------------------------------------------------------------------------- |
| dst_type | dtype        | PyTorch 为字符串或 PyTorch 数据类型，Paddle 为 字符串或 Paddle 数据类型，需要转写。 |

### 转写示例

#### dst_type 参数：数据类型

```python
# PyTorch 写法:
module = torch.nn.Module()
module.type(dst_type=torch.float32)

# Paddle 写法:
module = paddle.nn.Layer()
module.astype(dtype=paddle.float32)
```
