## [ 参数不一致 ]torch.Tensor.all

### [torch.Tensor.all](https://pytorch.org/docs/1.13/generated/torch.Tensor.all.html?highlight=torch+tensor+all#torch.Tensor.all)

```python
torch.Tensor.all(dim=None, keepdim=False)
```

### [paddle.all](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Tensor_cn.html#all-axis-none-keepdim-false-name-none)

```python
paddle.Tensor.all(axis=None,
           keepdim=False,
           name=None)
```

其中 Paddle 与 Pytorch 运算 Tensor 所支持的类型不一致，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| 运算 Tensor        | 运算 Tensor            | PyTorch 支持布尔和数值类型的输入，Paddle 仅支持布尔类型，需要进行转写。                   |
| -             | <font color='red'> axis </font>         | 计算逻辑与运算的维度，Pytorch 无，保持默认即可。               |
| -             | <font color='red'> keepdim </font>      | 是否在输出 Tensor 中保留减小的维度，Pytorch 无，保持默认即可。  |

### 转写示例
```python
# PyTorch 写法
y = x.all()

# Paddle 写法
y = x.astype('bool').all()
```
