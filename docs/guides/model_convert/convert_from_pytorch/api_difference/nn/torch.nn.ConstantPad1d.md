## torch.nn.ConstantPad1d
### [torch.nn.ConstantPad1d](https://pytorch.org/docs/1.13/generated/torch.nn.ConstantPad1d.html?highlight=constantpad1d#torch.nn.ConstantPad1d)

```python
torch.nn.ConstantPad1d(padding,
                       value)
```

### [paddle.nn.Pad1D](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Pad1D_cn.html#pad1d)

```python
paddle.nn.Pad1D(padding,
                mode='constant',
                value=0.0,
                data_format='NCL',
                name=None)
```

其中 Paddle 相比 Pytorch 支持更多其他参数，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| padding       | padding      | 填充大小，Pytorch 和 Paddle 的 padding 参数的类型分别为 (int/tuple) 和 (int/Tensor/list)。  |
| -             | mode         | padding 的四种模式，PyTorch 无此参数，Paddle 保持默认即可。  |
| -             | data_format  | 输入和输出的数据格式，PyTorch 无此参数，Paddle 保持默认即可。  |

### 转写示例
#### padding：填充大小
```python
# Pytorch 写法
m = nn.ConstantPad1d((3, 1), 3.5)
m(input)

# Paddle 写法
pad = paddle.to_tensor((3, 1))
m = nn.Pad1D(pad, value=3.5)
m(input)
```
