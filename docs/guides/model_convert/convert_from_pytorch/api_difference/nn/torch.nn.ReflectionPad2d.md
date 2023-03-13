## torch.nn.ReflectionPad2d
### [torch.nn.ReflectionPad2d](https://pytorch.org/docs/stable/generated/torch.nn.ReflectionPad2d.html?highlight=pad#torch.nn.ReflectionPad2d)

```python
torch.nn.ReflectionPad2d(padding)
```

### [paddle.nn.Pad2D](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Pad2D_cn.html#pad2d)

```python
paddle.nn.Pad2D(padding,
                mode='constant',
                value=0.0,
                data_format='NCHW',
                name=None)
```

其中 Paddle 相比 Pytorch 支持更多其他参数，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| padding       | padding      | 填充大小，Pytorch 和 Paddle 的 padding 参数的类型分别为 (int/tuple) 和 (int/Tensor/list)。  |
| -             | mode         | padding 的四种模式，PyTorch 无此参数，Paddle 需设置为`reflect`。  |
| -             | data_format  | 输入和输出的数据格式，PyTorch 无此参数，Paddle 保持默认即可。  |


### 转写示例
#### padding：填充大小
```python
# Pytorch 写法
m = nn.ReflectionPad2d((3, 1), 3.5)
m(input)

# Paddle 写法
pad = paddle.to_tensor((3, 1))
m = nn.Pad2D(pad, mode='reflect', value=3.5)
m(input)
```
