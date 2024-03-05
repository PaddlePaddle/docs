## [ 参数不一致 ]torch.nn.ReflectionPad2d
### [torch.nn.ReflectionPad2d](https://pytorch.org/docs/stable/generated/torch.nn.ReflectionPad2d.html?highlight=pad#torch.nn.ReflectionPad2d)

```python
torch.nn.ReflectionPad2d(padding)
```

### [paddle.nn.Pad2D](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/Pad2D_cn.html#pad2d)

```python
paddle.nn.Pad2D(padding,
                mode='constant',
                value=0.0,
                data_format='NCHW',
                name=None)
```

其中 Paddle 与 PyTorch 的 padding 所支持的参数类型不一致，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| padding       | padding      | 填充大小，PyTorch 和 Paddle 的 padding 参数的类型分别为 (int/tuple) 和 (int/Tensor/list)。  |
| -             | mode         | padding 的四种模式，PyTorch 无此参数，Paddle 需设置为`reflect`。  |
| -             | value  | 表示填充值，PyTorch 无此参数，Paddle 保持默认即可。  |
| -             | data_format  | 输入和输出的数据格式，PyTorch 无此参数，Paddle 保持默认即可。  |


### 转写示例
#### padding：填充大小
```python
# PyTorch 写法
m = nn.ReflectionPad2d((1, 0, 1, 2))
m(input)

# Paddle 写法
m = nn.Pad2D([1, 0, 1, 2], mode='reflect')
m(input)
```
