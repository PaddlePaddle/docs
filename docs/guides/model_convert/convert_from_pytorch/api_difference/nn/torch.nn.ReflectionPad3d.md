## [ 参数不一致 ]torch.nn.ReflectionPad3d
### [torch.nn.ReflectionPad3d](https://pytorch.org/docs/stable/generated/torch.nn.ReflectionPad3d.html?highlight=pad#torch.nn.ReflectionPad3d)

```python
torch.nn.ReflectionPad3d(padding)
```

### [paddle.nn.Pad3D](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Pad3D_cn.html#pad3d)

```python
paddle.nn.Pad3D(padding,
                mode='constant',
                value=0.0,
                data_format='NCDHW',
                name=None)
```

其中 Paddle 与 Pytorch 的 padding 所支持的参数类型不一致，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| padding       | padding      | 填充大小，Pytorch 和 Paddle 的 padding 参数的类型分别为 (int/tuple) 和 (int/Tensor/list)。  |
| -             | mode         | padding 的四种模式，PyTorch 无此参数，Paddle 需设置为`reflect`。  |
| -             | value  | 表示填充值，PyTorch 无此参数，Paddle 保持默认即可。  |
| -             | data_format  | 输入和输出的数据格式，PyTorch 无此参数，Paddle 保持默认即可。  |


### 转写示例
#### padding：填充大小
```python
# Pytorch 写法
m = nn.ReflectionPad3d((1, 0, 1, 2, 0, 0))
m(input)

# Paddle 写法
m = nn.Pad3D([1, 0, 1, 2, 0, 0], mode='reflect')
m(input)
```
