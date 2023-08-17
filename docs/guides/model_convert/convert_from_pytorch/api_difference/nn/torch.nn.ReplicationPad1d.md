## [ 参数不一致 ]torch.nn.ReplicationPad1d
### [torch.nn.ReplicationPad1d](https://pytorch.org/docs/stable/generated/torch.nn.ReplicationPad1d.html?highlight=pad#torch.nn.ReplicationPad1d)

```python
torch.nn.ReplicationPad1d(padding)
```

### [paddle.nn.Pad1D](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/Pad1D_cn.html#pad1d)

```python
paddle.nn.Pad1D(padding,
                mode='constant',
                value=0.0,
                data_format='NCL',
                name=None)
```

其中 Paddle 与 Pytorch 的 padding 所支持的参数类型不一致，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| padding       | padding      | 填充大小，Pytorch 和 Paddle 的 padding 参数的类型分别为 (int/tuple) 和 (int/Tensor/list)。  |
| -             | mode         | padding 的四种模式，PyTorch 无此参数，Paddle 需设置为`replicate`。  |
| -             | value  | 表示填充值，PyTorch 无此参数，Paddle 保持默认即可。  |
| -             | data_format  | 输入和输出的数据格式，PyTorch 无此参数，Paddle 保持默认即可。  |


### 转写示例
#### padding：填充大小
```python
# Pytorch 写法
m = nn.ReplicationPad1d((3, 1))
m(input)

# Paddle 写法
pad = paddle.to_tensor((3, 1))
m = nn.Pad1D(pad, mode='replicate')
m(input)
```
