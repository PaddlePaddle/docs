# torch.nn.PReLU
### [torch.nn.PReLU](https://pytorch.org/docs/1.13/generated/torch.nn.PReLU.html?highlight=prelu#torch.nn.PReLU)

```python
torch.nn.PReLU(num_parameters=1,
               init=0.25,
               device=None,
               dtype=None)
```

### [paddle.nn.PReLU](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/PReLU_cn.html#prelu)

```python
paddle.nn.PReLU(num_parameters=1,
                init=0.25,
                weight_attr=None,
                data_format='NCHW',
                name=None)
```

其中 Pytorch 的 count_include_pad 与 Paddle 的 exclusive 用法不一致，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| device        | -            | 指定设备，PaddlePaddle 无此功能，一般对网络训练结果影响不大，可直接删除。  |
| dtype         | -            | 指定数据类型，PaddlePaddle 无此功能。  |
| -             | weight_attr  | 指定权重参数属性的对象，Pytorch 无此参数，Paddle 保持默认即可。  |
| -             | data_format  | 指定输入的数据格式，PyTorch 无此参数，Paddle 保持默认即可。  |
