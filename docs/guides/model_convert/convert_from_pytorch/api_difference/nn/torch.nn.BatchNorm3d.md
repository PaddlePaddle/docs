## [ 参数不一致 ]torch.nn.BatchNorm3d
### [torch.nn.BatchNorm3d](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm3d.html?highlight=torch%20nn%20batchnorm3d#torch.nn.BatchNorm3d)

```python
torch.nn.BatchNorm3d(num_features,
                     eps=1e-05,
                     momentum=0.1,
                     affine=True,
                     track_running_stats=True)
```

### [paddle.nn.BatchNorm3D](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/BatchNorm3D_cn.html#batchnorm3d)

```python
paddle.nn.BatchNorm3D(num_features,
                      momentum=0.9,
                      epsilon=1e-05,
                      weight_attr=None,
                      bias_attr=None,
                      data_format='NCDHW',
                      use_global_stats=True,
                      name=None)
```

两者功能一致但参数不一致，部分参数名不同，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| num_features           | num_features      | 表示输入 Tensor 通道数。                                     |
| eps           | epsilon      | 为了数值稳定加在分母上的值。                                     |
| momentum           | momentum      | 表示归一化函数中的超参数， PyTorch 和 Paddle 公式实现细节不一致，两者正好是相反的，需要进行转写。                                     |
| -             | weight_attr  | 指定权重参数属性的对象。如果为 False, 则表示每个通道的伸缩固定为 1，不可改变。默认值为 None，表示使用默认的权重参数属性。 |
| -             | bias_attr    | 指定偏置参数属性的对象。如果为 False, 则表示每一个通道的偏移固定为 0，不可改变。默认值为 None，表示使用默认的偏置参数属性。 |
| -             | data_format  | 指定输入数据格式， Pytorch 无此参数，Paddle 保持默认即可。 |
| affine        | -            | 是否进行反射变换， PaddlePaddle 无此参数，需要进行转写。         |
| track_running_stats | use_global_stats | 表示是否已加载的全局均值和方差。         |

### 转写示例
#### affine：是否进行反射变换
```python
affine=False 时，表示不更新：

# PyTorch 写法
m = torch.nn.BatchNorm3D(24, affine=False)

# Paddle 写法
m = paddle.nn.BatchNorm3D(24, weight_attr=False, bias_attr=False)

affine=True 时，表示更新：

# PyTorch 写法
m = torch.nn.BatchNorm3D(24)

# Paddle 写法
m = paddle.nn.BatchNorm3D(24)
```

#### momentum：
```python
# PyTorch 写法
m = torch.nn.BatchNorm3D(24, momentum=0.2)

# Paddle 写法
m = paddle.nn.BatchNorm3D(24, momentum=0.8)
```
