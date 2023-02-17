## torch.linspace
### [torch.linspace](https://pytorch.org/docs/stable/generated/torch.linspace.html?highlight=linspace#torch.linspace)
```python
torch.linspace(start,
               end,
               steps,
               *,
               out=None,
               dtype=None,
               layout=torch.strided,
               device=None,
               requires_grad=False)
```

### [paddle.linspace](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/linspace_cn.html#linspace)
```python
paddle.linspace(start,
                stop,
                num,
                dtype=None,
                name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| end           | stop         | 区间结束的变量。               |
| steps         | num          | 给定区间内需要划分的区间数。               |
| out           | -            | 表示输出的 Tensor，PaddlePaddle 无此参数。               |
| layout        | -            | 表示布局方式，PaddlePaddle 无此参数，一般对网络训练结果影响不大，可直接删除。                   |
| device        | -            | 表示 Tensor 存放位置，PaddlePaddle 无此参数，一般对网络训练结果影响不大，可直接删除。                   |
| requires_grad | -            | 表示是否不阻断梯度传导，PaddlePaddle 无此参数。 |


### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.linspace(0, 10, 5, out=y)

# Paddle 写法
y = paddle.linspace(0, 10, 5)
```


#### requires_grad：是否需要求反向梯度，需要修改该 Tensor 的 stop_gradient 属性
```python
# Pytorch 写法
x = torch.linspace(0, 10, 5, requires_grad=True)

# Paddle 写法
x = paddle.linspace(0, 10, 5)
x.stop_gradient = False
```
