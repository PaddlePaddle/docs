## torch.arange

### [torch.arange](https://pytorch.org/docs/stable/generated/torch.arange.html?highlight=arange#torch.arange)

```python
torch.arange(start=0,
             end,
             step=1,
             *,
             out=None,
             dtype=None,
             layout=torch.strided,
             device=None,
             requires_grad=False)
```

### [paddle.arange](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/arange_cn.html#arange)

```python
paddle.arange(start=0,
              end=None,
              step=1,
              dtype=None,
              name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| out           | -            | 表示输出的 Tensor，PaddlePaddle 无此参数。               |
| layout        | -            | 表示布局方式，PaddlePaddle 无此参数，一般对网络训练结果影响不大，可直接删除。                   |
| device        | -            | 表示 Tensor 存放位置，PaddlePaddle 无此参数，一般对网络训练结果影响不大，可直接删除。                   |
| requires_grad | -            | 表示是否不阻断梯度传导，PaddlePaddle 无此参数。 |


### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.arange(5, out=y)

# Paddle 写法
y = paddle.arange(5)
```


#### requires_grad：是否需要求反向梯度，需要修改该 Tensor 的 stop_gradient 属性
```python
# Pytorch 写法
x = torch.arange(5, requires_grad=True)

# Paddle 写法
x = paddle.arange(5)
x.stop_gradient = False
```
