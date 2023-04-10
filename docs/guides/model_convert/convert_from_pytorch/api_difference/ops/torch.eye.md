## [torch 参数更多 ]torch.eye

### [torch.eye](https://pytorch.org/docs/stable/generated/torch.eye.html?highlight=eye#torch.eye)
```python
torch.eye(n,
          m=None,
          *,
          out=None,
          dtype=None,
          layout=torch.strided,
          device=None,
          requires_grad=False)
```

### [paddle.eye](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/eye_cn.html#eye)
```python
paddle.eye(num_rows,
           num_columns=None,
           dtype=None,
           name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'> n </font>           | <font color='red'> num_rows </font>     | 生成 2-D Tensor 的行数，仅参数名不一致。               |
| <font color='red'> m  </font>           | <font color='red'> num_columns </font> | 生成 2-D Tensor 的列数， 仅参数名不一致。                   |
| <font color='red'> out </font> | -  | 表示输出的 Tensor ， Paddle 无此参数，需要进行转写。    |
| <font color='red'> layout </font> | -       | 表示布局方式， Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。  |
| <font color='red'> device </font>     | -       | 表示 Tensor 存放设备位置，Paddle 无此参数，需要进行转写。 |
| <font color='red'> requires_grad </font> | -       | 表示是否计算梯度， Paddle 无此参数，需要进行转写。 |


### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.eye(3, out=y)

# Paddle 写法
paddle.assign(paddle.eye(3), y)
```
#### device: Tensor 的设备
```python
# Pytorch 写法
torch.eye(x, device=torch.device('cpu'))

# Paddle 写法
y = paddle.eye(x)
y.cpu()
```

#### requires_grad：是否需要求反向梯度，需要修改该 Tensor 的 stop_gradient 属性
```python
# Pytorch 写法
x = torch.eye(3, requires_grad=True)

# Paddle 写法
x = paddle.eye(3)
x.stop_gradient = False
```
