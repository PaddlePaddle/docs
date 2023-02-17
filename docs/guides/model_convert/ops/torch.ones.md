## torch.ones
### [torch.ones](https://pytorch.org/docs/stable/generated/torch.ones.html?highlight=ones#torch.ones)

```python
torch.ones(*size,
           *,
           out=None,
           dtype=None,
           layout=torch.strided,
           device=None,
           requires_grad=False)
```

### [paddle.ones](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/ones_cn.html#ones)

```python
paddle.ones(shape,
             dtype=None,
             name=None)
```

### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| *size         | shape        | 表示输出形状大小。                                     |
| out           | -            | 表示输出的 Tensor，PaddlePaddle 无此参数。               |
| layout        | -            | 表示布局方式，PaddlePaddle 无此参数，一般对网络训练结果影响不大，可直接删除。                   |
| device        | -            | 表示 Tensor 存放位置，PaddlePaddle 无此参数，一般对网络训练结果影响不大，可直接删除。                   |
| requires_grad | -            | 表示是否不阻断梯度传导，PaddlePaddle 无此参数。 |


### 功能差异

#### 使用方式
***PyTorch***：生成 Tensor 的形状大小以可变参数、list 或 tuple 的方式传入。
***PaddlePaddle***：生成 Tensor 的形状大小以 list 或 tuple 的方式传入。


### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.full([3, 5], 1., out=y)

# Paddle 写法
y = paddle.full([3, 5], 1.)
```


#### requires_grad：是否需要求反向梯度，需要修改该 Tensor 的 stop_gradient 属性
```python
# Pytorch 写法
x = torch.full([3, 5], 1., requires_grad=True)

# Paddle 写法
x = paddle.full([3, 5], 1.)
x.stop_gradient = False
```
