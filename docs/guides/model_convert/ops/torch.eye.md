## torch.eye

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
| n             | num_rows     | 生成 2-D Tensor 的行数。               |
| m             | num_columns  | 生成 2-D Tensor 的列数。                   |
| out           | -            | 表示输出的 Tensor，PaddlePaddle 无此参数。               |
| layout        | -            | 表示布局方式，PaddlePaddle 无此参数，一般对网络训练结果影响不大，可直接删除。                   |
| device        | -            | 表示 Tensor 存放位置，PaddlePaddle 无此参数，一般对网络训练结果影响不大，可直接删除。                   |
| requires_grad | -            | 表示是否不阻断梯度传导，PaddlePaddle 无此参数。 |


### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.eye(3, out=y)

# Paddle 写法
y = paddle.eye(3)
```


#### requires_grad：是否需要求反向梯度，需要修改该 Tensor 的 stop_gradient 属性
```python
# Pytorch 写法
x = torch.eye(3, requires_grad=True)

# Paddle 写法
x = paddle.eye(3)
x.stop_gradient = False
```
