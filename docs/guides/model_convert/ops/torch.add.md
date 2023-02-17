## torch.add
### [torch.add](https)

```python
torch.add(input,
          other,
          *,
          alpha=1,
          out=None)
```

### [paddle.add](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/add_cn.html#add)

```python
paddle.add(x,
           y,
           name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 输入的 Tensor。                                     |
| other         | y            | 输入的 Tensor。                                     |
| alpha         | -            | other 的乘数，PaddlePaddle 无此参数。                   |
| out           | -            | 表示输出的 Tensor，PaddlePaddle 无此参数。               |


### 转写示例
#### alpha：other 的乘数
```python
# Pytorch 写法
torch.add([3, 5], [2, 3], alpha=2)

# Paddle 写法
paddle.add([3, 5], 2 * [2, 3])
```

#### out：指定输出
```python
# Pytorch 写法
torch.add([3, 5], [2, 3], out=y)

# Paddle 写法
y = paddle.add([3, 5], [2, 3])
```
