## [ 仅参数名不一致 ]torch.ceil

### [torch.ceil](https://pytorch.org/docs/stable/generated/torch.ceil.html?highlight=ceil#torch.ceil)

```python
torch.ceil(input,
           *,
           out=None)
```

### [paddle.ceil](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/ceil_cn.html#ceil)

```python
paddle.ceil(x,
            name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'>input</font>| <font color='red'>x</font> | 表示输入的 Tensor ，仅参数名不一致。  |
| <font color='red'>out</font> | -  | 表示输出的 Tensor ， Paddle 无此参数，需要进行转写。    |

### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.ceil([-0.4, -0.2, 0.1, 0.3], out=y)

# Paddle 写法
paddle.assign(paddle.ceil([-0.4, -0.2, 0.1, 0.3]), y)
```
