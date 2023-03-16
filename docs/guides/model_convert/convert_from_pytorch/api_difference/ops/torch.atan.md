## [torch 参数更多 ]torch.atan
### [torch.atan](https://pytorch.org/docs/stable/generated/torch.atan.html?highlight=atan#torch.atan)

```python
torch.atan(input,
           *,
           out=None)
```

### [paddle.atan](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/atan_cn.html#atan)

```python
paddle.atan(x,
            name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'>input</font>| <font color='red'>x</font> | 表示输入的 Tensor ，仅参数名不同。  |
| <font color='red'>out</font> | -  | 表示输出的 Tensor ， Paddle 无此参数，需要进行转写。    |


### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.atan([ 0.2341,  0.2539, -0.6256, -0.6448], out=y)

# Paddle 写法
y = paddle.atan([ 0.2341,  0.2539, -0.6256, -0.6448])
```
