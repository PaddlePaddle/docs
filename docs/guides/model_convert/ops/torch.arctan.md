## torch.arctan
### [torch.arctan](https://pytorch.org/docs/stable/generated/torch.arctan.html?highlight=arctan#torch.arctan)

```python
torch.arctan(input,
             *,
             out=None)
```

### [paddle.atan](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/atan_cn.html#atan)

```python
paddle.atan(x,
            name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 输入的 Tensor。                                      |
| out           | -            | 表示输出的 Tensor，PaddlePaddle 无此参数。               |


### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.arctan([ 0.2341,  0.2539, -0.6256, -0.6448], out=y)

# Paddle 写法
y = paddle.atan([ 0.2341,  0.2539, -0.6256, -0.6448])
```
