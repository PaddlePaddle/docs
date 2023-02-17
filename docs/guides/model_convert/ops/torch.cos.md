## torch.cos
### [torch.cos](https://pytorch.org/docs/stable/generated/torch.cos.html?highlight=cos#torch.cos)

```python
torch.cos(input,
          *,
          out=None)
```

### [paddle.cos](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/cos_cn.html#cos)

```python
paddle.cos(x,
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
torch.cos([ 1.4309,  1.2706, -0.8562,  0.9796], out=y)

# Paddle 写法
y = paddle.cos([ 1.4309,  1.2706, -0.8562,  0.9796])
```
