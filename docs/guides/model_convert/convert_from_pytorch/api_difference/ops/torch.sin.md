## [torch 参数更多 ]torch.sin
### [torch.sin](https://pytorch.org/docs/stable/generated/torch.sin.html?highlight=sin#torch.sin)

```python
torch.sin(input,
          *,
          out=None)
```

### [paddle.sin](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/sin_cn.html#sin)

```python
paddle.sin(x,
           name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 输入的 Tensor ，仅参数名不一致。                                      |
| out           | -            | 表示输出的 Tensor，PaddlePaddle 无此参数，需要进行转写。               |


### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.sin([3, 5], out=y)

# Paddle 写法
y = paddle.sin([3, 5])
```
