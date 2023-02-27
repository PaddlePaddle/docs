## torch.exp
### [torch.exp](https://pytorch.org/docs/stable/generated/torch.exp.html?highlight=exp#torch.exp)

```python
torch.exp(input,
          *,
          out=None)
```

### [paddle.exp](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/exp_cn.html#exp)

```python
paddle.exp(x,
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
torch.exp([-0.4, -0.2, 0.1, 0.3], out=y)

# Paddle 写法
y = paddle.exp([-0.4, -0.2, 0.1, 0.3])
```
