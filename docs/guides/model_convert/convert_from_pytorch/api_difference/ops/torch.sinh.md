## torch.sinh
### [torch.sinh](https://pytorch.org/docs/stable/generated/torch.sinh.html?highlight=sinh#torch.sinh)

```python
torch.sinh(input,
           *,
           out=None)
```

### [paddle.sinh](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/sinh_cn.html#sinh)

```python
paddle.sinh(x,
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
torch.sinh([3, 5], out=y)

# Paddle 写法
y = paddle.sinh([3, 5])
```
