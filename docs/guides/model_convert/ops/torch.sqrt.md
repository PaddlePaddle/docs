## torch.sqrt
### [torch.sqrt](https://pytorch.org/docs/stable/generated/torch.sqrt.html?highlight=sqrt#torch.sqrt)

```python
torch.sqrt(input,
            *,
            out=None)
```

### [paddle.sqrt](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/sqrt_cn.html#sqrt)

```python
paddle.sqrt(x,
            name=None)
```
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 输入的 Tensor。                   |
| out           | -            | 表示输出的 Tensor，PaddlePaddle 无此参数。               |


### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.full([3, 5], 1., out=y)

# Paddle 写法
y = paddle.full([3, 5], 1.)
```
