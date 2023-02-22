## torch.matmul
### [torch.matmul](https://pytorch.org/docs/stable/generated/torch.matmul.html?highlight=matmul#torch.matmul)
```python
torch.matmul(input,
             other,
             out=None)
```

### [paddle.matmul](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/matmul_cn.html)
```python
paddle.matmul(x,
              y,
              transpose_x=False,
              transpose_y=False,
              name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 表示输入的第一个 Tensor。               |
| other         | y            | 表示输入的第二个 Tensor。             |
| -             | transpose_x  | 表示相乘前是否转置 x，PyTorch 无此参数，Paddle 保持默认即可。               |
| -             | transpose_y  | 表示相乘前是否转置 y，PyTorch 无此参数，Paddle 保持默认即可。             |
| out           | -            | 表示输出的 Tensor，PaddlePaddle 无此参数。  |


### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.matmul(a, b, out=y)

# Paddle 写法
y = paddle.matmul(a, b)
```
