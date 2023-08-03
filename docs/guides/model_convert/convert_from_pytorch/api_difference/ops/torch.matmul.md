## [torch 参数更多 ]torch.matmul
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

Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 表示输入的第一个 Tensor ，仅参数名不一致。               |
| other         | y            | 表示输入的第二个 Tensor ，仅参数名不一致。             |
| -             | transpose_x  | 表示相乘前是否转置 x，PyTorch 无此参数，Paddle 保持默认即可。               |
| -             | transpose_y  | 表示相乘前是否转置 y，PyTorch 无此参数，Paddle 保持默认即可。             |
| out           | -            | 表示输出的 Tensor，Paddle 无此参数，需要转写。  |


### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.matmul(a, b, out=y)

# Paddle 写法
paddle.assign(paddle.matmul(a, b), y)
```
