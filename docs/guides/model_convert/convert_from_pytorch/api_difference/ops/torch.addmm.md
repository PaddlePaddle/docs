## [仅 torch 参数更多]torch.addmm

### [torch.addmm](https://pytorch.org/docs/stable/generated/torch.addmm.html?highlight=addmm#torch.addmm)

```python
torch.addmm(input,mat1,mat2,*,beta=1,alpha=1,out=None)
```

### [paddle.addmm](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/addmm_cn.html)

```python
paddle.addmm(input,x,y,alpha=1.0,beta=1.0,name=None)
```

其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：

### 参数映射
|PyTorch|PaddlePaddle|备注|
| ------- | ------- | ------- |
|input|input|输入矩阵|
|mat1|x|矩阵x|
|mat2|y|矩阵y|
|beta|beta|乘以input的标量|
|alpha|alpha|乘以x*y的标量|
|out||输出矩阵，Paddle没有此字段，需要进行转写|

### 转写示例

```python
# Pytorch 写法
>>> mat1 = torch.ones(2, 2)
>>> mat2 = torch.ones(2, 2)
>>> input = torch.ones(2,2)
>>> torch.addmm(input, mat1, mat2, beta=0.5, alpha=5.0, out=y)
# [[10.5 10.5]
# [10.5 10.5]]

# Paddle 写法
 
import paddle

x = paddle.ones([2,2])
y = paddle.ones([2,2])
input = paddle.ones([2,2])

out = paddle.addmm( input=input, x=x, y=y, beta=0.5, alpha=5.0 )

print(out)
# [[10.5 10.5]
# [10.5 10.5]]

```
