## [ 组合替代实现 ]torch.chain_matmul

### [torch.chain_matmul](https://pytorch.org/docs/stable/generated/torch.chain_matmul.html?highlight=chain_matmul#torch.chain_matmul)
```python
torch.chain_matmul(*matrices, out=None)
```
Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
y = torch.chain_matmul(a, b, c)

# Paddle 写法
y = a @ b @ c
```
