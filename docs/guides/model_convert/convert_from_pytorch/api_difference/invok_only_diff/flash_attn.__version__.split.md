## [ 仅 API 调用方式不一致 ]flash_attn._\_version__.split
### [flash_attn._\_version__.split](https://github.com/Dao-AILab/flash-attention/blob/72e27c6320555a37a83338178caa25a388e46121/flash_attn/__init__.py)
```python
flash_attn.__version__.split(sep=None, maxsplit=-1)
```

### [paddle._\_version__.split](https://github.com/PaddlePaddle/Paddle/tree/develop)
```python
paddle.__version__.split(sep=None, maxsplit=-1)
```

### 转写示例

```python
# PyTorch 写法
version = flash_attn.__version__

# Paddle 写法
version = paddle.__version__
```
