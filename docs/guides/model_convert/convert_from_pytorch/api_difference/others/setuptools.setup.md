## [ 组合替代实现 ]setuptools.setup

### [setuptools.setup](https://setuptools.pypa.io/en/latest/userguide/quickstart.html)

```python
setuptools.setup(*args, **attrs)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
setuptools.setup(*args, **attrs)

# Paddle 写法
if "cmdclass" in attrs:
    if "paddle.utils.cpp_extension.BuildExtension" in attrs["cmdclass"]:
        paddle.utils.cpp_extension.setup(attrs)
```
