# CAPI tools
CAPI tools 用于一键生成 C++ 的 rst 文档。

## 调用方式
```python
python main.py <source dir> <target dir>
```

若不设置`source dir`和`target dir`，则默认先查找已安装的`paddlepaddle`包环境。

其中：
- source dir 是安装后的 Paddle C++ API 声明路径。 例如`venv/Lib/site-packages/paddle/include/paddle`。
- target dir 目标文件保存路径。

最终生成结果如下所示：
```python
target dir
| -cn
    |- index.rst
    |- Paddle
        |- fluid
        |- phi
        |- ...
| -en
    |- index.rst
    |- Paddle
        |- fluid
        |- phi
        |- ...
```

## 获取最新 PaddlePaddle
pip install python -m pip install paddlepaddle==0.0.0 -f https://www.paddlepaddle.org.cn/whl/windows/cpu-mkl-avx/develop.html

## 代码结构

### `main.py`文件主要用于处理和筛选包文件, 并调用`utils_helper.py`中的函数进行文件生成
```python
def analysis_file() # 用于解析文件内容(多线程不安全)

def generate_docs() # 用于创建目录并传值给 utils_helper.py 中的函数进行文件生成

def cpp2py() # 用于筛选出 cpp api 和 py api 相对应的函数名称
```

### `utils_helper.py`文件主要存放函数生成、解析, 以及文件写入的工作
```python

class func_helper(object) # 用于生成和解析方法
    decode() # 用于解析输出输出参数、函数名称、返回值、函数注释信息
class class_helper(object) # 用于生成和解析类
    decode() # 同 func_helper()

def generate_overview() # 用于生成 overview.rst 文件
```
