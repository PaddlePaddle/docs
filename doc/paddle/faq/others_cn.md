# 其他常见问题


##### 问题：import paddle.fluid后logging模块无法使用，如何解决？

+ 答复：操作方法可以参考[#issue17731](https://github.com/PaddlePaddle/Paddle/issues/17731)。

----------

##### 问题：使用X2paddle 从Caffe 转Paddle model时，报错 `TypeError: __new__() got an unexpected keyword argument 'serialized_options'` ，如何处理？

+ 答复：这是由于ProtoBuf版本较低导致，将protobuf升级到3.6.0即可解决。

----------

##### 问题：Windows环境下，出现"Windows not support stack backtrace yet"，如何处理？

+ 答复：Windows环境下，遇到程序报错不会详细跟踪内存报错内容。这些信息对底层开发者更有帮助，普通开发者不必关心这类警告。如果想得到完整内存追踪错误信息，可以尝试更换至Linux系统。

----------


##### 问题：报错信息C++栈如何打开？

+ 答复：在2.0正式版中，我们默认将报错信息中的C++栈进行隐藏，如果您仍需要C++栈进行debug，请设置环境变量FLAGS_call_stack_level 为2。设置方法为：
方法1、导入为环境变量： ```export FLAGS_call_stack_level=2```
方法2、执行之前声明：```FLAGS_call_stack_level=2 python xxx.py```

----------
