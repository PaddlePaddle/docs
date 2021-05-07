# 飞桨框架NPU版安装说明


## 安装方式：通过release/2.1分支源码编译安装

```bash
# cmake命令
cmake .. -DPY_VERSION=3 -DWITH_ASCEND=OFF -DPYTHON_EXECUTABLE=`which python3`  -DPYTHON_INCLUDE_DIR=`python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())"` -DWITH_ARM=ON -DWITH_ASCEND_CL=ON -DWITH_TESTING=ON -DCMAKE_BUILD_TYPE=Release -DON_INFER=ON -DWITH_XBYAK=OFF
# make
```
