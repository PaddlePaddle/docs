# 自定义硬件接入教程

本教程介绍如何为 PaddlePaddle 实现一个 CustomDevice 插件，添加一个名为 CustomCPU 的新硬件后端，并进行编译，打包，安装和使用。

> 注意：
> - 请确保已经正确安装了[飞桨develop](https://github.com/PaddlePaddle/Paddle)最新版本
> - 当前仅支持 `Linux`平台
> - 支持飞桨已通过头文件开放函数式声明的Kernel自定义编码与注册

## 自定义 Runtime

**InitPlugin**

InitPlugin 作为自定义 Runtime 的入口函数，插件需要实现该函数，并在该函数中检查参数，填充硬件信息，注册 Runtime API 。 PaddlePaddle 初始化时加载插件并调用 InitPlugin 完成插件初始化，注册 Runtime（整个过程由框架自动完成，只要动态链接库位于 site-packages/paddle-plugins/ 或 CUSTOM_DEVICE_ROOT 环境变量指定目录即可）。

例子：

```c++
#include "paddle/phi/backends/device_ext.h"

void InitPlugin(CustomRuntimeParams *params) {
  // 将检查版本兼容性并填充插件使用的自定义 Runtime 版本信息
  PADDLE_CUSTOM_RUNTIME_CHECK_VERSION(params);

  // 填充 Runtime 基本信息
  params->device_type = "CustomCPU";
  params->sub_device_type = "V1";

  // 注册 Runtime API
  params->interface->set_device = set_device;
  params->interface->get_device = get_device;
  params->interface->create_stream = create_stream;
  params->interface->destroy_stream = destroy_stream;
  params->interface->create_event = create_event;
  params->interface->destroy_event = destroy_event;
  params->interface->record_event = record_event;
  params->interface->synchronize_device = sync_device;
  params->interface->synchronize_stream = sync_stream;
  params->interface->synchronize_event = sync_event;
  params->interface->stream_wait_event = stream_wait_event;
  params->interface->memory_copy_h2d = memory_copy;
  params->interface->memory_copy_d2d = memory_copy;
  params->interface->memory_copy_d2h = memory_copy;
  params->interface->device_memory_allocate = allocate;
  params->interface->device_memory_deallocate = deallocate;
  params->interface->get_device_count = get_device_cout;
  params->interface->get_device_list = get_device_list;
  params->interface->device_memory_stats = memstats;
  params->interface->device_min_chunk_size = get_min_chunk_size;
}
```

插件首先需要检查 InitPlugin 的参数，框架设置该参数成员 size 为其类型的大小并传入 InitPlugin ， CustomRuntimeParams 与 C_DeviceInterface 的类型定义详见 [device_ext.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/backends/device_ext.h)。

然后，插件需要填充插件的基本信息以及版本号，以供 PaddlePaddle 管理插件以及检查版本兼容性。

- params->size 和 params->interface.size ： 自定义 Runtime 的后续版本中，会保证 size 和 interface 为 CustomRuntimeParams 类型的前两个成员。
- params->version ： 插件填充版本信息，其版本号在 [device_ext.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/backends/device_ext.h) 中定义， PaddlePaddle 在注册自定义 Runtime 时检查版本兼容性。
- params->device_type ： 硬件后端名，具有同名的插件已经注册时，则不会注册 Runtime 。
- params->sub_device_type ： 硬件后端子类型名。

最后，插件需要填充 params->interface 中的回调接口（至少实现 Required 接口，否则 Runtime 不会被注册），完成自定义 Runtime 的初始化。具体API的说明详见[自定义 Runtime 文档](./custom_runtime_cn.html)。

```c++
static size_t global_total_mem_size = 1 * 1024 * 1024 * 1024UL;
static size_t global_free_mem_size = global_total_mem_size;

C_Status set_device(const C_Device device) {
  return C_SUCCESS;
}

C_Status get_device(const C_Device device) {
  device->id = 0;
  return C_SUCCESS;
}

C_Status get_device_count(size_t *count) {
  *count = 1;
  return C_SUCCESS;
}

C_Status get_device_list(size_t *device) {
  *device = 0;
  return C_SUCCESS;
}

C_Status memory_copy(const C_Device device, void *dst, const void *src, size_t size) {
  memcpy(dst, src, size);
  return C_SUCCESS;
}

C_Status allocate(const C_Device device, void **ptr, size_t size) {
  if (size > global_free_mem_size) {
    return C_FAILED;
  }
  global_free_mem_size -= size;
  *ptr = malloc(size);
  return C_SUCCESS;
}

C_Status deallocate(const C_Device device, void *ptr, size_t size) {
  if (!ptr) {
    return C_FAILED;
  }
  global_free_mem_size += size;
  free(ptr);
  return C_SUCCESS;
}

C_Status create_stream(const C_Device device, C_Stream *stream) {
  stream = nullptr;
  return C_SUCCESS;
}

C_Status destroy_stream(const C_Device device, C_Stream stream) {
  return C_SUCCESS;
}

C_Status create_event(const C_Device device, C_Event *event) {
  return C_SUCCESS;
}

C_Status record_event(const C_Device device, C_Stream stream, C_Event event) {
  return C_SUCCESS;
}

C_Status destroy_event(const C_Device device, C_Event event) {
  return C_SUCCESS;
}

C_Status sync_device(const C_Device device) {
  return C_SUCCESS;
}

C_Status sync_stream(const C_Device device, C_Stream stream) {
  return C_SUCCESS;
}

C_Status sync_event(const C_Device device, C_Event event) {
  return C_SUCCESS;
}

C_Status stream_wait_event(const C_Device device, C_Stream stream, C_Event event) {
  return C_SUCCESS;
}

C_Status memstats(const C_Device device, size_t *total_memory, size_t *free_memory) {
  *total_memory = global_total_mem_size;
  *free_memory = global_free_mem_size
  return C_SUCCESS;
}

C_Status get_min_chunk_size(const C_Device device, size_t *size) {
  *size = 1;
  return C_SUCCESS;
}
```

## 自定义 Kernel

以 add 为例，介绍如何实现一个 kernel 并完成注册。

例子：

### 1.确定Kernel声明

查找飞桨发布的头文件`math_kernel.h`中，其Kernel函数声明如下：

```c++
// Add 内核函数
// 模板参数： T - 数据类型
//          Context - 设备上下文
// 参数： dev_ctx - Context 对象
//       x - DenseTensor 对象
//       y - DenseTensor 对象
//       out - DenseTensor 指针
// 返回： None
template <typename T, typename Context>
void AddKernel(const Context& dev_ctx,
               const DenseTensor& x,
               const DenseTensor& y,
               DenseTensor* out);

```

### 2.Kernel实现与注册

```c++
// add_kernel.cc

#include "paddle/phi/extension.h" // 自定义Kernel依赖头文件

namespace custom_cpu {

// Kernel函数体实现
template <typename T, typename Context>
void AddKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               const phi::DenseTensor& y,
               phi::DenseTensor* out) {
  // 使用dev_ctx的Alloc API为输出参数out分配模板参数T数据类型的内存空间
  dev_ctx.template Alloc<T>(out);
  // 使用DenseTensor的numel API获取Tensor元素数量
  auto numel = x.numel();
  // 使用DenseTensor的data API获取输入参数x的模板参数T类型的数据指针
  auto x_data = x.data<T>();
  // 使用DenseTensor的data API获取输入参数y的模板参数T类型的数据指针
  auto y_data = y.data<T>();
  // 使用DenseTensor的data API获取输出参数out的模板参数T类型的数据指针
  auto out_data = out->data<T>();
  // 完成计算逻辑
  for (auto i = 0; i < numel; ++i) {
    out_data[i] = x_data[i] + y_data[i];
  }
}

} // namespace custom_cpu

// 全局命名空间内使用注册宏完成Kernel注册
// CustomCPU的AddKernel注册
// 参数： add - Kernel名称
//       CustomCPU - 后端名称
//       ALL_LAYOUT - 内存布局
//       custom_cpu::AddKernel - Kernel函数名
//       int - 数据类型名
//       int64_t - 数据类型名
//       float - 数据类型名
//       double - 数据类型名
//       phi::dtype::float16 - 数据类型名
PD_REGISTER_PLUGIN_KERNEL(add,
                          CustomCPU,
                          ALL_LAYOUT,
                          custom_cpu::AddKernel,
                          int,
                          int64_t
                          float,
                          double,
                          phi::dtype::float16){}
```

## 编译

### CMake 编译

**编写CMakeLists.txt**

```
cmake_minimum_required(VERSION 3.10)

project(paddle-custom_cpu CXX C)

set(PLUGIN_NAME        "paddle_custom_cpu")
set(PLUGIN_VERSION      "0.0.1")

set(PADDLE_PLUGIN_DIR  "/opt/conda/lib/python3.7/site-packages/paddle-plugins/")
set(PADDLE_INC_DIR     "/opt/conda/lib/python3.7/site-packages/paddle/include/")
set(PADDLE_LIB_DIR     "/opt/conda/lib/python3.7/site-packages/paddle/fluid/")

include_directories(${PADDLE_INC_DIR})
link_directories(${PADDLE_LIB_DIR})

############ 编译插件
add_library(${PLUGIN_NAME} SHARED runtime.cc add_kernel.cc)
target_link_libraries(${PLUGIN_NAME} PRIVATE core_avx.so)  # special name

############ 打包插件
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/setup.py)

add_custom_command(TARGET ${PLUGIN_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_CURRENT_BINARY_DIR}/python/
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/python/
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/python/paddle-plugins/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_BINARY_DIR}/lib${PLUGIN_NAME}.so ${CMAKE_CURRENT_BINARY_DIR}/python/paddle-plugins/
    COMMENT "Creating plugin dirrectories------>>>"
)

add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/python/.timestamp
    COMMAND python3 ${CMAKE_CURRENT_BINARY_DIR}/setup.py bdist_wheel
    DEPENDS ${PLUGIN_NAME}
    COMMENT "Packing whl packages------>>>"
)

add_custom_target(python_package ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/python/.timestamp)
```

**编写setup.py.in**

CMake 根据 setup.py.in 生成 setup.py，再使用 setuptools 将插件封装成 wheel 包。

```
from setuptools import setup, Distribution

packages = []
package_data = {}

class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True

setup(
    name = '@CMAKE_PROJECT_NAME@',
    version='@PLUGIN_VERSION@',
    description='Paddle CustomCPU plugin',
    long_description='',
    long_description_content_type="text/markdown",
    author_email="Paddle-better@baidu.com",
    maintainer="PaddlePaddle",
    maintainer_email="Paddle-better@baidu.com",
    project_urls={},
    license='Apache Software License',
    packages= [
        'paddle-plugins',
    ],
    include_package_data=True,
    package_data = {
        '': ['*.so', '*.h', '*.py', '*.hpp'],
    },
    package_dir = {
        '': 'python',
    },
    zip_safe=False,
    distclass=BinaryDistribution,
    entry_points={
        'console_scripts': [
        ]
    },
    classifiers=[
    ],
    keywords='Paddle CustomCPU plugin',
)
```

通过如下命令完成插件编译。

```bash
$ mkdir build
$ cd build
$ cmake .. -DWITH_KERNELS=ON
$ make
```

编译完成后在 build/dist 目录下生成 wheel 包。

### setuptools 编译

**编写setup.py**

setuptools 也可以用于编译插件，并直接打包

```
from setuptools import setup, Distribution, Extension
from setuptools.command.build_ext import build_ext
import os
import shutil

packages = []
package_data = {}

class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True

for pkg_dir in ['build/python/paddle-plugins/']:
    if os.path.exists(pkg_dir):
        shutil.rmtree(pkg_dir)
    os.makedirs(pkg_dir)

ext_modules = [Extension(name='paddle-plugins.libpaddle_custom_cpu',
                         sources=['runtime.cc', 'add_kernel.cc'],
                         include_dirs=['/opt/conda/lib/python3.7/site-packages/paddle/include/'],
                         library_dirs=['/opt/conda/lib/python3.7/site-packages/paddle/fluid/'],
                         libraries=['core_avx.so'])]

setup(
    name='paddle-custom_cpu',
    version='0.0.1',
    description='Paddle CustomCPU plugin',
    long_description='',
    long_description_content_type="text/markdown",
    author_email="Paddle-better@baidu.com",
    maintainer="PaddlePaddle",
    maintainer_email="Paddle-better@baidu.com",
    project_urls={},
    license='Apache Software License',
    ext_modules=ext_modules,
    packages=[
        'paddle-plugins',
    ],
    include_package_data=True,
    package_data={
        '': ['*.so', '*.h', '*.py', '*.hpp'],
    },
    package_dir={
        '': 'build/python',
    },
    zip_safe=False,
    distclass=BinaryDistribution,
    entry_points={
        'console_scripts': [
        ]
    },
    classifiers=[
    ],
    keywords='Paddle CustomCPU plugin',
)
```

通过如下命令完成插件编译。

```
$ python setup.py bdist_wheel
```

编译完成后在以及 dist 目录下生成wheel包。

## 安装

通过 pip 安装 wheel 包。

```
$ pip install build/dist/paddle_custom_cpu-0.0.1-cp37-cp37m-linux_aarch64.whl
```

## 使用

安装插件到指定路径后（ site-packages/paddle-plugins ），我们就可以使用 PaddlePaddle 的 CustomCPU 硬件后端用于执行计算任务。

首先，需要查看 PaddlePaddle 目前已注册的自定义硬件。

```
>>> paddle.device.get_all_custom_device_type()
['CustomCPU']
```

接下来设置要使用的硬件后端。

```
>>> paddle.set_device('CustomCPU')
```

最后， 使用新硬件后端用于执行计算任务。

```
>>> x = paddle.to_tensor([1])
>>> x
Tensor(shape=[1], dtype=int64, place=Place(CustomCPU:0), stop_gradient=True,
       [1])
>>> x + x
Tensor(shape=[1], dtype=int64, place=Place(CustomCPU:0), stop_gradient=True,
       [2])
```
