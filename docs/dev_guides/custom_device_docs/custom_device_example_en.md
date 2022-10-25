# CustomDevice Example

In this section we will walk through the steps required to extend a fake hardware backend for PaddlePaddle by implementing a fake device named CustomCPU.

> Note：
> - Please make sure that you have correctly installed the latest version of [Paddle develop](https://github.com/PaddlePaddle/Paddle).
> - Only `Linux` is supported

## Step One: Implement Custom Runtime

**InitPlugin**

As a custom runtime entry function, InitPlugin is required to be implemented by the plug-in. The parameter in InitPlugin should also be checked, device information should be filled in, and the runtime API should be registered. In the initialization, PaddlePaddle loads the plug-in and invokes InitPlugin to initialize it, and register runtime (The whole process can be done automatically by the framework, only if the dynamic-link library is in site-packages/paddle-plugins/ or the designated directory of the enviornment variable of CUSTOM_DEVICE_ROOT).

Example:

```c++
#include "paddle/phi/backends/device_ext.h"

void InitPlugin(CustomRuntimeParams *params) {
  // Check compatibility of the version and fill in the information of the custom runtime version used by the plug-in.
  PADDLE_CUSTOM_RUNTIME_CHECK_VERSION(params);

  // Fill in the basic runtime information
  params->device_type = "CustomCPU";
  params->sub_device_type = "V1";

  // Register the Runtime API
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
  params->interface->get_device_count = get_device_count;
  params->interface->get_device_list = get_device_list;
  params->interface->device_memory_stats = memstats;
  params->interface->device_min_chunk_size = get_min_chunk_size;
}
```

The plug-in should first check the parameters of InitPlugin, and the framework should set the size to an optimal value and sent it to InitPlugin. For types of CustomRuntimeParams and C_DeviceInterface, please refer to[device_ext.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/backends/device_ext.h).

Then, the plug-in should fill in its basic information and version number, which can be helpful for PaddlePaddle to manage the plug-in and check the version compatibility.

- params->size and params->interface.size ： In the following custom runtime versions, the size and the interface will rank the first and the second respectively in all types of CustomRuntimeParams.
- params->version ： Information of the plug-in version is filled in. The definition of the version number can be found in [device_ext.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/backends/device_ext.h). And PaddlePaddle checks the version compatibility in the registration of custom runtime.
- params->device_type ： the appellation of the device backend. If there is another plug-in with the same name, the runtime will not be registered.
- params->sub_device_type ： the appellation of the sub-type of the device backend

Finally, some callback APIs in params->interface should be filled by the plug-in (At least the required APIs should be implemented, or the runtime will not be registered otherwise). Thus, the custom runtime can be initialized. For details of the APIS, please refer to [Custom Runtime Document](./custom_runtime_en.html)。

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

## Step Two：Add Custom Kernel

Taking the add as an example, this part will introduce how to implement a kernel and make it registered.

Example：

### 1. Determine the Kernel Statement

Find the kernel statement of the header file `math_kernel.h` released by PaddlePaddle:

```c++
// Add the kernel function
// Model parameters： T - Data type
//          Context - Device context
// Parameters： dev_ctx - Context object
//       x - DenseTensor object
//       y - DenseTensor object
//       out - DenseTensor point
// Return： None
template <typename T, typename Context>
void AddKernel(const Context& dev_ctx,
               const DenseTensor& x,
               const DenseTensor& y,
               DenseTensor* out);

```

### 2.Kernel Implementation and Registration

```c++
// add_kernel.cc

#include "paddle/phi/extension.h" // the header file on which the custom kernel depends

namespace custom_cpu {

// Kernel Implementation
template <typename T, typename Context>
void AddKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               const phi::DenseTensor& y,
               phi::DenseTensor* out) {
  // Use Alloc API of dev_ctx to allocate storage of the template parameter T for the output parameter--out.
  dev_ctx.template Alloc<T>(out);
  // Use numel API of DenseTensor to acquire the number of Tensor elements.
  auto numel = x.numel();
  // Use data API of DenseTensor to acquire the data pointer of the template parameter T of the input parameter--x.
  auto x_data = x.data<T>();
  // Use data API of DenseTensor to acquire the data pointer of the template parameter T of the input parameter--y.
  auto y_data = y.data<T>();
  // Use data API of DenseTensor to acquire the data pointer of the template parameter T of the output parameter--out.
  auto out_data = out->data<T>();
  // Get the computing logic done
  for (auto i = 0; i < numel; ++i) {
    out_data[i] = x_data[i] + y_data[i];
  }
}

} // namespace custom_cpu

// In the global namespace, use the macro of registration to register the kernel.
// Register AddKernel of CustomCPU
// Parameters： add - Kernel name
//       CustomCPU - Backend name
//       ALL_LAYOUT - Memory layout
//       custom_cpu::AddKernel - Name of the kernel function
//       int - Data type name
//       int64_t - Data type name
//       float - Data type name
//       double - Data type name
//       phi::dtype::float16 - Data type name
PD_REGISTER_PLUGIN_KERNEL(add,
                          CustomCPU,
                          ALL_LAYOUT,
                          custom_cpu::AddKernel,
                          int,
                          int64_t,
                          float,
                          double,
                          phi::dtype::float16){}
```

## Step Three：Compile and Install

### CMake Compilation

**Edit CMakeLists.txt**

```
cmake_minimum_required(VERSION 3.10)

project(paddle-custom_cpu CXX C)

set(PLUGIN_NAME        "paddle_custom_cpu")
set(PLUGIN_VERSION      "0.0.1")

set(PADDLE_PLUGIN_DIR  "/opt/conda/lib/python3.7/site-packages/paddle-plugins/")
set(PADDLE_INC_DIR     "/opt/conda/lib/python3.7/site-packages/paddle/include/")
set(PADDLE_LIB_DIR     "/opt/conda/lib/python3.7/site-packages/paddle/fluid/")

############ Third-party dependencies
set(BOOST_INC_DIR      "/path/to/Paddle/build/third_party/boost/src/extern_boost")
set(GFLAGS_INC_DIR     "/path/to/Paddle/build/third_party/install/gflags/include")
set(GLOG_INC_DIR       "/path/to/Paddle/build/third_party/install/glog/include")
set(THREAD_INC_DIR     "/path/to/Paddle/build/third_party/threadpool/src/extern_threadpool")
set(THIRD_PARTY_INC_DIR ${BOOST_INC_DIR} ${GFLAGS_INC_DIR} ${GLOG_INC_DIR} ${THREAD_INC_DIR})

include_directories(${PADDLE_INC_DIR} ${THIRD_PARTY_INC_DIR})
link_directories(${PADDLE_LIB_DIR})

add_definitions(-DPADDLE_WITH_CUSTOM_DEVICE)  # for out CustomContext temporarily
add_definitions(-DPADDLE_WITH_CUSTOM_KERNEL)  # for out fluid separate temporarily

############ Compile plug-ins
add_library(${PLUGIN_NAME} SHARED runtime.cc add_kernel.cc)
target_link_libraries(${PLUGIN_NAME} PRIVATE :core_avx.so)  # special name

############ Assembly plug-ins
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

**Edit setup.py.in**

CMake generates setup.py according to setup.py.in，and uses setuptools to encapsulate plug-ins into a wheel package.

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

Compile plug-ins by following the command:

```bash
$ mkdir build
$ cd build
$ cmake .. -DWITH_KERNELS=ON
$ make
```

After the compilation, make a wheel package under build/dist.

### Setuptools Compilation

**Edit setup.py**

setuptools can be used to compile plug-ins and directly package them.

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

Compile plug-ins by running the command:

```
$ python setup.py bdist_wheel
```

After the compilation, make a wheel package under the directory of dist.

### Pip Installation

Use pip to install a wheel package.

```
$ pip install build/dist/paddle_custom_cpu-0.0.1-cp37-cp37m-linux_aarch64.whl
```

## Step Four：Load and Use

After installing plug-ins to their designated paths (site-packages/paddle-plugins), we can use the device backend of CustomCPU of PaddlePaddle to execute computation.

First, check the custom devices of PaddlePaddle currently registered.

```
>>> paddle.device.get_all_custom_device_type()
['CustomCPU']
```

Then, set the device backend to be used.

```
>>> paddle.set_device('CustomCPU')
```

Finally, use the new backend for computing tasks.

```
>>> x = paddle.to_tensor([1])
>>> x
Tensor(shape=[1], dtype=int64, place=Place(CustomCPU:0), stop_gradient=True,
       [1])
>>> x + x
Tensor(shape=[1], dtype=int64, place=Place(CustomCPU:0), stop_gradient=True,
       [2])
```
