# Build Environment
## Use docker
### 1. Install docker
About the installation of docker, please refer to [offical document](https://docs.docker.com/install/)
### 2. Use docker to build environment
First we enter into the directory of paddle-mobile and run `docker build`
Take Linux/Mac as an example (It is recommended to run in 'Docker Quickstart Terminal' in windows)
```
$ docker build -t paddle-mobile:dev - < Dockerfile
```
Use `docker images` to show image we created
```
$ docker images
REPOSITORY      TAG     IMAGE ID       CREATED         SIZE
paddle-mobile   dev     33b146787711   45 hours ago    372MB
```
### 3. Use docker to build
Enter into the directory of paddle-mobile and perform docker run
```
$ docker run -it --mount type=bind, source=$PWD, target=/paddle-mobile paddle-mobile:dev
root@5affd29d4fc5:/ # cd /paddle-mobile
# Generate Makefile in the construction of android
root@5affd29d4fc5:/ # rm CMakeCache.txt
root@5affd29d4fc5:/ # cmake -DCMAKE_TOOLCHAIN_FILE=tools/toolchains/arm-android-neon.cmake
# Generate Makefile in the construction of linux
root@5affd29d4fc5:/ # rm CMakeCache.txt
root@5affd29d4fc5:/ # cmake -DCMAKE_TOOLCHAIN_FILE=tools/toolchains/arm-linux-gnueabi.cmake
```
### 4. Configure Options for Build
We can configure options for build with ccmake.
```
root@5affd29d4fc5:/ # ccmake .
                                                     Page 1 of 1
 CMAKE_ASM_FLAGS
 CMAKE_ASM_FLAGS_DEBUG
 CMAKE_ASM_FLAGS_RELEASE
 CMAKE_BUILD_TYPE
 CMAKE_INSTALL_PREFIX             /usr/local
 CMAKE_TOOLCHAIN_FILE             /paddle-mobile/tools/toolchains/arm-android-neon.cmake
 CPU                              ON
 DEBUGING                         ON
 FPGA                             OFF
 LOG_PROFILE                      ON
 MALI_GPU                         OFF
 NET                              googlenet
 USE_EXCEPTION                    ON
 USE_OPENMP                       OFF
```
After updating options, we can update Makefile according to `c`, `g` .
### 5. Build
Use command make to build
```
root@5affd29d4fc5:/ # make
```
### 6. Check Output of Build
Output of build can be checked in host machine. In the directory of paddle-mobile, build and test/build, you can use command adb or scp to make it run on device.

## Without docker
Without docker, you can directly use cmake to generate makefile and then build. It needs to appropriately configure NDK_ROOt to build android with ndk. To build linux applications needs to install arm-linux-gnueabi-gcc or similiar cross-fix build tools and may need to configure environment path like CC, CXX, or update arm-linux-gnueabi.cmake in tools/toolchains/ , or add toolchain file according to your requirements.