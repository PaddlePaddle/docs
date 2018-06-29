cmake_minimum_required(VERSION 3.0)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/")


if(ANDROID_ABI)
    set(ANDROID_COMPILER_FLAGS -ffunction-sections -fdata-sections)
    set(ANDROID_LINKER_FLAGS -Wl,--gc-sections)

    if(ANDROID_ABI STREQUAL "armeabi-v7a")
        set(CMAKE_SYSROOT "${ANDROID_STANDALONE_TOOLCHAIN}/sysroot")
        set(ANDROID_TOOLCHAIN_PREFIX 
        "${ANDROID_STANDALONE_TOOLCHAIN}/bin/arm-linux-androideabi")
        list(APPEND ANDROID_COMPILER_FLAGS -mfpu=neon)
        list(APPEND ANDROID_LINKER_FLAGS -Wl,--fix-cortex-a8)
    elseif(ANDROID_ABI STREQUAL "arm64-v8a")
        set(CMAKE_SYSROOT "${ANDROID_STANDALONE_TOOLCHAIN}/sysroot")
        set(ANDROID_TOOLCHAIN_PREFIX 
        "${ANDROID_STANDALONE_TOOLCHAIN}/bin/aarch64-linux-android")
        list(APPEND ANDROID_COMPILER_FLAGS -march=armv8-a)
    endif()
    string(REPLACE ";" " " ANDROID_COMPILER_FLAGS "${ANDROID_COMPILER_FLAGS}")
    string(REPLACE ";" " " ANDROID_LINKER_FLAGS "${ANDROID_LINKER_FLAGS}")

    set(CMAKE_C_FLAGS "${ANDROID_COMPILER_FLAGS} ${CMAKE_C_FLAGS}" CACHE STRING "C flags")
    set(CMAKE_CXX_FLAGS "${ANDROID_COMPILER_FLAGS} ${CMAKE_CXX_FLAGS}" CACHE STRING "CXX flags")
    set(CMAKE_SHARED_LINKER_FLAGS "${ANDROID_LINKER_FLAGS} ${CMAKE_SHARED_LINKER_FLAGS}" CACHE STRING "shared linker flags")
    set(CMAKE_EXE_LINKER_FLAGS "-pie -fPIE ${ANDROID_LINKER_FLAGS} ${CMAKE_EXE_LINKER_FLAGS}"  CACHE STRING "executable linker flags")

    set(CMAKE_C_COMPILER ${ANDROID_TOOLCHAIN_PREFIX}-gcc CACHE PATH "C compiler" FORCE)
    set(CMAKE_CXX_COMPILER ${ANDROID_TOOLCHAIN_PREFIX}-g++ CACHE PATH "CXX compiler" FORCE)
endif()
project(inference CXX C)
include(FindPaddle)

aux_source_directory(. SRC_LIST)
add_executable(${PROJECT_NAME} ${SRC_LIST})

set(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS)
set(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS)
if(ANDROID_ABI)
    target_link_libraries(${PROJECT_NAME}
                          -Wl,--start-group -Wl,--whole-archive -lpaddle_capi_layers
                          -Wl,--no-whole-archive -lpaddle_capi_engine -Wl,--end-group
                          ${THIRD_PARTY_LIBRARYS}
                          log)
else()
    target_link_libraries(${PROJECT_NAME}
                    -Wl,--start-group -Wl,--whole-archive -lpaddle_capi_whole
                    -Wl,--no-whole-archive -Wl,--end-group
                    ${THIRD_PARTY_LIBRARYS} -lrt -ldl -lpthread)
endif()
