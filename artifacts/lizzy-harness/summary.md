# Lizzy real checkpoint validation

- commit: 163805bae20c1c7bb5fbfb4d0a5f583ad2a6c188
- model: flwrlabs/Lizzy-7B
- build: 1
- download: 125
- CELEG semantic resolution: 125
- Transformers reference: 125
- logits/token comparison: 125

## build stderr
```text
build: FAIL: CMake configure failed.
```

## build stdout
```text
doctor: platform       linux
doctor: backend        cpu (requested cpu)
doctor: cmake          /usr/local/bin/cmake (3.31.6)
doctor: ninja          /usr/local/bin/ninja (1.13.2)
doctor: compiler       /usr/bin/x86_64-linux-gnu-g++-13 (13.3.0)
doctor: nvcc           <not found>
doctor: gpu            <not detected>
doctor: driver         <not detected>
doctor: architecture   native
doctor: checkpoint     <not cached>
doctor: RESULT: ready
+ /usr/local/bin/cmake --fresh -S /home/runner/work/celeg/celeg -B /home/runner/work/celeg/celeg/out/linux-cpu-release -G Ninja -DCMAKE_BUILD_TYPE=Release -DCELEG_ENABLE_CUDA=OFF -DCELEG_BUILD_TESTS=ON -DCELEG_RUN_CUDA_TESTS=ON
-- The C compiler identification is GNU 13.3.0
-- The CXX compiler identification is GNU 13.3.0
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: /usr/bin/cc - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/c++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Found Python3: /opt/hostedtoolcache/Python/3.12.13/x64/bin/python3.12 (found version "3.12.13") found components: Interpreter
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Success
-- Found Threads: TRUE
CMake Error at /usr/local/share/cmake-3.31/Modules/FindPackageHandleStandardArgs.cmake:233 (message):
  Could NOT find CURL (missing: CURL_LIBRARY CURL_INCLUDE_DIR)
Call Stack (most recent call first):
  /usr/local/share/cmake-3.31/Modules/FindPackageHandleStandardArgs.cmake:603 (_FPHSA_FAILURE_MESSAGE)
  /usr/local/share/cmake-3.31/Modules/FindCURL.cmake:203 (find_package_handle_standard_args)
  CMakeLists.txt:99 (find_package)


-- Configuring incomplete, errors occurred!
```

