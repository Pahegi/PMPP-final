rm -rf build
mkdir build
mkdir build/out
cp -r *.bin build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make -j 16
# cuda-gdb pmpp_ex1
# BREAKPOINTS BEI JEDEM KERNELAUFRUF: set cuda break_on_launch application
# GEHT NICHT: break histogram_kernel
# GEHT NICHT: break histogram_kernel if threadIdx.x == 0
# run
# continue
# cuda kernel x block y thread z
# cuda kernel block thread
# info cuda kernels
