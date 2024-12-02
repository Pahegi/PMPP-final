rm -rf build
mkdir build
cd build
cmake ..
make -j 32
cd ..
./build/test_pmpp_final