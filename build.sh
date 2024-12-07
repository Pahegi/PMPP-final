export localdir=$(pwd)
rm -rf $SCRATCH/build
mkdir $SCRATCH/build
mkdir $SCRATCH/build/out
cp -r *.bin $SCRATCH
cp -r *.bin $SCRATCH/build
cd $SCRATCH/build
cmake -DCMAKE_BUILD_TYPE=Release $localdir
make -j 16