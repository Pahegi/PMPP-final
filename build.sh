module load cuda/12.5 gcc/13.1.0
export localdir=$(pwd)
rm -rf $SCRATCH/build
mkdir $SCRATCH/build
mkdir $SCRATCH/build/out
cp -r *.bin $SCRATCH
cp -r *.bin $SCRATCH/build
cd $SCRATCH/build
cmake -DCMAKE_BUILD_TYPE=Release $localdir
make -j 96
