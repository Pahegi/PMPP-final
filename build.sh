if ! module list 2>&1 | grep -q 'cuda/12.5'; then
	module load cuda/12.5
fi

if ! module list 2>&1 | grep -q 'gcc/13.1.0'; then
	module load gcc/13.1.0
fi
export localdir=$(pwd)
rm -rf $SCRATCH/build
mkdir $SCRATCH/build
mkdir $SCRATCH/build/out
cp -r *.bin $SCRATCH
cp -r *.bin $SCRATCH/build
cd $SCRATCH/build
cmake -DCMAKE_BUILD_TYPE=Release $localdir
make -j 96
