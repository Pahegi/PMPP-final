if ! module list 2>&1 | grep -q 'cuda/12.5'; then
	module load cuda/12.5
fi

if ! module list 2>&1 | grep -q 'gcc/13.1.0'; then
	module load gcc/13.1.0
fi

export localdir=$(pwd)

# Define the build directory
BUILD_DIR="$SCRATCH/build"

# Check if the build directory exists
if [ ! -d "$BUILD_DIR" ]; then
    echo "Build directory does not exist. Creating it and running cmake."
    mkdir -p "$BUILD_DIR"
    cmake -S . -B "$BUILD_DIR" || exit 1
	cp -r *.bin $SCRATCH
	cp -r *.bin $BUILD_DIR
else
    # Check if any CMakeLists.txt or cmake file is newer than the build directory or Makefile
    if [ "$(find . -name 'CMakeLists.txt' -o -name '*.cmake' -newer "$BUILD_DIR" -print -quit)" ]; then
        echo "CMake configuration files changed. Running cmake."
        cmake -S . -B "$BUILD_DIR" || exit 1
    else
        echo "CMake is up-to-date."
    fi
fi
cd "$BUILD_DIR" || exit 1
make -j 96
