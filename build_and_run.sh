# build program
./build.sh

FILE="$SCRATCH/test.out"
ERR="$SCRATCH/test.err"

# remove old output files
rm $FILE
rm $ERR
rm -rf "$SCRATCH/out"

echo "Running job..."
sbatch --wait run.sh

# File exists, print its contents
echo "___________________________"
echo "OUT:"
cat "$FILE"
echo "___________________________"
echo ""
echo "___________________________"
echo "ERR:"
cat "$ERR"
echo "___________________________"

# Print nvprof output if it exists
if [ -f "$SCRATCH/out/nvprof_trace.txt" ]; then
	echo "___________________________"
	echo "NVPROF:"
	cat "$SCRATCH/out/nvprof_trace.txt"
	echo "___________________________"
fi
