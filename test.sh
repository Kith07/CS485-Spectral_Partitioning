#!/usr/bin/env bash
#
# run_spectral.sh
# Usage: ./run_spectral.sh <input_file> <num_clusters> [mpi_ranks]
#

set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 <input_file> <num_clusters> [mpi_ranks]"
  exit 1
fi

INPUT="$1"
CLUSTERS="$2"
RANKS="${3:-4}"

echo "Building all executables..."
make clean >/dev/null && make >/dev/null
echo

# arrays to hold outputs and timings
declare -A output time

# helper to run & time a command
# args: key, command...
run_and_time() {
  local key="$1"; shift
  local start end elapsed

  start=$(date +%s.%N)
  # capture stdout only
  output["$key"]="$("$@" )"
  end=$(date +%s.%N)

  # compute elapsed = end - start
  elapsed=$(awk "BEGIN {print $end - $start}")
  time["$key"]="$elapsed"
}

# run each variant
run_and_time serial   ./spec_ser      "$INPUT" "$CLUSTERS"
run_and_time cuda     ./spec_cu       "$INPUT" "$CLUSTERS"
run_and_time mpi      mpirun -np "$RANKS" ./spec_mpi      "$INPUT" "$CLUSTERS"
run_and_time cu_mpi   mpirun -np "$RANKS" ./spec_cu_mpi  "$INPUT" "$CLUSTERS"

# print results
echo "=== Serial ==="
printf "%s\n\n" "${output[serial]}"
echo "Time (s): ${time[serial]}"
echo

echo "=== CUDA-only ==="
printf "%s\n\n" "${output[cuda]}"
echo "Time (s): ${time[cuda]}"
echo

echo "=== MPI-only (${RANKS} ranks) ==="
printf "%s\n\n" "${output[mpi]}"
echo "Time (s): ${time[mpi]}"
echo

echo "=== CUDA+MPI (${RANKS} ranks) ==="
printf "%s\n\n" "${output[cu_mpi]}"
echo "Time (s): ${time[cu_mpi]}"
echo "All done."
