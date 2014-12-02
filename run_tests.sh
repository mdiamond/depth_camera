#!/usr/bin/env bash
# Run unit tests.
#
# Example usage: ./run_tests.sh

# Run unit tests.
if python -m unittest -v test_calibrator test_stereo_rig_model; then
    echo "All unit tests passing. Congrats!"
else
    echo "Some unit tests failing."
fi
