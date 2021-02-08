#!/bin/sh

echo "Removing older build"
rm -rf build && mkdir build && cd build || exit

echo "Make new build"
cmake .. && make || exit

#printf '%s\n' "${PWD##*/}"

echo "Running application"
./gaussian_naive_bayes
