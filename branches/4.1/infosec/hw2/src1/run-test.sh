#!/bin/sh

MODE="ECB CBC CFB OFB CTR"
TESTFILE="text.txt"

echo $MODE

for m in $MODE
do
	echo "Testing $m mode"
	echo "./aes -m $m -e -i $TESTFILE -o $m-c.dat"
	./aes -m $m -e -i $TESTFILE -o $m-c.dat
	echo "./aes -m $m -d -i $TESTFILE -o $m-p.dat"
	./aes -m $m -d -i $m-c.dat -o $m-p.dat
	echo ""
	echo "$m mode test finished"
	echo ""
done

rm -rf *.dat
