#!/bin/bash
# assembler for simips machine

USAGE="Usage: asm.sh asm-file"

if [ $# -lt 1 ]; then
	echo $USAGE
	exit 3
fi

filebase=${1%.*}

if [ -r $1 ]; then
	./parse.py $1 "$filebase.txt"
	./bin-gen "$filebase.txt" "$filebase.bin"
else
	echo "file $1 doesn't exist"
	exit 4
fi
