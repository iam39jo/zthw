#!/bin/bash
# Initialize the symbol link file needed for sim-safe compile and simips ISA

SIMSAFE_ROOT_PATH=`pwd`

ln -sf target-simips/config.h config.h
ln -sf target-simips/loader.c loader.c
ln -sf target-simips/simips.h machine.h
ln -sf target-simips/simips.c machine.c
ln -sf target-simips/simips.def machine.def
ln -sf target-simips/symbol.c symbol.c
ln -sf target-simips/syscall.c syscall.c
