#ifndef __TEXT_OPERATIONS_H__
#define __TEXT_OPERATIONS_H__

#include "types.h"

#define BLOCK_BIT_SIZE 128
#define BLOCK_BYTE_SIZE 16

void XORText(ubyte_t *dest, const ubyte_t *op, int length);

int FormatPlainText(const ubyte_t *text, ubyte_t **out, int length);

int ParsePlainText(const ubyte_t *text, ubyte_t **out, int length);

#endif
