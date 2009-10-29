#ifndef __PLAINTEXT_H__
#define __PLAINTEXT_H__

#include "types.h"

#define BLOCK_BIT_SIZE 128
#define BLOCK_BYTE_SIZE 16

int FormatPlainText(const ubyte_t *text, ubyte_t **out, int length);

int ParsePlainText(const ubyte_t *text, ubyte_t **out, int length);

#endif
