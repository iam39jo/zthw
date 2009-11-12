#ifndef __CFB_H__
#define __CFB_H__

#include "types.h"

#define J_BIT_SIZE 8
#define J_BYTE_SIZE 1

int CFBEncrypt(const ubyte_t *plain, const ubyte_t *key, const ubyte_t *v_array, ubyte_t **cipher, int length);

int CFBDecrypt(const ubyte_t *cipher, const ubyte_t *key, const ubyte_t *v_array, ubyte_t **plain, int length);

#endif
