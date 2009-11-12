#ifndef __OFB_H__
#define __OFB_H__

#include "types.h"

#define J_BIT_SIZE 8
#define J_BYTE_SIZE 1

int OFBEncrypt(const ubyte_t *plain, const ubyte_t *key, const ubyte_t *v_array, ubyte_t **cipher, int length);

int OFBDecrypt(const ubyte_t *cipher, const ubyte_t *key, const ubyte_t *v_array, ubyte_t **plain, int length);

#endif
