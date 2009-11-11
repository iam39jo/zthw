#ifndef __ECB_H__
#define __ECB_H__

#include "types.h"

int CBCEncrypt(const ubyte_t *plain, const ubyte_t *key, const ubyte_t *v_array, ubyte_t **cipher, int length);

int CBCDecrypt(const ubyte_t *cipher, const ubyte_t *key, const ubyte_t *v_array, ubyte_t **plain, int length);

#endif
