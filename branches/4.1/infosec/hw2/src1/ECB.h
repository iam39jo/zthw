#ifndef __ECB_H__
#define __ECB_H__

#include "types.h"

int ECBEncrypt(const ubyte_t *plain, const ubyte_t *key, ubyte_t **cipher, int length)

int ECBDecrypt(const ubyte_t *cipher, const ubyte_t *key, ubyte_t **plain, int length)

#endif
