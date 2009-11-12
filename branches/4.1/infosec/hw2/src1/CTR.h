#ifndef __CTR_H__
#define __CTR_H__

#include "types.h"


int CTREncrypt(const ubyte_t *plain, const ubyte_t *key, const ubyte_t *counter, ubyte_t **cipher, int length);

int CTRDecrypt(const ubyte_t *cipher, const ubyte_t *key, const ubyte_t *counter, ubyte_t **plain, int length);

#endif
