#ifndef __AES_H__
#define __AES_H__

#define ROW_COUNT 4
#include "types.h"

void AESEncrypt(ubyte_t * output, const ubyte_t * key, int Nb, int Nk, int Nr);

void AESDecrypt(ubyte_t * output, const ubyte_t * key, int Nb, int Nk, int Nr);

#endif
