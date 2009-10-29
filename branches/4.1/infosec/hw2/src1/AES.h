#ifndef __AES_H__
#define __AES_H__

#define ROW_COUNT 4

void AESEncrypt(ubyte_t * output, ubyte_t * key, int Nb, int Nk, int Nr);

void AESDecrypt(ubyte_t * output, ubyte_t * key, int Nb, int Nk, int Nr);

#endif
