#ifndef __NAND_ECC_BCH_HEADER__
#define __NAND_ECC_BCH_HEADER__

void bch_encoder(const unsigned char *indata, unsigned char *bch_code);

int bch_decoder(unsigned char *indata, const unsigned char *bch_code);

#endif
