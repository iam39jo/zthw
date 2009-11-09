#include <stdio.h>
#include "bch_static_data.h"

#define INFO_BYTE_SIZE	512
#define INFO_BIT_SIZE	4096
#define BCH_BYTE_SIZE	7
#define BCH_BIT_SIZE	52
#define BCH_EC_CAPA		4
#define BCH_EC_CAPA_X2	8
#define BCH_NN			8191
#define BCH_NN_SHORTEN	4148
#define BCH_PARALLEL	8
#define BCH_MAX_CORRECTION 20

void bch_encoder(const unsigned char *indata, unsigned char *bch_code);

int bch_decoder(unsigned char *indata, const unsigned char *bch_code);

int bch_decoder(unsigned char *indata, const unsigned char *bch_code)
{
	int elp_sum;
	int L[BCH_EC_CAPA_X2+3];			// Degree of ELP 
	int u_L[BCH_EC_CAPA_X2+3];		// Difference between step number and the degree of ELP
	int reg[BCH_EC_CAPA+3];			// Register state
	int elp[BCH_EC_CAPA_X2+4][BCH_EC_CAPA_X2+4]; 	// Error locator polynomial (ELP)
	int desc[BCH_EC_CAPA_X2+4];		// Discrepancy 'mu'th discrepancy
	int u;		// u = 'mu' + 1 and u ranges from -1 to 2*t (see L&C)
	int q;				//
	int err_count;
	int location[BCH_MAX_CORRECTION];	// Error location
	int s[BCH_BIT_SIZE];		// Syndrome values

	int Temp, loop_count;
	int bin_code[BCH_BIT_SIZE];
	int bin_code_temp[BCH_BIT_SIZE];
	int bin_data_p[BCH_PARALLEL][INFO_BIT_SIZE];
	int syn_error;

	int bin_recd[BCH_BIT_SIZE + INFO_BIT_SIZE];
	int idx_temp;
	int i, j, k;

	/* convert data&bch_code into bits */
	idx_temp = 0;
	for (i = 0; i < BCH_BYTE_SIZE-1; i++) {
		for (j = 7; j >= 0; j--) {
			if ((1 << j) & bch_code[i])
				bin_recd[idx_temp] = 1;
			else
				bin_recd[idx_temp] = 0;

			idx_temp++;
		}
	}
	for (i = 3; i >= 0; i--) {
		if ((1 << i) & bch_code[BCH_BYTE_SIZE-1])
			bin_recd[idx_temp] = 1;
		else
			bin_recd[idx_temp] = 0;

		idx_temp++;
	}

	for (i = 0; i < INFO_BYTE_SIZE; i++) {
		for (j = 7; j >= 0; j--) {
			if ((1 << j) & indata[i])
				bin_recd[idx_temp] = 1;
			else
				bin_recd[idx_temp] = 0;

			idx_temp++;
		}
	}

	/* BCH_PARALLEL_syndrome */
	loop_count = (BCH_NN_SHORTEN + BCH_PARALLEL - 1) / BCH_PARALLEL;

	for (i = 0; i < BCH_PARALLEL; i++)
		for (j = 0; j < loop_count; j++)
			if (i + j * BCH_PARALLEL < BCH_NN_SHORTEN)
				bin_data_p[i][j] = bin_recd[i + j * BCH_PARALLEL];
			else
				bin_data_p[i][j] = 0;

	for (i = 0; i < BCH_BIT_SIZE; i++)
		bin_code[i] = 0;

	for (k = loop_count - 1; k >= 0; k--) {
		for (i = 0; i < BCH_BIT_SIZE; i++) {
			Temp = 0;
			for (j = 0; j < BCH_BIT_SIZE; j++)
				if (bin_code[j] != 0 && bch_T_G_R[i][j] != 0)
					Temp ^= 1;
			bin_code_temp[i] = Temp;
		}

		for (i = 0; i < BCH_BIT_SIZE; i++)
			bin_code[i] = bin_code_temp[i];

		for (i = 0; i < BCH_PARALLEL; i++)
			bin_code[i] = bin_code[i] ^ bin_data_p[i][k];
	}


	syn_error = 0;
	for (i = 1; i < BCH_EC_CAPA_X2; i += 2) {
		s[i] = 0;
		for (j = 0; j < BCH_BIT_SIZE; j++)
			if (bin_code[j])
				s[i] ^= bch_alpha_to[(bch_index_of[bin_code[j]] + i*j) % BCH_NN];
		if (s[i] != 0)
			syn_error = 1;
	}
	printf("Syn_error = %d\n", syn_error);

	for (i = 2; i <= BCH_EC_CAPA_X2; i += 2) {
		j = i / 2;
		if (s[j] == 0)
			s[i] = 0;
		else
			s[i] = bch_alpha_to[(2 * bch_index_of[s[j]]) % BCH_NN];
	}

	/* decode */

	printf("Syn_error = %d\n", syn_error);
	// no errors
	if (!syn_error) {
		return 0;
	}

	printf("%s %d %s\n", __FILE__, __LINE__, __FUNCTION__);
	for (i = 1; i <= BCH_EC_CAPA_X2; i++)
		s[i] = bch_index_of[s[i]];

	desc[0] = 0;
	desc[1] = s[1];
	elp[0][0] = 1;
	elp[1][0] = 1;
	for (i = 1; i < BCH_EC_CAPA_X2; i++) {
		elp[0][i] = 0;
		elp[1][i] = 0;
	}
	L[0] = 0;
	L[1] = 0;
	u_L[0] = -1;
	u_L[1] = 0;
	u = -1;

	do {
		u += 2;

		if (desc[u] == -1) {
			L[u+2] = L[u];
			for (i = 0; i <= L[u]; i++)
				elp[u+2][i] = elp[u][i];
		} else {
			q = u - 2;
			if (q < 0)	q = 0;
			while ((desc[q] == -1) && (q > 0))
				q = q-2;
			if (q < 0)	q = 0;

			if (q > 0) {
				j = q;
				do {
					j = j-2;
					if (j < 0) j= 0;
					if ((desc[j] != -1) && (u_L[q] < u_L[j]))
						q = j;
				} while (j > 0);
			}

			if (L[u] > L[q] + u - q)
				L[u + 2] = L[u];
			else
				L[u + 2] = L[q] + u - q;

			for (i = 0; i < BCH_EC_CAPA_X2; i++)
				elp[u + 2][i] = 0;
			for (i = 0; i <= L[q]; i++)
				if (elp[q][i] != 0)
					elp[u+2][i+u-q] = bch_alpha_to[(desc[u] + BCH_NN - desc[q] + bch_index_of[elp[q][i]]) % BCH_NN];
			for (i = 0; i <= L[u]; i++)
				elp[u + 2][i] ^= elp[u][i];
		}
		u_L[u+2] = u+1 - L[u+2];


		if (u < BCH_EC_CAPA_X2) {
			if (s[u+2] != -1)
				desc[u+2] = bch_alpha_to[s[u+2]];
			else
				desc[u+2] = 0;

			for (i = 1; i <= L[u+2]; i++)
				if ((s[u+2-i] != -1) && (elp[u+2][i] != 0))
					desc[u+2] ^= bch_alpha_to[(s[u+2-i] + bch_index_of[elp[u+2][i]]) % BCH_NN];
			desc[u+2] = bch_index_of[desc[u+2]];
		}

	} while ((u < (BCH_EC_CAPA_X2 - 1)) && (L[u+2] <= BCH_EC_CAPA));

	u += 2;
	L[BCH_EC_CAPA_X2-1] = L[u];

	// Cannot recover the errors
	if (L[BCH_EC_CAPA_X2-1] > BCH_EC_CAPA)
		return -1;

	// recover the errors
	
	for (i = 1; i <= L[BCH_EC_CAPA_X2 - 1]; i++) {
		reg[i] = bch_index_of[elp[u][i]];
	}
	err_count = 0;

	for (i = 1; i <= BCH_NN; i++) {
		elp_sum = 1;
		for (j = 1; j <= L[BCH_EC_CAPA_X2 - 1]; j++)
			if (reg[j] != -1) {
				reg[j] = (reg[j] + j) % BCH_NN;
				elp_sum ^= bch_alpha_to[reg[j]];
			}

		if (!elp_sum) {
			location[err_count] = BCH_NN - i;
			err_count++;
		}
	}

	printf("error count = %d\n", err_count);
	for (i = 0; i < err_count; i++) {
		printf("At %d\n", location[i]);
	}

	if (err_count == L[BCH_EC_CAPA_X2 - 1]) {
		for (i = 0; i < L[BCH_EC_CAPA_X2 - 1]; i++)
			bin_recd[location[i]] ^= 1;

		/* recover indata */
		idx_temp = BCH_BIT_SIZE;
		for (i = 0; i < INFO_BYTE_SIZE; i++) {
			Temp = 0;
			for (j = 7; j >= 0; j--) {
				if (bin_recd[idx_temp])
					Temp += (1 << j);
				idx_temp++;
			}

			indata[i] = Temp;
		}

		return err_count;
	} else {
		return -1;
	}
}

void bch_encoder(const unsigned char *indata, unsigned char *bch_code)
{
	int bin_data[INFO_BIT_SIZE];
	int bin_data_p[BCH_PARALLEL][INFO_BIT_SIZE];
	int bin_code[BCH_BIT_SIZE];
	int bin_code_temp[BCH_BIT_SIZE];

	int loop_count;
	int i, j, k;
	int idx_temp;


	/* convert indata into bits */
	idx_temp = 0;
	for (i = 0; i < INFO_BYTE_SIZE; i++) {
		for (j = 7; j >= 0; j--) {
			if ((1 << j) & indata[i])
				bin_data[idx_temp] = 1;
			else
				bin_data[idx_temp] = 0;

			idx_temp++;
		}
	}

	/*printf("------------------------------------------------\n");*/
	/*for (i = 0; i < INFO_BIT_SIZE; i++) {*/
		/*printf("%d", bin_data[i]);*/
	/*}*/
	/*printf("\n");*/

	/* encode loop procedure */
	loop_count = (INFO_BIT_SIZE + BCH_PARALLEL - 1) / BCH_PARALLEL;

	for (i = 0; i < BCH_PARALLEL; i++) {
		for (j = 0; j < loop_count; j++) {
			if (i + j * BCH_PARALLEL < INFO_BIT_SIZE)
				bin_data_p[i][j] = bin_data[i + j * BCH_PARALLEL];
			else
				bin_data_p[i][j] = 0;
		}
	}

	for (i = 0; i < BCH_BIT_SIZE; i++)
		bin_code[i] = 0;

	for (k = loop_count - 1; k >= 0; k--) {
		for (i = 0; i < BCH_BIT_SIZE; i++)
			bin_code_temp[i] = bin_code[i];
		for (i = BCH_PARALLEL - 1; i >= 0; i--)
			bin_code_temp[BCH_BIT_SIZE - BCH_PARALLEL + i] = bin_code_temp[BCH_BIT_SIZE - BCH_PARALLEL + i] ^ bin_data_p[i][k];

		for (i = 0; i < BCH_BIT_SIZE; i++) {
			idx_temp = 0;
			for (j = 0; j < BCH_BIT_SIZE; j++)
				idx_temp = idx_temp ^ (bin_code_temp[j] * bch_T_G_R[i][j]);
			bin_code[i] = idx_temp;
		}
	}

	/*printf("[MY]");*/
	/*for (i = 0; i < BCH_BIT_SIZE; i++)*/
		/*printf("%d", bin_code[i]);*/
	/*printf("\n");*/


	/* convert & save bch code into output buffer */
	idx_temp = 0;
	for (i = 0; i < BCH_BYTE_SIZE - 1; i++) {
		bch_code[i] = 0;
		for (j = 7; j >= 0; j--) {
			if (bin_code[idx_temp])
				 bch_code[i] += (1 << j);
			idx_temp++;
		}
	}
	/* the last 4 bits */
	bch_code[BCH_BYTE_SIZE-1] = 0;
	for (i = 3; i >= 0; i--) {
		if (bin_code[idx_temp])
			bch_code[BCH_BYTE_SIZE-1] += (1 << i);
		idx_temp++;
	}

	/*for (i = 0; i < BCH_BYTE_SIZE; i++)
		printf("%02X", bch_code[i]);
	printf("\n");*/
}
