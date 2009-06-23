
#include "host.h"
#include "misc.h"
#include "loader.h"
#include "symbol.h"
/* symbol database in no particular order */
struct sym_sym_t *sym_db = NULL;

/* all symbol sorted by address */
int sym_nsyms = 0;
struct sym_sym_t **sym_syms = NULL;

/* all symbols sorted by name */
struct sym_sym_t **sym_syms_by_name = NULL;

/* text symbols sorted by address */
int sym_ntextsyms = 0;
struct sym_sym_t **sym_textsyms = NULL;

/* text symbols sorted by name */
struct sym_sym_t **sym_textsyms_by_name = NULL;

/* data symbols sorted by address */
int sym_ndatasyms = 0;
struct sym_sym_t **sym_datasyms = NULL;

/* data symbols sorted by name */
struct sym_sym_t **sym_datasyms_by_name = NULL;

/* load symbols out of FNAME */
void
sym_loadsyms(char *fname,		/* file name containing symbols */
	     int load_locals){}		/* load local symbols */

/* dump symbol SYM to output stream FD */
void
sym_dumpsym(struct sym_sym_t *sym,	/* symbol to display */
	    FILE *fd){}			/* output stream */

/* dump all symbols to output stream FD */
void
sym_dumpsyms(FILE *fd){}			/* output stream */

/* dump all symbol state to output stream FD */
void
sym_dumpstate(FILE *fd){}		/* output stream */


/* bind address ADDR to a symbol in symbol database DB, the address must
   match exactly if EXACT is non-zero, the index of the symbol in the
   requested symbol database is returned in *PINDEX if the pointer is
   non-NULL */
struct sym_sym_t *			/* symbol found, or NULL */
sym_bind_addr(md_addr_t addr,		/* address of symbol to locate */
	      int *pindex,		/* ptr to index result var */
	      int exact,		/* require exact address match? */
	      enum sym_db_t db){return NULL;}	/* symbol database to search */

/* bind name NAME to a symbol in symbol database DB, the index of the symbol
   in the requested symbol database is returned in *PINDEX if the pointer is
   non-NULL */
struct sym_sym_t *				/* symbol found, or NULL */
sym_bind_name(char *name,			/* symbol name to locate */
	      int *pindex,			/* ptr to index result var */
	      enum sym_db_t db){return NULL;}		/* symbol database to search */



