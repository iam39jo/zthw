
#include <stdio.h>
#include <stdlib.h>

#include "host.h"
#include "misc.h"
#include "machine.h"
#include "endian.h"
#include "regs.h"
#include "memory.h"
#include "sim.h"
#include "eio.h"
#include "loader.h"

/* amount of tail padding added to all loaded text segments */
#define TEXT_TAIL_PADDING 0 /* was: 128 */

md_addr_t ld_text_base = 0;
unsigned int ld_text_size = 0;
md_addr_t ld_data_base = 0;
md_addr_t ld_brk_point = 0;
unsigned int ld_data_size = 0;
unsigned int ld_stack_size = 0;
md_addr_t ld_stack_min = -1;
md_addr_t ld_environ_base = 0;

/* program base and bound */
md_addr_t ld_program_base = 0xffffffff;
md_addr_t ld_program_bound = 0;

/* program text (code) size in bytes */
unsigned int ld_program_size = 0;


/* program stack segment base (highest address in stack) */
md_addr_t ld_stack_base = 0;

/* program file name */
char *ld_prog_fname = NULL;

/* program entry point (initial PC) */
md_addr_t ld_prog_entry = 0;

/* target executable endian-ness, non-zero if big endian */
int ld_target_big_endian;

/* register simulator-specific statistics */
void
ld_reg_stats(struct stat_sdb_t *sdb)	/* stats data base */
{
  stat_reg_addr(sdb, "ld_program_base",
		"program base",
		&ld_program_base, ld_program_base, "0x%08p");
  stat_reg_addr(sdb, "ld_program_bound",
		"program bound",
		&ld_program_bound, ld_program_bound, "0x%08p");
  stat_reg_uint(sdb, "ld_program_size",
		"program  size in bytes",
		&ld_program_size, ld_program_size, NULL);
/*  stat_reg_addr(sdb, "ld_data_base",
		"program initialized data segment base",
		&ld_data_base, ld_data_base, "0x%08p");
  stat_reg_addr(sdb, "ld_data_bound",
		"program initialized data segment bound",
		&ld_data_bound, ld_data_bound, "0x%08p");
  stat_reg_uint(sdb, "ld_data_size",
		"program init'ed `.data' and uninit'ed `.bss' size in bytes",
		&ld_data_size, ld_data_size, NULL);
*/
  stat_reg_addr(sdb, "ld_stack_base",
		"program stack segment base (highest address in stack)",
		&ld_stack_base, ld_stack_base, "0x%08p");
#if 0 /* FIXME: broken... */
  stat_reg_addr(sdb, "ld_stack_min",
		"program stack segment lowest address",
		&ld_stack_min, ld_stack_min, "0x%08p");
#endif
/*
  stat_reg_uint(sdb, "ld_stack_size",
		"program initial stack size",
		&ld_stack_size, ld_stack_size, NULL);
  */
  stat_reg_addr(sdb, "ld_prog_entry",
		"program entry point (initial PC)",
		&ld_prog_entry, ld_prog_entry, "0x%08p");
  stat_reg_int(sdb, "ld_target_big_endian",
	       "target executable endian-ness, non-zero if big endian",
	       &ld_target_big_endian, ld_target_big_endian, NULL);
}


/* load program text and initialized data into simulated virtual memory
   space and initialize program segment range variables */
void
ld_load_prog(char *fname,		/* program to load */
	     int argc, char **argv,	/* simulated program cmd line args */
	     char **envp,		/* simulated program environment */
	     struct regs_t *regs,	/* registers to initialize for load */
	     struct mem_t *mem,		/* memory space to load prog into */
	     int zero_bss_segs)		/* zero uninit data segment? */
{
	int _length;
	byte_t * buffer=NULL;
	FILE * fobj;
	
	fobj = fopen (argv[0], "r");
	if(!fobj)
		fatal("cannot open executable '%s' ", argv[0]);

	if (fread(&_length, sizeof(int), 1, fobj) <1)
		fatal("error reading file length from '%s'", argv[0]);
	if (fread(&ld_prog_entry, sizeof(md_addr_t), 1, fobj) <1)
		fatal("error reading program entry from '%s'", argv[0]);
	
printf("%d,%08x\n",_length,ld_prog_entry);	
	/*printf("%d,%d\n",_length,_program_entry);*/
	buffer = (char *)calloc(_length, sizeof(word_t));
	if(!buffer)
		fatal("out of virtual memory");
	fread(buffer,sizeof(word_t),_length,fobj);
	/* copy the program into simulator target memory
	 *        func      , mem, R/W, begin_address, pro, length*/
	mem_bcopy(mem_access, mem, Write, 0x00000200, buffer, (_length * sizeof(word_t)));

      	ld_program_base = 0x00000200;
	ld_program_size = ld_program_base + _length; 
	ld_program_bound = ld_program_base + ld_program_size;
      	ld_stack_base = 0xc0000000;
  
	regs->regs_R[MD_REG_SP] = ld_stack_base;
	regs->regs_PC = ld_prog_entry;

}
