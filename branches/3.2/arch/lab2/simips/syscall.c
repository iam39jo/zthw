/*
 * syscall.c - proxy system call handler routines
 *
 * This file is a part of the SimpleScalar tool suite written by
 * Todd M. Austin as a part of the Multiscalar Research Project.
 *  
 * The tool suite is currently maintained by Doug Burger and Todd M. Austin.
 * 
 * Copyright (C) 1994, 1995, 1996, 1997 by Todd M. Austin
 *
 * This source file is distributed "as is" in the hope that it will be
 * useful.  The tool set comes with no warranty, and no author or
 * distributor accepts any responsibility for the consequences of its
 * use. 
 * 
 * Everyone is granted permission to copy, modify and redistribute
 * this tool set under the following conditions:
 * 
 *    This source code is distributed for non-commercial use only. 
 *    Please contact the maintainer for restrictions applying to 
 *    commercial use.
 *
 *    Permission is granted to anyone to make or distribute copies
 *    of this source code, either as received or modified, in any
 *    medium, provided that all copyright notices, permission and
 *    nonwarranty notices are preserved, and that the distributor
 *    grants the recipient permission for further redistribution as
 *    permitted by this document.
 *
 *    Permission is granted to distribute this file in compiled
 *    or executable form under the same conditions that apply for
 *    source code, provided that either:
 *
 *    A. it is accompanied by the corresponding machine-readable
 *       source code,
 *    B. it is accompanied by a written offer, with no time limit,
 *       to give anyone a machine-readable copy of the corresponding
 *       source code in return for reimbursement of the cost of
 *       distribution.  This written offer must permit verbatim
 *       duplication by anyone, or
 *    C. it is distributed by someone who received only the
 *       executable form, and is accompanied by a copy of the
 *       written offer of source code that they received concurrently.
 *
 * In other words, you are welcome to use, share and improve this
 * source file.  You are forbidden to forbid anyone else to use, share
 * and improve what you give them.
 *
 * INTERNET: dburger@cs.wisc.edu
 * US Mail:  1210 W. Dayton Street, Madison, WI 53706
 *
 * $Id: syscall.c,v 1.1.1.1 2000/11/29 14:53:54 cu-cs Exp $
 *
 * $Log: syscall.c,v $
 * Revision 1.1.1.1  2000/11/29 14:53:54  cu-cs
 * Grand unification of arm sources.
 *
 */

#include <stdio.h>
#include <stdlib.h>

/* only enable a minimal set of systen call proxies if on limited
   hosts or if in cross endian live execution mode */
#ifndef MIN_SYSCALL_MODE
#if defined(_MSC_VER) || defined(__CYGWIN32__) || defined(MD_CROSS_ENDIAN)
#define MIN_SYSCALL_MODE
#endif
#endif /* !MIN_SYSCALL_MODE */

/* live execution only support on same-endian hosts... */
#ifdef _MSC_VER
#include <io.h>
#else /* !_MSC_VER */
#include <unistd.h>
#endif
#include <fcntl.h>
#include <sys/types.h>
#ifndef _MSC_VER
#include <sys/param.h>
#endif
#include <errno.h>
#include <time.h>
#ifndef _MSC_VER
#include <sys/time.h>
#endif
#ifndef _MSC_VER
#include <sys/resource.h>
#endif
#include <signal.h>
#ifndef _MSC_VER
#include <sys/file.h>
#endif
#include <sys/stat.h>
#ifndef _MSC_VER
#include <sys/uio.h>
#endif
#include <setjmp.h>
#ifndef _MSC_VER
#include <sys/times.h>
#endif
#include <limits.h>
#ifndef _MSC_VER
#include <sys/ioctl.h>
#endif
#if defined(linux)
#include <utime.h>
#include <dirent.h>
#include <sys/vfs.h>
#endif
#if defined(_AIX)
#include <sys/statfs.h>
#else /* !_AIX */
#ifndef _MSC_VER
#include <sys/mount.h>
#endif
#endif /* !_AIX */
#if !defined(linux) && !defined(sparc) && !defined(hpux) && !defined(__hpux) && !defined(__CYGWIN32__) && !defined(ultrix)
#ifndef _MSC_VER
#include <sys/select.h>
#endif
#endif
#ifdef linux
#include <sgtty.h>
#include <netinet/tcp.h>
#include <netinet/udp.h>
#include <netinet/in.h>
#endif /* linux */

#if defined(__svr4__)
#include <sys/dirent.h>
#include <sys/filio.h>
#elif defined(__osf__)
#include <dirent.h>
/* -- For some weird reason, getdirentries() is not declared in any
 * -- header file under /usr/include on the Alpha boxen that I tried
 * -- SS-Alpha on. But the function exists in the libraries.
 */
int getdirentries(int fd, char *buf, int nbytes, long *basep);
#endif

#if defined(__svr4__) || defined(__osf__)
#include <sys/statvfs.h>
#define statfs statvfs
#include <sys/time.h>
#include <utime.h>
#include <sgtty.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#endif

#if defined(sparc) && defined(__unix__)
#if defined(__svr4__) || defined(__USLC__)
#include <dirent.h>
#else
#include <sys/dir.h>
#endif

#if 0
/* dorks */
#undef NL0
#undef NL1
#undef CR0
#undef CR1
#undef CR2
#undef CR3
#undef TAB0
#undef TAB1
#undef TAB2
#undef XTABS
#undef BS0
#undef BS1
#undef FF0
#undef FF1
#undef ECHO
#undef NOFLSH
#undef TOSTOP
#undef FLUSHO
#undef PENDIN
#endif

#if defined(hpux) || defined(__hpux)
#undef CR0
#endif

#ifdef __FreeBSD__
#include <sys/ioctl_compat.h>
#else
#ifndef _MSC_VER
#include <termio.h>
#endif
#endif

#if defined(hpux) || defined(__hpux)
/* et tu, dorks! */
#undef HUPCL
#undef ECHO
#undef B50
#undef B75
#undef B110
#undef B134
#undef B150
#undef B200
#undef B300
#undef B600
#undef B1200
#undef B1800
#undef B2400
#undef B4800
#undef B9600
#undef B19200
#undef B38400
#undef NL0
#undef NL1
#undef CR0
#undef CR1
#undef CR2
#undef CR3
#undef TAB0
#undef TAB1
#undef BS0
#undef BS1
#undef FF0
#undef FF1
#undef EXTA
#undef EXTB
#undef B900
#undef B3600
#undef B7200
#undef XTABS
#include <sgtty.h>
#include <utime.h>
#endif

#ifdef __CYGWIN32__
#include <sys/unistd.h>
#include <sys/vfs.h>
#endif

#include <sys/socket.h>
#include <sys/poll.h>

#ifdef _MSC_VER
#define access		_access
#define chmod		_chmod
#define chdir		_chdir
#define unlink		_unlink
#define open		_open
#define creat		_creat
#define pipe		_pipe
#define dup		_dup
#define dup2		_dup2
#define stat		_stat
#define fstat		_fstat
#define lseek		_lseek
#define read		_read
#define write		_write
#define close		_close
#define getpid		_getpid
#define utime		_utime
#include <sys/utime.h>
#endif /* _MSC_VER */
#endif

#include "host.h"
#include "misc.h"
#include "machine.h"
#include "regs.h"
#include "memory.h"
#include "loader.h"
#include "sim.h"
#include "endian.h"
#include "eio.h"
#include "syscall.h"

/* Syscall numbers */

#define SIMIPS_SYS_ni_syscall	0
#define SIMIPS_SYS_exit			1
#define SIMIPS_SYS_print			2
#define SIMIPS_SYS_delim		3

/* internal system call buffer size, used primarily for file name arguments,
   argument larger than this will be truncated */
#define MAXBUFSIZE 		1024


/* ARM SYSTEM CALL CONVENTIONS

	System call conventions as taken from unistd.h of arm architecture
	Depending on the call type r0, r1, r2, r3, r4 will contain the arguments
	of the system call.
	The actual system call number is found out using the link register after 
	an swi call is made. How this is done is shown in 
	kernel/arch/arm/kernel/entry-common.S file
	We need to decode the instruction to find out the system call number.
	
	Return value is returned in register 0. If the return value is between -1
	and -125 inclusive then there's an error.
*/

/* syscall proxy handler, architect registers and memory are assumed to be
   precise when this function is called, register and memory are updated with
   the results of the sustem call */

void
sys_syscall(struct regs_t *regs,	/* registers to access */
	    mem_access_fn mem_fn,	/* generic memory accessor */
	    struct mem_t *mem,		/* memory space to access */
	    md_inst_t inst,		/* system call inst */
	    int traceable)		/* traceable system call? */
{
  /* Figure out the system call number from the swi instruction
	  Last 24 bits gives us this number */

  word_t syscode = inst & 0x000fffff;


  /* first, check if an EIO trace is being consumed... */
  /* Left from Alpha code should be the same in ARM though */

  if (traceable && sim_eio_fd != NULL)
    {
      eio_read_trace(sim_eio_fd, sim_num_insn, regs, mem_fn, mem, inst);

#if 0
      /* kludge fix for sigreturn(), it modifies all registers */
      if (syscode == ARM_SYS_sigreturn)
	{
	  int i;
	  struct osf_sigcontext sc;
	  md_addr_t sc_addr = regs->regs_R[MD_REG_R0]; /* change the register name to r0 (first arg) */

	  mem_bcopy(mem_fn, mem, Read, sc_addr, 
		    &sc, sizeof(struct osf_sigcontext));
	  regs->regs_NPC = sc.sc_pc;
	  for (i=0; i < 32; ++i)
	    regs->regs_R[i] = sc.sc_regs[i];
	  for (i=0; i < 32; ++i)
	    regs->regs_F.q[i] = sc.sc_fpregs[i];
	  regs->regs_C.fpcr = sc.sc_fpcr;
	}
#endif

      /* fini... */
      return;
    }

  /* no, OK execute the live system call... */

  switch (syscode)
    {
    case SIMIPS_SYS_exit:
      /* exit jumps to the target set in main() */
      longjmp(sim_exit_buf,/* exitcode + fudge */0);
      break;

    case SIMIPS_SYS_print:
      {
	     md_addr_t _begin_addr;
	     word_t _length;
	     int * data = NULL;
	     int i;

	     _begin_addr = regs->regs_R[1];
	     _length = regs->regs_R[2];
	     
	     data = (int *) calloc(_length, sizeof(int));
	     mem_bcopy(mem_fn, mem, Read, _begin_addr, (char *)data, (sizeof(int)*_length));
	     for(i=0;i<_length;i++ )
		     printf("%d\n", data[i]);
      }
      break;

		case SIMIPS_SYS_delim:
			{
				printf("----------------------\n");
				break;
			}

    default:
      warn("invalid/unimplemented syscall %d, PC=0x%08p, winging it",
	   (int)syscode, regs->regs_PC);
    }

  if (verbose)
    fprintf(stderr, "syscall(%d):\n",(int)syscode);
}
