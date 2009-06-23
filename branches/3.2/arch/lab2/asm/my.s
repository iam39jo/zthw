#demo.s
#implement swap operation on 2 integers stored in memory
#$1,$2 for swi, $3=length, $4=MEM Base, $5=number1, $6=number2, $0=0


DATA SEG:
#number(length) of numbers (count in words)
	2
#first number
	1
#second number
	2
DATA END

CODE SEG:
	addi	$4,	$0,	0x00000200
	lw	$3,	0($4)			#load length in $3

	add 	$2,	$3,	$0
	addi	$1,	$4,	4
	swi	2				#display origion numbers, should be 1 2

	lw	$5,	4($4)
	lw	$6,	8($4)
#	swp	$5,	$6
	sw	$5,	4($4)
	sw	$6,	8($4)			#swap numbers and restore

	swi	2				#display swapped numbers, should be 2 1
	swi	1				#exit
CODE END
