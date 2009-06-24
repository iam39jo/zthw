# insert sort program using simips instruction set

DATA SEG:
	10	#number count
	12,45,1,4,7,5,0,3,80,2		#numbers going to sort
DATA END

CODE SEG:
	addi	$3, $0,	0x00000200
	lw		$4,	0($3)

	add		$2,	$4,	$0
	addi	$1,	$3,	4
	swi		2
	swi		3

	sll		$4, $4, 2
	add		$4, $4, $2		#up-lim of array

	addi		$5,	$2,	4


	swi		1
CODE END
