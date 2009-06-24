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
	add		$4, $4, $1		#up-lim of array
	addi	$5,	$1,	4			#for j = 1 to lenA, $5:j

	slt		$10, $5, $4
	beq		$10, $0, 15		#to for-end

	lw		$6,	 0($5)		#$6:key
	addi	$7,	 $5, -4		#$7:i

	slt		$10, $3, $7
	beq		$10, $0, 7		#to end-while
	lw		$8, 0($7)
	slt		$10, $6, $8
	beq		$10, $0, 4		#to end-while
	addi	$9, $7, 4
	sw		$8, 0($9)
	addi	$7, $7, -4
	beq		$0, $0, -9			#start-while

	addi	$9, $7, 4
	sw		$6, 0($9)
	addi	$5, $5, 4
	beq		$0, $0, -17
	
	swi 	2
	swi		1
CODE END
