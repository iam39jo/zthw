#!/usr/bin/env python

import string

fin=open('my.s', 'r')
fout=open('my.txt', 'wb')

data_len = 0
code_len = 0
parser_status = 0
lineno = 0
delimiter = "1073741824"

def get_reg_no(token):
    return int(token.strip()[1:])

def get_imm(token):
    if token[0:2] == "0x" or token[0:2] == "0X":
        return int(token, 16)
    elif token[0:2] == "0b" or token[0:2] == "0B":
        return int(token, 2)
    else:
        return int(token)

def parse_instr(statement):
    instr_fields = oneline.split(None, 1)
    instr_fields[0] = instr_fields[0].strip()
    if instr_fields[0] == "sll":
        op_fields = instr_fields[1].split(',')

        return 0

    elif instr_fields[0] == "add":
        op_fields = instr_fields[1].split(',')
        op_rd = get_reg_no(op_fields[0])
        op_rs = get_reg_no(op_fields[1])
        op_rt = get_reg_no(op_fields[2])
        instruction = (0b000000 << 26) | (0b100000);
        instruction = instruction | (op_rs << 21) | (op_rt << 16) | (op_rd << 11)
        return instruction
    
    elif instr_fields[0] == "sub":
        op_fields = instr_fields[1].split(',')
        op_rd = get_reg_no(op_fields[0])
        op_rs = get_reg_no(op_fields[1])
        op_rt = get_reg_no(op_fields[2])
        instruction = (0b000000 << 26) | (0b100010);
        instruction = instruction | (op_rs << 21) | (op_rt << 16) | (op_rd << 11)
        return instruction

    elif instr_fields[0] == "addi":
        op_fields = instr_fields[1].split(',')
        op_rs = get_reg_no(op_fields[1])
        op_rt = get_reg_no(op_fields[0])
        imme = get_imm(op_fields[2].strip())
        instruction = (0b001000 << 26)
        instruction = instruction | (op_rs << 21) | (op_rt << 16) | (imme & 0xffff)
        return instruction

    elif instr_fields[0] == "beq":
        op_fields = instr_fields[1].split(',')
        op_rs = get_reg_no(op_fields[0])
        op_rt = get_reg_no(op_fields[1])
        imme = get_imm(op_fields[2].strip())
        instruction = (0b000100 << 26)
        instruction = instruction | (op_rs << 21) | (op_rt << 16) | (imme & 0xffff)
        return instruction

    elif instr_fields[0].strip() == "sw":
        op_fields = instr_fields[1].split(',')
        op_rt = get_reg_no(op_fields[0])
        imme = get_imm(op_fields[1][0:op_fields[1].find('(')])
        op_rs = get_reg_no(op_fields[1][op_fields[1].find('(')+1:op_fields[1].find(')')])
        instruction = (0b101011 << 26);
        instruction = instruction | (op_rs << 21) | (op_rt << 16) | (imme & 0xffff)
        return instruction 

    elif instr_fields[0] == "lw":
        op_fields = instr_fields[1].split(',')
        op_rt = get_reg_no(op_fields[0])
        imme = get_imm(op_fields[1][0:op_fields[1].find('(')])
        op_rs = get_reg_no(op_fields[1][op_fields[1].find('(')+1:op_fields[1].find(')')])
        instruction = (0b100011 << 26);
        instruction = instruction | (op_rs << 21) | (op_rt << 16) | (imme & 0xffff)
        return instruction

    elif instr_fields[0] == "swi":
        imme = get_imm(instr_fields[1].strip())
        instruction = (0b111111 << 26);
        instruction = instruction | (imme);
        return instruction

    else:
        print("Invalid instruction @ "+str(lineno)+": "+oneline)
        exit(2)

for oneline in fin:
    lineno = lineno + 1
    oneline = oneline.strip()

    if len(oneline) == 0:
        continue

    ## Clear comments
    if oneline[0] == '#':
        continue
    oneline = oneline.split('#')[0].strip()


    if parser_status == 0:
        if oneline == "DATA SEG:":
            parser_status = 1
            continue
        elif oneline == "CODE SEG:":
            parser_status = 10
            # write the delimiter for data segment
            fout.write(delimiter+'\n')
            continue
        else:
            print("Unexpected token @ "+str(lineno)+": "+oneline)
            exit()
            
            
    if parser_status == 1:
        if oneline == "DATA END":
            parser_status = 2
            fout.write(delimiter+'\n')
            continue
        for eachnum in oneline.split(','):
            if eachnum.isalnum():
                fout.write(eachnum+'\n')
                continue
            else:
                print("Invalid numberic @ "+str(lineno)+": "+oneline)
                exit()

    if parser_status == 2:
        if oneline == "CODE SEG:":
            parser_status = 10
            continue
        else:
            print("Invalid statement @ "+str(lineno)+": "+oneline)
            exit()

    if parser_status == 10:
        if oneline == "CODE END":
            parser_status = 20
            fout.write(delimiter+'\n')
            continue
        else:
            instruction = parse_instr(oneline.strip())
            fout.write(str(instruction)+'\n')
            continue

    if parser_status == 20:
        if len(oneline) > 0:
            print("Invalid statement @ "+str(lineno)+": "+oneline)
            exit()

if parser_status != 20:
    print("Unexpected file end!")
    exit()




