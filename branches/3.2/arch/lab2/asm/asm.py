#!/usr/bin/env python

import string

fin=open('my.s', 'r')
fout=open('my.txt', 'wb')

data_len = 0
code_len = 0
parser_status = 0
lineno = 0

for oneline in fin:
    lineno = lineno + 1
    oneline=oneline.strip()

    if len(oneline) == 0:
        continue

    if oneline[0] == '#':
        continue
    
    if parser_status == 0:
        if oneline == "DATA SEG:":
            parser_status = 1
            fout.write("DATA"+'\n')
            continue
        elif oneline == "CODE SEG:":
            parser_status = 10
            fout.write("CODE"+'\n')
            continue
        else:
            print("Unexpected token @ "+str(lineno)+": "+oneline)
            exit()

    if parser_status == 1:
        if oneline.isalnum():
            fout.write(oneline+'\n')
            continue
        elif oneline == "DATA END":
            parser_status = 2
            fout.write("END"+'\n')
            continue
        else:
            print("Invalid numberic @ "+str(lineno)+": "+oneline)
            exit()



