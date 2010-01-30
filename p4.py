#!/usr/bin/env python

biggest = 0

def isPalindromic(num):
    num_str = str(num)
    num_halflen = len(num_str)/2
    for idx in range(0, num_halflen):
        if num_str[idx] != num_str[-(idx+1)]:
            return False
    return True

for x in range(100, 999):
    for y in range(100, 999):
        if isPalindromic(x*y) and x*y > biggest:
            biggest = x*y

print biggest
