#!/usr/bin/env python

sum_of_square = 0
for x in range(1, 101):
    sum_of_square += x**2

sum_tmp = 0
for x in range(1, 101):
    sum_tmp += x
square_of_sum = sum_tmp**2

delta = square_of_sum - sum_of_square
print delta
