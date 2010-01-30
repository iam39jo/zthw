#!/usr/bin/env python

def isPrimeUnder20(num):
    for tmp in range(2, num):
        if num%tmp == 0:
            return False
    return True

factor_list = []

for x in range(2, 20):
    if isPrimeUnder20(x):
        power_x = 1
        while x**power_x <= 20:
            power_x += 1
        factor_list.append(x**(power_x-1))
        print x,power_x-1

prod = 1
for x in factor_list:
    prod *= x

print prod
