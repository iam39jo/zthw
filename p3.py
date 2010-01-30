#!/usr/bin/env python  

num = 600851475143
seg_size = 10000
base_factor = 0
prime_factors = []

while base_factor*seg_size < num:
    for tmp in range(seg_size*base_factor, seg_size*(base_factor+1)):
        if tmp>1 and num%tmp == 0:
            prime_factors.append(tmp)
            while num%tmp == 0:
                num = num / tmp
            print "num:",num," ",tmp
        base_factor = base_factor + 1

print prime_factors
