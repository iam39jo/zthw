#!/usr/bin/env python  

sum = 0  
num1 = 0  
num2 = 1  
while num2 <= 4000000:  
    print num2  
    if num2 % 2 == 0:  
        sum += num2  
    temp = num2  
    num2 = num1+num2  
    num1 = temp  
print sum  

