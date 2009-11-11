
all: test

test: AES.o ECB.o Text_Operations.o test_main.o
	gcc test_main.o AES.o ECB.o Text_Operations.o -o test

clean:
	rm -rf *.o test
