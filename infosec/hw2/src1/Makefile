
all: test

test: AES.o ECB.o Text_Operations.o test_main.o CBC.o CFB.o OFB.o CTR.o
	gcc test_main.o AES.o ECB.o Text_Operations.o CBC.o CFB.o OFB.o CTR.o -o test

clean:
	rm -rf *.o test
