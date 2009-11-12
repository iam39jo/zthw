
all: test

test: AES.o ECB.o Text_Operations.o CBC.o CFB.o OFB.o CTR.o ui.o
	gcc ui.o AES.o ECB.o Text_Operations.o CBC.o CFB.o OFB.o CTR.o -o test

clean:
	rm -rf *.o test
