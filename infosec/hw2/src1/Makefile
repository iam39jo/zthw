
all: aes

aes: AES.o ECB.o Text_Operations.o CBC.o CFB.o OFB.o CTR.o ui.o
	gcc ui.o AES.o ECB.o Text_Operations.o CBC.o CFB.o OFB.o CTR.o -o aes

clean:
	rm -rf *.o aes
