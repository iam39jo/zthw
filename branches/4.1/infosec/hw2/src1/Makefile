
all: test

test: AES.o ECB.o PlainText.o test_main.o
	gcc test_main.o AES.o ECB.o PlainText.o -o test

clean:
	rm -rf *.o test
