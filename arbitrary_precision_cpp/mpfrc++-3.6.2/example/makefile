example: example.o
	g++ -std=c++11 example.o -o example -lmpfr -lgmp

example.o: example.cpp
	g++ -std=c++11 -c example.cpp

clean:
	rm -f *.o
