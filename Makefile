.PHONY: test clean

test: 
	g++ --std=c++20 -o test_binary -g src/util.cpp test/run_unit_tests.cpp && ./test_binary

clean:
	rm -f test_binary

