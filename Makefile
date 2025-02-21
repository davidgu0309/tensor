.PHONY: test clean

test: src/util.cpp test/run_unit_tests.cpp
	g++ --std=c++17 -o test_binary -g src/util.cpp test/run_unit_tests.cpp && ./test_binary

clean:
	rm -f test_binary

