#include "constructor_unit_tests.cpp"
#include "functional_unit_tests.cpp"
// #include "scalar_operation_unit_tests.cpp" // Besed on new framework

#include <iostream>

int main() {

    std::cout << "Running constructor unit tests." << std::endl;
    constructorUnitTests();

    std::cout << "Running functional unit tests." << std::endl;
    functionalUnitTests();
    
    // std::cout << "Running scalar operation unit tests" << std::endl;
    // tinytorch::scalarOpUnitTests();
}