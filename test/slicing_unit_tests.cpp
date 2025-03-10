#include "test_tensors.hpp"

void slicingUnitTests() {
    std::cout << iota_2x2x3 << "\n";
    MultiRange multi_range1 = {{0,1}, {0,2}, {0,3}};
    std::cout << iota_2x2x3.slice(multi_range1) << "\n";

    MultiRange multi_range2 = {{0,2}, {0,2}, {0,1}};
    std::cout << iota_2x2x3.slice(multi_range2) << "\n";

    MultiRange multi_range3 = {{1,2}, {0,1}, {2,3}};
    std::cout << iota_2x2x3.slice(multi_range3) << "\n";
}