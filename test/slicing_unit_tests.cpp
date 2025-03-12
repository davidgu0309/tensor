#include "test_tensors.hpp"

void slicingUnitTests() {
    std::cout << iota_2x2x3 << "\n";
    MultiRange multi_range1 = {{0,1}, {0,2}, {0,3}};
    std::cout << iota_2x2x3.slice(multi_range1) << "\n";

    MultiRange multi_range2 = {{0,2}, {0,2}, {0,1}};
    std::cout << iota_2x2x3.slice(multi_range2) << "\n";

    MultiRange multi_range3 = {{1,2}, {0,1}, {2,3}};
    std::cout << iota_2x2x3.slice(multi_range3) << "\n";

    std::cout << "--------------------------\n";

    std::vector<tensor::Tensor<int>> iota_2x2x3_unstacked = iota_2x2x3.unstack(2);
    for (tensor::Tensor<int> sub_tensor : iota_2x2x3_unstacked) {
        std::cout << sub_tensor << "\n";
    }

    std::cout << "--------------------------\n";

    std::vector<tensor::Tensor<int>> iota_2x2x3_unstacked2 = iota_2x2x3.unstack(0);
    for (tensor::Tensor<int> sub_tensor : iota_2x2x3_unstacked2) {
        std::cout << sub_tensor << "\n";
    }

    std::cout << "--------------------------\n";
    std::cout << tensor::aggregate<int, aggregator::sum<int>>(iota_2x2x3, 0) << "\n";

    std::cout << "--------------------------\n";
    std::cout << tensor::aggregate<int, aggregator::sum<int>>(iota_2x2x3, 1) << "\n";

    std::cout << "--------------------------\n";
    std::cout << tensor::aggregate<int, aggregator::sum<int>>(iota_2x2x3, 2) << "\n";

    std::cout << "--------------------------\n";
    std::cout << tensor::aggregate<int, aggregator::mean<int>>(iota_2x2x3, 2) << "\n";
}