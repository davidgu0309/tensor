// TO DO: find cleaner way to do this, maybe with some macros

#pragma once

#include "../include/tensor.hpp"

namespace tensor{

Tensor<int> zeros_3x4x5 = zeros<int>({3, 4, 5});
Tensor<int> ones_5 = ones<int>({5});
Tensor<int> twos_5 = constant<int>({5}, 2);
Tensor<int> threes_5 = constant<int>({5}, 3);
Tensor<int> ones_10 = ones<int>({10});
Tensor<int> ones_3x3 = ones<int>({3, 3});
Tensor<int> ones_5x5 = ones<int>({5, 5});
Tensor<int> ones_3x2x4 = ones<int>({3, 2, 4});
Tensor<int> id_5x5 = idLeft<int>({5, 5});
Tensor<int> t1 = Tensor<int>({3, 2, 4}, std::vector<int>(24, -1));
Tensor<int> t2 = Tensor<int>({10}, std::vector<int>(10, -1));
Tensor<int> t3 = Tensor<int>({3, 3}, std::vector<int>(9, -1));
Tensor<int> threes_3x3 = constant<int>({3, 3}, 3);
Tensor<int> iota_5 = iota<int>({5});
Tensor<int> scalar_1 = Tensor<int>(1);
Tensor<int> scalar_10 = Tensor<int>(10);
Tensor<int> scalar_15 = Tensor<int>(15);
Tensor<int> scalar_55 = Tensor<int>(55);
Tensor<int> iota_3x3 = iota<int>({3,3});
Tensor<int> iota_2x2x3 = iota<int>({2,2,3});
Tensor<int> iota_3x3_squared = Tensor<int>({3,3}, std::vector<int>({30,36,42,66,81,96,102,126,150}));
Tensor<int> iota_3D_times_2D_result = Tensor<int>({2,2,3}, std::vector<int>{30,36,42,66,81,96,102,126,150,138,171,204});

} // namespace tensor