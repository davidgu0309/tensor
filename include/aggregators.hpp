/**
 * @file aggregators.hpp
 * 
 * @brief Implementation of common aggregators.
 * 
 * @author David Gu
 * @author Mirco Paul
 * 
 * @date \today
 */
#pragma once
#include <vector>

namespace aggregator {
    template <typename T>
    T sum(const std::vector<T>& operands);

    template <typename T>
    T mean(const std::vector<T>& operands);
}

#include "../src/aggregators.tpp"
