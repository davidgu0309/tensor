/**
 * @file tensor.hpp
 * 
 * @brief Dynamic size tensors templated for type.
 * 
 * @author David Gu
 * @author Mirco Paul
 * 
 * @date \today
 */
#pragma once

#include "distribution.hpp"
#include "util.hpp"
#include "aggregators.hpp"

#include <cassert>
#include <iostream>
// #include <memory>
#include <numeric>
#include <queue>
#include <vector>
#include <functional>



// TO DO: implement .copy()

/**
 * @namespace tensor
 * 
 * @brief Namespace of the entire framework.
 * 
 */
namespace tensor {

    /**
     * @typedef Shape
     * 
     * @brief Type for tensor shapes.
     * 
     */
    typedef std::vector<size_t> Shape;

    /**
     * @brief Prints shape shape to outputstream out.
     */
    std::ostream& operator << (std::ostream& out, const Shape& shape);

    /**
     * @typedef MultiIndex
     * 
     * @brief 0-based tensor indexes.
     * 
     */
    typedef std::vector<size_t> MultiIndex;

    struct Range {
        size_t from, to;
    };

    /**
     * @typedef MultiIndex
     * 
     * @brief 0-based tensor indexes.
     * 
     */
    typedef std::vector<Range> MultiRange;


    /**
     * @brief Checks if multi_index is valid for shape.
     */
    bool multiIndexLegalityTest(const Shape shape, const MultiIndex multi_index);

    /**
     * @brief Prints multiindex index to outputstream out.
     */
    std::ostream& operator << (std::ostream& out, const MultiIndex& index);

    /**
     * @class Tensor
     * 
     * @brief Dynamic-size tensor.
     * 
     * @todo Comment members.
     * 
     * @tparam T Entry type.
     */
    template <typename T>
    class Tensor {

        // Default visibility is private
        Shape shape_;
        std::vector<T> data_;

    public:

        Tensor();   // Doesn't do anything, but is necessary
        Tensor(const T value);    // Returns scalar (shape {})
        Tensor(const std::vector<size_t> shape);
        Tensor(const std::vector<size_t> shape,
                    const std::vector<T>& data);        

        size_t size() const;
        const Shape& shape() const;
        Shape& shape();


        std::vector<T>& data();

        const std::vector<T>& data() const;

        void clear();   /** Zeroes the data. */

        T& getEntryUnsafe(MultiIndex index);
        const T& getEntryUnsafe(MultiIndex index) const;

        T& getEntrySafe(MultiIndex index);
        const T& getEntrySafe(MultiIndex index) const;

        Tensor<T> slice(MultiRange multi_range) const;
        std::vector<Tensor<T>> unstack(size_t d) const;

        // Comparison operators
        bool operator == (const Tensor<T>& other) const;
        bool shapeEqual(const Tensor<T>& other) const;

        template <typename U>
        friend std::ostream& operator << (std::ostream& out, const Tensor<U>& tensor);

    };

    // TODO: maybe move somewhere
    template <typename T>
    T kroneckerDelta(const MultiIndex i);

    // Common tensors
    template <typename T>
    Tensor<T> zeros(const Shape shape);

    template <typename T>
    Tensor<T> ones(const Shape shape);

    template <typename T>
    Tensor<T> constant(const Shape shape, T value);

    template <typename T>
    Tensor<T> idLeft(const Shape shape) ;

    template <typename T>
    Tensor<T> iota(const Shape shape);

    // TO DO: identity

    // TO DO: random
    template <typename T>
    Tensor<T> initializeWithGenerator(const Shape shape, std::function<T(MultiIndex)> generator);

    template <typename T>
    Tensor<T> realUniform(const Shape shape, const T lower, const T upper);

    std::vector<MultiIndex> indexesRowMajor(const Shape shape);
    MultiIndex concatIndexes(const MultiIndex& i, const MultiIndex& j);
    MultiIndex combineIndexes(const MultiIndex& i, const MultiIndex& j);

    template<typename T, T(*aggregator)(const std::vector<T>&)> 
    Tensor<T> aggregate(const Tensor<T>& tensor, size_t axis);
}

#include "../src/tensor.tpp"

