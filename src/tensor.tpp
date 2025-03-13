namespace tensor {

    bool multiIndexLegalityTest(const Shape shape, const MultiIndex index){
        if(shape.size() != index.size()) return false;
        for(int i = 0; i < shape.size(); ++i){
            if(index[i] >= shape[i]) return false;
        }
        return true;
    }

    std::ostream& operator << (std::ostream& out, const Shape& shape){
        for(size_t i : shape){
            out << i << " ";
        }
        // out << std::endl;
        return out;
    }

    // Constructors
    template <typename T>
    Tensor<T>::Tensor(){}
    
    template <typename T>
    Tensor<T>::Tensor(const T value) : shape_({}), data_(std::vector<T>(1, value)){}

    template <typename T>
    Tensor<T>::Tensor(const Shape shape) : shape_(shape), data_(std::vector<T>(numEntries(shape))){}

    template <typename T>
    Tensor<T>::Tensor(const Shape shape, const std::vector<T>& data) : shape_(shape), data_(data){}


    template <typename T>
    size_t Tensor<T>::size() const {
        return data_.size();
    }

    template <typename T>
    const Shape& Tensor<T>::shape() const {
        return shape_;
    }

    template <typename T>
    Shape& Tensor<T>::shape() {
        return shape_;
    }


    template <typename T>
    std::vector<T>& Tensor<T>::data() {
        return data_;
    }

    template <typename T>
    const std::vector<T>& Tensor<T>::data() const {
        return data_;
    }

    template <typename T>
    void Tensor<T>::clear(){
        for(size_t i = 0; i < data_.size(); ++i) data_[i] = 0;
    }

    template <typename T>
    T& Tensor<T>::getEntryUnsafe(MultiIndex index){
        size_t data_pos = index.size() ? index[0] : 0;
        // shape_ = {2, 2, 3}
        // (index[0] * shape_[1] + index[1]) * shape_[2] + index[2]
        // (index[0] * shape_[1] * shape_[2] + index[1] * shape_[2] + index[2])
        /*
        _ _ _
        _ _ _

        _ _ _
        _ _ _

        Horner Scheme

        sum_i a_i * x ** i = (a_i * x + a_(i - 1)) * x + ...

        */
        for(size_t d = 1; d < index.size(); ++d){
            data_pos *= shape_[d];
            data_pos += index[d];
        }
        return data()[data_pos];
    }

    template <typename T>
    const T& Tensor<T>::getEntryUnsafe(MultiIndex index) const{
        size_t data_pos = index.size() ? index[0] : 0;
        for(size_t d = 1; d < index.size(); ++d){
            data_pos *= shape_[d];
            data_pos += index[d];
        }
        return data()[data_pos];
    }

    template <typename T>
    T& Tensor<T>::getEntrySafe(MultiIndex index){
        if(!multiIndexLegalityTest(shape_, index)){
            std::cout << "Shape " << shape_ << std::endl;
            std::cout << "Index " << index << std::endl;
            assert(0);
        }
        return getEntryUnsafe(index);
    }

    template <typename T>
    const T& Tensor<T>::getEntrySafe(MultiIndex index) const{
        if(!multiIndexLegalityTest(shape_, index)){
            std::cout << "Shape " << shape_ << std::endl;
            std::cout << "Index " << index << std::endl;
            assert(0);
        }
        return getEntryUnsafe(index);
    }


    // Comparison operators
    template <typename T>
    bool Tensor<T>::shapeEqual (const Tensor<T>& other) const {
        return shape() == other.shape();
    }

    template <typename T>
    bool Tensor<T>::operator == (const Tensor<T>& other) const {
        const std::vector<T>& other_data = other.data();
        return shapeEqual(other) && (data() == other_data);
    }


    // Common tensors
    template <typename T>
    Tensor<T> constant(const Shape shape, T value){
        std::vector<T> data(numEntries(shape), value);
        return Tensor(shape, data);
    }

    template <typename T>
    Tensor<T> zeros(const Shape shape) {
        return constant<T>(shape, 0);
    }

    template <typename T>
    Tensor<T> ones(const Shape shape) {
        return constant<T>(shape, 1);
    }

    template <typename T>
    Tensor<T> iota(const Shape shape){
        std::vector<T> data(numEntries(shape));
        std::iota(data.begin(), data.end(), 1);
        return Tensor(shape, data);
    }

    template <typename T>
    Tensor<T> initializeWithGenerator(const Shape shape, std::function<T(MultiIndex)> generator) {
        std::vector<T> data(numEntries(shape));
        std::vector<MultiIndex> indexes = indexesRowMajor(shape);
        for (int i=0; i<data.size(); i++) {
            data[i] = generator(indexes[i]);
        }
        return Tensor(shape, data);
    }

    template <typename T>
    Tensor<T> realUniform(const Shape shape, const T lower, const T upper) {
        std::random_device rd;
        std::mt19937 gen(rd());
        return initializeWithGenerator<T>(shape, [lower, upper, gen](MultiIndex i) mutable {
            std::uniform_real_distribution<T> dist(lower, upper);
            return dist(gen);
        });
    }

    template <typename T>
    T kroneckerDelta(const MultiIndex i){
        if(i.size() > 1){
            for(size_t j = 1; j < i.size(); ++j){
                if(i[j - 1] != i[j]) return (T)0;
            }
        }
        return (T)1;
    }

    template <typename T>
    Tensor<T> idLeft(const Shape shape) {
        if(!shape.size()) return Tensor<T>(1);
        Shape id_shape = {shape.front(), shape.front()};
        // Inefficient, but good to test initializeWithGenerator
        // TODO: rewrite
        return initializeWithGenerator<T>(id_shape, kroneckerDelta<T>);
    }

    template <typename T>
    Tensor<T> idRight(const Shape shape) {
        if(!shape.size()) return Tensor<T>(1);
        Shape id_shape = {shape.back(), shape.back()};
        // Inefficient, but good to test initializeWithGenerator
        // TODO: rewrite
        return initializeWithGenerator<T>(id_shape, kroneckerDelta<T>);
    }


    /**
     * Writes tensor to output stream out. This enables std::cout << tensor ...
     *
     * @tparam U Tensor entry type.
     * 
     * @param out Output stream.
     * @param tensor Tensor to print.
     * 
     * @return Updated output stream.
     * 
     **/
    template<typename U>
    std::ostream& operator << (std::ostream& out, const Tensor<U>& tensor){
        // Iterate over all indexes and print
        Shape shape = tensor.shape();
        // out << shape << std::endl;
        size_t n = shape.size(), m = tensor.size();
        for(size_t i = 0; i < m; ++i){
            out << tensor.data()[i] << " ";
            // Inefficient but doesn't matter
            size_t j = 0, temp = i + 1;
            while(j < n && !(temp % shape[n - 1 - j])){
                out << std::endl;
                temp /= shape[n - 1 - j];
                ++j;
            }
        }
        return out;
    }

    // Unnecessary and inefficient, but nice
    std::vector<MultiIndex> indexesRowMajor(const Shape shape){
        std::queue<MultiIndex> indexes; // Multiindexes in "row-major" order
        indexes.push({});
        size_t n = shape.size();
        //shape = {2, 2, 3}
        //
        // {}
        // {0}, {1}
        // {0, 0}, {0, 1}, {1, 0}, {1, 1}
        // ...
        for(size_t d = 0; d < n; ++d){
            size_t m = indexes.size();
            // Iterate over all multiindexes of the previous dimension
            for(size_t i = 0; i < m; ++i){
                // For each one, add all possible indexes for the current dimension
                for(size_t j = 0; j < shape[d]; ++j){
                    MultiIndex index = indexes.front();
                    index.push_back(j);
                    indexes.push(index);
                }
                indexes.pop();
            }
        }
        std::vector<MultiIndex> row_major(indexes.size());
        size_t i = 0;
        while(indexes.size()){
            row_major[i] = indexes.front();
            indexes.pop();
            ++i;
        }
        return row_major;
    }

    MultiIndex concatIndexes(const MultiIndex& i, const MultiIndex& j){
        size_t dim_j = j.size();
        MultiIndex ij = i;
        for(size_t d = 0; d < dim_j; ++d) ij.push_back(j[d]);
        return ij;
    }

    MultiIndex combineIndexes(const MultiIndex& i, const MultiIndex& j){
        size_t dim_j = j.size();
        MultiIndex ij = i;
        for(size_t d = 1; d < dim_j; ++d) ij.push_back(j[d]);
        return ij;
    }

    std::vector<MultiIndex> indexesSlice(const MultiRange multi_range){
        std::queue<MultiIndex> indexes; // Multiindexes in "row-major" order
        indexes.push({});
        size_t n = multi_range.size();
        //shape = {2, 2, 3}
        //
        // {}
        // {0}, {1}
        // {0, 0}, {0, 1}, {1, 0}, {1, 1}
        // ...
        for(size_t d = 0; d < n; ++d){
            size_t m = indexes.size();
            // Iterate over all multiindexes of the previous dimension
            for(size_t i = 0; i < m; ++i){
                // For each one, add all possible indexes for the current dimension
                for(size_t j = multi_range[d].from; j < multi_range[d].to; ++j){
                    MultiIndex index = indexes.front();
                    index.push_back(j);
                    indexes.push(index);
                }
                indexes.pop();
            }
        }
        std::vector<MultiIndex> row_major(indexes.size());
        size_t i = 0;
        while(indexes.size()){
            row_major[i] = indexes.front();
            indexes.pop();
            ++i;
        }
        return row_major;
    }

    Shape sliceShape(const MultiRange multi_range) {
        Shape sliced_shape = {};
        for (size_t d = 0; d<multi_range.size(); d++) {
            sliced_shape.push_back(multi_range[d].to - multi_range[d].from);
        }
        return sliced_shape;
    }

    template <typename T>
    Tensor<T> Tensor<T>::slice(MultiRange multi_range) const{
        std::vector<MultiIndex> indexes = indexesSlice(multi_range);
        Tensor<T> result(sliceShape(multi_range));

        size_t counter = 0;
        for(const MultiIndex& index : indexes){
            result.data()[counter] = getEntryUnsafe(index);
            counter++;
        }
        return result;
    }

    template <typename T>
    std::vector<Tensor<T>> Tensor<T>::unstack(size_t d) const{
        std::vector<Tensor<T>> result;
        MultiRange multi_range;
        for (size_t tensor_dim : shape_) {
            multi_range.push_back({0, tensor_dim});
        }
        for (size_t i=0; i<shape_[d]; i++) {
            multi_range[d] = {i, i+1};
            result.push_back(slice(multi_range));
            result.back().shape().erase(result.back().shape().begin() + d);
        }
        return result;
    }

    // // Concept to detect C-style strings (both char arrays and const char*)
    // template<typename T>
    // concept is_c_string = 
    //     (std::is_array_v<std::remove_reference_t<T>> && 
    //     std::same_as<std::remove_extent_t<std::remove_reference_t<T>>, char>) || 
    //     std::same_as<std::remove_cvref_t<T>, const char*>;

    // // Generic << operator for iterable types other than strings
    // template<typename T>
    // requires (std::ranges::range<T> && !std::same_as<T, std::string> && !is_c_string<T>)
    // std::ostream& operator<<(std::ostream& os, const T& container);

    // // Print first operand and recurse on the rest
    // template<typename T, typename... Args>
    // void rec(std::ostream& os, size_t operand_idx, const T& first, const Args&... rest) {
    //     os << "Operand " << operand_idx << ":\n" << first;
    //     if constexpr (sizeof...(rest) > 0) {
    //         os << std::endl;
    //         rec(os, operand_idx + 1, rest...);
    //     }
    // }

    // // Overload << operator for std::tuple
    // template<typename... Args>
    // std::ostream& operator<<(std::ostream& os, const std::tuple<Args...>& tup) {
    //     std::apply([&os](const auto&... args) { rec(os, 0, args...); }, tup);
    //     return os;
    // }

    // // Generic << operator for iterable types other than strings
    // template<typename T>
    // requires (std::ranges::range<T> && !std::same_as<T, std::string> && !is_c_string<T>)
    // std::ostream& operator<<(std::ostream& os, const T& container) {
    //     os << "[";
    //     bool first = true;
    //     for (const auto& item : container) {
    //         if (!first) os << ", ";
    //         os << item;
    //         first = false;
    //     }
    //     os << "]";
    //     return os;
    // }

    template<typename T, T(*aggregator)(const std::vector<T>&)>
    Tensor<T> aggregate(const Tensor<T>& tensor, size_t axis) {
        Shape result_shape = tensor.shape();
        result_shape.erase(result_shape.begin() + axis);
        Tensor<T> result(result_shape);
        std::vector<Tensor<T>> unstacked = tensor.unstack(axis);

        std::vector<MultiIndex> multi_indexes = indexesRowMajor(result_shape);
        for (const MultiIndex &multi_index : multi_indexes) {
            std::vector<T> operands;

            for (size_t i=0; i<unstacked.size(); i++) {
                operands.push_back(unstacked[i].getEntrySafe(multi_index));
            }
            result.getEntryUnsafe(multi_index) = aggregator(operands);
        }
        return result;
    }


}
