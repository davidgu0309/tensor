namespace tensor{

    // TO DO: scalar * tensor, maybe generic broadcasting

    // Unary operations
    template<typename T, typename U, U (*unaryOp)(T)>
    Tensor<U> applyUnaryOp(const Tensor<T>& a){
        const std::vector<T>& data_a = a.data();  
        Tensor<U> result(a.shape());
        std::vector<U>& result_data = result.data();
        for (int i=0; i<data_a.size(); i++) {
            result_data[i] = unaryOp(data_a[i]);
        }
        return result;
    }

    template <typename T>
    Tensor<T> neg(const Tensor<T>& a){
        return applyUnaryOp<T, T, scalar::neg<T>>(a);
    }

    template <typename T>
    Tensor<T> inv(const Tensor<T>& a){
        return applyUnaryOp<T, T, scalar::inv<T>>(a);
    }

    template <typename T>
    Tensor<T> relu(const Tensor<T>& a){
        return applyUnaryOp<T, T, scalar::relu<T>>(a);
    }

    template <typename T>
    Tensor<T> sigmoid(const Tensor<T>& a){
        return applyUnaryOp<T, T, scalar::sigmoid<T>>(a);
    }

    // Binary operations
    template<typename T, typename U, typename V, V (*binaryOp)(T, U)>
    Tensor<V> applyBinaryOp(const Tensor<T>& a, const Tensor<U>& b){
        if(!a.shapeEqual(b)){
            std::cout << a.shape() << std::endl;
            std::cout << b.shape() << std::endl;
            assert(0);
        }
        const std::vector<T>& data_a = a.data();  
        const std::vector<U>& data_b = b.data();  
        Tensor<V> result(a.shape());
        std::vector<V>& result_data = result.data();
        for (int i=0; i<data_a.size(); i++) {
            result_data[i] = binaryOp(data_a[i], data_b[i]);
        }
        return result;
    }

    template <typename T>
    Tensor<T> add(const Tensor<T>& a, const Tensor<T>& b) {
        return applyBinaryOp<T, T, T, scalar::sum<T>>(a, b);
    }
    

    template <typename T>
    Tensor<T> mul(const Tensor<T>& a, const Tensor<T>& b) {
        return applyBinaryOp<T, T, T, scalar::product<T>>(a, b);
    }

    template <typename T>
    T dot(const Tensor<T>& a, const Tensor<T>& b) {
        if(a.shape() != b.shape()){
            std::cout << a.shape() << std::endl;
            std::cout << b.shape() << std::endl;
            assert(0);
        }
        T sum = 0;
        std::vector<MultiIndex> indexes = indexesRowMajor(a.shape());
        for(const MultiIndex& i : indexes){
            sum += a.get(i) * b.get(i);
        }
        return sum;
    }

    // TODO: implement slicing and rewrite with dot
    template <typename T>
    Tensor<T> evaluateDifferential(const Tensor<T>& x, const Tensor<T>& D, size_t gradient_dim){
        Shape input_shape = x.shape();
        Shape D_shape = D.shape();
        // TODO: check that dimensions that are dot-producted match
        Shape output_shape(D_shape.begin() + gradient_dim, D_shape.end());
        Shape result_shape(input_shape.begin(), input_shape.end() - gradient_dim);
        result_shape.insert(result_shape.end(), output_shape.begin(), output_shape.end());
        Tensor<T> diff = zeros<T>(result_shape);
        std::vector<MultiIndex> input_indexes = indexesRowMajor(input_shape), output_indexes = indexesRowMajor(output_shape);
        for(const MultiIndex& i : input_indexes){
            const T& x_i = x.getEntrySafe(i);
            MultiIndex prefix(i.begin(), i.end() - gradient_dim);
            MultiIndex suffix(i.end() - gradient_dim, i.end());
            for(const MultiIndex& j : output_indexes){
                MultiIndex pj = concatIndexes(prefix, j);
                MultiIndex sj = concatIndexes(suffix, j);
                diff.getEntrySafe(pj) += x_i * D.getEntrySafe(sj);
            }
        }
        return diff;
    }

    // a_shape = {a_1, ..., a_n}, b_shape = {b_1, ..., b_m}, ab_shape = {a_1, ..., a_(n - 1), b_2, ..., b_m}
    Shape matmulShape(const Shape a_shape, const Shape b_shape){
        
        size_t dim_a = a_shape.size(), dim_b = b_shape.size();
        
        // Compatibility test
        // assert(dim_a && dim_b && a_shape.back() == b_shape.front());    // TO DO: return {} if one of the two is {}
        if(!(dim_a && dim_b && a_shape.back() == b_shape.front())){
            std::cout << a_shape << std::endl;
            std::cout << b_shape << std::endl;
            assert(0);
        }

        // Compute result_shape
        Shape result_shape = a_shape;
        result_shape.pop_back();
        for(size_t d = 1; d < dim_b; ++d) result_shape.push_back(b_shape[d]);

        return result_shape;
    }

    template <typename T>
    Tensor<T> matmul(const Tensor<T>& a, const Tensor<T>& b){

        Shape a_shape = a.shape(), b_shape = b.shape();

        // Compute result shape
        Shape result_shape = matmulShape(a_shape, b_shape);
        // Initialize result with 0s
        Tensor<T> result = zeros<T>(result_shape);

        a_shape.pop_back();
        // TO DO (priority: low for now): think about performance in cache
        std::vector<MultiIndex> a_indexes = indexesRowMajor(a_shape), b_indexes = indexesRowMajor(b_shape);

        /*
        a:
        shape = {3, 2} : {0} {1} {2}
        1 2 
        3 4 
        5 6

        b:
        shape = {2, 2} : {0, 0} {0, 1} {1, 0} {1, 1}
        1 2
        3 4

        result:
        0 0
        0 0
        0 0
        */

        // Asymptotically this is optimal O(result.size() * a.shape.back())
        for(const MultiIndex& i : a_indexes){
            for(const MultiIndex& j : b_indexes){
                MultiIndex ii(i); ii.push_back(j.front());
                result.getEntrySafe(combineIndexes(i, j)) += a.getEntrySafe(ii) * b.getEntrySafe(j);
            }
        }

        return result;
    }

}
