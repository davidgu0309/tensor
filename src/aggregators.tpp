namespace aggregator {
    template <typename T>
    T sum(const std::vector<T>& operands) {
        T accul = 0;
        for (T operand : operands) {
            accul += operand;
        }
        return accul;
    } 

    template <typename T>
    T mean(const std::vector<T>& operands) {
        return operands.size() == 0 ? 0 : sum(operands)/operands.size();
    } 
}