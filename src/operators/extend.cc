#include "operators/extend.h"
#include "utils/operator_utils.h"

namespace infini {

ExtendObj::ExtendObj(GraphObj *graph, Tensor input, Tensor output, int dim,
                     int num)
    : OperatorObj(OpType::Extend, {input}, {output}), dim(dim), num(num) {
    int rank = input->getRank();
    dim = get_real_axis(dim, rank);
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> ExtendObj::inferShape(const TensorVec &inputs) {
    auto ret = inputs[0]->getDims();
    ret[dim] = ret[dim] * (num + 1);
    return {{ret}};
}

std::string ExtendObj::toString() const {
    std::ostringstream os;
    os << "Extend[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "dim=" << dim << ",";
    os << "num=" << num << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> ExtendObj::getWorkloadVector() const {
    vector<int> ret = inputs[0]->getDims();
    ret.emplace_back(dim);
    ret.emplace_back(num);
    ret.emplace(ret.begin(), type.underlying());
    return ret;
}

vector<int> ExtendObj::getOpAttrVector() const {
    return {type.underlying(), dim, num};
}

double ExtendObj::getComputeTime() const {
    double inputSize = inputs[0]->size();
    double outputSize = outputs[0]->size();
    double extendRatio = num + 1;
    double indexMappingCost = std::log2(extendRatio) * 0.1;
    return outputSize * (1.0 + indexMappingCost) / 1e9;
}

double ExtendObj::getMemoryCost() const {
    double inputSize = inputs[0]->size();
    double outputSize = outputs[0]->size();
    return inputSize + outputSize;
}

double ExtendObj::getParallelism() const {
    const auto &inputDims = inputs[0]->getDims();
    int64_t nonExtendSize = 1;
    for (size_t i = 0; i < inputDims.size(); ++i) {
        if (i != (size_t)dim) {
            nonExtendSize *= inputDims[i];
        }
    }
    int64_t extendDimSize = inputDims[dim] * (num + 1);
    double parallelism = nonExtendSize * std::min(extendDimSize, (int64_t)16);
    const double MAX_PARALLEL_UNITS = 1024.0;
    return std::min(parallelism, MAX_PARALLEL_UNITS);
}

} // namespace infini