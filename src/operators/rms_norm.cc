#include "operators/rms_norm.h"

namespace infini {
RMSNormObj::RMSNormObj(GraphObj *graph, Tensor input, Tensor weight,
                       Tensor output)
    : OperatorObj(OpType::RMSNorm, {input, weight}, {output}) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> RMSNormObj::inferShape(const TensorVec &inputs) {
    const auto A = inputs[0];
    auto input_dim = A->getDims();
    auto output_dim = input_dim;
    return {{output_dim}};
}

std::string RMSNormObj::toString() const {
    std::ostringstream os;
    os << type.toString() << "[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> RMSNormObj::getWorkloadVector() const {
    vector<int> ret{type.underlying()};
    const Shape shape = outputs[0]->getDims();
    ret.insert(ret.end(), shape.begin(), shape.end());
    return ret;
}

vector<int> RMSNormObj::getOpAttrVector() const { return {type.underlying()}; }

double RMSNormObj::getComputeTime() const {
    const auto &inputDims = inputs[0]->getDims();
    int64_t inputSize = inputs[0]->size();
    int64_t lastDimSize = inputDims.back();
    int64_t batchSize = inputSize / lastDimSize;
    
    double squareOps = inputSize;
    double sumOps = batchSize * lastDimSize;
    double rmsOps = batchSize * 2;
    double normalizeOps = inputSize;
    double scaleOps = inputSize;
    
    double totalOps = squareOps + sumOps + rmsOps + normalizeOps + scaleOps;
    return totalOps / 1.5e9;
}

double RMSNormObj::getMemoryCost() const {
    double inputCost = inputs[0]->size();
    double weightCost = inputs[1]->size();
    double outputCost = outputs[0]->size();
    
    const auto &inputDims = inputs[0]->getDims();
    int64_t lastDimSize = inputDims.back();
    int64_t batchSize = inputs[0]->size() / lastDimSize;
    double tempStorageCost = inputs[0]->size() + batchSize;
    
    return inputCost + weightCost + outputCost + tempStorageCost;
}

double RMSNormObj::getParallelism() const {
    const auto &inputDims = inputs[0]->getDims();
    int64_t lastDimSize = inputDims.back();
    int64_t batchSize = inputs[0]->size() / lastDimSize;
    
    double batchParallelism = batchSize;
    double normParallelism = std::min(16.0, std::log2(lastDimSize) * 4.0);
    double totalParallelism = batchParallelism * normParallelism;
    const double MAX_PARALLEL_UNITS = 1024.0;
    
    return std::min(totalParallelism, MAX_PARALLEL_UNITS);
}

}; // namespace infini