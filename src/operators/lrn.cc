#include "operators/lrn.h"
#include "utils/operator_utils.h"

namespace infini {

LRNObj::LRNObj(GraphObj *graph, Tensor input, Tensor output, float alpha,
               float beta, float bias, int size)
    : OperatorObj(OpType::LRN, TensorVec{input}, {output}), alpha_value(alpha),
      beta_value(beta), bias_value(bias), size_value(size) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> LRNObj::inferShape(const TensorVec &inputs) {
    const auto A = inputs[0];
    return {{A->getDims()}};
}

std::string LRNObj::toString() const {
    std::ostringstream os;
    os << "LRN[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> LRNObj::getWorkloadVector() const {
    vector<int> ret = getOutput()->getDims();
    ret.emplace(ret.begin(), type.underlying());
    return ret;
}

vector<int> LRNObj::getOpAttrVector() const { return {type.underlying()}; }

double LRNObj::getComputeTime() const {
    const auto &inputDims = inputs[0]->getDims();
    int64_t batchSize = inputDims[0];
    int64_t channels = inputDims[1];
    int64_t spatialSize = 1;
    for (size_t i = 2; i < inputDims.size(); ++i) {
        spatialSize *= inputDims[i];
    }
    double windowSize = std::min(static_cast<double>(size_value), static_cast<double>(channels));
    double sumSquareOps = batchSize * channels * spatialSize * windowSize;
    double normalizeOps = batchSize * channels * spatialSize * 5.0;
    double totalOps = sumSquareOps + normalizeOps;
    return totalOps / 1e9;
}

double LRNObj::getMemoryCost() const {
    double inputCost = inputs[0]->size();
    double outputCost = outputs[0]->size();
    const auto &inputDims = inputs[0]->getDims();
    int64_t batchSize = inputDims[0];
    int64_t channels = inputDims[1];
    int64_t spatialSize = 1;
    for (size_t i = 2; i < inputDims.size(); ++i) {
        spatialSize *= inputDims[i];
    }
    double tempStorageCost = batchSize * channels * spatialSize;
    double windowEfficiencyFactor = 1.0 + std::log2(size_value) * 0.1;
    return (inputCost * windowEfficiencyFactor) + outputCost + tempStorageCost;
}

double LRNObj::getParallelism() const {
    const auto &inputDims = inputs[0]->getDims();
    int64_t batchSize = inputDims[0];
    int64_t spatialSize = 1;
    for (size_t i = 2; i < inputDims.size(); ++i) {
        spatialSize *= inputDims[i];
    }
    double parallelism = batchSize * spatialSize;
    int64_t channels = inputDims[1];
    if (size_value < channels) {
        double channelParallelFactor = std::ceil(static_cast<double>(channels) / size_value);
        parallelism *= std::min(channelParallelFactor, 4.0);
    }
    const double MAX_PARALLEL_UNITS = 1024.0;
    return std::min(parallelism, MAX_PARALLEL_UNITS);
}

} // namespace infini