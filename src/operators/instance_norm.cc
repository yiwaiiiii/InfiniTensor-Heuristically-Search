#include "operators/instance_norm.h"
#include "utils/operator_utils.h"

namespace infini {
InstanceNormObj::InstanceNormObj(GraphObj *graph, Tensor input, Tensor output,
                                 Tensor scale, Tensor bias, float eps)
    : OperatorObj(OpType::InstanceNormalization, TensorVec{input, scale, bias},
                  {output}),
      eps(eps) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> InstanceNormObj::inferShape(const TensorVec &inputs) {
    return {{inputs[0]->getDims()}};
}

vector<DataType> InstanceNormObj::inferDataType(const TensorVec &inputs) const {
    return {inputs[0]->getDType()};
}

std::string InstanceNormObj::toString() const {
    std::ostringstream os;
    os << "InstanceNormalization[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "eps=" << eps << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "scale=" << inputs[1]->getGuid() << ",";
    os << "bias=" << inputs[2]->getGuid() << ",";
    os << "output=";
    for (auto output : outputs)
        os << output->getGuid() << ",";
    return os.str();
}

vector<int> InstanceNormObj::getWorkloadVector() const {
    vector<int> ret = inputs[0]->getDims();
    ret.emplace(ret.begin(), type.underlying());
    return ret;
}

vector<int> InstanceNormObj::getOpAttrVector() const {
    return {type.underlying()};
}

double InstanceNormObj::getComputeTime() const {
    const auto &inputDims = inputs[0]->getDims();
    int64_t batchSize = inputDims[0];
    int64_t channels = inputDims[1];
    int64_t featuresPerInstance = 1;
    for (size_t i = 2; i < inputDims.size(); ++i) {
        featuresPerInstance *= inputDims[i];
    }
    
    double opsForMean = batchSize * channels * featuresPerInstance + batchSize * channels;
    double opsForVariance = 3 * batchSize * channels * featuresPerInstance + batchSize * channels;
    double opsForNormalize = 3 * batchSize * channels * featuresPerInstance;
    double opsForScaleShift = 2 * batchSize * channels * featuresPerInstance;
    
    double totalOps = opsForMean + opsForVariance + opsForNormalize + opsForScaleShift;
    return totalOps / 1e9;
}

double InstanceNormObj::getMemoryCost() const {
    double inputCost = inputs[0]->size();
    double scaleCost = inputs[1]->size();
    double biasCost = inputs[2]->size();
    double outputCost = outputs[0]->size();
    const auto &inputDims = inputs[0]->getDims();
    int64_t batchSize = inputDims[0];
    int64_t channels = inputDims[1];
    double tempStorageCost = 2 * batchSize * channels;
    return inputCost + scaleCost + biasCost + outputCost + tempStorageCost;
}

double InstanceNormObj::getParallelism() const {
    const auto &inputDims = inputs[0]->getDims();
    int64_t batchSize = inputDims[0];
    int64_t channels = inputDims[1];
    double parallelism = batchSize * channels;
    const double MAX_PARALLEL_UNITS = 1024.0;
    return std::min(parallelism, MAX_PARALLEL_UNITS);
}

} // namespace infini