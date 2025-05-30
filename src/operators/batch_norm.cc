#include "operators/batch_norm.h"

namespace infini {
BatchNormObj::BatchNormObj(GraphObj *graph, Tensor input, Tensor output,
                           Tensor mean, Tensor var, Tensor scale, Tensor bias,
                           float momentum, float eps, bool trainingMode)
    : OperatorObj(OpType::BatchNormalization, {input, mean, var, scale, bias},
                  {output}),
      momentum(momentum), eps(eps), trainingMode(trainingMode) {
    if (trainingMode)
        IT_TODO_HALT();

    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> BatchNormObj::inferShape(const TensorVec &inputs) {
    auto input = inputs[0];
    auto mean = inputs[1];
    auto var = inputs[2];
    auto scale = inputs[3];
    auto bias = inputs[4];
    auto c = std::vector<int>{input->getDims()[1]};
    IT_ASSERT(mean->getRank() == 1 && mean->getDims() == c);
    IT_ASSERT(var->getRank() == 1 && var->getDims() == c);
    IT_ASSERT(scale->getRank() == 1 && scale->getDims() == c);
    IT_ASSERT(bias->getRank() == 1 && bias->getDims() == c);
    return {{input->getDims()}};
}

vector<DataType> BatchNormObj::inferDataType(const TensorVec &inputs) const {
    IT_ASSERT(inputs.size() == 5);
    auto index = inputs[1];
    IT_ASSERT(inputs[1]->getDType() == DataType::Float32);
    IT_ASSERT(inputs[2]->getDType() == DataType::Float32);
    IT_ASSERT(inputs[3]->getDType() == DataType::Float32);
    IT_ASSERT(inputs[4]->getDType() == DataType::Float32);
    return {inputs[0]->getDType()};
}

std::string BatchNormObj::toString() const {
    std::ostringstream os;
    os << "batchNormalization[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "momentum=" << momentum << ",";
    os << "eps=" << eps << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "mean=" << inputs[1]->getGuid() << ",";
    os << "var=" << inputs[2]->getGuid() << ",";
    os << "scale=" << inputs[3]->getGuid() << ",";
    os << "bias=" << inputs[4]->getGuid() << ",";
    os << "output=";
    for (auto output : outputs)
        os << output->getGuid() << ",";
    return os.str();
}

vector<int> BatchNormObj::getWorkloadVector() const {
    vector<int> ret = inputs[0]->getDims();
    ret.emplace(ret.begin(), type.underlying());
    return ret;
}

vector<int> BatchNormObj::getOpAttrVector() const {
    return {type.underlying()};
}

double BatchNormObj::getComputeTime() const {
    const auto &inputDims = inputs[0]->getDims();
    
    int64_t batchSize = inputDims[0];
    int64_t channels = inputDims[1];
    
    int64_t featureSize = 1;
    for (size_t i = 2; i < inputDims.size(); ++i) {
        featureSize *= inputDims[i];
    }
    
    double opsPerElement = trainingMode ? 8.0 : 5.0;
    double totalOps = batchSize * channels * featureSize * opsPerElement;
    
    return totalOps / 1e9;
}

double BatchNormObj::getMemoryCost() const {
    double inputCost = inputs[0]->size();
    
    double paramsCost = inputs[1]->size() + inputs[2]->size() + 
                       inputs[3]->size() + inputs[4]->size();
    
    double outputCost = outputs[0]->size();
    
    return inputCost + paramsCost + outputCost;
}

double BatchNormObj::getParallelism() const {
    const auto &inputDims = inputs[0]->getDims();
    int64_t channels = inputDims[1];
    
    double parallelism = channels;
    
    if (inputDims[0] > 1) {
        parallelism *= std::min(inputDims[0], 8);
    }
    
    const double MAX_PARALLEL_UNITS = 1024.0;
    return std::min(parallelism, MAX_PARALLEL_UNITS);
}

} // namespace infini