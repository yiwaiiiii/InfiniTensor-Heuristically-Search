#include "operators/layer_norm.h"
#include "utils/operator_utils.h"

namespace infini {
LayerNormObj::LayerNormObj(GraphObj *graph, Tensor input, Tensor scale,
                           Tensor output, [[maybe_unused]] Tensor bias,
                           float eps, int axis_, int stash_type)
    : OperatorObj(OpType::LayerNormalization,
                  bias ? TensorVec{input, scale, bias}
                       : TensorVec{input, scale},
                  {output}),
      eps(eps), stash_type(stash_type) {
    const auto size = input->getRank();
    axis = get_real_axis(axis_, size);
    IT_ASSERT(
        is_unidirectional_broadcasting(input->getDims(), scale->getDims()));
    if (bias) {
        IT_ASSERT(
            is_unidirectional_broadcasting(input->getDims(), bias->getDims()));
    }
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> LayerNormObj::inferShape(const TensorVec &inputs) {
    return {{inputs[0]->getDims()}};
}

vector<DataType> LayerNormObj::inferDataType(const TensorVec &inputs) const {
    IT_ASSERT(inputs.size() == 2 || inputs.size() == 3);
    return {inputs[0]->getDType()};
}

std::string LayerNormObj::toString() const {
    std::ostringstream os;
    os << "layerNormalization[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "axis=" << axis << ",";
    os << "eps=" << eps << ",";
    os << "stash_type=" << stash_type << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "scale=" << inputs[1]->getGuid() << ",";
    os << "output=";
    for (auto output : outputs)
        os << output->getGuid() << ",";
    return os.str();
}

vector<int> LayerNormObj::getWorkloadVector() const {
    vector<int> ret = inputs[0]->getDims();
    ret.emplace(ret.begin(), type.underlying());
    return ret;
}

vector<int> LayerNormObj::getOpAttrVector() const {
    return {type.underlying(), axis, stash_type};
}

double LayerNormObj::getComputeTime() const {
    const auto &inputDims = inputs[0]->getDims();
    int64_t outerSize = 1;
    int64_t normalizedSize = 1;
    
    for (int i = 0; i < axis; ++i) {
        outerSize *= inputDims[i];
    }
    for (int i = axis; i < static_cast<int>(inputDims.size()); ++i) {
        normalizedSize *= inputDims[i];
    }
    
    double opsForMean = outerSize * normalizedSize + outerSize;
    double opsForVariance = 3 * outerSize * normalizedSize + outerSize;
    double opsForNormalize = 3 * outerSize * normalizedSize;
    double hasBias = (inputs.size() > 2) ? 1.0 : 0.0;
    double opsForScaleShift = (1 + hasBias) * outerSize * normalizedSize;
    
    double totalOps = opsForMean + opsForVariance + opsForNormalize + opsForScaleShift;
    return totalOps / 1.5e9;
}

double LayerNormObj::getMemoryCost() const {
    double inputCost = inputs[0]->size();
    double scaleCost = inputs[1]->size();
    double biasCost = 0.0;
    if (inputs.size() > 2) {
        biasCost = inputs[2]->size();
    }
    double outputCost = outputs[0]->size();
    
    const auto &inputDims = inputs[0]->getDims();
    int64_t outerSize = 1;
    for (int i = 0; i < axis; ++i) {
        outerSize *= inputDims[i];
    }
    double tempStorageCost = 2 * outerSize;
    
    return inputCost + scaleCost + biasCost + outputCost + tempStorageCost;
}

double LayerNormObj::getParallelism() const {
    const auto &inputDims = inputs[0]->getDims();
    int64_t outerSize = 1;
    for (int i = 0; i < axis; ++i) {
        outerSize *= inputDims[i];
    }
    
    int64_t normalizedSize = 1;
    for (int i = axis; i < static_cast<int>(inputDims.size()); ++i) {
        normalizedSize *= inputDims[i];
    }
    
    double parallelism = outerSize * std::min(16.0, std::log2(normalizedSize) * 4.0);
    const double MAX_PARALLEL_UNITS = 1024.0;
    return std::min(parallelism, MAX_PARALLEL_UNITS);
}

} // namespace infini