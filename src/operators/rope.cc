#include "operators/rope.h"

namespace infini {
RoPEObj::RoPEObj(GraphObj *graph, Tensor pos, Tensor input, Tensor output)
    : OperatorObj(OpType::RoPE, {pos, input}, {output}) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> RoPEObj::inferShape(const TensorVec &inputs) {
    const auto A = inputs[1];
    auto input_dim = A->getDims();
    auto output_dim = input_dim;
    return {{output_dim}};
}

std::string RoPEObj::toString() const {
    std::ostringstream os;
    os << type.toString() << "[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> RoPEObj::getWorkloadVector() const {
    vector<int> ret{type.underlying()};
    const Shape shape = outputs[0]->getDims();
    ret.insert(ret.end(), shape.begin(), shape.end());
    return ret;
}

vector<int> RoPEObj::getOpAttrVector() const { return {type.underlying()}; }

double RoPEObj::getComputeTime() const {
    const auto &inputDims = inputs[1]->getDims();
    const auto &posDims = inputs[0]->getDims();
    int64_t inputSize = inputs[1]->size();
    int64_t lastDimSize = inputDims.back();
    int64_t rotationPairs = lastDimSize / 2;
    
    int64_t seqLen = 1;
    if (inputDims.size() >= 2) {
        seqLen = inputDims[inputDims.size() - 2];
    }
    
    int64_t batchSize = inputSize / (seqLen * lastDimSize);
    
    double angleCost = seqLen * rotationPairs * 0.5;
    double trigCost = seqLen * rotationPairs * 2 * 5.0;
    double rotationCost = batchSize * seqLen * rotationPairs * 4;
    
    double totalOps = angleCost + trigCost + rotationCost;
    return totalOps / 2e9;
}

double RoPEObj::getMemoryCost() const {
    double posCost = inputs[0]->size();
    double inputCost = inputs[1]->size();
    double outputCost = outputs[0]->size();
    
    const auto &inputDims = inputs[1]->getDims();
    int64_t lastDimSize = inputDims.back();
    int64_t rotationPairs = lastDimSize / 2;
    
    int64_t seqLen = 1;
    if (inputDims.size() >= 2) {
        seqLen = inputDims[inputDims.size() - 2];
    }
    
    double tempStorageCost = seqLen * rotationPairs * 2;
    return posCost + inputCost + outputCost + tempStorageCost;
}

double RoPEObj::getParallelism() const {
    const auto &inputDims = inputs[1]->getDims();
    int64_t inputSize = inputs[1]->size();
    int64_t lastDimSize = inputDims.back();
    
    int64_t seqLen = 1;
    if (inputDims.size() >= 2) {
        seqLen = inputDims[inputDims.size() - 2];
    }
    
    int64_t batchSize = inputSize / (seqLen * lastDimSize);
    int64_t rotationPairs = lastDimSize / 2;
    
    double batchParallelism = batchSize;
    double seqParallelism = seqLen;
    double dimParallelism = std::min(static_cast<double>(rotationPairs), 16.0);
    
    double totalParallelism = batchParallelism * seqParallelism * dimParallelism;
    const double MAX_PARALLEL_UNITS = 2048.0;
    
    return std::min(totalParallelism, MAX_PARALLEL_UNITS);
}

}; // namespace infini