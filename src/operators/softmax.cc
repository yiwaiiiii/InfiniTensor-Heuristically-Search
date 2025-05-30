#include "operators/softmax.h"
#include "utils/operator_utils.h"

namespace infini {

SoftmaxObj::SoftmaxObj(GraphObj *graph, Tensor input, Tensor output, int _axis)
    : OperatorObj(OpType::Softmax, {input}, {output}) {
    int rank = input->getRank();
    axis = get_real_axis(_axis, rank);
    IT_ASSERT(checkValid(graph));
}

std::string SoftmaxObj::toString() const {
    std::ostringstream os;
    os << type.toString() << "[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ",";
    os << "axis=" << axis << ")";
    return os.str();
}

vector<int> SoftmaxObj::getWorkloadVector() const {
    vector<int> ret{type.underlying(), axis};
    const Shape shape = outputs[0]->getDims();
    ret.insert(ret.end(), shape.begin(), shape.end());
    return ret;
}

vector<int> SoftmaxObj::getOpAttrVector() const {
    return {type.underlying(), axis};
}

double SoftmaxObj::getComputeTime() const {
    const auto &inputDims = inputs[0]->getDims();
    int64_t inputSize = inputs[0]->size();
    int64_t axisSize = inputDims[axis];
    int64_t batchSize = inputSize / axisSize;
    
    double findMaxOps = batchSize * axisSize;
    double expOps = inputSize * 5.0;
    double sumOps = batchSize * axisSize;
    double normalizeOps = inputSize;
    
    double totalOps = findMaxOps + expOps + sumOps + normalizeOps;
    double cacheFactor = 1.0;
    if (axis < static_cast<int>(inputDims.size()) - 1) {
        cacheFactor = 1.2 + 0.1 * (static_cast<int>(inputDims.size()) - 1 - axis);
    }
    
    return totalOps * cacheFactor / 1e9;
}

double SoftmaxObj::getMemoryCost() const {
    double inputCost = inputs[0]->size();
    double outputCost = outputs[0]->size();
    
    const auto &inputDims = inputs[0]->getDims();
    int64_t axisSize = inputDims[axis];
    int64_t batchSize = inputs[0]->size() / axisSize;
    double tempStorageCost = batchSize * 2;
    
    double memoryAccessFactor = 1.0;
    if (axis < static_cast<int>(inputDims.size()) - 1) {
        memoryAccessFactor = 1.3 + 0.1 * (static_cast<int>(inputDims.size()) - 1 - axis);
    }
    
    return (inputCost + outputCost) * memoryAccessFactor + tempStorageCost;
}

double SoftmaxObj::getParallelism() const {
    const auto &inputDims = inputs[0]->getDims();
    int64_t axisSize = inputDims[axis];
    int64_t inputSize = inputs[0]->size();
    int64_t batchSize = inputSize / axisSize;
    
    double batchParallelism = batchSize;
    double withinBatchParallelism = std::min(16.0, std::log2(axisSize) * 4.0);
    double totalParallelism = batchParallelism * withinBatchParallelism;
    const double MAX_PARALLEL_UNITS = 1024.0;
    
    return std::min(totalParallelism, MAX_PARALLEL_UNITS);
}

} // namespace infini