#include "operators/det.h"

namespace infini {
DetObj::DetObj(GraphObj *graph, Tensor input, Tensor output, Mode mode)
    : OperatorObj(OpType::Det, {input}, {output}), modeValue(mode) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> DetObj::inferShape(const TensorVec &inputs) {
    const auto A = inputs[0];
    auto input = A->getDims();
    int rank = A->getRank();
    if (rank == 2) {
        std::vector<int> output = {1};
        return {{output}};
    } else {
        std::vector<int> output(input.begin(), input.end() - 2);
        return {{output}};
    }
}

std::string DetObj::toString() const {
    std::ostringstream os;
    os << type.toString() << "[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> DetObj::getWorkloadVector() const {
    vector<int> ret{type.underlying()};
    const Shape shape = outputs[0]->getDims();
    ret.insert(ret.end(), shape.begin(), shape.end());
    return ret;
}

vector<int> DetObj::getOpAttrVector() const { return {type.underlying()}; }

double DetObj::getComputeTime() const {
    const auto &inputDims = inputs[0]->getDims();
    const int rank = inputDims.size();
    int64_t n = inputDims[rank - 1];
    IT_ASSERT(inputDims[rank - 1] == inputDims[rank - 2], "Matrix must be square");
    int64_t batchSize = 1;
    for (int i = 0; i < rank - 2; ++i) {
        batchSize *= inputDims[i];
    }
    double complexity;
    switch (modeValue) {
        case Mode::LogDet:
            complexity = 2.0;
            break;
        default:
            complexity = 1.0;
    }
    double operationsPerMatrix = std::pow(n, 3.0) * complexity;
    double totalOperations = operationsPerMatrix * batchSize;
    return totalOperations / 1e9;
}

double DetObj::getMemoryCost() const {
    double inputCost = inputs[0]->size();
    double outputCost = outputs[0]->size();
    const auto &inputDims = inputs[0]->getDims();
    const int rank = inputDims.size();
    int64_t n = inputDims[rank - 1];
    double tempStorageCost = inputs[0]->size();
    return inputCost + outputCost + tempStorageCost;
}

double DetObj::getParallelism() const {
    const auto &inputDims = inputs[0]->getDims();
    const int rank = inputDims.size();
    if (rank <= 2) {
        return 1.0;
    }
    int64_t batchSize = 1;
    for (int i = 0; i < rank - 2; ++i) {
        batchSize *= inputDims[i];
    }
    return static_cast<double>(batchSize);
}

} // namespace infini