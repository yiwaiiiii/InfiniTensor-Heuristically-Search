#include "operators/dropout.h"

namespace infini {

DropoutObj::DropoutObj(GraphObj *graph, Tensor data, Tensor output, Tensor mask,
                       float ratio, bool training_mode)
    : OperatorObj(OpType::Dropout, {data}, {output, mask}), ratio(ratio) {
    IT_ASSERT(0 <= ratio && ratio < 1);
    IT_ASSERT(!training_mode);
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> DropoutObj::inferShape(const TensorVec &inputs) {
    auto shape = inputs[0]->getDims();
    return {{shape, shape}};
}

std::string DropoutObj::toString() const {
    std::ostringstream os;
    os << "Dropout[" << getGuid() << "](" << vecToString(inputs[0]->getDims())
       << ", "
       << "ratio=" << ratio << ", "
       << "training_mode=false, "
       << "input=" << inputs[0]->getGuid() << ", "
       << "outputs=" << outputs[0]->getGuid() << ", " << outputs[1]->getGuid()
       << ")";
    return os.str();
}

vector<int> DropoutObj::getWorkloadVector() const {
    vector<int> ret = inputs[0]->getDims();
    ret.emplace_back(static_cast<int>(ratio));
    ret.emplace(ret.begin(), type.underlying());
    return ret;
}

vector<int> DropoutObj::getOpAttrVector() const {
    return {type.underlying(), static_cast<int>(ratio), false};
}

double DropoutObj::getComputeTime() const {
    int64_t size = inputs[0]->size();
    double opsPerElement = 3.0;
    double totalOps = size * opsPerElement;
    double randomGenFactor = 1.2;
    return totalOps * randomGenFactor / 1e9;
}

double DropoutObj::getMemoryCost() const {
    double inputCost = inputs[0]->size();
    double outputCost = outputs[0]->size();
    double maskCost = outputs[1]->size();
    return inputCost + outputCost + maskCost;
}

double DropoutObj::getParallelism() const {
    int64_t size = inputs[0]->size();
    const double MAX_PARALLEL_UNITS = 1024.0;
    double utilizationFactor = 0.95;
    return std::min(static_cast<double>(size) * utilizationFactor, MAX_PARALLEL_UNITS);
}

} // namespace infini