#include "operators/element_wise.h"
#include "utils/operator_utils.h"

namespace infini {
ElementWiseObj::ElementWiseObj(OpType type, GraphObj *graph, Tensor input0,
                               Tensor input1, Tensor output)
    : OperatorObj(type, {input0, input1}, {output}) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> ElementWiseObj::inferShape(const TensorVec &inputs) {
    const auto A = inputs[0], B = inputs[1];
    auto res = infer_broadcast(A->getDims(), B->getDims());
    return {{res}};
}

std::string ElementWiseObj::toString() const {
    std::ostringstream os;
    os << type.toString() << "[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << vecToString(inputs[1]->getDims()) << ",";
    os << "input0=" << inputs[0]->getGuid() << ",";
    os << "input1=" << inputs[1]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> ElementWiseObj::getWorkloadVector() const {
    vector<int> ret = outputs[0]->getDims();
    ret.emplace(ret.begin(), type.underlying());
    return ret;
}

vector<int> ElementWiseObj::getOpAttrVector() const {
    return {type.underlying()};
}

double ElementWiseObj::getComputeTime() const {
    double outputSize = outputs[0]->size();
    double complexityFactor = 1.0;
    switch (type.underlying()) {
        case OpType::Add:
        case OpType::Sub:
            complexityFactor = 1.0;
            break;
        case OpType::Mul:
            complexityFactor = 1.1;
            break;
        case OpType::Div:
        case OpType::Pow:
            complexityFactor = 1.3;
            break;
        case OpType::Equal:
        case OpType::Greater:
        case OpType::Less:
            complexityFactor = 0.8;
            break;
        default:
            complexityFactor = 1.0;
    }
    return outputSize * complexityFactor / 1e9;
}

double ElementWiseObj::getMemoryCost() const {
    double input0Size = inputs[0]->size();
    double input1Size = inputs[1]->size();
    double outputSize = outputs[0]->size();
    // 假设大部分情况下输入和输出可以复用缓存，实际访问量略大于输出
    double memoryEfficiency = 1.1; // 可根据实际情况调整
    return outputSize * memoryEfficiency + std::min(input0Size, input1Size) * 0.1;
}


double ElementWiseObj::getParallelism() const {
    double outputSize = outputs[0]->size();
    const double MAX_PARALLEL_UNITS = 1024.0;
    double utilizationFactor = 0.95;
    return std::min(outputSize * utilizationFactor, MAX_PARALLEL_UNITS);
}

MSELossObj::MSELossObj(GraphObj *graph, Tensor input0, Tensor input1,
                       Reduction reduction, Tensor output)
    : OperatorObj(OpType::MSELoss, {input0, input1}, {output}),
      reductionMode(reduction) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> MSELossObj::inferShape(const TensorVec &inputs) {
    const auto A = inputs[0], B = inputs[1];
    IT_ASSERT(A->getRank() == B->getRank());
    IT_ASSERT(A->getDims() == B->getDims());

    if (reductionMode == None) {
        return {{A->getDims()}};
    } else {
        Shape temp = {1};
        return {{temp}};
    }
}

std::string MSELossObj::toString() const {
    std::ostringstream os;
    os << type.toString() << "[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << vecToString(inputs[1]->getDims()) << ",";
    os << "input0=" << inputs[0]->getGuid() << ",";
    os << "input1=" << inputs[1]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> MSELossObj::getWorkloadVector() const {
    vector<int> ret = outputs[0]->getDims();
    ret.emplace(ret.begin(), type.underlying());
    return ret;
}

vector<int> MSELossObj::getOpAttrVector() const { return {type.underlying()}; }

double MSELossObj::getComputeTime() const {
    double inputSize = inputs[0]->size();
    double opsPerElement = 2.0;
    if (reductionMode != None) {
        opsPerElement += std::log2(inputSize) / inputSize;
    }
    return inputSize * opsPerElement / 1e9;
}

double MSELossObj::getMemoryCost() const {
    double inputsSize = inputs[0]->size() + inputs[1]->size();
    double outputSize = outputs[0]->size();
    double intermediateSize = inputs[0]->size();
    return inputsSize + outputSize + intermediateSize;
}

double MSELossObj::getParallelism() const {
    if (reductionMode == None) {
        return std::min(static_cast<double>(inputs[0]->size()), 1024.0);
    } else {
        double logParallelism = std::log2(inputs[0]->size());
        return std::min(logParallelism * 32.0, 512.0);
    }
}

} // namespace infini