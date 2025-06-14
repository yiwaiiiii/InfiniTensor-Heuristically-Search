#include "operators/activation_backward.h"

namespace infini {
ActivationBackwardObj::ActivationBackwardObj(OpType type, GraphObj *graph,
                                             Tensor y, Tensor diff_y, Tensor x,
                                             Tensor diff_x)
    : OperatorObj(type, {y, diff_y, x}, {diff_x}) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>>
ActivationBackwardObj::inferShape(const TensorVec &inputs) {
    return {{inputs[0]->getDims()}};
}

std::string ActivationBackwardObj::toString() const {
    std::ostringstream os;
    os << type.toString() << "[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> ActivationBackwardObj::getWorkloadVector() const {
    vector<int> ret{type.underlying()};
    const Shape shape = outputs[0]->getDims();
    ret.insert(ret.end(), shape.begin(), shape.end());
    return ret;
}

vector<int> ActivationBackwardObj::getOpAttrVector() const {
    return {type.underlying()};
}

double ActivationBackwardObj::getComputeTime() const {
    double complexity = 1.0;
    switch (type.underlying()) {
        case OpType::ReluBackward:
            complexity = 1.0;  
            break;
        case OpType::SigmoidBackward:
            complexity = 3.0;  
            break;
        case OpType::TanhBackward:
            complexity = 3.0;  
            break;
        default:
            complexity = 2.0;  
    }
    
    return inputs[0]->size() * complexity / 1e9;  
}

double ActivationBackwardObj::getMemoryCost() const {
    double cost = 0;
    for (const auto &input : inputs) {
        cost += input->size();  
    }
    cost += outputs[0]->size();  
    return cost;
}

double ActivationBackwardObj::getParallelism() const {
    return outputs[0]->size() / 16.0;  
}

} // namespace infini
