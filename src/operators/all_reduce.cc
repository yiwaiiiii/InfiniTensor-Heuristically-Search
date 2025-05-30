#include "operators/all_reduce.h"

namespace infini {
AllReduceBaseObj::AllReduceBaseObj(GraphObj *graph, OpType opType, Tensor input,
                                   Tensor output)
    : OperatorObj(opType, {input}, {output}) {
    IT_ASSERT(checkValid(graph));
}

std::string AllReduceBaseObj::toString() const {
    std::ostringstream os;
    os << type.toString() << "[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ",";
    return os.str();
}

vector<int> AllReduceBaseObj::getWorkloadVector() const {
    vector<int> ret{type.underlying()};
    const Shape shape = outputs[0]->getDims();
    ret.insert(ret.end(), shape.begin(), shape.end());
    return ret;
}

vector<int> AllReduceBaseObj::getOpAttrVector() const {
    return {type.underlying()};
}

double AllReduceBaseObj::getComputeTime() const {
    double dataSize = inputs[0]->size();
    int world_size = 8;
    double commCost = dataSize * std::log2(world_size) * 1e-7;
    double computeCost = dataSize * 1e-9; 
    return commCost + computeCost;
}

double AllReduceBaseObj::getMemoryCost() const {
    return inputs[0]->size() + outputs[0]->size();
}

double AllReduceBaseObj::getParallelism() const {
    int world_size = 8;
    double efficiency = 0.9; 
    return world_size * efficiency;
}

AllReduceSumObj::AllReduceSumObj(GraphObj *graph, Tensor input, Tensor output)
    : AllReduceBaseObj(graph, OpType::AllReduceSum, input, output) {}

double AllReduceSumObj::getComputeTime() const {
    double baseTime = AllReduceBaseObj::getComputeTime();
    return baseTime * 1.0; 
}

AllReduceProdObj::AllReduceProdObj(GraphObj *graph, Tensor input, Tensor output)
    : AllReduceBaseObj(graph, OpType::AllReduceProd, input, output) {}

double AllReduceProdObj::getComputeTime() const {
    double baseTime = AllReduceBaseObj::getComputeTime();
    return baseTime * 1.2; 
}

AllReduceMinObj::AllReduceMinObj(GraphObj *graph, Tensor input, Tensor output)
    : AllReduceBaseObj(graph, OpType::AllReduceMin, input, output) {}

double AllReduceMinObj::getComputeTime() const {
    double baseTime = AllReduceBaseObj::getComputeTime();
    return baseTime * 1.1; 
}

AllReduceMaxObj::AllReduceMaxObj(GraphObj *graph, Tensor input, Tensor output)
    : AllReduceBaseObj(graph, OpType::AllReduceMax, input, output) {}

double AllReduceMaxObj::getComputeTime() const {
    double baseTime = AllReduceBaseObj::getComputeTime();
    return baseTime * 1.1; 
}

AllReduceAvgObj::AllReduceAvgObj(GraphObj *graph, Tensor input, Tensor output)
    : AllReduceBaseObj(graph, OpType::AllReduceAvg, input, output) {}

double AllReduceAvgObj::getComputeTime() const {
    double baseTime = AllReduceBaseObj::getComputeTime();
    return baseTime * 1.3; 
}

} // namespace infini
