#include "operators/broadcast.h"

namespace infini {
BroadcastObj::BroadcastObj(GraphObj *graph, Tensor input, Tensor output,
                           int root)
    : OperatorObj(OpType::Broadcast, {input}, {output}), root(root) {
    IT_ASSERT(checkValid(graph));
}

vector<int> BroadcastObj::getWorkloadVector() const {
    vector<int> ret{type.underlying()};
    const Shape shape = inputs[0]->getDims();
    ret.insert(ret.end(), shape.begin(), shape.end());
    return ret;
}

vector<int> BroadcastObj::getOpAttrVector() const {
    return {type.underlying()};
}

std::string BroadcastObj::toString() const {
    std::ostringstream os;
    os << "Broadcast"
       << "[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ",";
    os << "root=" << root;
    os << ")";
    return os.str();
}

double BroadcastObj::getComputeTime() const {
    double dataSize = inputs[0]->size();
    
    int world_size = 8;
    
    double commCost = dataSize * std::log2(world_size) * 1e-7;
    
    return commCost;
}

double BroadcastObj::getMemoryCost() const {
    double inputCost = inputs[0]->size();
    double outputCost = outputs[0]->size();
    
    return inputCost + outputCost;
}

double BroadcastObj::getParallelism() const {
    int world_size = 8;
    
    return world_size - 1;
}

} // namespace infini