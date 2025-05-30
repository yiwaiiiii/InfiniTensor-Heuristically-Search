#include "operators/gather.h"
#include "utils/operator_utils.h"

namespace infini {
GatherElementsObj::GatherElementsObj(GraphObj *graph, Tensor input,
                                     Tensor indices, Tensor output, int axis)
    : GatherBaseObj(OpType::GatherElements, {input, indices}, {output}, axis) {
    int rank = input->getRank();
    this->axis = get_real_axis(axis, rank);
    IT_ASSERT(checkValid(graph));
}

bool checkShape(Tensor input, Tensor indices, int axis) {
    auto inputDims = input->getDims();
    auto indicesDims = indices->getDims();
    if (input->getRank() != indices->getRank()) {
        return false;
    }
    for (int i = 0; i < static_cast<int>(input->getRank()); ++i) {
        if (i != axis && inputDims[i] != indicesDims[i]) {
            return false;
        }
    }
    return true;
}

optional<vector<Shape>> GatherElementsObj::inferShape(const TensorVec &inputs) {
    IT_ASSERT(checkShape(inputs[0], inputs[1], axis));
    auto indicesDims = inputs[1]->getDims();
    return {{indicesDims}};
}

vector<DataType>
GatherElementsObj::inferDataType(const TensorVec &inputs) const {
    IT_ASSERT(inputs.size() == 2);
    auto indexDtype = inputs[1]->getDType();
    IT_ASSERT(indexDtype == DataType::Int32 || indexDtype == DataType::Int64);
    return {inputs[0]->getDType()};
}

std::string GatherElementsObj::toString() const {
    std::ostringstream os;
    os << "GatherElements"
       << "[" << getGuid() << "]";
    os << "(";
    if (inputs.size() == 2) {
        os << vecToString(inputs[0]->getDims()) << ",";
        os << vecToString(inputs[1]->getDims()) << ",";
    }
    os << "axis=" << axis << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> GatherElementsObj::getWorkloadVector() const {
    vector<int> ret = inputs[0]->getDims();
    ret.emplace(ret.begin(), type.underlying());
    for (auto it : inputs[1]->getDims())
        ret.emplace_back(it);
    ret.emplace_back(axis);
    return ret;
}

vector<int> GatherElementsObj::getOpAttrVector() const {
    return {type.underlying(), axis};
}

double GatherElementsObj::getComputeTime() const {
    double indicesSize = inputs[1]->size();
    const auto &inputDims = inputs[0]->getDims();
    const auto &indicesDims = inputs[1]->getDims();
    int rank = inputDims.size();
    
    double memoryDiscontinuityFactor = 1.0;
    if (axis < rank - 1) {
        int innerDimProduct = 1;
        for (int i = axis + 1; i < rank; ++i) {
            innerDimProduct *= inputDims[i];
        }
        memoryDiscontinuityFactor = 1.0 + std::log2(innerDimProduct) * 0.2;
    }
    
    double indexingOps = indicesSize * 3.0;
    return indexingOps * memoryDiscontinuityFactor / 5e8;
}

double GatherElementsObj::getMemoryCost() const {
    double indicesCost = inputs[1]->size();
    double inputCost = inputs[0]->size() * 0.1;
    double outputCost = outputs[0]->size();
    double randomAccessFactor = 2.0;
    return indicesCost + inputCost * randomAccessFactor + outputCost;
}

double GatherElementsObj::getParallelism() const {
    double outputSize = outputs[0]->size();
    double parallelEfficiency;
    const auto &inputDims = inputs[0]->getDims();
    int rank = inputDims.size();
    
    if (axis == rank - 1) {
        parallelEfficiency = 0.9;
    } else if (axis == 0) {
        parallelEfficiency = 0.6;
    } else {
        parallelEfficiency = 0.75;
    }
    
    const double MAX_PARALLEL_UNITS = 1024.0;
    return std::min(outputSize * parallelEfficiency, MAX_PARALLEL_UNITS);
}

} // namespace infini