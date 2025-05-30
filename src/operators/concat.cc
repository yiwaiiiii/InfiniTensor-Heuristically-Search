#include "operators/concat.h"
#include "utils/operator_utils.h"

namespace infini {
ConcatObj::ConcatObj(GraphObj *graph, TensorVec inputs, Tensor output, int _dim)
    : OperatorObj(OpType::Concat, inputs, {output}) {
    int rank = inputs[0]->getRank();
    dim = get_real_axis(_dim, rank);
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> ConcatObj::inferShape(const TensorVec &inputs) {
    Shape dims = inputs[0]->getDims();
    auto rank = inputs[0]->getRank();
    if (inputs.size() == 2) {
        for (size_t i = 0; i < inputs.size(); ++i) {
            if (inputs[i]->size() == 0) {
                return {{inputs[1 - i]->getDims()}};
            }
        }
    }
    ShapeElem n = dims.at(dim);
    for (auto itr = inputs.begin() + 1; itr != inputs.end(); ++itr) {
        auto input = *itr;
        auto iDims = input->getDims();
        IT_ASSERT(rank == input->getRank());
        for (auto i = 0; i < (int)rank; i++) {
            if (i == dim) {
                n += iDims.at(i);
                continue;
            }
            IT_ASSERT(iDims.at(i) == dims.at(i));
        }
    }
    dims[dim] = n;
    return {{dims}};
}

std::string ConcatObj::toString() const {
    std::ostringstream os;
    os << "Concat[" << getGuid() << "]";
    os << "(";
    for (auto input : inputs)
        os << vecToString(input->getDims()) << ",";
    os << "dim=" << dim << ",";
    os << "input=";
    for (auto input : inputs)
        os << input->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> ConcatObj::getWorkloadVector() const {
    vector<int> ret = getOutput()->getDims();
    ret.emplace(ret.begin(), (int)inputs.size());
    ret.emplace(ret.begin(), dim);
    ret.emplace(ret.begin(), type.underlying());
    return ret;
}

vector<int> ConcatObj::getOpAttrVector() const {
    return {type.underlying(), dim};
}

double ConcatObj::getComputeTime() const {
    double totalElements = 0;
    for (const auto &input : inputs) {
        totalElements += input->size();
    }
    
    return totalElements / 1e9;
}

double ConcatObj::getMemoryCost() const {
    double inputCost = 0;
    for (const auto &input : inputs) {
        inputCost += input->size();
    }
    
    double outputCost = outputs[0]->size();
    
    return inputCost + outputCost;
}

double ConcatObj::getParallelism() const {
    const auto &outDims = outputs[0]->getDims();
    
    int64_t parallelWorkload = 1;
    for (size_t i = 0; i < outDims.size(); ++i) {
        if (i != (size_t)dim) {
            parallelWorkload *= outDims[i];
        }
    }
    
    const double MAX_PARALLEL_UNITS = 1024.0;
    return std::min(static_cast<double>(parallelWorkload), MAX_PARALLEL_UNITS);
}

} // namespace infini