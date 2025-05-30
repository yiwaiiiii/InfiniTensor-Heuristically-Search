#include "operators/reduce.h"
#include "utils/operator_utils.h"

namespace infini {
ReduceBaseObj::ReduceBaseObj(GraphObj *graph, OpType opType, Tensor input,
                             Tensor output, const optional<vector<int>> &_axes,
                             bool keepDims)
    : OperatorObj(opType, {input}, {output}), keepDims(keepDims) {
    const auto size = input->getRank();
    if (_axes) {
        for (auto idx : *_axes) {
            idx = get_real_axis(idx, size);
            axes.emplace(idx);
        }
    } else
        for (size_t i = 0; i < size; ++i)
            axes.emplace(i);
    IT_ASSERT(checkValid(graph));
}

bool ReduceBaseObj::isReduced(int idx) const {
    return axes.find(idx) != axes.end();
}

optional<vector<Shape>> ReduceBaseObj::inferShape(const TensorVec &inputs) {
    auto dims = inputs[0]->getDims();
    auto rank = inputs[0]->getRank();

    if (keepDims) {
        Shape ret = dims;
        for (auto it : axes)
            ret[it] = 1;
        return {{ret}};
    } else {
        Shape ret;
        for (size_t i = 0; i < rank; ++i) {
            if (!isReduced(i))
                ret.emplace_back(dims[i]);
        }
        if (ret.empty())
            return {{{1}}};
        else
            return {{ret}};
    }
}

std::string ReduceBaseObj::toString() const {
    std::ostringstream os;
    os << type.toString() << "[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";

    std::string axisstr;
    axisstr.append("[");
    for (auto d : axes) {
        axisstr.append(std::to_string(d));
        axisstr.append(",");
    }
    if (!axes.empty())
        axisstr.pop_back();
    axisstr.append("]");
    os << "axes=" << axisstr << ",";
    os << "keepDims=" << keepDims << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> ReduceBaseObj::getWorkloadVector() const {
    vector<int> ret = inputs[0]->getDims();
    ret.emplace(ret.begin(), type.underlying());
    ret.emplace_back((int)keepDims);
    ret.insert(ret.end(), axes.begin(), axes.end());
    return ret;
}

vector<int> ReduceBaseObj::getOpAttrVector() const {
    vector<int> ret = {type.underlying(), (int)keepDims};
    ret.insert(ret.end(), axes.begin(), axes.end());
    return ret;
}

double ReduceBaseObj::getComputeTime() const {
    const auto &inputDims = inputs[0]->getDims();
    int64_t inputSize = inputs[0]->size();
    int64_t reducedSize = 1;
    int rank = inputDims.size();
    for (int i = 0; i < rank; ++i) {
        if (isReduced(i)) {
            reducedSize *= inputDims[i];
        }
    }
    int64_t outputSize = inputSize / reducedSize;
    if (outputSize == 0) outputSize = 1;
    double opsPerReduction;
    if (type == OpType::ReduceSum || type == OpType::ReduceMax || 
        type == OpType::ReduceMin) {
        opsPerReduction = 1.0;
    } else if (type == OpType::ReduceMean) {
        opsPerReduction = 1.0 + 1.0/reducedSize;
    } else if (type == OpType::ReduceProd) {
        opsPerReduction = 1.0;
    } else {
        opsPerReduction = 1.0;
    }
    double totalOps = outputSize * reducedSize * opsPerReduction;
    double axisEfficiencyFactor = 1.0;
    bool hasOuterAxisReduction = false;
    for (int i = 0; i < rank/2; ++i) {
        if (isReduced(i)) {
            hasOuterAxisReduction = true;
            break;
        }
    }
    if (hasOuterAxisReduction) {
        axisEfficiencyFactor = 1.5;
    }
    const double OPS_PER_SECOND = 1.5e9;
    return (totalOps * axisEfficiencyFactor) / OPS_PER_SECOND;
}

double ReduceBaseObj::getMemoryCost() const {
    double inputCost = inputs[0]->size();
    double outputCost = outputs[0]->size();
    double tempStorageCost = 0.0;
    double memAccessEfficiency = 1.0;
    int numReducedAxes = axes.size();
    int rank = inputs[0]->getRank();
    bool hasOuterAxisReduction = false;
    for (int i = 0; i < rank/2; ++i) {
        if (isReduced(i)) {
            hasOuterAxisReduction = true;
            break;
        }
    }
    if (hasOuterAxisReduction) {
        memAccessEfficiency = 1.0 + 0.2 * numReducedAxes;
    } else {
        memAccessEfficiency = 1.0 + 0.05 * numReducedAxes;
    }
    bool hasNonContiguousAxes = false;
    auto it = axes.begin();
    int prevAxis = *it;
    ++it;
    for (; it != axes.end(); ++it) {
        if (*it != prevAxis + 1) {
            hasNonContiguousAxes = true;
            break;
        }
        prevAxis = *it;
    }
    if (hasNonContiguousAxes) {
        tempStorageCost = inputs[0]->size() * 0.1;
    }
    return (inputCost * memAccessEfficiency) + outputCost + tempStorageCost;
}

double ReduceBaseObj::getParallelism() const {
    const auto &inputDims = inputs[0]->getDims();
    int rank = inputDims.size();
    double nonReducedSize = 1.0;
    for (int i = 0; i < rank; ++i) {
        if (!isReduced(i)) {
            nonReducedSize *= inputDims[i];
        }
    }
    double reducedSize = 1.0;
    for (int i = 0; i < rank; ++i) {
        if (isReduced(i)) {
            reducedSize *= inputDims[i];
        }
    }
    double reductionParallelism = 1.0;
    if (reducedSize > 1) {
        reductionParallelism = std::min(8.0, std::log2(reducedSize));
    }
    double totalParallelism = nonReducedSize * reductionParallelism;
    const double MAX_PARALLEL_UNITS = 1024.0;
    return std::min(totalParallelism, MAX_PARALLEL_UNITS);
}

ReduceMeanObj::ReduceMeanObj(GraphObj *graph, Tensor input, Tensor output,
                             const optional<vector<int>> &_axes, bool keepDims)
    : ReduceBaseObj(graph, OpType::ReduceMean, input, output, _axes, keepDims) {
}

ReduceSumObj::ReduceSumObj(GraphObj *graph, Tensor input, Tensor output,
                           const optional<vector<int>> &_axes, bool keepDims)
    : ReduceBaseObj(graph, OpType::ReduceSum, input, output, _axes, keepDims) {}
} // namespace infini