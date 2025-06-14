#include "operators/reshape.h"
#include "utils/operator_utils.h"
#include <numeric>

namespace infini {
ReshapeObj::ReshapeObj(GraphObj *graph, Tensor input, Tensor output, Shape dims)
    : OperatorObj(OpType::Reshape, {input}, {output}), dims(std::move(dims)) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> ReshapeObj::inferShape(const TensorVec &inputs) {
    int count = 0;
    for (auto x : dims) {
        if (x == -1) {
            count++;
        }
        IT_ASSERT(x == -1 || x >= 0);
    }
    IT_ASSERT(count == 0 || count == 1);
    auto inputShape = inputs[0]->getDims();
    int size = inputs[0]->size();
    int index = -1;
    outputShape = dims;
    for (int i = 0; i < (int)dims.size(); ++i) {
        if (dims[i] == 0) {
            outputShape[i] = inputShape[i];
        }
        if (dims[i] == -1) {
            index = i;
        }
    }
    if (index != -1) {
        outputShape[index] =
            size / (-std::accumulate(outputShape.begin(), outputShape.end(), 1,
                                     [](auto acc, auto x) { return acc * x; }));
    }
    int outputSize = std::accumulate(outputShape.begin(), outputShape.end(), 1,
                                     [](auto acc, auto x) { return acc * x; });
    IT_ASSERT(outputSize == size);

    return {{outputShape}};
}

std::string ReshapeObj::toString() const {
    std::ostringstream os;
    os << "Reshape[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "outputShape=" << vecToString(outputShape) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> ReshapeObj::getWorkloadVector() const {
    vector<int> ret = inputs[0]->getDims();
    ret.insert(ret.end(), outputShape.begin(), outputShape.end());
    ret.emplace(ret.begin(), type.underlying());
    return ret;
}
vector<int> ReshapeObj::getOpAttrVector() const {
    vector<int> ret = outputShape;
    ret.emplace(ret.begin(), type.underlying());
    return ret;
}

double ReshapeObj::getComputeTime() const {
    bool needsRearrangement = false;
    const auto &inputDims = inputs[0]->getDims();
    int inputRank = inputDims.size();
    int outputRank = outputShape.size();
    if (inputRank != outputRank) {
        needsRearrangement = true;
    } else {
        for (int i = 0; i < inputRank - 1; ++i) {
            if (inputDims[i] != outputShape[i]) {
                needsRearrangement = true;
                break;
            }
        }
    }
    if (needsRearrangement) {
        double dataSize = inputs[0]->size();
        return dataSize / 8e9;
    } else {
        return 1e-6;
    }
}

double ReshapeObj::getMemoryCost() const {
    bool needsRearrangement = false;
    const auto &inputDims = inputs[0]->getDims();
    int inputRank = inputDims.size();
    int outputRank = outputShape.size();
    if (inputRank != outputRank) {
        needsRearrangement = true;
    } else {
        for (int i = 0; i < inputRank - 1; ++i) {
            if (inputDims[i] != outputShape[i]) {
                needsRearrangement = true;
                break;
            }
        }
    }
    if (needsRearrangement) {
        double inputSize = inputs[0]->size();
        double outputSize = outputs[0]->size();
        return inputSize + outputSize;
    } else {
        return 0.0;
    }
}

double ReshapeObj::getParallelism() const {
    bool needsRearrangement = false;
    const auto &inputDims = inputs[0]->getDims();
    int inputRank = inputDims.size();
    int outputRank = outputShape.size();
    if (inputRank != outputRank) {
        needsRearrangement = true;
    } else {
        for (int i = 0; i < inputRank - 1; ++i) {
            if (inputDims[i] != outputShape[i]) {
                needsRearrangement = true;
                break;
            }
        }
    }
    if (needsRearrangement) {
        double dataSize = inputs[0]->size();
        const double MAX_PARALLEL_UNITS = 512.0;
        return std::min(dataSize / 128, MAX_PARALLEL_UNITS);
    } else {
        return 1.0;
    }
}

FlattenObj::FlattenObj(GraphObj *graph, Tensor input, Tensor output, int _axis)
    : OperatorObj(OpType::Flatten, {input}, {output}) {
    int rank = input->getRank();
    axis = get_real_axis(_axis, rank);
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> FlattenObj::inferShape(const TensorVec &inputs) {
    int sizeB = 1, sizeE = 1;
    auto dims = getInputs(0)->getDims();
    int rank = getInputs(0)->getRank();
    for (int i = 0; i < rank; ++i) {
        ((i < axis) ? sizeB : sizeE) *= dims.at(i);
    }
    return {{{sizeB, sizeE}}};
}

std::string FlattenObj::toString() const {
    std::ostringstream os;
    os << "Flatten[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ",";
    os << "axis=" << axis << ")";
    return os.str();
}

vector<int> FlattenObj::getWorkloadVector() const {
    vector<int> ret = inputs[0]->getDims();
    ret.emplace(ret.begin(), axis);
    ret.emplace(ret.begin(), type.underlying());
    return ret;
}

vector<int> FlattenObj::getOpAttrVector() const {
    return {type.underlying(), axis};
}

double FlattenObj::getComputeTime() const {
    bool needsRearrangement = false;
    if (axis > 1) {
        needsRearrangement = true;
    }
    if (needsRearrangement) {
        double dataSize = inputs[0]->size();
        return dataSize / 8e9;
    } else {
        return 1e-6;
    }
}

double FlattenObj::getMemoryCost() const {
    bool needsRearrangement = false;
    if (axis > 1) {
        needsRearrangement = true;
    }
    if (needsRearrangement) {
        double inputSize = inputs[0]->size();
        double outputSize = outputs[0]->size();
        return inputSize + outputSize;
    } else {
        return 0.0;
    }
}

double FlattenObj::getParallelism() const {
    bool needsRearrangement = false;
    if (axis > 1) {
        needsRearrangement = true;
    }
    if (needsRearrangement) {
        double dataSize = inputs[0]->size();
        const double MAX_PARALLEL_UNITS = 512.0;
        return std::min(dataSize / 128, MAX_PARALLEL_UNITS);
    } else {
        return 1.0;
    }
}

IdentityObj::IdentityObj(GraphObj *graph, Tensor input, Tensor output)
    : OperatorObj(OpType::Identity, {input}, {output}) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> IdentityObj::inferShape(const TensorVec &inputs) {
    return {{getInputs(0)->getDims()}};
}

std::string IdentityObj::toString() const {
    std::ostringstream os;
    os << "Identity[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> IdentityObj::getWorkloadVector() const {
    vector<int> ret = inputs[0]->getDims();
    ret.emplace(ret.begin(), type.underlying());
    return ret;
}
vector<int> IdentityObj::getOpAttrVector() const { return {type.underlying()}; }

double IdentityObj::getComputeTime() const {
    double dataSize = inputs[0]->size();
    return dataSize / 10e9;
}

double IdentityObj::getMemoryCost() const {
    double inputSize = inputs[0]->size();
    double outputSize = outputs[0]->size();
    return inputSize + outputSize;
}

double IdentityObj::getParallelism() const {
    double dataSize = inputs[0]->size();
    const double MAX_PARALLEL_UNITS = 1024.0;
    return std::min(dataSize / 64, MAX_PARALLEL_UNITS);
}
} // namespace infini