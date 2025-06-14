#include "operators/unsqueeze.h"
#include "utils/operator_utils.h"

namespace infini {
UnsqueezeObj::UnsqueezeObj(GraphObj *graph, Tensor input, Tensor output,
                           Shape axes)
    : OperatorObj(OpType::Unsqueeze, {input}, {output}), axes(std::move(axes)) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> UnsqueezeObj::inferShape(const TensorVec &inputs) {
    Shape inputDim = inputs[0]->getDims();
    auto rank = inputs[0]->getRank() + axes.size();
    Shape outputShape(rank, -1);
    for (size_t i = 0; i < axes.size(); ++i) {
        axes[i] = get_real_axis(axes[i], rank);
        IT_ASSERT(outputShape[axes[i]] == -1, "Axes have duplicate");
        outputShape[axes[i]] = 1;
    }
    auto it = inputDim.begin();
    for (size_t i = 0; i < outputShape.size(); ++i) {
        if (outputShape[i] == -1) {
            outputShape[i] = *it++;
        }
    }
    return {{outputShape}};
}

std::string UnsqueezeObj::toString() const {
    std::ostringstream os;
    os << "Unsqueeze[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "axes=" << vecToString(axes) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> UnsqueezeObj::getWorkloadVector() const {
    vector<int> ret = inputs[0]->getDims();
    ret.insert(ret.end(), axes.begin(), axes.end());
    ret.emplace(ret.begin(), type.underlying());
    return ret;
}

vector<int> UnsqueezeObj::getOpAttrVector() const {
    vector<int> ret = axes;
    ret.emplace(ret.begin(), type.underlying());
    return ret;
}

double UnsqueezeObj::getComputeTime() const {
    double inputSize = inputs[0]->size();
    bool needsRearrangement = false;
    int oldRank = inputs[0]->getRank();
    int newRank = outputs[0]->getRank();
    
    for (int axis : axes) {
        int realAxis = axis < 0 ? axis + newRank : axis;
        if (realAxis > 0 && realAxis < newRank - 1) {
            needsRearrangement = true;
            break;
        }
    }
    
    if (needsRearrangement) {
        return inputSize / 10e9;
    } else {
        return 1e-6;
    }
}

double UnsqueezeObj::getMemoryCost() const {
    double inputSize = inputs[0]->size();
    double outputSize = outputs[0]->size();
    bool needsRearrangement = false;
    int oldRank = inputs[0]->getRank();
    int newRank = outputs[0]->getRank();
    
    for (int axis : axes) {
        int realAxis = axis < 0 ? axis + newRank : axis;
        if (realAxis > 0 && realAxis < newRank - 1) {
            needsRearrangement = true;
            break;
        }
    }
    
    if (needsRearrangement) {
        return inputSize + outputSize;
    } else {
        return 0.0;
    }
}

double UnsqueezeObj::getParallelism() const {
    bool needsRearrangement = false;
    int oldRank = inputs[0]->getRank();
    int newRank = outputs[0]->getRank();
    
    for (int axis : axes) {
        int realAxis = axis < 0 ? axis + newRank : axis;
        if (realAxis > 0 && realAxis < newRank - 1) {
            needsRearrangement = true;
            break;
        }
    }
    
    if (needsRearrangement) {
        double dataSize = inputs[0]->size();
        const double MAX_PARALLEL_UNITS = 512.0;
        return std::min(dataSize / 128.0, MAX_PARALLEL_UNITS);
    } else {
        return 1.0;
    }
}

} // namespace infini