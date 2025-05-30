#include "operators/pad.h"

namespace infini {
PadObj::PadObj(GraphObj *graph, Tensor input, Tensor output,
               const vector<int> &_pads, const optional<vector<int>> &axes)
    : OperatorObj(OpType::Pad, {input}, {output}) {
    if (!axes)
        pads = _pads;
    else {
        auto nAxis = (*axes).size();
        IT_ASSERT(_pads.size() == nAxis * 2);
        auto nDims = input->getRank();
        pads = vector<int>(nDims * 2, 0);

        for (size_t i = 0; i < nAxis; ++i) {
            auto k = (*axes)[i];
            auto j = k < 0 ? nDims + k : k;
            pads[j] = _pads[i];
            pads[j + nDims] = _pads[i + nAxis];
        }
    }
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> PadObj::inferShape(const TensorVec &inputs) {
    auto dims = inputs[0]->getDims();
    int rank = inputs[0]->getRank();
    IT_ASSERT(rank * 2 == (int)pads.size());
    for (int i = 0; i < rank; ++i) {
        IT_ASSERT(pads[i] >= 0 && pads[i + rank] >= 0);
        dims[i] += pads[i] + pads[i + rank];
    }

    return {{dims}};
}

std::string PadObj::toString() const {
    std::ostringstream os;
    os << "Pad"
       << "[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "pads=" << vecToString(pads) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> PadObj::getWorkloadVector() const {
    vector<int> ret = inputs[0]->getDims();
    ret.insert(ret.end(), pads.begin(), pads.end());
    ret.emplace(ret.begin(), type.underlying());
    return ret;
}

vector<int> PadObj::getOpAttrVector() const {
    vector<int> ret = pads;
    ret.emplace(ret.begin(), type.underlying());
    return ret;
}

double PadObj::getComputeTime() const {
    double inputSize = inputs[0]->size();
    double outputSize = outputs[0]->size();
    double paddedElements = outputSize - inputSize;
    double copyOps = inputSize * 2.0;
    double padOps = paddedElements;
    double indexingOps = outputSize * 0.1;
    double totalOps = copyOps + padOps + indexingOps;
    return totalOps / 1e9;
}

double PadObj::getMemoryCost() const {
    double inputCost = inputs[0]->size();
    double outputCost = outputs[0]->size();
    double memoryEfficiencyFactor = 1.0;
    const auto &inputDims = inputs[0]->getDims();
    int rank = inputDims.size();
    bool hasPadOnInnerDims = false;

    for (int i = rank / 2; i < rank; ++i) {
        if (pads[i] > 0 || pads[i + rank] > 0) {
            hasPadOnInnerDims = true;
            break;
        }
    }

    if (hasPadOnInnerDims) {
        memoryEfficiencyFactor = 1.2;
    }

    return (inputCost + outputCost) * memoryEfficiencyFactor;
}

double PadObj::getParallelism() const {
    double outputSize = outputs[0]->size();
    const auto &inputDims = inputs[0]->getDims();
    int rank = inputDims.size();
    double outerPadRatio = 0.0;
    double totalPad = 0.0;

    for (int i = 0; i < rank; ++i) {
        double dimPad = pads[i] + pads[i + rank];
        totalPad += dimPad;

        if (i < rank / 2) {
            outerPadRatio += dimPad;
        }
    }

    if (totalPad > 0) {
        outerPadRatio /= totalPad;
    } else {
        outerPadRatio = 0.5;
    }

    double parallelEfficiency = 0.8 + outerPadRatio * 0.2;
    const double MAX_PARALLEL_UNITS = 1024.0;

    return std::min(outputSize * parallelEfficiency, MAX_PARALLEL_UNITS);
}

} // namespace infini