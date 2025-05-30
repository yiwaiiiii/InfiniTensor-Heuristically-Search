#include "operators/pooling.h"

namespace infini {

PoolingObj::PoolingObj(GraphObj *graph, OpType optype, Tensor input,
                       Tensor output, int kh, int kw, int dh, int dw, int ph,
                       int pw, int sh, int sw, int ceilMode)
    : OperatorObj(optype, {input}, {output}), kh(kh), kw(kw), dh(dh), dw(dw),
      ph(ph), pw(pw), sh(sh), sw(sw), ceilMode(ceilMode),
      n(input->getDims().at(0)), c(input->getDims().at(1)),
      h(input->getRank() == 3 ? 1 : input->getDims().at(2)),
      w(input->getRank() == 3 ? input->getDims().at(2)
                              : input->getDims().at(3)) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> PoolingObj::inferShape(const TensorVec &inputs) {
    const auto &input = inputs[0];
    int oh, ow;
    if (ceilMode) {
        oh = ceil(((float)(h + 2 * ph - dh * (kh - 1) - 1)) / sh + 1);
        ow = ceil(((float)(w + 2 * pw - dw * (kw - 1) - 1)) / sw + 1);
    } else {
        oh = floor(((float)(h + 2 * ph - dh * (kh - 1) - 1)) / sh + 1);
        ow = floor(((float)(w + 2 * pw - dw * (kw - 1) - 1)) / sw + 1);
    }

    auto ret = input->getDims();
    if (input->getRank() == 4) {
        ret[input->getRank() - 2] = oh;
    }
    ret[input->getRank() - 1] = ow;
    return {{ret}};
}

std::string PoolingObj::toString() const {
    std::ostringstream os;
    os << type.toString() << "[" << getGuid() << "]";
    os << "(";
    os << "k=[" << kh << "," << kw << "],";
    os << "p=[" << ph << "," << pw << "],";
    os << "s=[" << sh << "," << sw << "],";
    os << "d=[" << dh << "," << dw << "],";
    os << "ceil mode=" << ceilMode << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> PoolingObj::getWorkloadVector() const {
    return {type.underlying(), n, c, h, w, kh, kw, ph, pw, sh, sw, dh, dw,
            ceilMode};
}

vector<int> PoolingObj::getOpAttrVector() const {
    return {type.underlying(), kh, kw, ph, pw, sh, sw, dh, dw, ceilMode};
}

double PoolingObj::getComputeTime() const {
    int oh, ow;
    if (ceilMode) {
        oh = ceil(((float)(h + 2 * ph - dh * (kh - 1) - 1)) / sh + 1);
        ow = ceil(((float)(w + 2 * pw - dw * (kw - 1) - 1)) / sw + 1);
    } else {
        oh = floor(((float)(h + 2 * ph - dh * (kh - 1) - 1)) / sh + 1);
        ow = floor(((float)(w + 2 * pw - dw * (kw - 1) - 1)) / sw + 1);
    }

    int64_t outputElements = n * c * oh * ow;
    int64_t kernelSize = kh * kw;

    double opsPerElement;
    if (type == OpType::MaxPool) {
        opsPerElement = kernelSize - 1;
    } else if (type == OpType::AveragePool) {
        opsPerElement = kernelSize + 1;
    } else {
        opsPerElement = kernelSize;
    }

    double dilationFactor = 1.0;
    if (dh > 1 || dw > 1) {
        dilationFactor = 1.0 + std::log2(std::max(dh, dw)) * 0.1;
    }

    double totalOps = outputElements * opsPerElement * dilationFactor;
    return totalOps / 2e9;
}

double PoolingObj::getMemoryCost() const {
    int oh, ow;
    if (ceilMode) {
        oh = ceil(((float)(h + 2 * ph - dh * (kh - 1) - 1)) / sh + 1);
        ow = ceil(((float)(w + 2 * pw - dw * (kw - 1) - 1)) / sw + 1);
    } else {
        oh = floor(((float)(h + 2 * ph - dh * (kh - 1) - 1)) / sh + 1);
        ow = floor(((float)(w + 2 * pw - dw * (kw - 1) - 1)) / sw + 1);
    }

    int64_t outputSize = n * c * oh * ow;

    double inputAccessRatio = 1.5;
    double inputSize = inputs[0]->size() * inputAccessRatio;

    double strideEffect = 1.0;
    if (sh > 1 || sw > 1) {
        strideEffect = 1.0 + std::log2(std::max(sh, sw)) * 0.05;
    }

    double dilationEffect = 1.0;
    if (dh > 1 || dw > 1) {
        dilationEffect = 1.0 + std::log2(std::max(dh, dw)) * 0.1;
    }

    return (inputSize * strideEffect * dilationEffect + outputSize);
}

double PoolingObj::getParallelism() const {
    int oh, ow;
    if (ceilMode) {
        oh = ceil(((float)(h + 2 * ph - dh * (kh - 1) - 1)) / sh + 1);
        ow = ceil(((float)(w + 2 * pw - dw * (kw - 1) - 1)) / sw + 1);
    } else {
        oh = floor(((float)(h + 2 * ph - dh * (kh - 1) - 1)) / sh + 1);
        ow = floor(((float)(w + 2 * pw - dw * (kw - 1) - 1)) / sw + 1);
    }

    double batchParallel = n;
    double channelParallel = c;
    double spatialParallel = oh * ow;

    double effectiveChannelParallel = std::min(channelParallel, 64.0);
    double effectiveSpatialParallel = std::min(spatialParallel, 256.0);

    double totalParallelism = batchParallel * effectiveChannelParallel * effectiveSpatialParallel;

    const double MAX_PARALLEL_UNITS = 2048.0;

    return std::min(totalParallelism, MAX_PARALLEL_UNITS);
}

}; // namespace infini