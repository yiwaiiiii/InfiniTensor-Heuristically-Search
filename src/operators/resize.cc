#include "operators/resize.h"
#include <cmath>

namespace infini {
ResizeObj::ResizeObj(GraphObj *graph, Tensor input, Tensor output,
                     const std::optional<vector<int>> &axes, Tensor sizes,
                     Tensor scales, Tensor roi,
                     EKeepAspectRatioPolicy ratioPolicy,
                     ENearestMode nearestMode,
                     ECoordinateTransMode coordTransMode)
    : OperatorObj(OpType::Resize, {input}, {output}), coMode(coordTransMode),
      mode(ECoeffMode::nearest), nearestMode(nearestMode),
      ratioPolicy(ratioPolicy) {
    init(input, sizes, scales, roi, axes);
    IT_ASSERT(checkValid(graph));
}

ResizeObj::ResizeObj(GraphObj *graph, Tensor input, Tensor output,
                     const std::optional<vector<int>> &axes, Tensor sizes,
                     Tensor scales, Tensor roi, ECoeffMode mode,
                     EKeepAspectRatioPolicy ratioPolicy,
                     ECoordinateTransMode coordTransMode)
    : OperatorObj(OpType::Resize, {input}, {output}), coMode(coordTransMode),
      mode(mode), nearestMode(ENearestMode::none), ratioPolicy(ratioPolicy) {
    init(input, sizes, scales, roi, axes);
    IT_ASSERT(checkValid(graph));
}

void ResizeObj::init(const Tensor &input, const Tensor &sizes,
                     const Tensor &scales, const Tensor &roi,
                     const std::optional<vector<int>> &axes) {
    IT_ASSERT(!(nullptr != sizes && nullptr != scales));

    if (nullptr != sizes) {
        setGivenSizes(true);
        inputs.push_back(sizes);
        InitBySizes(input, sizes, axes);
    } else if (nullptr != scales) {
        setGivenSizes(false);
        inputs.push_back(scales);
        InitByScales(input, scales, axes);
    }

    if (ECoordinateTransMode::tfCropAndResize == coMode) {
        IT_ASSERT(nullptr != roi);
        inputs.push_back(roi);
        IT_ASSERT(roi->getRank() == 1);
        IT_ASSERT((size_t)roi->getDims()[0] == this->axes.size() * 2);

        size_t nDims = input->getRank();
        for (size_t i = 0; i < nDims; ++i) {
            this->roi.emplace_back(0);
        }
        for (size_t i = 0; i < nDims; ++i) {
            this->roi.emplace_back(1);
        }

        Runtime runtime = NativeCpuRuntimeObj::getInstance();
        std::shared_ptr<float> dataObj((float *)runtime->alloc(roi->getBytes()),
                                       [&](float *p) { runtime->dealloc(p); });
        auto data = dataObj.get();
        roi->getRuntime()->copyBlobToCPU(
            (void *)data, roi->getRawDataPtr<void *>(), roi->getBytes());

        for (size_t i = 0; i < this->axes.size(); ++i) {
            this->roi[this->axes[i]] = data[i];
            this->roi[this->axes[i] + nDims] = data[i + this->axes.size()];
        }
    }
}

void ResizeObj::InitBySizes(Tensor input, Tensor sizes,
                            const std::optional<vector<int>> &axes) {
    IT_ASSERT(sizes != nullptr);
    size_t size = sizes->getDims()[0];
    IT_ASSERT(size == input->getRank() ||
              (axes != std::nullopt && size == (*axes).size()));

    if (axes == std::nullopt) {
        for (size_t i = 0; i < input->getRank(); ++i) {
            this->axes.emplace_back(i);
        }
    } else {
        for (size_t i = 0; i < (*axes).size(); ++i) {
            auto val = (*axes)[i];
            if (val < 0) {
                IT_TODO_HALT();
            }
            IT_ASSERT((size_t)val < inputs[0]->getRank());
            this->axes.emplace_back(val);
        }
    }

    for (size_t i = 0; i < input->getRank(); ++i) {
        this->scales.emplace_back(1);
    }

    IT_ASSERT(sizes->getDataBlob() != nullptr);
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    std::shared_ptr<int64_t> dataObj(
        (int64_t *)runtime->alloc(sizes->getBytes()),
        [&](int64_t *p) { runtime->dealloc(p); });
    auto data = dataObj.get();
    sizes->getRuntime()->copyBlobToCPU(
        (void *)data, sizes->getRawDataPtr<void *>(), sizes->getBytes());

    auto inDims = input->getDims();
    int n = this->axes.size();
    switch (ratioPolicy) {
    case EKeepAspectRatioPolicy::stretch:
        for (int i = 0; i < n; ++i) {
            scales[this->axes[i]] =
                (float)data[i] / (float)inDims[this->axes[i]];
        }
        break;
    case EKeepAspectRatioPolicy::notLarger: {
        float scale = (float)data[0] / (float)inDims[this->axes[0]];
        for (int i = 1; i < n; ++i) {
            auto tmp = (float)data[i] / (float)inDims[this->axes[i]];
            scale = scale < tmp ? scale : tmp;
        }
        for (int i = 0; i < n; ++i) {
            scales[this->axes[i]] = scale;
        }
        break;
    }
    case EKeepAspectRatioPolicy::notSmaller: {
        float scale = (float)data[0] / (float)inDims[this->axes[0]];
        for (int i = 1; i < n; ++i) {
            auto tmp = (float)data[i] / (float)inDims[this->axes[i]];
            scale = scale > tmp ? scale : tmp;
        }
        for (int i = 0; i < n; ++i) {
            scales[this->axes[i]] = scale;
        }
        break;
    }
    default:
        IT_ASSERT(0);
    }
}

void ResizeObj::InitByScales(Tensor input, Tensor scales,
                             const std::optional<vector<int>> &axes) {
    IT_ASSERT(scales != nullptr);
    size_t size = scales->getDims()[0];
    IT_ASSERT(size == input->getRank() ||
              (axes != std::nullopt && size == (*axes).size()));

    IT_ASSERT(scales->getDataBlob() != nullptr);
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    std::shared_ptr<float> dataObj((float *)runtime->alloc(scales->getBytes()),
                                   [&](float *p) { runtime->dealloc(p); });
    auto data = dataObj.get();
    scales->getRuntime()->copyBlobToCPU(
        (void *)data, scales->getRawDataPtr<void *>(), scales->getBytes());

    for (size_t i = 0; i < input->getRank(); ++i) {
        this->scales.emplace_back(1);
    }

    if (axes == std::nullopt) {
        for (size_t i = 0; i < input->getRank(); ++i) {
            this->axes.emplace_back(i);
            IT_ASSERT(data[i] > 0);
            this->scales[i] = data[i];
        }
    } else {
        for (size_t i = 0; i < (*axes).size(); ++i) {
            auto val = (*axes)[i];
            if (val < 0) {
                IT_TODO_HALT();
            }
            IT_ASSERT((size_t)val < inputs[0]->getRank());
            this->axes.emplace_back(val);
            IT_ASSERT(data[i] > 0);
            this->scales[val] = data[i];
        }
    }
}

vector<DataType> ResizeObj::inferDataType(const TensorVec &inputs) const {
    IT_ASSERT(inputs.size() == 2 || inputs.size() == 3);
    if (inputs.size() == 3) {
        auto roi = inputs[2];
        IT_ASSERT(roi && roi->getDType() == DataType::Float32);
    }
    if (isResizeBySizes()) {
        auto sizes = inputs[1];
        IT_ASSERT(sizes && sizes->getDType() == DataType::Int64);
    } else {
        auto scales = inputs[1];
        IT_ASSERT(scales && scales->getDType() == DataType::Float32);
    }
    return {inputs[0]->getDType()};
}

float ResizeObj::round_int(float x) const {
    return (x > 0.0) ? floor(x + 0.5) : ceil(x - 0.5);
}

optional<vector<Shape>> ResizeObj::inferShape(const TensorVec &inputs) {
    auto inDims = inputs[0]->getDims();
    Shape ret = inDims;
    int rank = inputs[0]->getRank();
    for (int i = 0; i < rank; ++i) {
        int size = round_int(scales[i] * inDims[i]);
        ret[i] = size;
    }

    return {{ret}};
}

std::string ResizeObj::toString() const {
    std::ostringstream os;
    os << "Resize"
       << "[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    if (inputs.size() == 3) {
        os << "roi=" << vecToString(inputs[2]->getDims()) << ",";
    }
    if (isResizeBySizes()) {
        os << "sizes=" << vecToString(inputs[1]->getDims()) << ",";
    } else {
        os << "scales=" << vecToString(inputs[1]->getDims()) << ",";
    }
    os << "axes=" << vecToString(axes) << ",";
    os << "coMode=" << enum_to_underlying(coMode) << ",";
    os << "nearestMode=" << enum_to_underlying(nearestMode) << ",";
    os << "ratioPolicy=" << enum_to_underlying(ratioPolicy) << ",";

    os << "input=" << inputs[0]->getGuid() << ",";
    os << inputs[1]->getGuid() << ",";
    if (inputs.size() == 3) {
        os << inputs[2]->getGuid() << ",";
    }
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> ResizeObj::getWorkloadVector() const {
    vector<int> ret = inputs[0]->getDims();
    for (size_t i = 0; i < outputs[0]->getRank(); ++i) {
        ret.emplace_back(outputs[0]->getDims()[i]);
    }
    ret.emplace_back(enum_to_underlying(coMode));
    ret.emplace_back(enum_to_underlying(nearestMode));
    ret.emplace(ret.begin(), type.underlying());
    return ret;
}

vector<int> ResizeObj::getOpAttrVector() const {
    vector<int> ret = axes;
    ret.emplace_back(enum_to_underlying(coMode));
    ret.emplace_back(enum_to_underlying(nearestMode));
    ret.emplace_back(enum_to_underlying(ratioPolicy));
    ret.emplace(ret.begin(), type.underlying());
    return ret;
}

double ResizeObj::getComputeTime() const {
    const auto &inputDims = inputs[0]->getDims();
    const auto &outputDims = outputs[0]->getDims();
    int64_t inputSize = inputs[0]->size();
    int64_t outputSize = outputs[0]->size();
    double operationFactor;
    if (mode == ECoeffMode::nearest) {
        operationFactor = 1.0;
    } else if (mode == ECoeffMode::linear) {
        operationFactor = 2.5;
    } else if (mode == ECoeffMode::cubic) {
        operationFactor = 6.0;
    } else {
        operationFactor = 2.0;
    }
    double transformFactor = 1.0;
    if (coMode == ECoordinateTransMode::alignCorners) {
        transformFactor = 1.1;
    } else if (coMode == ECoordinateTransMode::tfCropAndResize) {
        transformFactor = 1.3;
    }
    int rank = inputDims.size();
    int resizeDims = axes.size();
    double dimFactor = 1.0 + 0.1 * resizeDims;
    double totalOps = outputSize * operationFactor * transformFactor * dimFactor;
    double processingRate = 1e9;
    return totalOps / processingRate;
}

double ResizeObj::getMemoryCost() const {
    double inputCost = inputs[0]->size();
    double paramCost = inputs[1]->size();
    double roiCost = 0.0;
    if (inputs.size() > 2 && coMode == ECoordinateTransMode::tfCropAndResize) {
        roiCost = inputs[2]->size();
    }
    double outputCost = outputs[0]->size();
    double accessFactor = 1.0;
    if (mode != ECoeffMode::nearest) {
        int interpolationPoints = 0;
        if (mode == ECoeffMode::linear) {
            interpolationPoints = 1 << axes.size();
        } else if (mode == ECoeffMode::cubic) {
            interpolationPoints = std::pow(4, axes.size());
        }
        inputCost *= interpolationPoints / 2.0;
    }
    return (inputCost + paramCost + roiCost) * accessFactor + outputCost;
}

double ResizeObj::getParallelism() const {
    double parallelism = outputs[0]->size();
    double effectiveFactor = 0.8;
    if (mode != ECoeffMode::nearest) {
        effectiveFactor *= 0.9;
    }
    if (coMode == ECoordinateTransMode::tfCropAndResize) {
        effectiveFactor *= 0.95;
    }
    const double MAX_PARALLEL_UNITS = 2048.0;
    double effectiveParallelism = parallelism * effectiveFactor;
    return std::min(effectiveParallelism, MAX_PARALLEL_UNITS);
}

} // namespace infini