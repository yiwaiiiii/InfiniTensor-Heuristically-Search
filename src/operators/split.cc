#include "operators/split.h"
#include "utils/operator_utils.h"
#include <numeric>

namespace infini {
SplitObj::SplitObj(GraphObj *graph, Tensor input,
                   std::optional<TensorVec> outputs, int dim, int num)
    : OperatorObj(OpType::Split, {input},
                  ((!outputs) ? TensorVec(num, nullptr) : std::move(*outputs))),
      dim(get_real_axis(dim, input->getRank())), num(num), ratio({}) {
    int dimSize = input->getDims().at(this->dim);
    int pieceSize = dimSize / num;
    int lastSize = dimSize - pieceSize * num;

    if (lastSize > 0) {
        ratio = std::vector<int>(num - 1, pieceSize);
        ratio.emplace_back(lastSize + pieceSize);
    } else
        ratio = std::vector<int>(num, pieceSize);

    IT_ASSERT(checkValid(graph));
}

SplitObj::SplitObj(GraphObj *graph, Tensor input,
                   std::optional<TensorVec> outputs, int dim,
                   const vector<int> &ratio)
    : OperatorObj(OpType::Split, {input},
                  ((!outputs) ? TensorVec{nullptr} : (*outputs))),
      dim(get_real_axis(dim, input->getRank())), num(-1), ratio(ratio) {
    num = ratio.size();
    if (!outputs) {
        TensorVec tmp(num, nullptr);
        this->outputs = tmp;
    }
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> SplitObj::inferShape(const TensorVec &inputs) {
    IT_ASSERT(num != -1 && ratio.size() != 0);
    auto inputDims = inputs[0]->getDims();
    int totalSize = inputDims.at(dim);
    int ratioSum = std::accumulate(ratio.begin(), ratio.end(), 0);
    IT_ASSERT(totalSize % ratioSum == 0);

    int pieceSize = totalSize / ratioSum;

    vector<Shape> ret;
    Shape outShape = inputDims;
    for (int i = 0; i < num; i++) {
        outShape[dim] = pieceSize * ratio.at(i);
        ret.push_back(outShape);
    }
    return {ret};
}

vector<int> SplitObj::getWorkloadVector() const {
    vector<int> ret = inputs[0]->getDims();
    ret.emplace(ret.begin(), type.underlying());
    ret.emplace_back(dim);
    ret.emplace_back(num);
    return ret;
}

vector<int> SplitObj::getOpAttrVector() const {
    return {type.underlying(), dim, num};
}

string SplitObj::toString() const {
    std::ostringstream os;
    os << "Split[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "dim=" << dim << ",";
    os << "num= " << num << ",";
    os << "ratio= " << vecToString(ratio) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=";
    for (auto i = 0; i < num; i++)
        os << outputs[i]->getGuid() << ",";
    os << ")";
    return os.str();
}

double SplitObj::getComputeTime() const {
    double inputSize = inputs[0]->size();
    const auto &inputDims = inputs[0]->getDims();
    int rank = inputDims.size();
    
    double copyFactor = 0.05;
    if (dim > 0) {
        copyFactor = 0.2 + (double)dim / rank * 0.8;
    }
    
    double splitFactor = 1.0 + std::log2(num) * 0.05;
    double totalOps = inputSize * copyFactor * splitFactor;
    return totalOps / 10e9;
}

double SplitObj::getMemoryCost() const {
    double inputCost = inputs[0]->size();
    double outputCost = 0.0;
    for (int i = 0; i < num; i++) {
        if (outputs[i] != nullptr) {
            outputCost += outputs[i]->size();
        }
    }
    
    double accessFactor = 1.0;
    const auto &inputDims = inputs[0]->getDims();
    int rank = inputDims.size();
    
    if (dim > rank / 2) {
        accessFactor = 1.0 + (double)(dim - rank / 2) / (rank / 2) * 0.5;
    }
    
    if (dim == 0) {
        accessFactor *= 0.2;
    }
    
    return (inputCost + outputCost) * accessFactor;
}

double SplitObj::getParallelism() const {
    double splitParallelism = std::min(static_cast<double>(num), 32.0);
    double withinSplitParallelism = 1.0;
    
    if (!outputs.empty() && outputs[0] != nullptr) {
        double avgOutputSize = outputs[0]->size();
        withinSplitParallelism = std::sqrt(avgOutputSize);
    } else if (!inputs.empty()) {
        double avgOutputSize = inputs[0]->size() / num;
        withinSplitParallelism = std::sqrt(avgOutputSize);
    }
    
    withinSplitParallelism = std::min(withinSplitParallelism, 64.0);
    double totalParallelism = splitParallelism * withinSplitParallelism;
    const double MAX_PARALLEL_UNITS = 1024.0;
    
    return std::min(totalParallelism, MAX_PARALLEL_UNITS);
}

} // namespace infini