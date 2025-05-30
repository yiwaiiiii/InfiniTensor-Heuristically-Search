#include "operators/GBMM.h"

namespace infini {

GBMMObj::GBMMObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, int dilation,
                 [[maybe_unused]] Tensor bias, ActType act)
    : OperatorObj(OpType::GBMM, {A, B}, {C}), dilation(dilation), act(act),
      b(A->getDims()[0]), m(A->getDims()[1]), w((A->getDims()[2] - 1) / 2),
      n(B->getDims()[2]) {
    IT_ASSERT(checkValid(graph));
}

string GBMMObj::toString() const {
    std::ostringstream os;
    os << "GBMM(["
       << ",act=" << (int)act << "],A=" << inputs[0]->getGuid()
       << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
       << ", TTbmwnd: " << this->getB() << ", " << this->getM() << ", "
       << this->getW() << ", " << this->getN() << ", " << this->getDilation()
       << ")";
    return os.str();
}

optional<vector<Shape>> GBMMObj::inferShape(const TensorVec &inputs) {
    auto A = inputs[0], B = inputs[1];
    b = A->getDims()[0];
    m = A->getDims()[1];
    w = (A->getDims()[2] - 1) / 2;
    n = B->getDims()[2];

    IT_ASSERT(A->getRank() == 3 && B->getRank() == 3);
    IT_ASSERT(A->getDims()[0] == B->getDims()[0]);
    IT_ASSERT(A->getDims()[1] == B->getDims()[1]);
    IT_ASSERT(A->getDims()[2] % 2 != 0);
    return {{{b, m, n}}};
}

vector<int> GBMMObj::getWorkloadVector() const {
    return {type.underlying(), b, m, w, n, dilation, enum_to_underlying(act)};
}

vector<int> GBMMObj::getOpAttrVector() const {
    return {type.underlying(), dilation, enum_to_underlying(act)};
}

double GBMMObj::getComputeTime() const {
    int64_t batchSize = b;
    int64_t seqLength = m;
    int64_t width = w;
    int64_t outDim = n;
    int64_t dilationFactor = dilation;
    int64_t actualBandWidth = 2 * width + 1;
    
    double multiplyAddOps = batchSize * seqLength * outDim * actualBandWidth;
    double dilationPenalty = 1.0;
    if (dilationFactor > 1) {
        dilationPenalty = 1.0 + std::log2(dilationFactor) * 0.15;
    }
    
    double actCost = 0.0;
    if (act != ActType::None) {
        actCost = batchSize * seqLength * outDim * 0.1;
    }
    
    double totalOps = multiplyAddOps * dilationPenalty + actCost;
    return totalOps / 2e9;
}

double GBMMObj::getMemoryCost() const {
    double costA = inputs[0]->size();
    double costB = inputs[1]->size();
    double costC = outputs[0]->size();
    double memoryEfficiencyFactor = 1.0;
    if (dilation > 1) {
        memoryEfficiencyFactor = 1.0 + dilation * 0.05;
    }
    return (costA + costB) * memoryEfficiencyFactor + costC;
}

double GBMMObj::getParallelism() const {
    double batchParallel = b;
    double seqParallel = m;
    double featureParallel = std::min(n, 16);
    double totalParallelism = batchParallel * seqParallel * featureParallel;
    const double MAX_PARALLEL_UNITS = 2048.0;
    double efficiencyFactor = 1.0;
    if (dilation > 1) {
        efficiencyFactor = 1.0 - std::log2(dilation) * 0.05;
    }
    return std::min(totalParallelism * efficiencyFactor, MAX_PARALLEL_UNITS);
}

} // namespace infini