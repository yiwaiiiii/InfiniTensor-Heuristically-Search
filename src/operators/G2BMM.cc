#include "operators/G2BMM.h"

namespace infini {

G2BMMObj::G2BMMObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, int width,
                   int dilation, [[maybe_unused]] Tensor bias, ActType act)
    : OperatorObj(OpType::G2BMM, {A, B}, {C}), width(width), dilation(dilation),
      act(act), b(A->getDims()[0]), m(A->getDims()[1]), k(A->getDims()[2]) {
    IT_ASSERT(checkValid(graph));
}

string G2BMMObj::toString() const {
    std::ostringstream os;
    os << "G2BMM(["
       << "width=" << width << ",act=" << enum_to_underlying(act)
       << "],A=" << inputs[0]->getGuid() << ",B=" << inputs[1]->getGuid()
       << ",C=" << outputs[0]->getGuid() << ", TTbmnkd: " << this->getB()
       << ", " << this->getM() << ", " << this->getWidth() << ", "
       << inputs[1]->getDims()[2] << ", " << this->getDilation() << ")";
    return os.str();
}

optional<vector<Shape>> G2BMMObj::inferShape(const TensorVec &inputs) {
    auto A = inputs[0], B = inputs[1];
    b = A->getDims()[0];
    m = A->getDims()[1];
    k = A->getDims()[2];

    IT_ASSERT(A->getRank() == 3 && B->getRank() == 3);
    IT_ASSERT(A->getDims()[0] == B->getDims()[0]);
    IT_ASSERT(A->getDims()[1] == B->getDims()[1]);
    IT_ASSERT(A->getDims()[2] == B->getDims()[2]);
    IT_ASSERT(width >= 0);
    int n(2 * width + 1);
    return {{{b, m, n}}};
}

vector<int> G2BMMObj::getWorkloadVector() const {
    return {type.underlying(),      b, m, k, width, dilation,
            enum_to_underlying(act)};
}

vector<int> G2BMMObj::getOpAttrVector() const {
    return {type.underlying(), width, dilation, enum_to_underlying(act)};
}

double G2BMMObj::getComputeTime() const {
    int64_t batchSize = getB();
    int64_t seqLength = getM();
    int64_t featureDim = getK();
    int64_t windowWidth = getWidth();
    int64_t dilationFactor = getDilation();
    int64_t outputWidth = 2 * windowWidth + 1;
    
    double multiplyAddOps = batchSize * seqLength * outputWidth * featureDim;
    double dilationPenalty = std::log2(dilationFactor + 1) * 0.1 + 1.0;
    
    double actCost = 0.0;
    if (act != ActType::None) {
        actCost = batchSize * seqLength * outputWidth * 0.1;
    }
    
    double totalOps = multiplyAddOps * dilationPenalty + actCost;
    return totalOps / 2e9;
}

double G2BMMObj::getMemoryCost() const {
    double costA = inputs[0]->size();
    double costB = inputs[1]->size();
    double costC = outputs[0]->size();
    double memoryEfficiencyFactor = 1.0 + dilation * 0.05;
    return (costA + costB) * memoryEfficiencyFactor + costC;
}

double G2BMMObj::getParallelism() const {
    int64_t batchParallel = getB();
    int64_t seqParallel = getM();
    int64_t windowParallel = std::min(2 * width + 1, 8);
    double totalParallelism = batchParallel * seqParallel * windowParallel;
    const double MAX_PARALLEL_UNITS = 2048.0;
    return std::min(totalParallelism, MAX_PARALLEL_UNITS);
}

} // namespace infini