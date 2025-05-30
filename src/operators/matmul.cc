#include "operators/matmul.h"
#include "utils/operator_utils.h"
#include <numeric>

namespace infini {

MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                     bool transB, [[maybe_unused]] Tensor bias, ActType act,
                     std::string computeType)
    : OperatorObj(OpType::MatMul,
                  bias ? TensorVec{A, B, bias} : TensorVec{A, B}, {C}),
      transA(transA), transB(transB), act(act), b(1), computeType(computeType) {
    IT_ASSERT(checkValid(graph));
}

string MatmulObj::toString() const {
    std::ostringstream os;
    os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B")
       << ",act=" << enum_to_underlying(act) << "],A=" << inputs[0]->getGuid()
       << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
       << ",bmnk=[" << b << "," << m << "," << n << "," << k << "])"
       << ",computeType=" << computeType;
    return os.str();
}

optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs) {
    auto A = inputs[0], B = inputs[1];
    auto shapeA = A->getDims();
    auto shapeB = B->getDims();
    int rankA = A->getRank();
    int rankB = B->getRank();
    Shape shapeA1(shapeA.begin(), shapeA.begin() + (rankA - 2));
    Shape shapeB1(shapeB.begin(), shapeB.begin() + (rankB - 2));
    Shape ret = infer_broadcast(shapeA1, shapeB1);
    if (ret.empty()) {
        b = 1;
    } else {
        b = std::accumulate(ret.begin(), ret.end(), 1, std::multiplies<int>());
    }
    auto kA = *(transA ? shapeA.rbegin() + 1 : shapeA.rbegin());
    auto kB = *(transB ? shapeB.rbegin() : shapeB.rbegin() + 1);
    IT_ASSERT(kA == kB);
    m = *(transA ? shapeA.rbegin() : shapeA.rbegin() + 1);
    n = *(transB ? shapeB.rbegin() + 1 : shapeB.rbegin());
    k = kA;
    ret.emplace_back(m);
    ret.emplace_back(n);
    return {{ret}};
}

vector<int> MatmulObj::getWorkloadVector() const {
    return {type.underlying(),      b, m, n, k, transA, transB,
            enum_to_underlying(act)};
}

vector<int> MatmulObj::getOpAttrVector() const {
    return {type.underlying(), transA, transB, enum_to_underlying(act)};
}

double MatmulObj::getComputeTime() const {
    int64_t batchSize = b;
    int64_t M = m;
    int64_t N = n;
    int64_t K = k;
    double basicOps = 2.0 * batchSize * M * N * K;
    double transposePenalty = 1.0;
    if (transA || transB) {
        transposePenalty = 1.05;
    }
    double actCost = 0.0;
    if (act != ActType::None) {
        actCost = batchSize * M * N * 0.1;
    }
    double biasCost = 0.0;
    if (inputs.size() > 2) {
        biasCost = batchSize * M * N;
    }
    double typeFactor = 1.0;
    if (computeType == "half") {
        typeFactor = 0.5;
    } else if (computeType == "double") {
        typeFactor = 2.0;
    }
    double totalOps = (basicOps * transposePenalty + actCost + biasCost) / typeFactor;
    return totalOps / 5e9;
}

double MatmulObj::getMemoryCost() const {
    double costA = inputs[0]->size();
    double costB = inputs[1]->size();
    double costBias = 0.0;
    if (inputs.size() > 2) {
        costBias = inputs[2]->size();
    }
    double costC = outputs[0]->size();
    double cacheFactor = 0.2;
    if (transA || transB) {
        cacheFactor *= 1.5;
    }
    return (costA + costB) * cacheFactor + costBias + costC;
}

double MatmulObj::getParallelism() const {
    double batchParallel = b;
    double outputParallel = m * n;
    double innerParallel = std::min(std::sqrt(k), 8.0);
    double totalParallelism = batchParallel * outputParallel * innerParallel;
    const double MAX_PARALLEL_UNITS = 4096.0;
    double transposeFactor = 1.0;
    if (transA && transB) {
        transposeFactor = 0.9;
    } else if (transA || transB) {
        transposeFactor = 0.95;
    }
    return std::min(totalParallelism * transposeFactor, MAX_PARALLEL_UNITS);
}

} // namespace infini