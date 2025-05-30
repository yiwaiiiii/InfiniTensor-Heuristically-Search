#include "operators/transpose.h"

namespace infini {
TransposeObj::TransposeObj(GraphObj *graph, Tensor input, Tensor output,
                           vector<int> permute)
    : OperatorObj(OpType::Transpose, {input}, {output}) {
    auto rank = input->getRank();
    if (permute.empty()) {
        for (size_t i = 0; i < rank; ++i) {
            transposePermute[i] = i;
        }
    } else {
        IT_ASSERT(rank == permute.size());
        transposePermute = std::move(permute);
    }
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> TransposeObj::inferShape(const TensorVec &inputs) {
    const auto A = inputs[0];
    auto input_dim = A->getDims();
    auto output_dim = input_dim;
    int rank = A->getRank();

    for (auto index : transposePermute) {
        IT_ASSERT(index < rank);
    }
    for (int i = 0; i < rank; ++i) {
        output_dim[i] = input_dim[transposePermute[i]];
    }
    return {{output_dim}};
}

std::string TransposeObj::toString() const {
    std::ostringstream os;
    os << type.toString() << "[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> TransposeObj::getWorkloadVector() const {
    vector<int> ret{type.underlying()};
    const Shape shape = outputs[0]->getDims();
    ret.insert(ret.end(), shape.begin(), shape.end());
    return ret;
}

vector<int> TransposeObj::getOpAttrVector() const {
    return {type.underlying()};
}

double TransposeObj::getComputeTime() const {
    double inputSize = inputs[0]->size();
    double complexityFactor = 1.0;
    auto inputDims = inputs[0]->getDims();
    int rank = inputDims.size();
    
    bool hasNonLocalPermutation = false;
    for (int i = 0; i < rank; ++i) {
        if (std::abs(transposePermute[i] - i) > 1) {
            hasNonLocalPermutation = true;
            complexityFactor += 0.2 * std::abs(transposePermute[i] - i);
        }
    }
    
    bool hasInnerDimTranspose = false;
    for (int i = rank - 2; i < rank; ++i) {
        if (i >= 0 && transposePermute[i] != i) {
            hasInnerDimTranspose = true;
            complexityFactor += 0.5;
        }
    }
    
    bool isIdentityPermutation = true;
    for (int i = 0; i < rank; ++i) {
        if (transposePermute[i] != i) {
            isIdentityPermutation = false;
            break;
        }
    }
    
    if (isIdentityPermutation) {
        return 1e-6;
    }
    
    double transposeSpeed = 2e9;
    if (hasNonLocalPermutation) {
        transposeSpeed /= 2.0;
    }
    if (hasInnerDimTranspose) {
        transposeSpeed /= 1.5;
    }
    
    return (inputSize * complexityFactor) / transposeSpeed;
}

double TransposeObj::getMemoryCost() const {
    double inputSize = inputs[0]->size();
    double outputSize = outputs[0]->size();
    double accessEfficiencyFactor = 1.0;
    auto inputDims = inputs[0]->getDims();
    int rank = inputDims.size();
    
    for (int i = rank - 2; i < rank; ++i) {
        if (i >= 0 && transposePermute[i] != i) {
            accessEfficiencyFactor += 0.5;
        }
    }
    
    for (int i = 0; i < rank; ++i) {
        if (std::abs(transposePermute[i] - i) > 1) {
            accessEfficiencyFactor += 0.1 * std::abs(transposePermute[i] - i);
        }
    }
    
    bool isIdentityPermutation = true;
    for (int i = 0; i < rank; ++i) {
        if (transposePermute[i] != i) {
            isIdentityPermutation = false;
            break;
        }
    }
    
    if (isIdentityPermutation) {
        return 0.0;
    }
    
    return (inputSize + outputSize) * accessEfficiencyFactor;
}

double TransposeObj::getParallelism() const {
    double parallelism = std::sqrt(inputs[0]->size());
    
    bool isIdentityPermutation = true;
    for (size_t i = 0; i < transposePermute.size(); ++i) {
        if (transposePermute[i] != static_cast<int>(i)) {
            isIdentityPermutation = false;
            break;
        }
    }
    
    if (isIdentityPermutation) {
        return 1.0;
    }
    
    double parallelEfficiencyFactor = 0.7;
    parallelEfficiencyFactor -= 0.05 * std::max(0, (int)transposePermute.size() - 3);
    
    int contiguousAxes = 0;
    for (size_t i = 0; i < transposePermute.size() - 1; ++i) {
        if (std::abs(transposePermute[i+1] - transposePermute[i]) == 1) {
            contiguousAxes++;
        }
    }
    parallelEfficiencyFactor += 0.05 * contiguousAxes;
    
    const double MAX_PARALLEL_UNITS = 512.0;
    double effectiveParallelism = parallelism * parallelEfficiencyFactor;
    
    return std::min(effectiveParallelism, MAX_PARALLEL_UNITS);
}

DepthToSpaceObj::DepthToSpaceObj(GraphObj *graph, Tensor input, Tensor output,
                                 int blocksize, std::string mode)
    : OperatorObj(OpType::DepthToSpace, {input}, {output}) {
    blockSize = blocksize;
    D2SMode = 0;
    D2SModeString = "DCR";
    if (mode == "CRD") {
        D2SMode = 1;
        D2SModeString = "CRD";
    }
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> DepthToSpaceObj::inferShape(const TensorVec &inputs) {
    const auto A = inputs[0];
    auto inputDim = A->getDims();
    IT_ASSERT(inputDim.size() == 4);
    if (D2SMode == 0) {
        reshapeDim[0] = inputDim[0];
        reshapeDim[1] = blockSize;
        reshapeDim[2] = blockSize;
        reshapeDim[3] = inputDim[1] / (blockSize * blockSize);
        reshapeDim[4] = inputDim[2];
        reshapeDim[5] = inputDim[3];
        transposeDim[0] = reshapeDim[0];
        transposeDim[1] = reshapeDim[3];
        transposeDim[2] = reshapeDim[4];
        transposeDim[3] = reshapeDim[1];
        transposeDim[4] = reshapeDim[5];
        transposeDim[5] = reshapeDim[2];
        outDim[0] = inputDim[0];
        outDim[1] = inputDim[1] / (blockSize * blockSize);
        outDim[2] = inputDim[2] * blockSize;
        outDim[3] = inputDim[3] * blockSize;
    } else {
        reshapeDim[0] = inputDim[0];
        reshapeDim[1] = inputDim[1] / (blockSize * blockSize);
        reshapeDim[2] = blockSize;
        reshapeDim[3] = blockSize;
        reshapeDim[4] = inputDim[2];
        reshapeDim[5] = inputDim[3];
        transposeDim[0] = reshapeDim[0];
        transposeDim[1] = reshapeDim[1];
        transposeDim[2] = reshapeDim[4];
        transposeDim[3] = reshapeDim[2];
        transposeDim[4] = reshapeDim[5];
        transposeDim[5] = reshapeDim[3];
        outDim[0] = inputDim[0];
        outDim[1] = inputDim[1] / (blockSize * blockSize);
        outDim[2] = inputDim[2] * blockSize;
        outDim[3] = inputDim[3] * blockSize;
    }

    return {{outDim}};
}

std::string DepthToSpaceObj::toString() const {
    std::ostringstream os;
    os << type.toString() << "[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> DepthToSpaceObj::getWorkloadVector() const {
    vector<int> ret{type.underlying()};
    const Shape shape = outputs[0]->getDims();
    ret.insert(ret.end(), shape.begin(), shape.end());
    return ret;
}

vector<int> DepthToSpaceObj::getOpAttrVector() const {
    return {type.underlying()};
}

double DepthToSpaceObj::getComputeTime() const {
    double inputSize = inputs[0]->size();
    double reshapeCost = inputSize * 0.01;
    double transposeCost = inputSize;
    
    if (D2SMode == 0) {
        transposeCost *= 1.2;
    }
    
    transposeCost *= (1.0 + std::log2(blockSize) * 0.1);
    double finalReshapeCost = inputSize * 0.01;
    double totalOps = reshapeCost + transposeCost + finalReshapeCost;
    return totalOps / 2e9;
}

double DepthToSpaceObj::getMemoryCost() const {
    double inputSize = inputs[0]->size();
    double outputSize = outputs[0]->size();
    double inputCost = inputSize;
    double transposeCost = inputSize;
    double memoryAccessFactor = 1.0 + std::log2(blockSize) * 0.1;
    
    if (D2SMode == 0) {
        memoryAccessFactor *= 1.1;
    }
    
    double outputCost = outputSize;
    return (inputCost + transposeCost + outputCost) * memoryAccessFactor;
}

double DepthToSpaceObj::getParallelism() const {
    double parallelism = std::sqrt(inputs[0]->size());
    double blockSizeFactor = 1.0 / (1.0 + std::log2(blockSize) * 0.05);
    double modeFactor = 1.0;
    if (D2SMode == 1) {
        modeFactor = 1.1;
    }
    
    const double MAX_PARALLEL_UNITS = 512.0;
    double effectiveParallelism = parallelism * blockSizeFactor * modeFactor;
    return std::min(effectiveParallelism, MAX_PARALLEL_UNITS);
}

}; // namespace infini