#ifdef BUILD_NNET

#include "operators/membound.h"
#include "nnet/Visitor/CheckOOBVisitor.h"
#include "nnet/Visitor/HashVisitor.h"
#include "nnet/Visitor/MergeMemboundMutator.h"

namespace infini {

MemBoundObj::MemBoundObj(GraphObj *graph, const TensorVec &input,
                         const TensorVec &output,
                         const std::vector<nnet::Tensor> &nnetInputs,
                         nnet::Expr expr, double exec_time, std::string hint)
    : OperatorObj(OpType::MemBound, input, output), nnetInputs(nnetInputs),
      expr(expr), exec_time(exec_time), hint(hint) {
    IT_ASSERT(checkValid(graph));
    IT_ASSERT(!checkOOB(expr));
    hash = calcHash(expr);

    if (auto mergedExpr =
            nnet::MergeMemboundMutator({expr}).merge(false, true)) {
        simplifiedExpr = mergedExpr;
        IT_ASSERT(!checkOOB(simplifiedExpr));
        simplifiedHash = calcHash(simplifiedExpr);
    } else {
        simplifiedExpr = expr;
        simplifiedHash = hash;
    }
}

string MemBoundObj::toString() const {
    std::ostringstream os;
    os << "MemBound[" << getGuid() << "](";
    for (size_t i = 0; i < inputs.size(); ++i) {
        os << "i" << i << "=" << inputs[i]->getGuid();
        if (i != inputs.size() - 1)
            os << " ";
    }
    os << ", ";
    for (size_t i = 0; i < outputs.size(); ++i) {
        os << "o" << i << "=" << outputs[i]->getGuid();
        if (i != outputs.size() - 1)
            os << " ";
    }
    os << ", ";
    os << "exec_time=" << exec_time << ", ";
    os << "NNet Inputs=[";
    for (const auto &tensor : nnetInputs)
        os << tensor->toReadable() << ",";
    os << "]";
    os << ", ExprHash=" << hash;
    os << ", SimplifiedExprHash=" << simplifiedHash;
    os << ")\n";
    os << ">>> Original expr\n"
       << (expr ? expr->toReadable() : "Empty expression") << "\n";
    os << ">>> Simplified expr\n"
       << (simplifiedExpr ? simplifiedExpr->toReadable() : "Empty expression")
       << "\n";
    return os.str();
}

optional<vector<Shape>> MemBoundObj::inferShape(const TensorVec &inputs) {
    if (inputs.size() != nnetInputs.size())
        return {};
    for (size_t i = 0; i < inputs.size(); ++i)
        if (inputs[i]->getDims() != nnetInputs[i]->getShape())
            return {};
    return {{nnet::as<nnet::RangeOpNode>(expr)->getOutputShape()}};
}

vector<int> MemBoundObj::getWorkloadVector() const {
    return {type.underlying(), (int)simplifiedHash};
}

vector<int> MemBoundObj::getOpAttrVector() const { return getWorkloadVector(); }

HashType MemBoundObj::calcHash(nnet::Expr expr) {
    return nnet::HashVisitor().dispatch(expr);
}

bool MemBoundObj::checkOOB(nnet::Expr expr) {
    return nnet::CheckOOBVisitor().checkRangeOp(
        nnet::as<nnet::RangeOpNode>(expr));
}

double MemBoundObj::getComputeTime() const {
    double baseTime = exec_time;
    double bandwidth = double(200) * 1024 * 1024 * 1024;
    double totalDataSize = 0.0;
    for (const auto &input : inputs) {
        totalDataSize += input->size();
    }
    for (const auto &output : outputs) {
        totalDataSize += output->size();
    }
    double memAccessTime = (totalDataSize * 4) / (bandwidth * 1e9) * 1000;
    double complexityFactor = 1.0;
    if (simplifiedExpr) {
        bool hasIrregularAccess = hint.find("irregular") != std::string::npos;
        if (hasIrregularAccess) {
            complexityFactor *= 1.5;
        }
        complexityFactor *= (1.0 + std::log(1.0 + (simplifiedHash % 1000)) / 10.0);
    }
    double finalTime;
    if (baseTime > 0.0) {
        finalTime = baseTime * (0.8 + 0.2 * complexityFactor);
    } else {
        finalTime = memAccessTime * complexityFactor;
    }
    if (hint.find("compute_overlap") != std::string::npos) {
        finalTime *= 0.8;
    }
    return finalTime / 1000.0;
}

double MemBoundObj::getMemoryCost() const {
    double inputCost = 0.0;
    for (const auto &input : inputs) {
        inputCost += input->size();
    }
    double outputCost = 0.0;
    for (const auto &output : outputs) {
        outputCost += output->size();
    }
    double intermediateAccessCost = 0.0;
    if (simplifiedExpr) {
        double totalSize = inputCost + outputCost;
        double complexityFactor = std::log(1.0 + simplifiedHash % 10000) / 10.0;
        intermediateAccessCost = totalSize * complexityFactor;
    }
    return inputCost + outputCost + intermediateAccessCost;
}

double MemBoundObj::getParallelism() const {
    double parallelism = 1.0;
    if (!outputs.empty()) {
        size_t maxOutputSize = 0;
        for (const auto &output : outputs) {
            maxOutputSize = std::max(maxOutputSize, output->size());
        }
        parallelism = std::sqrt(maxOutputSize) * 4.0;
    }
    if (hint.find("parallel") != std::string::npos) {
        parallelism *= 2.0;
    } else if (hint.find("sequential") != std::string::npos) {
        parallelism /= 2.0;
    }
    const double MAX_PARALLEL_UNITS = 1024.0;
    return std::min(parallelism, MAX_PARALLEL_UNITS);
}

} // namespace infini

#endif