#include "operators/send.h"

namespace infini {
SendObj::SendObj(GraphObj *graph, Tensor input, int source, int destination,
                 [[maybe_unused]] Tensor output)
    : OperatorObj(OpType::Send, TensorVec{input},
                  TensorVec{output ? output : nullptr}),
      source(source), destination(destination) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> SendObj::inferShape(const TensorVec &inputs) {
    return {{inputs[0]->getDims()}};
}

vector<DataType> SendObj::inferDataType(const TensorVec &inputs) const {
    return {{inputs[0]->getDType()}};
}

std::string SendObj::toString() const {
    std::ostringstream os;
    os << "Send"
       << "[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "input=" << inputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> SendObj::getWorkloadVector() const {
    vector<int> ret = inputs[0]->getDims();
    ret.emplace(ret.begin(), type.underlying());
    ret.emplace_back(source);
    ret.emplace_back(destination);
    return ret;
}

vector<int> SendObj::getOpAttrVector() const {
    vector<int> ret = inputs[0]->getDims();
    ret.emplace(ret.begin(), type.underlying());
    ret.emplace_back(source);
    ret.emplace_back(destination);
    return ret;
}

double SendObj::getComputeTime() const {
    double dataSize = inputs[0]->size() * inputs[0]->getDType().getSize();
    double bandwidthGBps = 10.0;
    double latencyMs = 0.01;
    if (std::abs(source - destination) > 1) {
        latencyMs += std::abs(source - destination) * 0.005;
    }
    double transferTime = (dataSize / (bandwidthGBps * 1e9)) + (latencyMs / 1000.0);
    double setupTime = 0.0001;
    return transferTime + setupTime;
}

double SendObj::getMemoryCost() const {
    double inputSize = inputs[0]->size() * inputs[0]->getDType().getSize();
    double sendBufferSize = inputSize;
    double protocolOverhead = 256.0;
    double outputCost = 0.0;
    if (!outputs.empty() && outputs[0] != nullptr) {
        outputCost = outputs[0]->size() * outputs[0]->getDType().getSize();
    }
    return inputSize + sendBufferSize + protocolOverhead + outputCost;
}

double SendObj::getParallelism() const {
    double baseParallelism = 1.0;
    double sizeBasedParallelism = 1.0;
    double dataSize = inputs[0]->size() * inputs[0]->getDType().getSize();
    if (dataSize > 1024 * 1024 * 10) {
        sizeBasedParallelism = std::min(dataSize / (1024.0 * 1024.0 * 10), 4.0);
    }
    double multiChannelFactor = 1.0;
    double totalParallelism = baseParallelism * sizeBasedParallelism * multiChannelFactor;
    const double MAX_COMM_PARALLEL_UNITS = 8.0;
    return std::min(totalParallelism, MAX_COMM_PARALLEL_UNITS);
}

} // namespace infini