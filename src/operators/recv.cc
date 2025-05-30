#include "operators/recv.h"

namespace infini {
RecvObj::RecvObj(GraphObj *graph, Tensor output, int source, int destination,
                 Shape dims, int outputType, [[maybe_unused]] Tensor input)
    : OperatorObj(OpType::Recv, input ? TensorVec{input} : TensorVec{},
                  TensorVec{output}),
      source(source), destination(destination), dims(std::move(dims)),
      outputType(outputType) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> RecvObj::inferShape(const TensorVec &inputs) {
    return {{dims}};
}

vector<DataType> RecvObj::inferDataType(const TensorVec &inputs) const {
    return {{DataType(outputType)}};
}

DataType RecvObj::getDType() const { return getOutput(0)->getDType(); }

std::string RecvObj::toString() const {
    std::ostringstream os;
    os << "Recv"
       << "[" << getGuid() << "]";
    os << "(";
    os << vecToString(dims) << ",";
    os << "output=" << outputs[0]->getGuid() << ",";
    os << "dims=" << vecToString(dims) << ")";
    return os.str();
}

vector<int> RecvObj::getWorkloadVector() const {
    vector<int> ret = dims;
    ret.insert(ret.end(), dims.begin(), dims.end());
    ret.emplace(ret.begin(), type.underlying());
    ret.emplace_back(source);
    ret.emplace_back(destination);
    return ret;
}

vector<int> RecvObj::getOpAttrVector() const {
    vector<int> ret = dims;
    ret.emplace(ret.begin(), type.underlying());
    ret.emplace_back(source);
    ret.emplace_back(destination);
    return ret;
}

double RecvObj::getComputeTime() const {
    double dataSize = outputs[0]->size() * outputs[0]->getDType().getSize();
    double bandwidthGBps = 10.0;
    double latencyMs = 0.01;
    if (std::abs(source - destination) > 1) {
        latencyMs += std::abs(source - destination) * 0.005;
    }
    double transferTime = (dataSize / (bandwidthGBps * 1e9)) + (latencyMs / 1000.0);
    double setupTime = 0.0001;
    return transferTime + setupTime;
}

double RecvObj::getMemoryCost() const {
    double outputSize = outputs[0]->size() * outputs[0]->getDType().getSize();
    double tempBufferSize = 0.0;
    if (outputs[0]->size() > 1024 * 1024) {
        tempBufferSize = std::min(outputSize * 0.1, 1024.0 * 1024.0);
    }
    double protocolOverhead = 256.0;
    return outputSize + tempBufferSize + protocolOverhead;
}

double RecvObj::getParallelism() const {
    double baseParallelism = 1.0;
    double sizeBasedParallelism = 1.0;
    double dataSize = outputs[0]->size() * outputs[0]->getDType().getSize();
    if (dataSize > 1024 * 1024 * 10) {
        sizeBasedParallelism = std::min(dataSize / (1024.0 * 1024.0 * 10), 4.0);
    }
    double multiStreamFactor = 1.0;
    double totalParallelism = baseParallelism * sizeBasedParallelism * multiStreamFactor;
    const double MAX_COMM_PARALLEL_UNITS = 8.0;
    return std::min(totalParallelism, MAX_COMM_PARALLEL_UNITS);
}
} // namespace infini