#include "operators/unary.h"

namespace infini {
UnaryObj::UnaryObj(OpType type, GraphObj *graph, Tensor input, Tensor output)
    : OperatorObj(type, {input}, {output}) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> UnaryObj::inferShape(const TensorVec &inputs) {
    const auto A = inputs[0];
    return {{A->getDims()}};
}

std::string UnaryObj::toString() const {
    std::ostringstream os;
    os << type.toString() << "[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> UnaryObj::getWorkloadVector() const {
    vector<int> ret{type.underlying()};
    const Shape shape = outputs[0]->getDims();
    ret.insert(ret.end(), shape.begin(), shape.end());
    return ret;
}

vector<int> UnaryObj::getOpAttrVector() const { return {type.underlying()}; }

double UnaryObj::getComputeTime() const {
    double inputSize = inputs[0]->size();
    double operationCost = 1.0;
    
    if (type == OpType::Neg || type == OpType::Abs || 
        type == OpType::Sign || type == OpType::Not) {
        operationCost = 0.5;
    }
    else if (type == OpType::Relu || type == OpType::Gelu || 
             type == OpType::Silu || type == OpType::Ceil || 
             type == OpType::Floor || type == OpType::Round || 
             type == OpType::Sigmoid || type == OpType::HardSigmoid || 
             type == OpType::Erf || type == OpType::HardSwish) {
        operationCost = 1.0;
    }
    else if (type == OpType::Exp || type == OpType::Log || 
             type == OpType::Sqrt || type == OpType::Tanh || 
             type == OpType::Sin || type == OpType::Cos || 
             type == OpType::Tan || type == OpType::Asin || 
             type == OpType::Acos || type == OpType::Atan || 
             type == OpType::Sinh || type == OpType::Cosh || 
             type == OpType::Asinh || type == OpType::Acosh || 
             type == OpType::Atanh) {
        operationCost = 3.0;
    }
    
    return (inputSize * operationCost) / 2e9;
}

double UnaryObj::getMemoryCost() const {
    double inputCost = inputs[0]->size();
    double outputCost = outputs[0]->size();
    double memoryEfficiencyFactor = 1.0;
    
    if (type == OpType::Exp || type == OpType::Log || 
        type == OpType::Tanh || type == OpType::Erf) {
        memoryEfficiencyFactor = 1.1;
    }
    
    return (inputCost + outputCost) * memoryEfficiencyFactor;
}

double UnaryObj::getParallelism() const {
    double inputSize = inputs[0]->size();
    double parallelEfficiency = 1.0;
    
    if (type == OpType::Neg || type == OpType::Abs || 
        type == OpType::Relu || type == OpType::Sign) {
        parallelEfficiency = 1.0;
    }
    else if (type == OpType::Exp || type == OpType::Log || 
             type == OpType::Sin || type == OpType::Cos) {
        parallelEfficiency = 0.8;
    }
    
    const double MAX_PARALLEL_UNITS = 1024.0;
    return std::min(inputSize * parallelEfficiency, MAX_PARALLEL_UNITS);
}

ClipObj::ClipObj(GraphObj *graph, Tensor input, Tensor output,
                 std::optional<float> min, std::optional<float> max)
    : OperatorObj(OpType::Clip, {input}, {output}), minValue(min),
      maxValue(max) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> ClipObj::inferShape(const TensorVec &inputs) {
    const auto A = inputs[0];
    return {{A->getDims()}};
}

std::string ClipObj::toString() const {
    std::ostringstream os;
    os << type.toString() << "[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> ClipObj::getWorkloadVector() const {
    vector<int> ret{type.underlying()};
    const Shape shape = outputs[0]->getDims();
    ret.insert(ret.end(), shape.begin(), shape.end());
    return ret;
}

vector<int> ClipObj::getOpAttrVector() const { return {type.underlying()}; }

double ClipObj::getComputeTime() const {
    double inputSize = inputs[0]->size();
    return inputSize * 1.2 / 5e9;
}

double ClipObj::getMemoryCost() const {
    return inputs[0]->size() + outputs[0]->size();
}

double ClipObj::getParallelism() const {
    const double MAX_PARALLEL_UNITS = 1024.0;
    return std::min(inputs[0]->size() * 0.8, MAX_PARALLEL_UNITS);
}

HardtanhObj::HardtanhObj(GraphObj *graph, Tensor input, Tensor output,
                         float min, float max)
    : OperatorObj(OpType::Hardtanh, {input}, {output}), minValue(min),
      maxValue(max) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> HardtanhObj::inferShape(const TensorVec &inputs) {
    const auto A = inputs[0];
    return {{A->getDims()}};
}

std::string HardtanhObj::toString() const {
    std::ostringstream os;
    os << type.toString() << "[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> HardtanhObj::getWorkloadVector() const {
    vector<int> ret{type.underlying()};
    const Shape shape = outputs[0]->getDims();
    ret.insert(ret.end(), shape.begin(), shape.end());
    return ret;
}

vector<int> HardtanhObj::getOpAttrVector() const { return {type.underlying()}; }

double HardtanhObj::getComputeTime() const {
    return inputs[0]->size() * 0.7 / 2e9;
}

double HardtanhObj::getMemoryCost() const {
    return inputs[0]->size() * 2;
}

double HardtanhObj::getParallelism() const {
    return std::min(outputs[0]->size(), 1024.0);
}

FillObj::FillObj(GraphObj *graph, Tensor input, Tensor output, float value)
    : OperatorObj(OpType::Fill, {input}, {output}), setValue(value) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> FillObj::inferShape(const TensorVec &inputs) {
    const auto A = inputs[0];
    return {{A->getDims()}};
}

std::string FillObj::toString() const {
    std::ostringstream os;
    os << type.toString() << "[" << getGuid() << "]";
    os << "(";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> FillObj::getWorkloadVector() const {
    vector<int> ret{type.underlying()};
    const Shape shape = outputs[0]->getDims();
    ret.insert(ret.end(), shape.begin(), shape.end());
    return ret;
}

vector<int> FillObj::getOpAttrVector() const { return {type.underlying()}; }

double FillObj::getComputeTime() const {
    return outputs[0]->size() * 0.2 / 2e9;
}

double FillObj::getMemoryCost() const {
    return outputs[0]->size();
}

double FillObj::getParallelism() const {
    return std::min(outputs[0]->size(), 1024.0);
}

L2LossObj::L2LossObj(GraphObj *graph, Tensor input, Tensor output)
    : OperatorObj(OpType::L2Loss, {input}, {output}) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> L2LossObj::inferShape(const TensorVec &inputs) {
    Shape temp = {1};
    return {{temp}};
}

std::string L2LossObj::toString() const {
    std::ostringstream os;
    os << type.toString() << "[" << getGuid() << "]";
    os << "(";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> L2LossObj::getWorkloadVector() const {
    vector<int> ret{type.underlying()};
    const Shape shape = outputs[0]->getDims();
    ret.insert(ret.end(), shape.begin(), shape.end());
    return ret;
}

vector<int> L2LossObj::getOpAttrVector() const { return {type.underlying()}; }

double L2LossObj::getComputeTime() const {
    double inputSize = inputs[0]->size();
    return (inputSize * 2 + 1) / 2e9;
}

double L2LossObj::getMemoryCost() const {
    return inputs[0]->size() + outputs[0]->size();
}

double L2LossObj::getParallelism() const {
    return std::min(inputs[0]->size() / 2, 256.0);
}

CastObj::CastObj(GraphObj *graph, Tensor input, Tensor output, CastType type)
    : OperatorObj(OpType::Cast, {input}, {output}), castType(type) {
    IT_ASSERT(checkValid(graph));
}

vector<DataType> CastObj::inferDataType(const TensorVec &inputs) const {
    auto input_dataType = inputs[0]->getDType();
    auto output_dataType = getOutputDataType();
    for (const auto &tensor : inputs)
        IT_ASSERT(input_dataType == tensor->getDType());
    return vector(numOutputs(), output_dataType);
}

optional<vector<Shape>> CastObj::inferShape(const TensorVec &inputs) {
    const auto A = inputs[0];
    return {{A->getDims()}};
}

std::string CastObj::toString() const {
    std::ostringstream os;
    os << type.toString() << "[" << getGuid() << "]";
    os << "(";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> CastObj::getWorkloadVector() const {
    vector<int> ret{type.underlying()};
    const Shape shape = outputs[0]->getDims();
    ret.insert(ret.end(), shape.begin(), shape.end());
    return ret;
}

vector<int> CastObj::getOpAttrVector() const { return {type.underlying()}; }

DataType CastObj::getOutputDataType() const {
    switch (castType) {
    case CastType::Float2Float16: return DataType::Float16;
    case CastType::Float2Int64: return DataType::Int64;
    case CastType::Float2Int32: return DataType::Int32;
    case CastType::Float2Int16: return DataType::Int16;
    case CastType::Float2Int8: return DataType::Int8;
    case CastType::Int322Float: return DataType::Float32;
    case CastType::Int322Int8: return DataType::Int8;
    case CastType::Int322Int16: return DataType::Int16;
    case CastType::Int162Float: return DataType::Float32;
    case CastType::Int162Int32: return DataType::Int32;
    case CastType::Int82Float: return DataType::Float32;
    case CastType::Int82Int16: return DataType::Int16;
    case CastType::Int82Int32: return DataType::Int32;
    case CastType::Uint82Float: return DataType::Float32;
    case CastType::Uint82Int32: return DataType::Int32;
    case CastType::Uint82Int64: return DataType::Int64;
    case CastType::Int322Int64: return DataType::Int64;
    case CastType::Int642Int32: return DataType::Int32;
    case CastType::Int642Uint32: return DataType::UInt32;
    case CastType::Int642Float: return DataType::Float32;
    case CastType::Uint322Int64: return DataType::Int64;
    case CastType::Float162Float: return DataType::Float32;
    case CastType::BFloat162Float: return DataType::Float32;
    case CastType::Float2BFloat16: return DataType::BFloat16;
    case CastType::Float2Float: return DataType::Float32;
    default: IT_TODO_HALT();
    }
}

double CastObj::getComputeTime() const {
    return inputs[0]->size() * 0.5 / 2e9;
}

double CastObj::getMemoryCost() const {
    return inputs[0]->size() + outputs[0]->size();
}

double CastObj::getParallelism() const {
    return std::min(outputs[0]->size(), 1024.0);
}

ShapeObj::ShapeObj(GraphObj *graph, Tensor input, Tensor output)
    : OperatorObj(OpType::Shape, {input}, {output}) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> ShapeObj::inferShape(const TensorVec &inputs) {
    return {{{static_cast<int>(inputs[0]->getRank())}}};
}

std::string ShapeObj::toString() const {
    std::ostringstream os;
    os << type.toString() << "[" << getGuid() << "]("
       << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

double ShapeObj::getComputeTime() const {
    return 1e-6;
}

double ShapeObj::getMemoryCost() const {
    return inputs[0]->getRank();
}

double ShapeObj::getParallelism() const {
    return 1.0;
}

PReluObj::PReluObj(GraphObj *graph, Tensor input, Tensor alpha, Tensor output)
    : OperatorObj(OpType::PRelu, {input, alpha}, {output}) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> PReluObj::inferShape(const TensorVec &inputs) {
    const auto A = inputs[0];
    return {{A->getDims()}};
}

std::string PReluObj::toString() const {
    std::ostringstream os;
    os << type.toString() << "[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> PReluObj::getWorkloadVector() const {
    vector<int> ret{type.underlying()};
    const Shape shape = outputs[0]->getDims();
    ret.insert(ret.end(), shape.begin(), shape.end());
    return ret;
}

vector<int> PReluObj::getOpAttrVector() const { return {type.underlying()}; }

double PReluObj::getComputeTime() const {
    return inputs[0]->size() * 1.2 / 2e9;
}

double PReluObj::getMemoryCost() const {
    return inputs[0]->size() * 2 + inputs[1]->size();
}

double PReluObj::getParallelism() const {
    return std::min(outputs[0]->size(), 1024.0);
}

LeakyReluObj::LeakyReluObj(GraphObj *graph, Tensor input, Tensor output,
                           float alpha)
    : OperatorObj(OpType::LeakyRelu, {input}, {output}), alphaValue(alpha) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> LeakyReluObj::inferShape(const TensorVec &inputs) {
    const auto A = inputs[0];
    return {{A->getDims()}};
}

std::string LeakyReluObj::toString() const {
    std::ostringstream os;
    os << type.toString() << "[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ",";
    os << "alpha=" << alphaValue << ")";
    return os.str();
}

vector<int> LeakyReluObj::getWorkloadVector() const {
    vector<int> ret{type.underlying()};
    const Shape shape = outputs[0]->getDims();
    ret.insert(ret.end(), shape.begin(), shape.end());
    return ret;
}

vector<int> LeakyReluObj::getOpAttrVector() const {
    return {type.underlying()};
}

double LeakyReluObj::getComputeTime() const {
    return inputs[0]->size() / 2e9;
}

double LeakyReluObj::getMemoryCost() const {
    return inputs[0]->size() * 2;
}

double LeakyReluObj::getParallelism() const {
    return std::min(outputs[0]->size(), 1024.0);
}

LogObj::LogObj(GraphObj *graph, Tensor input, Tensor output, LogType type)
    : OperatorObj(OpType::Log, {input}, {output}), logType(type) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> LogObj::inferShape(const TensorVec &inputs) {
    const auto A = inputs[0];
    return {{A->getDims()}};
}

std::string LogObj::toString() const {
    std::ostringstream os;
    os << type.toString() << "[" << getGuid() << "]";
    os << "(";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> LogObj::getWorkloadVector() const {
    vector<int> ret{type.underlying()};
    const Shape shape = outputs[0]->getDims();
    ret.insert(ret.end(), shape.begin(), shape.end());
    return ret;
}

vector<int> LogObj::getOpAttrVector() const { return {type.underlying()}; }

double LogObj::getComputeTime() const {
    return inputs[0]->size() * 5 / 2e9;
}

double LogObj::getMemoryCost() const {
    return inputs[0]->size() * 2;
}

double LogObj::getParallelism() const {
    return std::min(outputs[0]->size(), 1024.0);
}

EluObj::EluObj(GraphObj *graph, Tensor input, Tensor output, float alpha)
    : OperatorObj(OpType::Elu, {input}, {output}), alpha(alpha) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> EluObj::inferShape(const TensorVec &inputs) {
    return {{inputs[0]->getDims()}};
}

std::string EluObj::toString() const {
    std::ostringstream os;
    os << "Elu[" << getGuid() << "]";
    os << "(";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "alpha=" << alpha << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> EluObj::getWorkloadVector() const {
    vector<int> ret = getOutput()->getDims();
    ret.emplace(ret.begin(), type.underlying());
    return ret;
}

vector<int> EluObj::getOpAttrVector() const {
    return {type.underlying(), static_cast<int>(alpha)};
}

double EluObj::getComputeTime() const {
    return inputs[0]->size() * 3 / 2e9;
}

double EluObj::getMemoryCost() const {
    return inputs[0]->size() * 2;
}

double EluObj::getParallelism() const {
    return std::min(outputs[0]->size(), 1024.0);
}

}; // namespace infini