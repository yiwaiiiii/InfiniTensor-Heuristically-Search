#include "operators/where.h"
#include "utils/operator_utils.h"

namespace infini {

WhereObj::WhereObj(GraphObj *graph, Tensor inputX, Tensor inputY,
                   Tensor condition, Tensor output)
    : OperatorObj(OpType::Where, TensorVec{inputX, inputY, condition},
                  {output}) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> WhereObj::inferShape(const TensorVec &inputs) {
    auto shapeX = inputs[0]->getDims();
    auto shapeY = inputs[1]->getDims();
    auto shapeCon = inputs[2]->getDims();
    auto retXY = infer_broadcast(shapeX, shapeY);
    auto ret = infer_broadcast(retXY, shapeCon);
    return {{ret}};
}

std::string WhereObj::toString() const {
    std::ostringstream os;
    os << "Where[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[2]->getDims()) << ",";
    os << "inputX=" << inputs[0]->getGuid() << ",";
    os << "inputY=" << inputs[1]->getGuid() << ",";
    os << "condition=" << inputs[2]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> WhereObj::getWorkloadVector() const {
    vector<int> ret = getOutput()->getDims();
    ret.emplace(ret.begin(), type.underlying());
    return ret;
}

vector<int> WhereObj::getOpAttrVector() const { return {type.underlying()}; }

double WhereObj::getComputeTime() const {
    double outputSize = outputs[0]->size();
    bool needsBroadcasting = false;
    double broadcastingFactor = 1.0;
    
    const auto &shapeX = inputs[0]->getDims();
    const auto &shapeY = inputs[1]->getDims();
    const auto &shapeCond = inputs[2]->getDims();
    const auto &outputShape = outputs[0]->getDims();
    
    if (shapeX != outputShape) {
        needsBroadcasting = true;
        broadcastingFactor += 0.05;
    }
    
    if (shapeY != outputShape) {
        needsBroadcasting = true;
        broadcastingFactor += 0.05;
    }
    
    if (shapeCond != outputShape) {
        needsBroadcasting = true;
        broadcastingFactor += 0.1;
    }
    
    double opsPerElement = 2.0;
    
    if (needsBroadcasting) {
        opsPerElement *= broadcastingFactor;
    }
    
    return (outputSize * opsPerElement) / 3e9;
}

double WhereObj::getMemoryCost() const {
    double conditionCost = inputs[2]->size();
    double inputXCost = inputs[0]->size();
    double inputYCost = inputs[1]->size();
    double outputCost = outputs[0]->size();
    
    const auto &outputShape = outputs[0]->getDims();
    
    double xAccessFactor = 1.0;
    if (inputs[0]->getDims() != outputShape) {
        xAccessFactor = 1.2;
    }
    
    double yAccessFactor = 1.0;
    if (inputs[1]->getDims() != outputShape) {
        yAccessFactor = 1.2;
    }
    
    double condAccessFactor = 1.0;
    if (inputs[2]->getDims() != outputShape) {
        condAccessFactor = 1.2;
    }
    
    double condTypeRatio = 0.25;
    
    return (conditionCost * condAccessFactor * condTypeRatio) + 
           (inputXCost * xAccessFactor) + 
           (inputYCost * yAccessFactor) + 
           outputCost;
}

double WhereObj::getParallelism() const {
    double parallelism = outputs[0]->size();
    const auto &outputShape = outputs[0]->getDims();
    
    if (inputs[0]->getDims() != outputShape || 
        inputs[1]->getDims() != outputShape || 
        inputs[2]->getDims() != outputShape) {
        parallelism *= 0.9;
    }
    
    return parallelism;
}

} // namespace infini