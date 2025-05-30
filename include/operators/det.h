#pragma once
#include "core/operator.h"

namespace infini {
class DetObj : public OperatorObj {
  public:
    enum Mode { NormalDet = 0, LogDet };
    DetObj(GraphObj *graph, Tensor input, Tensor output, Mode mode);
    OP_CLONE(DetObj);
    optional<vector<Shape>> inferShape(const TensorVec &inputs) override;

    std::string toString() const override;
    int numInputs() const override { return 1; }
    int numOutputs() const override { return 1; }
    Mode getMode() const { return modeValue; }

    double getComputeTime() const override;
    double getMemoryCost() const override;
    double getParallelism() const override;    
    
  private:
    Mode modeValue;
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};
}; // namespace infini
