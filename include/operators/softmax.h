#pragma once
#include "core/operator.h"

namespace infini {
class SoftmaxObj : public OperatorObj {
    int axis;

  public:
    SoftmaxObj(GraphObj *graph, Tensor input, Tensor output, int axis);

    OP_CLONE(SoftmaxObj);

    optional<vector<Shape>> inferShape(const TensorVec &inputs) override {
        return {{inputs[0]->getDims()}};
    };

    std::string toString() const override;
    int numInputs() const override { return 1; }
    int numOutputs() const override { return 1; }

    int getAxis() const { return axis; }

    double getComputeTime() const override;
    double getMemoryCost() const override;
    double getParallelism() const override;

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};
} // namespace infini
