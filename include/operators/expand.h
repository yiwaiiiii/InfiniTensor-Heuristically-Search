#pragma once
#include "core/operator.h"

namespace infini {
/**
 *  @brief Broadcast the input tensor following the given shape and the
 * broadcast rule.
 *
 */
class ExpandObj : public OperatorObj {
    Shape dims;

  public:
    /**
     * @brief Construct a new Expand object.
     * @param graph The computation graph that this operator belongs to.
     * @param input The input tensor.
     * @param output The output tensor.
     * @param dims The shape you want to expand to, following the broadcast
     * rule.
     */
    ExpandObj(GraphObj *graph, Tensor input, Tensor output, Shape dims);
    OP_CLONE(ExpandObj);
    optional<vector<Shape>> inferShape(const TensorVec &inputs) override;

    std::string toString() const override;
    int numInputs() const override { return 1; }
    int numOutputs() const override { return 1; }
    Shape getShape() const { return dims; }

    double getComputeTime() const override;
    double getMemoryCost() const override;
    double getParallelism() const override;
    
  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};

} // namespace infini
