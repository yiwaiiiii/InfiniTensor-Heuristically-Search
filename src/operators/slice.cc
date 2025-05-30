#include "operators/slice.h"

namespace infini {
SliceObj::SliceObj(GraphObj *graph, Tensor input, Tensor output,
                   const vector<int> &starts, const vector<int> &ends,
                   const optional<vector<int>> &_axes,
                   const optional<vector<int>> &_steps)
    : OperatorObj(OpType::Slice, {input}, {output}) {
    auto shape = input->getDims();
    map<size_t, size_t> axes;
    vector<int> steps;
    {
        auto size = starts.size();
        IT_ASSERT(size == ends.size());

        if (_axes) {
            IT_ASSERT(size == _axes->size());
            IT_ASSERT(size == std::set(_axes->begin(), _axes->end()).size());

            for (size_t i = 0; i < size; ++i) {
                auto index = _axes->at(i);
                if (index < 0)
                    index += shape.size();
                axes[index] = i;
            }
        } else
            for (size_t i = 0; i < size; ++i)
                axes[i] = i;

        if (_steps) {
            IT_ASSERT(size == _steps->size());
            IT_ASSERT(std::find(_steps->begin(), _steps->end(), 0) == _steps->end());
            steps = *_steps;
        } else {
            steps.reserve(size);
            for (size_t i = 0; i < size; ++i)
                steps.push_back(1);
        }
    }

    auto size = shape.size();
    this->axes.reserve(size);
    for (size_t i = 0; i < size; ++i) {
        auto len = shape[i];
        if (auto _i = axes.find(i); _i != axes.end()) {
            auto __i = _i->second;
            auto start = starts[__i];
            auto end = ends[__i];
            if (start > len)
                start = len;
            if (end > len)
                end = len;
            this->axes.push_back({start >= 0 ? start : start + len,
                                  end >= 0 ? end : end + len, steps[__i]});
        } else {
            this->axes.push_back({0, len, 1});
        }
    }
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> SliceObj::inferShape(const TensorVec &inputs) {
    Shape ans;
    ans.reserve(axes.size());
    for (const auto &range : axes) {
        auto step = std::abs(range.step);
        ans.push_back((range.end - range.start + step - 1) / step);
    }
    return {{ans}};
}

std::string SliceObj::toString() const {
    std::ostringstream os;
    os << "Slice[" << getGuid() << "][";
    for (const auto &range : axes) {
        os << range.start << ':' << range.step << ':' << range.end << ", ";
    }
    os << "]("
       << "input=" << inputs[0]->getGuid() << ", "
       << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> SliceObj::getWorkloadVector() const {
    auto ans = getOpAttrVector();
    {
        auto i = inputs[0]->getDims();
        ans.insert(ans.end(), i.begin(), i.end());
    }
    if (!outputs.empty()) {
        auto o = outputs[0]->getDims();
        ans.insert(ans.end(), o.begin(), o.end());
    }
    return ans;
}

vector<int> SliceObj::getOpAttrVector() const {
    vector<int> ans{type.underlying()};
    for (const auto &range : axes) {
        ans.push_back(range.start);
        ans.push_back(range.end);
        ans.push_back(range.step);
    }
    return ans;
}

double SliceObj::getComputeTime() const {
    double outputSize = outputs[0]->size();
    double discontinuityFactor = 1.0;
    int nonUnitStepCount = 0;
    int innerAxisNonUnitStep = 0;
    
    for (size_t i = 0; i < axes.size(); ++i) {
        if (std::abs(axes[i].step) != 1) {
            nonUnitStepCount++;
            if (i >= axes.size() / 2) {
                innerAxisNonUnitStep++;
            }
        }
    }
    
    discontinuityFactor += nonUnitStepCount * 0.15;
    discontinuityFactor += innerAxisNonUnitStep * 0.25;
    double opsPerElement = 2.0;
    double totalOps = outputSize * opsPerElement * discontinuityFactor;
    return totalOps / 5e9;
}

double SliceObj::getMemoryCost() const {
    double outputSize = outputs[0]->size();
    double memoryCostFactor = 1.0;
    
    for (size_t i = 0; i < axes.size(); ++i) {
        if (std::abs(axes[i].step) != 1) {
            if (i >= axes.size() / 2) {
                memoryCostFactor += 0.4;
            } else {
                memoryCostFactor += 0.2;
            }
        }
    }
    
    double inputAccessCost = outputSize * memoryCostFactor;
    double outputAccessCost = outputSize;
    return inputAccessCost + outputAccessCost;
}

double SliceObj::getParallelism() const {
    double parallelism = outputs[0]->size();
    double parallelEfficiency = 1.0;
    
    for (const auto &range : axes) {
        if (std::abs(range.step) > 1) {
            parallelEfficiency *= (1.0 - 0.05 * std::min(std::abs(range.step) - 1, 5));
        }
    }
    
    const double MAX_PARALLEL_UNITS = 1024.0;
    double effectiveParallelism = parallelism * parallelEfficiency;
    return std::min(effectiveParallelism, MAX_PARALLEL_UNITS);
}

} // namespace infini