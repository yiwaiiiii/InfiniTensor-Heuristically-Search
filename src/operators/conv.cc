#include "operators/conv.h"

namespace infini {

ConvBaseObj::ConvBaseObj(OpType opType, TensorVec inputs, Tensor &output,
                         int ph, int pw, int sh, int sw, int dh, int dw,
                         const Tensor &inputInConvFWD,
                         const Tensor &weightInConvFWD, ActType act)
    : OperatorObj(opType, inputs, {output}), ph(ph), pw(pw), sh(sh), sw(sw),
      dh(dh), dw(dw), padding(PaddingMode::Other), act(act) {}
ConvBaseObj::ConvBaseObj(OpType opType, TensorVec inputs, Tensor &output,
                         PaddingMode mode, int sh, int sw, int dh, int dw,
                         const Tensor &inputInConvFWD,
                         const Tensor &weightInConvFWD, ActType act)
    : OperatorObj(opType, inputs, {output}), ph(-1), pw(-1), sh(sh), sw(sw),
      dh(dh), dw(dw), padding(mode), act(act) {
    IT_ASSERT(mode != PaddingMode::Other);
}

string ConvBaseObj::toString() const {
    std::ostringstream os;
    os << type.toString() << "[" << getGuid() << "]";
    os << "(";
    if (inputs.size() == 2) {
        os << vecToString(inputs[0]->getDims()) << ",";
        os << vecToString(inputs[1]->getDims()) << ",";
    }
    os << "p=[" << ph << "," << pw << "],";
    os << "s=[" << sh << "," << sw << "],";
    os << "d=[" << dh << "," << dw << "],";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "weight=" << inputs[1]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> ConvBaseObj::getWorkloadVector() const {
    return {type.underlying(), n, c, h, w, f, r, s, ph, pw, sh, sw, dh, dw};
}

vector<int> ConvBaseObj::getOpAttrVector() const {
    return {type.underlying(), c, f, r, s, ph, pw, sh, sw, dh, dw};
}

void ConvObj::setAuxilaryAttributes(PaddingMode mode) {
    const Tensor &input = inputs[0];
    const Tensor &weight = inputs[1];
    n = input->getDims()[0], c = input->getDims()[1], h = input->getDims()[2],
    w = input->getDims()[3], f = weight->getDims()[0], r = weight->getDims()[2],
    s = weight->getDims()[3];
    if (mode == PaddingMode::Same) {
        int oh = h / sh;
        int ow = w / sw;
        ph = (h - oh * sh + (r - sh) * dh) / 2;
        pw = (w - ow * sw + (s - sw) * dw) / 2;
    } else if (mode == PaddingMode::Valid) {
        ph = pw = 0;
    }
}

ConvObj::ConvObj(GraphObj *graph, Tensor input, Tensor weight, Tensor output,
                 int ph, int pw, int sh, int sw, int dh, int dw, Tensor bias,
                 ActType act)
    : ConvBaseObj(OpType::Conv, {input, weight}, output, ph, pw, sh, sw, dh, dw,
                  input, weight, act) {
    if (bias)
        IT_TODO_HALT();
    setAuxilaryAttributes(PaddingMode::Other);
    IT_ASSERT(checkValid(graph));
}

ConvObj::ConvObj(GraphObj *graph, Tensor input, Tensor weight, Tensor output,
                 PaddingMode mode, int sh, int sw, int dh, int dw, Tensor bias,
                 ActType act)
    : ConvBaseObj(OpType::Conv, {input, weight}, output, mode, sh, sw, dh, dw,
                  input, weight, act) {
    if (bias)
        IT_TODO_HALT();
    setAuxilaryAttributes(mode);
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> ConvObj::inferShape(const TensorVec &inputs) {
    const auto &input = inputs[0], &weight = inputs[1];
    n = input->getDims()[0];
    c = input->getDims()[1];
    h = input->getDims()[2];
    w = input->getDims()[3];
    f = weight->getDims()[0];
    r = weight->getDims()[2];
    s = weight->getDims()[3];
    int on = n, oc = f;
    int oh = 0, ow = 0;
    IT_ASSERT(input->getDims()[1] % weight->getDims()[1] == 0);
    if (padding == PaddingMode::Other) {
        oh = (h - (r - sh) * dh + ph * 2) / sh;
        ow = (w - (s - sw) * dw + pw * 2) / sw;
    } else if (padding == PaddingMode::Same) {
        oh = h / sh;
        ow = w / sw;
    } else if (padding == PaddingMode::Valid) {
        int ph = 0;
        int pw = 0;
        oh = (h - (r - sh) * dh + ph * 2) / sh;
        ow = (w - (s - sw) * dw + pw * 2) / sw;
    }
    return {{{on, oc, oh, ow}}};
}

void Conv3dObj::setAuxilaryAttributes(PaddingMode mode) {
    const Tensor &input = inputs[0];
    const Tensor &weight = inputs[1];
    n = input->getDims()[0];
    c = input->getDims()[1];
    d = input->getDims()[2];
    h = input->getDims()[3];
    w = input->getDims()[4];
    f = weight->getDims()[0];
    q = weight->getDims()[2];
    r = weight->getDims()[3];
    s = weight->getDims()[4];
    if (mode == PaddingMode::Same) {
        int od = d / sd;
        int oh = h / sh;
        int ow = w / sw;
        pd = (d - od * sd + (q - sd) * dd) / 2;
        ph = (h - oh * sh + (r - sh) * dh) / 2;
        pw = (w - ow * sw + (s - sw) * dw) / 2;
    } else if (mode == PaddingMode::Valid) {
        pd = 0;
        ph = 0;
        pw = 0;
    }
}

Conv3dObj::Conv3dObj(GraphObj *graph, Tensor input, Tensor weight,
                     Tensor output, int pd, int ph, int pw, int sd, int sh,
                     int sw, int dd, int dh, int dw, Tensor bias, ActType act)
    : ConvBaseObj(OpType::Conv3d, {input, weight}, output, ph, pw, sh, sw, dh,
                  dw, input, weight, act),
      pd(pd), sd(sd), dd(dd) {
    if (bias)
        IT_TODO_HALT();
    setAuxilaryAttributes(PaddingMode::Other);
    IT_ASSERT(checkValid(graph));
}

Conv3dObj::Conv3dObj(GraphObj *graph, Tensor input, Tensor weight,
                     Tensor output, PaddingMode mode, int sd, int sh, int sw,
                     int dd, int dh, int dw, Tensor bias, ActType act)
    : ConvBaseObj(OpType::Conv3d, {input, weight}, output, mode, sh, sw, dh, dw,
                  input, weight, act),
      sd(sd), dd(dd) {
    if (bias)
        IT_TODO_HALT();
    setAuxilaryAttributes(mode);
    IT_ASSERT(checkValid(graph));
}

string Conv3dObj::toString() const {
    std::ostringstream os;
    os << type.toString() << "[" << getGuid() << "]";
    os << "(";
    if (inputs.size() == 2) {
        os << vecToString(inputs[0]->getDims()) << ",";
        os << vecToString(inputs[1]->getDims()) << ",";
    }
    os << "p=[" << pd << "," << ph << "," << pw << "],";
    os << "s=[" << sd << "," << sh << "," << sw << "],";
    os << "d=[" << dd << "," << dh << "," << dw << "],";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "weight=" << inputs[1]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

optional<vector<Shape>> Conv3dObj::inferShape(const TensorVec &inputs) {
    const auto &input = inputs[0];
    const auto &weight = inputs[1];
    n = input->getDims()[0];
    c = input->getDims()[1];
    d = input->getDims()[2];
    h = input->getDims()[3];
    w = input->getDims()[4];
    f = weight->getDims()[0];
    q = weight->getDims()[2];
    r = weight->getDims()[3];
    s = weight->getDims()[4];
    int on = n;
    int oc = f;
    int od = 0;
    int oh = 0;
    int ow = 0;
    IT_ASSERT(input->getDims()[1] % weight->getDims()[1] == 0);
    if (padding == PaddingMode::Other) {
        od = (d - (q - sd) * dd + pd * 2) / sd;
        oh = (h - (r - sh) * dh + ph * 2) / sh;
        ow = (w - (s - sw) * dw + pw * 2) / sw;
    } else if (padding == PaddingMode::Same) {
        od = d / sd;
        oh = h / sh;
        ow = w / sw;
    } else if (padding == PaddingMode::Valid) {
        int pd = 0;
        int ph = 0;
        int pw = 0;
        od = (d - (q - sd) * dd + pd * 2) / sd;
        oh = (h - (r - sh) * dh + ph * 2) / sh;
        ow = (w - (s - sw) * dw + pw * 2) / sw;
    }
    return {{{on, oc, od, oh, ow}}};
}

ConvTransposed2dObj::ConvTransposed2dObj(GraphObj *graph, Tensor input,
                                         Tensor weight, Tensor output, int ph,
                                         int pw, int sh, int sw, int dh, int dw,
                                         int oph, int opw, int group,
                                         Tensor bias, ActType act)
    : ConvBaseObj(OpType::ConvTranspose, {input, weight}, output, ph, pw, sh,
                  sw, dh, dw, output, weight, act),
      oph(oph), opw(opw), group(group) {
    if (bias)
        IT_TODO_HALT();
    setAuxilaryAttributes(PaddingMode::Other);
    IT_ASSERT(checkValid(graph));
}

ConvTransposed2dObj::ConvTransposed2dObj(GraphObj *graph, Tensor input,
                                         Tensor weight, Tensor output,
                                         PaddingMode mode, int sh, int sw,
                                         int dh, int dw, int oph, int opw,
                                         int group, Tensor bias, ActType act)
    : ConvBaseObj(OpType::ConvTranspose, {input, weight}, output, mode, sh, sw,
                  dh, dw, output, weight, act),
      oph(oph), opw(opw), group(group) {
    if (bias)
        IT_TODO_HALT();
    setAuxilaryAttributes(mode);
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>>
ConvTransposed2dObj::inferShape(const TensorVec &inputs) {
    const Tensor &input = inputs[0], &weight = inputs[1];
    n = input->getDims()[0];
    f = input->getDims()[1];
    h = input->getDims()[2];
    w = input->getDims()[3];
    c = weight->getDims()[1];
    r = weight->getDims()[2];
    s = weight->getDims()[3];
    IT_ASSERT(f == weight->getDims()[0]);

    int on = n, oc = c * group;
    int oh = 0, ow = 0;
    oh = (h - 1) * sh - 2 * ph + dh * (r - 1) + oph + 1;
    ow = (w - 1) * sw - 2 * pw + dw * (s - 1) + opw + 1;
    return {{{on, oc, oh, ow}}};
}

void ConvTransposed2dObj::setAuxilaryAttributes(PaddingMode mode) {
    const Tensor &input = inputs[0];
    const Tensor &weight = inputs[1];
    n = input->getDims()[0], f = input->getDims()[1], h = input->getDims()[2],
    w = input->getDims()[3], c = weight->getDims()[1], r = weight->getDims()[2],
    s = weight->getDims()[3];
    if (mode == PaddingMode::Same) {
        int oh = h / sh;
        int ow = w / sw;
        ph = (h - oh * sh + (r - sh) * dh) / 2;
        pw = (w - ow * sw + (s - sw) * dw) / 2;
    } else if (mode == PaddingMode::Valid) {
        ph = pw = 0;
    }
}

void ConvBackwardFilterObj::setAuxilaryAttributes(PaddingMode mode) {
    const Tensor &inputX = inputs[0];
    const Tensor &diffY = inputs[1];
    n = inputX->getDims()[0], c = inputX->getDims()[1],
    h = inputX->getDims()[2], w = inputX->getDims()[3], f = diffY->getDims()[0],
    r = diffY->getDims()[2], s = diffY->getDims()[3];
    if (mode == PaddingMode::Same) {
        int oh = h / sh;
        int ow = w / sw;
        ph = (h - oh * sh + (r - sh) * dh) / 2;
        pw = (w - ow * sw + (s - sw) * dw) / 2;
    } else if (mode == PaddingMode::Valid) {
        ph = pw = 0;
    }
}

ConvBackwardFilterObj::ConvBackwardFilterObj(GraphObj *graph, Tensor inputX,
                                             Tensor diffY, Tensor diffW, int ph,
                                             int pw, int sh, int sw, int dh,
                                             int dw, Tensor bias, ActType act)
    : ConvBaseObj(OpType::Conv, {inputX, diffY}, diffW, ph, pw, sh, sw, dh, dw,
                  inputX, diffY),
      act(act) {
    if (bias)
        IT_TODO_HALT();
    setAuxilaryAttributes(PaddingMode::Other);
    IT_ASSERT(checkValid(graph));
}

ConvBackwardFilterObj::ConvBackwardFilterObj(GraphObj *graph, Tensor inputX,
                                             Tensor diffY, Tensor diffW,
                                             PaddingMode mode, int sh, int sw,
                                             int dh, int dw, Tensor bias,
                                             ActType act)
    : ConvBaseObj(OpType::Conv, {inputX, diffY}, diffW, mode, sh, sw, dh, dw,
                  inputX, diffY),
      act(act) {
    if (bias)
        IT_TODO_HALT();
    setAuxilaryAttributes(mode);
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>>
ConvBackwardFilterObj::inferShape(const TensorVec &inputs) {
    const auto &inputX = inputs[0], &diffY = inputs[1];
    n = inputX->getDims()[0];
    c = inputX->getDims()[1];
    h = inputX->getDims()[2];
    w = inputX->getDims()[3];
    f = diffY->getDims()[0];
    r = diffY->getDims()[2];
    s = diffY->getDims()[3];
    int on = n, oc = f;
    int oh = 0, ow = 0;
    IT_ASSERT(inputX->getDims()[1] % diffY->getDims()[1] == 0);
    if (padding == PaddingMode::Other) {
        oh = (h - (r - sh) * dh + ph * 2) / sh;
        ow = (w - (s - sw) * dw + pw * 2) / sw;
    } else if (padding == PaddingMode::Same) {
        oh = h / sh;
        ow = w / sw;
    } else if (padding == PaddingMode::Valid) {
        int ph = 0;
        int pw = 0;
        oh = (h - (r - sh) * dh + ph * 2) / sh;
        ow = (w - (s - sw) * dw + pw * 2) / sw;
    }
    return {{{on, oc, oh, ow}}};
}

ConvTransposed2dNHWCObj::ConvTransposed2dNHWCObj(GraphObj *graph, Tensor input,
                                                 Tensor weight, Tensor output,
                                                 int ph, int pw, int sh, int sw,
                                                 int dh, int dw, int oph,
                                                 int opw, int group,
                                                 Tensor bias, ActType act)
    : ConvBaseObj(OpType::ConvTransNHWC, {input, weight}, output, ph, pw, sh,
                  sw, dh, dw, output, weight, act),
      oph(oph), opw(opw), group(group) {
    if (bias)
        IT_TODO_HALT();
    setAuxilaryAttributes(PaddingMode::Other);
    IT_ASSERT(checkValid(graph));
}

ConvTransposed2dNHWCObj::ConvTransposed2dNHWCObj(GraphObj *graph, Tensor input,
                                                 Tensor weight, Tensor output,
                                                 PaddingMode mode, int sh,
                                                 int sw, int dh, int dw,
                                                 int oph, int opw, int group,
                                                 Tensor bias, ActType act)
    : ConvBaseObj(OpType::ConvTranspose, {input, weight}, output, mode, sh, sw,
                  dh, dw, output, weight, act),
      oph(oph), opw(opw), group(group) {
    if (bias)
        IT_TODO_HALT();
    setAuxilaryAttributes(mode);
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>>
ConvTransposed2dNHWCObj::inferShape(const TensorVec &inputs) {
    const Tensor &input = inputs[0], &weight = inputs[1];
    n = input->getDims()[0];
    f = input->getDims()[3];
    h = input->getDims()[1];
    w = input->getDims()[2];
    c = weight->getDims()[3];
    r = weight->getDims()[1];
    s = weight->getDims()[2];
    IT_ASSERT(f == weight->getDims()[0]);

    int on = n, oc = c * group;
    int oh = 0, ow = 0;
    oh = (h - 1) * sh - 2 * ph + dh * (r - 1) + oph + 1;
    ow = (w - 1) * sw - 2 * pw + dw * (s - 1) + opw + 1;
    return {{{on, oh, ow, oc}}};
}

void ConvTransposed2dNHWCObj::setAuxilaryAttributes(PaddingMode mode) {
    const Tensor &input = inputs[0];
    const Tensor &weight = inputs[1];
    n = input->getDims()[0], f = input->getDims()[3], h = input->getDims()[1],
    w = input->getDims()[2], c = weight->getDims()[3], r = weight->getDims()[1],
    s = weight->getDims()[2];
    if (mode == PaddingMode::Same) {
        int oh = h / sh;
        int ow = w / sw;
        ph = (h - oh * sh + (r - sh) * dh) / 2;
        pw = (w - ow * sw + (s - sw) * dw) / 2;
    } else if (mode == PaddingMode::Valid) {
        ph = pw = 0;
    }
}

double ConvBaseObj::getComputeTime() const {
    const auto &inputDims = inputs[0]->getDims();
    const auto &weightDims = inputs[1]->getDims();
    const auto &outputDims = outputs[0]->getDims();
    
    int64_t n, c, f, r, s;
    
    n = this->n;
    c = this->c;
    f = this->f;
    r = this->r;
    s = this->s;
    
    int64_t oh = outputDims.size() >= 4 ? outputDims[2] : 1;
    int64_t ow = outputDims.size() >= 4 ? outputDims[3] : 1;
    
    double operations = static_cast<double>(n) * f * oh * ow * r * s * c;
    
    return operations / 1.0e9;
}

double ConvBaseObj::getMemoryCost() const {
    double cost = 0.0;
    
    cost += inputs[0]->size();
    
    cost += inputs[1]->size();
    
    cost += outputs[0]->size();
    
    return cost;
}

double ConvBaseObj::getParallelism() const {
    const auto &outputDims = outputs[0]->getDims();
    
    int64_t n = outputDims[0];
    int64_t f = outputDims[1];
    
    double spatialParallelism = 1.0;
    if (outputDims.size() >= 4) {
        spatialParallelism = outputDims[2] * outputDims[3];
    } else if (outputDims.size() >= 5) {
        spatialParallelism = outputDims[2] * outputDims[3] * outputDims[4];
    }
    
    double parallelism = n * f * spatialParallelism;
    
    const double MAX_PARALLEL_UNITS = 10240.0;
    return std::min(parallelism, MAX_PARALLEL_UNITS);
}

double ConvObj::getComputeTime() const {
    return ConvBaseObj::getComputeTime();
}

double ConvObj::getMemoryCost() const {
    return ConvBaseObj::getMemoryCost();
}

double ConvObj::getParallelism() const {
    return ConvBaseObj::getParallelism();
}

double Conv3dObj::getComputeTime() const {
    const auto &inputDims = inputs[0]->getDims();
    const auto &weightDims = inputs[1]->getDims();
    const auto &outputDims = outputs[0]->getDims();
    
    int64_t n = inputDims[0];
    int64_t c = inputDims[1];
    
    int64_t f = weightDims[0];
    int64_t q = weightDims[2];
    int64_t r = weightDims[3];
    int64_t s = weightDims[4];
    
    int64_t od = outputDims[2];
    int64_t oh = outputDims[3];
    int64_t ow = outputDims[4];
    
    double operations = static_cast<double>(n) * f * od * oh * ow * q * r * s * c;
    
    return operations / 1.0e9;
}

double Conv3dObj::getMemoryCost() const {
    return ConvBaseObj::getMemoryCost() * 1.2;
}

double Conv3dObj::getParallelism() const {
    const auto &outputDims = outputs[0]->getDims();
    
    int64_t n = outputDims[0];
    int64_t f = outputDims[1];
    int64_t d = outputDims[2];
    int64_t h = outputDims[3];
    int64_t w = outputDims[4];
    
    double parallelism = n * f * d * h * w;
    
    const double MAX_PARALLEL_UNITS = 10240.0;
    return std::min(parallelism, MAX_PARALLEL_UNITS);
}

double ConvTransposed2dObj::getComputeTime() const {
    const auto &inputDims = inputs[0]->getDims();
    const auto &weightDims = inputs[1]->getDims();
    const auto &outputDims = outputs[0]->getDims();
    
    int64_t n = inputDims[0];
    int64_t f = inputDims[1];
    
    int64_t r = weightDims[2];
    int64_t s = weightDims[3];
    
    int64_t oh = outputDims[2];
    int64_t ow = outputDims[3];
    int64_t oc = outputDims[1];
    
    double operations = static_cast<double>(n) * oc * oh * ow * r * s * (f / group);
    
    return operations / 0.9e9;
}

double ConvBackwardFilterObj::getComputeTime() const {
    return ConvBaseObj::getComputeTime() * 2.5;
}

double ConvBackwardFilterObj::getMemoryCost() const {
    return ConvBaseObj::getMemoryCost() * 2.0;
}

double ConvBackwardFilterObj::getParallelism() const {
    return ConvBaseObj::getParallelism() * 0.9;
}

double ConvTransposed2dObj::getMemoryCost() const {
    return ConvBaseObj::getMemoryCost() * 1.3;
}

double ConvTransposed2dObj::getParallelism() const {
    return ConvBaseObj::getParallelism() * 0.95;
}

double ConvTransposed2dNHWCObj::getComputeTime() const {
    return ConvBaseObj::getComputeTime() * 1.05 * 1.2;
}

double ConvTransposed2dNHWCObj::getMemoryCost() const {
    return ConvBaseObj::getMemoryCost() * 1.3 * 0.95;
}

double ConvTransposed2dNHWCObj::getParallelism() const {
    return ConvBaseObj::getParallelism() * 0.95;
}

} // namespace infini