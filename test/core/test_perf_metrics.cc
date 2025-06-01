#include <gtest/gtest.h>
#include "core/graph.h"
#include "core/runtime.h"
#include "operators/matmul.h"
#include "operators/conv.h"
#include "operators/unary.h"
#include "operators/element_wise.h"
#include "operators/pooling.h"

namespace infini {

void printMetrics(const std::string &title, const PerfMetrics &metrics) {
    std::cout << title << std::endl;
    std::cout << "  计算时间: " << metrics.computeTime << " 秒" << std::endl;
    std::cout << "  内存开销: " << metrics.memoryCost << " 元素" << std::endl;
    std::cout << "  并行度: " << metrics.parallelism << std::endl;
}

TEST(PerfMetrics, BasicTest) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph graph = make_ref<GraphObj>(runtime);

    // MatMul
    auto A = graph->addTensor({64, 128}, DataType::Float32);
    auto B = graph->addTensor({128, 64}, DataType::Float32);
    auto matmul = graph->addOp<MatmulObj>(A, B, nullptr);

    PerfMetrics metrics = runtime->getPerfMetrics(graph, false);

    EXPECT_GT(metrics.computeTime, 0.0);
    EXPECT_GT(metrics.memoryCost, 0.0);
    EXPECT_GT(metrics.parallelism, 0.0);

    printMetrics("MatMul 性能指标:", metrics);
}

TEST(PerfMetrics, ConvTest) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph graph = make_ref<GraphObj>(runtime);

    // Conv
    auto input = graph->addTensor({1, 3, 32, 32}, DataType::Float32);
    auto weight = graph->addTensor({16, 3, 3, 3}, DataType::Float32);
    auto conv = graph->addOp<ConvObj>(input, weight, nullptr, 0, 0, 1, 1, 1, 1, nullptr);

    PerfMetrics metrics = runtime->getPerfMetrics(graph, false);

    EXPECT_GT(metrics.computeTime, 0.0);
    EXPECT_GT(metrics.memoryCost, 0.0);
    EXPECT_GT(metrics.parallelism, 0.0);

    printMetrics("Conv 性能指标:", metrics);
}

TEST(PerfMetrics, AddTest) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph graph = make_ref<GraphObj>(runtime);

    // 使用AddObj而不是ElementWiseObj
    auto A = graph->addTensor({64, 128}, DataType::Float32);
    auto B = graph->addTensor({64, 128}, DataType::Float32);
    // AddObj构造函数参数: graph, input0, input1, output
    auto add = graph->addOp<AddObj>(A, B, nullptr);

    PerfMetrics metrics = runtime->getPerfMetrics(graph, false);

    EXPECT_GT(metrics.computeTime, 0.0);
    EXPECT_GT(metrics.memoryCost, 0.0);
    EXPECT_GT(metrics.parallelism, 0.0);

    printMetrics("Add 性能指标:", metrics);
}

TEST(PerfMetrics, ReluTest) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph graph = make_ref<GraphObj>(runtime);

    // 使用具体的ReluObj
    auto A = graph->addTensor({64, 128}, DataType::Float32);
    // 修改: 用nullptr替代预创建的输出张量
    auto relu = graph->addOp<ReluObj>(A, nullptr);

    PerfMetrics metrics = runtime->getPerfMetrics(graph, false);

    EXPECT_GT(metrics.computeTime, 0.0);
    EXPECT_GT(metrics.memoryCost, 0.0);
    EXPECT_GT(metrics.parallelism, 0.0);

    printMetrics("Relu 性能指标:", metrics);
}

TEST(PerfMetrics, PoolTest) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph graph = make_ref<GraphObj>(runtime);

    // Pooling - 直接使用MaxPoolObj简化代码
    auto input = graph->addTensor({1, 3, 32, 32}, DataType::Float32);
    
    // 按照MaxPoolObj构造函数的要求提供参数
    auto pool = graph->addOp<MaxPoolObj>(
        input, nullptr,  // 使用nullptr作为输出张量
        2, 2,           // kh, kw (内核高度和宽度)
        1, 1,           // dh, dw (高度和宽度维度的扩张)
        0, 0,           // ph, pw (高度和宽度维度的填充)
        2, 2,           // sh, sw (高度和宽度维度的步幅)
        0               // ceilMode (是否向上取整)
    );

    PerfMetrics metrics = runtime->getPerfMetrics(graph, false);

    EXPECT_GT(metrics.computeTime, 0.0);
    EXPECT_GT(metrics.memoryCost, 0.0);
    EXPECT_GT(metrics.parallelism, 0.0);

    printMetrics("MaxPool 性能指标:", metrics);
}

TEST(PerfMetrics, FusionTest) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();

    // 1. MatMul + Relu 未融合
    Graph graphUnfused = make_ref<GraphObj>(runtime);
    auto A1 = graphUnfused->addTensor({64, 128}, DataType::Float32);
    auto B1 = graphUnfused->addTensor({128, 64}, DataType::Float32);
    auto matmul1 = graphUnfused->addOp<MatmulObj>(A1, B1, nullptr);
    auto C1 = matmul1->getOutput();
    auto relu1 = graphUnfused->addOp<ReluObj>(C1, nullptr);

    // 2. MatMul(act=Relu) 融合
    Graph graphFused = make_ref<GraphObj>(runtime);
    auto A2 = graphFused->addTensor({64, 128}, DataType::Float32);
    auto B2 = graphFused->addTensor({128, 64}, DataType::Float32);
    auto matmul2 = graphFused->addOp<MatmulObj>(A2, B2, nullptr, false, false, nullptr, ActType::Relu);

    // 3. Conv + Relu 未融合
    Graph graphConvUnfused = make_ref<GraphObj>(runtime);
    auto input1 = graphConvUnfused->addTensor({1, 3, 32, 32}, DataType::Float32);
    auto weight1 = graphConvUnfused->addTensor({16, 3, 3, 3}, DataType::Float32);
    auto conv1 = graphConvUnfused->addOp<ConvObj>(input1, weight1, nullptr, 0, 0, 1, 1, 1, 1, nullptr);
    auto convOut1 = conv1->getOutput();
    auto reluConv1 = graphConvUnfused->addOp<ReluObj>(convOut1, nullptr);

    // 4. Conv(act=Relu) 融合
    Graph graphConvFused = make_ref<GraphObj>(runtime);
    auto input2 = graphConvFused->addTensor({1, 3, 32, 32}, DataType::Float32);
    auto weight2 = graphConvFused->addTensor({16, 3, 3, 3}, DataType::Float32);
    auto conv2 = graphConvFused->addOp<ConvObj>(input2, weight2, nullptr, 0, 0, 1, 1, 1, 1, nullptr, ActType::Relu);

    // 5. Add + Relu 未融合
    Graph graphAddUnfused = make_ref<GraphObj>(runtime);
    auto AA = graphAddUnfused->addTensor({64, 128}, DataType::Float32);
    auto BB = graphAddUnfused->addTensor({64, 128}, DataType::Float32);
    auto add1 = graphAddUnfused->addOp<AddObj>(AA, BB, nullptr);
    auto CC = add1->getOutput();
    auto reluAdd = graphAddUnfused->addOp<ReluObj>(CC, nullptr);

    // 6. Add(act=Relu) 融合
    Graph graphAddFused = make_ref<GraphObj>(runtime);
    auto AA2 = graphAddFused->addTensor({64, 128}, DataType::Float32);
    auto BB2 = graphAddFused->addTensor({64, 128}, DataType::Float32);
    auto add2 = graphAddFused->addOp<AddObj>(AA2, BB2, nullptr);

    // 获取性能指标
    PerfMetrics metricsUnfused = runtime->getPerfMetrics(graphUnfused, false);
    PerfMetrics metricsFused = runtime->getPerfMetrics(graphFused, false);
    PerfMetrics metricsConvUnfused = runtime->getPerfMetrics(graphConvUnfused, false);
    PerfMetrics metricsConvFused = runtime->getPerfMetrics(graphConvFused, false);
    PerfMetrics metricsAddUnfused = runtime->getPerfMetrics(graphAddUnfused, false);
    PerfMetrics metricsAddFused = runtime->getPerfMetrics(graphAddFused, false);

    printMetrics("MatMul+Relu 未融合:", metricsUnfused);
    printMetrics("MatMul+Relu 融合:", metricsFused);
    printMetrics("Conv+Relu 未融合:", metricsConvUnfused);
    printMetrics("Conv+Relu 融合:", metricsConvFused);
    printMetrics("Add+Relu 未融合:", metricsAddUnfused);
    printMetrics("Add+Relu 融合:", metricsAddFused);

    EXPECT_TRUE(metricsFused.computeTime < metricsUnfused.computeTime);
    EXPECT_TRUE(metricsConvFused.computeTime < metricsConvUnfused.computeTime);
    EXPECT_TRUE(metricsAddFused.computeTime < metricsAddUnfused.computeTime);
}

} // namespace infini