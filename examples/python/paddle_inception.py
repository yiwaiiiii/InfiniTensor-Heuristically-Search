import paddle
import paddle.vision.transforms as T
from paddle.vision.datasets import Cifar10
from pyinfinitensor.onnx import OnnxStub, backend
import onnx
import itertools
import time
import gc
import numpy as np
import os
import traceback

class PerformanceTracker:
    def __init__(self):
        self.start_time = None
        self.stage_times = {}
        self.active_stages = set()
        
    def start_stage(self, stage_name):
        """开始计时一个阶段"""
        self.start_time = time.time()
        self.active_stages.add(stage_name)
        return self.start_time
        
    def end_stage(self, stage_name):
        """结束一个阶段的计时"""
        if self.start_time is None:
            return 0
        
        elapsed = time.time() - self.start_time
        self.stage_times[stage_name] = elapsed
            
        if stage_name in self.active_stages:
            self.active_stages.remove(stage_name)
            
        self.start_time = None
        return elapsed
    
    def end_all_active_stages(self):
        """结束所有活跃的阶段"""
        for stage in list(self.active_stages):
            self.end_stage(stage)
    
    def report(self):
        """打印性能报告"""
        # 确保所有阶段都已结束
        self.end_all_active_stages()
        
        print("\n" + "="*50)
        print("性能分析报告")
        print("="*50)
        print("\n时间分析:")
        for stage, elapsed in self.stage_times.items():
            print(f"{stage}: {elapsed:.4f} 秒")
            
        print("="*50)

def run_test(epochs, batch_size):
    print(f"\n{'-'*20} 测试配置: epochs={epochs}, batch_size={batch_size} {'-'*20}")
    
    # 创建性能跟踪器
    tracker = PerformanceTracker()
    
    try:
        tracker.start_stage("总体执行")
        
        # 确保输出目录存在
        os.makedirs('onnx.save', exist_ok=True)
        
        paddle.device.set_device("gpu")

        transform = T.Compose(
            [
                T.Resize(224),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                    to_rgb=True,
                ),
            ]
        )
        
        tracker.start_stage("数据集加载")
        # 下载数据集并初始化 DataSet
        train_dataset = paddle.vision.datasets.Cifar10(mode='train', transform=transform)
        test_dataset = paddle.vision.datasets.Cifar10(mode='test', transform=transform)
        tracker.end_stage("数据集加载")

        tracker.start_stage("模型创建")
        # 模型组网并初始化网络
        inception = paddle.vision.models.InceptionV3(num_classes=10)
        model = paddle.Model(inception)

        # 模型训练的配置准备，准备损失函数，优化器和评价指标
        model.prepare(paddle.optimizer.Adam(parameters=model.parameters()), 
                    paddle.nn.CrossEntropyLoss(),
                    paddle.metric.Accuracy())
        tracker.end_stage("模型创建")

        # 模型训练
        tracker.start_stage("模型训练")
        model.fit(train_dataset, epochs=epochs, batch_size=batch_size, verbose=1)
        tracker.end_stage("模型训练")
        
        # 模型评估
        tracker.start_stage("模型评估")
        model.evaluate(test_dataset, batch_size=batch_size, verbose=1)
        tracker.end_stage("模型评估")

        # 导出到ONNX
        tracker.start_stage("ONNX导出")
        save_path = f'onnx.save/inception_e{epochs}_b{batch_size}'
        x_spec = paddle.static.InputSpec([1, 3, 224, 224], 'float32', 'x')
        paddle.onnx.export(inception, save_path, input_spec=[x_spec], opset_version=11)
        tracker.end_stage("ONNX导出")

        # 强制垃圾回收
        gc.collect()
        
        # 加载onnx模型并放到Infinitensor中
        tracker.start_stage("InfiniTensor加载")
        model_path = save_path + ".onnx"
        onnx_model = onnx.load(model_path)
        gofusion_model = OnnxStub(onnx_model, backend.cuda_runtime())
        model = gofusion_model
        model.init()
        tracker.end_stage("InfiniTensor加载")

        # 启动推理
        tracker.start_stage("InfiniTensor推理")
        cifar10_test = Cifar10(
            mode="test",
            transform=transform,
            backend="cv2",
        )
        inference_batch_size = 1  # 推理时始终使用批次为1
        total_size = 0
        total_acc = 0.0
        
        # 测量单次推理时间
        inference_times = []
        
        # 热身推理
        tracker.start_stage("热身推理")
        warm_up_samples = 10
        for data in itertools.islice(iter(cifar10_test), warm_up_samples):
            images, _ = data
            next(model.inputs.items().__iter__())[1].copyin_float(images.reshape([3*224*224]).tolist())
            model.run()
            _ = next(model.outputs.items().__iter__())[1].copyout_float()
        tracker.end_stage("热身推理")
        
        # 正式推理测试
        for data in itertools.islice(iter(cifar10_test), 100):  # 只测试100个样本
            images, labels = data
            
            # 记录单次推理时间
            inference_start = time.time()
            
            next(model.inputs.items().__iter__())[1].copyin_float(images.reshape([3*224*224]).tolist())
            model.run()
            outputs = next(model.outputs.items().__iter__())[1].copyout_float()
            
            inference_end = time.time()
            inference_times.append(inference_end - inference_start)
            
            outputs = paddle.to_tensor(outputs)
            outputs = paddle.reshape(outputs, (1, 10))
            labels = paddle.to_tensor(labels)
            labels = paddle.reshape(labels, (1,1))
            acc = paddle.metric.accuracy(outputs, labels)
            total_acc += acc
            total_size += inference_batch_size
        
        # 计算推理性能指标
        avg_inference_time = np.mean(inference_times)
        inference_std = np.std(inference_times)
        inference_p95 = np.percentile(inference_times, 95)
        inference_p99 = np.percentile(inference_times, 99)
        throughput = 1.0 / avg_inference_time
        test_accuracy = total_acc.numpy() / total_size
        
        print(f"\n推理性能统计 (epochs={epochs}, batch_size={batch_size}):")
        print(f"平均单次推理时间: {avg_inference_time*1000:.2f} ms")
        print(f"推理时间标准差: {inference_std*1000:.2f} ms")
        print(f"P95推理时间: {inference_p95*1000:.2f} ms")
        print(f"P99推理时间: {inference_p99*1000:.2f} ms")
        print(f"测试准确率: {test_accuracy:.4f}")
        
        tracker.end_stage("InfiniTensor推理")
        tracker.end_stage("总体执行")
        
        # 记录结果到汇总表中
        return {
            "epochs": epochs,
            "batch_size": batch_size,
            "avg_inference_time": avg_inference_time*1000,
            "inference_std": inference_std*1000,
            "p95_latency": inference_p95*1000,
            "p99_latency": inference_p99*1000,
            "accuracy": test_accuracy
        }
        
    except Exception as e:
        print(f"执行过程中发生错误: {e}")
        traceback.print_exc()
        return None
    finally:
        # 无论是否发生异常，都确保性能报告被生成
        tracker.report()


def run_all_tests():
    # 定义要测试的配置
    test_configs = [
        {"epochs": 3, "batch_size": 32},
        {"epochs": 3, "batch_size": 64},
        {"epochs": 5, "batch_size": 32},
        {"epochs": 5, "batch_size": 64},
    ]
    
    # 存储所有结果
    results = []
    
    # 运行所有测试
    for config in test_configs:
        result = run_test(config["epochs"], config["batch_size"])
        if result:
            results.append(result)
    
    # 打印汇总结果
    print("\n" + "="*80)
    print("测试结果汇总")
    print("="*80)
    print(f"{'Epochs':^6} | {'Batch Size':^10} | {'平均推理时间 (ms)':^18} | {'标准差 (ms)':^14} | {'P95 延迟 (ms)':^14} | {'P99 延迟 (ms)':^14} | {'准确率':^8}")
    print("-"*80)
    
    for result in results:
        print(f"{result['epochs']:^6} | {result['batch_size']:^10} | {result['avg_inference_time']:^18.2f} | {result['inference_std']:^14.2f} | {result['p95_latency']:^14.2f} | {result['p99_latency']:^14.2f} | {result['accuracy']:^8.4f}")
    
    print("="*80)


if __name__ == "__main__":
    run_all_tests()
