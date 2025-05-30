#include "operators/attention_kvcache.h"
#include "utils/operator_utils.h"

namespace infini {
AttentionKVCacheObj::AttentionKVCacheObj(GraphObj *graph, Tensor input_k_cache,
                                         Tensor input_v_cache, Tensor input_q,
                                         Tensor input_k, Tensor input_v,
                                         Tensor position_id,
                                         Tensor output_matmul)
    : OperatorObj(OpType::AttentionKVCache,
                  TensorVec{input_k_cache, input_v_cache, input_q, input_k,
                            input_v, position_id},
                  {output_matmul}) {
    int rank = inputs[0]->getRank();
    IT_ASSERT(rank == 4);
    dim = 2;
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>>
AttentionKVCacheObj::inferShape(const TensorVec &inputs) {
    IT_ASSERT(inputs.size() == 6);
    Shape dims = inputs[0]->getDims();
    ShapeElem n = dims.at(dim);
    dims[dim] = n + 1;
    return {{inputs[2]->getDims()}};
}

std::string AttentionKVCacheObj::toString() const {
    std::ostringstream os;
    os << "AttentionKVCache[" << getGuid() << "]";
    os << "(";
    for (auto input : inputs)
        os << vecToString(input->getDims()) << ",";
    os << "dim=" << dim << ",";
    os << "input=";
    for (auto input : inputs)
        os << input->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> AttentionKVCacheObj::getWorkloadVector() const {
    vector<int> ret = getOutputs()[0]->getDims();
    ret.emplace(ret.begin(), (int)inputs.size());
    ret.emplace(ret.begin(), dim);
    ret.emplace(ret.begin(), type.underlying());
    return ret;
}

vector<int> AttentionKVCacheObj::getOpAttrVector() const {
    return {type.underlying(), dim};
}

double AttentionKVCacheObj::getComputeTime() const {
    const auto &q_dims = inputs[2]->getDims();      
    const auto &k_cache_dims = inputs[0]->getDims(); 
    
    int64_t batch_size = q_dims[0];
    int64_t seq_len = k_cache_dims[2]; 
    int64_t new_seq_len = seq_len + 1; 
    int64_t num_heads, head_dim;
    
    if (q_dims.size() >= 4) {
        num_heads = q_dims[1];
        head_dim = q_dims[3];
    } else {
        num_heads = 16;
        head_dim = 64;
    }
    double qk_cost = batch_size * num_heads * seq_len * new_seq_len * head_dim;
    double softmax_cost = batch_size * num_heads * seq_len * new_seq_len * 3.0; 
    double attn_v_cost = batch_size * num_heads * seq_len * head_dim * new_seq_len;
    double cache_update_cost = 2 * batch_size * num_heads * head_dim; 
    return (qk_cost + softmax_cost + attn_v_cost + cache_update_cost) / 1e9;
}

double AttentionKVCacheObj::getMemoryCost() const {
    double input_cost = 0;
    for (const auto &input : inputs) {
        input_cost += input->size();
    }
    double output_cost = outputs[0]->size();
    const auto &q_dims = inputs[2]->getDims();      
    const auto &k_cache_dims = inputs[0]->getDims(); 
    int64_t batch_size = q_dims[0];
    int64_t seq_len = k_cache_dims[2];
    int64_t new_seq_len = seq_len + 1;
    int64_t num_heads = q_dims.size() >= 2 ? q_dims[1] : 16;
    double attn_matrix_size = batch_size * num_heads * seq_len * new_seq_len;
    return input_cost + output_cost + attn_matrix_size;
}

double AttentionKVCacheObj::getParallelism() const {
    const auto &q_dims = inputs[2]->getDims(); 
    int64_t batch_size = q_dims[0];
    int64_t num_heads = q_dims.size() >= 2 ? q_dims[1] : 16;
    double parallelism = batch_size * num_heads;
    double utilization = 0.9; 
    return parallelism * utilization;
}

} // namespace infini
