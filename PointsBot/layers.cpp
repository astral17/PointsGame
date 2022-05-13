#include "layers.h"

#include <algorithm>
#include <fstream>
#include <cassert>
#include <stdexcept>
#include <numeric>
#include <initializer_list>

using namespace dnnl;

memory::dims GetStridesForDims(const memory::dims& dims)
{
    memory::dims result(dims.size());
    for (memory::dim i = result.size() - 1, prod = 1; i >= 0; i--)
    {
        result[i] = prod;
        prod *= dims[i];
    }
    return result;
}

size_t GetMemoryCount(const dnnl::memory& mem)
{
    auto dims = mem.get_desc().dims();
    return std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>());
}

size_t GetMemoryByteSize(const dnnl::memory& mem)
{
    return mem.get_desc().get_size();
}

void* MemoryMapData(const dnnl::memory& mem)
{
    dnnl::engine eng = mem.get_engine();

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    if (eng.get_kind() == dnnl::engine::kind::gpu)
        return mem.map_data();
#endif

    if (eng.get_kind() == dnnl::engine::kind::cpu)
        return mem.get_data_handle();

    assert(!"not expected");
}

void MemoryUnmapData(const dnnl::memory& mem, void* handle)
{
    {
        dnnl::engine eng = mem.get_engine();

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
        if (eng.get_kind() == dnnl::engine::kind::gpu)
        {
            mem.unmap_data(handle);
            return;
        }
#endif

        if (eng.get_kind() == dnnl::engine::kind::cpu)
            return;

        assert(!"not expected");
    }
}

void operator>>(const dnnl::memory& mem, void* handle)
{
    dnnl::engine eng = mem.get_engine();
    size_t size = GetMemoryByteSize(mem);
    if (!size)
        return;
    if (!handle)
        throw std::runtime_error("handle is nullptr.");

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    if (eng.get_kind() == dnnl::engine::kind::gpu) {
        void* mapped_ptr = mem.map_data();
        if (mapped_ptr)
            std::memcpy(handle, mapped_ptr, size);
        mem.unmap_data(mapped_ptr);
        return;
    }
#endif

    if (eng.get_kind() == dnnl::engine::kind::cpu) {
        void* src = mem.get_data_handle();
        if (!src)
            throw std::runtime_error("get_data_handle returned nullptr.");
        std::memcpy(handle, src, size);
        return;
    }

    assert(!"not expected");
}

void operator<<(const dnnl::memory& mem, void* handle)
{
    dnnl::engine eng = mem.get_engine();
    size_t size = GetMemoryByteSize(mem);

    if (!handle)
        throw std::runtime_error("handle is nullptr.");

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    if (eng.get_kind() == dnnl::engine::kind::gpu) {
        void* mapped_ptr = mem.map_data();
        if (mapped_ptr) std::memcpy(mapped_ptr, handle, size);
        mem.unmap_data(mapped_ptr);
        return;
    }
#endif

    if (eng.get_kind() == dnnl::engine::kind::cpu) {
        void* dst = mem.get_data_handle();
        if (!dst) throw std::runtime_error("get_data_handle returned nullptr.");
        std::memcpy(dst, handle, size);
        return;
    }

    assert(!"not expected");
}


dnnl::memory Layer::dst_grad_2()
{
    if (!dst_grad_2_)
        dst_grad_2_ = dnnl::memory(output_desc(), net().engine);
    return dst_grad_2_;
}

Layer::Layer(NNetwork& net) : net_(&net)
{
    net.layers.push_back(this);
}

Loss::Loss(NNetwork& net) : net_(&net)
{
}

NNetwork::NNetwork(const dnnl::engine& eng, dnnl::prop_kind kind, dnnl::memory::data_type data_type) : engine(eng), kind(kind), data_type(data_type)
{
}

void NNetwork::Build(const std::vector<Layer*>& output, float learn_rate, const std::vector<Loss*>& loss)
{
    this->learn_rate = learn_rate;
    if (kind == prop_kind::forward_training)
    {
        for (Layer* layer : output)
            layer->IncDependencies();
    }
    for (auto layer : layers)
        layer->Init();
    auto reverse_from = bwd.size();
    for (int i = 0; i < output.size(); i++)
    {
        Layer* layer = output[i];
        auto desc = layer->output_desc();
        outputs.push_back(layer->output());
        answers.push_back(memory(desc, engine));
        if (kind == prop_kind::forward_training)
        {
            Loss* cur_loss = loss[i];
            layer->DecDependencies();
            grads.push_back(layer->dst_grad());
            cur_loss->Init(outputs.back(), answers.back(), grads.back());
        }
    }
    std::reverse(bwd.begin() + reverse_from, bwd.end());
    std::reverse(bwd.begin(), bwd.end());
}

void NNetwork::RandomWeights(std::mt19937& mt, float min, float max)
{
    //std::normal_distribution<float> dist(0, 1);
    std::uniform_real_distribution<float> dist(min, max);
    //std::uniform_real_distribution<float> dist(-1e-5, 1e-5);
    for (memory& mem : weights)
    {
        std::vector<float> buffer(GetMemoryCount(mem));
        for (auto& x : buffer)
            x = dist(mt);
        mem << buffer;
    }
}

void NNetwork::SaveWeights(const std::string& file)
{
    std::ofstream fout(file, std::ios::binary);
    for (memory& mem : weights)
    {
        std::string s(GetMemoryByteSize(mem), 0);
        mem >> (void*)s.c_str();
        fout.write(s.c_str(), s.length());
    }
}

void NNetwork::LoadWeights(const std::string& file)
{
    std::ifstream fin(file, std::ios::binary);
    for (memory& mem : weights)
    {
        std::string s(GetMemoryByteSize(mem), 0);
        fin.read((char*)s.c_str(), s.length());
        mem << (void*)s.c_str();
    }
}

void NNetwork::Forward(const dnnl::stream& s)
{
    for (auto& cur : fwd)
    {
        cur.first.execute(s, cur.second);
        //for (auto m : cur.second)
        //{
        //    std::vector<float> tmp;
        //    m.second >> tmp;
        //    for (int j = 0; j < tmp.size(); j++)
        //        if (isnan(tmp[j]) || isinf(tmp[j]) || abs(tmp[j]) > 1e4)
        //        {
        //            exit(-777);
        //        }
        //    //float max_value = -123;
        //    //for (int j = 0; j < tmp.size(); j++)
        //    //    max_value = std::max(max_value, std::abs(tmp[j]));
        //    //if (max_value > 10000)
        //    //{
        //    //    max_value += 0;
        //    //}

        //}
    }
}

void NNetwork::Backward(const dnnl::stream& s)
{
    for (auto& cur : bwd)
    {
        cur.first.execute(s, cur.second);
        //if (cur.second.count(DNNL_ARG_DIFF_WEIGHTS))
        //{
        //for (auto m : cur.second)
        //{
        //    std::vector<float> tmp;
        //    m.second >> tmp;
        //    //float max_value = -123;
        //    for (int j = 0; j < tmp.size(); j++)
        //        if (isnan(tmp[j]) || isinf(tmp[j]) || abs(tmp[j]) > 1e4)
        //        {
        //            exit(-666);
        //        }
        //        //max_value = std::max(max_value, std::abs(tmp[j]));
        //    //max_value += 0;
        //}
        //}
    }
}

//float NNetwork::GetLoss(const dnnl::stream& s)
//{
//    return 123;
//}

InputLayer::InputLayer(NNetwork& net, const dnnl::memory::dims& adims) : InputLayer(net, {adims, net.data_type, GetStridesForDims(adims)})
{
}

InputLayer::InputLayer(NNetwork& net, const dnnl::memory::desc& input_desc) : Layer(net)
{
    net.input = output({ input_desc, net.engine });
}

InputLayer::InputLayer(NNetwork& net, const dnnl::memory::dims& adims, dnnl::memory::data_type adata_type, dnnl::memory::format_tag aformat_tag)
    : InputLayer(net, { adims, adata_type, aformat_tag }) {}

void InputLayer::Init()
{
}

void InputLayer::SetInput(void* handle)
{
    output() << handle;
}

DenseLayer::DenseLayer(Layer& input, int output_size, bool use_bias) : Layer(input.net()), input(input), output_size(output_size), use_bias(use_bias)
{
    input.IncDependencies();
}

void DenseLayer::Init()
{
    NNetwork& net = this->net();
    input.DecDependencies();
    auto src_md = input.output_desc();
    auto w_dims = src_md.dims();
    w_dims[0] = output_size;

    auto weight_md = memory::desc(w_dims, net.data_type, GetStridesForDims(w_dims));
    auto dst_md = memory::desc({ src_md.data.dims[0], output_size }, net.data_type, memory::format_tag::ab);

    weights = memory(weight_md, net.engine);
    output({ dst_md, net.engine });

    net.weights.push_back(weights);

    memory::desc bias_md;
    if (use_bias)
    {
        bias_md = memory::desc({ output_size }, net.data_type, memory::format_tag::a);
        bias = memory(bias_md, net.engine);
        net.weights.push_back(bias);
    }

    auto product_pd = use_bias
        ? inner_product_forward::primitive_desc({ net.kind, src_md, weight_md, bias_md, dst_md }, net.engine)
        : inner_product_forward::primitive_desc({ net.kind, src_md, weight_md, dst_md }, net.engine);
    auto fwd = inner_product_forward(product_pd);

    net.fwd.push_back({ fwd,
        {
            {DNNL_ARG_SRC, input.output()},
            {DNNL_ARG_WEIGHTS, weights},
            {DNNL_ARG_BIAS, bias},
            {DNNL_ARG_DST, output()},
        } });
    // Backward
    if (net.kind == dnnl::prop_kind::forward_training)
    {
        dst_grad({ dst_md, net.engine });

        auto reverse_from = net.bwd.size();
        if (input.dst_grad())
        {
            auto bwd_data = inner_product_backward_data({ {src_md, weight_md, dst_md}, net.engine, product_pd });
            net.bwd.push_back({ bwd_data,
            {
                {DNNL_ARG_DIFF_SRC, !input.dependency_count() ? input.dst_grad() : input.dst_grad_2()},
                {DNNL_ARG_WEIGHTS, weights},
                {DNNL_ARG_DIFF_DST, dst_grad()},
            } });

            if (input.dependency_count()) // grad saved to buffer, so should add to main memory
            {
                auto desc = input.output_desc();
                auto adder = binary({ {algorithm::binary_add, desc, desc, desc}, net.engine });
                net.bwd.push_back({ adder,
                {
                    {DNNL_ARG_SRC_0, input.dst_grad()},
                    {DNNL_ARG_SRC_1, input.dst_grad_2()},
                    {DNNL_ARG_DST, input.dst_grad()},
                } });
            }
        }

        auto d_weights = memory(weight_md, net.engine);
        auto d_bias = memory(bias_md, net.engine);
        auto bwd_weights = use_bias
            ? inner_product_backward_weights({ {src_md, weight_md, bias_md, dst_md}, net.engine, product_pd })
            : inner_product_backward_weights({ {src_md, weight_md, dst_md}, net.engine, product_pd });
        net.bwd.push_back({ bwd_weights,
            {
                {DNNL_ARG_SRC, input.output()},
                {DNNL_ARG_DIFF_WEIGHTS, d_weights},
                {DNNL_ARG_DIFF_BIAS, d_bias},
                {DNNL_ARG_DIFF_DST, dst_grad()},
            } });
        // TODO: Optimizer control
        auto bwd_weights_update = sum({ { 1, -net.learn_rate }, {weight_md, weight_md}, net.engine });
        net.bwd.push_back({ bwd_weights_update,
            {
                {DNNL_ARG_MULTIPLE_SRC + 0, weights},
                {DNNL_ARG_MULTIPLE_SRC + 1, d_weights},
                {DNNL_ARG_DST, weights},
            } });
        if (use_bias)
        {
            auto bwd_bias_update = sum({ { 1, -net.learn_rate }, {bias_md, bias_md}, net.engine });
            net.bwd.push_back({ bwd_bias_update,
                {
                    {DNNL_ARG_MULTIPLE_SRC + 0, bias},
                    {DNNL_ARG_MULTIPLE_SRC + 1, d_bias},
                    {DNNL_ARG_DST, bias},
                } });
        }
        std::reverse(net.bwd.begin() + reverse_from, net.bwd.end());
    }
}

EltwiseLayer::EltwiseLayer(Layer& input, algorithm algo, float alpha, float beta) : Layer(input.net()), input(input), algo(algo), alpha(alpha), beta(beta)
{
    input.IncDependencies();
}

void EltwiseLayer::Init()
{
    NNetwork& net = this->net();
    input.DecDependencies();
    auto desc = input.output_desc();
    auto elt_pd = eltwise_forward::primitive_desc({ net.kind, algo, desc, alpha, beta }, net.engine);
    auto elt_fwd = eltwise_forward(elt_pd);

    output({ desc, net.engine });

    net.fwd.push_back({ elt_fwd,
        {
            {DNNL_ARG_SRC, input.output()},
            {DNNL_ARG_DST, output()},
        } });
    // Backward
    if (net.kind == dnnl::prop_kind::forward_training)
    {
        dst_grad({ desc, net.engine });
        auto elt_bwd = eltwise_backward({ {algo, desc, desc, alpha, beta}, net.engine, elt_pd });
        auto reverse_from = net.bwd.size();
        if (input.dst_grad())
        {
            net.bwd.push_back({ elt_bwd,
            {
                {DNNL_ARG_DIFF_SRC, !input.dependency_count() ? input.dst_grad() : input.dst_grad_2()},
                {DNNL_ARG_SRC, input.output()},
                {DNNL_ARG_DIFF_DST, dst_grad()},
            } });

            if (input.dependency_count()) // grad saved to buffer, so should add to main memory
            {
                auto adder = binary({ {algorithm::binary_add, desc, desc, desc}, net.engine });
                net.bwd.push_back({ adder,
                {
                    {DNNL_ARG_SRC_0, input.dst_grad()},
                    {DNNL_ARG_SRC_1, input.dst_grad_2()},
                    {DNNL_ARG_DST, input.dst_grad()},
                } });
            }
        }
        std::reverse(net.bwd.begin() + reverse_from, net.bwd.end());
    }
}

ReluLayer::ReluLayer(Layer& input, float alpha) : EltwiseLayer(input, algorithm::eltwise_relu, alpha)
{
}

SigmoidLayer::SigmoidLayer(Layer& input) : EltwiseLayer(input, algorithm::eltwise_logistic)
{
}

TanhLayer::TanhLayer(Layer& input) : EltwiseLayer(input, algorithm::eltwise_tanh)
{
}

LayerAdder::LayerAdder(Layer& input1, Layer& input2) : Layer(input1.net()), input1(input1), input2(input2)
{
    assert(&input1.net() == &input2.net() && "Add layers from different networks impossible");
    input1.IncDependencies();
    input2.IncDependencies();
}

void LayerAdder::Init()
{
    NNetwork& net = this->net();
    input1.DecDependencies();
    input2.DecDependencies();

    auto desc = input1.output_desc();
    assert(desc == input2.output_desc());
    auto adder = binary({ {algorithm::binary_add, desc, desc, desc}, net.engine });

    output({ desc, net.engine });

    net.fwd.push_back({ adder,
    {
        {DNNL_ARG_SRC_0, input1.output()},
        {DNNL_ARG_SRC_1, input2.output()},
        {DNNL_ARG_DST, output()},
    } });

    // Backward
    if (net.kind == dnnl::prop_kind::forward_training)
    {
        dst_grad({ desc, net.engine });

        auto reverse_from = net.bwd.size();
        std::vector<Layer*> layers = { &input1, &input2 };
        for (Layer* layer : layers)
        {
            Layer& input = *layer;
            if (input.dst_grad())
            {
                // TODO: allow broadcasting
                if (input.dependency_count())
                {
                    auto bwd_update = sum({ { 1, 1 }, {desc, desc}, net.engine });
                    net.bwd.push_back({ bwd_update,
                    {
                        {DNNL_ARG_MULTIPLE_SRC + 0, input.dst_grad()},
                        {DNNL_ARG_MULTIPLE_SRC + 1, dst_grad()},
                        {DNNL_ARG_DST, input.dst_grad()},
                    } });
                }
                else
                {
                    // TODO: Replace by reorder?
                    auto bwd_update = sum({ { 1 }, {desc}, net.engine });
                    net.bwd.push_back({ bwd_update,
                    {
                        {DNNL_ARG_MULTIPLE_SRC + 0, dst_grad()},
                        {DNNL_ARG_DST, input.dst_grad()},
                    } });
                }
                //net.bwd.back().second.insert({ DNNL_ARG_MULTIPLE_SRC + 1, input1.dst_grad });
            }
        }
        std::reverse(net.bwd.begin() + reverse_from, net.bwd.end());
    }
}

ReorderLayer::ReorderLayer(Layer& input, dnnl::memory::desc dst_md) : Layer(input.net()), input(input), dst_md(dst_md)
{
    input.IncDependencies();
}

void ReorderLayer::Init()
{
    NNetwork& net = this->net();
    input.DecDependencies();

    auto src_md = input.output_desc();
    output({ dst_md, net.engine });
    auto reorder_fwd = reorder({ input.output(), output() });
    net.fwd.push_back({ reorder_fwd,
    {
        {DNNL_ARG_FROM, input.output()},
        {DNNL_ARG_TO, output()},
    } });
    // Backward
    if (net.kind == dnnl::prop_kind::forward_training)
    {
        dst_grad({ dst_md, net.engine });

        auto reverse_from = net.bwd.size();
        if (input.dst_grad())
        {
            auto reorder_bwd = reorder({ dst_grad(), input.dst_grad() });
            net.bwd.push_back({ reorder_bwd,
            {
                {DNNL_ARG_SRC, dst_grad()},
                {DNNL_ARG_DST, !input.dependency_count() ? input.dst_grad() : input.dst_grad_2()},
            } });

            if (input.dependency_count()) // grad saved to buffer, so should add to main memory
            {
                auto adder = binary({ {algorithm::binary_add, src_md, src_md, src_md}, net.engine });
                net.bwd.push_back({ adder,
                {
                    {DNNL_ARG_SRC_0, input.dst_grad()},
                    {DNNL_ARG_SRC_1, input.dst_grad_2()},
                    {DNNL_ARG_DST, input.dst_grad()},
                } });
            }
        }

        std::reverse(net.bwd.begin() + reverse_from, net.bwd.end());
    }
}

PoolingLayer::PoolingLayer(Layer& input, dnnl::algorithm algo, const dnnl::memory::dims& strides, const dnnl::memory::dims& kernel, const dnnl::memory::dims& padding_l, const dnnl::memory::dims& padding_r)
    : Layer(input.net()), input(input), algo(algo), strides(strides), kernel(kernel), padding_l(padding_l), padding_r(padding_r)
{
    input.IncDependencies();
}

void PoolingLayer::Init()
{
    NNetwork& net = this->net();
    input.DecDependencies();
    //dnnl::pooling_v2_forward::desc()
    auto src_md = input.output_desc();

    auto w_dims = src_md.dims();

    for (memory::dim i = 2; i < w_dims.size(); i++)
    {
        w_dims[i] = (w_dims[i] - kernel[i - 2] + padding_l[i - 2] + padding_r[i - 2]) / strides[i - 2] + 1;
    }

    auto dst_md = memory::desc(w_dims, net.data_type, GetStridesForDims(w_dims));
    auto pool_pd = pooling_forward::primitive_desc({ net.kind, algo, src_md, dst_md, strides, kernel, padding_l, padding_r }, net.engine);
    auto pool_fwd = pooling_forward(pool_pd);

    auto pool_workspace_memory = memory(pool_pd.workspace_desc(), net.engine);
    output({ dst_md, net.engine });

    net.fwd.push_back({ pool_fwd,
    {
        {DNNL_ARG_SRC, input.output()},
        {DNNL_ARG_WORKSPACE, pool_workspace_memory},
        {DNNL_ARG_DST, output()},
    } });
    // Backward
    if (net.kind == dnnl::prop_kind::forward_training)
    {
        dst_grad({ dst_md, net.engine });
        auto pool_bwd = pooling_backward({ { algo, src_md, dst_md, strides, kernel, padding_l, padding_r }, net.engine, pool_pd });
        auto reverse_from = net.bwd.size();
        if (input.dst_grad())
        {
            net.bwd.push_back({ pool_bwd,
            {
                {DNNL_ARG_DIFF_SRC, !input.dependency_count() ? input.dst_grad() : input.dst_grad_2()},
                {DNNL_ARG_WORKSPACE, pool_workspace_memory},
                {DNNL_ARG_DIFF_DST, dst_grad()},
            } });

            if (input.dependency_count()) // grad saved to buffer, so should add to main memory
            {
                auto adder = binary({ {algorithm::binary_add, src_md, src_md, src_md}, net.engine });
                net.bwd.push_back({ adder,
                {
                    {DNNL_ARG_SRC_0, input.dst_grad()},
                    {DNNL_ARG_SRC_1, input.dst_grad_2()},
                    {DNNL_ARG_DST, input.dst_grad()},
                } });
            }
        }
        std::reverse(net.bwd.begin() + reverse_from, net.bwd.end());
    }
}

GlobalPoolingLayer::GlobalPoolingLayer(Layer& input, algorithm algo) : Layer(input.net()), pool(input, algo, {}, {}, {}, {})
{
}

void GlobalPoolingLayer::Init()
{
    auto dims = pool.input.output_desc().dims();
    dims.erase(dims.begin(), dims.begin() + 2);
    pool.kernel = dims;
    pool.strides = memory::dims(pool.kernel.size(), 1);
    pool.padding_l = pool.padding_r = memory::dims(pool.kernel.size(), 0);
}

ConvLayer::ConvLayer(Layer& input, const memory::dims& kernel, const memory::dims& strides, const memory::dims& padding_l, const memory::dims& padding_r, bool use_bias)
    : Layer(input.net()), input(input), kernel(kernel), strides(strides), padding_l(padding_l), padding_r(padding_r), use_bias(use_bias)
{
    input.IncDependencies();
}

void ConvLayer::Init()
{
    NNetwork& net = this->net();
    input.DecDependencies();
    auto src_md = input.output_desc();

    auto w_dims = src_md.dims();

    for (memory::dim i = 2; i < w_dims.size(); i++)
    {
        w_dims[i] = (w_dims[i] - kernel[i] + padding_l[i - 2] + padding_r[i - 2]) / strides[i - 2] + 1;
    }
    w_dims[1] = kernel[0];

    auto weights_md = memory::desc(kernel, net.data_type, GetStridesForDims(kernel));
    auto dst_md = memory::desc(w_dims, net.data_type, GetStridesForDims(w_dims));

    weights = memory(weights_md, net.engine);
    net.weights.push_back(weights);

    output({ dst_md, net.engine });

    memory::desc bias_md;
    if (use_bias)
    {
        bias_md = memory::desc({ kernel[0] }, net.data_type, memory::format_tag::a);
        bias = memory(bias_md, net.engine);
        net.weights.push_back(bias);
    }

    auto conv_pd = use_bias
        ? convolution_forward::primitive_desc({ net.kind, algorithm::convolution_auto, src_md, weights_md, bias_md, dst_md, strides, padding_l, padding_r }, net.engine)
        : convolution_forward::primitive_desc({ net.kind, algorithm::convolution_auto, src_md, weights_md, dst_md, strides, padding_l, padding_r }, net.engine);

    auto conv_fwd = convolution_forward(conv_pd);

    net.fwd.push_back({ conv_fwd,
    {
        {DNNL_ARG_SRC, input.output()},
        {DNNL_ARG_WEIGHTS, weights},
        {DNNL_ARG_BIAS, bias},
        {DNNL_ARG_DST, output()},
    } });
    // Backward
    if (net.kind == dnnl::prop_kind::forward_training)
    {
        dst_grad({ dst_md, net.engine });
        auto conv_bwd_data = convolution_backward_data({ { algorithm::convolution_auto, src_md, weights_md, dst_md, strides, padding_l, padding_r }, net.engine, conv_pd });
        auto reverse_from = net.bwd.size();
        if (input.dst_grad())
        {
            net.bwd.push_back({ conv_bwd_data,
            {
                {DNNL_ARG_DIFF_SRC, !input.dependency_count() ? input.dst_grad() : input.dst_grad_2()},
                {DNNL_ARG_WEIGHTS, weights},
                {DNNL_ARG_DIFF_DST, dst_grad()},
            } });

            if (input.dependency_count()) // grad saved to buffer, so should add to main memory
            {
                auto adder = binary({ {algorithm::binary_add, src_md, src_md, src_md}, net.engine });
                net.bwd.push_back({ adder,
                {
                    {DNNL_ARG_SRC_0, input.dst_grad()},
                    {DNNL_ARG_SRC_1, input.dst_grad_2()},
                    {DNNL_ARG_DST, input.dst_grad()},
                } });
            }
        }
        auto d_weights = memory(weights_md, net.engine);
        auto d_bias = memory(bias_md, net.engine);
        auto conv_bwd_weights = use_bias
            ? convolution_backward_weights({ { algorithm::convolution_auto, src_md, weights_md, bias_md, dst_md, strides, padding_l, padding_r }, net.engine, conv_pd })
            : convolution_backward_weights({ { algorithm::convolution_auto, src_md, weights_md, dst_md, strides, padding_l, padding_r }, net.engine, conv_pd });
        net.bwd.push_back({ conv_bwd_weights,
            {
                {DNNL_ARG_SRC, input.output()},
                {DNNL_ARG_DIFF_WEIGHTS, d_weights},
                {DNNL_ARG_DIFF_BIAS, d_bias},
                {DNNL_ARG_DIFF_DST, dst_grad()},
            } });
        // TODO: Optimizer control
        auto bwd_weights_update = sum({ { 1, -net.learn_rate }, {weights_md, weights_md}, net.engine });
        net.bwd.push_back({ bwd_weights_update,
        {
            {DNNL_ARG_MULTIPLE_SRC + 0, weights},
            {DNNL_ARG_MULTIPLE_SRC + 1, d_weights},
            {DNNL_ARG_DST, weights},
        } });
        if (use_bias)
        {
            auto bwd_bias_update = sum({ { 1, -net.learn_rate }, {bias_md, bias_md}, net.engine });
            net.bwd.push_back({ bwd_bias_update,
            {
                {DNNL_ARG_MULTIPLE_SRC + 0, bias},
                {DNNL_ARG_MULTIPLE_SRC + 1, d_bias},
                {DNNL_ARG_DST, bias},
            } });
        }
        std::reverse(net.bwd.begin() + reverse_from, net.bwd.end());
    }
}

MultiplyLayer::MultiplyLayer(Layer& input1, Layer& input2) : Layer(input1.net()), input1(input1), input2(input2)
{
    assert(&input1.net() == &input2.net() && "Add layers from different networks impossible");
    input1.IncDependencies();
    input2.IncDependencies();
}

void MultiplyLayer::Init()
{
    NNetwork& net = this->net();
    input1.DecDependencies();
    input2.DecDependencies();

    auto src1_md = input1.output_desc();
    auto src2_md = input2.output_desc();

    auto mul = binary({ {algorithm::binary_mul, src1_md, src2_md, src1_md}, net.engine });

    output({ src1_md, net.engine });

    net.fwd.push_back({ mul,
    {
        {DNNL_ARG_SRC_0, input1.output()},
        {DNNL_ARG_SRC_1, input2.output()},
        {DNNL_ARG_DST, output()},
    } });

    // Backward
    if (net.kind == dnnl::prop_kind::forward_training)
    {
        dst_grad({ src1_md, net.engine });

        auto reverse_from = net.bwd.size();
        std::vector<Layer*> layers = { &input1, &input2 };
        for (int i = 0; i < 2; i++)
        {
            Layer& input = *layers[i];
            Layer& input_other = *layers[1 - i];
            if (input.dst_grad())
            {
                auto bwd_update = binary({ {algorithm::binary_mul, output_desc(), input_other.output_desc(), output_desc()}, net.engine });
                std::vector< std::pair<dnnl::primitive, std::unordered_map<int, dnnl::memory>>> tmp;
                tmp.push_back({ bwd_update,
                {
                    {DNNL_ARG_SRC_0, dst_grad()},
                    {DNNL_ARG_SRC_1, input_other.output()},
                    {DNNL_ARG_DST, input.dst_grad()},
                } });
                // If used broadcasting should use reduction
                if (output_desc() != input.output_desc())
                {
                    auto tmp_mem = memory(output_desc(), net.engine);
                    auto bwd_reduce = reduction({ {algorithm::reduction_sum, output_desc(), input.output_desc(), 0, 0}, net.engine });
                    tmp.back().second[DNNL_ARG_DST] = tmp_mem;
                    tmp.push_back({bwd_reduce,
                    {
                        {DNNL_ARG_SRC, tmp_mem},
                        {DNNL_ARG_DST, input.dst_grad()},
                    } });
                }
                if (input.dependency_count())
                {
                    auto desc = input.output_desc();
                    tmp.back().second[DNNL_ARG_DST] = input.dst_grad_2();
                    for (auto& x : tmp)
                        net.bwd.push_back(x);
                    auto adder = binary({ {algorithm::binary_add, desc, desc, desc}, net.engine });
                    net.bwd.push_back({ adder,
                    {
                        {DNNL_ARG_SRC_0, input.dst_grad()},
                        {DNNL_ARG_SRC_1, input.dst_grad_2()},
                        {DNNL_ARG_DST, input.dst_grad()},
                    } });
                }
                else
                {
                    for (auto& x : tmp)
                        net.bwd.push_back(x);
                }
            }
        }
        std::reverse(net.bwd.begin() + reverse_from, net.bwd.end());
    }
}

ReshapeLayer::ReshapeLayer(Layer& input, dnnl::memory::dims output_dims) : Layer(input.net()), input(input), output_dims(output_dims)
{
}

ReshapeLayer::ReshapeLayer(Layer& input, const std::function<dnnl::memory::dims(dnnl::memory::dims)>& src2dst) : Layer(input.net()), input(input), src2dst(src2dst)
{
}

void ReshapeLayer::Init()
{
    if (src2dst)
        output_dims = src2dst(input.output_desc().dims());
    output_md = input.output_desc().reshape(output_dims);
}

SELayer::SELayer(Layer& input, int inner_size)
    : Layer(input.net()), input(input), inner_size(inner_size),
    // Initialization order as defined in header
    pool(input, algorithm::pooling_avg),
    dense1(pool, 0),
    relu(dense1),
    dense2(relu, 0),
    sigm(dense2),
    reshape(sigm, [&input](dnnl::memory::dims x) -> dnnl::memory::dims
    {
        x.resize(input.output_desc().dims().size(), 1);
        return x;
    }),
    mul(input, reshape)
{
};

void SELayer::Init()
{
    int channel_count = input.output_desc().dims()[1];
    dense1.output_size = inner_size;
    dense2.output_size = channel_count;
}

BatchNormLayer::BatchNormLayer(Layer& input, float eps) : Layer(input.net()), input(input), eps(eps)
{
    input.IncDependencies();
}

void BatchNormLayer::Init()
{
    NNetwork& net = this->net();
    input.DecDependencies();
    auto desc = input.output_desc();
    auto c = desc.dims()[1];
    auto arg_d = net.kind == prop_kind::forward_training
        ? memory::desc({ c }, net.data_type, memory::format_tag::a)
        : memory::desc();
    auto mean_mem = memory(arg_d, net.engine);
    auto var_mem = memory(arg_d, net.engine);

    auto norm_pd = batch_normalization_forward::primitive_desc({ net.kind, desc, eps, normalization_flags::none }, net.engine);
    auto norm_fwd = batch_normalization_forward(norm_pd);

    output({ desc, net.engine });

    net.fwd.push_back({ norm_fwd,
    {
        {DNNL_ARG_SRC, input.output()},
        {DNNL_ARG_MEAN, mean_mem},
        {DNNL_ARG_VARIANCE, var_mem},
        {DNNL_ARG_DST, output()},
    } });
    // Backward
    if (net.kind == dnnl::prop_kind::forward_training)
    {
        dst_grad({ desc, net.engine });
        auto norm_bwd = batch_normalization_backward({ { prop_kind::backward_data, desc, desc, eps, normalization_flags::none }, net.engine, norm_pd });
        auto reverse_from = net.bwd.size();
        if (input.dst_grad())
        {
            net.bwd.push_back({ norm_bwd,
            {
                {DNNL_ARG_DIFF_SRC, !input.dependency_count() ? input.dst_grad() : input.dst_grad_2()},
                {DNNL_ARG_SRC, input.output()},
                {DNNL_ARG_MEAN, mean_mem},
                {DNNL_ARG_VARIANCE, var_mem},
                {DNNL_ARG_DIFF_DST, dst_grad()},
            } });

            if (input.dependency_count()) // grad saved to buffer, so should add to main memory
            {
                auto adder = binary({ {algorithm::binary_add, desc, desc, desc}, net.engine });
                net.bwd.push_back({ adder,
                {
                    {DNNL_ARG_SRC_0, input.dst_grad()},
                    {DNNL_ARG_SRC_1, input.dst_grad_2()},
                    {DNNL_ARG_DST, input.dst_grad()},
                } });
            }
        }
        std::reverse(net.bwd.begin() + reverse_from, net.bwd.end());
    }
}

SoftMaxLayer::SoftMaxLayer(Layer& input, int axis) : Layer(input.net()), input(input), axis(axis)
{
    input.IncDependencies();
}

void SoftMaxLayer::Init()
{
    NNetwork& net = this->net();
    input.DecDependencies();
    auto desc = input.output_desc();

    if (axis == -1)
    {
        auto dims = desc.dims();
        for (size_t i = 2; i < dims.size(); i++)
            dims[1] *= dims[i];
        dims.resize(2);
        desc = desc.reshape(dims);
        axis = 1;
    }

    auto soft_pd = softmax_forward::primitive_desc({ net.kind, desc, axis }, net.engine);
    auto soft_fwd = softmax_forward(soft_pd);

    output({ desc, net.engine });

    net.fwd.push_back({ soft_fwd,
    {
        {DNNL_ARG_SRC, input.output()},
        {DNNL_ARG_DST, output()},
    } });
    // Backward
    if (net.kind == dnnl::prop_kind::forward_training)
    {
        dst_grad({ desc, net.engine });
        auto soft_bwd = softmax_backward({ { desc, desc, axis }, net.engine, soft_pd });
        auto reverse_from = net.bwd.size();
        if (input.dst_grad())
        {
            net.bwd.push_back({ soft_bwd,
            {
                {DNNL_ARG_DIFF_SRC, !input.dependency_count() ? input.dst_grad() : input.dst_grad_2()},
                {DNNL_ARG_DST, output()},
                {DNNL_ARG_DIFF_DST, dst_grad()},
            } });

            if (input.dependency_count()) // grad saved to buffer, so should add to main memory
            {
                auto adder = binary({ {algorithm::binary_add, desc, desc, desc}, net.engine });
                net.bwd.push_back({ adder,
                {
                    {DNNL_ARG_SRC_0, input.dst_grad()},
                    {DNNL_ARG_SRC_1, input.dst_grad_2()},
                    {DNNL_ARG_DST, input.dst_grad()},
                } });
            }
        }
        std::reverse(net.bwd.begin() + reverse_from, net.bwd.end());
    }
}

MeanSquaredLoss::MeanSquaredLoss(NNetwork& net) : Loss(net)
{
}

void MeanSquaredLoss::Init(dnnl::memory output, dnnl::memory answer, dnnl::memory grad)
{
    auto desc = answer.get_desc();
    auto subber = binary({ {algorithm::binary_sub, desc, desc, desc}, net().engine });
    net().bwd.push_back({ subber,
    {
        {DNNL_ARG_SRC_0, output},
        {DNNL_ARG_SRC_1, answer},
        {DNNL_ARG_DST, grad},
    } });
}

CrossEntropyLoss::CrossEntropyLoss(NNetwork& net) : Loss(net)
{
}

void CrossEntropyLoss::Init(dnnl::memory output, dnnl::memory answer, dnnl::memory grad)
{
    auto desc = answer.get_desc();
    //auto dims = memory::dims(desc.dims().size(), 1);
    //auto eps_mem = memory::desc(dims, net().data_type, GetStridesForDims(dims));
    //auto add_p = binary({ {algorithm::binary_add, desc, desc, desc}, net().engine });
    auto clip_p = eltwise_forward({ {prop_kind::forward_inference, algorithm::eltwise_clip, desc, 1e-7, 1 - 1e-7}, net().engine });
    auto inv_p = eltwise_forward({ {prop_kind::forward_inference, algorithm::eltwise_pow, desc, -1, -1}, net().engine });
    auto mul_p = binary({ {algorithm::binary_mul, desc, desc, desc}, net().engine });
    net().bwd.push_back({ clip_p,
    {
        {DNNL_ARG_SRC, output},
        {DNNL_ARG_DST, grad},
    } });

    net().bwd.push_back({ inv_p,
    {
        {DNNL_ARG_SRC, grad},
        {DNNL_ARG_DST, grad},
    } });

    net().bwd.push_back({ mul_p,
    {
        {DNNL_ARG_SRC_0, grad},
        {DNNL_ARG_SRC_1, answer},
        {DNNL_ARG_DST, grad},
    } });
}
