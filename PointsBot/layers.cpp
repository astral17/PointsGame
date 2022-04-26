#include "layers.h"

#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <initializer_list>

using namespace dnnl;

void operator>>(const dnnl::memory& mem, void* handle)
{
    dnnl::engine eng = mem.get_engine();
    size_t size = mem.get_desc().get_size();

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
    size_t size = mem.get_desc().get_size();

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


Layer::Layer(NNetwork& net) : net(&net)
{
    net.layers.push_back(this);
}

NNetwork::NNetwork(dnnl::engine eng, dnnl::prop_kind kind) : engine(eng), kind(kind) {}

void NNetwork::Build(float learn_rate)
{
    this->learn_rate = learn_rate;
    for (auto layer : layers)
        layer->Init();

    auto desc = output.get_desc();
    auto subber = binary({ {algorithm::binary_sub, desc, desc, desc}, engine });
    answer = memory(desc, engine);
    bwd.push_back({ subber,
        {
            {DNNL_ARG_SRC_0, output},
            {DNNL_ARG_SRC_1, answer},
            {DNNL_ARG_DST, grad},
        } });
    std::reverse(bwd.begin(), bwd.end());
}

void NNetwork::Forward(const dnnl::stream& s)
{
    for (auto cur : fwd)
        cur.first.execute(s, cur.second);
}

void NNetwork::Backward(const dnnl::stream& s)
{
    for (auto cur : bwd)
    {
        //std::cerr << "NEXT:\n";
        //for (auto it = cur.second.begin(); it != cur.second.end(); it++)
        //    std::cerr << it->first << " ";
        //std::cerr << "\n";
        cur.first.execute(s, cur.second);
    }
}

InputLayer::InputLayer(NNetwork& net, const dnnl::memory::desc& input_desc) : Layer(net)
{
    net.input = output = dnnl::memory(input_desc, net.engine);
}

InputLayer::InputLayer(NNetwork& net, const dnnl::memory::dims& adims, dnnl::memory::data_type adata_type, dnnl::memory::format_tag aformat_tag)
    : InputLayer(net, { adims, adata_type, aformat_tag }) {}

void InputLayer::Init()
{
}

void InputLayer::SetInput(void* handle)
{
    output << handle;
}

DenseLayer::DenseLayer(Layer& input, int output_size) : Layer(*input.net), input(input), output_size(output_size)
{
    input.dependency_count++;
}

void DenseLayer::Init()
{
    NNetwork& net = *this->net;
    input.dependency_count--;
    auto src_md = input.output.get_desc();
    auto w_dims = src_md.dims();
    w_dims[0] = output_size;
    memory::dims w_strides(w_dims.size());
    for (memory::dim i = w_strides.size() - 1, prod = 1; i >= 0; i--)
    {
        w_strides[i] = prod;
        prod *= w_dims[i];
    }
    auto weight_md = memory::desc(
        w_dims,
        memory::data_type::f32,
        w_strides
    );
    auto dst_md = memory::desc(
        { src_md.data.dims[0], output_size },
        memory::data_type::f32,
        memory::format_tag::ab
    );

    weights = memory(weight_md, net.engine);
    output = memory(dst_md, net.engine);

    auto product_pd = inner_product_forward::primitive_desc({ net.kind, src_md, weight_md, dst_md }, net.engine);
    auto fwd = inner_product_forward(product_pd);

    net.fwd.push_back({ fwd,
        {
            {DNNL_ARG_SRC, input.output},
            {DNNL_ARG_WEIGHTS, weights},
            {DNNL_ARG_DST, output},
        } });
    // Backward
    if (net.kind == dnnl::prop_kind::forward_training)
    {
        dst_grad = memory(dst_md, net.engine);

        auto reverse_from = net.bwd.size();
        if (input.dst_grad)
        {
            auto bwd_data = inner_product_backward_data({ {src_md, weight_md, dst_md}, net.engine, product_pd });
            net.bwd.push_back({ bwd_data,
            {
                {DNNL_ARG_DIFF_SRC, !input.dependency_count ? input.dst_grad :
                    input.dst_grad_2 ? input.dst_grad_2 : input.dst_grad_2 = memory(src_md, net.engine)},
                {DNNL_ARG_WEIGHTS, weights},
                {DNNL_ARG_DIFF_DST, dst_grad},
            } });

            if (input.dependency_count) // grad saved to buffer, so should add to main memory
            {
                auto desc = input.dst_grad.get_desc();
                auto adder = binary({ {algorithm::binary_add, desc, desc, desc}, net.engine });
                net.bwd.push_back({ adder,
                {
                    {DNNL_ARG_SRC_0, input.dst_grad},
                    {DNNL_ARG_SRC_1, input.dst_grad_2},
                    {DNNL_ARG_DST, input.dst_grad},
                } });
            }
        }

        auto d_weights = memory(weight_md, net.engine);
        auto bwd_weights = inner_product_backward_weights({ {src_md, weight_md, dst_md}, net.engine, product_pd });
        net.bwd.push_back({ bwd_weights,
            {
                {DNNL_ARG_SRC, input.output},
                {DNNL_ARG_DIFF_WEIGHTS, d_weights},
                {DNNL_ARG_DIFF_DST, dst_grad},
            } });
        // TODO: Optimizer control
        auto bwd_weights_update = sum({ { 1, -net.learn_rate }, {weight_md, weight_md}, net.engine });
        net.bwd.push_back({ bwd_weights_update,
            {
                {DNNL_ARG_MULTIPLE_SRC + 0, weights},
                {DNNL_ARG_MULTIPLE_SRC + 1, d_weights},
                {DNNL_ARG_DST, weights},
            } });
        std::reverse(net.bwd.begin() + reverse_from, net.bwd.end());
    }
    net.output = output;
    net.grad = dst_grad;
}

void DenseLayer::SetWeights(void* handle)
{
    weights << handle;
}

EltwiseLayer::EltwiseLayer(Layer& input, algorithm algo, float alpha, float beta) : Layer(*input.net), input(input), algo(algo), alpha(alpha), beta(beta)
{
    input.dependency_count++;
}

void EltwiseLayer::Init()
{
    NNetwork& net = *this->net;
    input.dependency_count--;
    auto desc = input.output.get_desc();
    auto elt_pd = eltwise_forward::primitive_desc({ net.kind, algo, desc, alpha, beta }, net.engine);
    auto elt_fwd = eltwise_forward(elt_pd);

    output = memory(desc, net.engine);

    net.fwd.push_back({ elt_fwd,
        {
            {DNNL_ARG_SRC, input.output},
            {DNNL_ARG_DST, output},
        } });
    // Backward
    if (net.kind == dnnl::prop_kind::forward_training)
    {
        dst_grad = memory(desc, net.engine);
        auto elt_bwd = eltwise_backward({ {algo, desc, desc, alpha, beta}, net.engine, elt_pd });
        auto reverse_from = net.bwd.size();
        if (input.dst_grad)
        {
            if (input.dependency_count)
            {
                if (!input.dst_grad_2)
                    input.dst_grad_2 = memory(desc, net.engine);
                net.bwd.push_back({ elt_bwd,
                {
                    {DNNL_ARG_SRC, input.output},
                    {DNNL_ARG_DIFF_SRC, input.dst_grad_2},
                    {DNNL_ARG_DIFF_DST, dst_grad},
                } });

                auto adder = binary({ {algorithm::binary_add, desc, desc, desc}, net.engine });
                net.bwd.push_back({ adder,
                {
                    {DNNL_ARG_SRC_0, input.dst_grad},
                    {DNNL_ARG_SRC_1, input.dst_grad_2},
                    {DNNL_ARG_DST, input.dst_grad},
                } });
            }
            else
            {
                net.bwd.push_back({ elt_bwd,
                {
                    {DNNL_ARG_SRC, input.output},
                    {DNNL_ARG_DIFF_SRC, input.dst_grad},
                    {DNNL_ARG_DIFF_DST, dst_grad},
                } });
            }
        }
        std::reverse(net.bwd.begin() + reverse_from, net.bwd.end());
    }
    net.output = output;
    net.grad = dst_grad;
}

ReluLayer::ReluLayer(Layer& input, float alpha) : EltwiseLayer(input, algorithm::eltwise_relu, alpha)
{
}

LayerAdder::LayerAdder(Layer& input1, Layer& input2) : Layer(*input1.net), input1(input1), input2(input2)
{
    assert(input1.net == input2.net && "Add layers from different networks impossible");
    input1.dependency_count++;
    input2.dependency_count++;
}

void LayerAdder::Init()
{
    NNetwork& net = *this->net;
    input1.dependency_count--;
    input2.dependency_count--;

    auto desc = input1.output.get_desc();
    assert(desc == input2.output.get_desc());
    auto adder = binary({ {algorithm::binary_add, desc, desc, desc}, net.engine });

    output = memory(desc, net.engine);

    net.fwd.push_back({ adder,
    {
        {DNNL_ARG_SRC_0, input1.output},
        {DNNL_ARG_SRC_1, input2.output},
        {DNNL_ARG_DST, output},
    } });

    // Backward
    if (net.kind == dnnl::prop_kind::forward_training)
    {
        dst_grad = memory(desc, net.engine);

        auto reverse_from = net.bwd.size();
        std::vector<Layer*> layers = { &input1, &input2 };
        for (Layer* layer : layers)
        {
            Layer& input = *layer;
            if (input.dst_grad)
            {
                if (input.dependency_count)
                {
                    auto bwd_update = sum({ { 1, 1 }, {desc, desc}, net.engine });
                    net.bwd.push_back({ bwd_update,
                    {
                        {DNNL_ARG_MULTIPLE_SRC + 0, input.dst_grad},
                        {DNNL_ARG_MULTIPLE_SRC + 1, dst_grad},
                        {DNNL_ARG_DST, input.dst_grad},
                    } });
                }
                else
                {
                    auto bwd_update = sum({ { 1 }, {desc}, net.engine });
                    net.bwd.push_back({ bwd_update,
                    {
                        {DNNL_ARG_MULTIPLE_SRC + 0, dst_grad},
                        {DNNL_ARG_DST, input.dst_grad},
                    } });
                }
                //net.bwd.back().second.insert({ DNNL_ARG_MULTIPLE_SRC + 1, input1.dst_grad });
            }
        }
        std::reverse(net.bwd.begin() + reverse_from, net.bwd.end());
    }
    net.output = output;
    net.grad = dst_grad;
}

ReorderLayer::ReorderLayer(Layer& input, dnnl::memory::desc dst_md) : Layer(*input.net), input(input), dst_md(dst_md)
{
    input.dependency_count++;
}

void ReorderLayer::Init()
{
    NNetwork& net = *this->net;
    input.dependency_count--;

    auto src_md = input.output.get_desc();
    output = memory(dst_md, net.engine);
    auto reorder_fwd = reorder({ input.output, output });
    net.fwd.push_back({ reorder_fwd,
    {
        {DNNL_ARG_FROM, input.output},
        {DNNL_ARG_TO, output},
    } });
    // Backward
    if (net.kind == dnnl::prop_kind::forward_training)
    {
        dst_grad = memory(dst_md, net.engine);

        auto reverse_from = net.bwd.size();
        if (input.dst_grad)
        {
            auto reorder_bwd = reorder({ dst_grad, input.dst_grad });
            if (input.dependency_count)
            {
                if (!input.dst_grad_2)
                    input.dst_grad_2 = memory(src_md, net.engine);
                net.fwd.push_back({ reorder_bwd,
                {
                    {DNNL_ARG_FROM, dst_grad},
                    {DNNL_ARG_TO, input.dst_grad_2},
                } });
                auto adder = binary({ {algorithm::binary_add, src_md, src_md, src_md}, net.engine });
                net.bwd.push_back({ adder,
                {
                    {DNNL_ARG_SRC_0, input.dst_grad},
                    {DNNL_ARG_SRC_1, input.dst_grad_2},
                    {DNNL_ARG_DST, input.dst_grad},
                } });
            }
            else
            {
                net.fwd.push_back({ reorder_bwd,
                {
                    {DNNL_ARG_FROM, dst_grad},
                    {DNNL_ARG_TO, input.dst_grad},
                } });
            }
        }

        std::reverse(net.bwd.begin() + reverse_from, net.bwd.end());
    }
    net.output = output;
    net.grad = dst_grad;
}

PoolingLayer::PoolingLayer(Layer& input, dnnl::algorithm algo, const dnnl::memory::dims& strides, const dnnl::memory::dims& kernel, const dnnl::memory::dims& padding_l, const dnnl::memory::dims& padding_r) : Layer(*input.net), input(input), algo(algo), strides(strides), kernel(kernel), padding_l(padding_l), padding_r(padding_r)
{
    input.dependency_count++;
}

void PoolingLayer::Init()
{
    NNetwork& net = *this->net;
    input.dependency_count--;
    //dnnl::pooling_v2_forward::desc()
    auto src_md = input.output.get_desc();

    auto w_dims = src_md.dims();

    for (memory::dim i = 2; i < w_dims.size(); i++)
    {
        w_dims[i] = (w_dims[i] - kernel[i - 2] + padding_l[i - 2] + padding_r[i - 2]) / strides[i - 2] + 1;
    }
    memory::dims w_strides(w_dims.size());
    for (memory::dim i = w_strides.size() - 1, prod = 1; i >= 0; i--)
    {
        w_strides[i] = prod;
        prod *= w_dims[i];
    }

    auto dst_md = memory::desc(w_dims, memory::data_type::f32, w_strides);
    auto pool_pd = pooling_forward::primitive_desc({ net.kind, algo, src_md, dst_md, strides, kernel, padding_l, padding_r }, net.engine);
    auto pool_fwd = pooling_forward(pool_pd);

    auto pool_workspace_memory = memory(pool_pd.workspace_desc(), net.engine);
    output = memory(dst_md, net.engine);

    net.fwd.push_back({ pool_fwd,
        {
            {DNNL_ARG_SRC, input.output},
            {DNNL_ARG_WORKSPACE, pool_workspace_memory},
            {DNNL_ARG_DST, output},
        } });
    // Backward
    if (net.kind == dnnl::prop_kind::forward_training)
    {
        dst_grad = memory(dst_md, net.engine);
        auto pool_bwd = pooling_backward({ { algo, src_md, dst_md, strides, kernel, padding_l, padding_r }, net.engine, pool_pd });
        auto reverse_from = net.bwd.size();
        if (input.dst_grad)
        {
            if (input.dependency_count)
            {
                if (!input.dst_grad_2)
                    input.dst_grad_2 = memory(src_md, net.engine);
                net.bwd.push_back({ pool_bwd,
                {
                    {DNNL_ARG_DIFF_SRC, input.dst_grad_2},
                    {DNNL_ARG_WORKSPACE, pool_workspace_memory},
                    {DNNL_ARG_DIFF_DST, dst_grad},
                } });

                auto adder = binary({ {algorithm::binary_add, src_md, src_md, src_md}, net.engine });
                net.bwd.push_back({ adder,
                {
                    {DNNL_ARG_SRC_0, input.dst_grad},
                    {DNNL_ARG_SRC_1, input.dst_grad_2},
                    {DNNL_ARG_DST, input.dst_grad},
                } });
            }
            else
            {
                net.bwd.push_back({ pool_bwd,
                {
                    {DNNL_ARG_DIFF_SRC, input.dst_grad},
                    {DNNL_ARG_WORKSPACE, pool_workspace_memory},
                    {DNNL_ARG_DIFF_DST, dst_grad},
                } });
            }
        }
        std::reverse(net.bwd.begin() + reverse_from, net.bwd.end());
    }
    net.output = output;
    net.grad = dst_grad;
}

ConvLayer::ConvLayer(Layer& input, const memory::dims& kernel, const memory::dims& strides, const memory::dims& padding_l, const memory::dims& padding_r) : Layer(*input.net), input(input), kernel(kernel), strides(strides), padding_l(padding_l), padding_r(padding_r)
{
    input.dependency_count++;
}

void ConvLayer::Init()
{
    NNetwork& net = *this->net;
    input.dependency_count--;
    auto src_md = input.output.get_desc();

    auto w_dims = src_md.dims();

    for (memory::dim i = 2; i < w_dims.size(); i++)
    {
        w_dims[i] = (w_dims[i] - kernel[i] + padding_l[i - 2] + padding_r[i - 2]) / strides[i - 2] + 1;
    }
    w_dims[1] = kernel[0];

    memory::dims k_strides(w_dims.size());
    for (memory::dim i = k_strides.size() - 1, prod = 1; i >= 0; i--)
    {
        k_strides[i] = prod;
        prod *= kernel[i];
    }
    memory::dims d_strides(w_dims.size());
    for (memory::dim i = d_strides.size() - 1, prod = 1; i >= 0; i--)
    {
        d_strides[i] = prod;
        prod *= w_dims[i];
    }

    auto weights_md = memory::desc(kernel, memory::data_type::f32, k_strides);
    auto dst_md = memory::desc(w_dims, memory::data_type::f32, d_strides);

    auto conv_pd = convolution_forward::primitive_desc({ net.kind, algorithm::convolution_auto, src_md, weights_md, dst_md, strides, padding_l, padding_r }, net.engine);

    auto conv_fwd = convolution_forward(conv_pd);

    weights = memory(weights_md, net.engine);
    output = memory(dst_md, net.engine);

    net.fwd.push_back({ conv_fwd,
        {
            {DNNL_ARG_SRC, input.output},
            {DNNL_ARG_WEIGHTS, weights},
            {DNNL_ARG_DST, output},
        } });
    // Backward
    if (net.kind == dnnl::prop_kind::forward_training)
    {
        dst_grad = memory(dst_md, net.engine);
        auto conv_bwd_data = convolution_backward_data({ { algorithm::convolution_auto, src_md, weights_md, dst_md, strides, padding_l, padding_r }, net.engine, conv_pd });
        auto reverse_from = net.bwd.size();
        if (input.dst_grad)
        {
            if (input.dependency_count)
            {
                if (!input.dst_grad_2)
                    input.dst_grad_2 = memory(src_md, net.engine);
                net.bwd.push_back({ conv_bwd_data,
                {
                    {DNNL_ARG_DIFF_SRC, input.dst_grad_2},
                    {DNNL_ARG_WEIGHTS, weights},
                    {DNNL_ARG_DIFF_DST, dst_grad},
                } });

                auto adder = binary({ {algorithm::binary_add, src_md, src_md, src_md}, net.engine });
                net.bwd.push_back({ adder,
                {
                    {DNNL_ARG_SRC_0, input.dst_grad},
                    {DNNL_ARG_SRC_1, input.dst_grad_2},
                    {DNNL_ARG_DST, input.dst_grad},
                } });
            }
            else
            {
                net.bwd.push_back({ conv_bwd_data,
                {
                    {DNNL_ARG_DIFF_SRC, input.dst_grad},
                    {DNNL_ARG_WEIGHTS, weights},
                    {DNNL_ARG_DIFF_DST, dst_grad},
                } });
            }
        }
        auto d_weights = memory(weights_md, net.engine);
        auto conv_bwd_weights = convolution_backward_weights({ { algorithm::convolution_auto, src_md, weights_md, dst_md, strides, padding_l, padding_r }, net.engine, conv_pd });
        net.bwd.push_back({ conv_bwd_weights,
            {
                {DNNL_ARG_SRC, input.output},
                {DNNL_ARG_DIFF_WEIGHTS, d_weights},
                {DNNL_ARG_DIFF_DST, dst_grad},
            } });
        // TODO: Optimizer control
        auto bwd_weights_update = sum({ { 1, -net.learn_rate }, {weights_md, weights_md}, net.engine });
        net.bwd.push_back({ bwd_weights_update,
            {
                {DNNL_ARG_MULTIPLE_SRC + 0, weights},
                {DNNL_ARG_MULTIPLE_SRC + 1, d_weights},
                {DNNL_ARG_DST, weights},
            } });
        std::reverse(net.bwd.begin() + reverse_from, net.bwd.end());
    }
    net.output = output;
    net.grad = dst_grad;
}
