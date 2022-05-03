#pragma once

#include <dnnl.hpp>
#include <random>

size_t GetMemoryCount(const dnnl::memory& mem);
size_t GetMemoryByteSize(const dnnl::memory& mem);

void operator>>(const dnnl::memory& mem, void* handle);

template<typename T>
void operator>>(const dnnl::memory& mem, std::vector<T>& v)
{
    v.resize(GetMemoryCount(mem));
    mem >> v.data();
}

//void operator>>(const dnnl::memory& mem, std::string& s)
//{
//    s.resize(GetMemoryByteSize(mem));
//    mem >> (void*)s.c_str();
//}

void operator<<(const dnnl::memory& mem, void* handle);

template<typename T>
void operator<<(const dnnl::memory& mem, std::vector<T>& v)
{
    mem << v.data();
}

//void operator<<(const dnnl::memory& mem, std::string& s)
//{
//    mem << (void*)s.c_str();
//}

struct NNetwork;

// TODO: Layer classes just shared_ptr to memory, for in place build network

struct Layer
{
public:
    NNetwork* net = nullptr;
    dnnl::memory output;
    dnnl::memory dst_grad;
    // Temp buffer, if dst_grad already used
    dnnl::memory dst_grad_2;

    int dependency_count = 0;
    Layer(NNetwork& net);
    virtual void Init() = 0;
};

struct NNetwork
{
public:
    dnnl::engine engine;
    dnnl::prop_kind kind;
    std::vector<Layer*> layers;
    std::vector<dnnl::memory> weights;
    std::vector< std::pair<dnnl::primitive, std::unordered_map<int, dnnl::memory>>> fwd;
    std::vector< std::pair<dnnl::primitive, std::unordered_map<int, dnnl::memory>>> bwd;
    // TODO: Optimizer
    float learn_rate = 0.01;

    //InputLayer input_layer;

    //NNetwork(dnnl::engine eng, int input_size, dnnl::prop_kind kind = dnnl::prop_kind::forward_training) : engine(eng), kind(kind) {}

    NNetwork(dnnl::engine eng, dnnl::prop_kind kind = dnnl::prop_kind::forward_training);
    
    // Set loss function, optimizer, prepare forward and backward list
    void Build(float learn_rate);
    void RandomWeights(std::mt19937 &mt, float min = -1, float max = 1);
    void SaveWeights(const std::string &file);
    void LoadWeights(const std::string& file);
    void Forward(const dnnl::stream& s);
    void Backward(const dnnl::stream& s);
    // GetInputMemory
    // GetOutputMemory
    dnnl::memory input;
    dnnl::memory output;
    dnnl::memory grad;
    dnnl::memory answer;
};

struct InputLayer : Layer
{
public:
    InputLayer(NNetwork& net, const dnnl::memory::desc& input_desc);
    InputLayer(NNetwork& net, const dnnl::memory::dims& adims, dnnl::memory::data_type adata_type, dnnl::memory::format_tag aformat_tag);
    virtual void Init() override;
    void SetInput(void* handle);
};

// TODO: Bias

struct DenseLayer : Layer
{
    Layer& input;
    int output_size;
public:
    dnnl::memory weights;

    DenseLayer(Layer& input, int output_size);

    virtual void Init() override;
};

struct EltwiseLayer : Layer
{
    Layer& input;
    dnnl::algorithm algo;
    float alpha, beta;
public:
    EltwiseLayer(Layer& input, dnnl::algorithm algo, float alpha = 0, float beta = 0);

    virtual void Init() override;
};

struct ReluLayer : EltwiseLayer
{
public:
    ReluLayer(Layer& input, float alpha = 0);
};

struct SigmoidLayer : EltwiseLayer
{
public:
    SigmoidLayer(Layer& input, float alpha = 0);
};

struct LayerAdder : Layer
{
    Layer& input1;
    Layer& input2;
public:
    LayerAdder(Layer& input1, Layer& input2);

    virtual void Init() override;
};

// Mb unuseful
struct ReorderLayer : Layer
{
    Layer& input;
    dnnl::memory::desc dst_md;
public:
    ReorderLayer(Layer& input, dnnl::memory::desc dst_md);
    virtual void Init() override;
};

struct PoolingLayer : Layer
{
    Layer& input;
    dnnl::algorithm algo;
    dnnl::memory::dims strides, kernel, padding_l, padding_r;
public:
    PoolingLayer(Layer& input, dnnl::algorithm algo, const dnnl::memory::dims &strides, const dnnl::memory::dims& kernel, const dnnl::memory::dims& padding_l, const dnnl::memory::dims& padding_r);
    virtual void Init() override;
};

// TODO: allow use GlobalPoolingLayer as input instead pool field
struct GlobalPoolingLayer : Layer
{
    // Temporarily use this as input for next layers
    PoolingLayer pool;
public:
    GlobalPoolingLayer(Layer& input, dnnl::algorithm algo);
    virtual void Init() override;
};

// TODO: Bias

struct ConvLayer : Layer
{
    Layer& input;
    dnnl::memory::dims kernel, strides, padding_l, padding_r;
public:
    dnnl::memory weights;
    ConvLayer(Layer& input, const dnnl::memory::dims& kernel, const dnnl::memory::dims& strides, const dnnl::memory::dims& padding_l, const dnnl::memory::dims& padding_r);
    virtual void Init() override;
};

struct MultiplyLayer : Layer
{
    Layer& input1;
    Layer& input2;
public:
    MultiplyLayer(Layer& input1, Layer& input2);
    virtual void Init() override;
};

// TODO: allow use SELayer as input instead mul field
struct SELayer : Layer
{
    Layer& input;
    int ratio;
    // Don't shuffle layers cause of initialization order

    GlobalPoolingLayer pool;
    DenseLayer dense1;
    ReluLayer relu;
    DenseLayer dense2;
    SigmoidLayer sigm;
    // Temporarily use this as input for next layers
    MultiplyLayer mul;
public:
    SELayer(Layer& input, int ratio);
    virtual void Init() override;
};
