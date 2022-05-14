#pragma once

#include <dnnl.hpp>
#include <random>
#include <functional>

#include <assert.h>

dnnl::memory::dims GetStridesForDims(const dnnl::memory::dims& dims);
size_t GetMemoryCount(const dnnl::memory& mem);
size_t GetMemoryByteSize(const dnnl::memory& mem);

void* MemoryMapData(const dnnl::memory& mem);

void MemoryUnmapData(const dnnl::memory& mem, void* handle);

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

class Layer
{
    // TODO: separate interface from variable defenition?

    NNetwork* net_ = nullptr;
    dnnl::memory output_;
    dnnl::memory dst_grad_;
    dnnl::memory dst_grad_2_;

    int dependency_count_ = 0;
public:
    NNetwork& net() const { return *net_; }
    // Getter for output/dst_grad/dst_grad_2 memory descriptor
    virtual dnnl::memory::desc output_desc() const { return output().get_desc(); }
    // Getter
    virtual dnnl::memory output() const { return output_; }
    // Setter
    virtual dnnl::memory output(const dnnl::memory& output) { return output_ = output; }
    // Getter, init field if needed
    virtual dnnl::memory dst_grad() const { return dst_grad_; }
    virtual dnnl::memory dst_grad(const dnnl::memory& dst_grad) { return dst_grad_ = dst_grad; }
    // Temp buffer, if dst_grad already used
    // Getter, init field if needed
    virtual dnnl::memory dst_grad_2();
    virtual int dependency_count() const { return dependency_count_; }
    virtual int dependency_count(int count) { return dependency_count_ = count; }
    void IncDependencies() { dependency_count(dependency_count() + 1); }
    void DecDependencies() { dependency_count(dependency_count() - 1); }
    Layer(NNetwork& net);
    virtual void Init() = 0;
};

class Loss
{
    NNetwork* net_ = nullptr;
public:
    NNetwork& net() const { return *net_; }
    Loss(NNetwork& net);
    virtual void Init(dnnl::memory output, dnnl::memory answer, dnnl::memory grad) = 0;
};

struct NNetwork
{
public:
    dnnl::engine engine;
    dnnl::prop_kind kind;
    dnnl::memory::data_type data_type;
    std::vector<Layer*> layers;
    std::vector<dnnl::memory> weights;
    std::vector< std::pair<dnnl::primitive, std::unordered_map<int, dnnl::memory>>> fwd;
    std::vector< std::pair<dnnl::primitive, std::unordered_map<int, dnnl::memory>>> bwd;
    // TODO: Optimizer
    float learn_rate = 0.01;

    //InputLayer input_layer;

    //NNetwork(dnnl::engine eng, int input_size, dnnl::prop_kind kind = dnnl::prop_kind::forward_training) : engine(eng), kind(kind) {}

    NNetwork(const dnnl::engine& eng, dnnl::prop_kind kind = dnnl::prop_kind::forward_training, dnnl::memory::data_type data_type = dnnl::memory::data_type::f32);
    
    // Set loss function, optimizer, prepare forward and backward list
    void Build(const std::vector<Layer*>& output, float learn_rate, const std::vector<Loss*>& loss = {});
    void RandomWeights(std::mt19937 &mt, float min = -1, float max = 1);
    void SaveWeights(const std::string &file);
    void LoadWeights(const std::string& file);
    void Forward(const dnnl::stream& s);
    void Backward(const dnnl::stream& s);
    //float GetLoss(const dnnl::stream& s);
    dnnl::memory input;
    std::vector<dnnl::memory> outputs;
    std::vector<dnnl::memory> grads;
    std::vector<dnnl::memory> answers;
    dnnl::memory output(size_t index = 0) { return outputs[index]; }
    dnnl::memory grad(size_t index = 0) { return grads[index]; }
    dnnl::memory answer(size_t index = 0) { return answers[index]; }
};

struct InputLayer : Layer
{
public:
    InputLayer(NNetwork& net, const dnnl::memory::dims& adims);
    InputLayer(NNetwork& net, const dnnl::memory::desc& input_desc);
    InputLayer(NNetwork& net, const dnnl::memory::dims& adims, dnnl::memory::data_type adata_type, dnnl::memory::format_tag aformat_tag);
    virtual void Init() override;
    void SetInput(void* handle);
};

struct DenseLayer : Layer
{
    Layer& input;
    int output_size;
    bool use_bias;
public:
    dnnl::memory weights;
    dnnl::memory bias;

    DenseLayer(Layer& input, int output_size, bool use_bias = true);

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
    SigmoidLayer(Layer& input);
};
struct TanhLayer : EltwiseLayer
{
public:
    TanhLayer(Layer& input);
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

struct GlobalPoolingLayer : Layer
{
    PoolingLayer pool;
public:
    GlobalPoolingLayer(Layer& input, dnnl::algorithm algo);
    // Getter
    virtual dnnl::memory output() const override { return pool.output(); }
    // Setter
    virtual dnnl::memory output(const dnnl::memory& output) override { return pool.output(output); }
    // Getter, init field if needed
    virtual dnnl::memory dst_grad() const override { return pool.dst_grad(); }
    virtual dnnl::memory dst_grad(const dnnl::memory& dst_grad) override { return pool.dst_grad(dst_grad); }
    // Temp buffer, if dst_grad already used
    // Getter, init field if needed
    virtual dnnl::memory dst_grad_2() override { return pool.dst_grad_2(); }
    virtual int dependency_count() const override { return pool.dependency_count(); }
    virtual int dependency_count(int count) override { return pool.dependency_count(count); }
    virtual void Init() override;
};

// TODO: Bias

struct ConvLayer : Layer
{
    Layer& input;
    dnnl::memory::dims kernel, strides, padding_l, padding_r;
    bool use_bias;
public:
    dnnl::memory weights;
    dnnl::memory bias;
    ConvLayer(Layer& input, const dnnl::memory::dims& kernel, const dnnl::memory::dims& strides, const dnnl::memory::dims& padding_l, const dnnl::memory::dims& padding_r, bool use_bias = true);
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

struct ReshapeLayer : Layer
{
    Layer& input;
    dnnl::memory::desc output_md;
    dnnl::memory::dims output_dims;
    std::function<dnnl::memory::dims(const dnnl::memory::dims&)> src2dst;
public:
    ReshapeLayer(Layer& input, dnnl::memory::dims output_dims);
    ReshapeLayer(Layer& input, const std::function<dnnl::memory::dims(dnnl::memory::dims)>& src2dst);

    // Getter for output/dst_grad/dst_grad_2 memory descriptor
    virtual dnnl::memory::desc output_desc() const override { return output_md; }
    // Getter
    virtual dnnl::memory output() const override { return input.output(); }
    // Setter
    virtual dnnl::memory output(const dnnl::memory& output) override { return input.output(output); }
    // Getter, init field if needed
    virtual dnnl::memory dst_grad() const override { return input.dst_grad(); }
    virtual dnnl::memory dst_grad(const dnnl::memory& dst_grad) override { return input.dst_grad(dst_grad); }
    // Temp buffer, if dst_grad already used
    // Getter, init field if needed
    virtual dnnl::memory dst_grad_2() override { return input.dst_grad_2(); }
    virtual int dependency_count() const override { return input.dependency_count(); }
    virtual int dependency_count(int count) override { return input.dependency_count(count); }

    virtual void Init() override;
};

struct SELayer : Layer
{
    Layer& input;
    int inner_size;
    // Don't shuffle layers cause of initialization order

    GlobalPoolingLayer pool;
    DenseLayer dense1;
    ReluLayer relu;
    DenseLayer dense2;
    SigmoidLayer sigm;
    ReshapeLayer reshape;
    MultiplyLayer mul;
public:
    SELayer(Layer& input, int inner_size);

    // Getter
    virtual dnnl::memory output() const override { return mul.output(); }
    // Setter
    virtual dnnl::memory output(const dnnl::memory& output) override { return mul.output(output); }
    // Getter, init field if needed
    virtual dnnl::memory dst_grad() const override { return mul.dst_grad(); }
    virtual dnnl::memory dst_grad(const dnnl::memory& dst_grad) override { return mul.dst_grad(dst_grad); }
    // Temp buffer, if dst_grad already used
    // Getter, init field if needed
    virtual dnnl::memory dst_grad_2() override { return mul.dst_grad_2(); }
    virtual int dependency_count() const override { return mul.dependency_count(); }
    virtual int dependency_count(int count) override { return mul.dependency_count(count); }

    virtual void Init() override;
};

struct BatchNormLayer : Layer
{
    Layer& input;
    float eps;
public:
    BatchNormLayer(Layer& input, float eps = 1e-3);
    virtual void Init() override;
};

struct SoftMaxLayer : Layer
{
    Layer& input;
    int axis;
public:
    SoftMaxLayer(Layer& input, int axis = -1);
    virtual void Init() override;
};

struct LogSoftMaxLayer : Layer
{
    Layer& input;
    int axis;
public:
    LogSoftMaxLayer(Layer& input, int axis = -1);
    virtual void Init() override;
};

struct MeanSquaredLoss : Loss
{
    MeanSquaredLoss(NNetwork& net);
    virtual void Init(dnnl::memory output, dnnl::memory answer, dnnl::memory grad);
};

struct CrossEntropyLoss : Loss
{
    CrossEntropyLoss(NNetwork& net);
    virtual void Init(dnnl::memory output, dnnl::memory answer, dnnl::memory grad);
};

struct NLLLoss : Loss
{
    NLLLoss(NNetwork& net);
    virtual void Init(dnnl::memory output, dnnl::memory answer, dnnl::memory grad);
};
