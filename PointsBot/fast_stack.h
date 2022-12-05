#pragma once
#include <algorithm>
#include <memory>

template<typename T, typename IndexType = short>
class FastStack
{
public:
	FastStack(IndexType capacity) : stk(new T[capacity]), capacity(capacity), size(0) {}
	FastStack(const FastStack<T, IndexType>& other) : stk(new T[other.capacity]), capacity(other.capacity), size(other.size)
	{
		std::copy_n(other.stk.get(), other.size, stk.get());
	}

	inline const T operator[](const int i) const { return stk[i]; }
	inline void clear() { size = 0; }
	inline bool empty() const { return !size; }
	inline void push(const T x) { stk[size++] = x; }
	template<class... Args>
	inline void emplace(Args&&... args) { stk[size++] = T(args...); }
	inline T top() const { return stk[size - 1]; }
	inline void pop() { --size; }
	inline void swap(FastStack& other)
	{
		std::swap(stk, other.stk);
		std::swap(capacity, other.capacity);
		std::swap(size, other.size);
	}
public:
	std::unique_ptr<T[]> stk;
	IndexType capacity, size;
};