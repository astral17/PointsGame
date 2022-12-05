#pragma once
#include <algorithm>
#include <memory>

template<typename T, typename IndexType = short>
class FastQueue
{
public:
	FastQueue(IndexType size) : q(new T[size]), size(size), l(0), r(0) {}
	FastQueue(const FastQueue& other) : q(new T[other.size]), size(other.size), l(other.l), r(other.r)
	{
		std::copy_n(other.q.get(), other.r, q.get());
	}

	inline const T operator[](const int i) const { return q[i]; }
	inline void clear() { l = r = 0; }
	inline bool empty() const { return l == r; }
	inline void push(const T x) { q[r++] = x; }
	template<class... Args>
	inline void emplace(Args&&... args) { q[r++] = T(args...); }
	inline T front() const { return q[l]; }
	inline void pop() { l++; }
public:
	std::unique_ptr<T[]> q;
	IndexType size, l, r;
};