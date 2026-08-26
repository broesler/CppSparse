//==============================================================================
//     File: Vector.h
//  Created: 2026-07-27 21:57
//   Author: Bernie Roesler
//
//  Description: A vector class that support math operations.
//
//==============================================================================

#pragma once

#include "types.hpp"

#include <algorithm>        // transform
#include <format>           // format
#include <functional>       // plus, etc.
#include <initializer_list>
#include <numeric>          // accumulate
#include <ranges>           // from_range_t, views::transform
#include <stdexcept>        // invalid_argument
#include <type_traits>      // is_arithmetic_v
#include <vector>

namespace cs {


template <Arithmetic T>
class Vector
{

public:
    // -------------------------------------------------------------------------
    //         STL Container Type Aliases
    // -------------------------------------------------------------------------
    using value_type = T;
    using size_type = size_t;
    using difference_type = ptrdiff_t;
    using reference = T&;
    using const_reference = const T&;
    using pointer = T*;
    using const_pointer = const T*;
    using iterator = typename std::vector<T>::iterator;
    using const_iterator = typename std::vector<T>::const_iterator;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    // -------------------------------------------------------------------------
    //         Constructors
    // -------------------------------------------------------------------------
    Vector() = default;
    explicit Vector(size_t count) : data_(count) {}
    Vector(size_t count, const T& value) : data_(count, value) {}
    Vector(std::initializer_list<T> init) : data_(init) {}
    Vector(const std::vector<T>& vec) : data_(vec) {}
    Vector(std::vector<T>&& vec) : data_(std::move(vec)) {}

    template <std::input_iterator It>
    Vector(It first, It last) : data_(first, last) {}

    // Usage: Vector<T> v(std::from_range, some_range);
    template <std::ranges::range R>
    explicit Vector(std::from_range_t, R&& r)
        : data_(std::ranges::begin(r), std::ranges::end(r)) {}

    void assign(size_t count, const T& value) { data_.assign(count, value); }

    template <std::input_iterator It>
    void assign(It first, It last) { data_.assign(first, last); }

    void assign(std::initializer_list<T> ilist) { data_.assign(ilist); }

    Vector& operator=(std::initializer_list<T> ilist) {
        data_ = ilist;
        return *this;
    }

    Vector(const Vector&) = default;
    Vector(Vector&&) = default;
    ~Vector() = default;

    Vector& operator=(const Vector&) = default;
    Vector& operator=(Vector&&) = default;

    // -------------------------------------------------------------------------
    //         Accessors
    // -------------------------------------------------------------------------
    T& at(size_t i) { return data_.at(i); }
    T& operator[](size_t i) { return data_[i]; }
    const T& at(size_t i) const { return data_.at(i); }
    const T& operator[](size_t i) const { return data_[i]; }

    T& front() { return data_.front(); }
    T& back() { return data_.back(); }
    T* data() { return data_.data(); }
    const T& front() const { return data_.front(); }
    const T& back() const { return data_.back(); }
    const T* data() const { return data_.data(); }

    bool empty() const noexcept { return data_.empty(); }
    size_t size() const noexcept { return data_.size(); }
    size_t max_size() const noexcept { return data_.max_size(); }
    void reserve(size_t new_cap) { data_.reserve(new_cap); }
    size_t capacity() const noexcept { return data_.capacity(); }
    void shrink_to_fit() { data_.shrink_to_fit(); }

    // -------------------------------------------------------------------------
    //         Iterators
    // -------------------------------------------------------------------------
    auto begin() noexcept { return data_.begin(); }
    auto end() noexcept { return data_.end(); }

    auto begin() const noexcept { return data_.begin(); }
    auto end() const noexcept { return data_.end(); }

    auto cbegin() const noexcept { return data_.cbegin(); }
    auto cend() const noexcept { return data_.cend(); }

    auto rbegin() noexcept { return data_.rbegin(); }
    auto rend() noexcept { return data_.rend(); }

    auto rbegin() const noexcept { return data_.rbegin(); }
    auto rend() const noexcept { return data_.rend(); }

    auto crbegin() const noexcept { return data_.crbegin(); }
    auto crend() const noexcept { return data_.crend(); }

    // -------------------------------------------------------------------------
    //         Modifiers
    // -------------------------------------------------------------------------
    void clear() noexcept { data_.clear(); }

    auto insert(const_iterator pos, const T& value) { return data_.insert(pos, value); }
    auto insert(const_iterator pos, T&& value) { return data_.insert(pos, std::move(value)); }
    auto insert(const_iterator pos, size_t count, const T& value) { return data_.insert(pos, count, value); }

    template <typename It>
    auto insert(const_iterator pos, It first, It last) {
        return data_.insert(pos, first, last);
    }

    auto insert(const_iterator pos, std::initializer_list<T> ilist) { return data_.insert(pos, ilist); }

    template <typename... Args>
    auto emplace(const_iterator pos, Args&&... args) {
        return data_.emplace(pos, std::forward<Args>(args)...);
    }

    auto erase(iterator pos) { return data_.erase(pos); }
    auto erase(const_iterator pos) { return data_.erase(pos); }
    auto erase(iterator first, iterator last) { return data_.erase(first, last); }
    auto erase(const_iterator first, const_iterator last) { return data_.erase(first, last); }

    void push_back(const T& value) { data_.push_back(value); }
    void push_back(T&& value) { data_.push_back(std::move(value)); }

    template <typename... Args>
    decltype(auto) emplace_back(Args&&... args) {
        data_.emplace_back(std::forward<Args>(args)...);
    }

    void pop_back() { data_.pop_back(); }

    void resize(size_t count) { data_.resize(count); }
    void resize(size_t count, const T& value) { data_.resize(count, value); }

    void swap(Vector& other) noexcept { data_.swap(other.data_); }
    friend void swap(Vector& lhs, Vector& rhs) noexcept { lhs.swap(rhs); }  // for std::swap

    // -------------------------------------------------------------------------
    //         Comparison Operators
    // -------------------------------------------------------------------------
    bool operator==(const Vector& rhs) const { return data_ == rhs.data_; }
    bool operator!=(const Vector& rhs) const { return !(*this == rhs); }

    // TODO
    // friend Vector<bool> operator==(const Vector& lhs, const Vector& rhs) {
    //     return lhs.compare_elementwise(rhs, std::equal_to<>{});
    // }

    // friend Vector<bool> operator!=(const Vector& lhs, const Vector& rhs) {
    //     return lhs.compare_elementwise(rhs, std::not_equal_to<>{});
    // }

    // friend Vector<bool> operator<(const Vector& lhs, const Vector& rhs) {
    //     return lhs.compare_elementwise(rhs, std::less<>{});
    // }

    // friend Vector<bool> operator<=(const Vector& lhs, const Vector& rhs) {
    //     return lhs.compare_elementwise(rhs, std::less_equal<>{});
    // }

    // friend Vector<bool> operator>(const Vector& lhs, const Vector& rhs) {
    //     return lhs.compare_elementwise(rhs, std::greater<>{});
    // }

    // friend Vector<bool> operator>=(const Vector& lhs, const Vector& rhs) {
    //     return lhs.compare_elementwise(rhs, std::greater_equal<>{});
    // }

    // -------------------------------------------------------------------------
    //         Assignment Operators
    // -------------------------------------------------------------------------
    // Vector-Vector
    Vector& operator+=(const Vector& rhs) { return apply_elementwise_(rhs, std::plus<>()); }

    template <std::ranges::contiguous_range R>
    Vector& operator-=(R&& rhs) {
        return apply_elementwise_(std::forward<R>(rhs), std::minus<>());
    }

    Vector& operator*=(const Vector& rhs) { return apply_elementwise_(rhs, std::multiplies<>()); }
    Vector& operator/=(const Vector& rhs) { return apply_elementwise_(rhs, std::divides<>()); }

    // Vector-scalar
    Vector& operator+=(T scalar) { return apply_scalar_(scalar, std::plus<>()); }
    Vector& operator-=(T scalar) { return apply_scalar_(scalar, std::minus<>()); }
    Vector& operator*=(T scalar) { return apply_scalar_(scalar, std::multiplies<>()); }
    Vector& operator/=(T scalar) { return apply_scalar_(scalar, std::divides<>()); }

    // Binary operators (hidden friends)
    // LHS is passed by value, mutated, and returned
    // Vector-vector
    friend Vector operator+(Vector lhs, const Vector& rhs) { return lhs += rhs; }
    friend Vector operator-(Vector lhs, const Vector& rhs) { return lhs -= rhs; }
    friend Vector operator*(Vector lhs, const Vector& rhs) { return lhs *= rhs; }
    friend Vector operator/(Vector lhs, const Vector& rhs) { return lhs /= rhs; }

    // Vector-scalar
    friend Vector operator+(Vector lhs, T scalar) { return lhs += scalar; }
    friend Vector operator-(Vector lhs, T scalar) { return lhs -= scalar; }
    friend Vector operator*(Vector lhs, T scalar) { return lhs *= scalar; }
    friend Vector operator/(Vector lhs, T scalar) { return lhs /= scalar; }

    // Scalar-vector
    // Commutative
    friend Vector operator+(T scalar, Vector rhs) { return rhs += scalar; }
    friend Vector operator*(T scalar, Vector rhs) { return rhs *= scalar; }

    // Non-commutative
    friend Vector operator-(T scalar, const Vector& rhs) {
        Vector result(rhs.size());
        std::ranges::transform(
            rhs, result.begin(), [scalar](T x) { return scalar - x; }
        );
        return result;
    }

    friend Vector operator/(T scalar, const Vector& rhs) {
        Vector result(rhs.size());
        std::ranges::transform(
            rhs, result.begin(), [scalar](T x) { return scalar / x; }
        );
        return result;
    }

    // Unary operators
    Vector operator+() const { return *this; }
    Vector operator-() const { return (*this) * T(-1); }

    // -------------------------------------------------------------------------
    //         Methods
    // -------------------------------------------------------------------------
    T min() const {
        check_empty_();
        return *std::ranges::min_element(data_);
    }

    T max() const {
        check_empty_();
        return *std::ranges::max_element(data_);
    }

    T sum() const {
        check_empty_();
        return std::accumulate(begin(), end(), T(0));
    }

    T mean() const {
        check_empty_();
        return sum() / static_cast<T>(size());
    }

private:
    std::vector<T> data_;

    void check_empty_() const {
        if (empty()) {
            throw std::runtime_error("Vector is empty");
        }
    }

    template <std::ranges::contiguous_range R>
    void check_same_size_(R&& rhs) const {
        if (size() != rhs.size()) {
            throw std::invalid_argument(
                std::format("Vector size mismatch: {} vs {}", size(), rhs.size())
            );
        }
    }

    template <typename BinaryOp>
    Vector& apply_scalar_(T scalar, BinaryOp op) {
        std::ranges::transform(
            data_, begin(), [scalar, op](T x) { return op(x, scalar); }
        );
        return *this;
    }

    template <std::ranges::contiguous_range R, typename BinaryOp>
    Vector& apply_elementwise_(R&& rhs, BinaryOp op) {
        check_same_size_(rhs);
        std::ranges::transform(data_, rhs, begin(), op);
        return *this;
    }

    // TODO
    // template <typename BinaryOp>
    // Vector<bool> compare_elementwise(const Vector& rhs, BinaryOp op) const {
    //     check_same_size(rhs);
    //     Vector<bool> result(size());
    //     std::ranges::transform(data_, rhs, result.begin(), op);
    //     return result;
    // }
};  // class Vector


}  // namespace cs

//==============================================================================
//==============================================================================
