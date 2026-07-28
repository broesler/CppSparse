//==============================================================================
//     File: Vector.h
//  Created: 2026-07-27 21:57
//   Author: Bernie Roesler
//
//  Description: A vector class that support math operations.
//
//==============================================================================

#pragma once

#include <algorithm>        // transform
#include <functional>       // plus, etc.
#include <initializer_list>
#include <stdexcept>        // invalid_argument
#include <type_traits>      // is_arithmetic_v
#include <vector>

#include "types.h"

// TODO 
// in types.h:
// template <typename T>
// concept Arithmetic = std::integral<T> || std::floating_point<T>;
// then, template <Arithmetic T> class Vector { ... } instead of requires
// or can do same for CSCMatrix

namespace cs {

template <typename T>
requires std::is_arithmetic_v<T>
class Vector
{

public:
    // -------------------------------------------------------------------------
    //         Constructors
    // -------------------------------------------------------------------------
    Vector() = default;
    explicit Vector(size_t count, const T& value = T()) : data_(count, value) {}
    Vector(std::initializer_list<T> init) : data_(init) {}
    Vector(const std::vector<T>& vec) : data_(vec) {}
    Vector(std::vector<T> vec) : data_(std::move(vec)) {}

    // -------------------------------------------------------------------------
    //         Accessors
    // -------------------------------------------------------------------------
    T& at(size_t i) noexcept { return data_.at(i); }
    T& operator[](size_t i) noexcept { return data_[i]; }
    const T& at(size_t i) const noexcept { return data_.at(i); }
    const T& operator[](size_t i) const noexcept { return data_[i]; }

    T& front() noexcept { return data_.front(); }
    T& back() noexcept { return data_.back(); }
    T* data() noexcept { return data_.data(); }
    const T& front() const noexcept { return data_.front(); }
    const T& back() const noexcept { return data_.back(); }
    const T* data() const noexcept { return data_.data(); }

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

    using iterator = typename std::vector<T>::iterator;
    using const_iterator = typename std::vector<T>::const_iterator;

    auto insert(const_iterator pos, const T& value) { return data_.insert(pos, value); }
    auto insert(const_iterator pos, T&& value) { return data_.insert(pos, std::move(value)); }
    auto insert(const_iterator pos, size_t count, const T& value) { return data_.insert(pos, count, value); }

    template <typename InputIt>
    auto insert(const_iterator pos, InputIt first, InputIt last) { return data_.insert(pos, first, last); }

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
    // Vector-vector
    Vector& operator+=(const Vector& rhs) { return apply_elementwise(rhs, std::plus<>()); }
    Vector& operator-=(const Vector& rhs) { return apply_elementwise(rhs, std::minus<>()); }
    Vector& operator*=(const Vector& rhs) { return apply_elementwise(rhs, std::multiplies<>()); }
    Vector& operator/=(const Vector& rhs) { return apply_elementwise(rhs, std::divides<>()); }

    // Vector-scalar
    Vector& operator+=(T scalar) { return apply_scalar(scalar, std::plus<>()); }
    Vector& operator-=(T scalar) { return apply_scalar(scalar, std::minus<>()); }
    Vector& operator*=(T scalar) { return apply_scalar(scalar, std::multiplies<>()); }
    Vector& operator/=(T scalar) { return apply_scalar(scalar, std::divides<>()); }

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
        std::transform(
            rhs.data_,
            result.begin(),
            [scalar](T x) { return scalar - x; }
        );
        return result;
    }

    friend Vector operator/(T scalar, const Vector& rhs) {
        Vector result(rhs.size());
        std::transform(
            rhs.data_,
            result.begin(),
            [scalar](T x) { return scalar / x; }
        );
        return result;
    }

    // Unary operators
    Vector operator+() const { return *this; }
    Vector operator-() const { return (*this) * T(-1); }

private:
    std::vector<T> data_;

    void check_same_size(const Vector& rhs) const {
        if (size() != rhs.size()) {
            throw std::invalid_argument(
                std::format("Vector size mismatch: {} vs {}", size(), rhs.size())
            );
        }
    }

    template <typename BinaryOp>
    Vector& apply_scalar(T scalar, BinaryOp op) {
        std::ranges::transform(
            data_, begin(), [scalar, op](T x) { return op(x, scalar); }
        );
        return *this;
    }

    template <typename BinaryOp>
    Vector& apply_elementwise(const Vector& rhs, BinaryOp op) {
        check_same_size(rhs);
        std::ranges::transform(data_, rhs.data_, begin(), op);
        return *this;
    }

    // TODO
    // template <typename BinaryOp>
    // Vector<bool> compare_elementwise(const Vector& rhs, BinaryOp op) const {
    //     check_same_size(rhs);
    //     Vector<bool> result(size());
    //     std::ranges::transform(data_, rhs.data_, result.begin(), op);
    //     return result;
    // }
};  // class Vector

}  // namespace cs

//==============================================================================
//==============================================================================
