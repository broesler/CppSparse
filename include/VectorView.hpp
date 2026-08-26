//==============================================================================
//     File: VectorView.h
//  Created: 2026-08-14 20:53
//   Author: Bernie Roesler
//
//  Description: Define the VectorView class.
//
//==============================================================================

#pragma once

#include <algorithm>        // transform
#include <concepts>         // integral, floating_point
#include <format>           // format
#include <functional>       // plus, etc.
#include <initializer_list>
#include <ranges>           // from_range_t, views::transform
#include <stdexcept>        // invalid_argument
#include <type_traits>      // is_same_v
#include <iterator>         // const_iterator

#include "Vector.hpp"


namespace cs {


/// A class that provides a contiguous view onto a Vector.
template <Arithmetic T>
class VectorView
{
public:

    // -------------------------------------------------------------------------
    //         STL Container Type Aliases
    // -------------------------------------------------------------------------
    using element_type = T;
    using value_type = std::remove_cv_t<T>;
    using size_type = size_t;
    using difference_type = ptrdiff_t;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using iterator = typename std::span<T>::iterator;
    using const_iterator = typename std::span<std::add_const_t<T>>::iterator;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    // -------------------------------------------------------------------------
    //         Constructors
    // -------------------------------------------------------------------------
    VectorView() = default;
    explicit VectorView(std::span<T> s) : data_(s) {}

    VectorView(Vector<value_type>& v) : data_(v) {}
    VectorView(const Vector<value_type>& v)
        requires std::is_const_v<T>
        : data_(v) {}

    template <std::contiguous_iterator It>
    VectorView(It first, size_t count) : data_(first, count) {}

    template <std::contiguous_iterator It, std::sized_sentinel_for<It> End>
    VectorView(It first, End last) : data_(first, last) {}

    // NOTE the "requires" line prevents "hijacking" the copy constructor
    template <std::ranges::contiguous_range R>
    requires (!std::is_same_v<std::remove_cvref_t<R>, VectorView>) 
    VectorView(R&& range) : data_(std::forward<R>(range)) {}

    VectorView(const VectorView&) = default;
    VectorView(VectorView&&) = default;
    ~VectorView() = default;

    VectorView& operator=(const VectorView&) = default;
    VectorView& operator=(VectorView&&) = default;

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

    auto crend() const noexcept { return data_.crend(); }

    // -------------------------------------------------------------------------
    //         Element Access
    // -------------------------------------------------------------------------
    T& front() const { return data_.front(); }
    T& back() const { return data_.back(); }
    // T& at(size_t i) const { return data_.at(i); }  // C++26 only
    T& operator[](size_t i) const { return data_[i]; }
    T* data() const noexcept { return data_.data(); }

    // -------------------------------------------------------------------------
    //         Observers
    // -------------------------------------------------------------------------
    size_t size() const noexcept { return data_.size(); }
    bool empty() const noexcept { return data_.empty(); }

    // -------------------------------------------------------------------------
    //         Subviews
    // -------------------------------------------------------------------------
    VectorView first(size_t count) const { return VectorView(data_.first(count)); }
    VectorView last(size_t count) const { return VectorView(data_.last(count)); }
    VectorView subspan(size_t offset, size_t count = std::dynamic_extent) const {
        return VectorView(data_.subspan(offset, count));
    }

    // -------------------------------------------------------------------------
    //         Comparison Operators
    // -------------------------------------------------------------------------
    constexpr bool operator==(const VectorView& rhs) const {
        return std::ranges::equal((*this), rhs);
    }

    // -------------------------------------------------------------------------
    //         Operators
    // -------------------------------------------------------------------------
    // Vector-Vector
    VectorView& operator+=(VectorView<const T> rhs) { return apply_elementwise_(rhs, std::plus<>()); }
    VectorView& operator-=(VectorView<const T> rhs) { return apply_elementwise_(rhs, std::minus<>()); }
    VectorView& operator*=(VectorView<const T> rhs) { return apply_elementwise_(rhs, std::multiplies<>()); }
    VectorView& operator/=(VectorView<const T> rhs) { return apply_elementwise_(rhs, std::divides<>()); }

private:
    std::span<T> data_;

    void check_same_size_(VectorView<const T> rhs) const {
        if (size() != rhs.size()) {
            throw std::invalid_argument(
                std::format("Vector size mismatch: {} vs {}", size(), rhs.size())
            );
        }
    }

    template <typename BinaryOp>
    VectorView& apply_elementwise_(VectorView<const T> rhs, BinaryOp op) {
        check_same_size_(rhs);
        std::ranges::transform((*this), rhs, begin(), op);
        return *this;
    }

};  // VectorView


using VectorViewD = VectorView<double>;
using cVectorViewD = VectorView<const double>;


// -----------------------------------------------------------------------------
//         Operators
// -----------------------------------------------------------------------------
template <typename T>
Vector<T> operator+(VectorView<const T> a, VectorView<const T> b)
{
    Vector<T> x(a.begin(), a.end());
    x += b;
    return x;
}


template <typename T>
Vector<T> operator-(VectorView<const T> a, VectorView<const T> b)
{
    Vector<T> x(a.begin(), a.end());
    x -= b;
    return x;
};


template <typename T>
Vector<T> operator-(VectorView<const T> a)
{
    Vector<T> x(a.begin(), a.end());
    return -x;
}



}  // namespace cs

//==============================================================================
//==============================================================================
