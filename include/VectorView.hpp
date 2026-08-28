//==============================================================================
//     File: VectorView.h
//  Created: 2026-08-14 20:53
//   Author: Bernie Roesler
//
//  Description: Define the VectorView class.
//
//==============================================================================

#pragma once

#include "types.hpp"
#include "Vector.hpp"

#include <algorithm>        // transform
#include <concepts>         // integral, floating_point
#include <format>           // format
#include <functional>       // plus, etc.
#include <initializer_list>
#include <ranges>           // from_range_t, views::transform
#include <stdexcept>        // invalid_argument
#include <type_traits>      // is_same_v
#include <iterator>         // const_iterator

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
    explicit constexpr VectorView(std::span<T> s) : data_(s) {}

    constexpr VectorView(Vector<value_type>& v) : data_(v) {}
    constexpr VectorView(const Vector<value_type>& v)
        requires std::is_const_v<T>
        : data_(v) {}

    template <std::contiguous_iterator It>
    constexpr VectorView(It first, size_t count) : data_(first, count) {}

    template <std::contiguous_iterator It, std::sized_sentinel_for<It> End>
    constexpr VectorView(It first, End last) : data_(first, last) {}

    // NOTE the "requires" line prevents "hijacking" the copy constructor
    template <std::ranges::contiguous_range R>
    requires (!std::is_same_v<std::remove_cvref_t<R>, VectorView>) 
    constexpr VectorView(R&& range) : data_(std::forward<R>(range)) {}

    VectorView(const VectorView&) = default;
    VectorView(VectorView&&) = default;
    ~VectorView() = default;

    VectorView& operator=(const VectorView&) = default;
    VectorView& operator=(VectorView&&) = default;

    // -------------------------------------------------------------------------
    //         Iterators
    // -------------------------------------------------------------------------
    constexpr auto begin() noexcept { return data_.begin(); }
    constexpr auto end() noexcept { return data_.end(); }

    constexpr auto begin() const noexcept { return data_.begin(); }
    constexpr auto end() const noexcept { return data_.end(); }

    constexpr auto cbegin() const noexcept { return data_.cbegin(); }
    constexpr auto cend() const noexcept { return data_.cend(); }

    constexpr auto rbegin() noexcept { return data_.rbegin(); }
    constexpr auto rend() noexcept { return data_.rend(); }

    constexpr auto rbegin() const noexcept { return data_.rbegin(); }
    constexpr auto rend() const noexcept { return data_.rend(); }

    constexpr auto crend() const noexcept { return data_.crend(); }

    // -------------------------------------------------------------------------
    //         Element Access
    // -------------------------------------------------------------------------
    constexpr T& front() const { return data_.front(); }
    constexpr T& back() const { return data_.back(); }
    // T& at(size_t i) const { return data_.at(i); }  // C++26 only
    constexpr T& operator[](size_t i) const { return data_[i]; }
    constexpr T* data() const noexcept { return data_.data(); }

    // -------------------------------------------------------------------------
    //         Observers
    // -------------------------------------------------------------------------
    constexpr size_t size() const noexcept { return data_.size(); }
    constexpr bool empty() const noexcept { return data_.empty(); }

    // -------------------------------------------------------------------------
    //         Subviews
    // -------------------------------------------------------------------------
    constexpr VectorView first(size_t count) const { return VectorView(data_.first(count)); }
    constexpr VectorView last(size_t count) const { return VectorView(data_.last(count)); }
    constexpr VectorView subspan(size_t offset, size_t count = std::dynamic_extent) const {
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
    template <std::ranges::contiguous_range R>
    constexpr VectorView& operator+=(const R& rhs) { return apply_elementwise_(rhs, std::plus<>()); }

    template <std::ranges::contiguous_range R>
    constexpr VectorView& operator-=(const R& rhs) { return apply_elementwise_(rhs, std::minus<>()); }

    template <std::ranges::contiguous_range R>
    constexpr VectorView& operator*=(const R& rhs) { return apply_elementwise_(rhs, std::multiplies<>()); }

    template <std::ranges::contiguous_range R>
    constexpr VectorView& operator/=(const R& rhs) { return apply_elementwise_(rhs, std::divides<>()); }


private:
    std::span<T> data_;

    template <std::ranges::contiguous_range R>
    constexpr void check_same_size_(const R& rhs) const {
        if (size() != rhs.size()) {
            throw std::invalid_argument(
                std::format("Vector size mismatch: {} vs {}", size(), rhs.size())
            );
        }
    }

    template <std::ranges::contiguous_range R, typename BinaryOp>
    constexpr VectorView& apply_elementwise_(const R& rhs, BinaryOp op) {
        check_same_size_(rhs);
        std::ranges::transform(data_, rhs, begin(), op);
        return *this;
    }

};  // VectorView


}  // namespace cs

//==============================================================================
//==============================================================================
