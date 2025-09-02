#pragma once

#include <rclcpp/rclcpp.hpp>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <iterator>
#include <algorithm>
#include <string>

namespace ros_pbar {

// -------------------- Chronometer --------------------
class Chronometer {
public:
    Chronometer() : start_(std::chrono::steady_clock::now()) {}

    double reset() {
        auto previous = start_;
        start_ = std::chrono::steady_clock::now();
        return elapsed_seconds(previous, start_);
    }

    double peek() const {
        auto now = std::chrono::steady_clock::now();
        return elapsed_seconds(start_, now);
    }

private:
    using time_point_t = std::chrono::time_point<std::chrono::steady_clock>;

    static double elapsed_seconds(time_point_t from, time_point_t to) {
        using seconds = std::chrono::duration<double>;
        return std::chrono::duration_cast<seconds>(to - from).count();
    }

    time_point_t start_;
};

// -------------------- ROS2 Progress Bar --------------------
class ProgressBar {
public:
    explicit ProgressBar(rclcpp::Logger logger)
        : logger_(logger)
    {}

    void update(double progress) {
        progress = std::clamp(progress, 0.0, 1.0);

        double percent = progress * 100.0;
        int bar_width = 40;
        int filled = static_cast<int>(std::round(bar_width * progress));

        std::stringstream bar;
        if (!prefix_.empty()) {
            bar << prefix_ << " ";
        }

        bar << std::fixed << std::setprecision(1)
            << percent << "% ["
            << std::string(filled, '#')
            << std::string(bar_width - filled, ' ')
            << "]";

        if (!suffix_.empty()) {
            bar << " " << suffix_;
        }

        // Log to ROS2
        RCLCPP_INFO(logger_, "%s", bar.str().c_str());
    }

    void set_prefix(const std::string &s) { prefix_ = s; }
    void set_suffix(const std::string &s) { suffix_ = s; }

private:
    rclcpp::Logger logger_;
    std::string prefix_;
    std::string suffix_;
};

// -------------------- Iterator Wrapper --------------------
template <typename ForwardIter, typename Parent>
class IterWrapper {
public:
    IterWrapper(ForwardIter it, Parent* parent)
        : current_(it), parent_(parent)
    {}

    auto operator*() { return *current_; }

    void operator++() { ++current_; }

    template <typename Other>
    bool operator!=(const Other& other) const {
        parent_->update();
        return current_ != other;
    }

    bool operator!=(const IterWrapper<ForwardIter, Parent>& other) const {
        parent_->update();
        return current_ != other.current_;
    }

private:
    ForwardIter current_;
    Parent* parent_;
};

// -------------------- ros_pbar_for_lvalues --------------------
template <typename ForwardIter, typename EndIter = ForwardIter>
class ros_pbar_for_lvalues {
public:
    using iterator = IterWrapper<ForwardIter, ros_pbar_for_lvalues>;
    using index = std::ptrdiff_t;

    ros_pbar_for_lvalues(ForwardIter begin, EndIter end, rclcpp::Logger logger)
        : first_(begin, this), last_(end), num_iters_(std::distance(begin, end)), bar_(logger)
    {}

    template <typename Container>
    explicit ros_pbar_for_lvalues(Container& c, rclcpp::Logger logger)
        : ros_pbar_for_lvalues(c.begin(), c.end(), logger)
    {}

    iterator begin() {
        iters_done_ = 0;
        return first_;
    }

    EndIter end() const { return last_; }

    void update() {
        ++iters_done_;
        double progress = static_cast<double>(iters_done_) / (num_iters_ + 1e-12);
        bar_.update(progress);
    }

    void set_suffix(const std::string& s) { bar_.set_suffix(s); }
    void set_prefix(const std::string& s) { bar_.set_prefix(s); }

private:
    iterator first_;
    EndIter last_;
    index num_iters_;
    index iters_done_{0};
    ProgressBar bar_;
};

// -------------------- ros_pbar helper --------------------
template <typename Container>
auto ros_pbar(Container& c, rclcpp::Logger logger) {
    return ros_pbar_for_lvalues<typename Container::iterator>(c, logger);
}

template <typename ForwardIter>
auto ros_pbar(ForwardIter begin, ForwardIter end, rclcpp::Logger logger) {
    return ros_pbar_for_lvalues<ForwardIter>(begin, end, logger);
}

} // namespace ros_pbar
