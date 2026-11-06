// Chapter 09：声明重复测量样本的基础统计接口。
// 职责：保留样本数量、范围、均值和中位数，不负责判断实验是否可比。
// 验证：chapter09_statistics_test 覆盖空样本、排序无关性和奇偶样本数。
#pragma once

#include <cstddef>
#include <vector>

namespace chapter09_performance {

struct SampleStatistics {
    std::size_t count = 0;
    double minimum = 0.0;
    double maximum = 0.0;
    double mean = 0.0;
    double median = 0.0;
};

SampleStatistics summarize(const std::vector<double>& samples);

}  // namespace chapter09_performance
