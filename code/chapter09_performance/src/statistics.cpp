// Chapter 09：实现与输入顺序无关的基础统计聚合。
// 契约：空输入返回 count=0；非空输入使用排序副本计算中位数。
// 验证：chapter09_statistics_test 核对均值、范围和中位数。
#include "chapter09_performance/statistics.h"

#include <algorithm>
namespace chapter09_performance {

SampleStatistics summarize(const std::vector<double>& samples) {
    SampleStatistics result;
    if (samples.empty()) return result;

    std::vector<double> ordered = samples;
    std::sort(ordered.begin(), ordered.end());
    result.count = ordered.size();
    result.minimum = ordered.front();
    result.maximum = ordered.back();
    long double sum = 0.0L;
    for (const double sample : ordered) sum += static_cast<long double>(sample);
    result.mean = static_cast<double>(sum / static_cast<long double>(ordered.size()));
    const std::size_t middle = ordered.size() / 2;
    result.median = ordered.size() % 2 == 0
                        ? ordered[middle - 1] / 2.0 + ordered[middle] / 2.0
                        : ordered[middle];
    return result;
}

}  // namespace chapter09_performance
