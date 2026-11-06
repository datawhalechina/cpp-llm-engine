// Converts Chapter 07 benchmark output into a profiling report or A/B decision.
// Input is read as key=value lines so the tool can consume real script output without fake traces.
// Run the executable through scripts/run.sh or pipe a benchmark result into it.
#include "chapter08_profiling/ab_compare.h"

#include <fstream>
#include <iostream>

namespace {

chapter08_profiling::ProfileRecord read_file(const char* path) {
    std::ifstream input(path);
    return chapter08_profiling::parse_benchmark_output(input);
}

}  // namespace

int main(int argc, char** argv) {
    using namespace chapter08_profiling;
    if (argc == 1) {
        const ProfileRecord record = parse_benchmark_output(std::cin);
        std::string error;
        if (!record.valid(&error)) {
            std::cerr << "profile input invalid: " << error << '\n';
            return 1;
        }
        std::cout << render_report(record);
        return 0;
    }
    if (argc == 4 && std::string(argv[1]) == "--compare") {
        const ProfileRecord baseline = read_file(argv[2]);
        const ProfileRecord experiment = read_file(argv[3]);
        const ComparisonResult result = compare(baseline, experiment);
        std::cout << render_comparison(result);
        return result.accepted ? 0 : 1;
    }
    std::cerr << "usage: chapter08_profile_report [--compare baseline experiment]\n";
    return 2;
}
