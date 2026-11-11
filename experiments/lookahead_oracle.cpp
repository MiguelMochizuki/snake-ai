/**
 * lookahead_oracle.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Phase 0a of the beat-the-teacher design: an upper bound on what a safe player can save over the safe A*, using a lookahead that sees the real future foods
 */
#include "snake_ai/eval/paired.hpp"
#include "snake_ai/planning/lookahead.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

/**
 * Prints one arm's summary
 *
 * Parameters:
 * const char* name: Arm label
 * const std::vector<GameResult>& r: Results
 *
 * Returns double: Mean steps
 */
double report(const char* name, const std::vector<GameResult>& r) {
	double steps = 0;
	int died = 0, notFull = 0;
	for (const GameResult& g : r) {
		steps += g.steps / static_cast<double>(r.size());
		died += g.died;
		notFull += !g.full;
	}
	std::printf("%-26s mean steps %9.1f  deaths %d/%zu  not full %d/%zu\n", name, steps, died, r.size(), notFull, r.size());
	return steps;
}

}  // namespace

/**
 * Usage: lookahead_oracle <validate|test> [foods=3] [threads=12]
 * validate uses the seeds 20000..20019, test the seeds 100000..100029
 */
int main(int argc, char** argv) {
	if (argc < 2 || (std::strcmp(argv[1], "validate") != 0 && std::strcmp(argv[1], "test") != 0)) {
		std::fprintf(stderr, "usage: %s <validate|test> [foods=3] [threads=12]\n", argv[0]);
		return 2;
	}
	const bool validate = std::strcmp(argv[1], "validate") == 0;
	const int foods = argc > 2 ? std::atoi(argv[2]) : 3;
	const int threads = argc > 3 ? std::atoi(argv[3]) : 12;
	std::vector<unsigned> seeds;
	for (unsigned g = 0; g < (validate ? 20u : 30u); ++g) seeds.push_back((validate ? 20000u : 100000u) + g);

	std::printf("lookahead_oracle %s: foods K=%d, %zu seeds, 20x20 (the oracle sees the true future foods: upper bound only)\n", argv[1], foods, seeds.size());
	std::vector<GameResult> base = playGames(20, seeds, [](SnakeEnv& e, SafeAgent& a) { return teacherAction(snapshotOf(e), a); }, threads);
	std::vector<GameResult> look = playGames(20, seeds, [foods](SnakeEnv& e, SafeAgent& a) { return lookaheadLeaked(e, a, foods); }, threads);
	report("safe A* (teacher)", base);
	report("leaked lookahead", look);

	std::vector<double> b, t;
	for (size_t i = 0; i < seeds.size(); ++i) {
		b.push_back(base[i].steps);
		t.push_back(look[i].steps);
	}
	Interval iv = bootstrapRelDiff(b, t, 10000, 1);
	std::printf("relative change in steps %+.2f%%  95%% CI [%+.2f%%, %+.2f%%]\n", 100 * iv.mean, 100 * iv.lo, 100 * iv.hi);
	std::printf("GATE A: %s\n", passesGate(iv) ? "PASS" : "FAIL");
	return 0;
}
