/**
 * lookahead_teacher.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Phase 0b of the beat-the-teacher design: the honest lookahead, which scores moves on sampled futures and never sees the real foods to come
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
 * Usage: lookahead_teacher <validate|test> [foods=6] [samples=8] [threads=12]
 * validate uses the seeds 20000..20019, test the seeds 100000..100029. The sampled futures use a fixed base seed, so results do not depend on the thread count
 */
int main(int argc, char** argv) {
	// Line-buffered, so a redirected log shows the parsed arguments and each finished arm while a long run is still going
	std::setvbuf(stdout, nullptr, _IOLBF, 0);
	// Exactly <mode> [foods] [samples] [threads], each a whole positive integer; anything else is refused instead of guessed
	LookaheadConfig cfg;
	cfg.foods = 6;
	cfg.samples = 8;
	int threads = 12;
	const bool modeOk = argc >= 2 && (std::strcmp(argv[1], "validate") == 0 || std::strcmp(argv[1], "test") == 0);
	if (!modeOk || argc > 5 || (argc > 2 && !parseInt(argv[2], 1, cfg.foods)) || (argc > 3 && !parseInt(argv[3], 1, cfg.samples))
		|| (argc > 4 && !parseInt(argv[4], 1, threads))) {
		std::fprintf(stderr, "usage: %s <validate|test> [foods=6] [samples=8] [threads=12]  (integers >= 1, nothing else)\n", argv[0]);
		return 2;
	}
	const bool validate = std::strcmp(argv[1], "validate") == 0;
	std::vector<unsigned> seeds;
	for (unsigned g = 0; g < (validate ? 20u : 30u); ++g) seeds.push_back((validate ? 20000u : 100000u) + g);

	std::printf("lookahead_teacher %s: foods K=%d, samples M=%d, threads %d, %zu seeds, 20x20 (futures are sampled; the real ones are never seen)\n", argv[1], cfg.foods, cfg.samples, threads, seeds.size());
	std::vector<GameResult> base = playGames(20, seeds, [](SnakeEnv& e, SafeAgent& a) { return teacherAction(snapshotOf(e), a); }, threads);
	std::vector<GameResult> look = playGames(20, seeds, [cfg](SnakeEnv& e, SafeAgent& a) { return lookaheadHonest(snapshotOf(e), a, cfg); }, threads);
	report("safe A* (teacher)", base);
	report("honest lookahead", look);

	std::vector<double> b, t;
	for (size_t i = 0; i < seeds.size(); ++i) {
		b.push_back(base[i].steps);
		t.push_back(look[i].steps);
	}
	Interval iv = bootstrapRelDiff(b, t, 10000, 1);
	std::printf("relative change in steps %+.2f%%  95%% CI [%+.2f%%, %+.2f%%]\n", 100 * iv.mean, 100 * iv.lo, 100 * iv.hi);
	std::printf("GATE B: %s\n", passesGate(iv) ? "PASS" : "FAIL");
	return 0;
}
