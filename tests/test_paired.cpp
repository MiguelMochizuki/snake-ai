/**
 * test_paired.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Tests for the paired evaluation harness
 */

#include "doctest/doctest.h"
#include "snake_ai/eval/paired.hpp"
#include "snake_ai/planning/modes.hpp"

#include <stdexcept>

TEST_CASE("bootstrapRelDiff on a constant 10 percent drop") {
	std::vector<double> base(30, 100.0), test(30, 90.0);
	Interval iv = bootstrapRelDiff(base, test, 500, 1);
	CHECK(iv.mean == doctest::Approx(-0.10));
	CHECK(iv.lo == doctest::Approx(-0.10));
	CHECK(iv.hi == doctest::Approx(-0.10));
	CHECK(passesGate(iv));
}

TEST_CASE("bootstrapRelDiff on identical data does not pass the gate") {
	std::vector<double> base(30, 100.0);
	Interval iv = bootstrapRelDiff(base, base, 500, 1);
	CHECK(iv.mean == doctest::Approx(0.0));
	CHECK_FALSE(passesGate(iv));
}

TEST_CASE("a noisy 5 percent drop whose interval straddles zero fails the gate") {
	std::vector<double> base, test;
	for (int i = 0; i < 30; ++i) {
		double b = i % 2 ? 150.0 : 50.0;
		base.push_back(b);
		test.push_back(b + (i % 2 ? 20.0 : -30.0));  // paired differences of -30 and +20: mean -5, spread large
	}
	Interval iv = bootstrapRelDiff(base, test, 2000, 1);
	CHECK(iv.mean == doctest::Approx(-0.05));
	CHECK(iv.hi > 0.0);
	CHECK_FALSE(passesGate(iv));
}

TEST_CASE("playGames does not depend on the thread count") {
	auto teacher = [](SnakeEnv& env, SafeAgent& agent) { return teacherAction(snapshotOf(env), agent); };
	std::vector<unsigned> seeds{1, 2, 3, 4, 5, 6};
	std::vector<GameResult> one = playGames(10, seeds, teacher, 1), four = playGames(10, seeds, teacher, 4);
	REQUIRE(one.size() == seeds.size());
	for (size_t i = 0; i < seeds.size(); ++i) {
		CHECK(one[i].steps == four[i].steps);
		CHECK(one[i].full);
		CHECK_FALSE(one[i].died);
		CHECK(one[i].score == 97);
	}
}

TEST_CASE("parseInt accepts only a whole decimal integer at or above the minimum") {
	int v = -7;
	CHECK(parseInt("12", 1, v));
	CHECK(v == 12);
	CHECK(parseInt("1", 1, v));
	for (const char* bad : {"", " 3", "3 ", "3 4", "6x", "abc", "-1", "0", "1.5", "99999999999", "+"}) {
		CAPTURE(bad);
		v = -7;
		CHECK_FALSE(parseInt(bad, 1, v));
		CHECK(v == -7);  // a rejected value never touches the output
	}
	CHECK_FALSE(parseInt(nullptr, 1, v));
}

TEST_CASE("modes map to disjoint seed ranges and board sizes") {
	CHECK(isRunMode("validate"));
	CHECK(isRunMode("test"));
	CHECK(isRunMode("smoke"));
	CHECK_FALSE(isRunMode("valid"));
	CHECK_FALSE(isRunMode(""));
	CHECK_FALSE(isRunMode(nullptr));

	std::vector<unsigned> val = seedsFor("validate"), tst = seedsFor("test"), smk = seedsFor("smoke");
	REQUIRE(val.size() == 20);
	REQUIRE(tst.size() == 30);
	REQUIRE(smk.size() == 2);
	CHECK(val.front() == 20000u);
	CHECK(tst.front() == 100000u);
	CHECK(boardFor("validate") == 20);
	CHECK(boardFor("test") == 20);
	CHECK(boardFor("smoke") == 10);
	for (unsigned a : val) {
		for (unsigned b : tst) CHECK(a != b);
	}
	CHECK_THROWS_AS(seedsFor("nope"), std::invalid_argument);
	CHECK_THROWS_AS(boardFor("nope"), std::invalid_argument);
}
