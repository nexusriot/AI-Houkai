"""Tests for the retrieval-quality metrics and evaluation harness."""

from __future__ import annotations

import pytest

from ai_houkai.eval import (
    EvalCase,
    average_precision,
    dcg_at_k,
    evaluate,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from ai_houkai.memory_system import MemoryStore


class TestMetricFunctions:
    def test_recall_at_k(self):
        assert recall_at_k(["a", "b", "c"], ["b"], 2) == 1.0
        assert recall_at_k(["a", "b", "c"], ["c"], 2) == 0.0      # c is at rank 3
        assert recall_at_k(["a", "b"], ["a", "b"], 2) == 1.0
        assert recall_at_k(["a", "b"], ["a", "x"], 2) == 0.5

    def test_recall_empty_relevant_is_zero(self):
        assert recall_at_k(["a"], [], 5) == 0.0

    def test_precision_at_k(self):
        assert precision_at_k(["a", "b"], ["a"], 2) == 0.5
        assert precision_at_k(["a", "b"], ["a", "b"], 2) == 1.0
        assert precision_at_k([], ["a"], 2) == 0.0

    def test_reciprocal_rank(self):
        assert reciprocal_rank(["a", "b", "c"], ["b"]) == 0.5
        assert reciprocal_rank(["a", "b"], ["a"]) == 1.0
        assert reciprocal_rank(["a", "b"], ["x"]) == 0.0

    def test_average_precision(self):
        # relevant at ranks 1 and 3 → (1/1 + 2/3) / 2
        assert average_precision(["a", "x", "c"], ["a", "c"]) == pytest.approx((1.0 + 2 / 3) / 2)

    def test_ndcg_perfect_and_partial(self):
        assert ndcg_at_k(["b", "a"], ["b"], 2) == pytest.approx(1.0)  # relevant first
        partial = ndcg_at_k(["a", "b"], ["b"], 2)                     # relevant second
        assert 0.0 < partial < 1.0
        assert ndcg_at_k(["a"], [], 2) == 0.0

    def test_dcg_monotonic(self):
        assert dcg_at_k(["b"], ["b"], 1) > dcg_at_k(["a", "b"], ["b"], 2) * 0  # sanity
        assert dcg_at_k(["b", "a"], ["b"], 2) > dcg_at_k(["a", "b"], ["b"], 2)

    def test_metrics_bounded_with_duplicate_ids(self):
        # A duplicated relevant id must not push metrics above their [0,1] range.
        assert recall_at_k(["a", "a"], ["a"], 5) == 1.0
        assert ndcg_at_k(["a", "a"], ["a"], 5) == pytest.approx(1.0)
        assert average_precision(["a", "a"], ["a"]) == 1.0
        assert recall_at_k(["a", "a", "b"], ["a", "b"], 5) == 1.0


class TestEvaluateHarness:
    def _seed(self, store: MemoryStore):
        return {
            "deploy": store.remember("Deploy the service with make release",
                                     type="procedural", tags=["deploy"]).id,
            "test": store.remember("Run pytest with tmp_path isolation",
                                   type="procedural", tags=["testing"]).id,
            "db": store.remember("We use PostgreSQL for storage",
                                 type="semantic", tags=["db"]).id,
        }

    def test_evaluate_runs_and_aggregates(self, store: MemoryStore):
        ids = self._seed(store)
        cases = [
            EvalCase(query="how do we deploy", relevant_ids=[ids["deploy"]], k=3),
            EvalCase(query="test isolation", relevant_ids=[ids["test"]], k=3),
        ]
        res = evaluate(store, cases, default_mode="hybrid")
        assert res.n == 2
        assert 0.0 <= res.recall_at_k <= 1.0
        assert 0.0 <= res.mrr <= 1.0
        assert 0.0 <= res.ndcg_at_k <= 1.0
        assert len(res.per_case) == 2
        assert "recall@" in res.summary()

    def test_evaluate_is_read_only(self, store: MemoryStore):
        ids = self._seed(store)
        cases = [EvalCase(query="deploy service", relevant_ids=[ids["deploy"]])]
        evaluate(store, cases)
        # touch=False inside evaluate → access_count must stay 0
        for mid in ids.values():
            mem = store._get_by_id(mid)
            assert mem.access_count == 0

    def test_evaluate_forwards_recall_kwargs(self, store: MemoryStore):
        ids = self._seed(store)
        cases = [EvalCase(query="deploy", relevant_ids=[ids["deploy"]])]
        # fusion=rrf is a recall kwarg; should be accepted and not crash
        res = evaluate(store, cases, default_mode="hybrid", fusion="rrf")
        assert res.n == 1

    def test_default_k_and_mode_are_honored(self, store: MemoryStore):
        ids = self._seed(store)
        # EvalCase leaves k/mode unset → must fall back to the evaluate() defaults.
        cases = [EvalCase(query="deploy", relevant_ids=[ids["deploy"]])]
        res = evaluate(store, cases, default_k=3, default_mode="semantic")
        assert res.k == 3                      # labelled with the k actually used
        assert res.per_case[0]["k"] == 3
        assert "recall@3" in res.summary()

    def test_per_case_override_and_mixed_k_label(self, store: MemoryStore):
        ids = self._seed(store)
        cases = [
            EvalCase(query="deploy", relevant_ids=[ids["deploy"]], k=2),
            EvalCase(query="testing", relevant_ids=[ids["test"]], k=5),
        ]
        res = evaluate(store, cases, default_k=10)
        assert {c["k"] for c in res.per_case} == {2, 5}
        assert res.k == -1                     # mixed
        assert "mixed" in res.summary()


# Golden metric values, duplicated verbatim in go/internal/eval/eval_test.go
# (TestPortParityGoldenValues). The eval harness's whole purpose is measuring
# ranking changes, so the two ports disagreeing by even a rounding step would
# make a cross-port comparison meaningless. Any change to these numbers must be
# made in both files, deliberately.
PORT_PARITY_GOLDEN = [
    # (retrieved, relevant, k, recall, precision, rr, ap, ndcg)
    (["a", "x", "b", "y"], ["a", "b"], 4,
     1.000000, 0.500000, 1.000000, 0.833333, 0.919721),
    (["x", "y", "z"], ["a"], 3,
     0.000000, 0.000000, 0.000000, 0.000000, 0.000000),
    # A duplicated retrieved id is credited once, so nothing exceeds 1.0.
    (["a", "a", "b"], ["a", "b"], 3,
     1.000000, 1.000000, 1.000000, 0.833333, 0.919721),
    # k truncates recall but not RR/AP, which score the full ranking.
    (["b", "a"], ["a", "b"], 1,
     0.500000, 1.000000, 1.000000, 1.000000, 1.000000),
    ([], ["a"], 5,
     0.000000, 0.000000, 0.000000, 0.000000, 0.000000),
]


class TestPortParityGoldenValues:
    @pytest.mark.parametrize(
        "retrieved,relevant,k,want_recall,want_precision,want_rr,want_ap,want_ndcg",
        PORT_PARITY_GOLDEN)
    def test_matches_the_go_port(self, retrieved, relevant, k, want_recall,
                                 want_precision, want_rr, want_ap, want_ndcg):
        assert recall_at_k(retrieved, relevant, k) == pytest.approx(
            want_recall, abs=1e-6)
        assert precision_at_k(retrieved, relevant, k) == pytest.approx(
            want_precision, abs=1e-6)
        assert reciprocal_rank(retrieved, relevant) == pytest.approx(
            want_rr, abs=1e-6)
        assert average_precision(retrieved, relevant) == pytest.approx(
            want_ap, abs=1e-6)
        assert ndcg_at_k(retrieved, relevant, k) == pytest.approx(
            want_ndcg, abs=1e-6)
