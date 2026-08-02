"""The eval harness reaches the CLI and MCP (F2).

`ai_houkai.eval` scored rankings for months but was reachable from no surface,
so every ranking constant in the project — graph damping, the β=0.20 lexical
weight, the MMR defaults — was set by intuition with no way to measure a
change. These tests pin the two surfaces that make it usable.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

import ai_houkai.mcp_server.server as srv
from ai_houkai.cli.commands.eval_cmd import load_goldset
from ai_houkai.cli.main import app


@pytest.fixture()
def mcp_store(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_HOUKAI_PATH", str(tmp_path / "chroma"))
    monkeypatch.setenv("AI_HOUKAI_COLLECTION", "eval_mcp")
    monkeypatch.setattr(srv, "_store", None)
    yield
    if srv._store is not None:
        srv._store.client.close()
        srv._store = None


class TestGoldsetParsing:
    def test_parses_cases_and_skips_noise(self, tmp_path):
        path = tmp_path / "gold.jsonl"
        path.write_text(
            "# a comment\n"
            '{"query": "alpha", "relevant_ids": ["a"]}\n'
            "\n"
            '{"query": "beta", "relevant_ids": ["b", "c"], "k": 3, "mode": "semantic"}\n'
        )
        cases = load_goldset(path)
        assert len(cases) == 2
        assert cases[0].query == "alpha" and cases[0].relevant_ids == ["a"]
        assert cases[0].k is None  # falls back to the CLI default
        assert cases[1].k == 3 and cases[1].mode == "semantic"

    def test_reports_the_offending_line(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text('{"query": "ok", "relevant_ids": ["a"]}\nnot json\n')
        with pytest.raises(ValueError, match=r"bad\.jsonl:2: invalid JSON"):
            load_goldset(path)

    @pytest.mark.parametrize("line,match", [
        ('{"relevant_ids": ["a"]}', "missing 'query'"),
        ('{"query": "q"}', "non-empty list"),
        ('{"query": "q", "relevant_ids": []}', "non-empty list"),
        ('["not", "an", "object"]', "expected a JSON object"),
    ])
    def test_rejects_malformed_cases(self, tmp_path, line, match):
        path = tmp_path / "bad.jsonl"
        path.write_text(line + "\n")
        with pytest.raises(ValueError, match=match):
            load_goldset(path)

    def test_empty_goldset_is_an_error(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("# only comments\n\n")
        with pytest.raises(ValueError, match="no evaluation cases"):
            load_goldset(path)

    def test_resolves_id_prefixes(self, tmp_path, fake_store):
        """A gold set written by eye from `houkai list` uses 8-char prefixes."""
        mem = fake_store.remember("prefix resolution subject")
        path = tmp_path / "gold.jsonl"
        path.write_text(json.dumps(
            {"query": "prefix", "relevant_ids": [mem.id[:8]]}) + "\n")
        cases = load_goldset(path, fake_store)
        assert cases[0].relevant_ids == [mem.id]

    def test_unresolvable_id_is_an_error_not_a_zero_score(self, tmp_path, fake_store):
        """A typo must not masquerade as a ranking regression."""
        fake_store.remember("anchor")
        path = tmp_path / "gold.jsonl"
        path.write_text('{"query": "q", "relevant_ids": ["deadbeef"]}\n')
        with pytest.raises(ValueError, match="No memory with id prefix"):
            load_goldset(path, fake_store)


class TestEvalCli:
    def _run(self, tmp_path, *args):
        return CliRunner().invoke(app, ["--store", str(tmp_path / "chroma"), *args])

    def _seed(self, tmp_path):
        self._run(tmp_path, "remember", "the deploy runbook lives in ops/deploy.md")
        self._run(tmp_path, "remember", "postgres vacuum runs nightly")
        listing = self._run(tmp_path, "list", "--format", "json")
        return {m["text"]: m["id"] for m in json.loads(listing.stdout)}

    def test_perfect_goldset_scores_one(self, tmp_path):
        ids = self._seed(tmp_path)
        target = ids["the deploy runbook lives in ops/deploy.md"]
        gold = tmp_path / "gold.jsonl"
        gold.write_text(json.dumps({
            "query": "the deploy runbook lives in ops/deploy.md",
            "relevant_ids": [target],
        }) + "\n")

        res = self._run(tmp_path, "eval", str(gold), "--json")
        assert res.exit_code == 0, res.stdout
        out = json.loads(res.stdout)
        assert out["n"] == 1
        assert out["recall_at_k"] == 1.0
        assert out["mrr"] == 1.0
        assert out["ndcg_at_k"] == 1.0

    def test_wrong_goldset_scores_zero(self, tmp_path):
        self._seed(tmp_path)
        gold = tmp_path / "gold.jsonl"
        # A syntactically valid id that is not in the store: nothing can match.
        gold.write_text(json.dumps({
            "query": "the deploy runbook",
            "relevant_ids": ["00000000-0000-0000-0000-000000000000"],
        }) + "\n")
        out = json.loads(self._run(tmp_path, "eval", str(gold), "--json").stdout)
        assert out["recall_at_k"] == 0.0
        assert out["mrr"] == 0.0

    def test_records_the_config_under_test(self, tmp_path):
        """Without this the numbers are unattributable across A/B runs."""
        ids = self._seed(tmp_path)
        gold = tmp_path / "gold.jsonl"
        gold.write_text(json.dumps({
            "query": "postgres vacuum runs nightly",
            "relevant_ids": [ids["postgres vacuum runs nightly"]],
        }) + "\n")
        out = json.loads(self._run(
            tmp_path, "eval", str(gold), "--json",
            "--mode", "hybrid", "--fusion", "rrf", "--graph", "0.15",
            "--diversity", "0.7", "--expand-rerank").stdout)
        cfg = out["config"]
        assert cfg["mode"] == "hybrid" and cfg["fusion"] == "rrf"
        assert cfg["graph"] == 0.15 and cfg["diversity"] == 0.7
        assert cfg["expand_rerank"] is True

    def test_eval_is_read_only(self, tmp_path):
        """Evaluating twice must not change what is being evaluated."""
        ids = self._seed(tmp_path)
        target = ids["postgres vacuum runs nightly"]
        gold = tmp_path / "gold.jsonl"
        gold.write_text(json.dumps({
            "query": "postgres vacuum runs nightly", "relevant_ids": [target],
        }) + "\n")
        self._run(tmp_path, "eval", str(gold), "--json")
        shown = self._run(tmp_path, "show", target)
        assert "access_count" not in shown.stdout or "0" in shown.stdout

        first = json.loads(self._run(tmp_path, "eval", str(gold), "--json").stdout)
        second = json.loads(self._run(tmp_path, "eval", str(gold), "--json").stdout)
        assert first["recall_at_k"] == second["recall_at_k"]

    def test_per_case_rows(self, tmp_path):
        ids = self._seed(tmp_path)
        gold = tmp_path / "gold.jsonl"
        gold.write_text(json.dumps({
            "query": "postgres vacuum runs nightly",
            "relevant_ids": [ids["postgres vacuum runs nightly"]],
        }) + "\n")
        out = json.loads(self._run(
            tmp_path, "eval", str(gold), "--json", "--per-case").stdout)
        assert len(out["per_case"]) == 1
        assert out["per_case"][0]["query"] == "postgres vacuum runs nightly"
        assert "retrieved" in out["per_case"][0]

    def test_missing_goldset_is_a_clean_error(self, tmp_path):
        self._seed(tmp_path)
        res = self._run(tmp_path, "eval", str(tmp_path / "absent.jsonl"))
        assert res.exit_code == 1
        assert "Traceback" not in (res.stderr or "")


class TestEvalRecallMcpTool:
    def test_scores_a_gold_set(self, mcp_store):
        a = srv.remember(text="the deploy runbook lives in ops/deploy.md")
        srv.remember(text="postgres vacuum runs nightly")
        out = srv.eval_recall(cases=[{
            "query": "the deploy runbook lives in ops/deploy.md",
            "relevant_ids": [a["id"]],
        }])
        assert out["n"] == 1
        assert out["recall_at_k"] == 1.0
        assert out["mrr"] == 1.0
        assert out["per_case"][0]["retrieved"][0] == a["id"]

    def test_reports_the_k_actually_used(self, mcp_store):
        a = srv.remember(text="mixed k subject one")
        b = srv.remember(text="mixed k subject two")
        uniform = srv.eval_recall(cases=[
            {"query": "mixed k subject one", "relevant_ids": [a["id"]]},
            {"query": "mixed k subject two", "relevant_ids": [b["id"]]},
        ], k=4)
        assert uniform["k"] == 4
        mixed = srv.eval_recall(cases=[
            {"query": "mixed k subject one", "relevant_ids": [a["id"]], "k": 2},
            {"query": "mixed k subject two", "relevant_ids": [b["id"]], "k": 5},
        ])
        assert mixed["k"] == -1

    def test_is_read_only(self, mcp_store):
        created = srv.remember(text="mcp eval read-only subject")
        srv.eval_recall(cases=[{"query": "mcp eval read-only subject",
                                "relevant_ids": [created["id"]]}])
        assert srv.get(memory_id=created["id"])["access_count"] == 0

    @pytest.mark.parametrize("cases,match", [
        ([], "no cases"),
        (["bare string"], "expected an object"),
        ([{"relevant_ids": ["x"]}], "missing 'query'"),
        ([{"query": "q"}], "non-empty list"),
    ])
    def test_rejects_malformed_input(self, mcp_store, cases, match):
        assert match in srv.eval_recall(cases=cases)["error"]

    def test_ranking_knobs_are_accepted(self, mcp_store):
        a = srv.remember(text="knob-accepting subject")
        out = srv.eval_recall(
            cases=[{"query": "knob-accepting subject", "relevant_ids": [a["id"]]}],
            mode="hybrid", fusion="rrf", graph=0.15, diversity=0.6,
            dedup_threshold=0.95, min_cosine=-1.0,
        )
        assert out["n"] == 1 and "error" not in out
