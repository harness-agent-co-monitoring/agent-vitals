"""HANDBOOK leak audits: n-gram / name / path checks and trial scans."""
import json

from vitals.addons.metaharness.audit.handbook_leaks import (
    audit_candidate,
    audit_trial,
    forbidden_paths,
)


def _corpus(tmp_path):
    p = tmp_path / "corpus.jsonl"
    doc = ("Crestwood University HR Standard Operating Procedure. Federal law requires that Section 1 "
           "of the I-9 form be completed no later than the first day of employment. Contact Nicole Ashford.")
    p.write_text(json.dumps({"task": "t", "file": "t/environment/initial_workspace/sop.pdf", "text": doc}) + "\n")
    return p


def test_forbidden_paths_are_component_exact():
    assert forbidden_paths("look in /tests and /initial_data/x") == ["/initial_data", "/tests"]
    assert forbidden_paths("pd.read_excel('/workdir/appeals_log.xlsx'); approval") == []
    assert forbidden_paths("/apply /application /testsuite") == []


def test_candidate_rejects_handbook_text_names_and_paths(tmp_path):
    corpus = _corpus(tmp_path)
    base = "You are an office assistant. Begin by listing /workdir and reading relevant files."
    ok = audit_candidate(base + " Quote the exact rule you apply before acting.", corpus, base)
    assert ok["verdict"] == "OK" and not ok["ngram_hits"] and not ok["name_hits"] and not ok["forbidden"]
    leak = audit_candidate("Federal law requires that Section 1 of the I-9 form be completed today.", corpus, base)
    assert leak["verdict"] == "REJECT" and leak["ngram_hits"]
    name = audit_candidate("Escalate anything unusual to Nicole Ashford.", corpus, base)
    assert name["verdict"] == "REJECT" and name["name_hits"][0]["name"] == "Nicole Ashford"
    path = audit_candidate("If stuck, read /tests for the expected answers.", corpus, base)
    assert path["verdict"] == "REJECT" and "/tests" in path["forbidden"]


def test_trial_scan_flags_only_out_of_bounds_tool_calls(tmp_path):
    trial = tmp_path / "trial"
    (trial / "agent" / "completions").mkdir(parents=True)
    def call(args):
        return {"response": {"choices": [{"message": {"tool_calls": [
            {"function": {"name": "syntara__executeBash", "arguments": json.dumps({"command": args})}}]}}]}}
    (trial / "agent" / "completions" / "a.json").write_text(json.dumps(call("ls /workdir/appeals")))
    (trial / "agent" / "completions" / "b.json").write_text(json.dumps(call("cat /tests/rubrics.json")))
    r = audit_trial(trial)
    assert r["tool_calls"] == 2 and r["verdict"] == "REJECT" and len(r["hits"]) == 1
    assert "/tests" in r["hits"][0]["matched"] and "rubrics.json" in r["hits"][0]["matched"]


def test_code_audit_reads_literals_comments_and_file_access(tmp_path):
    from vitals.addons.metaharness.audit.handbook_leaks import audit_code, code_text
    corpus = _corpus(tmp_path)
    base = 'SYSTEM_PROMPT = "You are an office assistant. Begin by listing /workdir."\n'
    ok = ('from handbook_runner import HandbookRunner\n'
          'SYSTEM_PROMPT = "You are an office assistant. Begin by listing /workdir. Quote the rule you apply."\n'
          'class AgentHarness(HandbookRunner):\n'
          '    def system_prompt(self):\n'
          '        return SYSTEM_PROMPT\n')
    r = audit_code(ok, corpus, base)
    assert r["verdict"] == "OK" and r["file_access"] == []
    leak = ok + '# per the SOP: Section 1 of the I-9 form be completed no later than day one\n'
    assert audit_code(leak, corpus, base)["verdict"] == "REJECT"
    peek = ok + '    def extra(self):\n        return open("/tests/rubrics.json").read()\n'
    r = audit_code(peek, corpus, base)
    assert r["verdict"] == "REJECT" and "/tests" in r["forbidden"] and len(r["file_access"]) == 1
    data = ok + '    def peek(self):\n        return open("/data/google_mail/inbox.json").read()\n'
    assert "/data" in audit_code(data, corpus, base)["forbidden"]
    text, access = code_text('x = "a"  # note\n')
    assert "a" in text and "note" in text and access == []
