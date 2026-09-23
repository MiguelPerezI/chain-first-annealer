"""CPU-only orchestration tests. Synthetic results never attest an S6 cover."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import run_s6_ladder as ladder


class LadderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name) / "campaign"
        self.plan = ladder.load_plan(ladder.OPTI / "experiments/s6_ladder.json")
        self.calls = []
        self.audit_rc = 0
        self.patch = patch.object(ladder, "runtime_info", return_value={"synthetic_cpu_test": True})
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def runner(self, coverages, bad_rc=None, stale=False):
        coverages = iter(coverages)

        def run(args, log):
            self.calls.append(args)
            log.write_text("Synthetic CPU test; no GPU search performed.\n")
            if "independent_check.py" in args[2]:
                return self.audit_rc
            cov = next(coverages)
            options = dict(zip(args[3::2], args[4::2]))  # --require-cover has no value; parameters precede it
            index = int(log.parent.name.removeprefix("rung"))
            params = ladder.parameters(self.plan, index)
            for key, value in params.items():
                self.assertEqual(options["--" + key.replace("_", "-")], str(value))
            chain = [[v // 8 for v in range(64)]] * params["cmax"] + [[v % 8 for v in range(64)]]
            full = cov == ladder.NF
            summary = dict(complex=dict(nv=8, num_facets=ladder.NF), params=params,
                           coverage=cov, best_chains=[chain] * 3,
                           sc_strict_bound=2 if full else None, all_domains_verified=full)
            if full:
                summary.update(n_domains=3, n_verified=3)
            else:
                summary["status"] = "incomplete"
            (log.parent / "summary.json").write_text(json.dumps(summary))
            if full or stale:
                for file in ["cover.txt", "planners.txt"]:
                    (log.parent / file).write_text("synthetic fixture; not a certificate\n")
            return bad_rc if bad_rc is not None else (0 if full else 2)
        return run

    def run_campaign(self, coverages, **kwargs):
        return ladder.campaign(self.plan, self.out, Path(sys.executable),
                               runner=self.runner(coverages), **kwargs)

    def state(self):
        return json.loads((self.out / "campaign.json").read_text())

    def test_budget_and_dry_run_has_no_side_effects(self):
        r = subprocess.run([sys.executable, str(Path(ladder.__file__)), "--outdir", str(self.out)],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        plan = json.loads(r.stdout)
        self.assertEqual(plan["max_epochs"], 17)
        self.assertEqual(plan["max_moves"], 710400)
        self.assertFalse(self.out.exists())
        self.assertNotIn("numba", sys.modules)

    def test_invalid_plans_fail_before_launch(self):
        cases = []
        p = copy.deepcopy(self.plan); p["rungs"][1]["epochs"] = 1; cases.append(p)
        p = copy.deepcopy(self.plan); p["rungs"][2]["cmax"] = 8; cases.append(p)
        p = copy.deepcopy(self.plan); p["rungs"][0]["epochs"] = 2; cases.append(p)
        p = copy.deepcopy(self.plan); p["common"]["S"] = 1; cases.append(p)
        p = copy.deepcopy(self.plan); p["rungs"][1]["guide_max"] = 1; cases.append(p)
        p = copy.deepcopy(self.plan); p["init"] = "s6_starcover/summary.json"; cases.append(p)
        p = copy.deepcopy(self.plan); p["common"]["gamma"] = -1; cases.append(p)
        for p in cases:
            path = Path(self.tmp.name) / "plan.json"
            path.write_text(json.dumps(p))
            with self.assertRaises(ValueError):
                ladder.load_plan(path)

    def test_partial_rungs_escalate_from_best_not_latest(self):
        self.assertEqual(self.run_campaign([100, 90, 150, 140, 160]), 2)
        self.assertNotIn("--init", self.calls[0])
        for call, expected in zip(self.calls[1:], ["rung00", "rung00", "rung02", "rung02"]):
            path = Path(call[call.index("--init") + 1])
            self.assertEqual(path.parent.name, expected)
            self.assertEqual(path.parent.parent, self.out)
            self.assertIn("--require-cover", call)
        s = self.state()
        self.assertEqual(s["best_coverage"], 160)
        self.assertEqual(s["status"], "incomplete")
        self.assertIsNone(s["sc_strict_bound"])
        self.assertEqual(len(s["rungs"]), 5)
        self.assertIsNotNone(s["rungs"][1]["init_sha256"])

    def test_full_coverage_stops_only_after_audit(self):
        self.assertEqual(self.run_campaign([100, ladder.NF]), 0)
        self.assertEqual(len(self.calls), 3)  # two searches, one audit
        self.assertIn("independent_check.py", self.calls[-1][2])
        self.assertNotIn("-O", self.calls[-1])
        self.assertEqual(self.state()["status"], "verified")
        self.assertEqual(self.state()["sc_strict_bound"], 2)
        self.assertIn("audit.log", self.state()["rungs"][-1]["artifact_sha256"])

    def test_audit_failure_never_grants_bound(self):
        self.audit_rc = 7
        with self.assertRaisesRegex(ValueError, "audit failed"):
            self.run_campaign([ladder.NF])
        self.assertEqual(self.state()["status"], "failed")
        self.assertIsNone(self.state()["sc_strict_bound"])

    def test_process_failure_or_bad_partial_contract_stops_ladder(self):
        for rc, stale in [(9, False), (0, False), (2, True)]:
            self.out = Path(self.tmp.name) / f"bad_{rc}_{stale}"
            with self.assertRaises(ValueError):
                ladder.campaign(self.plan, self.out, Path(sys.executable),
                                runner=self.runner([100], bad_rc=rc, stale=stale))
            self.assertEqual(self.state()["status"], "failed")
            self.assertEqual(self.state()["rungs"], [])
            self.assertIsNone(self.state()["sc_strict_bound"])

    def test_pause_resume_preserves_best_and_checks_artifact_hashes(self):
        self.assertEqual(self.run_campaign([100, 90], stop_after=2), 3)
        self.assertEqual(self.state()["status"], "paused")
        self.assertEqual(self.run_campaign([150, 140, 160], resume=True), 2)
        self.assertEqual(len(self.state()["rungs"]), 5)
        self.assertIn("rung00", self.calls[2][-1])
        (self.out / "rung00/summary.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "modified rung artifact"):
            self.run_campaign([], resume=True)

    def test_refuse_overwrite_changed_plan_and_interrupted_resume(self):
        self.run_campaign([100], stop_after=1)
        with self.assertRaises(FileExistsError):
            self.run_campaign([])
        changed = copy.deepcopy(self.plan)
        changed["rungs"][2]["guide"] = 9
        with self.assertRaisesRegex(ValueError, "same plan"):
            ladder.campaign(changed, self.out, Path(sys.executable), resume=True)
        s = self.state(); s["status"] = "interrupted"
        ladder.save(self.out / "campaign.json", s)
        with self.assertRaisesRegex(ValueError, "completed rung boundary"):
            self.run_campaign([], resume=True)

    def test_corrupt_full_output_rejected_by_real_independent_checker(self):
        self.run_campaign([ladder.NF])  # mock audit passes a synthetic fixture
        log = Path(self.tmp.name) / "real_audit.log"
        rc = ladder.run_logged([sys.executable, "-B", str(ladder.SRC / "independent_check.py"),
                                str(self.out / "rung00")], log)
        self.assertNotEqual(rc, 0)

    def test_auditor_subprocess_removes_pythonoptimize(self):
        log = Path(self.tmp.name) / "optimization.log"
        with patch.dict(os.environ, PYTHONOPTIMIZE="2"):
            rc = ladder.run_logged([sys.executable, "-c", "print(__debug__); raise SystemExit(0 if __debug__ else 1)"], log)
        self.assertEqual(rc, 0)
        self.assertIn("True", log.read_text())


if __name__ == "__main__":
    unittest.main()
