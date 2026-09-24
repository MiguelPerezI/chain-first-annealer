"""CPU regressions for certificate export, CLI bounds and shell error handling."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import gpu_sc
from validation import validate_parameters

HERE = Path(__file__).resolve().parent


class CertificateContracts(unittest.TestCase):
    def setUp(self):
        self.P = gpu_sc.build('circle')
        self.parts = []
        used = set()
        for v in range(3):
            ids = [i for i, f in enumerate(self.P.facets) if i not in used
                   and v in {int(x)//3 for x in f} and v in {int(x)%3 for x in f}]
            used.update(ids)
            self.parts.append((ids, [self.P.pi1.copy(), np.full(9,v,np.int8), self.P.pi2.copy()]))

    def test_export_and_independent_audit(self):
        with tempfile.TemporaryDirectory() as out:
            s=gpu_sc.write_certificate(self.P,self.parts,out,{})
            self.assertEqual(s['certification']['independent_audit'],'not_run')
            self.assertIn('gpu_sc.py',s['provenance']['source_sha256'])
            r=subprocess.run([sys.executable,'-B',str(HERE/'independent_check.py'),out],capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stderr)

    def test_invalid_export_has_no_side_effects(self):
        bad=[(list(range(self.P.nf)),[self.P.pi1,self.P.pi2])]
        with tempfile.TemporaryDirectory() as root:
            out=Path(root)/'invalid'
            with self.assertRaises(ValueError):gpu_sc.write_certificate(self.P,bad,out,{})
            self.assertFalse(out.exists())

    def test_malformed_certificates(self):
        cases=[]
        x=copy.deepcopy(self.parts);x[0][0].append(x[0][0][0]);cases.append(x)
        x=copy.deepcopy(self.parts);x[0][0][0]=-1;cases.append(x)
        x=copy.deepcopy(self.parts);x[0][0][0]=0.5;cases.append(x)
        x=copy.deepcopy(self.parts);x[0][1].clear();cases.append(x)
        x=copy.deepcopy(self.parts);x[0][1][1][:]=3;cases.append(x)
        x=copy.deepcopy(self.parts);x[0][1][0][:]=2;cases.append(x)
        cases.extend([[],self.parts[:1]])
        for parts in cases:
            with self.subTest(parts=str(parts)[:70]):
                with self.assertRaises(ValueError):gpu_sc.verify_system(self.P,parts)

    def test_metadata_cannot_override_certification(self):
        with tempfile.TemporaryDirectory() as out:
            with self.assertRaises(ValueError):
                gpu_sc.write_certificate(self.P,self.parts,out,{'sc_strict_bound':0})
            self.assertEqual(list(Path(out).iterdir()),[])


class ExecutionContracts(unittest.TestCase):
    def test_parameter_boundaries(self):
        validate_parameters(cmax=255,T0=0.6,T1=0.03,ptrack=1,seed=0)
        validate_parameters(crossover=0,h1_guide=0)
        validate_parameters(crossover=1,h1_guide=1)
        for values in [dict(cmax=256),dict(cmax=1),dict(S=0),dict(k=0),dict(iters=0),
                       dict(epochs=0),dict(T0=0),dict(T1=float('nan')),
                       dict(ptrack=1.1),dict(tournament_every=0),dict(time_limit=-1),
                       dict(crossover=-0.1),dict(crossover=1.1),dict(crossover=float('nan')),
                       dict(crossover=True),dict(h1_guide=-1),dict(h1_guide=1.5),
                       dict(h1_guide=True)]:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):validate_parameters(**values)
        with self.assertRaises(ValueError):gpu_sc.ChainAnnealer(None,2,256,1)

    def test_cli_rejects_before_gpu_access(self):
        r=subprocess.run([sys.executable,'-B',str(HERE/'gpu_sc.py'),'--cmax','256'],capture_output=True,text=True)
        self.assertEqual(r.returncode,2)
        self.assertIn('cmax must',r.stderr)
        self.assertNotIn('CudaSupportError',r.stderr)
        r=subprocess.run([sys.executable,'-B',str(HERE/'gpu_sc.py'),'--h1-guide','-1'],capture_output=True,text=True)
        self.assertEqual(r.returncode,2)
        self.assertIn('h1_guide must',r.stderr)
        self.assertNotIn('CudaSupportError',r.stderr)

    def test_incomplete_cli_status_and_record(self):
        P=gpu_sc.build('circle')
        chains=np.array([[P.pi1,P.pi2]])
        ann=SimpleNamespace(kernel_time=0.1,moves=1)
        for strict,expected in [(False,0),(True,2)]:
            with tempfile.TemporaryDirectory() as out:
                argv=['gpu_sc.py','--complex','circle','--k','1','--outdir',out]
                if strict:argv.append('--require-cover')
                with patch.object(sys,'argv',argv), patch.object(gpu_sc,'anneal_cover',return_value=(chains,0,[],ann)):
                    self.assertEqual(gpu_sc.cli(),expected)
                summary=json.loads((Path(out)/'summary.json').read_text())
                self.assertEqual(summary['status'],'incomplete')
                self.assertIsNone(summary['sc_strict_bound'])
                self.assertFalse((Path(out)/'cover.txt').exists())

    def test_audit_failure_propagates(self):
        with tempfile.TemporaryDirectory() as root:
            auditor=Path(root)/'fail.py';auditor.write_text('raise SystemExit(7)\n')
            env=dict(os.environ,PY=sys.executable,AUDIT=str(auditor),CHECKS=str(HERE.parent/'reproduction_checks.sh'))
            r=subprocess.run(['bash','-c','source "$CHECKS"; audit missing'],env=env,capture_output=True)
            self.assertEqual(r.returncode,7)

    def test_incomplete_record(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root);s=dict(status='incomplete',sc_strict_bound=None,all_domains_verified=False,
                                coverage=17,complex=dict(num_facets=18))
            (p/'summary.json').write_text(json.dumps(s))
            env=dict(os.environ,PY=sys.executable,CHECKS=str(HERE.parent/'reproduction_checks.sh'),OUT=root)
            cmd=['bash','-c','source "$CHECKS"; check_incomplete "$OUT"']
            self.assertEqual(subprocess.run(cmd,env=env,capture_output=True).returncode,0)
            (p/'cover.txt').write_text('stale certificate')
            self.assertNotEqual(subprocess.run(cmd,env=env,capture_output=True).returncode,0)


if __name__=='__main__':unittest.main()
