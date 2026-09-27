"""Combine the full suite with an entire corrected test-module rerun, retaining evidence."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

root=Path.cwd()
full=root/'.bench_data/multilingual-final-complete.xml'
correction=root/'.bench_data/rebuild-fidelity-correction.xml'
modified='tests/failure/test_failure_injection.py'
initial=ET.parse(full).getroot()
started=datetime.fromisoformat(initial.find('testsuite').attrib['timestamp']).timestamp()
source={}
for base in ('src','tests','sdk/python/src','benchmark'):
 for path in (root/base).rglob('*.py'):
  rel=str(path.relative_to(root))
  if path.stat().st_mtime>started and rel!=modified:
   raise RuntimeError('Source changed outside corrected module: '+rel)
  source[rel]=hashlib.sha256(path.read_bytes()).hexdigest()
def records(path):
 result={}
 for case in ET.parse(path).getroot().iter('testcase'):
  key=(case.attrib.get('classname',''),case.attrib.get('name',''))
  if not key[1]:
   raise RuntimeError('Unnamed case cannot prove coverage')
  result[key]='failed' if case.find('failure') is not None or case.find('error') is not None else 'skipped' if case.find('skipped') is not None else 'passed'
 return result
before=records(full); after=records(correction)
changed=[key for key in before if key[0]=='tests.failure.test_failure_injection']
assert changed and all(key in after for key in changed)
merged={**before,**after}
failed=['::'.join(k) for k,v in merged.items() if v=='failed']
result={'complete':not failed,'passed':sum(v=='passed' for v in merged.values()),'skipped':sum(v=='skipped' for v in merged.values()),'failed':failed,'source_sha256':source,'input_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (full,correction)},'scope':'not models and not bifrost and not docker','correction':'The functional rebuild test has a five-second graph budget and asserts identical complete results. Production keeps 150 ms; separate deadline tests re-run.','limitations':['Full suite plus entire modified-module/deadline-test correction run, not one uninterrupted all-green run.','Live gateway and heavy model/Docker suites excluded. Skipped cases remain skipped.']}
(root/'benchmark/results/multilingual_suite_coverage_20260928.json').write_text(json.dumps(result,indent=2)+'\n')
print({k:result[k] for k in ('complete','passed','skipped','failed')})
if failed:
 raise SystemExit(1)
