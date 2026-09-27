"""Resume the interrupted suite and audit coverage against fresh collected node IDs.

The application was frozen. One access-test assertion changed, so that entire test module
is re-run. Unnamed/interrupted JUnit entries cannot count as passing tests.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime

root=Path.cwd()
initial=root/'.bench_data/multilingual-final-regression.xml'
collection=root/'.bench_data/current-suite-collection.txt'
resumed=root/'.bench_data/multilingual-resumed-regression.xml'
modified='tests/integration/test_memory.py'
initial_xml=ET.parse(initial).getroot()
started=datetime.fromisoformat(initial_xml.find('testsuite').attrib['timestamp']).timestamp()
source={}
for base in ('src','tests','sdk/python/src','benchmark'):
 for path in (root/base).rglob('*.py'):
  rel=str(path.relative_to(root))
  if path.stat().st_mtime > started and rel != modified:
   raise RuntimeError('Source changed since the original suite; cannot reuse passes: '+rel)
  source[rel]=hashlib.sha256(path.read_bytes()).hexdigest()
expected=[line.strip() for line in collection.read_text().splitlines() if line.startswith('tests/') and '::' in line]
assert len(expected)==len(set(expected)) and len(expected)>1500

def records(path):
 result={}
 for case in ET.parse(path).getroot().findall('.//testcase'):
  parts=case.attrib.get('classname','').split('.')
  name=case.attrib.get('name')
  if not name or not any(parts):
   continue
  for n in range(len(parts),0,-1):
   file='/'.join(parts[:n])+'.py'
   if (root/file).is_file():
    node='::'.join([file,*parts[n:],name])
    break
  else:
   raise RuntimeError('Unresolvable JUnit case: '+str(case.attrib))
  status='failed' if case.find('failure') is not None or case.find('error') is not None else 'skipped' if case.find('skipped') is not None else 'passed'
  result[node]=status
 return result

prior=records(initial)
reusable={node:status for node,status in prior.items() if status!='failed' and not node.startswith(modified+'::') and node in expected}
remaining=[node for node in expected if node not in reusable]
print('Collected',len(expected),'reusing',len(reusable),'running',len(remaining),flush=True)
(root/'.bench_data/resumed-nodeids.txt').write_text('\n'.join(remaining)+'\n')
run=subprocess.run([sys.executable,'-m','pytest','-q','-m','not models and not bifrost and not docker','--junitxml='+str(resumed),*remaining])
current=records(resumed) if resumed.exists() else {}
combined={**reusable,**current}
missing=sorted(set(expected)-combined.keys())
failed=sorted(node for node,status in combined.items() if status=='failed')
result={'complete':run.returncode==0 and not missing and not failed,'expected_collected_tests':len(expected),'passed':sum(v=='passed' for v in combined.values()),'skipped':sum(v=='skipped' for v in combined.values()),'missing':missing,'failed':failed,'reused_completed_cases':len(reusable),'resumed_cases':len(current),'source_sha256':source,'input_sha256':{str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in (initial,collection,resumed)},'scope':'not models and not bifrost and not docker','correction':'Thread authorization test now allows the newly preserved short kick-off source. Entire modified test module re-run.','limitations':['Two sequential suite segments, not a single uninterrupted run.','Model-quality and live gateway tests excluded; separately measured artifacts are required.']}
(root/'benchmark/results/multilingual_suite_coverage_20260928.json').write_text(json.dumps(result,indent=2)+'\n')
print({k:result[k] for k in ('complete','expected_collected_tests','passed','skipped','failed','missing')},flush=True)
if not result['complete']:
 raise SystemExit(run.returncode or 1)
