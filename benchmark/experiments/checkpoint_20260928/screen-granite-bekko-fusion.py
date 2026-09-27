"""One fixed equal-weight RRF hypothesis over already measured candidate lists.

No inference or label-driven weight search. English and multilingual specialists use
separate ranking lists; this is not the score of concatenated embedding vectors.
"""
import hashlib
import json
import random
from pathlib import Path
from benchmark.public.metrics import evaluate_run, ndcg_at_k, recall_at_k

root=Path('benchmark/results')
paths=[root/'scifact_dense_granite.json',root/'scifact_dense_bekko.json']
a,b=[json.loads(p.read_text()) for p in paths]
data_path=Path('/Users/ricky/usage_data/agent-memory-service/benchmark/data/scifact.json')
data=json.loads(data_path.read_text())
data_hash=hashlib.sha256(data_path.read_bytes()).hexdigest()
assert data_hash==a['manifest']['data_sha256']==b['manifest']['data_sha256']
ordinal={str(d['id']):i for i,d in enumerate(data['corpus'])}
qrels={str(q['id']):{str(d):1 for d in q['relevant']} for q in data['queries']}
other={row['query_id']:row for row in b['candidates']}
def fuse(arms):
 scores={}
 for arm in arms:
  for rank,doc in enumerate(arm,1):
   scores[doc]=scores.get(doc,0)+1/(1+rank)
 return sorted(scores,key=lambda doc:(-scores[doc],ordinal[doc]))[:10]
baseline={row['query_id']:fuse((row['dense'],row['sparse'])) for row in a['candidates']}
assert evaluate_run(baseline,qrels,recall_ks=(10,))==a['metrics']['hybrid_k1']
run={row['query_id']:fuse((row['dense'],other[row['query_id']]['dense'],row['sparse'])) for row in a['candidates']}
result={'hypothesis':'Equal-weight RRF k=1 over English Granite dense, Bekko a8m dense and BM25; no parameter search', 'complete':True,'llm_calls':0,'inference_calls':0,'dataset_sha256':data_hash,'input_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'metrics':evaluate_run(run,qrels,recall_ks=(10,)), 'baseline':a['metrics']['hybrid_k1'],'paired':{},'run':run,'limitations':['Test-set screen, not held-out promotion evidence.','Two-encoder architecture is not implemented or timed.','Cannot establish multilingual ensemble or end-to-end RAG quality.']}
for name,metric in [('ndcg@10',ndcg_at_k),('recall@10',recall_at_k)]:
 differences=[metric(run[q],rels,10)-metric(baseline[q],rels,10) for q,rels in qrels.items()]
 rng=random.Random(20260928)
 samples=sorted(sum(rng.choices(differences,k=len(differences)))/len(differences) for _ in range(10000))
 result['paired'][name]={'mean_delta':sum(differences)/len(differences),'ci95':[samples[249],samples[9749]],'wins':sum(x>1e-12 for x in differences),'losses':sum(x < -1e-12 for x in differences),'ties':sum(abs(x)<=1e-12 for x in differences)}
result['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(root/'scifact_granite_bekko_fusion_screen.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ('metrics','baseline','paired')},indent=2))
