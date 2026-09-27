"""Predeclared three-arm RRF screen: all 12 languages, fixed 100 IDs per language.

No fitted weights, translation, generation, API calls, or language-dependent routing.
Native encoders are measured separately; fusion consumes cached rank lists.
"""
import asyncio
import hashlib
import json
from pathlib import Path
import numpy as np
from benchmark.multilingual_dense import ranking, metrics, record
from benchmark.multilingual_sparse import SparseCorpus, load_dataset
from memory_service.adapters.models.embeddings import load_dense
from memory_service.config.constants import DenseModel

ROOT=Path.cwd()
DATA=Path('/Users/ricky/usage_data/ams-hindsight-integration/benchmark/data/xquad')
SPECS={
 'granite':Path('/Users/ricky/usage_data/ams-hindsight-integration/.bench_data/models/granite-baseline.json'),
 'bekko':ROOT/'.bench_data/models/bekko-a8m.json',
}
OUTPUT=ROOT/'benchmark/results/multilingual_granite_bekko_fusion_screen.json'
CACHE=ROOT/'.bench_data/multilingual-fusion-vectors'
CACHE.mkdir(exist_ok=True)
manifest=json.loads((DATA/'manifest.json').read_text())
langs=sorted(manifest['files'])
corpora={}; questions={}
for lang in langs:
 path=DATA/f'xquad.{lang}.json'
 assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['files'][lang]['sha256']
 corpora[lang],all_queries=load_dataset(path)
 questions[lang]=[all_queries[i] for i in np.linspace(0,len(all_queries)-1,100,dtype=int)]
en_gold={q['id']:q['gold'] for q in load_dataset(DATA/'xquad.en.json')[1]}

def fuse(arms):
 scores={}
 for arm in arms:
  for rank,doc in enumerate(arm,1):
   scores[doc]=scores.get(doc,0)+1/(1+rank)
 return sorted(scores,key=lambda doc:(-scores[doc],doc))[:10]

async def encode(name,path):
 spec=DenseModel.model_validate_json(path.read_text())
 model=load_dense(spec)
 matrices={}
 try:
  for lang in langs:
   key=hashlib.sha256(json.dumps({'encoder':model.fingerprint(),'file':manifest['files'][lang]['sha256'],'ids':[q['id'] for q in questions[lang]]},sort_keys=True).encode()).hexdigest()
   cache=CACHE/(key+'.npz')
   if cache.exists():
    with np.load(cache,allow_pickle=False) as rows:
     matrices[lang]=(rows['documents'],rows['queries'])
   else:
    docs=np.asarray(await model.embed_documents(corpora[lang]),dtype=np.float32)
    queries=np.asarray([await model.embed_query(q['query']) for q in questions[lang]],dtype=np.float32)
    with cache.open('wb') as handle:
     np.savez_compressed(handle,documents=docs,queries=queries)
    matrices[lang]=(docs,queries)
   print(name,lang,'encoded',flush=True)
 finally:
  model.close()
 return matrices

async def main():
 vectors={name:await encode(name,path) for name,path in SPECS.items()}
 result={'complete':False,'dataset':manifest,'specs':{n:json.loads(p.read_text()) for n,p in SPECS.items()},'queries_per_language':100,'llm_calls':0,'hypothesis':'Equal RRF k=1, depth50, three independent arms vs two arms; no weights fitted','languages':{},'limitations':['Fixed 100 evenly spaced IDs per language; a screen, not full XQuAD.','Small-corpus paragraph retrieval; not generated-answer accuracy or full service latency.','Exploratory public test-set selection; no held-out promotion claim.']}
 for lang in langs:
  sparse=SparseCorpus(corpora[lang]); english_sparse=SparseCorpus(corpora['en'])
  arms={name:[] for name in ('same_bekko','same_ensemble','cross_bekko','cross_ensemble')}
  for idx,q in enumerate(questions[lang]):
   for prefix,target,lexicon,gold in [('same',lang,sparse,q),('cross','en',english_sparse,{**q,'gold':en_gold[q['id']]})]:
    english=ranking(vectors['granite'][lang][1][idx],vectors['granite'][target][0])
    multilingual=ranking(vectors['bekko'][lang][1][idx],vectors['bekko'][target][0])
    lexical,_=lexicon.retrieve(q['query'],50)
    record(arms[prefix+'_bekko'],gold,fuse((multilingual,lexical)))
    record(arms[prefix+'_ensemble'],gold,fuse((english,multilingual,lexical)))
  result['languages'][lang]={name:metrics(rows) for name,rows in arms.items()}
  OUTPUT.write_text(json.dumps(result,indent=2)+'\n')
  print(lang,{n:m['recall']['10'] for n,m in result['languages'][lang].items()},flush=True)
 result['complete']=True
 result['mean_recall_at_10']={name:sum(rows[name]['recall']['10'] for rows in result['languages'].values())/len(langs) for name in arms}
 result['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 OUTPUT.write_text(json.dumps(result,indent=2)+'\n')
 print(result['mean_recall_at_10'],flush=True)

asyncio.run(main())
