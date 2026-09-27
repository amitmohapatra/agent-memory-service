"""One isolated named-vector fusion screen. No inference, paid APIs or shared stores."""
import asyncio
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from qdrant_client import AsyncQdrantClient, models
from benchmark.harness import stats
from benchmark.public.metrics import evaluate_run
from memory_service.adapters.models.sparse import Bm25SparseEncoder
from memory_service.adapters.search.qdrant_store import _filter
from memory_service.ports.search import SearchFilter

ROOT=Path.cwd()
OUTPUT=ROOT/'benchmark/results/scifact_qdrant_ensemble_screen.json'
COLLECTION='ensemble_scifact_20260928'
DATA=Path('/Users/ricky/usage_data/agent-memory-service/benchmark/data/scifact.json')
CANDIDATES=(
 ('english','granite',Path('/Users/ricky/usage_data/ams-hindsight-integration/.bench_data/document-vectors')),
 ('multilingual','bekko',ROOT/'.bench_data/document-vectors'),
)
IMAGE='sha256:75eab8c4ba42096724fdcfde8b4de0b5713d529dde32f285a1f86fdcb2c9e50c'

def sha(path):
 return hashlib.sha256(path.read_bytes()).hexdigest()

async def main():
 data=json.loads(DATA.read_text())
 assert len(data['corpus'])==5183 and len(data['queries'])==300
 ids=[str(doc['id']) for doc in data['corpus']]
 ordinal={doc:i for i,doc in enumerate(ids)}
 qrels={str(q['id']):{str(doc):1 for doc in q['relevant']} for q in data['queries']}
 vectors={}; manifests={}
 for field,candidate,cache in CANDIDATES:
  artifact=ROOT/'benchmark/results'/('scifact_dense_'+candidate+'.json')
  result=json.loads(artifact.read_text())
  assert result['complete'] and result['manifest']['data_sha256']==sha(DATA)
  key=hashlib.sha256(json.dumps(result['manifest'],sort_keys=True).encode()).hexdigest()
  path=cache/(key+'.npz')
  with np.load(path,allow_pickle=False) as saved:
   docs,queries=saved['documents'],saved['queries']
  assert docs.shape==(5183,384) and queries.shape==(300,384)
  assert np.isfinite(docs).all() and np.isfinite(queries).all()
  vectors[field]=(docs,queries)
  manifests[field]={'artifact_sha256':sha(artifact),'vectors_sha256':sha(path),'model':result['manifest']}
 encoder=Bm25SparseEncoder()
 texts=[doc['title']+'\n\n'+doc['text'] for doc in data['corpus']]
 sparse=encoder.encode_documents(texts)
 client=AsyncQdrantClient(url='http://localhost:26333',grpc_port=26334,prefer_grpc=True,timeout=10)
 result={'complete':False,'image':IMAGE,'dataset_sha256':sha(DATA),'inputs':manifests,
 'benchmark_sha256':sha(Path(__file__)),'source_filter_sha256':sha(ROOT/'src/memory_service/adapters/search/qdrant_store.py'),
 'inference_calls':0,'llm_calls':0,'arms':{},'limitations':[
 'Isolated Qdrant component with cached vectors; excludes query encoding, SQL, HTTP API and context construction.',
 'One title+abstract per document; no document parsing/chunk/parent-expansion measurement.',
 'Shared host; separate Qdrant capped at one CPU and 512 MiB. Not an 8-vCPU deployment SLO.',
 'Small fresh collection may use exact scans; record index state, do not extrapolate scale.',
 'Model-selection test-set screen, not independent held-out promotion evidence.']}
 try:
  if await client.collection_exists(COLLECTION):
   raise RuntimeError('Refusing to reuse an existing experiment collection')
  await client.create_collection(COLLECTION,
   vectors_config={field:models.VectorParams(size=384,distance=models.Distance.COSINE) for field in vectors},
   sparse_vectors_config={'bm25':models.SparseVectorParams(modifier=models.Modifier.IDF)},
   on_disk_payload=False)
  for field in ('tenant_id','visibility_keys'):
   await client.create_payload_index(COLLECTION,field,models.PayloadSchemaType.KEYWORD,wait=True)
  for start in range(0,len(ids),128):
   points=[]
   for i in range(start,min(start+128,len(ids))):
    point={field:matrix[i].tolist() for field,(matrix,_) in vectors.items()}
    point['bm25']=models.SparseVector(indices=sparse[i].indices,values=sparse[i].values)
    points.append(models.PointStruct(id=i,vector=point,payload={'document_id':ids[i],'tenant_id':'ensemble-screen','visibility_keys':['reader']}))
   await client.upsert(COLLECTION,points,wait=True)
  info=await client.get_collection(COLLECTION)
  result['index']={'points_count':info.points_count,'indexed_vectors_count':info.indexed_vectors_count,'status':str(info.status)}
  flt=_filter(SearchFilter(tenant_id='ensemble-screen',must_any={'visibility_keys':['reader']}))
  async def query(number,ensemble):
   sparse_query=encoder.encode_query(data['queries'][number]['text'])
   arms=[models.Prefetch(query=vectors['english'][1][number].tolist(),using='english',limit=50,filter=flt)]
   if ensemble:
    arms.append(models.Prefetch(query=vectors['multilingual'][1][number].tolist(),using='multilingual',limit=50,filter=flt))
   arms.append(models.Prefetch(query=models.SparseVector(indices=sparse_query.indices,values=sparse_query.values),using='bm25',limit=50,filter=flt))
   start=time.perf_counter()
   response=await client.query_points(COLLECTION,prefetch=arms,query=models.FusionQuery(fusion=models.Fusion.RRF),query_filter=flt,limit=50,with_payload=['document_id'])
   elapsed=(time.perf_counter()-start)*1000
   found=sorted(response.points,key=lambda point:(-point.score,int(point.id)))
   return [str(point.payload['document_id']) for point in found[:10]],elapsed
  await query(0,False);await query(0,True)
  runs={'baseline':{},'ensemble':{}};timings={'baseline':[],'ensemble':[]}
  for i,question in enumerate(data['queries']):
   for name in (('baseline','ensemble') if i%2==0 else ('ensemble','baseline')):
    run,latency=await query(i,name=='ensemble');runs[name][str(question['id'])]=run;timings[name].append(latency)
   if (i+1)%50==0:
    print('queries',i+1,flush=True)
  for name in runs:
   result['arms'][name]={'metrics':evaluate_run(runs[name],qrels,recall_ks=(10,)),'search_ms':stats(timings[name]),'run':runs[name]}
  # Add high-scoring unauthorized points only AFTER scoring, so the measured corpus and
  # BM25 document frequencies remain exactly 5,183 documents in both quality arms.
  first_query=encoder.encode_query(data['queries'][0]['text'])
  sentinels=[]
  for offset,tenant,scope in ((0,'other-tenant','reader'),(1,'ensemble-screen','sibling')):
   point={field:matrix[0].tolist() for field,(_,matrix) in vectors.items()}
   point['bm25']=models.SparseVector(indices=first_query.indices,values=first_query.values)
   sentinels.append(models.PointStruct(id=900000+offset,vector=point,payload={'document_id':'unauthorized-'+str(offset),'tenant_id':tenant,'visibility_keys':[scope]}))
  await client.upsert(COLLECTION,sentinels,wait=True)
  leaks=[]
  for ensemble in (False,True):
   found,_=await query(0,ensemble)
   leaks.extend(doc for doc in found if doc.startswith('unauthorized'))
  result['isolation']={'unauthorized_points':2,'leaks':leaks,'passed':not leaks}
  assert not leaks
  result['complete']=True
 finally:
  OUTPUT.write_text(json.dumps(result,indent=2)+'\n')
  await client.close()
 print(json.dumps({k:{x:y for x,y in v.items() if x!='run'} for k,v in result['arms'].items()},indent=2),flush=True)

asyncio.run(main())
