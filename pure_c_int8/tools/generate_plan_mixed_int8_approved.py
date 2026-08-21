#!/usr/bin/env python3
import json, math, sys
from pathlib import Path

root=Path(sys.argv[1] if len(sys.argv)>1 else '.')
graph=json.loads((root/'qat/deployment/v9/integer_reference/v9_integer_execution_plan.json').read_text())
man=json.loads((root/'qat/deployment/v9/approved_pure_c/corrected_int8_manifest.json').read_text())
out=root/'pure_c_int8/generated'; out.mkdir(parents=True,exist_ok=True)
nodes=graph['nodes']; layers=man['layers']

def refs(v):
 if isinstance(v,dict) and 'node' in v:return [v['node']]
 if isinstance(v,list):
  z=[]
  for x in v:z+=refs(x)
  return z
 return []

def qm(r):
 if r==0:return 0,0
 s,e=math.frexp(float(r));m=round(s*(1<<31))
 if m==(1<<31):m//=2;e+=1
 return int(max(-(1<<31),min((1<<31)-1,m))),int(e)
compute=[n for n in nodes if n['op'] in ('placeholder','call_function','output')]
convnodes=[n for n in compute if n['op']=='call_function' and n['target'].startswith('conv2d')]
assert len(convnodes)==63
convpos={n['name']:i for i,n in enumerate(convnodes)}
cons={}
for n in nodes:
 for r in refs(n.get('args',[])):cons.setdefault(r,[]).append(n['name'])
by={n['name']:n for n in nodes}
def nextscale(name):
 q=[name];seen=set()
 while q:
  x=q.pop(0)
  if x in seen:continue
  seen.add(x)
  for y in cons.get(x,[]):
   if y in convpos:return float(layers[convpos[y]]['input_scale'])
   q.append(y)
 return None
ids={'x':0}; aliases={}; scales={'x':float(layers[0]['input_scale'])}; shapes={'x':[1,13,256,256]}; ops=[]; nextid=1
def resolve(n):
 while n in aliases:n=aliases[n]
 return n
for node in compute:
 name,t=node['name'],node['target']
 if node['op'] in ('placeholder','output'):continue
 rr=[resolve(r) for r in refs(node.get('args',[]))]
 if t.startswith('batch_norm'):
  aliases[name]=rr[0];continue
 oid=nextid;nextid+=1;ids[name]=oid
 shape=node.get('shape') or shapes[rr[0]];shapes[name]=shape
 if t.startswith('conv2d'):
  i=convpos[name]; src=rr[0]; ss=scales[src];ts=float(layers[i]['input_scale']);m,s=qm(ss/ts)
  scales[name]=float(layers[i]['output_scale']);ops.append(('CONV',oid,ids[src],-1,i,m,s,0,0,ss/ts,1.0,shape))
 elif t.startswith('hardtanh'):
  src=rr[0];scales[name]=scales[src];ops.append(('RELU6',oid,ids[src],-1,-1,0,0,round(6/scales[src]),0,1.0,1.0,shape))
 elif t.startswith('relu'):
  src=rr[0];scales[name]=scales[src];ops.append(('RELU',oid,ids[src],-1,-1,0,0,0,0,1.0,1.0,shape))
 elif t.startswith('add'):
  a,b=rr[:2];ts=nextscale(name) or max(scales[a],scales[b]);m1,s1=qm(scales[a]/ts);m2,s2=qm(scales[b]/ts);scales[name]=ts;ops.append(('ADD',oid,ids[a],ids[b],-1,m1,s1,m2,s2,scales[a]/ts,scales[b]/ts,shape))
 elif t.startswith('upsample_nearest2d'):
  src=rr[0];scales[name]=scales[src];ops.append(('RESIZE',oid,ids[src],-1,-1,0,0,0,0,1.0,1.0,shape))
 elif t.startswith('cat'):
  assert len(rr)==2, (name,rr);a,b=rr;ts=nextscale(name) or max(scales[a],scales[b]);m1,s1=qm(scales[a]/ts);m2,s2=qm(scales[b]/ts);scales[name]=ts;ops.append(('CONCAT',oid,ids[a],ids[b],-1,m1,s1,m2,s2,scales[a]/ts,scales[b]/ts,shape))
 else:raise RuntimeError((name,t))
# files
h='''#pragma once\n#include <stdint.h>\ntypedef enum {OP_CONV,OP_RELU6,OP_RELU,OP_ADD,OP_RESIZE,OP_CONCAT} OpKind;\ntypedef struct {int kind,out,a,b,layer; int32_t m1,s1,m2,s2; double r1,r2; int n,c,h,w;} Op;\ntypedef struct {int oc,icg,kh,kw,sh,sw,ph,pw,dh,dw,groups; const char *wf,*bf,*mf,*sf;} Layer;\nextern const Op OPS[]; extern const int NOPS,NTENSORS; extern const Layer LAYERS[]; extern const int NLAYERS,FINAL_TENSOR;\n'''
(out/'v9_plan.h').write_text(h)
c=['#include "v9_plan.h"','const Layer LAYERS[] = {']
for l in layers:
 sh=l['weight_shape'];c.append('{%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,"%s","%s","%s","%s"},'%(sh[0],sh[1],sh[2],sh[3],*l['stride'],*l['padding'],*l['dilation'],l['groups'],l['weight_file'].replace('\\','/'),l['bias_file'].replace('\\','/'),l['multiplier_file'].replace('\\','/'),l['shift_file'].replace('\\','/')))
c+=['};','const Op OPS[] = {']
kind={'CONV':'OP_CONV','RELU6':'OP_RELU6','RELU':'OP_RELU','ADD':'OP_ADD','RESIZE':'OP_RESIZE','CONCAT':'OP_CONCAT'}
for o in ops:
 k,oid,a,b,li,m1,s1,m2,s2,r1,r2,sh=o;c.append('{%s,%d,%d,%d,%d,%d,%d,%d,%d,%.17g,%.17g,%d,%d,%d,%d},'%(kind[k],oid,a,b,li,m1,s1,m2,s2,r1,r2,*sh))
c += ['};',f'const int NOPS={len(ops)}, NTENSORS={nextid}, NLAYERS={len(layers)}, FINAL_TENSOR={ids[convnodes[-1]["name"]]};']
(out/'v9_plan.c').write_text('\n'.join(c)+'\n')
print(json.dumps({'ops':len(ops),'tensors':nextid,'layers':len(layers),'final_tensor':ids[convnodes[-1]['name']]},indent=2))
