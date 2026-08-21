#!/usr/bin/env python3
import copy,json,math,time
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from cloudsen12_models import cloudsen12
ROOT=Path.cwd();OUT=ROOT/'qat/deployment/v9/q31_retry_strict';OUT.mkdir(parents=True,exist_ok=True)
MAN=json.loads((ROOT/'qat/deployment/v9/approved_pure_c/corrected_int8_manifest.json').read_text());CK=ROOT/'qat/deployment/v9/deployment_qat/deployment_qat_best_locked.pth'
class DS(Dataset):
 def __init__(self,p):self.f=sorted(Path(p).glob('*.npz'))
 def __len__(self):return len(self.f)
 def __getitem__(self,i):
  with np.load(self.f[i]) as d:return torch.from_numpy(d['image'].astype(np.float32)/10000),torch.from_numpy(d['target'].astype(np.int64))
def fold_pair(conv,bn):
 w=conv.weight.detach();b=conv.bias.detach() if conv.bias is not None else torch.zeros(w.shape[0],device=w.device);g=bn.weight.detach() if bn.affine else torch.ones_like(b);be=bn.bias.detach() if bn.affine else torch.zeros_like(b);rm=bn.running_mean;rv=bn.running_var;s=g/torch.sqrt(rv+bn.eps)
 conv.weight=torch.nn.Parameter(w*s.reshape(-1,1,1,1));conv.bias=torch.nn.Parameter((b-rm)*s+be);return conv
def fold_bn(mod):
 for name,ch in list(mod.named_children()):fold_bn(ch)
 items=list(mod._modules.items())
 for i in range(len(items)-1):
  n1,a=items[i];n2,b=items[i+1]
  if isinstance(a,torch.nn.Conv2d) and isinstance(b,torch.nn.BatchNorm2d):mod._modules[n1]=fold_pair(a,b);mod._modules[n2]=torch.nn.Identity()
def qm(r):
 if r==0:return 0,0
 s,e=math.frexp(float(r));m=round(s*(1<<31))
 if m==(1<<31):m//=2;e+=1
 return int(max(-(1<<31),min((1<<31)-1,m))),int(e)
class FQConv(torch.nn.Module):
 def __init__(self,m,l):
  super().__init__();self.weight=m.weight;self.bias=m.bias;self.stride=m.stride;self.padding=m.padding;self.dilation=m.dilation;self.groups=m.groups;self.ins=float(l['input_scale']);self.outs=float(l['output_scale'])
 def fq(self,x,s):q=torch.clamp(torch.round(x/s),-127,127)*s;return x+(q-x).detach()
 def forward(self,x):
  x=self.fq(x,self.ins);w=self.weight;ws=w.detach().flatten(1).abs().amax(1).clamp_min(1e-12)/127;sh=[w.shape[0]]+[1]*(w.ndim-1);qw=torch.clamp(torch.round(w/ws.reshape(sh)),-127,127)*ws.reshape(sh);qw=w+(qw-w).detach();return self.fq(F.conv2d(x,qw,self.bias,self.stride,self.padding,self.dilation,self.groups),self.outs)
def replace(model):
 it=iter(MAN['layers']);n=0
 def rec(m):
  nonlocal n
  for k,c in list(m.named_children()):
   if isinstance(c,torch.nn.Conv2d):m._modules[k]=FQConv(c,next(it));n+=1
   else:rec(c)
 rec(model);assert n==63
def evaluate(m,t,ldr,d):
 m.eval();t.eval();ok=ag=tot=0
 with torch.no_grad():
  for x,y in ldr:x=x.to(d);y=y.to(d);p=m(x).argmax(1);q=t(x).argmax(1);ok+=(p==y).sum().item();ag+=(p==q).sum().item();tot+=y.numel()
 return {'pixel_accuracy':ok/tot,'agreement_vs_fp32':ag/tot}
def export(m):
 pkg=OUT/'package';
 for d in ['weights_int8','biases_int32','fixed_point']:(pkg/d).mkdir(parents=True,exist_ok=True)
 nm=copy.deepcopy(MAN);mods=[x for x in m.modules() if isinstance(x,FQConv)]
 for l,x in zip(nm['layers'],mods):
  w=x.weight.detach().cpu();oc=w.shape[0];ws=w.flatten(1).abs().amax(1).clamp_min(1e-12)/127;q=torch.clamp(torch.round(w/ws.reshape(oc,*([1]*(w.ndim-1)))),-127,127).to(torch.int8).numpy();b=x.bias.detach().cpu().numpy();bi=np.clip(np.round(b/(float(l['input_scale'])*ws.numpy())),np.iinfo(np.int32).min,np.iinfo(np.int32).max).astype(np.int32);mu=np.zeros(oc,np.int32);sh=np.zeros(oc,np.int32)
  for i,r in enumerate(float(l['input_scale'])*ws.numpy()/float(l['output_scale'])):mu[i],sh[i]=qm(r)
  ident=l['identifier'];spec=[('weight_file','weights_int8',f'{ident}.bin',q),('bias_file','biases_int32',f'{ident}.bin',bi),('multiplier_file','fixed_point',f'{ident}_multipliers.bin',mu),('shift_file','fixed_point',f'{ident}_shifts.bin',sh)]
  for key,folder,fn,a in spec:a.tofile(pkg/folder/fn);l[key]=f'{folder}/{fn}'
 (pkg/'corrected_int8_manifest.json').write_text(json.dumps(nm,indent=2)+'\n');return pkg
def main():
 d=torch.device('cuda');base=cloudsen12.load_model_by_name(name='UNetMobV2_V2',weights_folder='weights',device=d).model;teacher=copy.deepcopy(base).eval();fold_bn(base);replace(base); ck=torch.load(CK,map_location=d,weights_only=False)
 state={
  k:v for k,v in ck['student_state_dict'].items()
  if not k.endswith('.input_scale')
  and not k.endswith('.output_scale')
 }
 result=base.load_state_dict(state,strict=True)
 assert len(result.missing_keys)==0
 assert len(result.unexpected_keys)==0
 print('STRICT LOAD OK: missing=0 unexpected=0')
 tr=DataLoader(DS('qat/data/train'),batch_size=4,shuffle=True,num_workers=0);va=DataLoader(DS('qat/data/validation'),batch_size=4,num_workers=0);initial=evaluate(base,teacher,va,d);print('initial',json.dumps(initial));opt=torch.optim.AdamW(base.parameters(),lr=2e-8,weight_decay=1e-6);best=initial['pixel_accuracy'];torch.save({'epoch':0,'student_state_dict':base.state_dict(),'metrics':initial},OUT/'best.pth');hist=[dict(epoch=0,**initial)]
 for ep in range(1,4):
  base.train();ls=[];st=time.time()
  for i,(x,y) in enumerate(tr,1):
   x=x.to(d);y=y.to(d);opt.zero_grad(set_to_none=True)
   with torch.no_grad():tl=teacher(x);tp=tl.argmax(1)
   sl=base(x);loss=.70*F.cross_entropy(sl,tp)+.25*F.kl_div(F.log_softmax(sl/2,1),F.softmax(tl/2,1),reduction='batchmean')*4+.05*F.cross_entropy(sl,y);loss.backward();torch.nn.utils.clip_grad_norm_(base.parameters(),.5);opt.step();ls.append(loss.item())
   if i%50==0:print(f'epoch={ep}/3 batch={i}/{len(tr)} loss={loss.item():.4f}',flush=True)
  met=evaluate(base,teacher,va,d);met.update(epoch=ep,loss=float(np.mean(ls)),seconds=time.time()-st);hist.append(met);print(json.dumps(met,indent=2))
  if met['pixel_accuracy']>best:best=met['pixel_accuracy'];torch.save({'epoch':ep,'student_state_dict':base.state_dict(),'metrics':met},OUT/'best.pth')
 bc=torch.load(OUT/'best.pth',map_location=d,weights_only=False);base.load_state_dict(bc['student_state_dict']);pkg=export(base);res={'strict_load':True,'missing':0,'unexpected':0,'best':bc['metrics'],'history':hist,'package':str(pkg)};(OUT/'result.json').write_text(json.dumps(res,indent=2)+'\n');print('COMPLETE',json.dumps(res,indent=2))
if __name__=='__main__':main()

