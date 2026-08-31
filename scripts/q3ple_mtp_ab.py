import ctypes,hashlib,json,os,socket,subprocess,threading,time,urllib.request
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'artifacts/models/AtomicChat/Qwen3.8-Flash-Next-AD-4.27bpw-Q3_PLE-M64/Qwen3.8-Flash-Next-AD-4.27bpw-Q3_PLE-M64-00001-of-00033.gguf'
SIDECAR=ROOT/'artifacts/models/Qwen3.8-Flash-Next-MTP-Q4_K_M-FC-HC/mtp-Qwen3.8-Flash-Next-Q4_K_M-FC-HC.gguf'
BIN=ROOT/'workstreams/llama.cpp-q3ple-mtp/build-win-cuda-q3ple-mtp/bin'; EXE=BIN/'llama-server.exe'
OUT=ROOT/'results/QWEN38-MTP-PROTOTYPE-001/q3ple_mtp_matched_ab.json'; LOG=ROOT/'logs/QWEN38-MTP-PROTOTYPE-001/q3ple_mtp_ab'; PORT=18086
EXPECTED='''A01 alpha bravo charlie delta echo
A02 foxtrot golf hotel india juliet
A03 kilo lima mike november oscar
A04 papa quebec romeo sierra tango
A05 uniform victor whiskey xray yankee
A06 zulu amber birch cedar denim
A07 ember frost granite harbor ivory
A08 jade kelp linen maple nickel
A09 olive pearl quartz river stone
A10 timber umber velvet willow xenon
A11 yellow azure bronze copper drift
A12 elm fern glass hazel iron
A13 juniper kite lemon moss navy
A14 opal pine reed slate teal
A15 urn violet wheat yarrow zinc
A16 acorn berry clover dune elder'''
SYSTEM='Follow the user instruction precisely. Output exactly the requested text with no extra whitespace or commentary.'
USER='Return exactly the text between BEGIN_EXPECTED and END_EXPECTED, excluding the marker lines.\nBEGIN_EXPECTED\n'+EXPECTED+'\nEND_EXPECTED'
def gpu():
 r=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')
 return {'used_mib':int(r[0]),'free_mib':int(r[1]),'util_pct':int(r[2])}
def snap(proc=None):
 v=psutil.virtual_memory(); s=psutil.swap_memory(); x={'t':time.time(),'ram_available':v.available,'swap_used':s.used,'gpu':gpu()}
 if proc:
  try:x.update({'rss':proc.memory_info().rss,'read':proc.io_counters().read_bytes})
  except:pass
 return x
def port_free():
 s=socket.socket()
 try:s.bind(('127.0.0.1',PORT)); return True
 except:return False
 finally:s.close()
def post(path,body,timeout=240):
 q=urllib.request.Request(f'http://127.0.0.1:{PORT}{path}',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(q,timeout=timeout) as r:return json.loads(r.read().decode())
def stop(p):
 if p.poll() is not None:return
 p.terminate()
 try:p.wait(15)
 except: p.kill();p.wait(10)
def set_working_set_cap(p,max_bytes):
 if os.name!='nt':raise RuntimeError('working-set cap is Windows-only')
 kernel32=ctypes.WinDLL('kernel32',use_last_error=True)
 set_ws=kernel32.SetProcessWorkingSetSizeEx
 set_ws.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_size_t,ctypes.c_uint32];set_ws.restype=ctypes.c_int
 handle=ctypes.c_void_p(int(p._handle));minimum=64*1024;flags=0x00000002|0x00000004
 ctypes.set_last_error(0)
 if not set_ws(handle,minimum,max_bytes,flags):raise ctypes.WinError(ctypes.get_last_error())
 get_ws=kernel32.GetProcessWorkingSetSizeEx
 get_ws.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_uint32)];get_ws.restype=ctypes.c_int
 actual_min=ctypes.c_size_t();actual_max=ctypes.c_size_t();actual_flags=ctypes.c_uint32()
 if not get_ws(handle,ctypes.byref(actual_min),ctypes.byref(actual_max),ctypes.byref(actual_flags)):raise ctypes.WinError(ctypes.get_last_error())
 return {'requested_max':max_bytes,'actual_min':actual_min.value,'actual_max':actual_max.value,'flags':actual_flags.value}
def base_args():
 return [str(EXE),'--model',str(MODEL),'--host','127.0.0.1','--port',str(PORT),'--ctx-size','16384','--parallel','1','--threads','8','--threads-batch','8','--batch-size','2048','--ubatch-size','512','--cache-type-k','q8_0','--cache-type-v','q8_0','--flash-attn','on','--load-mode','mmap','--tensor-read-lazy','auto','--n-gpu-layers','auto','--fit','on','--fit-target','1024','--n-cpu-moe','45','--split-mode','none','--device','CUDA0','--no-cache-prompt','--cache-ram','0','--no-warmup','--jinja','--reasoning','off','--reasoning-budget','0','--override-tensor',r'per_layer_token_embd\.weight=CPU','--log-verbosity','3']
def args_for(mode):
 a=base_args()
 if mode=='target': return a+['--spec-type','none']
 return a+['-md',str(SIDECAR),'--spec-type','draft-mtp','--spec-draft-n-max','2','--spec-draft-p-min','0.75','--spec-draft-device','none','--spec-draft-ngl','0','--spec-draft-cpu-moe','--spec-draft-type-k','f16','--spec-draft-type-v','f16']
def run(mode):
 pre=snap()
 if pre['ram_available']<40*1024**3 or pre['gpu']['free_mib']<8192 or pre['gpu']['util_pct']>15: raise RuntimeError(f'preflight {pre}')
 if not port_free(): raise RuntimeError('port busy')
 LOG.mkdir(parents=True,exist_ok=True); lp=LOG/f'{mode}.log'; lf=lp.open('w',encoding='utf-8')
 env=os.environ.copy(); tmp=ROOT/f'artifacts/.cache/q3ple-mtp-ab/{mode}'; tmp.mkdir(parents=True,exist_ok=True); env.update({'TEMP':str(tmp),'TMP':str(tmp),'HF_HOME':str(tmp/'hf')})
 p=subprocess.Popen(args_for(mode),cwd=BIN,env=env,stdout=lf,stderr=subprocess.STDOUT,text=True); proc=psutil.Process(p.pid); samples=[]; violation=[]; done=threading.Event(); cap_bytes=int(float(os.environ.get('QWEN38_WORKING_SET_CAP_GIB','0'))*1024**3); cap_events=[]
 def mon():
  while not done.wait(.25) and p.poll() is None:
   x=snap(proc);samples.append(x)
   if cap_bytes and not cap_events and x.get('rss',0)>=cap_bytes:
    try:cap_events.append({'t':time.time(),'rss_before':x.get('rss',0),**set_working_set_cap(p,cap_bytes)})
    except Exception as e:violation.append(f'working-set-cap-failed:{e}');stop(p);return
   if x['ram_available']<6*1024**3:violation.append('ram<6GiB');stop(p);return
   if x['gpu']['free_mib']<768:violation.append('vram<768MiB');stop(p);return
   if x.get('rss',0)>50*1024**3:violation.append('rss>50GiB');stop(p);return
   if x['swap_used']-pre['swap_used']>1024**3:violation.append('swap-growth>1GiB');stop(p);return
 t=threading.Thread(target=mon,daemon=True);t.start(); ready=False
 for _ in range(1200):
  if violation or p.poll() is not None:break
  try:
   with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health',timeout=.5) as r:
    if r.status==200:ready=True;break
  except:pass
  time.sleep(.25)
 if not ready: done.set();stop(p);lf.close();raise RuntimeError(f'not ready {violation}')
 runs=[]; req={'model':'model','messages':[{'role':'system','content':SYSTEM},{'role':'user','content':USER}],'temperature':0,'seed':38027,'max_tokens':256,'stream':False,'cache_prompt':False}
 for rep in (1,2):
  r=post('/v1/chat/completions',req); out=r['choices'][0]['message'].get('content') or ''; ids=post('/tokenize',{'content':out,'add_special':False}).get('tokens'); tm=r.get('timings') or {}
  runs.append({'repeat':rep,'prompt_tokens':r.get('usage',{}).get('prompt_tokens'),'completion_tokens':r.get('usage',{}).get('completion_tokens'),'prefill_tps':tm.get('prompt_per_second'),'decode_tps':tm.get('predicted_per_second'),'draft_n':tm.get('draft_n',0),'draft_n_accepted':tm.get('draft_n_accepted',0),'finish_reason':r['choices'][0].get('finish_reason'),'output_sha256':hashlib.sha256(out.encode()).hexdigest(),'ids_sha256':hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),'exact':out==EXPECTED})
  if violation:break
 done.set();t.join(2);stop(p);lf.close();time.sleep(2)
 peak={'min_ram':min([pre['ram_available']]+[x['ram_available'] for x in samples]),'min_vram':min([pre['gpu']['free_mib']]+[x['gpu']['free_mib'] for x in samples]),'max_rss':max([0]+[x.get('rss',0) for x in samples]),'max_swap':max([pre['swap_used']]+[x['swap_used'] for x in samples]),'swap_growth':max([pre['swap_used']]+[x['swap_used'] for x in samples])-pre['swap_used']}
 return {'mode':mode,'preflight':pre,'args':args_for(mode),'runs':runs,'peak':peak,'violation':violation,'working_set_cap_bytes':cap_bytes or None,'working_set_cap_events':cap_events,'port_free_after':port_free(),'log':str(lp)}
def main():
 if OUT.exists(): raise SystemExit('refusing overwrite')
 if not EXE.is_file() or not MODEL.is_file() or not SIDECAR.is_file(): raise SystemExit('missing artifact')
 target=run('target'); time.sleep(4); mtp=run('mtp') if all(x['exact'] for x in target['runs']) else {'mode':'mtp','skipped':'target parity failed'}
 rec={'schema':1,'target':target,'mtp':mtp}
 if 'runs' in mtp and len(target['runs'])==2 and len(mtp['runs'])==2:
  t=target['runs'][1];m=mtp['runs'][1]; rec['warm']={'target_decode':t['decode_tps'],'mtp_decode':m['decode_tps'],'decode_gain_pct':100*(m['decode_tps']/t['decode_tps']-1),'target_prefill':t['prefill_tps'],'mtp_prefill':m['prefill_tps'],'acceptance':m['draft_n_accepted']/m['draft_n'] if m['draft_n'] else None}
 OUT.write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps(rec.get('warm',rec),indent=2))
if __name__=='__main__':main()
