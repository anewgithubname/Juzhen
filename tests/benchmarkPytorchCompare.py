"""Compare Juzhen and PyTorch Transformer training-step performance.

Numerical parity is delegated to testTransformerTorch.py so weight loading,
dump parsing, tolerances, and error reporting have a single implementation.
"""
import os, re, subprocess, sys, time
try:
    import torch
except ImportError as exc:
    print(f"SKIP: {exc}"); sys.exit(77)
sys.path.insert(0, os.path.dirname(__file__))
from demo_transformer import TransformerBlock

def run(cmd, env=None):
    p=subprocess.run(cmd, text=True, capture_output=True, env=env)
    print(p.stdout, end="")
    if p.returncode: print(p.stderr, end="", file=sys.stderr); raise RuntimeError(cmd)
    return p.stdout

def parity(dump_exe, root):
    run([dump_exe])
    dump_path=os.path.join(root,"res","transformer_torch_dump.bin")
    parity_script=os.path.join(root,"tests","testTransformerTorch.py")
    output=run([sys.executable,parity_script,dump_path])
    errors=re.findall(r"max_abs=([0-9.eE+-]+)",output)
    if len(errors) != 2:
        raise RuntimeError("could not parse parity metrics from testTransformerTorch.py")
    return float(errors[0]),float(errors[1])

def main():
    train_exe,dump_exe,root=sys.argv[1:4]
    env=os.environ.copy(); env.update(JUZHEN_BENCH_WARMUP="3",JUZHEN_BENCH_ITERS="20")
    text=run([train_exe],env); fields=dict(re.findall(r"(\w+)=([^ ]+)",re.search(r"^RESULT .+$",text,re.M).group()))
    device="cuda" if fields["backend"]=="CUDA" and torch.cuda.is_available() else "cpu"
    torch.manual_seed(42); d=dk=128; ff=512; seq=64; batch=2; heads=4
    model=TransformerBlock(d,dk,ff,seq,heads).to(device); opt=torch.optim.Adam(model.parameters(),lr=1e-4)
    x=torch.randn(seq*batch,d,device=device); g=torch.randn_like(x)
    def step():
        opt.zero_grad(set_to_none=True); y=model(x); (y*g).sum().backward(); opt.step()
    for _ in range(3): step()
    if device=="cuda": torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    samples=[]
    for _ in range(20):
        if device=="cuda":
            a=torch.cuda.Event(True); b=torch.cuda.Event(True); a.record(); step(); b.record(); b.synchronize(); samples.append(a.elapsed_time(b))
        else:
            a=time.perf_counter(); step(); samples.append((time.perf_counter()-a)*1000)
    samples.sort(); mean=sum(samples)/len(samples); peak=torch.cuda.max_memory_allocated()/2**20 if device=="cuda" else 0.0
    fwd,bwd=parity(dump_exe,root)
    print(f"PYTORCH device={device} mean_ms={mean:.3f} p50_ms={samples[len(samples)//2]:.3f} p95_ms={samples[int(.95*(len(samples)-1))]:.3f} tokens_per_second={seq*batch*1000/mean:.3f} peak_device_mb={peak:.3f}")
    print(f"COMPARISON speed_ratio_pytorch_over_juzhen={mean/float(fields['mean_ms']):.3f} forward_max_abs={fwd:.3e} backward_max_abs={bwd:.3e}")
    return 0 if fwd<=1e-4 and bwd<=1e-4 else 1
if __name__=="__main__": sys.exit(main())
