"""PyTorch oracle for every elementwise/reduce call form in the C++ dump."""
import os,struct,sys
try:
    import numpy as np
    import torch
except ImportError as exc:
    print(f"SKIP: {exc}"); sys.exit(77)

def read_matrix(f):
    r,c=struct.unpack("<ii",f.read(8))
    return np.frombuffer(f.read(4*r*c),dtype="<f4").reshape(c,r).T.copy()

def compare(got,expected,name,tol=1e-6):
    expected=expected.detach().cpu().numpy()
    maximum=float(np.max(np.abs(got-expected)))
    print(f"[{'PASS' if maximum<=tol else 'FAIL'}] {name}: shape={got.shape} max_abs={maximum:.3e}")
    return maximum<=tol

def main():
    path=sys.argv[1] if len(sys.argv)>1 else os.path.join(os.path.dirname(__file__),"..","res","elementwise_reduce_torch_dump.bin")
    with open(path,"rb") as f:
        if f.read(8)!=b"JZERDMP1": raise RuntimeError("invalid dump")
        x,elem_l,elem_r,red_l0,red_l1,red_r0,red_r1=[read_matrix(f) for _ in range(7)]
    t=torch.from_numpy(x)
    ew=t*t+2.0*t-0.5
    # Juzhen dim=0 reduces rows for every column; dim=1 reduces columns for every row.
    sum0=t.sum(dim=0,keepdim=True)
    sum1=t.sum(dim=1,keepdim=True)
    stats0=torch.stack((t.sum(dim=0),t.max(dim=0).values),dim=0)
    stats1=torch.stack((t.sum(dim=1),t.max(dim=1).values),dim=1)
    checks=[(elem_l,ew,"elementwise(const Matrix&)"),(elem_r,ew,"elementwise(Matrix&&)"),
            (red_l0,sum0,"reduce lvalue dim=0 k=1"),(red_l1,sum1,"reduce lvalue dim=1 k=1"),
            (red_r0,stats0,"reduce rvalue->const& dim=0 k=2"),(red_r1,stats1,"reduce rvalue->const& dim=1 k=2")]
    return 0 if all(compare(*item) for item in checks) else 1
if __name__=="__main__": sys.exit(main())
