"""Malformed files are rejected before allocation; legacy valid inputs retain behavior."""
from pathlib import Path
import os,struct,subprocess,sys,json

def matrix(rows,cols,trans=0,data=()):
    return struct.pack('=iii',rows,cols,trans)+struct.pack('='+('f'*len(data)),*data)

def main(binary,root,strict=True):
    # The caller explicitly chooses the fixture/output directory.
    root=Path(root).resolve()
    root.mkdir(parents=True,exist_ok=True)
    if os.name=='posix':
        import resource
        resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    values=(1,4,2,5,3,6);normal=matrix(2,3,0,values);trans=matrix(2,3,1,values)
    cases={
      'valid':(normal,0),'transposed':(trans,0),'empty':(matrix(0,3),0),
      'file_sequence':(normal+trans,0),'pipe':(normal,0),
      'stream_lifetime':(None,0),'stream_empty':(matrix(0,3),0),
      'missing':(None,1),'truncated_header':(b'\0'*7,1),
      'empty_truncated_header':(struct.pack('=ii',0,3),1),
      'invalid_transpose':(matrix(2,3,2,values),1),'negative':(matrix(-1,3),1),
      'short_payload':(normal[:-1],1),'file_stream':(normal[:-1],1),
      'huge_header':(matrix(2147483647,2147483647),1),
      'double_overflow':(matrix(2147483647,2147483647),1),'null_stream':(None,1),
      'csv':(None,0),'csv_negative':(None,1),'csv_missing':(None,1),
      'loader_valid':(None,0),'loader_mismatch':(None,1),'loader_label_rows':(None,1),
      'loader_empty':(None,1),'loader_zero_features':(None,1),'loader_zero_batch':(None,1),
      'loader_short_input':(None,1),'loader_short_labels':(None,1),'loader_short_header':(None,1),
      'loader_negative':(None,1),'loader_transposed':(None,1),'loader_missing':(None,1)}
    result=[]
    for name,(payload,expected) in cases.items():
        d=root/name;d.mkdir(exist_ok=True)
        if payload is not None:(d/'matrix.bin').write_bytes(payload)
        if name in ('csv','csv_negative'):(d/'matrix.csv').write_text('1,2,9\n3,4,8\n5,6,7\n')
        if name.startswith('loader') and name!='loader_missing':
            n=0 if name=='loader_empty' else 5;dim=0 if name=='loader_zero_features' else 2
            features=[v for i in range(n) for v in (float(i),float(100+i))][:n*dim]
            x=matrix(dim,n,int(name=='loader_transposed'),features)
            ny=6 if name=='loader_mismatch' else n;dy=2 if name=='loader_label_rows' else 1
            y=struct.pack('=iii',dy,ny,0)+struct.pack('='+('i'*(dy*ny)),*range(dy*ny))
            if name=='loader_negative':x=matrix(-1,n)
            if name=='loader_short_input':x=x[:-1]
            if name=='loader_short_labels':y=y[:-1]
            if name=='loader_short_header':x=x[:5]
            (d/'train_x.matrix').write_bytes(x);(d/'train_y.matrix').write_bytes(y)
        env=dict(os.environ,JUZHEN_IO_CASE=name,JUZHEN_IO_ROOT=str(d))
        try:
            p=subprocess.run([str(binary)],env=env,capture_output=True,text=True,timeout=15)
            actual=p.returncode;output=p.stdout+p.stderr
        except subprocess.TimeoutExpired:actual='timeout';output='timeout'
        ok=actual==expected or (name=='pipe' and actual==77)
        (d/'output.txt').write_text(output)
        result.append(dict(case=name,expected=expected,actual=actual,passed=ok));print(('PASS' if ok else 'FAIL'),name,actual,flush=True)
    (root/'results.json').write_text(json.dumps(result,indent=2))
    if strict:assert all(x['passed'] for x in result),[x for x in result if not x['passed']]
    return result
if __name__=='__main__':main(Path(sys.argv[1]).resolve(),sys.argv[2],strict='--record-only' not in sys.argv)
