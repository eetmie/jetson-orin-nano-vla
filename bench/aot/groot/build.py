"""Build checked GR00T attention replacements in an isolated cache, one graph per process."""
import argparse,gc,json,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from candidate import pin,sha256,validate_base


def build_one(a):
    import tensorrt as trt
    import tensorrt.plugin as trtp
    import plugin
    from bench.vendor.groot_trt import configure_accumulate
    T=trt.LayerType
    logger=trt.Logger(trt.Logger.WARNING)
    builder=trt.Builder(logger)
    net=builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser=trt.OnnxParser(net,logger)
    assert parser.parse_from_file(str(a.bundle/(a.graph+'.onnx'))),[str(parser.get_error(i)) for i in range(parser.num_errors)]
    layers=[net.get_layer(i) for i in range(net.num_layers)]
    producers={l.get_output(j).name:l for l in layers for j in range(l.num_outputs)}
    classes={T.SHUFFLE:trt.IShuffleLayer,T.SLICE:trt.ISliceLayer,T.CONSTANT:trt.IConstantLayer,
             T.ELEMENTWISE:trt.IElementWiseLayer,T.UNARY:trt.IUnaryLayer,T.CAST:trt.ICastLayer,
             T.MATRIX_MULTIPLY:trt.IMatrixMultiplyLayer,T.CONCATENATION:trt.IConcatenationLayer,T.SOFTMAX:trt.ISoftMaxLayer,
             T.SELECT:trt.ISelectLayer,T.FILL:trt.IFillLayer}
    np_types={trt.float16:np.float16,trt.float32:np.float32,trt.int64:np.int64,trt.int32:np.int32,trt.bool:np.bool_}
    def cls(l):
        l.__class__=classes.get(l.type,l.__class__);return l
    def users(t):
        return [(l,j) for l in layers for j in range(l.num_inputs) if l.get_input(j) is not None and l.get_input(j).name==t.name]
    def evaluate(t):
        l=cls(producers[t.name]);ins=[l.get_input(i) for i in range(l.num_inputs)]
        if l.type==T.CONSTANT:return np.asarray(l.weights).reshape(tuple(t.shape))
        if l.type==T.SHAPE:
            assert all(d>=0 for d in ins[0].shape);return np.array(tuple(ins[0].shape),np.int64)
        if l.type==T.CONCATENATION:return np.concatenate([evaluate(i) for i in ins],axis=l.axis)
        x=evaluate(ins[0])
        if l.type==T.CAST:return x.astype(np_types[t.dtype])
        if l.type in (T.SHUFFLE,T.SLICE,T.UNSQUEEZE,T.SQUEEZE):return emulate(l,x)
        if l.type==T.UNARY:
            assert l.op==trt.UnaryOperation.SQRT;return np.sqrt(x).astype(np_types[t.dtype])
        if l.type==T.ELEMENTWISE:
            ops={trt.ElementWiseOperation.DIV:np.divide,trt.ElementWiseOperation.PROD:np.multiply,
                 trt.ElementWiseOperation.SUM:np.add,trt.ElementWiseOperation.SUB:np.subtract,
                 trt.ElementWiseOperation.MIN:np.minimum,trt.ElementWiseOperation.MAX:np.maximum,
                 trt.ElementWiseOperation.EQUAL:np.equal}
            return ops[l.op](x,evaluate(ins[1])).astype(np_types[t.dtype])
        if l.type==T.SELECT:
            return np.where(x,evaluate(ins[1]),evaluate(ins[2])).astype(np_types[t.dtype])
        raise AssertionError((l.name,l.type,'nonconstant expression'))
    def emulate(l,x):
        l=cls(l)
        if l.type==T.CAST:return x  # Index maps track layout, independently of numerical casts.
        if l.type in (T.UNSQUEEZE,T.SQUEEZE):return x.reshape(tuple(l.get_output(0).shape))
        if l.type==T.SLICE:
            assert l.mode in (trt.SampleMode.STRICT_BOUNDS,trt.SampleMode.WRAP),l.name
            def dims(i,static):return tuple(int(v) for v in evaluate(l.get_input(i)).ravel()) if l.num_inputs>i and l.get_input(i) is not None else tuple(static())
            start=dims(1,lambda:l.start);size=dims(2,lambda:l.shape);stride=dims(3,lambda:l.stride)
            try:axes=dims(5,lambda:l.axes)
            except ValueError:axes=tuple(range(x.ndim))
            if not axes:axes=tuple(range(x.ndim))
            assert len(axes)==len(start)==len(size)==len(stride),(l.name,axes,start,size,stride)
            for axis,s,n,k in zip(axes,start,size,stride):
                axis=axis%x.ndim
                ix=s+np.arange(n)*k
                if l.mode==trt.SampleMode.WRAP:ix=ix%x.shape[axis]
                assert not len(ix) or (ix.min()>=0 and ix.max()<x.shape[axis]),(l.name,start,size,stride,x.shape,tuple(l.get_output(0).shape),axis)
                x=np.take(x,ix,axis=axis)
            assert x.shape==tuple(l.get_output(0).shape),(l.name,x.shape,tuple(l.get_output(0).shape))
            return x
        assert l.type==T.SHUFFLE,l.name
        x=x.transpose(tuple(l.first_transpose)[:x.ndim])
        if l.num_inputs>1 and l.get_input(1) is not None:shape=tuple(int(v) for v in evaluate(l.get_input(1)).ravel())
        else:
            try:shape=tuple(l.reshape_dims)
            except ValueError:shape=()
        if shape:
            if l.zero_is_placeholder:shape=tuple(x.shape[i] if d==0 else d for i,d in enumerate(shape))
            x=x.reshape(shape)
        x=x.transpose(tuple(l.second_transpose)[:x.ndim])
        assert x.shape==tuple(l.get_output(0).shape),(l.name,x.shape,l.get_output(0).shape)
        return x
    views=(T.SHUFFLE,T.CAST,T.SLICE,T.UNSQUEEZE,T.SQUEEZE)
    def native(t,allow_scale=False):
        chain=[];scale=1.
        while t.name in producers:
            l=cls(producers[t.name])
            if l.type in views:chain.append(l);t=l.get_input(0);continue
            if allow_scale and l.type==T.ELEMENTWISE and l.op==trt.ElementWiseOperation.PROD:
                x,c=l.get_input(0),l.get_input(1)
                if np.prod(c.shape)!=1:x,c=c,x
                assert np.prod(c.shape)==1
                scale*=float(evaluate(c).ravel()[0]);t=x;continue
            break
        indices=np.arange(np.prod(t.shape),dtype=np.int64).reshape(tuple(t.shape))
        for l in reversed(chain):indices=emulate(l,indices)
        return t,indices,scale
    def flat(t,b,n,w):
        assert tuple(t.shape) in [(b,n,w),(b,n,w//plugin.SPECS[key]['D'],plugin.SPECS[key]['D'])],tuple(t.shape)
        if len(t.shape)==3:return t
        l=net.add_shuffle(t);l.reshape_dims=(b,n,w);return l.get_output(0)
    family='n17' if json.loads((a.bundle/'bundle.json').read_text())['schema'].startswith('groot-n1.7') else 'n16'
    is_vision=a.graph.startswith('vision')
    is_llm=a.graph.startswith('llm')
    patches=[]
    softmaxes=[cls(l) for l in layers if l.type==T.SOFTMAX]

    def strides(m,name):
        """(head stride, row stride, heads per K/V head) of an index map [1,H,S,D] whose
        rows and channels are affine in the root tensor; refuses anything else."""
        m=m[0];H_,S_,D_=m.shape;g=1
        while g<H_ and m[g,0,0]==m[0,0,0]:g+=1
        sh=int(m[g,0,0]-m[0,0,0]) if g<H_ else 0;ss=int(m[0,1,0]-m[0,0,0])
        expect=(np.arange(H_)//g)[:,None,None]*sh+np.arange(S_)[None,:,None]*ss+np.arange(D_)[None,None,:]
        assert m[0,0,0]==0 and np.array_equal(m,expect),(name,'layout not affine')
        return sh,ss,g

    def patch_llm(soft):
        """Causal GQA self-attention of an LLM chunk -> kernels.causal_aot."""
        cast=cls(producers[soft.get_input(0).name])
        assert cast.type==T.CAST and cast.get_input(0).dtype==trt.float16
        add=cls(producers[cast.get_input(0).name])
        assert add.type==T.ELEMENTWISE and add.op==trt.ElementWiseOperation.SUM
        x,mask=add.get_input(0),add.get_input(1)
        if cls(producers[x.name]).type==T.CONSTANT:x,mask=mask,x
        mval=evaluate(mask).astype(np.float16);S_=mval.shape[-1]
        tri=np.triu(np.full((S_,S_),-1e4,np.float16),1)
        assert mval.shape==(1,1,S_,S_) and np.array_equal(mval[0,0],tri),'mask is not the causal triu(-1e4)'
        mul=cls(producers[x.name])
        assert mul.type==T.ELEMENTWISE and mul.op==trt.ElementWiseOperation.PROD
        left,c=mul.get_input(0),mul.get_input(1)
        if np.prod(c.shape)!=1:left,c=c,left
        scale=float(evaluate(c).ravel()[0]);mm=cls(producers[left.name])
        assert mm.type==T.MATRIX_MULTIPLY and mm.op0==mm.op1==trt.MatrixOperation.NONE
        q,qmap,qs=native(mm.get_input(0));k,kmap,ks=native(mm.get_input(1))
        assert qs==ks==1.
        prob=soft.get_output(0);dtypes=[]
        while True:
            (l,slot),=users(prob);l=cls(l)
            if l.type==T.MATRIX_MULTIPLY:pv=l;break
            if l.type==T.SHUFFLE:     # parser reshapes; only identities allowed
                ids=np.arange(np.prod(prob.shape)).reshape(tuple(prob.shape))
                assert np.array_equal(emulate(l,ids),ids),(l.name,'non-identity shuffle')
                prob=l.get_output(0);continue
            assert l.type in (T.CAST,T.IDENTITY),(l.name,l.type)
            prob=l.get_output(0);dtypes.append(prob.dtype)
        # FP32 softmax -> FP16 first, then only lossless FP16<->FP32 hops: probabilities are FP16.
        assert dtypes and dtypes[0]==dtypes[-1]==trt.float16 and set(dtypes)<={trt.float16,trt.float32}
        assert slot==0 and pv.op0==pv.op1==trt.MatrixOperation.NONE
        v,vmap,vs=native(pv.get_input(1));assert vs==1.
        _,H_,_,D_=qmap.shape
        sqh,sqs,gq=strides(qmap,'Q');skh,sks,g=strides(kmap.transpose(0,1,3,2),'K');svh,svs,gv=strides(vmap,'V')
        assert gq==1 and g==gv and H_%g==0 and q.dtype==k.dtype==v.dtype==trt.float16
        output=pv.get_output(0);outmap=np.arange(H_*S_*D_).reshape(1,H_,S_,D_)
        while tuple(output.shape)!=(1,S_,H_*D_):
            (l,_),=users(output);outmap=emulate(l,outmap);output=l.get_output(0)
        assert np.array_equal(outmap,np.arange(H_*S_*D_).reshape(1,H_,S_,D_).transpose(0,2,1,3).reshape(1,S_,H_*D_)),'output layout'
        key=plugin.register_causal(family,S_,H_,g,D_,scale,(sqh,sqs,skh,sks,svh,svs))
        custom=net.add_plugin(getattr(trtp.op.nano_vla,plugin.op_name(key,True))(q,k,v,q),aot=True)
        custom.name='triton_'+soft.name
        for l,slot in users(output):l.set_input(slot,custom.get_output(0))
        patches.append(dict(softmax=soft.name,shape=key,q=q.name,k=k.name,v=v.name,replaced=output.name))

    for soft in softmaxes if a.mode!='baseline' else []:
        if is_llm:
            if a.mode in ('llm','dit-llm','all'):patch_llm(soft)
            continue
        assert soft.axes==(1<<(len(soft.get_input(0).shape)-1)),soft.name
        cast=cls(producers[soft.get_input(0).name]);assert cast.type==T.CAST and cast.get_input(0).dtype==trt.float16
        score=cls(producers[cast.get_input(0).name]);bias=None;score_scale=1.
        if is_vision:
            assert score.type==T.ELEMENTWISE and score.op==trt.ElementWiseOperation.PROD
            left,right=score.get_input(0),score.get_input(1)
            if np.prod(right.shape)!=1:left,right=right,left
            score_scale=float(evaluate(right).ravel()[0]);mm=cls(producers[left.name])
        else:
            if score.type==T.ELEMENTWISE:
                assert score.op==trt.ElementWiseOperation.SUM
                left,bias=score.get_input(0),score.get_input(1)
                if producers[left.name].type!=T.MATRIX_MULTIPLY:left,bias=bias,left
                mm=cls(producers[left.name])
            else:mm=score
        assert mm.type==T.MATRIX_MULTIPLY and mm.op0==mm.op1==trt.MatrixOperation.NONE
        q,qmap,qs=native(mm.get_input(0),not is_vision)
        k,kmap,ks=native(mm.get_input(1),not is_vision)
        # Gathered bundles: cross-attention reads a kv_* input and carries no mask.
        gathered=not is_vision and bias is None and k.name.startswith('kv_')
        cross=bias is not None or gathered
        if a.mode=='vision' and not is_vision:continue
        if a.mode=='dit-cross' and (is_vision or not cross):continue
        if a.mode in ('dit-all','dit-llm') and is_vision:continue
        if a.mode=='llm':continue
        if gathered:key=plugin.register_gathered(family,int(k.shape[1]))
        else:key=family+('_vision' if is_vision else '_cross' if cross else '_self')
        spec=plugin.SPECS[key];b,nq,nk,h,d=[spec[x] for x in ['B','NQ','NK','H','D']];w=h*d
        assert tuple(mm.get_input(0).shape)==(b,h,nq,d)
        assert tuple(mm.get_input(1).shape)==(b,h,d,nk)
        assert qs==ks==spec['QK_SCALE'] and score_scale==spec['SCORE_SCALE'],(qs,ks,score_scale,key)
        prob=soft.get_output(0)
        while True:
            (l,_),=users(prob);l=cls(l);prob=l.get_output(0)
            if l.type==T.CAST:
                assert prob.dtype==trt.float16;break
            assert l.type==T.SHUFFLE and tuple(prob.shape)==tuple(soft.get_output(0).shape)
        (pv,slot),=users(prob);pv=cls(pv)
        assert slot==0 and pv.type==T.MATRIX_MULTIPLY and pv.op0==pv.op1==trt.MatrixOperation.NONE
        v,vmap,vs=native(pv.get_input(1));assert vs==1.
        qids=np.arange(b*nq*w).reshape(b,nq,h,d)
        assert np.array_equal(qmap,qids.transpose(0,2,1,3)),(key,'Q layout')
        if cross:
            assert k.name==v.name and tuple(k.shape)==(b,nk,2*w)
            assert k.dtype==(trt.float16 if spec.get('KV_HALF') else trt.float32),(key,k.dtype)
            ids=np.arange(b*nk*2*w).reshape(b,nk,2,h,d)
            assert np.array_equal(kmap,ids[:,:,0].transpose(0,2,3,1)),(key,'packed K layout')
            assert np.array_equal(vmap,ids[:,:,1].transpose(0,2,1,3)),(key,'packed V layout')
        if gathered:
            bv=vv=flat(q,b,nq,w)         # unused placeholders: no mask, K|V packed
        elif cross:
            biasroot,biasmap,bs=native(bias)
            assert biasroot.name in ['text_bias','image_bias'] and tuple(biasroot.shape)==(1,1,nk) and biasroot.dtype==trt.float32
            assert bs==1. and np.array_equal(biasmap,np.broadcast_to(np.arange(nk),biasmap.shape)),(key,'bias broadcast')
            bv=biasroot;vv=flat(q,b,nq,w)
        else:
            ids=np.arange(b*nk*w).reshape(b,nk,h,d)
            assert np.array_equal(kmap,ids.transpose(0,2,3,1)),(key,'K layout')
            if spec.get('V_PACKED'):
                packed_ids=np.arange(b*nk*3*w).reshape(b,nk,3,h,d)
                assert tuple(v.shape)==(b,nk,3*w) and np.array_equal(vmap,packed_ids[:,:,2].transpose(0,2,1,3)),(key,'packed V layout')
                vv=v
            else:
                assert np.array_equal(vmap,ids.transpose(0,2,1,3)),(key,'V layout')
                vv=flat(v,b,nk,w)
            k=flat(k,b,nk,w);bv=vv
        q=flat(q,b,nq,w)
        assert q.dtype==vv.dtype==trt.float16
        output=pv.get_output(0);outmap=np.arange(b*h*nq*d).reshape(b,h,nq,d)
        while tuple(output.shape)!=(b,nq,w):
            (l,_),=users(output);outmap=emulate(l,outmap);output=l.get_output(0)
        assert np.array_equal(outmap,np.arange(b*h*nq*d).reshape(b,h,nq,d).transpose(0,2,1,3).reshape(b,nq,w)),(key,'output layout')
        custom=net.add_plugin(getattr(trtp.op.nano_vla,plugin.op_name(key,a.two_pass or spec.get('EXACT',False)))(q,k,vv,bv),aot=True)
        custom.name='triton_'+soft.name
        for l,slot in users(output):l.set_input(slot,custom.get_output(0))
        patches.append(dict(softmax=soft.name,shape=key,q=q.name,k=k.name,bias=bv.name,replaced=output.name))
    if a.mode!='baseline':assert patches,(a.graph,a.mode,'no eligible attention')
    if a.check_only:
        print(json.dumps(dict(graph=a.graph,patches=patches,status='graph_matching_pass')),flush=True)
        return
    cfg=builder.create_builder_config();cfg.builder_optimization_level=2
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,512<<20)
    cfg.profiling_verbosity=trt.ProfilingVerbosity.DETAILED
    accumulated=configure_accumulate(net,cfg,a.accumulate)
    timing=a.base_cache/'timing.cache'
    cfg.set_timing_cache(cfg.create_timing_cache(timing.read_bytes() if timing.exists() else b''),ignore_mismatch=False)
    start=time.time();blob=builder.build_serialized_network(net,cfg)
    assert blob is not None,a.graph
    (a.out/(a.graph+'.engine')).write_bytes(blob)
    row=dict(graph=a.graph,patches=patches,fp32_accumulated_matmuls=accumulated,build_s=round(time.time()-start,2),engine_bytes=(a.out/(a.graph+'.engine')).stat().st_size)
    (a.out/(a.graph+'.build.json')).write_text(json.dumps(row,indent=2)+'\n')
    print(json.dumps(row),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--base-cache',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--mode',choices=['baseline','vision','dit-cross','dit-all','all','llm','dit-llm'],required=True)
    p.add_argument('--graphs',help='comma-separated graph names; defaults to all eligible graphs')
    p.add_argument('--check-only',action='store_true',help='validate one --graph without building')
    p.add_argument('--graph',help=argparse.SUPPRESS);p.add_argument('--two-pass',action='store_true')
    p.add_argument('--accumulate',choices=['fp32','auto'],default='fp32')
    a=p.parse_args();a.bundle=a.bundle.expanduser().resolve();a.base_cache=a.base_cache.expanduser().resolve();a.out=a.out.expanduser().resolve()
    if a.graph:build_one(a);return
    if a.check_only:p.error("--check-only requires --graph")
    from bench.vendor.groot_trt import _trt_version, _cudart_version
    stack=f"tensorrt={_trt_version()} cudart={_cudart_version()} opt_level=2 workspace_mb=512 strongly_typed"
    if a.accumulate != "auto":stack+=f" accumulate={a.accumulate}"
    validate_base(a.base_cache,a.bundle,a.accumulate,stack)
    meta=json.loads((a.bundle/'bundle.json').read_text());assert meta['schema'] in ['groot-n1.6-split/2','groot-n1.7-split/2']
    available=[g['name'] for g in meta['graphs']]
    names=a.graphs.split(',') if a.graphs else [n for n in available if (n.startswith('vision') and a.mode in ['vision','all','baseline']) or (n.startswith('dit') and a.mode in ['dit-cross','dit-all','all','baseline','dit-llm']) or (n.startswith('llm') and a.mode in ['llm','dit-llm','all'])]
    assert names and all(n in available and n.startswith(('vision','dit','llm')) for n in names)
    assert a.base_cache!=a.out
    a.out.mkdir(parents=True,exist_ok=False)
    for src in sorted(a.base_cache.iterdir()):
        if src.is_file() and src.name!='candidate.json' and not (src.suffix=='.engine' and src.stem in names):shutil.copy2(src,a.out/src.name)
    for name in names:
        command=[sys.executable,__file__,'--bundle',str(a.bundle),'--base-cache',str(a.base_cache),'--out',str(a.out),'--mode',a.mode,'--accumulate',a.accumulate,'--graph',name]
        if a.two_pass:command+=['--two-pass']
        subprocess.run(command,check=True)
    import tensorrt as trt
    here=Path(__file__).parent
    record=dict(mode=a.mode,two_pass=a.two_pass,accumulate=a.accumulate,bundle=str(a.bundle),base_cache=str(a.base_cache),tensorrt=trt.__version__,opt_level=2,
                graphs=names,platform_stack=stack,source_sha256_code={p.name:sha256(p) for p in here.glob('*.py')},
                precision='FP16 weights/activations; FP32 normalization and dot accumulation; '+('normalized' if a.two_pass else 'online unnormalized')+' probabilities rounded to FP16',**pin(a.out,a.bundle))
    (a.out/'candidate.json').write_text(json.dumps(record,indent=2)+'\n')
    print('DONE',a.mode,a.out,flush=True)

if __name__=='__main__':main()
