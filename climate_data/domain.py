"""核对有限因子、道路身份及允许状态引用。"""
from .io import unique,fraction

def summarize(v):
    factors=unique(v['factors'],'id');known={}
    for f in v['factors']:
        states=unique(f['states'],'id');known[f['id']]=states
        probs=[fraction(s['probability']) for s in f['states']]
        if any(p<0 for p in probs) or sum(probs)!=1:raise ValueError('状态概率无效')
    nodes=set(v['nodes'])
    if len(nodes)!=len(v['nodes']) or not set(v['terminals'])<=nodes or len(set(v['terminals']))<2:raise ValueError('节点或终端不合法')
    edges=unique(v['edges'],'id')
    for e in v['edges']:
        if e['u'] not in nodes or e['v'] not in nodes or e['u']==e['v']:raise ValueError('道路端点不合法')
        for factor,states in e['allowed'].items():
            if factor not in factors or not set(states)<=known[factor]:raise ValueError('道路因子引用无效')
    for s in v['fortifications']:
        if not set(s['edges'])<=edges:raise ValueError('加固道路未知')
    return {'nodes':len(nodes),'edges':len(edges),'factors':len(factors),'factor_state_space':__import__('math').prod(len(s) for s in known.values()),'terminals':sorted(v['terminals'])}
