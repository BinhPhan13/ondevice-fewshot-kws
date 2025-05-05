from typing import Any, Dict

Json = Dict[str, Any]
JsonStr = Dict[str, str]

def filter_opt(opt, tag):
    ret = { }

    for k,v in opt.items():
        tokens = k.split('.')
        if tokens[0] == tag:
            ret['.'.join(tokens[1:])] = v

    return ret


def npow2(x: int):
    if x < 1: return 1
    is_pow2 = (x & (x-1)) == 0
    return x if is_pow2 else 1 << x.bit_length()

