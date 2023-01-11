kernels = [
]
from dpu import DpuSet
import numpy as np
from dataclasses import dataclass
from typing import NamedTuple, Union, Callable, Tuple
i8 = i16 = i32 = i64 = u8 = u16 = u32 = u64 = int; f32 = f64 = float; char = string = str

def main():
    v0 = 1
    v1 = 2
    v2 = 3
    print((v0, v1, v2))
    v3 = v2 * 2
    del v2
    v4 = v1 * 2
    del v1
    v5 = v0 * 2
    del v0
    print((v5, v4, v3))
    del v3, v4, v5
    return 

if __name__ == '__main__': print(main())
