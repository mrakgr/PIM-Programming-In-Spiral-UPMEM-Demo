kernels = [
   r"""
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
__host int32_t v0;
__host int32_t v1;
int32_t main(){
    int32_t v2;
    v2 = v0 + v1;
    return 0l;
}
""",
   r"""
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
__host uint64_t v0;
__host uint64_t v1;
int32_t main(){
    uint64_t v2;
    v2 = v0 + v1;
    return 0l;
}
""",
   r"""
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
__host float v0;
__host float v1;
int32_t main(){
    float v2;
    v2 = v0 + v1;
    return 0l;
}
""",
]
from dpu import DpuSet
import numpy as np
from dataclasses import dataclass
from typing import NamedTuple, Union, Callable, Tuple
i8 = i16 = i32 = i64 = u8 = u16 = u32 = u64 = int; f32 = f64 = float; char = string = str

def main():
    v0 = 2
    v1 = 1
    v2 = 0
    print(v2,(v0, v1))
    del v0, v1, v2
    v3 = 2
    v4 = 5
    v5 = 1
    print(v5,(v3, v4))
    del v3, v4, v5
    v6 = 2.5
    v7 = 4.4
    print(v6,v7)
    v8 = 2
    print(v8,(v6, v7))
    del v6, v7, v8
    return 

if __name__ == '__main__': print(main())
