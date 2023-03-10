kernels = [
   r"""
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <mram.h>
__mram_noinit uint8_t buffer[1024*1024*64];
bool method_while0(uint32_t v0){
    bool v1;
    v1 = v0 < 16ul;
    return v1;
}
bool method_while1(uint32_t v0, uint32_t v1){
    bool v2;
    v2 = v1 < v0;
    return v2;
}
int32_t main(){
    __dma_aligned int32_t v0[8ul];
    __dma_aligned int32_t v1[8ul];
    __dma_aligned int32_t v2[8ul];
    __dma_aligned int32_t v3[8ul];
    __dma_aligned int32_t v4[8ul];
    uint32_t v5 = 0ul;
    while (method_while0(v5)){
        uint32_t v7;
        v7 = v5 + 8ul;
        bool v8;
        v8 = 16ul < v7;
        uint32_t v9;
        if (v8){
            v9 = 16ul;
        } else {
            v9 = v7;
        }
        uint32_t v10;
        v10 = v9 - v5;
        __mram_ptr int32_t * v11;
        v11 = (__mram_ptr int32_t *) (buffer + 0ul);
        mram_read(v11 + v5,v0,v10 * sizeof(int32_t));
        __mram_ptr int32_t * v12;
        v12 = (__mram_ptr int32_t *) (buffer + 64ul);
        mram_read(v12 + v5,v1,v10 * sizeof(int32_t));
        __mram_ptr int32_t * v13;
        v13 = (__mram_ptr int32_t *) (buffer + 128ul);
        mram_read(v13 + v5,v2,v10 * sizeof(int32_t));
        uint32_t v14 = 0ul;
        while (method_while1(v10, v14)){
            int32_t v16;
            v16 = v0[v14];
            int32_t v17;
            v17 = v1[v14];
            int32_t v18;
            v18 = v2[v14];
            int32_t v19;
            v19 = v16 + v17;
            int32_t v20;
            v20 = v19 + v18;
            int32_t v21;
            v21 = v16 * v17;
            int32_t v22;
            v22 = v21 * v18;
            v3[v14] = v20;
            v4[v14] = v22;
            v14 += 1ul;
        }
        __mram_ptr int32_t * v23;
        v23 = (__mram_ptr int32_t *) (buffer + 192ul);
        mram_write(v3,v23 + v5,v10 * sizeof(int32_t));
        __mram_ptr int32_t * v24;
        v24 = (__mram_ptr int32_t *) (buffer + 256ul);
        mram_write(v4,v24 + v5,v10 * sizeof(int32_t));
        v5 += 8ul;
    }
    return 0l;
}
""",
]
from dpu import DpuSet
import numpy as np
from dataclasses import dataclass
from typing import NamedTuple, Union, Callable, Tuple
i8 = i16 = i32 = i64 = u8 = u16 = u32 = u64 = int; f32 = f64 = float; char = string = str

import os
from io import StringIO
from sys import stdout
import struct
def method1(v0 : i32) -> None:
    if not os.path.exists('kernels'): os.mkdir('kernels')
    v1 = open(f'kernels/g{v0}.c','w')
    v1.write(kernels[v0])
    del v1
    if os.system(f'dpu-upmem-dpurte-clang -o kernels/g{v0}.dpu kernels/g{v0}.c') != 0: raise Exception('Compilation failed.')
    del v0
    return 
def method2(v0 : DpuSet, v1 : np.ndarray, v2 : u32) -> None:
    v3 = v1.nbytes % 8 == 0
    v4 = v3 == False
    del v3
    if v4:
        raise Exception("The input array has to be divisible by 8")
    else:
        pass
    del v4
    v0.copy('buffer',bytearray(v1),offset=v2)
    del v0, v1, v2
    return 
def method3(v0 : DpuSet, v1 : np.ndarray, v2 : u32) -> None:
    v3 = v1.nbytes % 8 == 0
    v4 = v3 == False
    del v3
    if v4:
        raise Exception("The input array has to be divisible by 8")
    else:
        pass
    del v4
    v5 = bytearray(v1.nbytes)
    v0.copy(v5,'buffer',offset=v2)
    del v0, v2
    np.copyto(v1,np.frombuffer(v5,dtype=v1.dtype))
    del v1, v5
    return 
def method0(v0 : np.ndarray, v1 : np.ndarray, v2 : np.ndarray, v3 : np.ndarray, v4 : np.ndarray) -> None:
    v5 = 0
    method1(v5)
    v6 = DpuSet(nr_dpus=1, binary=f'kernels/g{v5}.dpu', profile='backend=simulator')
    del v5
    v7 = 0
    method2(v6, v0, v7)
    del v0, v7
    v8 = 64
    method2(v6, v1, v8)
    del v1, v8
    v9 = 128
    method2(v6, v2, v9)
    del v2, v9
    v6.exec()
    v10 = 192
    method3(v6, v3, v10)
    del v3, v10
    v11 = 256
    return method3(v6, v4, v11)
def main():
    v0 = np.arange(0,16,dtype=np.int32)
    v1 = np.arange(0,16,dtype=np.int32)
    v2 = np.arange(0,16,dtype=np.int32)
    print((v0, v1, v2, 16))
    v3 = np.empty(16,dtype=np.int32)
    v4 = np.empty(16,dtype=np.int32)
    method0(v0, v1, v2, v3, v4)
    del v0, v1, v2
    print((v3, v4, 16))
    del v3, v4
    return 

if __name__ == '__main__': print(main())
