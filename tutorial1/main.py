kernels = [
]
from dpu import DpuSet
import numpy as np
from dataclasses import dataclass
from typing import NamedTuple, Union, Callable, Tuple
i8 = i16 = i32 = i64 = u8 = u16 = u32 = u64 = int; f32 = f64 = float; char = string = str

def method0() -> i32:
    return 3
def method1() -> i32:
    return 7
def method2(v0 : i32, v1 : i32) -> i32:
    v2 = v1 + v0
    del v0, v1
    return v2
def main():
    v0 = method0()
    del v0
    v1 = method1()
    del v1
    v2 = 5
    v3 = 6
    v4 = method2(v3, v2)
    del v2, v3, v4
    return 0

if __name__ == '__main__': print(main())
