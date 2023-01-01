<!-- TOC -->

- [Introduction](#introduction)
- [Important Notes on UPMEM hardware](#important-notes-on-upmem-hardware)
- [How To Make A Language Backend Tutorial](#how-to-make-a-language-backend-tutorial)
    - [Prerequisites](#prerequisites)
    - [test8c](#test8c)
        - [Intro to inverse arrays](#intro-to-inverse-arrays)
        - [The main program](#the-main-program)
        - [Globals](#globals)
        - [Macros](#macros)
    - [test1](#test1)
    - [test3](#test3)
    - [test4a](#test4a)
    - [test4b](#test4b)
    - [test5](#test5)
    - [test6](#test6)

<!-- /TOC -->

Date Of Writing: Early January 2023

# Introduction

In mid-December 2022 I found the [PIM course](https://www.youtube.com/playlist?list=PL5Q2soXY2Zi8KzG2CQYRNQOVD0GOBrnKy) by the Safari group, posted it on the PL sub and a UPMEM compiler dev messaged me, pointing me at their [SDK](https://sdk.upmem.com/) which has a simulator for UPMEM devices. Since [Spiral](https://github.com/mrakgr/The-Spiral-Language) was designed to make using novel hardware easier, I lounged at the bait and decided to do a backend for the language. This is to show off my skills as well as demonstrate how useful Spiral could be in novel contexts. I want to sell on how useful the language could be in the particular niche of programming heterogeneous architectures it is targetting.

The goal of this document is to inform you of the possibilities that Spiral offers you as well as serve as a language tutorial. Also, the purpose of this demo is not to specifically focus on UPMEM devices, but to illustrate how a language backend could be done for any kind of device. What makes UPMEM special is that it is the first commercialized PIM chip, but it is just the first in line, leading the shift towards Process In Memory (PIM) programming.

# Important Notes on UPMEM hardware

The course itself has excellent coverage of device specifics, but to summarize:

* UPMEM sells devices that look like DRAM sticks you could stick into your home rig slot.
* Each of those has 8Gb of RAM spread across 128 DPUs (Data Processing Units).
* That means that each DPU has 64Mb of RAM.
* They have 64kb of WRAM memory each.
* They have 24kb of IRAM memory each.
* They are 32-bit systems with 32-bit pointers.
* They can't communicate with each other by sending messages for example. Instead they have to send their data back to the host CPU and communicate that way.
* They can only do integer arithmetic using on-board device logic, and have to rely on software emulation for floats. This results in 10x lower performance compared to ints. And unlike CPUs which are memory bound, these devices are compute bound.

[This repo](https://github.com/CMU-SAFARI/prim-benchmarks) has a paper with the benchmark results. Because of these last two points, the devices are no good for neural nets for example. Also internal to each DPU, they have software threads they call tasklets. According to the paper, on most tasks an 11 of them is needed to fully saturate the device. Unlike the GPUs threads, they aren't tied to each other and are independent like regular threads are on a CPU.

I only have access to the SDK simulator that goes up to simulating a single DPU, so I won’t be demonstrating how to program multiple DPUs or how to deal with software concurrency on them. Merely, I want to show how to do something very basic, that would nonetheless be very difficult to do in any other language.

# How To Make A Language Backend Tutorial

## Prerequisites

If you want to run these example you need:

* Spiral v2.3.6. You can get it as an VS Code extension.
* The UPMEM SDK. [Here](https://sdk.upmem.com/2021.4.0/01_Install.html) are the install instructions.
* (Opt) If you are on Windows like me, just install [WSL](https://learn.microsoft.com/en-us/windows/wsl/install). My distro is Ubuntu 20.04.1.

## test8c

### Intro to inverse arrays

This is the final result. It is a generic map kernel that takes in inverse arrays and maps them using a provided function. It is similar to how in F# you could map over an array like the following...

```fs
 [|1;2;3|] |> Array.map (fun x -> x*2)
```

To get the array with the elements multiplied by 2. Passing the above into F# interactive, I get:

```
val it: int array = [|2; 4; 6|]
```

You can use `Array.map` to map over arbitrary data structures such as tuples.

```fs
 [|(1,2);(3,4);(5,6)|] |> Array.map (fun (x,y) -> x+y, x-y, x*y)
```
```
val it: (int * int * int) array = [|(3, -1, 2); (7, -1, 12); (11, -1, 30)|]
```

A regular tuple array would have each of the elements be a tuple. It would be a single array with the elements laid out in memory as was written `[|(1,2);(3,4);(5,6)|]`.

An inverse tuple array would have each of the tuple fields in a separate arrays. It would have two regular arrays of `[|1;3;5|]` and `[|2;4;6|]`.

Since you can only transfer primitive ints and floats over as well as arrays of them over interop boundaries, they are especially useful for language interop. It is not actually possible to transfer an Python array of tuples to the DPU, or an F# array of heap allocated tuples to the GPU, so splitting them up into primitives is absolutely necessary in order to do this cleanly.

### The main program

Here is an example of a Spiral function that maps over them. I'll explain what it does step by step as I go along.

```sl
open inv
open upmem
open upmem_loop
open utils_real
open utils

// Maps the input array inplace.
inl map_inp f = run fun input output =>
    global "#include <mram.h>"
    global "__mram_noinit uint8_t buffer[1024*1024*64];"
    inl block_size = 8
    // Creates the WRAM buffers as inverse arrays.
    inl buf_in = create block_size
    inl buf_out = create block_size
    inl len = length input
    forBy {from=0; nearTo=len; by=block_size} fun from =>
        inl nearTo = min len (from + block_size)
        // Reads the MRAM into the WRAM buffer.
        mram_read input buf_in {from nearTo}
        for {from=0; nearTo=nearTo - from} fun i => 
            set buf_out i (f (index buf_in i))
        mram_write buf_out output {from nearTo}
    0

// Maps the input array.
inl map f input = inl output = create (length input) in map_inp f input output . output

inl main () =
    global "import os"
    global "from io import StringIO"
    global "from sys import stdout"
    global "import struct"

    inl test_size = 16
    inl input = 
        zip (arange test_size)
        <| zip (arange test_size)
        <| arange test_size
    $"print(!input)"
    inl output = map (fun a,b,c => a+b+c, a*b*c) input
    $"print(!output)"
    ()
```

In VS Code using the Spiral v2.3.6 compiler, I compile the above using Ctrl + F1 into `main.py`, start up the `powershell` terminal, go into `wsl`, source the UPMEM variables using `. ~/upmem-sdk/upmem_env.sh` and then run the following. I also don't forget to Ctrl + , into Settings and search for Spiral before picking the UPMEM backend.

```
mrakgr@Lain:/mnt/e/PIM-Programming-In-Spiral-UPMEM-Demo/test8c$ python3 main.py
(array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15],
      dtype=int32), array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15],
      dtype=int32), array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15],
      dtype=int32), 16)
(array([ 0,  3,  6,  9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45],
      dtype=int32), array([   0,    1,    8,   27,   64,  125,  216,  343,  512,  729, 1000,
       1331, 1728, 2197, 2744, 3375], dtype=int32), 16)
```

This gets printed (plus some manually erased warnings), proving that what I have works. If you play around with it, you'll see that you can have an arbitrary number of fields in the input and the output tuple. You can for example even use records. If you don't know what those are, or anything else I do not take the time to explain in this tutorial, take a look at the Spiral docs on the language repo.

```sl
    inl input = 
        zip (arange test_size)
        <| zip (arange test_size)
        <| arange test_size
    inl input = rezip (fun x,y,z => {x y z}) input
```

You can define functions in the map kernel.

```sl
    inl test_size = 16
    inl input = 
        zip (arange test_size)
        <| zip (arange test_size)
        <| arange test_size
    inl input = rezip (fun x,y,z => {x y z}) input
    $"print(!input)"
    inl output = 
        map (fun {x y z} => 
            inl sqr x = x*x
            sqr (x + y + z)
            ) input
    $"print(!output)"
```

Here is what I get when I compile and run the above.

```
mrakgr@Lain:/mnt/e/PIM-Programming-In-Spiral-UPMEM-Demo/test8c$ python3 main.py
(array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15],
      dtype=int32), array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15],
      dtype=int32), array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15],
      dtype=int32), 16)
(array([   0,    9,   36,   81,  144,  225,  324,  441,  576,  729,  900,
       1089, 1296, 1521, 1764, 2025], dtype=int32), 16)
```

Looks good to me.

What I will do here is explain the code in the `main.spi` file, starting with the simplest.

### Globals

```sl
    global "import os"
    global "from io import StringIO"
    global "from sys import stdout"
    global "import struct"
```

These are just the globals. If you look at the compiled `main.py` file, these statements are what trigger the Python imports to be printed in the resulting fire.

```py
import os
from io import StringIO
from sys import stdout
import struct
```

Similarly...

```sl
// Maps the input array inplace.
inl map_inp f = run fun input output =>
    global "#include <mram.h>"
    global "__mram_noinit uint8_t buffer[1024*1024*64];"
```

These globals get printed in the kernels string itself.

```c
#include <mram.h>
__mram_noinit uint8_t buffer[1024*1024*64];
```

They get printed like so. This is a part of the way that Spiral integrates with other languages. The UPMEM backend compiles itself to Python and C, and it is easy to reuse libraries from both those languages as a result. This is the ideal solution. In many cases, for example when compiling to C++, it is impossible to fetch the specific data needed to have first class integration.

### Macros

The other is macros.

```sl
    $"print(!input)"
    // ...
    $"print(!output)"
```

If you install the extension and take a look at it in the file, the hightlighting will make it more clear what is going on. The `!input`  and `!output` will be light blue.

```py
    print((v0, v1, v2, 16))
    # ...
    print((v3, 16))
```

In the Python code, these get printed as a tuple. The `input` variable is made out of 3 Numpy arrays and the length, so that is how it shows up in the code. The output on the other hand has only a single array plus the length, so that is how it shows up.

```sl
inl arange (len : u32) : inv_array a u32 i32 = 
    inl arrays : a u32 i32 = $"np.arange(0,!len,dtype=np.int32)"
    real inv_array `a `u32 `i32 {len arrays}
```

In the `arange` function you can see how macros are used to generate the Numpy int32 arrays before they are passed as singleton fields of the inv_array type.

In the generated code they look like...

```py
    v0 = np.arange(0,16,dtype=np.int32)
    v1 = np.arange(0,16,dtype=np.int32)
    v2 = np.arange(0,16,dtype=np.int32)
```

That is how integration works in Spiral. It works the same on the C side as well. That is how you can make use of foreign libraries.

## test1

Now, what I want to do next is explain how the backend integration works. If you go into `test8c/utils.spi` and look at the `run` function, you'll see something that is too much to take in all at once. Instead of trying to understand that, let us move to the first `test1` and take a look at this.

```sl
// Does the basic kernel compile?

inl main () = 
    inl a,b = join_backend UPMEM_C_Kernel
        inl x = $"int v$[1]" : $"int *"
        0i32

    a
```

These tests retrace how I've build up the program step by step by myself. Since the usual macros cannot declare static C arrays due to its awkward syntax, I've introduced the special `v$` form just for the C backend. What it does is substitute the generated binding directly into the macro itself, so in the generated code it gets printed as `v0` in this case rather than `v$`.

Here is the Python output.

```py
kernels = [
   r"""
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
int32_t main(){
    int v0[1];
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
    v0 = 0
    return v0

if __name__ == '__main__': print(main())
```

Those top level imports in Python and the includes in the C kernel string are dumped there automatically by the respective generators.

The backend join points are different from ordinary ones.

The way the ordinary join points work is that they compile into functions and function calls. For example...

```sl
inl main() =
    inl f (a,b) = join
        a+b : i32
    inl _ = f (1,2)
    inl _ = f (3,4)
    inl _ = f (dyn (5,6))
    0i32
```

```py
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
```

This is easy to deal with in the codegen as all it has to do is go over the statements and print them out. But interaction between different backends is more difficult.

```sl
// Does the basic kernel compile?

inl main () = 
    inl a,b = join_backend UPMEM_C_Kernel
        inl x = $"int v$[1]" : $"int *"
        0i32

    a
```

Going back to this example, you'll see that it generates the C code for this, but not the scaffolding to actually compile it. What are these return arguments `a` and `b`? If you hover over them in the editor you'll see that they say `i32` and `.tuple_of_free_vars`. The first argument is the index into the `kernels` array.

You can use it with a macro like `kernels[!a]` to get the actual code string. The second argument is not actually a symbol, the type you see in the editor is an outright lie, though it does hint what it is supposed to be.

```sl
print_static b
```
```
()
```

If I insert a `print_static` statement to get the contents of `b`, I get `()` back. The unit type is what it actually is. The body has no runtime variables in this example.

## test3

Here is a more interesting example

```sl
// Does the add kernel compile?

inl main () = 
    inl add ~a ~b = join_backend UPMEM_C_Kernel 
        ignore (a + b)
        0i32

    inl f (a,b) : () = $"print(!a,!b)"
    f (add 2i32 1)
    f (add 2u64 5)
    inl ~(x,y) = 2.5f32, 4.4
    $"print(!x,!y)"
    f (add x y)
```

```py
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
```

In the Python `main` function, the variables `v2`, `v5` and `v8` and the kernel indices while the rest are the backend join point call arguments. As you can see, the second argument returns the free (runtime) variables of the join points, that is those that aren't compile time literals.

Now if you look at the C kernels, the float add one for example...

```c
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
```

You'll see that there are these `__host` variables. UPMEM is annoying in that it uses globals to pass in arguments from host into the device. Unlike Cuda and OpenCL where you use the stack. This makes interop more challenging, and likely a lot less efficient to launch the kernel as it uses symbol matching to assign the variables, but still making use of it can be automated.

## test4a

This example demonstrates how the kernels can actually be run.

```sl
// Does the most basic kernel run succesfully?

inl main () = 
    global "import os"
    global "from io import StringIO"
    global "from sys import stdout"
    global "import struct"

    let f = join_backend UPMEM_C_Kernel 
        $'printf("Size of a pointer is %i.\\n", sizeof(int *))'
        $'printf("Size of a mram pointer is %i.\\n", sizeof(__mram int *))'
        0i32
    inl kernel_i, vars = f
    inl file_name = $"f'kernels/g{!kernel_i}'" : string
    $"if not os.path.exists('kernels'): os.mkdir('kernels')"
    inl file = $"open(f'{!file_name}.c','w')" : $"object"
    $"!file.write(kernels[!kernel_i])"
    $"if os.system(f'dpu-upmem-dpurte-clang -o {!file_name}.dpu {!file_name}.c') \!= 0: raise Exception('Compilation failed.')"

    inl dpu = $"DpuSet(nr_dpus=1, binary=f'{!file_name}.dpu')" : $"DpuSet"
    $"!dpu.exec(log=stdout)" // Note: Putting in log=stdout and not using a printf inside the kernel results in an error.
    ()
```

```py
kernels = [
   r"""
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
int32_t main(){
    printf("Size of a pointer is %i.\n", sizeof(int *));
    printf("Size of a mram pointer is %i.\n", sizeof(__mram int *));
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
def main():
    v0 = 0
    v1 = f'kernels/g{v0}'
    if not os.path.exists('kernels'): os.mkdir('kernels')
    v2 = open(f'{v1}.c','w')
    v2.write(kernels[v0])
    del v0, v2
    if os.system(f'dpu-upmem-dpurte-clang -o {v1}.dpu {v1}.c') != 0: raise Exception('Compilation failed.')
    v3 = DpuSet(nr_dpus=1, binary=f'{v1}.dpu')
    del v1
    v3.exec(log=stdout)
    del v3
    return 

if __name__ == '__main__': print(main())
```

Here is how to run it from the command line.

```
PS E:\PIM-Programming-In-Spiral-UPMEM-Demo> wsl
mrakgr@Lain:/mnt/e/PIM-Programming-In-Spiral-UPMEM-Demo$ . ~/upmem-sdk/upmem_env.sh 
Setting UPMEM_HOME to /home/mrakgr/upmem-sdk and updating PATH/LD_LIBRARY_PATH/PYTHONPATH
mrakgr@Lain:/mnt/e/PIM-Programming-In-Spiral-UPMEM-Demo$ cd test4a
mrakgr@Lain:/mnt/e/PIM-Programming-In-Spiral-UPMEM-Demo/test4a$ python3 main.py
```

Then I get a ton of warnings, but...

```
Size of a pointer is 4.
Size of a mram pointer is 4.
```

Does get printed. The example is more complicated than I'd like. If you look at the provided code on the [SDK page](https://sdk.upmem.com/2021.4.0/PythonAPI/modules.html#overview) you might think that I could be able to just pass the kernel into the `c_string` argument, but what I've found when I tried this is that the error messages get eaten. Instead of the excellent Clang error messages, I would instead get a noncommital something-went-wrong style of error. So instead I take the kernel code as a string, write it into a file, and then have `dpu-upmem-dpurte-clang` compile it. That way if something goes wrong in the C kernel, I can hear the Clang telling me what it is.

It is easy to get macros wrong, and while I was working on this I was adjusting the codegen worked. Having to guess what the problem is would have made work impossible.

## test4b

```sl
// Does the most basic kernel run succesfully?

inl main () = 
    global "import os"
    global "from io import StringIO"
    global "from sys import stdout"
    global "import struct"

    let f = join_backend UPMEM_C_Kernel 
        global "#include <mram.h>"
        global "__mram_noinit uint8_t t[1024*1024*64];"
        // global "__mram_noinit uint8_t t1[1024*1024*64];"
        inl q = $"__dma_aligned uint8_t v$[512]" : $"uint8_t *"
        $"mram_read(t, !q, sizeof(!q))"
        0i32
    inl kernel_i, vars = f
    inl file_name = $"f'kernels/g{!kernel_i}'" : string
    $"if not os.path.exists('kernels'): os.mkdir('kernels')"
    inl file = $"open(f'{!file_name}.c','w')" : $"object"
    $"!file.write(kernels[!kernel_i])"
    $"if os.system(f'dpu-upmem-dpurte-clang -o {!file_name}.dpu {!file_name}.c') \!= 0: raise Exception('Compilation failed.')"

    inl dpu = $"DpuSet(nr_dpus=1, binary=f'{!file_name}.dpu')" : $"DpuSet"
    $"!dpu.exec()" // Note: Putting in log=stdout and not using a printf inside the kernel results in an error.
    ()
```

These examples are a lot nicer to look at in the editor, so I'd recommend it. Anyway, here the kernel still does not do anything interesting, but in it I've confirmed how to make local arrays, read from MRAM into the WRAM buffer, as well as verify the total memory size of the DPU. While I was working on the backend, I didn't actually know until over 10 days in that DPUs had only 64mb of memory each, so I was very beffudled as to why the compiler would not let me allocate more than 64mb. If you uncomment that `// global "__mram_noinit uint8_t t1[1024*1024*64];"` line the compiler would actually give an error statically. I thought that DPUs would be a bit like Cuda cores and have access to the entirety of their memory. In fact, I thought that a single DPU was responsible for the whole 8gb, and I found very confusing why the system is 32-bit when it had so much memory. This test made it clear to me that these things in fact have 64mb. This isn't actually written anywhere in the user guide.

It is in fact stated in the PIM course, but who is going to remember that the first time around?

So thus far, we know we can generate C kernels for the UPMEM device and we can run them. In this next example, I demonstrate how to transfer variables across language boundaries.

## test5

```sl
// Does the variable transfer work?

open utils

let compile_kernel (kernel_i : i32) =
    $"if not os.path.exists('kernels'): os.mkdir('kernels')"
    inl file = $"open(f'kernels/g{!kernel_i}.c','w')" : $"object"
    $"!file.write(kernels[!kernel_i])"
    $"if os.system(f'dpu-upmem-dpurte-clang -o kernels/g{!kernel_i}.dpu kernels/g{!kernel_i}.c') \!= 0: raise Exception('Compilation failed.')"

inl main () =
    global "import os"
    global "from io import StringIO"
    global "from sys import stdout"
    global "import struct"

    let f (a,b,c,d) = join_backend UPMEM_C_Kernel 
        inl q : i32 = a + b - c - d
        inl w = a * b * c * d
        $"!a = !q"
        $"!b = !w"
        0i32
    inl kernel_i, vars = f (1,2,3,4)
    compile_kernel kernel_i
    inl dpu = $"DpuSet(nr_dpus=1, binary=f'kernels/g{!kernel_i}.dpu', profile='backend=simulator')" : $"DpuSet"
    real dpu_pack dpu vars
    $"print(!vars)"
    $"!dpu.exec()" // Note: Putting in log=stdout and not using a printf inside the kernel results in an error.
    inl x = real dpu_unpack dpu vars
    $"print(!x)"
    ()
```

This is where the examples start to get a bit involved.

```sl
    let f (a,b,c,d) = join_backend UPMEM_C_Kernel 
        inl q : i32 = a + b - c - d
        inl w = a * b * c * d
        $"!a = !q"
        $"!b = !w"
        0i32
    inl kernel_i, vars = f (1,2,3,4)
    compile_kernel kernel_i
    inl dpu = $"DpuSet(nr_dpus=1, binary=f'kernels/g{!kernel_i}.dpu', profile='backend=simulator')" : $"DpuSet"
```

By now you should understand what this part does. Here I define the backend function, then I run it to get back the index and the tuple of runtime variables that need to be passed into the kernel. Then I take the kernel index and compile it as well as allocate the DPU. Once the `exec()` function is run the DPU program will execute. But before that...

```sl
real dpu_pack dpu vars
```

This is the part that actually transfers the variables to the DPU. If this was GPU programming, I'd just plop the args right there into the kernel launch function, but UPMEM uses globals to pass arguments between the host and device.

So in `utils.spir` I define the following function.

```sl
open real_core

inl dpu_pack dpu vars =
    inl rec loop i = function
        | () => ()
        | (a,b) => 
            typecase `a with
            | i8 => $"!dpu.v!i = bytearray(struct.pack('b',!a))" : ()
            | i16 => $"!dpu.v!i = bytearray(struct.pack('h',!a))" : ()
            | i32 => $"!dpu.v!i = bytearray(struct.pack('i',!a))" : ()
            | i64 => $"!dpu.v!i = bytearray(struct.pack('q',!a))" : ()
            | u8 => $"!dpu.v!i = bytearray(struct.pack('B',!a))" : ()
            | u16 => $"!dpu.v!i = bytearray(struct.pack('H',!a))" : ()
            | u32 => $"!dpu.v!i = bytearray(struct.pack('I',!a))" : ()
            | u64 => $"!dpu.v!i = bytearray(struct.pack('Q',!a))" : ()
            | f32 => $"!dpu.v!i = bytearray(struct.pack('f',!a))" : ()
            | f64 => $"!dpu.v!i = bytearray(struct.pack('d',!a))" : ()
            loop (i+1) b
    loop 0 vars
```

This is written in the bottom-up segment of Spiral. In languages like F#, Haskell and Ocaml, this kind of code would result in a type error. That is, the first `()` case would get inferred as an unit type, and the `(a,b)` case would show an error. But in the `.spir` files as well as the `real` segments in `.spi` files, this is perfectly valid.

If you are used to functional programming, the above fragment is not at all different from functionaly iterating over a list, but is guaranteed to be done at compile time because pairs and tuples are purely a compile time construct in Spiral.

It results in code like this.

```py
    v5.v0 = bytearray(struct.pack('i',v3))
    v5.v1 = bytearray(struct.pack('i',v2))
    v5.v2 = bytearray(struct.pack('i',v1))
    v5.v3 = bytearray(struct.pack('i',v0))
```

This is what I meant by UPMEM using globals to transfer variables. It actually uses symbol names to resolve them. 

```py
    v5.copy('v0', bytearray(struct.pack('i',v3)))
```

You could also write code like the above to achieve the same thing.

```sl
    $"print(!vars)"
    $"!dpu.exec()" // Note: Putting in log=stdout and not using a printf inside the kernel results in an error.
    inl x = real dpu_unpack dpu vars
    $"print(!x)"
```

In the next section I print the call arguments, execute the kernel, unpack the free vars and print them out again. Unpacking is done similar to packing.

```sl
inl dpu_unpack dpu vars = 
    inl rec loop i = function
        | () => ()
        | (a,b) => 
            inl r =
                typecase `a with
                | i8 => $"!dpu.v!i.int8()" : `a
                | i16 => $"!dpu.v!i.int16()" : `a
                | i32 => $"!dpu.v!i.int32()" : `a
                | i64 => $"!dpu.v!i.int64()" : `a
                | u8 => $"!dpu.v!i.uint8()" : `a
                | u16 => $"!dpu.v!i.uint16()" : `a
                | u32 => $"!dpu.v!i.uint32()" : `a
                | u64 => $"!dpu.v!i.uint64()" : `a
                | f32 => $"!dpu.v!i.float32()" : `a
                | f64 => $"!dpu.v!i.float64()" : `a

            r, loop (i+1) b
    loop 0 vars
```

I just iterate over the tuple, fetching it from the DPU's WRAM. When I run it I get the following output.

```
(4, 3, 2, 1)
(4, 3, 24, -4)
```

This isn't what you'd expect. I mean, when I wrote it the first time it certainly wasn't what I expected and I made the language! For some reason the arguments were getting reversed. I spent a day going over the compiler code, refreshing my memory of it, and the reason why this is going on is due to the way pattern matching on pairs is compiled. This wasn't an actual program error, but merely an error in my understanding of how things should function.

Spiral itself makes no guarantees in the order the arguments will get passed into the join point, and so we get results like these.

Anyway, compiling the program results in...

```py
kernels = [
   r"""
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
__host int32_t v0;
__host int32_t v1;
__host int32_t v2;
__host int32_t v3;
int32_t main(){
    int32_t v4;
    v4 = v3 + v2;
    int32_t v5;
    v5 = v4 - v1;
    int32_t v6;
    v6 = v5 - v0;
    int32_t v7;
    v7 = v3 * v2;
    int32_t v8;
    v8 = v7 * v1;
    int32_t v9;
    v9 = v8 * v0;
    v3 = v6;
    v2 = v9;
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
def method0(v0 : i32) -> None:
    if not os.path.exists('kernels'): os.mkdir('kernels')
    v1 = open(f'kernels/g{v0}.c','w')
    v1.write(kernels[v0])
    del v1
    if os.system(f'dpu-upmem-dpurte-clang -o kernels/g{v0}.dpu kernels/g{v0}.c') != 0: raise Exception('Compilation failed.')
    del v0
    return 
def main():
    v0 = 1
    v1 = 2
    v2 = 3
    v3 = 4
    v4 = 0
    method0(v4)
    v5 = DpuSet(nr_dpus=1, binary=f'kernels/g{v4}.dpu', profile='backend=simulator')
    del v4
    v5.v0 = bytearray(struct.pack('i',v3))
    v5.v1 = bytearray(struct.pack('i',v2))
    v5.v2 = bytearray(struct.pack('i',v1))
    v5.v3 = bytearray(struct.pack('i',v0))
    print((v3, v2, v1, v0))
    del v0, v1, v2, v3
    v5.exec()
    v6 = v5.v0.int32()
    v7 = v5.v1.int32()
    v8 = v5.v2.int32()
    v9 = v5.v3.int32()
    del v5
    print((v6, v7, v8, v9))
    del v6, v7, v8, v9
    return 

if __name__ == '__main__': print(main())
```

You'll note that `1` and `2` are in fact getting assigned to indicating that the kernel itself works. I realized after creating this test that in order to read from the variables in the order I want I needed to keep track of their tags.

## test6

...