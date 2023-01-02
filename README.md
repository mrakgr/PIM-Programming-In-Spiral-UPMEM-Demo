<!-- TOC -->

- [Introduction](#introduction)
- [Important Notes on UPMEM hardware](#important-notes-on-upmem-hardware)
- [How To Make A Language Backend Tutorial](#how-to-make-a-language-backend-tutorial)
    - [Prerequisites](#prerequisites)
    - [test8 (Part 1)](#test8-part-1)
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
    - [test7](#test7)
        - [main1 (Part 1)](#main1-part-1)
            - [`function_term_slots_get` and `function_term_slots_get`](#function_term_slots_get-and-function_term_slots_get)
        - [main1 (Part 2)](#main1-part-2)
            - [Type Lying](#type-lying)
        - [main2](#main2)
    - [test8 (Part 2)](#test8-part-2)
- [Why This Is So Great](#why-this-is-so-great)
    - [What Is Not Great About Spiral](#what-is-not-great-about-spiral)
    - [What Is Truly Great About Spiral](#what-is-truly-great-about-spiral)

<!-- /TOC -->

Date Of Writing: Early January 2023

# Introduction

In mid-December 2022 I found the [PIM course](https://www.youtube.com/playlist?list=PL5Q2soXY2Zi8KzG2CQYRNQOVD0GOBrnKy) by the Safari group, posted it on the PL sub and a UPMEM compiler dev messaged me, pointing me at their [SDK](https://sdk.upmem.com/) which has a simulator for UPMEM devices. Since [Spiral](https://github.com/mrakgr/The-Spiral-Language) was designed to make using novel hardware easier, I lounged at the bait and decided to do a backend for the language. This is to show off my skills as well as demonstrate how useful Spiral could be in novel contexts. I want to sell on how useful the language could be in the particular niche of programming heterogeneous architectures it is targetting.

The goal of this document is to inform you of the possibilities that Spiral offers you as well as serve as a language tutorial. Also, the purpose of this demo is not to specifically focus on UPMEM devices, but to illustrate how a language backend could be done for any kind of device. What makes the UPMEM DPU special is that it is the first commercialized PIM chip, leading the shift towards Process In Memory (PIM) programming, but others will come along and Spiral has been designed with that eventuality in mind.

# Important Notes on UPMEM hardware

The course itself has excellent coverage of device specifics, but to summarize:

* UPMEM sells devices that look like DRAM sticks you could stick into your home rig slot.
* Each of those has 8Gb of RAM spread across 128 DPUs (Data Processing Units).
* That means that each DPU has 64Mb of (main) MRAM.
* They have 64kb of (working) WRAM memory each.
* They have 24kb of (instruction) IRAM memory each.
* They are 32-bit systems with 32-bit pointers.
* They can't communicate with each other by sending messages for example. Instead they have to send their data back to the host CPU and communicate that way.
* They can only do integer arithmetic using on-board device logic, and have to rely on software emulation for floats. This results in 10x lower performance compared to ints. And unlike CPUs which are memory bound, these devices are compute bound.

[This repo](https://github.com/CMU-SAFARI/prim-benchmarks) has a paper with the benchmark results. Because of these last two points, the devices are no good for neural nets for example. Also internal to each DPU, they have software threads they call tasklets. According to the paper, on most tasks an 11 of them is needed to fully saturate the device. Unlike the GPUs threads, they aren't tied to each other and are independent like regular threads are on a CPU.

I only have access to the SDK simulator that goes up to simulating a single DPU, so I won’t be demonstrating how to program multiple DPUs or how to deal with software concurrency on them. Merely, I want to show how to do something very basic, that would nonetheless be very difficult to do in any other language.

# How To Make A Language Backend Tutorial

This tutorial is technical and if you just want the rantz instead of me guiding you through 1,900 lines of code, skip to the last section where I discuss informally why Spiral is great and various other things on my mind.

## Prerequisites

If you want to run these example you need:

* Spiral v2.3.6. You can get it as an VS Code extension.
* The UPMEM SDK. [Here](https://sdk.upmem.com/2021.4.0/01_Install.html) are the install instructions.
* (Opt) If you are on Windows like me, just install [WSL](https://learn.microsoft.com/en-us/windows/wsl/install). My distro is Ubuntu 20.04.1.

## test8 (Part 1)

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
mrakgr@Lain:/mnt/e/PIM-Programming-In-Spiral-UPMEM-Demo/test8$ python3 main.py
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
mrakgr@Lain:/mnt/e/PIM-Programming-In-Spiral-UPMEM-Demo/test8$ python3 main.py
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

Now, what I want to do next is explain how the backend integration works. If you go into `test8/utils.spi` and look at the `run` function, you'll see something that is too much to take in all at once. Instead of trying to understand that, let us move to the first `test1` and take a look at this.

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

This is what I meant by UPMEM using globals to transfer variables. It actually uses symbol names to resolve them. When you assign them like this, it actually copies the integers to WRAM.

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

You'll note that `1` and `2` are in fact getting assigned to indicating that the kernel itself works. I realized after creating this test that in order to read from the variables in the order I want I needed to keep track of their tags. If you'll look at the main function, it should be very straightforward. It compiles the kernel in `method0`, allocates the `DpuSet`, copies the variables to WRAM `v5.v0 = bytearray(struct.pack('i',v3))`, lauches the kernel `v5.exec()`, then transfers them back to host `v6 = v5.v0.int32()`.

To summarize:

* We have a way to compile the kernels and transfer the variables back and forth. This is a lot.

What we are missing:

* We are still missing a way to get specific outputs out.
* We need a way to copy arrays back and forth to MRAM.

The following tests tackle this.

## test6

We'll start with the first one.

```sl
// Does the scalar map work?

open utils

let compile_kernel (kernel_i : i32) =
    $"if not os.path.exists('kernels'): os.mkdir('kernels')"
    inl file = $"open(f'kernels/g{!kernel_i}.c','w')" : $"object"
    $"!file.write(kernels[!kernel_i])"
    $"if os.system(f'dpu-upmem-dpurte-clang -o kernels/g{!kernel_i}.dpu kernels/g{!kernel_i}.c') \!= 0: raise Exception('Compilation failed.')"

inl run forall a b. (f : a -> b -> i32) (input : a) (output : b) : b = join
    real if has_free_vars f then real_core.error_type "The body of the kernel function in this version of `run` should not have any free variables. Pass them in through input and output instead."
    inl kernel_i, free_vars = join_backend UPMEM_C_Kernel f input output
    inl free_vars_s = real dpu_tags free_vars
    compile_kernel kernel_i
    inl dpu = $"DpuSet(nr_dpus=1, binary=f'kernels/g{!kernel_i}.dpu', profile='backend=simulator')" : $"DpuSet"
    real dpu_pack dpu free_vars_s (real_core.free_vars input)
    $"!dpu.exec()" // Note: Putting in log=stdout and not using a printf inside the kernel results in an error.
    real dpu_unpack dpu {vars=free_vars_s} output
    output
```

A lot of this should be familiar to you now. The `compile_kernel` should be old news.

```sl
real if has_free_vars f then real_core.error_type "The body of the kernel function in this version of `run` should not have any free variables. Pass them in through input and output instead."
```

This part simply makes sure that the `f`, the input function does not have any runtime variables. There is an op in the language `FreeVars` which you can use to get all the free vars in an expression. The `free_vars` function in `real_core` can be used to do the same thing. This doesn't mean the function's variable slots are empty. They might simply be compile time literals and similar constructs.

```sl
inl kernel_i, free_vars = join_backend UPMEM_C_Kernel f input output
```

Again, this should be old hat. The backend join point returns the kernel index as well as the runtime free vars of its body.

```sl
inl free_vars_s = real dpu_tags free_vars
```

This one is new. Here is how `dpu_tags` is implemented.

```sl
// Iterates over the tuple of free vars returning var tags mapped to a linear sequence of integer indices.
// This is intented to be in the order the call args would get generated in the kernel.
inl dpu_tags vars =
    inl rec loop s i = function
        | () => s
        | (a,b) =>
            inl k = !!!!TagToSymbol(!!!!VarTag(a))
            loop {s with $k=i} (i + 1) b 
    loop {} 0 vars
```

If you'd have a variable `v3` for example in the generated code, the `VarTag` op returns 3 as an 32-bit int compile time literal. This is then passed to `TagToSymbol` which turns it into a symbol. Symbols in Spiral are used to index into records `some_record.some_field`. The `.some_field` would be the symbol. This allows me at compile time to build an associative map of call argument tags to kernel __host variables.

```sl
real dpu_pack dpu free_vars_s (real_core.free_vars input)
```

Then what I do in this line is actually transfer the runtime variables to the WRAM memory.

```sl
// Copies the host runtime vars to WRAM.
inl dpu_pack dpu s vars =
    inl rec loop = function
        | () => ()
        | (a,b) =>
            inl i = s !!!!TagToSymbol(!!!!VarTag(a))
            typecase `a with
            | i8 =>  $"!dpu.v!i = bytearray(struct.pack('b',!a))" : ()
            | i16 => $"!dpu.v!i = bytearray(struct.pack('h',!a))" : ()
            | i32 => $"!dpu.v!i = bytearray(struct.pack('i',!a))" : ()
            | i64 => $"!dpu.v!i = bytearray(struct.pack('q',!a))" : ()
            | u8 =>  $"!dpu.v!i = bytearray(struct.pack('B',!a))" : ()
            | u16 => $"!dpu.v!i = bytearray(struct.pack('H',!a))" : ()
            | u32 => $"!dpu.v!i = bytearray(struct.pack('I',!a))" : ()
            | u64 => $"!dpu.v!i = bytearray(struct.pack('Q',!a))" : ()
            | f32 => $"!dpu.v!i = bytearray(struct.pack('f',!a))" : ()
            | f64 => $"!dpu.v!i = bytearray(struct.pack('d',!a))" : ()
            loop b 
    loop vars
```

It is very similar to what I had before, but now I make use of the record created in `dpu_tags` to get the kernel variable tags. After `dpu_unpack` statements are executed in the Python code, the variables will be in DPU's WRAM.

```sl
$"!dpu.exec()" // Note: Putting in log=stdout and not using a printf inside the kernel results in an error.
```

This runs the kernel. After it finishes it is time to get the results out.

```sl
    real dpu_unpack dpu {vars=free_vars_s} output
    output
```

Note here how unlike when packing the `input`s where I simply got the `free_vars`, here I instead pass in the `output` directly. This preserves its structure. But it makes iterating over it harder.

```sl
// For transfering the variables back to host.
inl dpu_unpack dpu s vars = 
    inl rec loop = function
        | () => ()
        | {} as x => record_iter (fun {key value} => loop value) x
        | (a,b) => loop a . loop b
        | a => 
            inl i = s.vars !!!!TagToSymbol(!!!!VarTag(a))
            typecase `a with
            | i8 =>  $"!a = !dpu.v!i.int8()" : ()
            | i16 => $"!a = !dpu.v!i.int16()" : ()
            | i32 => $"!a = !dpu.v!i.int32()" : ()
            | i64 => $"!a = !dpu.v!i.int64()" : ()
            | u8 =>  $"!a = !dpu.v!i.uint8()" : ()
            | u16 => $"!a = !dpu.v!i.uint16()" : ()
            | u32 => $"!a = !dpu.v!i.uint32()" : ()
            | u64 => $"!a = !dpu.v!i.uint64()" : ()
            | f32 => $"!a = !dpu.v!i.float32()" : ()
            | f64 => $"!a = !dpu.v!i.float64()" : ()
    loop vars
```

Not that much harder though in this example. All I have to do is also consider the records. The the generated code this results in code such as...

```py
v2 = v4.v2.int32()
```

```py
v3 = v5.v3.int32()
```

And so on.

Since we can pass vars in and out of kernels, we can do a scalar map.

```
inl assign forall t. (a : t) (b : t) : () = real assign a b
inl scalar_map_inp f = run (fun inp out => assign out (f inp) . 0i32)
inl scalar_map forall a b. (f : a -> b) ~(inp : a) = scalar_map_inp f inp (real default `b)
```

What `assign` does is generate code such as `v0 = 3` on the C side. Regular C assignments cannot deal with the full range of Spiral types, so a little helper is required. Here is how it is implemented.

```sl
inl rec assign a b = real
    assert (`(`(a)) `= `(`(b))) "The two types should be equal."
    match a,b with
    | (), () => ()
    | {}, {} => record_iter (fun {key value} => assign value (b key)) a
    | (a,b), (a',b') => assign a a' . assign b b'
    | a,b when prim_is a && prim_is b => $"!a = !b" : ()
```

It iterates over the tuples and records, and generates the primitive assignments. 

```sl
inl scalar_map forall a b. (f : a -> b) ~(inp : a) = scalar_map_inp f inp (real default `b)
```

The `default` function used in `scalar_map` here just makes some default values based on purely the type.

```sl
inl default forall t. : t = 
    inl rec loop forall t. = 
        typecase t with
        | ~a * ~b => loop `a, loop `b
        | {} => record_type_map (fun k => forall v. => loop `v) `t
        | i8  => $"0" : t
        | i16 => $"0" : t
        | i32 => $"0" : t
        | i64 => $"0" : t
        | u8  => $"0" : t
        | u16 => $"0" : t
        | u32 => $"0" : t
        | u64 => $"0" : t
        | f32 => $"0.0" : t
        | f64 => $"0.0" : t
    loop `t
```

Just like ordinary records, Spiral has function in `real_core` for iterating over record types. So a type like...let me show it in code.

```sl
inl main () =
    inl x = real default `(i32 * {x : f64; y : f64})
    $"print(!x)"
```

The above results in the following Python code.

```py
def main():
    v0 = 0
    v1 = 0.0
    v2 = 0.0
    print((v0, v1, v2))
```

The type of `x` is `(i32 * {x : f64; y : f64})`.

Finally, here is the `main` function.

```sl
inl main () =
    global "import os"
    global "from io import StringIO"
    global "from sys import stdout"
    global "import struct"

    inl x : i32 = scalar_map (fun (a,b) => a+b) (1,2)
    $"print(!x)"

    inl x : i32 = scalar_map (fun (a,b,c) => a+b-c) (1,2,3)
    $"print(!x)"

    inl x : i32 = scalar_map (fun (a,b,c,d) => (a+b)*d/c) (1,2,3,4)
    $"print(!x)"

    inl ~q = 1,2,3 // Make sure the arguments are dyn'd before passing them into the function otherwise it won't work.
    inl x : i32 = scalar_map_inp (fun (a,b,c) => a-b-c) q (fst q)
    $"print(!x)"

    ()
```

When I compile and run this, I get...

```
3
0
4
-4
```

Plus some warnings I've ommitted by hand.

That covers `test6` and scalar maps. Now we know how to deal with passing variables and back and forth from the DPU. At this point `main.py` is lengthy so I won't paste it here, but you can check it out in the `test6` folder. While we have something useful now, any realistic use of the DPU would involve arrays. That is what we need to deal with next.

## test7

### main1 (Part 1)

Here is an example of a kernel that adds two vectors together.

```sl
open upmem
open upmem_loop
open utils_real
open utils

inl add2 forall dim {number} t {number}. = run fun (b,c) (a : a dim t) =>
    global "__mram_noinit uint8_t buffer[1024*1024*1];"
    for {from=0; nearTo=length a} fun i =>
        set a i (index b i + index c i)
    0

inl main () =
    global "import os"
    global "from io import StringIO"
    global "from sys import stdout"
    global "import struct"

    inl test_size = 64
    inl a,b,c = arange test_size, arange test_size, arange test_size
    $"print(!a)"
    add2 (b,c) a
    $"print(!a)"

    ()
```

I pass in 2 input vectors, and 1 output vector and add the two into the output. When I run this I get...

```
[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47
 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63]
[  0   2   4   6   8  10  12  14  16  18  20  22  24  26  28  30  32  34
  36  38  40  42  44  46  48  50  52  54  56  58  60  62  64  66  68  70
  72  74  76  78  80  82  84  86  88  90  92  94  96  98 100 102 104 106
 108 110 112 114 116 118 120 122 124 126]
```

So it works. Let me cover that changes needed to make it work. First comes the run function.

```sl
inl run forall a b. (f : a -> b -> i32) (input : a) (output : b) : () = join
    real if has_free_vars f then real_core.error_type "The body of the kernel function in this version of `run` should not have any free variables. Pass them in through input and output instead."
    inl input, in_s : a * _ = real dpu_arrays 0u32 input
    inl output', out_s : b * _ = real dpu_arrays in_s.offset output
    inl kernel_i, free_vars = join_backend UPMEM_C_Kernel f input output'
    inl free_vars_s = real dpu_tags free_vars
    compile_kernel kernel_i
    inl dpu = $"DpuSet(nr_dpus=1, binary=f'kernels/g{!kernel_i}.dpu', profile='backend=simulator')" : $"DpuSet"
    real dpu_pack dpu free_vars_s free_vars
    real dpu_arrays_transfer dpu in_s.arrays
    $"!dpu.exec()" // Note: Putting in log=stdout and not using a printf inside the kernel results in an error.
    real dpu_unpack dpu {arrays=out_s.arrays; vars=free_vars_s} output
```

There are some new lines specifically for handling arrays.

```sl
    inl input, in_s : a * _ = real dpu_arrays 0u32 input
    inl output', out_s : b * _ = real dpu_arrays in_s.offset output
```

What I do here is transform numpy arrays into records of their length and offset in the kernel `buffer` array.

```sl
real dpu_arrays_transfer dpu in_s.arrays
```

This is the function I call to actually transfer the buffers to the DPU.

```sl
real dpu_pack dpu free_vars_s free_vars
```

Also this time around I transfer both the input and the output runtime variables. I realized while working on this example that if I only transfered the inputs that the array lengths and offsets wouldn't be moved to the DPU for the outputs!

```sl
inl dpu_arrays offset f =
    inl rec loop s = function
        | () => (), s
        | (a,b) =>
            inl a,s = loop s a
            inl b,s = loop s b
            (a,b), s
        | {} & a =>
            record_fold (fun {state=(m,s) key value} => 
                inl a,s = loop s value
                {m with $key=a}, s
                ) ({}, s) a
        | a when function_is a => case_fun s a
        | a =>
            typecase `a with
            | a ~dim ~t =>
                inl k, offset = !!!!TagToSymbol(!!!!VarTag(a)), s.offset
                inl len = match s with {len} => len | _ => $"len(!a)" : dim
                inl nbytes = len * conv `dim (sizeof `t)
                upmem_kernel_array `dim `t {offset len}, {s with offset#=fun o => roundup (o + nbytes) 8u32; arrays#=fun ar => {ar with $k={var=a; offset}}}
            | _ =>
                a, s
    and inl case_fun s f =
        inl a,s = loop s (function_term_slots_get f)
        function_term_slots_set f a, s

    loop {offset arrays={}} f
```

Let me go over the cases in turn.

```sl
        | () => (), s
        | (a,b) =>
            inl a,s = loop s a
            inl b,s = loop s b
            (a,b), s
        | {} & a =>
            record_fold (fun {state=(m,s) key value} => 
                inl a,s = loop s value
                {m with $key=a}, s
                ) ({}, s) a
```

This is just straightforward iteration over tuples and records. The `record_fold` folds over all the key/value pairs in the record while threading state through the computation. This is all done at compile time. It is very similar to F#'s `fold` and folds in other functional languages. While in languages like F#, Haskell and Ocaml you can only do that over runtime structures like immutable maps, lists and arrays, in Spiral you can do it over compile time records. And this is very useful for interop situations such as these. In particular this is why I wanted these capabalities, for a situation none other than when I am doing a backend for some novel piece of hardware!

If I didn't have this, I'd have build a lot of this functionality into the code generator itself and that would have been a lot harder to deal with.

Though of course, powerful language features such as intensional pattern matching are useful for more than just interop. Let us move on.

```sl
| a when function_is a => case_fun s a
```

In here I do something special when the variable is a compile time function.

```sl
    and inl case_fun s f =
        inl a,s = loop s (function_term_slots_get f)
        function_term_slots_set f a, s
```

Let me cover `function_term_slots_get` and `function_term_slots_get`.

#### `function_term_slots_get` and `function_term_slots_get`

Here is an example that demonstrates their functionality.

```sl
inl main () =
    inl ~(a,b,c) = 1i32,2i32,3i32
    inl f () = a,b,c
    inl x = f()
    $"print(!x)"
    
    inl g x : i32 = x * 2
    inl f' = real 
        open real_core
        inl slots = function_term_slots_get f
        inl rec loop = function
            | () => ()
            | a,b => g a, loop b
        function_term_slots_set f (loop slots)
    inl x = f'()
    $"print(!x)"
    ()
```

When I run this I get...

```
mrakgr@Lain:/mnt/e/PIM-Programming-In-Spiral-UPMEM-Demo/test7$ python3 term_slots.py
(1, 2, 3)
(2, 4, 6)
```

What `function_term_slots_get` does is fetches the compile time environment of the function as a tuple, allowing me to operate on it. After that I pass into the `loop` function which applies `g` to every element except the unit at the end. Then the tuple is used to immutably update the enviroment of `f`. When `f'` is called, it ends up returning the variables doubled.

You wouldn't use these for something as trivial as doubling the slots of a function, but it is very useful if you want to replace the Numpy arrays the function has with DPU native ones.

### main1 (Part 2)

```sl
    and inl case_fun s f =
        inl a,s = loop s (function_term_slots_get f)
        function_term_slots_set f a, s
```

Originally when I was working on these examples I was passing the arrays inside the body, but that had the disadvantage of copying the output arrays into the kernel. Later I changed so that inputs and outputs are denoted explicitly by being passed separate arguments, so this code is not useful, but if you for example pass in a function in the input argument, it will kick in and replace all the arrays inside it with DPU native ones as well as transfer them to the DPU.

```sl
        | a =>
            typecase `a with
            | a ~dim ~t =>
                inl k, offset = !!!!TagToSymbol(!!!!VarTag(a)), s.offset
                inl len = match s with {len} => len | _ => $"len(!a)" : dim
                inl nbytes = len * conv `dim (sizeof `t)
                upmem_kernel_array `dim `t {offset len}, {s with offset#=fun o => roundup (o + nbytes) 8u32; arrays#=fun ar => {ar with $k={var=a; offset}}}
            | _ =>
                a, s
```

Here is the array case. If the array does not have a primitive int or float type as t, the `sizeof` function will give a type error.

```sl
inl k, offset = !!!!TagToSymbol(!!!!VarTag(a)), s.offset
```

Here I take the tag and the current offset.

```sl
inl len = match s with {len} => len | _ => $"len(!a)" : dim
```

This is just the length of array in elements.

```sl
inl nbytes = len * conv `dim (sizeof `t)
```

This is the number of bytes. `sizeof` return an i32 literal, so it has to be converted to the arrays native dimension. 

```sl
upmem_kernel_array `dim `t {offset len}
```

The `dpu_arrays` function is not reponsible for transfering the contents of the arrays to the DPU. Rather it just modifies the structure of the input into the one that can be transfered across language boundaries. That mean, converting them into primitives as well as pointer to them. If you think about it, pointer are just unsigned ints anyway, that is what makes them transferable across language and platform boundaries.

All other datatypes, even those such as bools, differ from language to language. One language might have 4 byte floats, another 1 byte. So they are out as well.

```sl
{s with offset#=fun o => roundup (o + nbytes) 8u32; arrays#=fun ar => {ar with $k={var=a; offset}}}
```

The MRAM arrays have to be 8 byte aligned, so after adding the nbytes to the original offset, I round them up to 8.

```sl
inl roundup a b = (a + (b - 1u32)) / b * b
```

This is a little trick to do that. Try reasoning it out and you should be able to prove to yourself that it in fact rounds up the offsets to 8.

```sl
arrays#=fun ar => {ar with $k={var=a; offset}}
```

This is where the data for the transfer function is built up.

```sl
inl dpu_arrays_transfer dpu arrays = record_iter (fun {key value={var offset}} => dpu_push_array dpu var offset) arrays
```

It just iterates over the `arrays` record and copies them into MRAM.

```sl
let dpu_push_array dpu a offset =
    assert ($"!a.nbytes % 8 == 0" : bool) "The input array has to be divisible by 8"
    $"!dpu.copy('buffer',bytearray(!a),offset=!offset)" : ()
```

Though now that I think about it, the rounding up of offsets is somewhat redundant. I implemented it that way at first, but later found out that in fact the transfer will fail if the array itself is not divisible by 8. I'll leave it like this for now.

```sl
// For transfering the output arrays and variables back to host.
inl dpu_unpack dpu s vars = 
    inl rec loop = function
        | () => ()
        | {} as x => record_iter (fun {key value} => loop value) x
        | (a,b) => loop a . loop b
        | a _ as a =>
            inl {offset} = s.arrays !!!!TagToSymbol(!!!!VarTag(a))
            dpu_pull_array dpu a offset
```

The rest is much the same, but now `dpu_unpack` has a part where it copies the MRAM arrays back to host.

I won't paste the contents of `dpu_pull_array` here. You can look it up in the file.

#### Type Lying

Let us go back to the original example.

```sl
inl add2 forall dim {number} t {number}. = run fun (b,c) (a : a dim t) =>
    global "__mram_noinit uint8_t buffer[1024*1024*1];"
    for {from=0; nearTo=length a} fun i =>
        set a i (index b i + index c i)
    0
```

If you hover the cursor over `b` or `c` you will see that the type of these variables is `a dim t`. That mean they are arrays who indexing type is `dim` and their element type is `t`.

This is actually fake. By the time you access them inside the kernel they will have been replaced and their type will be...

```sl
inl add2 forall dim {number} t {number}. = run fun (b,c) (a : a dim t) =>
    print_static a
```

When I compile this, I get the following in the Spiral language server terminal.

```
{len : u32; offset : u32} : upmem_kernel_array u32 i32
```

In other words, their actual type is `upmem_kernel_array dim t`. This is what the `dpu_arrays` function I've demonstrated does. It is used in `run`. It goes over the inputs and replaces the Numpy arrays in them with the `upmem_kernel_array`s.

As long as you don't put these arrays into other arrays, return them from join points or if statements, you'll be just fine using them the way they are.

The reason why things are like this is because the the top-down type system Spiral uses type unification to do inference. That means, it just resolves equalities. It does very straightforward reasoning that sometimes seems magical, but is actually very primitive. It is very easy to tell it that some type is equal to another.

But it is impossible to say to it: iterate over this type and replace the Numpy arrays with these UPMEM ones. That would go beyond resolving equalities and into program execution. If I tried designing a type system that can do what Spiral's bottom-up partial evaluator can I'd be working on it for years, so I've wisely opted for this course of action. At least, I'd have to drop the current type system that is easy to use and implement and introduce dependent types. Better not. It would be too much work. It would also require the user to start placing annotations everywhere and who wants that?

Let us take a look at `upmem.spi`.

```sl
nominal upmem_kernel_array dim t = {len : dim;  offset : dim}
nominal global_array_ptr dim t = $"__mram_ptr `t *"
```

Here is how the `upmem_kernel_array` and another helper are defined. You can see that the `upmem_kernel_array` is a nominal with two fields. Easy to pass through language boundaries.

```sl
inl at_upmem forall dim t. (upmem_kernel_array {len offset} : _ dim t) : global_array_ptr dim t = $"(__mram_ptr `t *) (buffer + !offset)"
inl index_upmem forall dim t. (upmem_kernel_array {len offset} as a : _ dim t) (i : dim) : t = inl a = at_upmem a in $"!a[!i]"
inl set_upmem forall dim t. (upmem_kernel_array {len offset} as a : _ dim t) (i : dim) (x : t) : () = inl a = at_upmem a in $"!a[!i] = !x"
inl length_upmem forall dim t. (upmem_kernel_array {len offset} : _ dim t) : dim = len
```

Here are the basic index, set and length for it. These arrays can't be directly. The offset first needs to be added to them. That is what the `inl a = at_upmem a` step does. Then it is a piece of cake to access them much like a regular C array `$"!a[!i]"`. The same goes for set.

```sl
inl at forall dim t. (a : a dim t) : global_array_ptr dim t = real 
    match a with 
    | upmem_kernel_array _ => at_upmem `dim `t a

inl index forall dim t. (a : a dim t) (i : dim) : t = real 
    match a with
    | upmem_kernel_array _ => index_upmem `dim `t a i
    | _ => typecase `a with ~ar ~dim ~t => index `ar `dim `t a i

inl set forall dim t. (a : a dim t) (i : dim) (x : t) : () = real 
    match a with
    | upmem_kernel_array _ => set_upmem `dim `t a i x
    | _ => typecase `a with ~ar ~dim ~t => set `ar `dim `t a i x

inl length forall dim t. (a : a dim t) : dim = real 
    match a with
    | upmem_kernel_array _ => length_upmem `dim `t a
    | _ => typecase `a with ~ar ~dim ~t => length `ar `dim `t a
```

These are the overloads that do type lying. You'll note that the top level annotation says `(a : a dim t)`, but in the body I promptly open a real block and test whether that is actually the case. If it is an `upmem_kernel_array` I route them to the respective functions, but otherwise I get the actual type and call the nominal prototype implementations.

```sl
nominal array_ptr dim t = $"`t *"

inl ptr_index forall dim t. (a : array_ptr dim t) (i : dim) : t = $"!a[!i]"
inl ptr_set forall dim t. (a : array_ptr dim t) (i : dim) (x : t) : () = $"!a[!i] = !x"

inl local_ptr forall dim t. (block_size : dim) : array_ptr dim t = 
    assert (lit_is block_size) "The block size should be a compile time literal."
    $"__dma_aligned `t v$[!block_size]"

inl mram_read forall dim {number} t. (src : a dim t) (dst : array_ptr dim t) {from nearTo} =
    inl size : dim = nearTo - from
    inl src = at src
    $"mram_read(!src + !from,!dst,!size * sizeof(`t))" : ()

inl mram_write forall dim {number} t. (src : array_ptr dim t) (dst : a dim t) {from nearTo} =
    inl size : dim = nearTo - from
    inl dst = at dst
    $"mram_write(!src,!dst + !from,!size * sizeof(`t))" : ()
```

I also need local aligned arrays for the sake of `mram_read` and `mram_write` functions. These are easy to index and set as they are just pointers.

Note how...

```sl
inl local_ptr forall dim t. (block_size : dim) : array_ptr dim t = 
    assert (lit_is block_size) "The block size should be a compile time literal."
    $"__dma_aligned `t v$[!block_size]"
```

Here I am finally making use of the `v$` style macros.

In `test8` I dispense these rough overloads in favor of prototypal instances, but the code here was a good starting point to where I could start working towards an inverse array implementation.

```sl
inl add2 forall dim {number} t {number}. = run fun (b,c) (a : a dim t) =>
    global "__mram_noinit uint8_t buffer[1024*1024*1];"
    for {from=0; nearTo=length a} fun i =>
        set a i (index b i + index c i)
    0
```

I've demonstrated this example already. Here one that makes use of all the features discussed so far.

### main2

```sl
// Does the buffered MRAM read work?

open upmem
open upmem_loop
open utils_real
open utils

inl add2 forall dim {number} t {number}. = run fun (b,c) (a : a dim t) =>
    global "#include <mram.h>"
    global "__mram_noinit uint8_t buffer[1024*1024*64];"
    inl block_size = 8 // Values less that 8 freeze the terminal when I try to run this. Maybe the min byte size needs to be 32.
    inl buf_a = local_ptr block_size
    inl buf_b = local_ptr block_size
    inl buf_c = local_ptr block_size
    inl len = length a
    forBy {from=0; nearTo=len; by=block_size} fun from =>
        inl nearTo = min len (from + block_size)
        mram_read b buf_b {from nearTo}
        mram_read c buf_c {from nearTo}
        for {from=0; nearTo=nearTo - from} fun i =>
            ptr_set buf_a i (ptr_index buf_b i + ptr_index buf_c i)
        mram_write buf_a a {from nearTo}
    0
    
inl main () =
    global "import os"
    global "from io import StringIO"
    global "from sys import stdout"
    global "import struct"

    inl test_size = 64
    inl a,b,c = arange test_size, arange test_size, arange test_size
    $"print(!a)"
    add2 (b,c) a
    $"print(!a)"

    ()
```

Running this I get...

```
[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47
 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63]
[  0   2   4   6   8  10  12  14  16  18  20  22  24  26  28  30  32  34
  36  38  40  42  44  46  48  50  52  54  56  58  60  62  64  66  68  70
  72  74  76  78  80  82  84  86  88  90  92  94  96  98 100 102 104 106
 108 110 112 114 116 118 120 122 124 126]
```

It works. If you look at the actual kernel in `main2.py` you'll see that it looks quite good.

Now that I've walked you through it, you should see that this is by no means complicated code. All I am doing is pointing out the obvious. And if you've understood the code so far, you should have no trouble understanding `test8`

## test8 (Part 2)

Here it is, once again.

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
    inl input = rezip (fun x,y,z => {x y z}) input
    $"print(!input)"
    inl output = 
        map (fun {x y z} => 
            inl sqr x = x*x
            sqr (x + y + z)
            ) input
    $"print(!output)"
    ()
```

Running it, I get...

```sl
(array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15],
      dtype=int32), array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15],
      dtype=int32), array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15],
      dtype=int32), 16)
(array([   0,    9,   36,   81,  144,  225,  324,  441,  576,  729,  900,
       1089, 1296, 1521, 1764, 2025], dtype=int32), 16)
```

To summarize what the main function does: I create 3 inverse arrays, zip them together, rezip the tuple fields into a record, before passing them into the map function. The map function then sums them up and squares them.

```sl
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
```

In the previous example, with the regular arrays, I only demonstrated the `add2` kernel rather than generic map functionality. Because only primitive arrays can be sent over the language boundaries, it would have been hard to make a generic map. But this one is very good. Since `run` is identical how it was in the previous example, I'll skip it over. You can check out the details of `utils_read.spir` yourself if you want. What I'll cover here is how inverse arrays are actually implemented.

```sl
    inl buf_in = create block_size
    inl buf_out = create block_size
```

For somebody used to regular programming languages with weak type systems it might be surprising that `create` can be used to make regular, inverse or even an inverse array of local pointers like there.

```sl
    inl buf_in = create block_size
    print_static buf_in
```

Here is the structure I get printed in the Spiral language server terminal.

```
{arrays : {x : i32 * : array_ptr u32 i32; y : i32 * : array_ptr u32 i32; z : i32 * : array_ptr u32 i32}; len : 8u32} : inv_array array_ptr u32 {x : i32; y : i32; z : i32}
```

If you hover over `buf_in` with the cursor you should see: `inv_array array_ptr 'c 'a`. And those `array_ptr` are in fact those local WRAM arrays. Just inferring the type is enough for `create` to make the correct kind of arrays even in the C backend. In this kernel it knows what to make because they are being passed into `mram_read` and `mram_write`.

Here is how they are implemented in `upmem.spi`

```sl
inl mram_read' forall dim {number} t. (src : a dim t) (dst : array_ptr dim t) {from nearTo} =
    inl size : dim = nearTo - from
    inl src = at src
    $"mram_read(!src + !from,!dst,!size * sizeof(`t))" : ()

inl mram_write' forall dim {number} t. (src : array_ptr dim t) (dst : a dim t) {from nearTo} =
    inl size : dim = nearTo - from
    inl dst = at dst
    $"mram_write(!src,!dst + !from,!size * sizeof(`t))" : ()

open inv
inl mram_read forall dim {number} t. (src : inv_array a dim t) (dst : inv_array array_ptr dim t) (r : {from : dim; nearTo : dim}) : () =
    real iam_real.iter2 (fun a b => 
        typecase `a with _ ~t => mram_read' `dim `t a b r
        ) src.arrays dst.arrays

inl mram_write forall dim {number} t. (src : inv_array array_ptr dim t) (dst : inv_array a dim t) (r : {from : dim; nearTo : dim}) : () =
    real iam_real.iter2 (fun a b => 
        typecase `a with _ ~t => mram_write' `dim `t a b r
        ) src.arrays dst.arrays
```

The originals I've renamed to `mram_read'` and `mram_write'`, and am making use of them on the individual elements of the inverse array.

```sl
nominal upmem_kernel_array dim t = {len : dim;  offset : dim}
nominal global_array_ptr dim t = $"__mram_ptr `t *"

inl at_upmem forall dim t. (upmem_kernel_array {len offset} : _ dim t) : global_array_ptr dim t = $"(__mram_ptr `t *) (buffer + !offset)"

instance index upmem_kernel_array = fun (upmem_kernel_array {len offset} as a) i => inl a = at_upmem a in $"!a[!i]"
instance set upmem_kernel_array = fun (upmem_kernel_array {len offset} as a) i x => inl a = at_upmem a in $"!a[!i] = !x"
instance length upmem_kernel_array = fun (upmem_kernel_array {len offset}) : dim => len

nominal array_ptr dim t = $"`t *"

instance create array_ptr = fun block_size =>
    assert (lit_is block_size) "The block size should be a compile time literal."
    $"__dma_aligned `el v$[!block_size]"
instance index array_ptr = fun a i => $"!a[!i]"
instance set array_ptr = fun a i x => $"!a[!i] = !x"
```

For `upmem_kernel_array` and `array_ptr` I dispense with the rough overloads in favor of nominal prototype instances. This allows the inverse array implementation to call them without needing any `upmem_kernel_array` or `array_ptr` specific code to handle them.

Here is how the inverse array is implemented in `inv.spi`. Let us start with the type. It is a bit different from the inverse array implementation in the core library.

```sl
open iam_real
nominal inv_array (ar : * -> * -> *) dim el = {len : dim; arrays : `(infer `ar `dim `el)}
```

If you are familiar with functional programming you might be familiar with most of this, but...

```sl
`(infer `ar `dim `el)
```

This should be new to you. Spiral's nominals are special in that you can go from the type to the term level and execute a program.

Forget the inverse array for a second, and consider the following example.

```sl
nominal qwe = i32 * `( "Hello." )
```

The type of this is `i32 * string`. The `i32` on the left side should be self explanatory. What that that right side does between the parenthesis is receive a "Hello." string literal. They type of that expression is string, so it gets converted into a string type for the nominal.

```sl
nominal qwe = i32 * `( "Hello." )

inl main() =
    inl x = qwe (1, "")
    ()
```

Suppose you try construct it like this, you'll get a type error in the editor saying.

```
Unification failure.
Got:      .<term>
Expected: string
```

This is one of the cases where the type system is being lied to for the greater good. If you try to construct the type in the bottom-up segment you'll see that it works.

```sl
nominal qwe = i32 * `( "Hello." )

inl main() =
    inl x = real qwe (1, "") // No prob
    ()
```

This works. If you try passing something that the expected fields...

```sl
nominal qwe = i32 * `( "Hello." )

inl main() =
    inl x = real qwe (1, true) // Error during partial evaluation.
    ()
```

```
Error trace on line: 3, column: 9 in module: e:\PIM-Programming-In-Spiral-UPMEM-Demo\test8\nominal_example.spi.
inl main() =
        ^
Error trace on line: 4, column: 5 in module: e:\PIM-Programming-In-Spiral-UPMEM-Demo\test8\nominal_example.spi.
    inl x = real qwe (1, true) // Error during partial evaluation.
    ^
Error trace on line: 4, column: 18 in module: e:\PIM-Programming-In-Spiral-UPMEM-Demo\test8\nominal_example.spi.
    inl x = real qwe (1, true) // Error during partial evaluation.
                 ^
Type error in nominal constructor.
Got: i32 * bool
Expected: i32 * string
```

The partial evaluator is a lot more powerful than the top down type system and does type checking on its own. Going back to `inv_array`...

```sl
open iam_real
nominal inv_array (ar : * -> * -> *) dim el = {len : dim; arrays : `(infer `ar `dim `el)}
```

You should now understand that it is calling the `infer` function and passing it the 3 type arguments. Here is how it is implemented in `iam_real` in the `core` package. You can go into it using Ctrl + LMB in the project package file.

```sl
inl infer_templ forall el. g =
    inl rec f forall el. =
        typecase el with
        | ~a * ~b => f `a, f `b
        | {} => record_type_map (fun k => f) `el
        | _ => g `el
    f `el

// Is only supposed to be used for inference.
inl infer forall ar dim el. = infer_templ `el (forall el. => ``(ar dim el))
```

In the above code segment the body of `infer_templ` should be a piece of cake to you by now.

```
``(ar dim el)
```

This thing with the two apostrophes creates a variable of the type `ar dim el` out of nothing. The type inference code will never get generated so it doesn't matter, but if the code generator encountered it, it would give out a type error. In Spiral code generators can return errors along with the trace as well.

Using an exception instead of this would have also worked. Something like...

```
failwith `(ar dim el) "Not supposed to run this."
```

This would have returned the right type, but the `TypeToVar` way is more to the point.

```sl
open iam_real
nominal inv_array (ar : * -> * -> *) dim el = {len : dim; arrays : `(infer `ar `dim `el)}
```

Anyway, now you should be convinced that the inverse arrays in fact genuine typing in Spiral. You can't construct nominals in arbitrary ways, their construction has to adhere to their blueprint.

```sl
// Creates an inverse form arrays.
inl create' forall ar el. size = infer_templ `el (forall el. => create `ar `(`size) `el size)

inl index' ar i =
    inl rec f = function
        | a, b => f a, f b
        | {} as ar => record_map (fun {value} => f value) ar
        | ar => typecase `ar with ~ar ~dim ~el => index `ar `dim `el ar i
    f ar

inl iter2 g a b =
    inl rec f = function
        | (a, b), (va,vb) => f (a, va) . f (b, vb)
        | ({} & ar, {} & v) => record_iter (fun {key value} => f (ar key, value)) v
        | ar,v => g ar v
    f (a, b)

inl set' ar i v = iter2 (fun ar v => typecase `ar with ~ar ~dim ~el => set `ar `dim `el ar i v) ar v
```

Setting and indexing them is really straightforward. As a reminder the `iter2` function that `mram_read` and `mram_write` use is this one. The same one `set'` uses.

```sl
instance create inv_array ar = fun len => inv_array {len arrays=real create' `ar `el len}
instance index inv_array ar = fun (inv_array {arrays}) i => real index' arrays i
instance set inv_array ar = fun (inv_array {arrays}) i v => real set' arrays i v
instance length inv_array ar = fun (inv_array {len}) => len
```

Then to actually implement inverse arrays all I have to do is make prototypal instance for `create`, `index`, `set` and `length`. In them I route to the relevant functions in `iam_core`. With all the pieces in place, I am free to write a generic map function using inverse arrays. It is as simple as writing two nested loops.

```sl
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
```

And that is how all this works! Pretty good, right? This is the right way to write a map function.

# Why This Is So Great

It might not seem impressive at first glance, but you have to consider what Spiral's competition is...

## What Is Not Great About Spiral

First of all, let me do an overview of what Spiral is not competing against. At the time of writing it has a F#, C and Python backends in addition to the UPMEM one presented here.

Compared to F# for example, Spiral gives tight control over inlining and specialization, so I can imagine it being used in performance sensitive applications on the .NET platform. It would probably crush C# for such a purpose thanks to its generative capabilities. But even so, this isn't really my aim as a language designer. If people want to use it for this purpose that is fine, but I am not going to try to push Spiral into such a crowded field by myself. The people in that domain can handle themselves. Maybe if Spiral gets popular, people will realize just how far staged functional programming can get them.

Compared to C, Spiral is way better. It has a ref counted backends, I could do some extensions to the system to manage file handles and GPU memory and it would be quite nice to use. Compared to other functional languages, Spiral is very efficient, so the negative impacts of ref counting on performance would be minimized. Still, just how many C language replacements are there? Do I really want to say my goal is to push into that crowded field? Every year, 10 C replacement languages get made. It is very easy to be better than C. I'll stay out of that warzone.

Compared to Python, Spiral has minimal advantages when using it as a platform. I realize that in 2021, back when Spiral had a Cython backend instead and I tried benchmarking it. Cython presents itself as a speedy alternative to Python, but the reality is that if you are allocating any kind of Python objects, and you are going to be doing so for any code of note, you performance will crater down to the Python level. I benchmarked almost identical kinds of code generated by the F# and the Cython backends, and the F# one was 1,000x times faster! It would have been cool to present a speedy alternative to Python, but Spiral's advantages don't matter here.

If you try programming in Spiral and target F# or Python, you'll quickly see a huge disadvantages of having to use macros for everything. No editor support. No autocomplete. And this really matters to one's enjoyment of programming in a language.

## What Is Truly Great About Spiral

It is essentially times like this. A company has novel piece of hardware. It has maybe a Python and a C backend, and is filled to the brim with low level C programmers. Maybe it is aware that interop with existing platforms as well as writing reusable libraries is a problem its needs to resolve, but has no idea how to proceed. None of the existing languages offer a clean solution to this problem.

I mean which other language support programming in the staged functional programming style apart from Spiral? None, right? And that is what they need.

...