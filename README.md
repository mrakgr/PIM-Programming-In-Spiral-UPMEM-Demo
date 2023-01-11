<!-- TOC -->

- [Intro](#intro)
    - [Main Points Of The Example](#main-points-of-the-example)
    - [Example](#example)
    - [Output](#output)

<!-- /TOC -->

# Intro

I am demoing the UPMEM Python + C backend for Spiral here. I want Spiral to be a language suitable for future generations of computing devices, and since UPMEM commercialized the first PIM chip, it is the ideal target for a demo. These backends are easy to make. They can be done in 0.5-1 week's time, after which one can go from the level of programming in C to a level of programming in a highly expressive functional language. One of the goals of Spiral's design is to be efficient enough for writing GPU kernels in it, and it has been met. Spiral has the most efficient possible design for a functional language without sacrificing expressivity.

Spiral's intended niche is not any device or platform in particular, but the intersection of them. Spiral is peerlessly well-suited for heterogeneous computing architectures of the future, and trivializes the data transfer between different pieces of hardware in the system.

I hope to get support for this kind of work, so that I can demonstrate my claims on different classes of hardware. If you are a company struggling with C programming on some novel piece of hardware, don't hesitate to get in touch. Going from programming in a language like C to programming in a high level language like Spiral will hugely raise your productivity. 

At the moment, I am also open for paid work on such systems personally. I'd find it satisfying to significantly improve your company's efficiency in programming using the power of (staged) functional programming, so if you find C programming torturous and are open minded enough to try it, don't hesitate to get in touch. Spiral offers a novel programming style that does not have the inefficiencies associated with regular functional PLs and is suitable for devices with restricted memory.

## Main Points Of The Example

* This is a map function, similar to the ones in [F#](https://fsharp.github.io/fsharp-core-docs/reference/fsharp-collections-arraymodule.html#map) and [OCaml](http://www.mega-nerd.com/erikd/Blog/CodeHacking/Ocaml/iter_and_map.html).
* You could implement it in a couple of lines of code in an impure functional language like those two. The same goes for Spiral.
* It would take about 10 minutes to implement.
* Since this map kernel uses a nested loop to make use of intermediate buffers, it is a bit harder, and would take 20 minutes.
* That is only if you have a language and backend for the particular device you are targeting. If you don't it would take a couple of years on top of that.
* The example also demonstrates the use of a [structure of arrays](https://en.wikipedia.org/wiki/AoS_and_SoA) datatype. In Spiral, they are called inverse arrays.
* Spiral is powerful enough as a language enough that you can implement such structures directly in it. It does not require special language support.
* If you look at the resulting code, you'll also see that the example demonstrates inlining of constants across platform boundaries. In the example provided, the array length and offsets are known statically, and the join points are specialized for them.
* There is a much longer article in the [`long`](/long/) directory that shows how the backend is built up from scratch, as well as how the inverse array is implemented.

## Example

You can find the full code [here](/test8/main.spi).

```spiral
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
    inl output = map (fun x, y, z => x+y+z,x*y*z) input
    $"print(!output)"
    ()
```

## Output

Compiling the above with Ctrl+F1 using Spiral generates the [`main.py`](/test8/main.py) file that has a bunch Python as well a C kernels in strings. Running it prints the inputs as well as the outputs.

```
mrakgr@Lain:/mnt/e/PIM-Programming-In-Spiral-UPMEM-Demo$ . ~/upmem-sdk/upmem_env.sh 
Setting UPMEM_HOME to /home/mrakgr/upmem-sdk and updating PATH/LD_LIBRARY_PATH/PYTHONPATH
mrakgr@Lain:/mnt/e/PIM-Programming-In-Spiral-UPMEM-Demo$ cd /test8
mrakgr@Lain:/mnt/e/PIM-Programming-In-Spiral-UPMEM-Demo/test8$ python3 main.py
(array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15],
      dtype=int32), array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15],
      dtype=int32), array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15],
      dtype=int32), 16)
(array([ 0,  3,  6,  9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45],
      dtype=int32), array([   0,    1,    8,   27,   64,  125,  216,  343,  512,  729, 1000,
       1331, 1728, 2197, 2744, 3375], dtype=int32), 16)
```