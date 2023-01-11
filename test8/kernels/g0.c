
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
