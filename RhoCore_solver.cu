#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <cassert>
#include <vector>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <map>

// Host-seitige Abhängigkeit für das Parsen von Parametern
#include "json.hpp"

using json = nlohmann::json;

// =============================================================
// TYPEN & STRUKTUREN
// =============================================================

struct scalar_n {
    uint64_t v[4];
};

struct jacobian_point {
    scalar_n X, Y, Z;
};

struct affine_point {
    scalar_n x, y;
};

// =============================================================
// KONSTANTEN & DEFINITIONEN
// =============================================================

#define MAX_NAF_BITS 257

// --- Arithmetik Device-Konstanten ---
static __device__ const scalar_n GROUP_ORDER = {{
    0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL
}};
static __device__ const scalar_n FIELD_P = {{
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFEFFFFFC2FULL
}};
static __device__ const uint64_t FIELD_P_INV_MONT = 0xD838091DD2253531ULL;
static __device__ const scalar_n GROUP_ORDER_HALF = {{
    0xDFE92F46681B20A0ULL, 0x5D576E7357A4501DULL,
    0xFFFFFFFFFFFFFFFFULL, 0x7FFFFFFFFFFFFFFFULL
}};
static __device__ const scalar_n BETA = {{
    0x7AE96A2B657C0710ULL, 0x852A7C8585D87B38ULL,
    0x400DE2D6E6447736ULL, 0xDD688A57AE96A2B6ULL
}};

// GLV Zerlegungskonstanten
static __device__ const scalar_n N_MINUS_1_DIV_2 = {{
    0xDFE92F46681B20A0ULL, 0x5D576E7357A4501DULL,
    0xFFFFFFFFFFFFFFFFULL, 0x7FFFFFFFFFFFFFFFULL
}};
static __device__ const scalar_n G1 = {{
    0x3086D221A7D46BCDU, 0xE86C90E49284EB15, 0, 0
}};
static __device__ const scalar_n G2 = {{
    0xE4437ED6010E8828, 0x6F547FA90ABFE4C3, 0, 0
}};

// Precomputed tables for point multiplication
__device__ jacobian_point WNAF5_LUT_G[16];
__device__ jacobian_point WNAF5_LUT_LAMBDA_G[16];

// =============================================================
// BIG-INTEGER ARITHMETIK (MODULO N für Skalare)
// =============================================================

__host__ __device__ __forceinline__ void scalar_copy(scalar_n* dst, const scalar_n* src) {
    dst->v[0] = src->v[0]; dst->v[1] = src->v[1];
    dst->v[2] = src->v[2]; dst->v[3] = src->v[3];
}

__host__ __device__ __forceinline__ int scalar_cmp(const scalar_n* a, const scalar_n* b) {
    for (int i = 3; i >= 0; i--) {
        if (a->v[i] < b->v[i]) return -1;
        if (a->v[i] > b->v[i]) return 1;
    }
    return 0;
}

__host__ __device__ __forceinline__ bool scalar_is_zero(const scalar_n* a) {
    return (a->v[0] | a->v[1] | a->v[2] | a->v[3]) == 0ULL;
}

__host__ __device__ __forceinline__ bool scalar_is_odd(const scalar_n* a) {
    return (a->v[0] & 1ULL) != 0ULL;
}

__host__ __device__ __forceinline__ void scalar_rshift1(scalar_n* r) {
    r->v[0] = (r->v[0] >> 1) | (r->v[1] << 63);
    r->v[1] = (r->v[1] >> 1) | (r->v[2] << 63);
    r->v[2] = (r->v[2] >> 1) | (r->v[3] << 63);
    r->v[3] = (r->v[3] >> 1);
}

__host__ __device__ void scalar_add_mod_n(scalar_n* r, const scalar_n* a, const scalar_n* b) {
    unsigned __int128 s = 0;
    uint64_t carry = 0ULL;
    for (int i = 0; i < 4; i++) {
        s = (unsigned __int128)a->v[i] + b->v[i] + carry;
        r->v[i] = (uint64_t)s;
        carry = (uint64_t)(s >> 64);
    }
    if (carry || scalar_cmp(r, &GROUP_ORDER) >= 0) {
        unsigned __int128 d;
        uint64_t br = 0ULL;
        for (int i = 0; i < 4; i++) {
            d = (unsigned __int128)r->v[i] - GROUP_ORDER.v[i] - br;
            r->v[i] = (uint64_t)d;
            br = (uint64_t)((d >> 127) & 1ULL);
        }
    }
}

__host__ __device__ void scalar_sub_mod_n(scalar_n* r, const scalar_n* a, const scalar_n* b) {
    unsigned __int128 diff = 0;
    uint64_t borrow = 0;
    scalar_n tmp;
    for (int i = 0; i < 4; i++) {
        diff = (unsigned __int128)a->v[i] - b->v[i] - borrow;
        tmp.v[i] = (uint64_t)diff;
        borrow = (diff >> 127) & 1;
    }
    if (borrow) {
        unsigned __int128 sum = 0;
        uint64_t carry = 0;
        for (int i = 0; i < 4; i++) {
            sum = (unsigned __int128)tmp.v[i] + GROUP_ORDER.v[i] + carry;
            tmp.v[i] = (uint64_t)sum;
            carry = (uint64_t)(sum >> 64);
        }
    }
    scalar_copy(r, &tmp);
}

__host__ __device__ void scalar_negate_mod_n(scalar_n* r, const scalar_n* a) {
    if (scalar_is_zero(a)) {
        r->v[0] = r->v[1] = r->v[2] = r->v[3] = 0ULL;
    } else {
       scalar_sub_mod_n(r, &GROUP_ORDER, a);
    }
}


// =============================================================
// FELDELEMENT-ARITHMETIK (MODULO P)
// =============================================================

__host__ __device__ void field_add(scalar_n *r, const scalar_n *a, const scalar_n *b) {
    unsigned __int128 carry = 0;
    for(int i=0; i<4; ++i) {
        unsigned __int128 sum = (unsigned __int128)a->v[i] + b->v[i] + carry;
        r->v[i] = (uint64_t)sum;
        carry = sum >> 64;
    }
    if (carry || scalar_cmp(r, &FIELD_P) >= 0) {
        scalar_sub_mod_n(r, r, &FIELD_P);
    }
}

__host__ __device__ void field_sub(scalar_n *r, const scalar_n *a, const scalar_n *b) {
    scalar_n tmp;
    unsigned __int128 diff=0; uint64_t borrow=0;
    for(int i=0; i<4; i++) {
        diff = (unsigned __int128)a->v[i] - b->v[i] - borrow;
        tmp.v[i] = (uint64_t)diff;
        borrow = (diff >> 127) & 1;
    }
    if (borrow) {
        scalar_add_mod_n(&tmp, &tmp, &FIELD_P);
    }
    scalar_copy(r, &tmp);
}


__host__ __device__ void field_mul_montgomery(scalar_n &res, const scalar_n &a, const scalar_n &b) {
    uint64_t t[8] = {0};
    for(int i=0;i<4;i++){
        uint64_t carry=0;
        for(int j=0;j<4;j++){
            unsigned __int128 prod = (unsigned __int128)a.v[i]*b.v[j] + t[i+j] + carry;
            t[i+j] = (uint64_t)prod;
            carry = (uint64_t)(prod>>64);
        }
        t[i+4] = carry;
    }
    for(int i=0;i<4;i++){
        uint64_t m = t[i]*FIELD_P_INV_MONT;
        uint64_t carry=0;
        for(int j=0;j<4;j++){
            unsigned __int128 prod = (unsigned __int128)m*FIELD_P.v[j]+t[i+j]+carry;
            t[i+j]=(uint64_t)prod;
            carry=(uint64_t)(prod>>64);
        }
        int k=i+4;
        while(carry){
            unsigned __int128 sum = (unsigned __int128)t[k]+carry;
            t[k]=(uint64_t)sum;
            carry=(uint64_t)(sum>>64);
            k++;
        }
    }
    for(int i=0;i<4;i++) res.v[i]=t[i+4];
    if(scalar_cmp(&res, &FIELD_P) >= 0) {
        field_sub(&res, &res, &FIELD_P);
    }
}

__host__ __device__ void field_inverse(scalar_n* r, const scalar_n* a) {
    scalar_n u, v, x1, x2;
    scalar_copy(&u, a);
    scalar_copy(&v, &FIELD_P);
    x1 = {{1,0,0,0}};
    x2 = {{0,0,0,0}};

    while(!scalar_is_zero(&u) && !scalar_is_zero(&v)) {
        while(!scalar_is_odd(&u)) {
            scalar_rshift1(&u);
            if(scalar_is_odd(&x1)) field_add(&x1, &x1, &FIELD_P);
            scalar_rshift1(&x1);
        }
         while(!scalar_is_odd(&v)) {
            scalar_rshift1(&v);
            if(scalar_is_odd(&x2)) field_add(&x2, &x2, &FIELD_P);
            scalar_rshift1(&x2);
        }
        if(scalar_cmp(&u, &v) >= 0) {
            field_sub(&u, &u, &v);
            field_sub(&x1, &x1, &x2);
        } else {
            field_sub(&v, &v, &u);
            field_sub(&x2, &x2, &x1);
        }
    }
    scalar_copy(r, scalar_is_zero(&u) ? &x2 : &x1);
}

// =============================================================
// KORREKTE JACOBI-PUNKTARITHMETIK (MODULO P)
// =============================================================

__host__ __device__ void point_set_infinity(jacobian_point* p) {
    p->X = p->Y = {{0,0,0,0}};
    p->Z = {{0,0,0,0}};
}
__host__ __device__ bool point_is_infinity(const jacobian_point* p) {
    return scalar_is_zero(&p->Z);
}
__host__ __device__ void point_assign(jacobian_point* dst, const jacobian_point* src) {
    scalar_copy(&dst->X, &src->X);
    scalar_copy(&dst->Y, &src->Y);
    scalar_copy(&dst->Z, &src->Z);
}
__host__ __device__ void point_negate(jacobian_point* r, const jacobian_point* p) {
    scalar_copy(&r->X, &p->X);
    field_sub(&r->Y, &FIELD_P, &p->Y);
    scalar_copy(&r->Z, &p->Z);
}

__host__ __device__ void point_double(jacobian_point* r, const jacobian_point* p){
    if(point_is_infinity(p)){ point_set_infinity(r); return; }
    scalar_n XX, YY, YYYY, S, M, T;
    field_mul_montgomery(XX, p->X, p->X);
    field_mul_montgomery(YY, p->Y, p->Y);
    field_mul_montgomery(YYYY, YY, YY);
    field_mul_montgomery(S, p->X, YY);
    field_add(&S, &S, &S); field_add(&S, &S, &S);
    field_add(&M, &XX, &XX); field_add(&M, &M, &XX);
    field_mul_montgomery(T, M, M);
    field_sub(&T, &T, &S); field_sub(&r->X, &T, &S); 
    field_sub(&T, &S, &r->X);
    field_mul_montgomery(r->Y, M, T);
    scalar_n YYYY8;
    field_add(&YYYY8, &YYYY, &YYYY); field_add(&YYYY8, &YYYY8, &YYYY8); field_add(&YYYY8, &YYYY8, &YYYY8);
    field_sub(&r->Y, &r->Y, &YYYY8);
    field_mul_montgomery(r->Z, p->Y, p->Z);
    field_add(&r->Z, &r->Z, &r->Z);
}

__host__ __device__ void point_add(jacobian_point* r, const jacobian_point* p, const jacobian_point* q){
    if(point_is_infinity(p)){ point_assign(r, q); return; }
    if(point_is_infinity(q)){ point_assign(r, p); return; }

    scalar_n Z1Z1, Z2Z2, U1, U2, S1, S2, H, I, J, r_val, V;
    field_mul_montgomery(Z1Z1, p->Z, p->Z);
    field_mul_montgomery(Z2Z2, q->Z, q->Z);
    field_mul_montgomery(U1, p->X, Z2Z2);
    field_mul_montgomery(U2, q->X, Z1Z1);
    
    scalar_n Z2_cubed;
    field_mul_montgomery(Z2_cubed, q->Z, Z2Z2);
    field_mul_montgomery(S1, p->Y, Z2_cubed);
    
    scalar_n Z1_cubed;
    field_mul_montgomery(Z1_cubed, p->Z, Z1Z1);
    field_mul_montgomery(S2, q->Y, Z1_cubed);
    
    field_sub(&H, &U2, &U1);
    field_sub(&r_val, &S2, &S1);

    if(scalar_is_zero(&H)){
        if(scalar_is_zero(&r_val)){ point_double(r, p); return; } 
        else { point_set_infinity(r); return; }
    }

    field_add(&I, &H, &H);
    field_mul_montgomery(I, I, I);
    field_mul_montgomery(J, H, I);
    field_mul_montgomery(V, U1, I);
    
    field_mul_montgomery(r->X, r_val, r_val);
    field_sub(&r->X, &r->X, &J);
    field_sub(&r->X, &r->X, &V);
    field_sub(&r->X, &r->X, &V);

    scalar_n tmp;
    field_sub(&tmp, &V, &r->X);
    field_mul_montgomery(r->Y, r_val, tmp);
    field_mul_montgomery(tmp, S1, J);
    field_add(&tmp, &tmp, &tmp);
    field_sub(&r->Y, &r->Y, &tmp);

    field_add(&tmp, &p->Z, &q->Z);
    field_mul_montgomery(tmp, tmp, tmp);
    field_sub(&tmp, &tmp, &Z1Z1);
    field_sub(&tmp, &tmp, &Z2Z2);
    field_mul_montgomery(r->Z, tmp, H);
}

// =============================================================
// GLV SKALARMULTIPLIKATION
// =============================================================
__host__ __device__ void scalar_mul_128(uint64_t* h, uint64_t* l, const scalar_n* s, const uint64_t m0, const uint64_t m1) {
    unsigned __int128 p0 = (unsigned __int128)s->v[0] * m0;
    unsigned __int128 p1 = (unsigned __int128)s->v[0] * m1;
    unsigned __int128 p2 = (unsigned __int128)s->v[1] * m0;
    unsigned __int128 p3 = (unsigned __int128)s->v[1] * m1;
    unsigned __int128 p4 = (unsigned __int128)s->v[2] * m0;
    unsigned __int128 p5 = (unsigned __int128)s->v[2] * m1;
    unsigned __int128 p6 = (unsigned __int128)s->v[3] * m0;
    unsigned __int128 p7 = (unsigned __int128)s->v[3] * m1;
    
    p1 += p0 >> 64; p2 += (uint64_t)p1; p3 += p2 >> 64; p3 += p1 >> 64;
    p4 += (uint64_t)p3; p5 += p4 >> 64; p5 += p3 >> 64;
    p6 += (uint64_t)p5; p7 += p6 >> 64; p7 += p5 >> 64;
    
    *h = (uint64_t)p7;
    *l = (uint64_t)p6;
}

__host__ __device__ void mul_256x256_to_512(uint64_t out[8], const scalar_n* A, const scalar_n* B) {
    #pragma unroll
    for (int i = 0; i < 8; i++) out[i] = 0ULL;
    for (int i = 0; i < 4; i++) {
        uint64_t carry = 0ULL;
        for (int j = 0; j < 4; j++) {
            unsigned __int128 prod = (unsigned __int128)A->v[i] * B->v[j] + out[i+j] + carry;
            out[i+j] = (uint64_t)prod;
            carry = (uint64_t)(prod >> 64);
        }
        int pos = i + 4;
        while (carry && pos < 8) {
            unsigned __int128 sum2 = (unsigned __int128)out[pos] + carry;
            out[pos] = (uint64_t)sum2;
            carry = (uint64_t)(sum2 >> 64);
            pos++;
        }
    }
}


__device__ void scalar_split_glv(scalar_n* k1, scalar_n* k2, const scalar_n* k) {
    uint64_t c1h, c1l, c2h, c2l;
    scalar_mul_128(&c1h, &c1l, k, G1.v[0], G1.v[1]);
    scalar_mul_128(&c2h, &c2l, k, G2.v[0], G2.v[1]);

    scalar_n c1 = {{c1l, c1h, 0, 0}};
    scalar_n c2 = {{c2l, c2h, 0, 0}};

    scalar_n tmp1, tmp2;
    uint64_t p1[8], p2[8];
    mul_256x256_to_512(p1, &c1, &G1);
    mul_256x256_to_512(p2, &c2, &G2);

    scalar_add_mod_n(&tmp1, (scalar_n*)p1, (scalar_n*)p2);
    scalar_sub_mod_n(k1, k, &tmp1);
    
    mul_256x256_to_512(p1, &c1, &G2);
    mul_256x256_to_512(p2, &c2, &G1);
    
    scalar_sub_mod_n(&tmp1, (scalar_n*)p1, (scalar_n*)p2);
    scalar_negate_mod_n(k2, &tmp1);
}

__device__ void scalar_to_wnaf(int* wnaf, int& len, const scalar_n* s, int w) {
    len = 0;
    scalar_n d;
    scalar_copy(&d, s);
    int width = 1 << w;
    int half_width = 1 << (w - 1);

    while (!scalar_is_zero(&d)) {
        int val;
        if (scalar_is_odd(&d)) {
            val = d.v[0] & (width - 1);
            if (val >= half_width) {
                val -= width;
            }
            if(val > 0) {
                scalar_n val_s = {{(uint64_t)val,0,0,0}};
                scalar_sub_mod_n(&d, &d, &val_s);
            }
            else {
                scalar_n val_s = {{(uint64_t)-val,0,0,0}};
                scalar_add_mod_n(&d, &d, &val_s);
            }
        } else {
            val = 0;
        }
        wnaf[len++] = val;
        scalar_rshift1(&d);
    }
}

__device__ void point_mul_glv(jacobian_point* r, const scalar_n* k) {
    scalar_n k1, k2;
    scalar_split_glv(&k1, &k2, k);
    
    int wnaf1[MAX_NAF_BITS], wnaf2[MAX_NAF_BITS];
    int len1, len2;
    scalar_to_wnaf(wnaf1, len1, &k1, 5);
    scalar_to_wnaf(wnaf2, len2, &k2, 5);

    int len = len1 > len2 ? len1 : len2;

    point_set_infinity(r);
    for (int i = len - 1; i >= 0; i--) {
        point_double(r, r);
        if(i < len1 && wnaf1[i] != 0) {
            if(wnaf1[i] > 0) point_add(r, r, &WNAF5_LUT_G[wnaf1[i]/2]);
            else {
                jacobian_point neg_G;
                point_negate(&neg_G, &WNAF5_LUT_G[-wnaf1[i]/2]);
                point_add(r, r, &neg_G);
            }
        }
        if(i < len2 && wnaf2[i] != 0) {
            if(wnaf2[i] > 0) point_add(r, r, &WNAF5_LUT_LAMBDA_G[wnaf2[i]/2]);
            else {
                jacobian_point neg_LG;
                point_negate(&neg_LG, &WNAF5_LUT_LAMBDA_G[-wnaf2[i]/2]);
                point_add(r, r, &neg_LG);
            }
        }
    }
}


// =============================================================
// AFFIN-KONVERTIERUNG, HASHING, SUCHE
// =============================================================

__device__ void jacobian_to_affine(const jacobian_point* P, affine_point* R) {
    if(point_is_infinity(P)) { R->x = R->y = {{0,0,0,0}}; return; }
    scalar_n z_inv, z_inv_sq, z_inv_cubed;
    field_inverse(&z_inv, &P->Z);
    field_mul_montgomery(z_inv_sq, z_inv, z_inv);
    field_mul_montgomery(z_inv_cubed, z_inv_sq, z_inv);
    field_mul_montgomery(R->x, P->X, z_inv_sq);
    field_mul_montgomery(R->y, P->Y, z_inv_cubed);
}

__device__ void affine_to_compressed_pubkey(const affine_point* P, uint8_t* out33) {
    out33[0] = (P->y.v[0] & 1) ? 0x03 : 0x02;
    for(int i=0; i<4; ++i) {
        uint64_t limb = P->x.v[3-i];
        for(int j=0; j<8; ++j) out33[1 + i*8 + j] = (limb >> (56 - j*8)) & 0xFF;
    }
}

__device__ __forceinline__ uint32_t ROTR(uint32_t x, uint32_t n){ return (x>>n)|(x<<(32-n)); }
__device__ __forceinline__ uint32_t CH(uint32_t x, uint32_t y, uint32_t z){ return (x&y) ^ (~x & z); }
__device__ __forceinline__ uint32_t MAJ(uint32_t x, uint32_t y, uint32_t z){ return (x&y) ^ (x&z) ^ (y&z); }
__device__ __forceinline__ uint32_t EP0(uint32_t x){ return ROTR(x,2)^ROTR(x,13)^ROTR(x,22); }
__device__ __forceinline__ uint32_t EP1(uint32_t x){ return ROTR(x,6)^ROTR(x,11)^ROTR(x,25); }
__device__ __forceinline__ uint32_t SIG0(uint32_t x){ return ROTR(x,7)^ROTR(x,18)^(x>>3); }
__device__ __forceinline__ uint32_t SIG1(uint32_t x){ return ROTR(x,17)^ROTR(x,19)^(x>>10); }

__device__ void sha256_gpu(const uint8_t *data, int len, uint8_t *hash); // Prototyp
__device__ void ripemd160_gpu(const uint8_t *data, int len, uint8_t *hash); // Prototyp

__device__ void sha256_ripe160(const uint8_t* input, size_t len, uint8_t* out20) {
    uint8_t sha256_hash[32];
    sha256_gpu(input, len, sha256_hash);
    ripemd160_gpu(sha256_hash, 32, out20);
}

__device__ __forceinline__ int memcmp_20(const uint8_t* hash_a, const uint8_t* hash_b) {
    #pragma unroll
    for (int i = 0; i < 20; ++i) {
        if (hash_a[i] < hash_b[i]) return -1;
        if (hash_a[i] > hash_b[i]) return 1;
    }
    return 0;
}

__device__ bool binary_search_hash160(const uint8_t* sorted_db, uint64_t db_entry_count, const uint8_t* target_hash) {
    uint64_t low = 0;
    uint64_t high = db_entry_count > 0 ? db_entry_count - 1 : 0;
    while (low <= high) {
        uint64_t mid = low + (high - low) / 2;
        const uint8_t* mid_hash = sorted_db + (mid * 20);
        int cmp = memcmp_20(target_hash, mid_hash);
        if (cmp == 0) return true;
        else if (cmp < 0) { if (mid == 0) break; high = mid - 1; } 
        else low = mid + 1;
    }
    return false;
}

// =============================================================
// KERNEL ZUR SCHLÜSSELSÜCHE
// =============================================================
__global__ void key_search_kernel(
    const scalar_n* private_keys,
    size_t num_keys,
    const uint8_t* hash160_db,
    uint64_t db_count,
    scalar_n* found_key,
    int* found_flag) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_keys) return;

    // 1. Berechne den Public Key mit GLV
    scalar_n private_key = private_keys[idx];
    jacobian_point public_key;
    point_mul_glv(&public_key, &private_key);

    // 2. Konvertiere zu Affin und komprimiere
    affine_point pub_affine;
    jacobian_to_affine(&public_key, &pub_affine);
    uint8_t pubkey_compressed[33];
    affine_to_compressed_pubkey(&pub_affine, pubkey_compressed);

    // 3. Hashe den Public Key
    uint8_t hash_result[20];
    sha256_ripe160(pubkey_compressed, 33, hash_result);

    // 4. Suche in der Datenbank
    if (binary_search_hash160(hash160_db, db_count, hash_result)) {
        if (atomicCAS(found_flag, 0, 1) == 0) {
            *found_key = private_key;
        }
    }
}

__device__ void sha256_gpu(const uint8_t *data, int len, uint8_t *hash) {
    const uint32_t K[64] = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
    };
    uint32_t H[8] = {
        0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
        0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19
    };
    uint8_t block[64];
    
    for(int i=0; i<len; ++i) block[i] = data[i];
    block[len] = 0x80;
    for(int i=len+1; i<56; ++i) block[i] = 0;
    uint64_t bit_len = (uint64_t)len * 8;
    for(int i=0; i<8; ++i) block[56+i] = (bit_len >> (56 - i*8)) & 0xFF;

    uint32_t W[64];
    for(int i=0;i<16;i++) W[i]=(uint32_t)(block[i*4]<<24)|(uint32_t)(block[i*4+1]<<16)|(uint32_t)(block[i*4+2]<<8)|(uint32_t)block[i*4+3];
    for(int i=16;i<64;i++) W[i]=SIG1(W[i-2])+W[i-7]+SIG0(W[i-15])+W[i-16];

    uint32_t a=H[0],b=H[1],c=H[2],d=H[3],e=H[4],f=H[5],g=H[6],h=H[7];
    for(int i=0;i<64;i++){
        uint32_t T1=h+EP1(e)+CH(e,f,g)+K[i]+W[i];
        uint32_t T2=EP0(a)+MAJ(a,b,c);
        h=g; g=f; f=e; e=d+T1; d=c; c=b; b=a; a=T1+T2;
    }
    H[0]+=a; H[1]+=b; H[2]+=c; H[3]+=d; H[4]+=e; H[5]+=f; H[6]+=g; H[7]+=h;
    
    for(int i=0;i<8;i++){
        hash[i*4+0]=(H[i]>>24)&0xFF; hash[i*4+1]=(H[i]>>16)&0xFF;
        hash[i*4+2]=(H[i]>>8)&0xFF; hash[i*4+3]=H[i]&0xFF;
    }
}

__device__ __forceinline__ uint32_t ROL(uint32_t x,int n){ return (x<<n)|(x>>(32-n)); }

__device__ void ripemd160_process_block(uint32_t H[5], const uint32_t M[16]) {
    const int r[80]={ 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
        7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8,
        3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12,
        1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2,
        4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13};
    const int s[80]={11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8,
        7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12,
        11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5,
        11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12,
        9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6};
    const uint32_t K[5]={0x00000000,0x5A827999,0x6ED9EBA1,0x8F1BBCDC,0xA953FD4E};
    const uint32_t J[5]={0x50A28BE6,0x5C4DD124,0x6D703EF3,0x7A6D76E9,0x00000000};
    
    uint32_t A=H[0],B=H[1],C=H[2],D=H[3],E=H[4];
    uint32_t Ap=H[0],Bp=H[1],Cp=H[2],Dp=H[3],Ep=H[4];

    for(int i=0;i<80;i++){
        uint32_t T=ROL(A+((i<16)?(B^C^D):((i<32)?((B&C)|(~B&D)):((i<48)?((B|~C)^D):((i<64)?((B&D)|(C&~D)):(B^(C|~D))))))+M[r[i]]+K[i/16],s[i])+E;
        A=E; E=D; D=ROL(C,10); C=B; B=T;
        T=ROL(Ap+((i<16)?(Bp^(Cp|~Dp)):((i<32)?((Bp&Dp)|(Cp&~Dp)):((i<48)?((Bp|~Cp)^Dp):((i<64)?((Bp&Cp)|(~Bp&Dp)):(Bp^Cp^Dp)))))+M[r[79-i]]+J[i/16],s[79-i])+Ep;
        Ap=Ep; Ep=Dp; Dp=ROL(Cp,10); Cp=Bp; Bp=T;
    }
    D+=C+Ep; H[1]+=D+Ap; H[2]+=E+Bp; H[3]+=A+Cp; H[4]+=B+Dp; H[0]=D;
}

__device__ void ripemd160_gpu(const uint8_t *data, int len, uint8_t *hash) {
    uint32_t H[5]={0x67452301,0xEFCDAB89,0x98BADCFE,0x10325476,0xC3D2E1F0};
    uint32_t M[16] = {0};
    uint64_t bit_len = (uint64_t)len * 8;

    int current_byte = 0;
    while (current_byte + 64 <= len) {
        for (int i = 0; i < 16; i++) {
            M[i] = (uint32_t)(data[current_byte + i*4 + 0]) |
                   (uint32_t)(data[current_byte + i*4 + 1] << 8) |
                   (uint32_t)(data[current_byte + i*4 + 2] << 16) |
                   (uint32_t)(data[current_byte + i*4 + 3] << 24);
        }
        ripemd160_process_block(H, M);
        current_byte += 64;
    }

    int remaining_bytes = len - current_byte;
    for (int i = 0; i < 16; i++) M[i] = 0;
    for (int i = 0; i < remaining_bytes; i++) {
        M[i/4] |= (uint32_t)data[current_byte + i] << (8 * (i % 4));
    }
    M[remaining_bytes/4] |= (uint32_t)0x80 << (8 * (remaining_bytes % 4));

    if (remaining_bytes >= 56) {
        ripemd160_process_block(H, M);
        for (int i = 0; i < 16; i++) M[i] = 0;
    }
    M[14] = (uint32_t)(bit_len & 0xFFFFFFFF);
    M[15] = (uint32_t)((bit_len >> 32) & 0xFFFFFFFF);
    ripemd160_process_block(H, M);

    for(int i=0;i<5;i++){
        hash[i*4+0]=H[i]&0xFF; hash[i*4+1]=(H[i]>>8)&0xFF;
        hash[i*4+2]=(H[i]>>16)&0xFF; hash[i*4+3]=(H[i]>>24)&0xFF;
    }
}

// =============================================================
// HOST-CODE
// =============================================================

json load_params(const char* file) {
    std::ifstream f(file);
    if (!f.is_open()) {
        throw std::runtime_error(std::string("Could not open parameter file: ") + file);
    }
    json j;
    f >> j;
    return j;
}

std::vector<uint8_t> hex_to_bytes(const std::string& hex) {
    std::vector<uint8_t> bytes;
    for (size_t i = 0; i < hex.length(); i += 2) {
        std::string byte_str = hex.substr(i, 2);
        bytes.push_back(static_cast<uint8_t>(std::stoul(byte_str, nullptr, 16)));
    }
    return bytes;
}

std::string scalar_to_hex(const scalar_n& s) {
    char hex_chars[65];
    hex_chars[64] = '\0';
    for (int i = 0; i < 4; ++i) {
        snprintf(&hex_chars[i * 16], 17, "%016llx", s.v[3 - i]);
    }
    return std::string(hex_chars);
}

bool has_hex_triples(const std::string& hex) {
    if (hex.length() < 3) return false;
    for (size_t i = 0; i <= hex.length() - 3; ++i) {
        if (hex[i] == hex[i+1] && hex[i+1] == hex[i+2]) return true;
    }
    return false;
}

bool has_adjacent_double_pairs(const std::string& hex) {
    if (hex.length() < 4) return false;
    for (size_t i = 0; i <= hex.length() - 4; ++i) {
        if (hex[i] == hex[i+1] && hex[i+2] == hex[i+3] && hex[i] != hex[i+2]) return true;
    }
    return false;
}

int count_set_bits(const scalar_n& s) {
    int count = 0;
    for (int i = 0; i < 4; ++i) {
        count += __builtin_popcountll(s.v[i]);
    }
    return count;
}

bool is_hex_score_ok(const std::string& hex, const std::map<char, int>& penalties) {
    for (char c : hex) {
        auto it = penalties.find(c);
        if (it != penalties.end() && it->second >= 5) return false;
    }
    return true;
}

void generate_filtered_privkeys(
    std::vector<scalar_n>& candidates,
    const json& params,
    size_t n
) {
    std::vector<uint8_t> range_min_bytes = hex_to_bytes(params["range_min"].get<std::string>());
    scalar_n current_key = {{0}};
    for (size_t i = 0; i < range_min_bytes.size() && i < 32; ++i) {
        current_key.v[i/8] |= (uint64_t)range_min_bytes[range_min_bytes.size() - 1 - i] << (8 * (i % 8));
    }

    std::vector<uint8_t> range_max_bytes = hex_to_bytes(params["range_max"].get<std::string>());
    scalar_n end_key = {{0}};
    for (size_t i = 0; i < range_max_bytes.size() && i < 32; ++i) {
        end_key.v[i/8] |= (uint64_t)range_max_bytes[range_max_bytes.size() - 1 - i] << (8 * (i % 8));
    }

    auto hamming_range = params["hamming_weight_range"].get<std::vector<int>>();
    std::map<char, int> hex_penalties = params["hex_score_penalties"].get<std::map<char, int>>();

    size_t found_count = 0;
    while (found_count < n && scalar_cmp(&current_key, &end_key) <= 0) {
        std::string hex_str = scalar_to_hex(current_key);
        
        bool passes_all_filters = true;

        int hamming_weight = count_set_bits(current_key);
        bool hamming_ok = false;
        for (int w : hamming_range) {
            if (hamming_weight == w) { hamming_ok = true; break; }
        }
        if (!hamming_ok) passes_all_filters = false;
        
        if (passes_all_filters && has_hex_triples(hex_str)) passes_all_filters = false;
        if (passes_all_filters && has_adjacent_double_pairs(hex_str)) passes_all_filters = false;
        if (passes_all_filters && !is_hex_score_ok(hex_str, hex_penalties)) passes_all_filters = false;

        if (passes_all_filters) {
            candidates.push_back(current_key);
            found_count++;
        }

        current_key.v[0]++; 
        if (current_key.v[0] == 0) {
            current_key.v[1]++;
            if (current_key.v[1] == 0) {
                current_key.v[2]++;
                if (current_key.v[2] == 0) {
                    current_key.v[3]++;
                }
            }
        }
    }
}

std::tuple<uint8_t*, uint64_t> load_db_to_gpu(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) throw std::runtime_error("Could not open database file: " + filepath);

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    if (size % 20 != 0) throw std::runtime_error("Invalid file size. Expected a multiple of 20 bytes.");
    
    std::vector<uint8_t> host_db(size);
    if (!file.read(reinterpret_cast<char*>(host_db.data()), size)) throw std::runtime_error("Failed to read database file.");

    uint8_t* d_sorted_db;
    cudaMalloc(&d_sorted_db, size);
    cudaMemcpy(d_sorted_db, host_db.data(), size, cudaMemcpyHostToDevice);
    
    return std::make_tuple(d_sorted_db, size / 20);
}

void setup_constants() {
    std::vector<jacobian_point> lut_g(16);
    std::vector<jacobian_point> lut_lambda_g(16);

    const scalar_n G_X = {{0x79BE667EF9DCBBACULL, 0x55A06295CE870B07ULL, 0x029BFCDB2DCE28D9ULL, 0x59F2815B16F81798ULL}};
    const scalar_n G_Y = {{0x483ADA7726A3C465ULL, 0x5DA4FBFC0E1108A8ULL, 0xFD17B448A6855419ULL, 0x9C47D08FFB10D4B8ULL}};
    const scalar_n G_Z = {{1, 0, 0, 0}};
    jacobian_point G = {G_X, G_Y, G_Z};
    
    jacobian_point lambda_G;
    field_mul_montgomery(lambda_G.X, G.X, BETA);
    lambda_G.Y = G.Y;
    lambda_G.Z = G.Z;

    jacobian_point two_G, two_lambda_G;
    point_double(&two_G, &G);
    point_double(&two_lambda_G, &lambda_G);

    lut_g[0] = G;
    lut_lambda_g[0] = lambda_G;

    for (int i = 1; i < 16; ++i) {
        point_add(&lut_g[i], &lut_g[i-1], &two_G);
        point_add(&lut_lambda_g[i], &lut_lambda_g[i-1], &two_lambda_G);
    }

    cudaMemcpyToSymbol(WNAF5_LUT_G, lut_g.data(), 16 * sizeof(jacobian_point));
    cudaMemcpyToSymbol(WNAF5_LUT_LAMBDA_G, lut_lambda_g.data(), 16 * sizeof(jacobian_point));
}

int main(int argc, char** argv) {
    try {
        setup_constants();
        
        json params = load_params("RhoCore_params.json");
        std::string db_filepath = "hash160_sorted.bin";
        uint8_t* d_sorted_db;
        uint64_t db_count;
        std::tie(d_sorted_db, db_count) = load_db_to_gpu(db_filepath);
        std::cout << "Successfully loaded " << db_count << " hashes from " << db_filepath << " to GPU.\n";

        size_t num_candidates = 1024 * 1024; // Beispiel: 1 Million Kandidaten
        std::vector<scalar_n> host_candidates;
        host_candidates.reserve(num_candidates);
        
        std::cout << "Generating " << num_candidates << " filtered private key candidates...\n";
        generate_filtered_privkeys(host_candidates, params, num_candidates);
        if (host_candidates.empty()) {
            std::cout << "No candidates generated. Exiting." << std::endl;
            return 0;
        }
        num_candidates = host_candidates.size();
        std::cout << "Generated " << num_candidates << " candidates." << std::endl;
        
        scalar_n* d_scalars;
        cudaMalloc(&d_scalars, num_candidates * sizeof(scalar_n));
        cudaMemcpy(d_scalars, host_candidates.data(), num_candidates * sizeof(scalar_n), cudaMemcpyHostToDevice);

        scalar_n* d_found_key;
        int* d_found_flag;
        cudaMalloc(&d_found_key, sizeof(scalar_n));
        cudaMalloc(&d_found_flag, sizeof(int));
        cudaMemset(d_found_flag, 0, sizeof(int));

        int threads_per_block = 256;
        int num_blocks = (num_candidates + threads_per_block - 1) / threads_per_block;

        std::cout << "Launching key search kernel with " << num_blocks << " blocks and " << threads_per_block << " threads...\n";
        key_search_kernel<<<num_blocks, threads_per_block>>>(
            d_scalars, num_candidates, d_sorted_db, db_count, d_found_key, d_found_flag
        );
        
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA error after kernel launch: %s\n", cudaGetErrorString(err));
        }

        int found_flag = 0;
        cudaMemcpy(&found_flag, d_found_flag, sizeof(int), cudaMemcpyDeviceToHost);

        if (found_flag) {
            scalar_n found_key;
            cudaMemcpy(&found_key, d_found_key, sizeof(scalar_n), cudaMemcpyDeviceToHost);
            std::cout << "\n!!! MATCH FOUND !!!\n";
            std::cout << "Private Key: " << scalar_to_hex(found_key) << std::endl;
        } else {
            std::cout << "\nNo match found in this batch." << std::endl;
        }
        
        // Speicherfreigabe
        cudaFree(d_scalars);
        cudaFree(d_sorted_db);
        cudaFree(d_found_key);
        cudaFree(d_found_flag);

    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}

