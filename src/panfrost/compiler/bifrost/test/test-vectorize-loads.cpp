/*
 * Copyright 2026 panvk-9300
 * SPDX-License-Identifier: MIT
 */

#include "compiler/nir/nir_builder.h"
#include "util/bitset.h"
#include "util/format/u_formats.h"
#include "util/shader_stats.h"
#include "util/u_dynarray.h"

extern "C" {
#include "panfrost/compiler/pan_compiler.h"
}

#include <gtest/gtest.h>

#include <algorithm>
#include <utility>
#include <vector>

/* Buffer loads reach bifrost_postprocess_nir with an index def of their own
 * each: PanVK's descriptor lowering (valhall_pack_buf_idx) packs the index
 * again for every access. The load/store vectorizer only merges accesses
 * whose index is the same SSA value.
 */
class VectorizeBufferLoads : public testing::Test {
 protected:
   VectorizeBufferLoads()
   {
      glsl_type_singleton_init_or_ref();
      b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE,
         pan_get_nir_shader_compiler_options(12, MESA_SHADER_COMPUTE, false),
         "vectorize_buffer_loads");
   }

   ~VectorizeBufferLoads()
   {
      ralloc_free(b.shader);
      glsl_type_singleton_decref();
   }

   /* Four consecutive words from a 16-byte aligned base, word i from the UBO
    * index[i], every load with an index def of its own.
    */
   void load_words(const uint32_t index[4])
   {
      nir_def *base = nir_ishl_imm(&b, nir_load_local_invocation_index(&b), 4);
      nir_def *sum = nir_imm_int(&b, 0);

      for (unsigned i = 0; i < 4; i++) {
         nir_def *word = nir_load_ubo(&b, 1, 32, nir_imm_int(&b, index[i]),
                                      nir_iadd_imm(&b, base, 4 * i));
         nir_intrinsic_instr *load = nir_def_as_intrinsic(word);
         nir_intrinsic_set_align(load, 16, 4 * i);
         nir_intrinsic_set_range(load, ~0);
         sum = nir_iadd(&b, sum, word);
      }

      nir_store_global(&b, sum, nir_undef(&b, 1, 64));
   }

   /* Four words at base + offsets[i] of one UBO, with the address built the
    * way PanVK leaves it for a compute shader: a vec3 (index, array index,
    * offset) per access from the descriptor lowering and
    * nir_lower_explicit_io (nir_build_addr_iadd_imm), and an index packed
    * per access by valhall_pack_buf_idx. A constant offset is then still
    * iadd(0, k) when bifrost_postprocess_nir starts.
    */
   void load_words_explicit_io(const unsigned offsets[4], bool dynamic_base)
   {
      nir_def *addr = nir_vec3(&b, nir_imm_int(&b, 0x12), nir_imm_int(&b, 0),
                               nir_imm_int(&b, 0));
      nir_def *base =
         dynamic_base
            ? nir_ishl_imm(&b, nir_load_local_invocation_index(&b), 6)
            : NULL;
      nir_def *sum = nir_imm_int(&b, 0);

      for (unsigned i = 0; i < 4; i++) {
         nir_def *a = addr;
         if (base)
            a = nir_vector_insert_imm(
               &b, a, nir_iadd(&b, nir_channel(&b, a, 2), base), 2);
         if (offsets[i])
            a = nir_vector_insert_imm(
               &b, a,
               nir_iadd(&b, nir_channel(&b, a, 2), nir_imm_int(&b, offsets[i])),
               2);

         nir_def *index = nir_trim_vector(&b, a, 2);
         nir_def *packed = nir_iadd(&b, nir_channel(&b, index, 0),
                                    nir_channel(&b, index, 1));
         nir_def *word = nir_load_ubo(&b, 1, 32, packed, nir_channel(&b, a, 2));
         nir_intrinsic_instr *load = nir_def_as_intrinsic(word);
         nir_intrinsic_set_align(load, 4, 0);
         nir_intrinsic_set_range(load, ~0);
         sum = nir_iadd(&b, sum, word);
      }

      nir_store_global(&b, sum, nir_undef(&b, 1, 64));
   }

   /* Component counts of every load_ubo, sorted. */
   std::vector<unsigned> ubo_load_sizes()
   {
      std::vector<unsigned> sizes;
      nir_foreach_block(block, nir_shader_get_entrypoint(b.shader)) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_load_ubo)
               sizes.push_back(intr->num_components);
         }
      }
      std::sort(sizes.begin(), sizes.end());
      return sizes;
   }

   void postprocess(nir_variable_mode robust_modes = (nir_variable_mode)0)
   {
      struct pan_compile_inputs inputs = {};
      inputs.arch = 12;
      inputs.robust_modes = robust_modes;
      struct pan_shader_info info = {};
      pan_postprocess_nir(b.shader, &inputs, &info);
   }

   /* (index, components) of every load_ubo, sorted. */
   std::vector<std::pair<uint32_t, unsigned>> ubo_loads()
   {
      std::vector<std::pair<uint32_t, unsigned>> loads;
      nir_foreach_block(block, nir_shader_get_entrypoint(b.shader)) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_load_ubo)
               loads.push_back({(uint32_t)nir_src_as_uint(intr->src[0]),
                                intr->num_components});
         }
      }
      std::sort(loads.begin(), loads.end());
      return loads;
   }

   nir_builder b;
};

TEST_F(VectorizeBufferLoads, MergesWordsOfOneBuffer)
{
   const uint32_t index[4] = {0x12, 0x12, 0x12, 0x12};
   load_words(index);
   postprocess();

   const std::vector<std::pair<uint32_t, unsigned>> expected = {{0x12, 4}};
   EXPECT_EQ(ubo_loads(), expected);
}

TEST_F(VectorizeBufferLoads, KeepsBuffersApart)
{
   const uint32_t index[4] = {0x12, 0x12, 0x13, 0x13};
   load_words(index);
   postprocess();

   const std::vector<std::pair<uint32_t, unsigned>> expected = {{0x12, 2},
                                                                {0x13, 2}};
   EXPECT_EQ(ubo_loads(), expected);
}

/* Constant offsets that nothing folded: the vectorizer would read each k as a
 * term of its own. */
TEST_F(VectorizeBufferLoads, MergesConstantOffsetsOfComputeShaders)
{
   const unsigned offsets[4] = {0x0, 0x4, 0x8, 0xc};
   load_words_explicit_io(offsets, false);
   postprocess();

   EXPECT_EQ(ubo_load_sizes(), std::vector<unsigned>({4}));
}

/* A dynamic base (an array element) plus constant offsets. */
TEST_F(VectorizeBufferLoads, MergesDynamicBasePlusConstant)
{
   const unsigned offsets[4] = {0x0, 0x4, 0x8, 0xc};
   load_words_explicit_io(offsets, true);
   postprocess();

   EXPECT_EQ(ubo_load_sizes(), std::vector<unsigned>({4}));
}

/* With robust UBO access a merged load stays inside one 16-byte block: words
 * 0x2c..0x38 become one word and one vec3, not a vec4 across 0x30. */
TEST_F(VectorizeBufferLoads, RobustUboMergeStaysInOneBlock)
{
   const unsigned offsets[4] = {0x2c, 0x30, 0x34, 0x38};
   load_words_explicit_io(offsets, true);
   postprocess(nir_var_mem_ubo);

   EXPECT_EQ(ubo_load_sizes(), std::vector<unsigned>({1, 3}));
}

/* Without robustness the same words may cross the block. */
TEST_F(VectorizeBufferLoads, NonRobustUboMergeMayCross)
{
   const unsigned offsets[4] = {0x2c, 0x30, 0x34, 0x38};
   load_words_explicit_io(offsets, true);
   postprocess();

   EXPECT_EQ(ubo_load_sizes(), std::vector<unsigned>({4}));
}
