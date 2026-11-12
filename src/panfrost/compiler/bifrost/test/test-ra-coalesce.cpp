/*
 * Copyright 2026 panvk-9300 contributors
 * SPDX-License-Identifier: MIT
 */

#include "bi_builder.h"
#include "bi_test.h"
#include "compiler.h"

#include <gtest/gtest.h>

class RegisterAllocation : public testing::Test {
 protected:
   RegisterAllocation()
   {
      mem_ctx = ralloc_context(NULL);
   }

   ~RegisterAllocation()
   {
      ralloc_free(mem_ctx);
   }

   void *mem_ctx;
};

static uint32_t
colour_word(unsigned target, unsigned component)
{
   return 0x100 * (target + 1) + component;
}

/* RA precolours the colour of every BLEND to R0-R3. A shader that computes the
 * colours of two render targets before it blends either, as a G-buffer pass
 * does, must still hand each BLEND its own colour.
 */
TEST_F(RegisterAllocation, TwoBlendTargetsKeepTheirColours)
{
   bi_builder *b = bit_builder(mem_ctx, 10);
   bi_context *ctx = b->shader;
   ctx->stage = MESA_SHADER_FRAGMENT;

   bi_index colour[2][4];
   for (unsigned rt = 0; rt < 2; ++rt) {
      for (unsigned c = 0; c < 4; ++c)
         colour[rt][c] = bi_mov_i32(b, bi_imm_u32(colour_word(rt, c)));
   }

   for (unsigned rt = 0; rt < 2; ++rt) {
      bi_instr *vec = bi_collect_i32_to(b, bi_temp(ctx), 4);
      bi_foreach_src(vec, c)
         vec->src[c] = colour[rt][c];

      bi_instr *blend = bi_blend_to(
         b, bi_temp(ctx), vec->dest[0], bi_register(60),
         bi_fau(BIR_FAU_BLEND_0, false), bi_fau(BIR_FAU_BLEND_0, true),
         bi_null(), BI_REGISTER_FORMAT_F32, 4, 4);
      blend->blend_target = rt;
   }

   bi_register_allocate(ctx);

   /* Run the allocated program: it is moves and blends only. */
   uint32_t regs[64] = {0};
   unsigned rt = 0;
   bi_foreach_instr_global(ctx, I) {
      if (I->op == BI_OPCODE_MOV_I32) {
         ASSERT_EQ(I->dest[0].type, BI_INDEX_REGISTER);
         if (I->src[0].type == BI_INDEX_CONSTANT) {
            regs[I->dest[0].value] = I->src[0].value;
         } else {
            ASSERT_EQ(I->src[0].type, BI_INDEX_REGISTER);
            regs[I->dest[0].value] = regs[I->src[0].value];
         }
         continue;
      }

      ASSERT_EQ(I->op, BI_OPCODE_BLEND) << bi_opcode_props[I->op].name;
      ASSERT_EQ(I->src[0].type, BI_INDEX_REGISTER);
      ASSERT_LT(rt, 2u);
      for (unsigned c = 0; c < 4; ++c) {
         EXPECT_EQ(regs[I->src[0].value + c], colour_word(rt, c))
            << "render target " << rt << ", component " << c;
      }
      ++rt;
   }

   EXPECT_EQ(rt, 2u);
}
