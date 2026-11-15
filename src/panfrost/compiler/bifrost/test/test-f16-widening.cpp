/*
 * Copyright 2026 panvk-9300 contributors
 * SPDX-License-Identifier: MIT
 */

#include "bi_builder.h"
#include "bi_test.h"
#include "compiler.h"
#include "va_compiler.h"
#include "valhall.h"

#include <gtest/gtest.h>

/* Valhall widens a half on the way into many 32-bit float sources, so
 * bi_opt_mod_prop_forward folds F16_TO_F32 into its users. A widened half
 * must stay in sources whose final encoding has the widen, including after
 * FABSNEG.f32 composition and the Valhall lowering that follows.
 */

static void
bi_optimizer(bi_context *ctx)
{
   bi_opt_mod_prop_forward(ctx);
   bi_opt_mod_prop_backward(ctx);
   bi_opt_dce(ctx, true);
}

#define CASE(instr, expected)                                                  \
   INSTRUCTION_CASE(                                                           \
      {                                                                        \
         UNUSED bi_index reg = bi_temp(b->shader);                             \
         instr;                                                                \
         bi_kaboom(b, reg);                                                    \
      },                                                                       \
      {                                                                        \
         UNUSED bi_index reg = bi_temp(b->shader);                             \
         expected;                                                             \
         bi_kaboom(b, reg);                                                    \
      },                                                                       \
      bi_optimizer);

#define NEGCASE(instr) CASE(instr, instr)

/* The passes between mod prop and packing that change opcodes. */
static void
va_lower(bi_context *ctx)
{
   bi_opt_mod_prop_forward(ctx);
   bi_opt_mod_prop_backward(ctx);
   bi_opt_dce(ctx, false);
   bi_lower_opt_instructions(ctx);
   va_lower_isel(ctx);
   bi_foreach_instr_global(ctx, I)
      va_fuse_add_imm(I);
}

/* Half-word swizzles on 32-bit sources that the final opcode cannot encode
 * (va_pack would reject them, and FROUND.f32 has no widen from v11 on). */
static unsigned
unencodable_widens(bi_context *ctx)
{
   unsigned bad = 0;
   bi_foreach_instr_global(ctx, I) {
      const struct va_opcode_info *info = &valhall_opcodes[I->op];
      if (!info->exact)
         continue;
      bi_foreach_src(I, s) {
         if (s >= info->nr_srcs)
            break;
         enum bi_swizzle swz = I->src[s].swizzle;
         if (swz != BI_SWIZZLE_H0 && swz != BI_SWIZZLE_H1)
            continue;
         struct va_src_info src = va_src_info(I->op, s);
         if (src.size != VA_SIZE_32)
            continue;
         if (!(src.swizzle || src.combine) ||
             (I->op == BI_OPCODE_FROUND_F32 && ctx->arch >= 11)) {
            bi_print_instr(I, stderr);
            bad++;
         }
      }
   }
   return bad;
}

class F16Widening : public testing::Test {
 protected:
   F16Widening()
   {
      mem_ctx = ralloc_context(NULL);
      arch = 12;
      x = bi_register(1);
      y = bi_register(2);
      z = bi_register(3);
   }

   ~F16Widening()
   {
      ralloc_free(mem_ctx);
   }

   void *mem_ctx;
   unsigned arch;
   bi_index x, y, z;
};

/* log2(abs(float(h))) at 32 bits: bi_flog2_32 reads |s0| in FREXPE.f32,
 * FADD_LSCALE.f32 and FLOGD.f32; FADD_LSCALE.f32 cannot widen. */
TEST_F(F16Widening, Log2OfAbsStaysEncodable)
{
   for (unsigned a = 9; a <= 14; ++a) {
      bi_builder *b = bit_builder(mem_ctx, a);
      bi_index w = bi_f16_to_f32(b, bi_half(x, false));
      bi_index s0 = bi_fabsneg_f32(b, bi_abs(w));
      bi_index e = bi_frexpe_f32(b, s0, true, false);
      bi_index ei = bi_s32_to_f32(b, e);
      bi_index add = bi_fadd_lscale_f32(b, bi_imm_f32(-1.0f), s0);
      bi_index d = bi_fma_f32(b, bi_flogd_f32(b, s0), add, ei);
      bi_kaboom(b, d);
      va_lower(b->shader);
      EXPECT_EQ(unencodable_widens(b->shader), 0u) << "arch " << a;
   }
}

/* exp2(-float(h)) at 32 bits: bi_fexp_32 reads -s0 in FMA_RSCALE.f32. */
TEST_F(F16Widening, Exp2OfNegStaysEncodable)
{
   for (unsigned a = 9; a <= 14; ++a) {
      bi_builder *b = bit_builder(mem_ctx, a);
      bi_index w = bi_f16_to_f32(b, bi_half(x, true));
      bi_index s0 = bi_fabsneg_f32(b, bi_neg(w));
      bi_index scale = bi_fma_rscale_f32(b, s0, bi_imm_f32(1.0f), bi_negzero(),
                                         bi_imm_u32(24), BI_SPECIAL_NONE);
      bi_index fixed = bi_f32_to_s32(b, scale);
      bi_index d = bi_fexp_f32(b, fixed, scale);
      bi_kaboom(b, d);
      va_lower(b->shader);
      EXPECT_EQ(unencodable_widens(b->shader), 0u) << "arch " << a;
   }
}

/* trunc(abs(float(h))) at 32 bits: FROUND.f32 has no widen on v11+. */
TEST_F(F16Widening, TruncOfAbsStaysEncodable)
{
   for (unsigned a = 9; a <= 14; ++a) {
      bi_builder *b = bit_builder(mem_ctx, a);
      bi_index w = bi_f16_to_f32(b, bi_half(x, false));
      bi_index d = bi_fround_f32(b, bi_fabsneg_f32(b, bi_abs(w)), BI_ROUND_RTZ);
      bi_kaboom(b, d);
      va_lower(b->shader);
      EXPECT_EQ(unencodable_widens(b->shader), 0u) << "arch " << a;
   }
}

/* A cube coordinate from -float(h): CUBEFACE1 takes .neg but no widen. */
TEST_F(F16Widening, CubefaceOfNegStaysEncodable)
{
   for (unsigned a = 9; a <= 14; ++a) {
      bi_builder *b = bit_builder(mem_ctx, a);
      bi_index w = bi_f16_to_f32(b, bi_half(x, false));
      bi_index d = bi_cubeface1(b, bi_fabsneg_f32(b, bi_neg(w)), y, z);
      bi_kaboom(b, d);
      va_lower(b->shader);
      EXPECT_EQ(unencodable_widens(b->shader), 0u) << "arch " << a;
   }
}

/* Through FABSNEG.f32, the user's own modifiers compose with the fold. */
TEST_F(F16Widening, AbsNegCompose)
{
   for (arch = 9; arch <= 14; ++arch) {
      CASE(bi_fadd_f32_to(b, reg,
                          bi_fabsneg_f32(b, bi_neg(bi_f16_to_f32(
                                               b, bi_half(x, false)))),
                          y),
           bi_fadd_f32_to(b, reg, bi_neg(bi_half(x, false)), y));

      CASE(bi_fma_f32_to(
              b, reg, y, z,
              bi_fabsneg_f32(b, bi_abs(bi_f16_to_f32(b, bi_half(x, true))))),
           bi_fma_f32_to(b, reg, y, z, bi_abs(bi_half(x, true))));

      /* -|float(h)| */
      CASE(bi_fadd_f32_to(b, reg,
                          bi_fabsneg_f32(b, bi_neg(bi_fabsneg_f32(
                                               b, bi_abs(bi_f16_to_f32(
                                                     b, bi_half(x, false)))))),
                          y),
           bi_fadd_f32_to(b, reg, bi_neg(bi_abs(bi_half(x, false))), y));
   }
}

/* Modifiers on the conversion's own source compose too. */
TEST_F(F16Widening, ConversionModifiersCompose)
{
   /* -float(-h) = float(h) */
   CASE(bi_fadd_f32_to(b, reg,
                       bi_fabsneg_f32(b, bi_neg(bi_f16_to_f32(
                                            b, bi_neg(bi_half(x, false))))),
                       y),
        bi_fadd_f32_to(b, reg, bi_half(x, false), y));

   /* |float(-h)| = |h| */
   CASE(bi_fadd_f32_to(b, reg,
                       bi_fabsneg_f32(b, bi_abs(bi_f16_to_f32(
                                            b, bi_neg(bi_half(x, false))))),
                       y),
        bi_fadd_f32_to(b, reg, bi_abs(bi_half(x, false)), y));
}

/* A user that reads half of the f32 result does not take the fold. */
TEST_F(F16Widening, SwizzledUseStays)
{
   NEGCASE({
      bi_index w = bi_f16_to_f32(b, bi_half(x, false));
      bi_fadd_f32_to(b, reg, bi_half(w, true), y);
   });
}

/* FCMP.f32 folds: it is encoded as FCMP_OR.f32, which widens. */
TEST_F(F16Widening, Fcmp)
{
   CASE(bi_fcmp_f32_to(b, reg, bi_f16_to_f32(b, bi_half(x, true)), y,
                       BI_CMPF_LT, BI_RESULT_TYPE_M1),
        bi_fcmp_f32_to(b, reg, bi_half(x, true), y, BI_CMPF_LT,
                       BI_RESULT_TYPE_M1));
}

/* A constant half is not folded: immediate folding would pack its raw bits. */
TEST_F(F16Widening, ConstantStays)
{
   NEGCASE(bi_fadd_f32_to(b, reg,
                          bi_f16_to_f32(b, bi_half(bi_imm_u32(0x3c00), false)),
                          y));
}

/* Bifrost has no source widening: F16_TO_F32 stays. */
TEST_F(F16Widening, BifrostUntouched)
{
   arch = 7;
   NEGCASE(bi_fadd_f32_to(b, reg, bi_f16_to_f32(b, bi_half(x, false)), y));
   NEGCASE(bi_fround_f32_to(b, reg, bi_f16_to_f32(b, bi_half(x, true)),
                            BI_ROUND_RTN));
   arch = 6;
   NEGCASE(bi_fma_f32_to(b, reg, y, bi_f16_to_f32(b, bi_half(x, true)), y));
}

static void
forward_dce_backward(bi_context *ctx)
{
   bi_opt_mod_prop_forward(ctx);
   bi_opt_dce(ctx, true);
   bi_opt_mod_prop_backward(ctx);
   bi_opt_dce(ctx, true);
}

/* clamp(unpackHalf2x16(bits(a + b)).y): the clamp reads only the high half of
 * the sum widened, so it must not move into the f32 add. */
TEST_F(F16Widening, ClampOfWidenedHalfStays)
{
   INSTRUCTION_CASE(
      {
         UNUSED bi_index reg = bi_temp(b->shader);
         bi_index t = bi_fadd_f32(b, x, y);
         bi_instr *c =
            bi_fclamp_f32_to(b, reg, bi_f16_to_f32(b, bi_half(t, true)));
         c->clamp = BI_CLAMP_CLAMP_0_1;
         bi_kaboom(b, reg);
      },
      {
         UNUSED bi_index reg = bi_temp(b->shader);
         bi_index t = bi_fadd_f32(b, x, y);
         UNUSED bi_index cvt = bi_temp(b->shader);
         bi_instr *c = bi_fclamp_f32_to(b, reg, bi_half(t, true));
         c->clamp = BI_CLAMP_CLAMP_0_1;
         bi_kaboom(b, reg);
      },
      forward_dce_backward);
}
