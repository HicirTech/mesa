/*
 * Copyright 2026 panvk-9300
 * SPDX-License-Identifier: MIT
 */

#include "compiler/nir/nir_builder.h"

extern "C" {
#include "panfrost/compiler/bifrost/valhall/valhall.h"
#include "panfrost/compiler/pan_compiler.h"

/* bifrost_compile.h, which declares it, does not build as C++. */
void bifrost_handle_unified_idvs_shader(nir_shader *nir);
}

#include <gtest/gtest.h>

#include <vector>

/* On v12+ a vertex shader is one binary: bifrost_handle_unified_idvs_shader
 * puts a copy of the shader behind each bit of shader_output and turns the
 * output tests inside each copy into constants, so the position copy keeps
 * only the position store and what feeds it. The output stores come from
 * pan_nir_lower_vs_outputs, behind their own tests at the end of the shader.
 */
class UnifiedIdvs : public testing::Test {
 protected:
   UnifiedIdvs()
   {
      glsl_type_singleton_init_or_ref();
      b = nir_builder_init_simple_shader(
         MESA_SHADER_VERTEX,
         pan_get_nir_shader_compiler_options(12, MESA_SHADER_VERTEX, false),
         "unified_idvs");
   }

   ~UnifiedIdvs()
   {
      ralloc_free(b.shader);
      glsl_type_singleton_decref();
   }

   /* An output store the way pan_nir_lower_vs_outputs builds it: the
    * position buffer is resource (61, 0), the varying buffer (61, 1). */
   void store(nir_def *data, nir_def *index, unsigned res_index)
   {
      nir_def *addr =
         nir_lea_buf_pan(&b, nir_imm_int(&b, pan_res_handle(61, res_index)),
                         index);
      nir_store_global(&b, data, nir_pack_64_2x32(&b, addr));
   }

   /* An atomic whose result reaches no output, guarded where it is the way
    * bifrost_nir_lower_vs_atomics guards it: a load of shader_output right
    * before it and an if on the position bit. Inside an if or a loop of the
    * shader, that load is in a nested block of every copy. */
   void guarded_atomic()
   {
      nir_def *so = nir_load_shader_output_pan(&b);
      nir_push_if(&b, nir_i2b(&b, nir_iand_imm(&b, so,
                                               VA_SHADER_OUTPUT_POSITION_BIT)));
      nir_intrinsic_instr *atomic =
         nir_intrinsic_instr_create(b.shader, nir_intrinsic_global_atomic);
      atomic->num_components = 1;
      atomic->src[0] = nir_src_for_ssa(nir_imm_int64(&b, 0x1000));
      atomic->src[1] = nir_src_for_ssa(nir_imm_int(&b, 1));
      nir_intrinsic_set_atomic_op(atomic, nir_atomic_op_iadd);
      nir_def_init(&atomic->instr, &atomic->def, 1, 32);
      nir_builder_instr_insert(&b, &atomic->instr);
      nir_pop_if(&b, NULL);
   }

   /* A position and one varying from the vertex ID. With control_flow, the
    * varying goes through an if/else first, as in any shader with a branch
    * before its outputs are written. */
   void build(bool control_flow)
   {
      nir_def *x = nir_u2f32(&b, nir_load_raw_vertex_id(&b));
      nir_def *pos = nir_vec4(&b, x, x, x, nir_imm_float(&b, 1.0f));
      nir_def *v = nir_fmul_imm(&b, x, 2.0f);

      if (control_flow) {
         nir_def *c = nir_ilt_imm(&b, nir_load_instance_id(&b), 4);
         nir_push_if(&b, c);
         nir_def *then_v = nir_fadd_imm(&b, v, 1.0f);
         nir_push_else(&b, NULL);
         nir_def *else_v = nir_fadd_imm(&b, v, 3.0f);
         nir_pop_if(&b, NULL);
         v = nir_if_phi(&b, then_v, else_v);
      }

      nir_def *vary = nir_vec4(&b, v, v, v, v);

      nir_def *so = nir_load_shader_output_pan(&b);
      nir_def *index = nir_load_idvs_output_buf_index_pan(&b);

      nir_def *is_pos =
         nir_i2b(&b, nir_iand_imm(&b, so, VA_SHADER_OUTPUT_POSITION_BIT));
      nir_push_if(&b, is_pos);
      store(pos, index, 0);
      nir_pop_if(&b, NULL);

      nir_def *is_vary =
         nir_i2b(&b, nir_iand_imm(&b, so, VA_SHADER_OUTPUT_VARY_BIT));
      nir_push_if(&b, is_vary);
      store(vary, index, 1);
      nir_pop_if(&b, NULL);
   }

   /* Resource index (0 position, 1 varying) a store's address comes from. */
   static int store_resource(nir_def *def, unsigned depth = 0)
   {
      if (depth > 8)
         return -1;
      nir_instr *instr = nir_def_instr(def);
      if (instr->type == nir_instr_type_intrinsic) {
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_lea_buf_pan &&
             nir_src_is_const(intr->src[0]))
            return pan_res_handle_get_index(nir_src_as_uint(intr->src[0]));
         return -1;
      }
      if (instr->type == nir_instr_type_alu) {
         nir_alu_instr *alu = nir_instr_as_alu(instr);
         for (unsigned s = 0; s < nir_op_infos[alu->op].num_inputs; s++) {
            int r = store_resource(alu->src[s].src.ssa, depth + 1);
            if (r >= 0)
               return r;
         }
      }
      return -1;
   }

   /* For every top-level if of the shader, in order, the resource indices
    * of the output stores inside it. */
   std::vector<std::vector<int>> stores_per_top_level_if()
   {
      std::vector<std::vector<int>> out;
      nir_function_impl *impl = nir_shader_get_entrypoint(b.shader);
      foreach_list_typed(nir_cf_node, node, node, &impl->body) {
         if (node->type != nir_cf_node_if)
            continue;
         std::vector<int> stores;
         nir_foreach_block_in_cf_node(block, node) {
            nir_foreach_instr(instr, block) {
               if (instr->type != nir_instr_type_intrinsic)
                  continue;
               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               if (intr->intrinsic == nir_intrinsic_store_global)
                  stores.push_back(store_resource(intr->src[1].ssa));
            }
         }
         out.push_back(stores);
      }
      return out;
   }

   /* For every top-level if of the shader, in order, the number of atomics
    * inside it. */
   std::vector<int> atomics_per_top_level_if()
   {
      std::vector<int> out;
      nir_function_impl *impl = nir_shader_get_entrypoint(b.shader);
      foreach_list_typed(nir_cf_node, node, node, &impl->body) {
         if (node->type != nir_cf_node_if)
            continue;
         int atomics = 0;
         nir_foreach_block_in_cf_node(block, node) {
            nir_foreach_instr(instr, block) {
               if (instr->type == nir_instr_type_intrinsic &&
                   nir_instr_as_intrinsic(instr)->intrinsic ==
                      nir_intrinsic_global_atomic)
                  atomics++;
            }
         }
         out.push_back(atomics);
      }
      return out;
   }

   /* Non-constant instructions outside the copies: the shader_output load
    * and, for each remaining copy, the iand and ine of its test. Values the
    * copies share stay inside each copy: hoisting them in front of the copies
    * would keep them live through the position copy.
    */
   unsigned instrs_outside_copies()
   {
      unsigned n = 0;
      nir_function_impl *impl = nir_shader_get_entrypoint(b.shader);
      foreach_list_typed(nir_cf_node, node, node, &impl->body) {
         if (node->type != nir_cf_node_block)
            continue;
         nir_foreach_instr(instr, nir_cf_node_as_block(node)) {
            if (instr->type != nir_instr_type_load_const)
               n++;
         }
      }
      return n;
   }

   /* Whether the first top-level if (the position copy) holds another if. */
   bool position_copy_branches()
   {
      nir_function_impl *impl = nir_shader_get_entrypoint(b.shader);
      foreach_list_typed(nir_cf_node, node, node, &impl->body) {
         if (node->type != nir_cf_node_if)
            continue;
         foreach_list_typed(nir_cf_node, inner, node,
                            &nir_cf_node_as_if(node)->then_list) {
            if (inner->type != nir_cf_node_block)
               return true;
         }
         return false;
      }
      return false;
   }

   nir_builder b;
};

/* Without control flow in the shader each copy keeps its own store. */
TEST_F(UnifiedIdvs, StraightLineShaderIsSpecialized)
{
   build(false);
   bifrost_handle_unified_idvs_shader(b.shader);

   const std::vector<std::vector<int>> expected = {{0}, {1}};
   EXPECT_EQ(stores_per_top_level_if(), expected);
   EXPECT_EQ(instrs_outside_copies(), 5u);
}

/* A branch before the outputs must not keep the varying store (and the
 * branch feeding it) in the position copy, nor leave an attribute copy. */
TEST_F(UnifiedIdvs, ShaderWithBranchIsSpecialized)
{
   build(true);
   bifrost_handle_unified_idvs_shader(b.shader);

   const std::vector<std::vector<int>> expected = {{0}, {1}};
   EXPECT_EQ(stores_per_top_level_if(), expected);
   EXPECT_EQ(instrs_outside_copies(), 5u);
   EXPECT_FALSE(position_copy_branches());
}

/* Every block of a copy is specialised, including nested ones: here the
 * outputs and their tests sit inside an if of the shader itself. */
TEST_F(UnifiedIdvs, NestedOutputTestIsSpecialized)
{
   nir_def *x = nir_u2f32(&b, nir_load_raw_vertex_id(&b));
   nir_def *pos = nir_vec4(&b, x, x, x, nir_imm_float(&b, 1.0f));
   nir_def *vary = nir_vec4(&b, x, x, x, x);

   nir_push_if(&b, nir_ilt_imm(&b, nir_load_instance_id(&b), 4));
   {
      nir_def *so = nir_load_shader_output_pan(&b);
      nir_def *index = nir_load_idvs_output_buf_index_pan(&b);

      nir_def *is_pos =
         nir_i2b(&b, nir_iand_imm(&b, so, VA_SHADER_OUTPUT_POSITION_BIT));
      nir_push_if(&b, is_pos);
      store(pos, index, 0);
      nir_pop_if(&b, NULL);

      nir_def *is_vary =
         nir_i2b(&b, nir_iand_imm(&b, so, VA_SHADER_OUTPUT_VARY_BIT));
      nir_push_if(&b, is_vary);
      store(vary, index, 1);
      nir_pop_if(&b, NULL);
   }
   nir_pop_if(&b, NULL);

   bifrost_handle_unified_idvs_shader(b.shader);

   const std::vector<std::vector<int>> expected = {{0}, {1}};
   EXPECT_EQ(stores_per_top_level_if(), expected);
}

/* The guard bifrost_nir_lower_vs_atomics puts on an atomic in an if or a
 * loop of the shader is specialized too: left with the run-time test, the
 * varying copy would keep the atomic and the attribute copy would survive
 * with it, running it again in an invocation with both the position and the
 * attribute bit. */
TEST_F(UnifiedIdvs, AtomicGuardInIfIsSpecialized)
{
   nir_push_if(&b, nir_ilt_imm(&b, nir_load_instance_id(&b), 4));
   guarded_atomic();
   nir_pop_if(&b, NULL);
   build(false);
   bifrost_handle_unified_idvs_shader(b.shader);

   const std::vector<std::vector<int>> stores = {{0}, {1}};
   EXPECT_EQ(stores_per_top_level_if(), stores);
   const std::vector<int> atomics = {1, 0};
   EXPECT_EQ(atomics_per_top_level_if(), atomics);
}

TEST_F(UnifiedIdvs, AtomicGuardInLoopIsSpecialized)
{
   nir_variable *i = nir_local_variable_create(b.impl, glsl_uint_type(), "i");
   nir_store_var(&b, i, nir_imm_int(&b, 0), 1);
   nir_loop *loop = nir_push_loop(&b);
   nir_def *iv = nir_load_var(&b, i);
   nir_break_if(&b, nir_uge(&b, iv, nir_load_instance_id(&b)));
   guarded_atomic();
   nir_store_var(&b, i, nir_iadd_imm(&b, iv, 1), 1);
   nir_pop_loop(&b, loop);
   build(false);
   NIR_PASS(_, b.shader, nir_lower_vars_to_ssa);
   bifrost_handle_unified_idvs_shader(b.shader);

   const std::vector<std::vector<int>> stores = {{0}, {1}};
   EXPECT_EQ(stores_per_top_level_if(), stores);
   const std::vector<int> atomics = {1, 0};
   EXPECT_EQ(atomics_per_top_level_if(), atomics);
}
