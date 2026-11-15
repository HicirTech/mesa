/*
 * Copyright 2026 panvk-9300 contributors
 * SPDX-License-Identifier: MIT
 */

#include "panvk_nir.h"

#include "nir.h"
#include "nir_builder.h"

/* Residency codes for a device without sparse residency.
 *
 * shaderResourceResidency is false, so a conformant SPIR-V module never asks
 * for residency. vkd3d-proton's VKD3D_FEATURE_LEVEL override raises the D3D12
 * tiled resources tier regardless of the device (d3d12_device_caps_override),
 * and then accepts DXIL that calls CheckAccessFullyMapped and emits the SPIR-V
 * sparse image instructions for it. The reserved textures those shaders read
 * are committed allocations on this driver (vkd3d-proton's fallback when the
 * format has no sparse support), so every texel is resident: return the
 * resident code (0, the convention radv and ir3 use) instead of leaving the
 * extra component undefined, which neither back-end writes.
 *
 * Without the override nothing here matches.
 */

static nir_def *
append_resident_code(nir_builder *b, nir_def *value, unsigned value_comps)
{
   nir_def *comps[NIR_MAX_VEC_COMPONENTS];

   for (unsigned i = 0; i < value_comps; i++)
      comps[i] = nir_channel(b, value, i);
   comps[value_comps] = nir_imm_intN_t(b, 0, value->bit_size);

   return nir_vec(b, comps, value_comps + 1);
}

static bool
lower_sparse_tex(nir_builder *b, nir_tex_instr *tex)
{
   if (!tex->is_sparse)
      return false;

   b->cursor = nir_after_instr(&tex->instr);

   tex->is_sparse = false;
   tex->def.num_components--;

   nir_def *v = append_resident_code(b, &tex->def, tex->def.num_components);
   nir_def_rewrite_uses_after(&tex->def, v);
   return true;
}

static nir_intrinsic_op
non_sparse_image_load(nir_intrinsic_op op)
{
   switch (op) {
   case nir_intrinsic_image_deref_sparse_load:
      return nir_intrinsic_image_deref_load;
   case nir_intrinsic_image_sparse_load:
      return nir_intrinsic_image_load;
   case nir_intrinsic_bindless_image_sparse_load:
      return nir_intrinsic_bindless_image_load;
   case nir_intrinsic_image_heap_sparse_load:
      return nir_intrinsic_image_heap_load;
   default:
      return nir_num_intrinsics;
   }
}

static bool
lower_sparse_intrinsic(nir_builder *b, nir_intrinsic_instr *intr)
{
   switch (intr->intrinsic) {
   case nir_intrinsic_image_deref_sparse_load:
   case nir_intrinsic_image_sparse_load:
   case nir_intrinsic_bindless_image_sparse_load:
   case nir_intrinsic_image_heap_sparse_load: {
      /* Same sources and indices as the plain load, one fewer component. */
      b->cursor = nir_after_instr(&intr->instr);

      intr->intrinsic = non_sparse_image_load(intr->intrinsic);
      intr->num_components--;
      intr->def.num_components--;

      nir_def *v =
         append_resident_code(b, &intr->def, intr->def.num_components);
      nir_def_rewrite_uses_after(&intr->def, v);
      return true;
   }

   case nir_intrinsic_is_sparse_texels_resident:
      b->cursor = nir_before_instr(&intr->instr);
      nir_def_replace(&intr->def, nir_ieq_imm(b, intr->src[0].ssa, 0));
      return true;

   case nir_intrinsic_sparse_residency_code_and:
      b->cursor = nir_before_instr(&intr->instr);
      nir_def_replace(&intr->def,
                      nir_ior(b, intr->src[0].ssa, intr->src[1].ssa));
      return true;

   default:
      return false;
   }
}

static bool
lower_sparse_instr(nir_builder *b, nir_instr *instr, UNUSED void *data)
{
   switch (instr->type) {
   case nir_instr_type_tex:
      return lower_sparse_tex(b, nir_instr_as_tex(instr));
   case nir_instr_type_intrinsic:
      return lower_sparse_intrinsic(b, nir_instr_as_intrinsic(instr));
   default:
      return false;
   }
}

bool
panvk_nir_lower_sparse_residency(nir_shader *nir)
{
   return nir_shader_instructions_pass(nir, lower_sparse_instr,
                                       nir_metadata_control_flow, NULL);
}
