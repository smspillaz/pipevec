/*
 * /pipevec/pipevec-tensor.h
 *
 * Forward declarations for Pipevec Tensor.
 *
 * Copyright (C) 2019 Sam Spilsbury.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#pragma once

#include <glib-object.h>
#include <stdint.h>

G_BEGIN_DECLS

#define PIPEVEC_TYPE_TENSOR pipevec_tensor_get_type ()
G_DECLARE_FINAL_TYPE (PipevecTensor, pipevec_tensor, PIPEVEC, TENSOR, GObject)

gboolean pipevec_tensor_set_data (PipevecTensor  *tensor,
                                  GArray         *contents,
                                  GArray         *shape,
                                  GError        **error);

GArray * pipevec_tensor_get_data (PipevecTensor *tensor);

PipevecTensor * pipevec_tensor_copy (PipevecTensor  *tensor,
                                     GError        **error);

gboolean pipevec_tensor_reshape (PipevecTensor  *tensor,
                                 GArray         *shape,
                                 GError        **error);

typedef float (*PipevecTensorMapFunction) (float     element,
                                           GArray   *indices,
                                           gpointer  user_data);

PipevecTensor * pipevec_tensor_map (PipevecTensor             *src,
                                    PipevecTensorMapFunction   func,
                                    gpointer                  *user_data,
                                    GError                   **error);

PipevecTensor * pipevec_tensor_add_tensor (PipevecTensor  *lhs,
                                           PipevecTensor  *rhs,
                                           GError        **error);

PipevecTensor * pipevec_tensor_add_scalar (PipevecTensor  *lhs,
                                           float           rhs,
                                           GError        **error);

PipevecTensor * pipevec_tensor_sub_tensor (PipevecTensor  *lhs,
                                           PipevecTensor  *rhs,
                                           GError        **error);

PipevecTensor * pipevec_tensor_sub_scalar (PipevecTensor  *lhs,
                                           float           rhs,
                                           GError        **error);

PipevecTensor * pipevec_tensor_inner_product_tensor (PipevecTensor  *lhs,
                                                     PipevecTensor  *rhs,
                                                     GError        **error);

PipevecTensor * pipevec_tensor_multiply_tensor (PipevecTensor  *lhs,
                                                PipevecTensor  *rhs,
                                                GError        **error);

PipevecTensor * pipevec_tensor_multiply_scalar (PipevecTensor  *lhs,
                                                float           rhs,
                                                GError        **error);

PipevecTensor * pipevec_tensor_divide_tensor (PipevecTensor  *lhs,
                                              PipevecTensor  *rhs,
                                              GError        **error);

PipevecTensor * pipevec_tensor_divide_scalar (PipevecTensor  *lhs,
                                              float           rhs,
                                              GError        **error);


PipevecTensor * pipevec_tensor_new (GArray  *shape,
                                    GArray  *contents,
                                    GError **error);


G_END_DECLS
