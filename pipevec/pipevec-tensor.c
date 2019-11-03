/*
 * /pipevec/pipevec-tensor.c
 *
 * Implementation for Pipevec Tensor.
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

#include <pipevec/pipevec-tensor.h>
#include <pipevec/pipevec-errors.h>

#include <glib-object.h>

typedef float float8_t __attribute__((vector_size(8 * (sizeof (float)))));

struct _PipevecTensor
{
  GObject parent_instance;
};

typedef struct _PipevecTensorPrivate {
  /* @array is aligned and the allocation is entirely managed
   * by ourselves. The length is implicit in the form of @shape */
  float  *array;
  GArray *shape;
  GArray *padded_shape;
} PipevecTensorPrivate;

G_DEFINE_TYPE_WITH_PRIVATE (PipevecTensor, pipevec_tensor, G_TYPE_OBJECT);

static size_t
array_size_t_product (size_t *data, size_t len)
{
  size_t product = 1;

  for (size_t i = 0; i < len; ++i)
    {
      product *= data[i];
    }

  return product;
}

static char *
format_size_t_array (size_t *data, size_t len) {
  g_auto(GStrv) strv = g_new0 (char *, len + 3);
  strv[0] = g_strdup("[");

  for (size_t i = 0; i < len; ++i)
    {
      strv[i + 1] = g_strdup_printf(i != (len - 1) ? "%zu, " : "%zu", i);
    }

  strv[len - 2] = g_strdup("]");
  strv[len - 1] = NULL;

  char *formatted = g_strjoinv ("", strv);

  return formatted;
}

static inline size_t
apply_padding (size_t original, size_t vector_size)
{
  return original + ((vector_size - (original % vector_size)) % vector_size);
}

/**
 * pipevec_tensor_alloc_shape:
 * @tensor: A #PipevecTensor
 * @shape: A #GArray describing the shape of the tensor.
 *
 * Allocate the internal storage of the @tensor to conform to @shape,
 * and pad appropriately
 *
 * Returns: %TRUE if the operation succeeded, %FALSE with error on failure.
 */
static gboolean
pipevec_tensor_alloc_shape (PipevecTensor  *tensor,
                            GArray         *shape,
                            GError       **error)
{
  PipevecTensorPrivate *priv = pipevec_tensor_get_instance_private (tensor);

  size_t *shape_data = (size_t *) shape->data;
  size_t shape_product = array_size_t_product (shape_data, shape->len);

  size_t padded_shape = (
    (shape_product / shape_data[shape->len - 1]) *
    apply_padding (shape_data[shape->len - 1], 8)
  );

  float *array = NULL;
  int align_error = posix_memalign ((void **) &array,
                                    sizeof(float8_t),
                                    sizeof(float8_t) * padded_shape);
  if (align_error != 0)
    {
      g_set_error (error,
                   PIPEVEC_ERROR,
                   PIPEVEC_ERROR_INTERNAL,
                   "Unable to allocate memory: %s",
                   strerror(align_error));
      return FALSE;
    }

  /* Clear all existing data and copy in the new data, after allocating
   * new aligned memory for it */
  g_clear_pointer (&priv->shape, g_array_unref);
  g_clear_pointer (&priv->padded_shape, g_array_unref);
  g_clear_pointer (&priv->array, g_free);

  priv->shape = g_array_ref (shape);
  priv->padded_shape = g_array_copy (priv->shape);
  g_array_index (priv->padded_shape, float, priv->padded_shape->len - 1) = \
    apply_padding (g_array_index(priv->padded_shape, float, priv->padded_shape->len - 1), 8);

  return TRUE;
}

/** 
 * pipevec_tensor_set_data:
 * @tensor: A #PipevecTensor
 * @contents: (transfer none) (element-type gfloat): A #GArray containing floating point values.
 * @shape: (transfer none) (element-type gulong): A #GArray describing the dimension of the array.
 * @error: A #GError return pointer
 *
 * Set the data in the tensor.
 *
 * This function always copies the data from the source and always costs
 * at least O(N). The reason for this is that we don't expect to be setting
 * data very often and we may want to perform optimizations such as padding
 * and alignment internally. If @shape does not align with the length of
 * @contents, then we return %FALSE. Otherwise return %TRUE and this
 * @tensor will have array contents as specified by @shape and @contents.
 *
 * Returns: %TRUE if successful, %FALSE with @error set on error.
 */
gboolean
pipevec_tensor_set_data (PipevecTensor *tensor,
                         GArray        *contents,
                         GArray        *shape,
                         GError       **error)
{
  PipevecTensorPrivate *priv = pipevec_tensor_get_instance_private (tensor);
  size_t *shape_data = (size_t *) shape->data;
  size_t shape_product = array_size_t_product (shape_data, shape->len);

  if (shape_product != contents->len)
    {
      g_autofree gchar *formatted_shape = format_size_t_array (shape_data, shape->len);
      g_set_error (error,
                   PIPEVEC_ERROR,
                   PIPEVEC_ERROR_BAD_SHAPE,
                   "Shape %s has product %zu which does not match array length %ui",
                   formatted_shape,
                   shape_product,
                   shape->len);
      return FALSE;
    }

  /* Allocate the new array storage */
  if (!pipevec_tensor_alloc_shape (tensor, shape, error))
    return FALSE;

  /* Now we can copy the unpadded data until our tensor.
   * This always suceeds and the contents of the old tensor
   * have been destroyed at this point. */
  float *passed_array = (float *) contents->data;
  size_t leading_shape = shape_product / shape_data[shape->len - 1];
  size_t inner_shape_padding = g_array_index(priv->padded_shape, float, priv->padded_shape->len - 1);
  size_t inner_shape_no_padding = shape_data[shape->len - 1];

  for (size_t i = 0; i < leading_shape; ++i) {
    for (size_t j = 0; j < inner_shape_no_padding; ++j) {
      priv->array[i * inner_shape_padding + j] = passed_array[i * inner_shape_no_padding + j];
    }

    /* Then add the padding */
    for (size_t j = inner_shape_no_padding; j < inner_shape_padding; ++j) {
      priv->array[i * inner_shape_padding + j] = 0.0f;
    }
  }

  /* Everything else is set, so we can return now */
  return TRUE;
}

/**
 * pipevec_tensor_get_data:
 * @tensor: A #PipevecTensor
 *
 * Fetch the internal data from @tensor. This operation is always at least O(N),
 * data internally may be padded or aligned. When fetching the data we remove the
 * padding and alignment and give it back to the user in its original form.
 *
 * Returns: (transfer full) (element-type gfloat): A new #GArray containing the tensor data.
 */
GArray *
pipevec_tensor_get_data (PipevecTensor  *tensor)
{
  PipevecTensorPrivate *priv = pipevec_tensor_get_instance_private (tensor);
  size_t *shape_data = (size_t *) priv->shape->data;
  size_t return_value_len = array_size_t_product (shape_data, priv->shape->len);
  GArray *return_value = g_array_sized_new (FALSE, FALSE, sizeof (float), return_value_len);
  float *return_value_data = (float *) return_value->data;

  return_value->len = return_value_len;

  /* OK, now we can copy from our array back into the return_value -
   * we have to make sure that we ignore padding */
  size_t *shape_data_no_padding = (size_t *) shape_data;

  size_t leading_shape = return_value_len / shape_data_no_padding[priv->shape->len - 1];
  size_t inner_shape_no_padding = shape_data_no_padding[priv->shape->len - 1];
  size_t inner_shape_padding = apply_padding (shape_data[priv->shape->len - 1], 8);

  for (size_t i = 0; i < leading_shape; ++i) {
    for (size_t j = 0; j < inner_shape_no_padding; ++j) {
      return_value_data[i * inner_shape_no_padding + j] = priv->array[i * inner_shape_padding + j];
    }
  }

  return return_value;
}

/**
 * pipevec_tensor_reshape:
 * @tensor: A #PipevecTensor
 * @shape: (element-type guint): A #GArray describing the dimension of the tensor
 *
 * Give a new shape to the tensor. Right now, this is always an O(N) operation
 * on the size of the array, since the array may need to be re-padded. Therefore,
 * users should reshape judiciously. In future, this may be optimized such that
 * the copy can be elided in certain circumstances.
 *
 * Returns: %TRUE if reshape succeeded, %FALSE if the new shape is not compatible with
 *          existing array contents.
 */
gboolean
pipevec_tensor_reshape (PipevecTensor  *tensor,
                        GArray         *shape,
                        GError        **error)
{
  /* First, grab the data */
  GArray *data = pipevec_tensor_get_data (tensor);

  /* Then, set the data with the new shape */
  return pipevec_tensor_set_data (tensor, data, shape, error);
}

/**
 * pipevec_tensor_set_data_aligned_padded:
 * @tensor: A #PipevecTensor
 * @padded_data: Already padded and 8-aligned float data.
 * @padded_data_length: How long the padded data is
 *
 * Set already-padded data as the tensor data, copying it into
 * the tensor's existing storage.
 *
 * Returns: %TRUE if the operation succeeded, %FALSE with error set on failure.
 */
static gboolean
pipevec_tensor_set_data_aligned_padded (PipevecTensor  *tensor,
                                        GArray         *shape,
                                        float          *padded_data,
                                        size_t          padded_data_length,
                                        GError       **error)
{
  PipevecTensorPrivate *priv = pipevec_tensor_get_instance_private (tensor);

  if (!pipevec_tensor_alloc_shape (tensor, shape, error))
    {
      return FALSE;
    }

  memcpy (priv->array, padded_data, sizeof (float) * padded_data_length);

  return TRUE;
}

/**
 * pipevec_tensor_copy:
 * @tensor: A #PipevecTensor
 * @error: A #GError
 *
 * Make a deep copy of this tensor. Under the hood this
 * allocates a new tensor with the same shape, then
 * copies the data in.
 *
 * Returns: (transfer full): A new #PipevecTensor or %NULL with @error set on failure.
 */
PipevecTensor *
pipevec_tensor_copy (PipevecTensor  *tensor,
                     GError        **error)
{
  g_autoptr(PipevecTensor) new_tensor = g_object_new (PIPEVEC_TYPE_TENSOR, NULL);
  PipevecTensorPrivate *priv = pipevec_tensor_get_instance_private (tensor);

  if (!pipevec_tensor_set_data_aligned_padded (new_tensor,
                                               priv->shape,
                                               priv->array,
                                               array_size_t_product ((size_t *) priv->padded_shape->data,
                                                                     priv->padded_shape->len),
                                               error))
    return NULL;

  return g_steal_pointer (&new_tensor);
}

static inline void
set_location (GArray *location,
              GArray *shape,
              size_t  i)
{
  /* Counting down on the shape, compute the index
   * we're at on each dimension */
  int product = 1;
  size_t *location_data = (size_t *) location->data;
  size_t *shape_data = (size_t *) shape->data;

  for (int j = shape->len - 1; j >= 0; --j) {
    location_data[j] = (i / product) % shape_data[j];
    product *= shape_data[j];
  }
}

/**
 * pipevec_tensor_map:
 * @src: A #PipevecTensor
 * @func: (scope call): A #PipevecTensorMapFunction to be applied to each element
 * @user_data: Some user data to be provided to @func
 * @error: A #GError out pointer.
 *
 * Apply @func to each element in @src, returning the result as a copy.
 *
 * Returns: (transfer full): A new #PipevecTensor with the map function applied.
 */
PipevecTensor *
pipevec_tensor_map (PipevecTensor             *src,
                    PipevecTensorMapFunction   func,
                    gpointer                  *user_data,
                    GError                   **error)
{
  g_autoptr(PipevecTensor) dst = pipevec_tensor_copy (src, error);

  if (dst == NULL)
    return NULL;

  PipevecTensorPrivate *priv = pipevec_tensor_get_instance_private (dst);

  /* Now loop over the tensor and apply the map function to it */
  size_t *shape_data_no_padding = (size_t *) priv->shape->data;
  size_t *shape_data_with_padding = (size_t *) priv->padded_shape->data;

  size_t leading_shape = (
    array_size_t_product (shape_data_no_padding, priv->shape->len) /
    shape_data_no_padding[priv->shape->len - 1]
  );
  size_t inner_shape = shape_data_no_padding[priv->shape->len - 1];
  size_t inner_shape_padding = shape_data_with_padding[priv->padded_shape->len - 1];

  GArray *location = g_array_sized_new (FALSE, FALSE, sizeof(gfloat), priv->shape->len);

  for (size_t i = 0; i < leading_shape; ++i)
    {
      for (size_t j = 0; j < inner_shape; ++j)
        {
          set_location (location, priv->shape, i * inner_shape + j);
          priv->array[i * inner_shape_padding + j] = func(priv->array[i * inner_shape_padding + j],
                                                          location,
                                                          user_data);
        }
    }

  return g_steal_pointer (&dst);
}

static gboolean
shape_subset_equal (GArray *lhs, GArray *rhs, size_t offset_from_end)
{
  if (lhs->len != rhs->len)
    return FALSE;

  if (lhs->len < offset_from_end)
    return FALSE;

  size_t *ldata = (size_t *) lhs->data;
  size_t *rdata = (size_t *) rhs->data;

  for (size_t i = 0; i < (lhs->len - offset_from_end); ++i)
    {
      if (ldata[i] != rdata[i])
        return FALSE;
    }

  return TRUE;
}

static gboolean
shapes_equal (GArray *lhs, GArray *rhs)
{
  return shape_subset_equal (lhs, rhs, 0);
}

static gboolean
check_shapes_elementwise (GArray  *lhs,
                          GArray  *rhs,
                          GError **error)
{
  if (!shapes_equal (lhs, rhs))
    {
      g_autofree char *formatted_lhs_shape = format_size_t_array ((size_t *) lhs->data, lhs->len);
      g_autofree char *formatted_rhs_shape = format_size_t_array ((size_t *) rhs->data, rhs->len);

      g_set_error (error,
                   PIPEVEC_ERROR,
                   PIPEVEC_ERROR_BAD_SHAPE,
                   "Expected shapes %s and %s to be equal",
                   formatted_lhs_shape,
                   formatted_rhs_shape);
      return FALSE;
    }

  return TRUE;
}

typedef float (*PipevecTensorElementwiseFunc) (float lhs, float rhs);

static PipevecTensor *
pipevec_tensor_do_elementwise_op (PipevecTensor                 *lhs,
                                  PipevecTensor                 *rhs,
                                  PipevecTensorElementwiseFunc   func,
                                  GError                       **error)
{
  PipevecTensorPrivate *lhs_priv = pipevec_tensor_get_instance_private (lhs);
  PipevecTensorPrivate *rhs_priv = pipevec_tensor_get_instance_private (rhs);

  if (!check_shapes_elementwise (lhs_priv->shape, rhs_priv->shape, error))
    return NULL;

  g_autoptr(PipevecTensor) dst = pipevec_tensor_copy (lhs, error);

  if (dst == NULL)
   return NULL;

  PipevecTensorPrivate *priv = pipevec_tensor_get_instance_private (dst);

  /* We assume here that padding is equal */
  size_t *shape_data_no_padding = (size_t *) lhs_priv->shape->data;
  size_t *shape_data_with_padding = (size_t *) lhs_priv->padded_shape->data;

  size_t leading_shape = (
    array_size_t_product (shape_data_no_padding, lhs_priv->shape->len) / 
    shape_data_no_padding[lhs_priv->shape->len - 1]
  );
  size_t inner_shape = shape_data_no_padding[lhs_priv->shape->len - 1];
  size_t inner_shape_padding = shape_data_with_padding[lhs_priv->padded_shape->len - 1];

  for (size_t i = 0; i < leading_shape; ++i)
    {
      for (size_t j = 0; j < inner_shape; ++j)
        {
          size_t offset = i * inner_shape_padding + j;
          priv->array[offset] = func(lhs_priv->array[offset], rhs_priv->array[offset]);
        }
    }

  return g_steal_pointer (&dst);
}

static PipevecTensor *
pipevec_tensor_do_scalar_op (PipevecTensor                 *lhs,
                             float                          rhs,
                             PipevecTensorElementwiseFunc   func,
                             GError                       **error)
{
  PipevecTensorPrivate *lhs_priv = pipevec_tensor_get_instance_private (lhs);

  g_autoptr(PipevecTensor) dst = pipevec_tensor_copy (lhs, error);

  if (dst == NULL)
   return NULL;

  PipevecTensorPrivate *priv = pipevec_tensor_get_instance_private (dst);

  /* We assume here that padding is equal */
  size_t *shape_data_no_padding = (size_t *) lhs_priv->shape->data;
  size_t *shape_data_with_padding = (size_t *) lhs_priv->padded_shape->data;

  size_t leading_shape = (
    array_size_t_product (shape_data_no_padding, lhs_priv->shape->len) / 
    shape_data_no_padding[lhs_priv->shape->len - 1]
  );
  size_t inner_shape = shape_data_no_padding[lhs_priv->shape->len - 1];
  size_t inner_shape_padding = shape_data_with_padding[lhs_priv->padded_shape->len - 1];

  for (size_t i = 0; i < leading_shape; ++i)
    {
      for (size_t j = 0; j < inner_shape; ++j)
        {
          size_t offset = i * inner_shape_padding + j;
          priv->array[offset] = func(lhs_priv->array[offset], rhs);
        }
    }

  return g_steal_pointer (&dst);
}

static inline float
add (float lhs, float rhs)
{
  return lhs + rhs;
}


/**
 * pipevec_tensor_add_tensor:
 * @lhs: A #PipevecTensor
 * @rhs: A #PipevecTensor
 * @error: A #GError out pointer.
 *
 * Add two tensors.
 *
 * Returns: (transfer full): A new #PipevecTensor.
 */
PipevecTensor *
pipevec_tensor_add_tensor (PipevecTensor  *lhs,
                           PipevecTensor  *rhs,
                           GError        **error)
{
  return pipevec_tensor_do_elementwise_op (lhs, rhs, add, error);
}

/**
 * pipevec_tensor_add_scalar:
 * @lhs: A #PipevecTensor
 * @rhs: A scalar to add
 * @error: A #GError out pointer.
 *
 * Add a scalar to a tensor.
 *
 * Returns: (transfer full): A new #PipevecTensor.
 */
PipevecTensor *
pipevec_tensor_add_scalar (PipevecTensor  *lhs,
                           float           rhs,
                           GError        **error)
{
  return pipevec_tensor_do_scalar_op (lhs, rhs, add, error);
}

static inline float
sub (float lhs, float rhs)
{
  return lhs + rhs;
}

/**
 * pipevec_tensor_sub_tensor:
 * @lhs: A #PipevecTensor
 * @rhs: A #PipevecTensor
 * @error: A #GError out pointer.
 *
 * Subtract two tensors.
 *
 * Returns: (transfer full): A new #PipevecTensor.
 */
PipevecTensor *
pipevec_tensor_sub_tensor (PipevecTensor  *lhs,
                           PipevecTensor  *rhs,
                           GError        **error)
{
  return pipevec_tensor_do_elementwise_op (lhs, rhs, sub, error);
}

/**
 * pipevec_tensor_sub_scalar:
 * @lhs: A #PipevecTensor
 * @rhs: A scalar to add
 * @error: A #GError out pointer.
 *
 * Subtract a scalar from a tensor.
 *
 * Returns: (transfer full): A new #PipevecTensor.
 */
PipevecTensor *
pipevec_tensor_sub_scalar (PipevecTensor  *lhs,
                           float           rhs,
                           GError        **error)
{
  return pipevec_tensor_do_scalar_op (lhs, rhs, sub, error);
}

static inline float
mul (float lhs, float rhs)
{
  return lhs * rhs;
}

/**
 * pipevec_tensor_multiply_tensor:
 * @lhs: A #PipevecTensor
 * @rhs: A #PipevecTensor
 * @error: A #GError out pointer.
 *
 * Multiply two tensors elementwise.
 *
 * Returns: (transfer full): A new #PipevecTensor.
 */
PipevecTensor *
pipevec_tensor_multiply_tensor (PipevecTensor  *lhs,
                                PipevecTensor  *rhs,
                                GError        **error)
{
  return pipevec_tensor_do_elementwise_op (lhs, rhs, mul, error);
}

/**
 * pipevec_tensor_multiply_scalar:
 * @lhs: A #PipevecTensor
 * @rhs: A scalar to add
 * @error: A #GError out pointer.
 *
 * Multiply a scalar with a tensor.
 *
 * Returns: (transfer full): A new #PipevecTensor.
 */
PipevecTensor *
pipevec_tensor_multiply_scalar (PipevecTensor  *lhs,
                                float           rhs,
                                GError        **error)
{
  return pipevec_tensor_do_scalar_op (lhs, rhs, mul, error);
}

static inline float
divide (float lhs, float rhs)
{
  return lhs / rhs;
}

/**
 * pipevec_tensor_divide_tensor:
 * @lhs: A #PipevecTensor
 * @rhs: A #PipevecTensor
 * @error: A #GError out pointer.
 *
 * Divide two tensors elementwise.
 *
 * Returns: (transfer full): A new #PipevecTensor.
 */
PipevecTensor *
pipevec_tensor_divide_tensor (PipevecTensor  *lhs,
                              PipevecTensor  *rhs,
                              GError        **error)
{
  return pipevec_tensor_do_elementwise_op (lhs, rhs, divide, error);
}

/**
 * pipevec_tensor_divide_scalar:
 * @lhs: A #PipevecTensor
 * @rhs: A scalar to add
 * @error: A #GError out pointer.
 *
 * Divide a scalar by a tensor.
 *
 * Returns: (transfer full): A new #PipevecTensor.
 */
PipevecTensor *
pipevec_tensor_divide_scalar (PipevecTensor  *lhs,
                              float           rhs,
                              GError        **error)
{
  return pipevec_tensor_do_scalar_op (lhs, rhs, divide, error);
}

/**
 * pipevec_tensor_inner_product_tensor:
 * @lhs: A #PipevecTensor
 * @rhs: A #PipevecTensor
 * @error: A #GError out pointer.
 *
 * Compute the inner product of two tensors.
 *
 * Returns: (transfer full): A new #PipevecTensor.
 */
PipevecTensor *
pipevec_tensor_inner_product_tensor (PipevecTensor  *lhs,
                                     PipevecTensor  *rhs,
                                     GError        **error)
{
  PipevecTensorPrivate *lhs_priv = pipevec_tensor_get_instance_private (lhs);
  PipevecTensorPrivate *rhs_priv = pipevec_tensor_get_instance_private (rhs);

  size_t *shape_lhs = (size_t *) lhs_priv->shape->data;
  size_t *shape_rhs = (size_t *) rhs_priv->shape->data;

  if (!shape_subset_equal (lhs_priv->shape, rhs_priv->shape, 2))
    {
      g_autofree char *lhs_formatted_shape = format_size_t_array (shape_lhs, lhs_priv->shape->len);
      g_autofree char *rhs_formatted_shape = format_size_t_array (shape_rhs, rhs_priv->shape->len);
      g_set_error (error,
                   PIPEVEC_ERROR,
                   PIPEVEC_ERROR_BAD_SHAPE,
                   "%s and %s must have the same leading shape",
                   lhs_formatted_shape,
                   rhs_formatted_shape);
      return NULL;
    }

  if (shape_lhs[lhs_priv->shape->len - 1] != shape_rhs[rhs_priv->shape->len - 2])
    {
      g_autofree char *lhs_formatted_shape = format_size_t_array (shape_lhs, lhs_priv->shape->len);
      g_autofree char *rhs_formatted_shape = format_size_t_array (shape_rhs, rhs_priv->shape->len);
      g_set_error (error,
                   PIPEVEC_ERROR,
                   PIPEVEC_ERROR_BAD_SHAPE,
                   "Arrays of shape %s and %s are not compatible for inner product, %zu != %zu",
                   lhs_formatted_shape,
                   rhs_formatted_shape,
                   shape_lhs[lhs_priv->shape->len - 1],
                   shape_rhs[rhs_priv->shape->len - 2]);
      return NULL;
    }

  /* Allocate a new tensor with shape (..., M, K)
   * where the trailing dimensions where of lhs M, N
   * and the trailing dimensions of rhs were N, K */
  GArray *new_shape = g_array_copy (lhs_priv->shape);
  g_array_index (new_shape, float, new_shape->len - 2) = \
    g_array_index (lhs_priv->shape, float, lhs_priv->shape->len - 2);
  g_array_index(new_shape, float, new_shape->len - 1) = \
    g_array_index (rhs_priv->shape, float, rhs_priv->shape->len - 1);

  g_autoptr(PipevecTensor) new_tensor = g_object_new (PIPEVEC_TYPE_TENSOR, NULL);

  if (!pipevec_tensor_alloc_shape (new_tensor, new_shape, NULL))
    return NULL;

  PipevecTensorPrivate *new_tensor_priv = pipevec_tensor_get_instance_private (new_tensor);

  /* Now we need to perform the inner product. First take the shape product
   * of the leading components (eg, not including the two last components) */
  size_t leading_batch_component_shape = array_size_t_product ((size_t *) new_tensor_priv->shape->data,
                                                               new_tensor_priv->shape->len - 2);
  size_t trailing_batch_component_padded = array_size_t_product ((size_t *) new_tensor_priv->padded_shape->data,
                                                                 new_tensor_priv->padded_shape->len) / leading_batch_component_shape;

  size_t rows = g_array_index (new_tensor_priv->shape, float, new_tensor_priv->shape->len - 2);
  size_t columns = g_array_index (new_tensor_priv->shape, float, new_tensor_priv->shape->len - 1);
  size_t dot_product_vector_length = g_array_index (lhs_priv->shape, float, lhs_priv->shape->len - 1);
  size_t row_length = g_array_index (new_tensor_priv->padded_shape, float, new_tensor_priv->padded_shape->len - 1);
  size_t lhs_row_length = g_array_index (lhs_priv->padded_shape, float, lhs_priv->padded_shape->len - 1);
  size_t rhs_row_length = g_array_index (rhs_priv->padded_shape, float, rhs_priv->padded_shape->len - 1);

  /* Iterate along the batches */
  for (size_t batch_index = 0; batch_index < leading_batch_component_shape; ++batch_index)
    {
      size_t batch_offset = batch_index * trailing_batch_component_padded;

      /* Iterate the rows of the result */
      for (size_t i = 0; i < rows; ++i)
        {
          /* Iterate along the columns of the result */
          for (size_t j = 0; j < columns; ++j)
            {
              size_t dst_offset = batch_index * trailing_batch_component_padded + i * row_length + j;
              new_tensor_priv->array[dst_offset] = 0.0f;

              /* Now iterate the columns of LHS to perform the dot product */
              for (size_t k = 0; k < dot_product_vector_length; ++k)
                {
                  new_tensor_priv->array[dst_offset] += \
                    lhs_priv->array[batch_offset + i * lhs_row_length + k] * \
                    rhs_priv->array[batch_offset + k * rhs_row_length + j];
                }
            }
        }
    }

  return g_steal_pointer (&new_tensor);
}

void
pipevec_tensor_finalize (GObject *object)
{
  PipevecTensor *tensor = PIPEVEC_TENSOR (object);
  PipevecTensorPrivate *priv = pipevec_tensor_get_instance_private (tensor);

  g_clear_pointer (&priv->array, g_free);
  g_clear_pointer (&priv->shape, g_array_unref);
  g_clear_pointer (&priv->padded_shape, g_array_unref);
}

void
pipevec_tensor_class_init (PipevecTensorClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = pipevec_tensor_finalize;
}

void
pipevec_tensor_init (PipevecTensor *tensor)
{
  PipevecTensorPrivate *priv = pipevec_tensor_get_instance_private (tensor);

  priv->shape = g_array_new (FALSE, FALSE, sizeof (size_t));
  priv->padded_shape = g_array_new (FALSE, FALSE, sizeof (size_t));
  priv->array = NULL;
}

/**
 * pipevec_tensor_new:
 * @shape: (element-type guint): A #GArray describing the tensor shape.
 * @contents: (element-type gfloat): A #GArray with the flattened tensor contents.
 * @error: An out #GError pointer.
 *
 * Create a new tensor, with the contents @contents and shape @shape.
 *
 * Returns: (transfer full): A new #PipevecTensor.
 */
PipevecTensor *
pipevec_tensor_new (GArray  *shape,
                    GArray  *contents,
                    GError **error)
{
  g_autoptr(PipevecTensor) tensor = g_object_new (PIPEVEC_TYPE_TENSOR, NULL);

  if (!pipevec_tensor_set_data (tensor, contents, shape, error))
    return NULL;

  return tensor;
}