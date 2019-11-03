/*
 * /pipevec/pipevec-errors.h
 *
 * Error codes and domain for pipevec.
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

#include <glib.h>

G_BEGIN_DECLS

/**
 * PipevecError
 * @PIPEVEC_ERROR_INTERNAL: Internal error occurred in pipevec or another library.
 * @PIPEVEC_ERROR_BAD_SHAPE: The data does not conform to the requested shape.
 * @PIPEVEC_ERROR_DIMENSION_MISMATCH: Dimensions mismatch such that the operation cannot be performed.
 *
 * Error enumeration for Scorch related errors.
 */
typedef enum {
  PIPEVEC_ERROR_INTERNAL,
  PIPEVEC_ERROR_BAD_SHAPE,
  PIPEVEC_ERROR_DIMENSION_MISMATCH
} PipevecError;

#define PIPEVEC_ERROR pipevec_error_quark ()
GQuark pipevec_error_quark (void);

G_END_DECLS
