# /pipevec/meson.build
#
# Build the libpipevec library.
#
# Copyright (C) 2019 Sam Spilsbury.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

api_version = '0'

pipevec_toplevel_headers = files([
  'pipevec.h',
  'pipevec-errors.h',
  'pipevec-tensor.h'
])
pipevec_introspectable_sources = files([
  'pipevec-errors.c',
  'pipevec-tensor.c'
])
pipevec_private_headers = files([
])
pipevec_private_sources = files([
])

pipevec_headers_subdir = 'pipevec'

install_headers(pipevec_toplevel_headers, subdir: pipevec_headers_subdir)

pipevec_sources = pipevec_introspectable_sources + pipevec_private_sources

glib = dependency('glib-2.0')
gobject = dependency('gobject-2.0')

pipevec_lib = shared_library(
  'pipevec',
  pipevec_sources,
  soversion: api_version,
  install: true,
  include_directories: [ pipevec_inc ],
  dependencies: [
    glib,
    gobject
  ]
)

pipevec_dep = declare_dependency(
  link_with: pipevec_lib,
  include_directories: [ pipevec_inc ],
)

introspection_sources = [ pipevec_introspectable_sources, pipevec_toplevel_headers ]

gnome = import('gnome')
gnome.generate_gir(
  pipevec_lib,
  extra_args: ['--warn-all', '--warn-error'],
  identifier_prefix: 'Pipevec',
  include_directories: pipevec_inc,
  includes: ['GLib-2.0', 'GObject-2.0'],
  install: true,
  namespace: 'Pipevec',
  nsversion: api_version,
  sources: introspection_sources,
  symbol_prefix: 'pipevec'
)

pkg = import('pkgconfig')
pkg.generate(
  description: 'Algorithms for GLib applications',
  name: 'libpipevec',
  filebase: 'libpipevec-' + api_version,
  version: meson.project_version(),
  libraries: pipevec_lib,
  install_dir: join_paths(get_option('libdir'), 'pkgconfig')
)
