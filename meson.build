# /meson.build
#
# Toplevel meson build file for pipevec.
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

project('pipevec', ['c', 'cpp'],
        version: '0.0.0',
        default_options : ['cpp_std=c++17'],
        license: 'LGPL2+',
        meson_version: '>= 0.46.0')
c_compiler = meson.get_compiler('c')
cpp_compiler = meson.get_compiler('cpp')

pipevec_inc = include_directories('.')

python = import('python')
gnome = import('gnome')

glib = dependency('glib-2.0', version: '>=2.26')
gobject = dependency('gobject-2.0')

gtest_project = subproject('googletest')
gtest_dep = gtest_project.get_variable('gtest_dep')
gtest_main_dep = gtest_project.get_variable('gtest_main_dep')
gmock_dep = gtest_project.get_variable('gmock_dep')

tests_inc = include_directories('tests')

subdir('pipevec')
subdir('tests')
