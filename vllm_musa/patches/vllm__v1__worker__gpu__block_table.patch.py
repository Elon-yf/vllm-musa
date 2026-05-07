# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

PATCHES = [
    (
        "_load_ptr(dst_block_table_ptrs + group_id, tl.int32)",
        "_load_ptr(dst_block_table_ptrs + group_id)",
    ),
    (
        "_load_ptr(src_block_table_ptrs + group_id, tl.int32)",
        "_load_ptr(src_block_table_ptrs + group_id)",
    ),
    (
        "_load_ptr(block_table_ptrs + group_id, tl.int32)",
        "_load_ptr(block_table_ptrs + group_id)",
    ),
    (
        """@triton.jit
def _load_ptr(ptr_to_ptr, elem_dtype):
    ptr = tl.load(ptr_to_ptr)
    ptr = tl.cast(ptr, tl.pointer_type(elem_dtype))
    return tl.multiple_of(ptr, 16)""",
        """@triton.jit
def _load_ptr(ptr_to_ptr):
    ptr = tl.load(ptr_to_ptr)
    ptr = ptr.to(tl.pointer_type(tl.int32))
    return tl.multiple_of(ptr, 16)""",
    ),
]
