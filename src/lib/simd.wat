(module
  ;; Import shared memory from JavaScript host
  (import "env" "memory" (memory 1))

  ;; normalize(ptr: i32, dimensions: i32)
  ;; Normalizes a vector in-place to unit length using SIMD.
  ;; ptr: byte offset of the vector in memory
  ;; dimensions: number of f32 elements
  ;; Optimized with 4x loop unrolling (16 floats/iteration) and multiple accumulators.
  (func (export "normalize") (param $ptr i32) (param $dim i32)
    (local $i i32)
    (local $acc0 v128)
    (local $acc1 v128)
    (local $acc2 v128)
    (local $acc3 v128)
    (local $sum f32)
    (local $mag f32)
    (local $inv_mag f32)
    (local $inv_vec v128)
    (local $unroll_end i32)
    (local $simd_end i32)
    (local $offset i32)

    ;; Phase 1: Compute sum of squares
    ;; 4x unrolled: process 16 floats per iteration with 4 independent accumulators
    (local.set $acc0 (v128.const f32x4 0 0 0 0))
    (local.set $acc1 (v128.const f32x4 0 0 0 0))
    (local.set $acc2 (v128.const f32x4 0 0 0 0))
    (local.set $acc3 (v128.const f32x4 0 0 0 0))
    (local.set $unroll_end (i32.and (local.get $dim) (i32.const -16)))  ;; dim & ~15
    (local.set $simd_end (i32.and (local.get $dim) (i32.const -4)))     ;; dim & ~3
    (local.set $i (i32.const 0))

    (block $break_sum_u
      (loop $loop_sum_u
        (br_if $break_sum_u (i32.ge_u (local.get $i) (local.get $unroll_end)))
        (local.set $offset (i32.add (local.get $ptr) (i32.shl (local.get $i) (i32.const 2))))

        (local.set $acc0 (f32x4.add (local.get $acc0)
          (f32x4.mul (v128.load (local.get $offset)) (v128.load (local.get $offset)))))
        (local.set $acc1 (f32x4.add (local.get $acc1)
          (f32x4.mul (v128.load offset=16 (local.get $offset)) (v128.load offset=16 (local.get $offset)))))
        (local.set $acc2 (f32x4.add (local.get $acc2)
          (f32x4.mul (v128.load offset=32 (local.get $offset)) (v128.load offset=32 (local.get $offset)))))
        (local.set $acc3 (f32x4.add (local.get $acc3)
          (f32x4.mul (v128.load offset=48 (local.get $offset)) (v128.load offset=48 (local.get $offset)))))

        (local.set $i (i32.add (local.get $i) (i32.const 16)))
        (br $loop_sum_u)
      )
    )

    ;; Merge 4 accumulators into one
    (local.set $acc0 (f32x4.add (f32x4.add (local.get $acc0) (local.get $acc1))
                                (f32x4.add (local.get $acc2) (local.get $acc3))))

    ;; Handle remaining 4-wide chunks
    (block $break_sum4
      (loop $loop_sum4
        (br_if $break_sum4 (i32.ge_u (local.get $i) (local.get $simd_end)))
        (local.set $offset (i32.add (local.get $ptr) (i32.shl (local.get $i) (i32.const 2))))
        (local.set $acc0 (f32x4.add (local.get $acc0)
          (f32x4.mul (v128.load (local.get $offset)) (v128.load (local.get $offset)))))
        (local.set $i (i32.add (local.get $i) (i32.const 4)))
        (br $loop_sum4)
      )
    )

    ;; Horizontal sum of SIMD lanes
    (local.set $sum
      (f32.add
        (f32.add (f32x4.extract_lane 0 (local.get $acc0)) (f32x4.extract_lane 1 (local.get $acc0)))
        (f32.add (f32x4.extract_lane 2 (local.get $acc0)) (f32x4.extract_lane 3 (local.get $acc0)))
      )
    )

    ;; Handle scalar remainder (dim % 4)
    (block $break_rem_sum
      (loop $loop_rem_sum
        (br_if $break_rem_sum (i32.ge_u (local.get $i) (local.get $dim)))
        (local.set $offset (i32.add (local.get $ptr) (i32.shl (local.get $i) (i32.const 2))))
        (local.set $sum (f32.add (local.get $sum)
          (f32.mul (f32.load (local.get $offset)) (f32.load (local.get $offset)))))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $loop_rem_sum)
      )
    )

    ;; Compute magnitude and check for zero
    (local.set $mag (f32.sqrt (local.get $sum)))
    (if (f32.eq (local.get $mag) (f32.const 0))
      (then (return))
    )

    ;; Phase 2: Divide each element by magnitude using SIMD (4x unrolled)
    (local.set $inv_mag (f32.div (f32.const 1) (local.get $mag)))
    (local.set $inv_vec (f32x4.splat (local.get $inv_mag)))
    (local.set $i (i32.const 0))

    (block $break_norm_u
      (loop $loop_norm_u
        (br_if $break_norm_u (i32.ge_u (local.get $i) (local.get $unroll_end)))
        (local.set $offset (i32.add (local.get $ptr) (i32.shl (local.get $i) (i32.const 2))))

        (v128.store (local.get $offset)
          (f32x4.mul (v128.load (local.get $offset)) (local.get $inv_vec)))
        (v128.store offset=16 (local.get $offset)
          (f32x4.mul (v128.load offset=16 (local.get $offset)) (local.get $inv_vec)))
        (v128.store offset=32 (local.get $offset)
          (f32x4.mul (v128.load offset=32 (local.get $offset)) (local.get $inv_vec)))
        (v128.store offset=48 (local.get $offset)
          (f32x4.mul (v128.load offset=48 (local.get $offset)) (local.get $inv_vec)))

        (local.set $i (i32.add (local.get $i) (i32.const 16)))
        (br $loop_norm_u)
      )
    )

    ;; Handle remaining 4-wide chunks
    (block $break_norm4
      (loop $loop_norm4
        (br_if $break_norm4 (i32.ge_u (local.get $i) (local.get $simd_end)))
        (local.set $offset (i32.add (local.get $ptr) (i32.shl (local.get $i) (i32.const 2))))
        (v128.store (local.get $offset)
          (f32x4.mul (v128.load (local.get $offset)) (local.get $inv_vec)))
        (local.set $i (i32.add (local.get $i) (i32.const 4)))
        (br $loop_norm4)
      )
    )

    ;; Handle scalar remainder
    (block $break_rem_norm
      (loop $loop_rem_norm
        (br_if $break_rem_norm (i32.ge_u (local.get $i) (local.get $dim)))
        (local.set $offset (i32.add (local.get $ptr) (i32.shl (local.get $i) (i32.const 2))))
        (f32.store (local.get $offset)
          (f32.mul (f32.load (local.get $offset)) (local.get $inv_mag)))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $loop_rem_norm)
      )
    )
  )

  ;; search_all(query_ptr: i32, db_ptr: i32, scores_ptr: i32, db_size: i32, dimensions: i32)
  ;; Computes dot products of query against every vector in the database.
  ;; Optimized with 4x loop unrolling (16 floats/iteration) and multiple accumulators
  ;; for improved instruction-level parallelism.
  (func (export "search_all") (param $query_ptr i32) (param $db_ptr i32) (param $scores_ptr i32) (param $db_size i32) (param $dim i32)
    (local $i i32)
    (local $j i32)
    (local $acc0 v128)
    (local $acc1 v128)
    (local $acc2 v128)
    (local $acc3 v128)
    (local $dot f32)
    (local $vec_ptr i32)
    (local $unroll_end i32)
    (local $simd_end i32)
    (local $q_offset i32)
    (local $v_offset i32)
    (local $bytes_per_vec i32)

    (local.set $unroll_end (i32.and (local.get $dim) (i32.const -16)))  ;; dim & ~15
    (local.set $simd_end (i32.and (local.get $dim) (i32.const -4)))     ;; dim & ~3
    (local.set $bytes_per_vec (i32.shl (local.get $dim) (i32.const 2)))
    (local.set $i (i32.const 0))

    (block $break_outer
      (loop $loop_outer
        (br_if $break_outer (i32.ge_u (local.get $i) (local.get $db_size)))

        ;; Pointer to the i-th database vector
        (local.set $vec_ptr
          (i32.add (local.get $db_ptr) (i32.mul (local.get $i) (local.get $bytes_per_vec)))
        )

        ;; 4 independent SIMD accumulators for ILP
        (local.set $acc0 (v128.const f32x4 0 0 0 0))
        (local.set $acc1 (v128.const f32x4 0 0 0 0))
        (local.set $acc2 (v128.const f32x4 0 0 0 0))
        (local.set $acc3 (v128.const f32x4 0 0 0 0))
        (local.set $j (i32.const 0))

        ;; 4x unrolled inner loop: 16 floats per iteration
        (block $break_inner_u
          (loop $loop_inner_u
            (br_if $break_inner_u (i32.ge_u (local.get $j) (local.get $unroll_end)))
            (local.set $q_offset (i32.add (local.get $query_ptr) (i32.shl (local.get $j) (i32.const 2))))
            (local.set $v_offset (i32.add (local.get $vec_ptr) (i32.shl (local.get $j) (i32.const 2))))

            (local.set $acc0 (f32x4.add (local.get $acc0)
              (f32x4.mul (v128.load (local.get $q_offset)) (v128.load (local.get $v_offset)))))
            (local.set $acc1 (f32x4.add (local.get $acc1)
              (f32x4.mul (v128.load offset=16 (local.get $q_offset)) (v128.load offset=16 (local.get $v_offset)))))
            (local.set $acc2 (f32x4.add (local.get $acc2)
              (f32x4.mul (v128.load offset=32 (local.get $q_offset)) (v128.load offset=32 (local.get $v_offset)))))
            (local.set $acc3 (f32x4.add (local.get $acc3)
              (f32x4.mul (v128.load offset=48 (local.get $q_offset)) (v128.load offset=48 (local.get $v_offset)))))

            (local.set $j (i32.add (local.get $j) (i32.const 16)))
            (br $loop_inner_u)
          )
        )

        ;; Merge 4 accumulators
        (local.set $acc0 (f32x4.add (f32x4.add (local.get $acc0) (local.get $acc1))
                                    (f32x4.add (local.get $acc2) (local.get $acc3))))

        ;; Handle remaining 4-wide chunks
        (block $break_inner4
          (loop $loop_inner4
            (br_if $break_inner4 (i32.ge_u (local.get $j) (local.get $simd_end)))
            (local.set $q_offset (i32.add (local.get $query_ptr) (i32.shl (local.get $j) (i32.const 2))))
            (local.set $v_offset (i32.add (local.get $vec_ptr) (i32.shl (local.get $j) (i32.const 2))))
            (local.set $acc0 (f32x4.add (local.get $acc0)
              (f32x4.mul (v128.load (local.get $q_offset)) (v128.load (local.get $v_offset)))))
            (local.set $j (i32.add (local.get $j) (i32.const 4)))
            (br $loop_inner4)
          )
        )

        ;; Horizontal sum of SIMD accumulator
        (local.set $dot
          (f32.add
            (f32.add (f32x4.extract_lane 0 (local.get $acc0)) (f32x4.extract_lane 1 (local.get $acc0)))
            (f32.add (f32x4.extract_lane 2 (local.get $acc0)) (f32x4.extract_lane 3 (local.get $acc0)))
          )
        )

        ;; Handle scalar remainder (dim % 4)
        (block $break_rem
          (loop $loop_rem
            (br_if $break_rem (i32.ge_u (local.get $j) (local.get $dim)))
            (local.set $q_offset (i32.add (local.get $query_ptr) (i32.shl (local.get $j) (i32.const 2))))
            (local.set $v_offset (i32.add (local.get $vec_ptr) (i32.shl (local.get $j) (i32.const 2))))
            (local.set $dot (f32.add (local.get $dot)
              (f32.mul (f32.load (local.get $q_offset)) (f32.load (local.get $v_offset)))))
            (local.set $j (i32.add (local.get $j) (i32.const 1)))
            (br $loop_rem)
          )
        )

        ;; Store score for vector i
        (f32.store
          (i32.add (local.get $scores_ptr) (i32.shl (local.get $i) (i32.const 2)))
          (local.get $dot)
        )

        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $loop_outer)
      )
    )
  )
)
