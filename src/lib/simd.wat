(module
  ;; Import shared memory from JavaScript host
  (import "env" "memory" (memory 1))

  ;; normalize(ptr: i32, dimensions: i32)
  ;; Normalizes a vector in-place to unit length using SIMD.
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

    ;; Phase 1: Sum of squares with 4x unroll and 4 independent accumulators
    (local.set $acc0 (v128.const f32x4 0 0 0 0))
    (local.set $acc1 (v128.const f32x4 0 0 0 0))
    (local.set $acc2 (v128.const f32x4 0 0 0 0))
    (local.set $acc3 (v128.const f32x4 0 0 0 0))
    (local.set $unroll_end (i32.and (local.get $dim) (i32.const -16)))
    (local.set $simd_end (i32.and (local.get $dim) (i32.const -4)))
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

    ;; Merge 4 accumulators
    (local.set $acc0 (f32x4.add (f32x4.add (local.get $acc0) (local.get $acc1))
                                (f32x4.add (local.get $acc2) (local.get $acc3))))

    ;; Remaining 4-wide chunks
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

    ;; Horizontal sum
    (local.set $sum
      (f32.add
        (f32.add (f32x4.extract_lane 0 (local.get $acc0)) (f32x4.extract_lane 1 (local.get $acc0)))
        (f32.add (f32x4.extract_lane 2 (local.get $acc0)) (f32x4.extract_lane 3 (local.get $acc0)))))

    ;; Scalar remainder
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

    ;; Magnitude check
    (local.set $mag (f32.sqrt (local.get $sum)))
    (if (f32.eq (local.get $mag) (f32.const 0))
      (then (return)))

    ;; Phase 2: Scale by inverse magnitude (4x unrolled)
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

    ;; Remaining 4-wide chunks
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

    ;; Scalar remainder
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

  ;; search_all(query_ptr, db_ptr, scores_ptr, db_size, dimensions)
  ;; Computes dot products of query against every vector in the database.
  ;; Optimized with:
  ;;   - 2-vector outer loop unrolling (halves query memory reads)
  ;;   - 4x inner loop unrolling (16 floats/iteration, 4 accumulators per vector)
  (func (export "search_all") (param $query_ptr i32) (param $db_ptr i32) (param $scores_ptr i32) (param $db_size i32) (param $dim i32)
    (local $i i32)
    (local $j i32)
    (local $accA0 v128)
    (local $accA1 v128)
    (local $accA2 v128)
    (local $accA3 v128)
    (local $accB0 v128)
    (local $accB1 v128)
    (local $accB2 v128)
    (local $accB3 v128)
    (local $q0 v128)
    (local $q1 v128)
    (local $q2 v128)
    (local $q3 v128)
    (local $dotA f32)
    (local $dotB f32)
    (local $vec_ptrA i32)
    (local $vec_ptrB i32)
    (local $unroll_end i32)
    (local $simd_end i32)
    (local $q_offset i32)
    (local $vA_offset i32)
    (local $vB_offset i32)
    (local $bytes_per_vec i32)
    (local $pair_end i32)

    (local.set $unroll_end (i32.and (local.get $dim) (i32.const -16)))
    (local.set $simd_end (i32.and (local.get $dim) (i32.const -4)))
    (local.set $bytes_per_vec (i32.shl (local.get $dim) (i32.const 2)))
    (local.set $pair_end (i32.and (local.get $db_size) (i32.const -2)))
    (local.set $i (i32.const 0))

    ;; Main loop: process 2 database vectors per iteration
    (block $break_outer
      (loop $loop_outer
        (br_if $break_outer (i32.ge_u (local.get $i) (local.get $pair_end)))

        (local.set $vec_ptrA
          (i32.add (local.get $db_ptr) (i32.mul (local.get $i) (local.get $bytes_per_vec))))
        (local.set $vec_ptrB
          (i32.add (local.get $vec_ptrA) (local.get $bytes_per_vec)))

        (local.set $accA0 (v128.const f32x4 0 0 0 0))
        (local.set $accA1 (v128.const f32x4 0 0 0 0))
        (local.set $accA2 (v128.const f32x4 0 0 0 0))
        (local.set $accA3 (v128.const f32x4 0 0 0 0))
        (local.set $accB0 (v128.const f32x4 0 0 0 0))
        (local.set $accB1 (v128.const f32x4 0 0 0 0))
        (local.set $accB2 (v128.const f32x4 0 0 0 0))
        (local.set $accB3 (v128.const f32x4 0 0 0 0))
        (local.set $j (i32.const 0))

        ;; Inner loop: load query once, dot product against both vectors
        (block $break_inner
          (loop $loop_inner
            (br_if $break_inner (i32.ge_u (local.get $j) (local.get $unroll_end)))
            (local.set $q_offset (i32.add (local.get $query_ptr) (i32.shl (local.get $j) (i32.const 2))))
            (local.set $vA_offset (i32.add (local.get $vec_ptrA) (i32.shl (local.get $j) (i32.const 2))))
            (local.set $vB_offset (i32.add (local.get $vec_ptrB) (i32.shl (local.get $j) (i32.const 2))))

            ;; Load query chunk (shared between both vectors)
            (local.set $q0 (v128.load (local.get $q_offset)))
            (local.set $q1 (v128.load offset=16 (local.get $q_offset)))
            (local.set $q2 (v128.load offset=32 (local.get $q_offset)))
            (local.set $q3 (v128.load offset=48 (local.get $q_offset)))

            ;; Vector A
            (local.set $accA0 (f32x4.add (local.get $accA0)
              (f32x4.mul (local.get $q0) (v128.load (local.get $vA_offset)))))
            (local.set $accA1 (f32x4.add (local.get $accA1)
              (f32x4.mul (local.get $q1) (v128.load offset=16 (local.get $vA_offset)))))
            (local.set $accA2 (f32x4.add (local.get $accA2)
              (f32x4.mul (local.get $q2) (v128.load offset=32 (local.get $vA_offset)))))
            (local.set $accA3 (f32x4.add (local.get $accA3)
              (f32x4.mul (local.get $q3) (v128.load offset=48 (local.get $vA_offset)))))

            ;; Vector B (reuses query loads)
            (local.set $accB0 (f32x4.add (local.get $accB0)
              (f32x4.mul (local.get $q0) (v128.load (local.get $vB_offset)))))
            (local.set $accB1 (f32x4.add (local.get $accB1)
              (f32x4.mul (local.get $q1) (v128.load offset=16 (local.get $vB_offset)))))
            (local.set $accB2 (f32x4.add (local.get $accB2)
              (f32x4.mul (local.get $q2) (v128.load offset=32 (local.get $vB_offset)))))
            (local.set $accB3 (f32x4.add (local.get $accB3)
              (f32x4.mul (local.get $q3) (v128.load offset=48 (local.get $vB_offset)))))

            (local.set $j (i32.add (local.get $j) (i32.const 16)))
            (br $loop_inner)
          )
        )

        ;; Merge accumulators
        (local.set $accA0 (f32x4.add (f32x4.add (local.get $accA0) (local.get $accA1))
                                     (f32x4.add (local.get $accA2) (local.get $accA3))))
        (local.set $accB0 (f32x4.add (f32x4.add (local.get $accB0) (local.get $accB1))
                                     (f32x4.add (local.get $accB2) (local.get $accB3))))

        ;; 4-wide cleanup (both vectors)
        (block $break_inner4
          (loop $loop_inner4
            (br_if $break_inner4 (i32.ge_u (local.get $j) (local.get $simd_end)))
            (local.set $q_offset (i32.add (local.get $query_ptr) (i32.shl (local.get $j) (i32.const 2))))
            (local.set $vA_offset (i32.add (local.get $vec_ptrA) (i32.shl (local.get $j) (i32.const 2))))
            (local.set $vB_offset (i32.add (local.get $vec_ptrB) (i32.shl (local.get $j) (i32.const 2))))
            (local.set $q0 (v128.load (local.get $q_offset)))
            (local.set $accA0 (f32x4.add (local.get $accA0) (f32x4.mul (local.get $q0) (v128.load (local.get $vA_offset)))))
            (local.set $accB0 (f32x4.add (local.get $accB0) (f32x4.mul (local.get $q0) (v128.load (local.get $vB_offset)))))
            (local.set $j (i32.add (local.get $j) (i32.const 4)))
            (br $loop_inner4)
          )
        )

        ;; Horizontal sums
        (local.set $dotA
          (f32.add
            (f32.add (f32x4.extract_lane 0 (local.get $accA0)) (f32x4.extract_lane 1 (local.get $accA0)))
            (f32.add (f32x4.extract_lane 2 (local.get $accA0)) (f32x4.extract_lane 3 (local.get $accA0)))))
        (local.set $dotB
          (f32.add
            (f32.add (f32x4.extract_lane 0 (local.get $accB0)) (f32x4.extract_lane 1 (local.get $accB0)))
            (f32.add (f32x4.extract_lane 2 (local.get $accB0)) (f32x4.extract_lane 3 (local.get $accB0)))))

        ;; Scalar remainder (both vectors)
        (block $break_rem
          (loop $loop_rem
            (br_if $break_rem (i32.ge_u (local.get $j) (local.get $dim)))
            (local.set $q_offset (i32.add (local.get $query_ptr) (i32.shl (local.get $j) (i32.const 2))))
            (local.set $vA_offset (i32.add (local.get $vec_ptrA) (i32.shl (local.get $j) (i32.const 2))))
            (local.set $vB_offset (i32.add (local.get $vec_ptrB) (i32.shl (local.get $j) (i32.const 2))))
            (local.set $dotA (f32.add (local.get $dotA)
              (f32.mul (f32.load (local.get $q_offset)) (f32.load (local.get $vA_offset)))))
            (local.set $dotB (f32.add (local.get $dotB)
              (f32.mul (f32.load (local.get $q_offset)) (f32.load (local.get $vB_offset)))))
            (local.set $j (i32.add (local.get $j) (i32.const 1)))
            (br $loop_rem)
          )
        )

        ;; Store scores for vectors i and i+1
        (f32.store
          (i32.add (local.get $scores_ptr) (i32.shl (local.get $i) (i32.const 2)))
          (local.get $dotA))
        (f32.store
          (i32.add (local.get $scores_ptr) (i32.shl (i32.add (local.get $i) (i32.const 1)) (i32.const 2)))
          (local.get $dotB))

        (local.set $i (i32.add (local.get $i) (i32.const 2)))
        (br $loop_outer)
      )
    )

    ;; Handle last vector if db_size is odd
    (if (i32.lt_u (local.get $i) (local.get $db_size))
      (then
        (local.set $vec_ptrA
          (i32.add (local.get $db_ptr) (i32.mul (local.get $i) (local.get $bytes_per_vec))))
        (local.set $accA0 (v128.const f32x4 0 0 0 0))
        (local.set $accA1 (v128.const f32x4 0 0 0 0))
        (local.set $accA2 (v128.const f32x4 0 0 0 0))
        (local.set $accA3 (v128.const f32x4 0 0 0 0))
        (local.set $j (i32.const 0))

        (block $break_last
          (loop $loop_last
            (br_if $break_last (i32.ge_u (local.get $j) (local.get $unroll_end)))
            (local.set $q_offset (i32.add (local.get $query_ptr) (i32.shl (local.get $j) (i32.const 2))))
            (local.set $vA_offset (i32.add (local.get $vec_ptrA) (i32.shl (local.get $j) (i32.const 2))))
            (local.set $accA0 (f32x4.add (local.get $accA0)
              (f32x4.mul (v128.load (local.get $q_offset)) (v128.load (local.get $vA_offset)))))
            (local.set $accA1 (f32x4.add (local.get $accA1)
              (f32x4.mul (v128.load offset=16 (local.get $q_offset)) (v128.load offset=16 (local.get $vA_offset)))))
            (local.set $accA2 (f32x4.add (local.get $accA2)
              (f32x4.mul (v128.load offset=32 (local.get $q_offset)) (v128.load offset=32 (local.get $vA_offset)))))
            (local.set $accA3 (f32x4.add (local.get $accA3)
              (f32x4.mul (v128.load offset=48 (local.get $q_offset)) (v128.load offset=48 (local.get $vA_offset)))))
            (local.set $j (i32.add (local.get $j) (i32.const 16)))
            (br $loop_last)
          )
        )

        (local.set $accA0 (f32x4.add (f32x4.add (local.get $accA0) (local.get $accA1))
                                     (f32x4.add (local.get $accA2) (local.get $accA3))))

        (block $break_last4
          (loop $loop_last4
            (br_if $break_last4 (i32.ge_u (local.get $j) (local.get $simd_end)))
            (local.set $q_offset (i32.add (local.get $query_ptr) (i32.shl (local.get $j) (i32.const 2))))
            (local.set $vA_offset (i32.add (local.get $vec_ptrA) (i32.shl (local.get $j) (i32.const 2))))
            (local.set $accA0 (f32x4.add (local.get $accA0)
              (f32x4.mul (v128.load (local.get $q_offset)) (v128.load (local.get $vA_offset)))))
            (local.set $j (i32.add (local.get $j) (i32.const 4)))
            (br $loop_last4)
          )
        )

        (local.set $dotA
          (f32.add
            (f32.add (f32x4.extract_lane 0 (local.get $accA0)) (f32x4.extract_lane 1 (local.get $accA0)))
            (f32.add (f32x4.extract_lane 2 (local.get $accA0)) (f32x4.extract_lane 3 (local.get $accA0)))))

        (block $break_last_rem
          (loop $loop_last_rem
            (br_if $break_last_rem (i32.ge_u (local.get $j) (local.get $dim)))
            (local.set $q_offset (i32.add (local.get $query_ptr) (i32.shl (local.get $j) (i32.const 2))))
            (local.set $vA_offset (i32.add (local.get $vec_ptrA) (i32.shl (local.get $j) (i32.const 2))))
            (local.set $dotA (f32.add (local.get $dotA)
              (f32.mul (f32.load (local.get $q_offset)) (f32.load (local.get $vA_offset)))))
            (local.set $j (i32.add (local.get $j) (i32.const 1)))
            (br $loop_last_rem)
          )
        )

        (f32.store
          (i32.add (local.get $scores_ptr) (i32.shl (local.get $i) (i32.const 2)))
          (local.get $dotA))
      )
    )
  )
)
