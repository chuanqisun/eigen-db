(module
  ;; Import shared memory from JavaScript host
  (import "env" "memory" (memory 1))

  ;; normalize(ptr: i32, dimensions: i32)
  ;; Normalizes a vector in-place to unit length using SIMD.
  (func (export "normalize") (param $ptr i32) (param $dim i32)
    (local $i i32)
    (local $acc v128)
    (local $sum f32)
    (local $mag f32)
    (local $inv_mag f32)
    (local $inv_vec v128)
    (local $simd_end i32)
    (local $offset i32)

    ;; Phase 1: Sum of squares
    (local.set $acc (v128.const f32x4 0 0 0 0))
    (local.set $simd_end (i32.and (local.get $dim) (i32.const -4)))
    (local.set $i (i32.const 0))

    ;; SIMD loop: 4 floats per iteration
    (block $break_sum
      (loop $loop_sum
        (br_if $break_sum (i32.ge_u (local.get $i) (local.get $simd_end)))
        (local.set $offset (i32.add (local.get $ptr) (i32.shl (local.get $i) (i32.const 2))))
        (local.set $acc (f32x4.add (local.get $acc)
          (f32x4.mul (v128.load (local.get $offset)) (v128.load (local.get $offset)))))
        (local.set $i (i32.add (local.get $i) (i32.const 4)))
        (br $loop_sum)
      )
    )

    ;; Horizontal sum
    (local.set $sum
      (f32.add
        (f32.add (f32x4.extract_lane 0 (local.get $acc)) (f32x4.extract_lane 1 (local.get $acc)))
        (f32.add (f32x4.extract_lane 2 (local.get $acc)) (f32x4.extract_lane 3 (local.get $acc)))))

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

    ;; Phase 2: Scale by inverse magnitude
    (local.set $inv_mag (f32.div (f32.const 1) (local.get $mag)))
    (local.set $inv_vec (f32x4.splat (local.get $inv_mag)))
    (local.set $i (i32.const 0))

    ;; SIMD loop
    (block $break_norm
      (loop $loop_norm
        (br_if $break_norm (i32.ge_u (local.get $i) (local.get $simd_end)))
        (local.set $offset (i32.add (local.get $ptr) (i32.shl (local.get $i) (i32.const 2))))
        (v128.store (local.get $offset)
          (f32x4.mul (v128.load (local.get $offset)) (local.get $inv_vec)))
        (local.set $i (i32.add (local.get $i) (i32.const 4)))
        (br $loop_norm)
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
  (func (export "search_all") (param $query_ptr i32) (param $db_ptr i32) (param $scores_ptr i32) (param $db_size i32) (param $dim i32)
    (local $i i32)
    (local $j i32)
    (local $acc v128)
    (local $dot f32)
    (local $vec_ptr i32)
    (local $simd_end i32)
    (local $q_offset i32)
    (local $v_offset i32)
    (local $bytes_per_vec i32)

    (local.set $simd_end (i32.and (local.get $dim) (i32.const -4)))
    (local.set $bytes_per_vec (i32.shl (local.get $dim) (i32.const 2)))
    (local.set $i (i32.const 0))

    ;; Outer loop: one database vector per iteration
    (block $break_outer
      (loop $loop_outer
        (br_if $break_outer (i32.ge_u (local.get $i) (local.get $db_size)))

        (local.set $vec_ptr
          (i32.add (local.get $db_ptr) (i32.mul (local.get $i) (local.get $bytes_per_vec))))
        (local.set $acc (v128.const f32x4 0 0 0 0))
        (local.set $j (i32.const 0))

        ;; SIMD inner loop: dot product, 4 floats per iteration
        (block $break_inner
          (loop $loop_inner
            (br_if $break_inner (i32.ge_u (local.get $j) (local.get $simd_end)))
            (local.set $q_offset (i32.add (local.get $query_ptr) (i32.shl (local.get $j) (i32.const 2))))
            (local.set $v_offset (i32.add (local.get $vec_ptr) (i32.shl (local.get $j) (i32.const 2))))
            (local.set $acc (f32x4.add (local.get $acc)
              (f32x4.mul (v128.load (local.get $q_offset)) (v128.load (local.get $v_offset)))))
            (local.set $j (i32.add (local.get $j) (i32.const 4)))
            (br $loop_inner)
          )
        )

        ;; Horizontal sum
        (local.set $dot
          (f32.add
            (f32.add (f32x4.extract_lane 0 (local.get $acc)) (f32x4.extract_lane 1 (local.get $acc)))
            (f32.add (f32x4.extract_lane 2 (local.get $acc)) (f32x4.extract_lane 3 (local.get $acc)))))

        ;; Scalar remainder
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

        ;; Store score
        (f32.store
          (i32.add (local.get $scores_ptr) (i32.shl (local.get $i) (i32.const 2)))
          (local.get $dot))

        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $loop_outer)
      )
    )
  )
)
