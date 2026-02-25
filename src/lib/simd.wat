(module
  ;; Import shared memory from JavaScript host
  (import "env" "memory" (memory 1))

  ;; normalize(ptr: i32, dimensions: i32)
  ;; Normalizes a vector in-place to unit length using SIMD.
  ;; ptr: byte offset of the vector in memory
  ;; dimensions: number of f32 elements (must be a multiple of 4 for SIMD path)
  (func (export "normalize") (param $ptr i32) (param $dim i32)
    (local $i i32)
    (local $sum_vec v128)
    (local $sum f32)
    (local $mag f32)
    (local $inv_mag f32)
    (local $inv_vec v128)
    (local $simd_end i32)
    (local $remainder i32)
    (local $offset i32)

    ;; Phase 1: Compute sum of squares using SIMD (4 floats at a time)
    (local.set $sum_vec (v128.const f32x4 0 0 0 0))
    (local.set $simd_end (i32.and (local.get $dim) (i32.const -4)))  ;; dim & ~3
    (local.set $i (i32.const 0))

    (block $break_sum
      (loop $loop_sum
        (br_if $break_sum (i32.ge_u (local.get $i) (local.get $simd_end)))
        (local.set $offset (i32.add (local.get $ptr) (i32.shl (local.get $i) (i32.const 2))))
        (local.set $sum_vec
          (f32x4.add
            (local.get $sum_vec)
            (f32x4.mul
              (v128.load (local.get $offset))
              (v128.load (local.get $offset))
            )
          )
        )
        (local.set $i (i32.add (local.get $i) (i32.const 4)))
        (br $loop_sum)
      )
    )

    ;; Horizontal sum of SIMD lanes
    (local.set $sum
      (f32.add
        (f32.add
          (f32x4.extract_lane 0 (local.get $sum_vec))
          (f32x4.extract_lane 1 (local.get $sum_vec))
        )
        (f32.add
          (f32x4.extract_lane 2 (local.get $sum_vec))
          (f32x4.extract_lane 3 (local.get $sum_vec))
        )
      )
    )

    ;; Handle remainder elements (dim % 4)
    (local.set $remainder (local.get $simd_end))
    (block $break_rem_sum
      (loop $loop_rem_sum
        (br_if $break_rem_sum (i32.ge_u (local.get $remainder) (local.get $dim)))
        (local.set $offset (i32.add (local.get $ptr) (i32.shl (local.get $remainder) (i32.const 2))))
        (local.set $sum
          (f32.add
            (local.get $sum)
            (f32.mul
              (f32.load (local.get $offset))
              (f32.load (local.get $offset))
            )
          )
        )
        (local.set $remainder (i32.add (local.get $remainder) (i32.const 1)))
        (br $loop_rem_sum)
      )
    )

    ;; Compute magnitude and check for zero
    (local.set $mag (f32.sqrt (local.get $sum)))
    (if (f32.eq (local.get $mag) (f32.const 0))
      (then (return))
    )

    ;; Phase 2: Divide each element by magnitude using SIMD
    (local.set $inv_mag (f32.div (f32.const 1) (local.get $mag)))
    (local.set $inv_vec (f32x4.splat (local.get $inv_mag)))
    (local.set $i (i32.const 0))

    (block $break_norm
      (loop $loop_norm
        (br_if $break_norm (i32.ge_u (local.get $i) (local.get $simd_end)))
        (local.set $offset (i32.add (local.get $ptr) (i32.shl (local.get $i) (i32.const 2))))
        (v128.store
          (local.get $offset)
          (f32x4.mul
            (v128.load (local.get $offset))
            (local.get $inv_vec)
          )
        )
        (local.set $i (i32.add (local.get $i) (i32.const 4)))
        (br $loop_norm)
      )
    )

    ;; Handle remainder elements
    (local.set $remainder (local.get $simd_end))
    (block $break_rem_norm
      (loop $loop_rem_norm
        (br_if $break_rem_norm (i32.ge_u (local.get $remainder) (local.get $dim)))
        (local.set $offset (i32.add (local.get $ptr) (i32.shl (local.get $remainder) (i32.const 2))))
        (f32.store
          (local.get $offset)
          (f32.mul
            (f32.load (local.get $offset))
            (local.get $inv_mag)
          )
        )
        (local.set $remainder (i32.add (local.get $remainder) (i32.const 1)))
        (br $loop_rem_norm)
      )
    )
  )

  ;; search_all(query_ptr: i32, db_ptr: i32, scores_ptr: i32, db_size: i32, dimensions: i32)
  ;; Computes dot products of query against every vector in the database.
  ;; Uses 128-bit SIMD for 4-wide f32 multiply-accumulate.
  (func (export "search_all") (param $query_ptr i32) (param $db_ptr i32) (param $scores_ptr i32) (param $db_size i32) (param $dim i32)
    (local $i i32)
    (local $j i32)
    (local $acc v128)
    (local $dot f32)
    (local $vec_ptr i32)
    (local $simd_end i32)
    (local $remainder i32)
    (local $q_offset i32)
    (local $v_offset i32)
    (local $bytes_per_vec i32)

    (local.set $simd_end (i32.and (local.get $dim) (i32.const -4)))
    (local.set $bytes_per_vec (i32.shl (local.get $dim) (i32.const 2)))
    (local.set $i (i32.const 0))

    (block $break_outer
      (loop $loop_outer
        (br_if $break_outer (i32.ge_u (local.get $i) (local.get $db_size)))

        ;; Pointer to the i-th database vector
        (local.set $vec_ptr
          (i32.add (local.get $db_ptr) (i32.mul (local.get $i) (local.get $bytes_per_vec)))
        )

        ;; SIMD dot product accumulator
        (local.set $acc (v128.const f32x4 0 0 0 0))
        (local.set $j (i32.const 0))

        (block $break_inner
          (loop $loop_inner
            (br_if $break_inner (i32.ge_u (local.get $j) (local.get $simd_end)))
            (local.set $q_offset (i32.add (local.get $query_ptr) (i32.shl (local.get $j) (i32.const 2))))
            (local.set $v_offset (i32.add (local.get $vec_ptr) (i32.shl (local.get $j) (i32.const 2))))
            (local.set $acc
              (f32x4.add
                (local.get $acc)
                (f32x4.mul
                  (v128.load (local.get $q_offset))
                  (v128.load (local.get $v_offset))
                )
              )
            )
            (local.set $j (i32.add (local.get $j) (i32.const 4)))
            (br $loop_inner)
          )
        )

        ;; Horizontal sum of SIMD accumulator
        (local.set $dot
          (f32.add
            (f32.add
              (f32x4.extract_lane 0 (local.get $acc))
              (f32x4.extract_lane 1 (local.get $acc))
            )
            (f32.add
              (f32x4.extract_lane 2 (local.get $acc))
              (f32x4.extract_lane 3 (local.get $acc))
            )
          )
        )

        ;; Handle remainder elements (dim % 4)
        (local.set $remainder (local.get $simd_end))
        (block $break_rem
          (loop $loop_rem
            (br_if $break_rem (i32.ge_u (local.get $remainder) (local.get $dim)))
            (local.set $q_offset (i32.add (local.get $query_ptr) (i32.shl (local.get $remainder) (i32.const 2))))
            (local.set $v_offset (i32.add (local.get $vec_ptr) (i32.shl (local.get $remainder) (i32.const 2))))
            (local.set $dot
              (f32.add
                (local.get $dot)
                (f32.mul
                  (f32.load (local.get $q_offset))
                  (f32.load (local.get $v_offset))
                )
              )
            )
            (local.set $remainder (i32.add (local.get $remainder) (i32.const 1)))
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
