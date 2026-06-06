# Nested loops (shootout: nestedloop).  n*n iterations, each adding 1 -> n*n;
# a pure call-assembler / tail-loop stress with no allocation.
let rec inner j n acc = if j < n then inner (j + 1) n (acc + 1) else acc ;;
let rec outer i n acc = if i < n then outer (i + 1) n (inner 0 n acc) else acc ;;
let rec main n = outer 0 n 0 ;;
