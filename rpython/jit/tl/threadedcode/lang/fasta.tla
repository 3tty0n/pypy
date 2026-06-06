# FASTA-style numeric generator.  TLA has no strings/stdout, so this emits the
# same LCG-shaped nucleotide stream into an encoded array and returns a checksum.
let rec fill a i n seed acc =
  if i >= n then acc
  else
    let seed = (seed * 3877 + 29573) % 139968 in
    let base = seed % 4 in
    fill (a[i] <- base) (i + 1) n seed (acc + base * (i + 1))
;;
let rec main n = fill (array n 0) 0 n 42 0 ;;
