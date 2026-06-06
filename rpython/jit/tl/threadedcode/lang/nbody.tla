# N-body kernel over the usual five bodies.  Positions, velocities and masses
# are stored in float arrays; main n advances n steps and returns scaled energy.
let rec set5 a a0 a1 a2 a3 a4 =
  let a = a[0] <- a0 in
  let a = a[1] <- a1 in
  let a = a[2] <- a2 in
  let a = a[3] <- a3 in
  a[4] <- a4
;;
let rec init_x z = set5 (array 5 (tofloat 0)) (tofloat 0) (tofloat 4) (tofloat 8) (tofloat 12) (tofloat 16) ;;
let rec init_y z = set5 (array 5 (tofloat 0)) (tofloat 0) (tofloat 3) (tofloat 6) (tofloat 9) (tofloat 12) ;;
let rec init_z z = set5 (array 5 (tofloat 0)) (tofloat 0) (tofloat 1) (tofloat 2) (tofloat 3) (tofloat 4) ;;
let rec init_vx z = set5 (array 5 (tofloat 0)) (tofloat 0) (tofloat 0) (tofloat 0) (tofloat 0) (tofloat 0) ;;
let rec init_vy z = set5 (array 5 (tofloat 0)) (tofloat 0) (tofloat 0) (tofloat 0) (tofloat 0) (tofloat 0) ;;
let rec init_vz z = set5 (array 5 (tofloat 0)) (tofloat 0) (tofloat 0) (tofloat 0) (tofloat 0) (tofloat 0) ;;
let rec init_m z = set5 (array 5 (tofloat 0)) (tofloat 1000) (tofloat 1) (tofloat 2) (tofloat 3) (tofloat 4) ;;

let rec pair x y z vx vy vz m i j =
  let dx = x[i] - x[j] in
  let dy = y[i] - y[j] in
  let dz = z[i] - z[j] in
  let d2 = dx * dx + dy * dy + dz * dz + (tofloat 1) in
  let mag = (tofloat (1)) / (d2 * sqrt d2) in
  let mi = m[i] * mag in
  let mj = m[j] * mag in
  let vx = vx[i] <- (vx[i] - dx * mj) in
  let vy = vy[i] <- (vy[i] - dy * mj) in
  let vz = vz[i] <- (vz[i] - dz * mj) in
  let vx = vx[j] <- (vx[j] + dx * mi) in
  let vy = vy[j] <- (vy[j] + dy * mi) in
  vz[j] <- (vz[j] + dz * mi)
;;
let rec pairs_j x y z vx vy vz m i j =
  if j >= 5 then vz else pairs_j x y z vx vy (pair x y z vx vy vz m i j) m i (j + 1)
;;
let rec pairs_i x y z vx vy vz m i =
  if i >= 4 then vz else pairs_i x y z vx vy (pairs_j x y z vx vy vz m i (i + 1)) m (i + 1)
;;
let rec move_i x y z vx vy vz i dt =
  if i >= 5 then z
  else
    let x = x[i] <- (x[i] + dt * vx[i]) in
    let y = y[i] <- (y[i] + dt * vy[i]) in
    move_i x y (z[i] <- (z[i] + dt * vz[i])) vx vy vz (i + 1) dt
;;
let rec advance x y z vx vy vz m step n =
  if step >= n then z
  else
    let vz = pairs_i x y z vx vy vz m 0 in
    let z = move_i x y z vx vy vz 0 ((tofloat (1)) / (tofloat (100))) in
    advance x y z vx vy vz m (step + 1) n
;;
let rec kinetic vx vy vz m i acc =
  if i >= 5 then acc
  else kinetic vx vy vz m (i + 1) (acc + ((tofloat (1)) / (tofloat (2))) * m[i] * (vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i]))
;;
let rec potential_j x y z m i j acc =
  if j >= 5 then acc
  else
    let dx = x[i] - x[j] in
    let dy = y[i] - y[j] in
    let dz = z[i] - z[j] in
    potential_j x y z m i (j + 1) (acc - (m[i] * m[j]) / sqrt (dx * dx + dy * dy + dz * dz + (tofloat 1)))
;;
let rec potential_i x y z m i acc =
  if i >= 4 then acc else potential_i x y z m (i + 1) (potential_j x y z m i (i + 1) acc)
;;
let rec main n =
  let x = init_x 0 in
  let y = init_y 0 in
  let z = init_z 0 in
  let vx = init_vx 0 in
  let vy = init_vy 0 in
  let vz = init_vz 0 in
  let m = init_m 0 in
  let z = advance x y z vx vy vz m 0 n in
  toint ((kinetic vx vy vz m 0 (tofloat 0) + potential_i x y z m 0 (tofloat 0)) * (tofloat 1000))
;;
