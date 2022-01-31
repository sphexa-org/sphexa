#  SITUACION DE LAS VARIABLES EN EL FICHERO DE DATOS
####################################################
#
#  1: radio al centro de masas (r) 
#  2: Density (rho)
#  3: Energy (U)
#  4: Pressure (P)
#  5: velocidad (v)
#
####################################################

set size ratio 1

set autoscale x
set autoscale y

set term pngcairo dashed enhanced font 'Verdana,14'


# color definitions
set style line 1 lc rgb '#ff0000' pt 2  ps 1 lt 1 lw 2 # --- red
set style line 2 lc rgb '#00ff00' pt 5  ps 1 lt 1 lw 2 # --- green
set style line 3 lc rgb '#0000ff' pt 7  ps 1 lt 1 lw 2 # --- blue
set style line 4 lc rgb '#ffff00' pt 9  ps 1 lt 1 lw 2 # --- yellow
set style line 5 lc rgb '#ff00ff' pt 11 ps 1 lt 1 lw 2 # --- magenta
set style line 6 lc rgb '#00ffff' pt 13 ps 1 lt 1 lw 2 # --- cyan
set style line 7 lc rgb '#000000' pt 2  ps 1 lt 1 lw 2 # --- black

set style line 12 lc rgb '#808080' lt 0 lw 1
set grid back ls 12


solSedov="theoretical.dat"

######################################################################################################

 set xlabel "R (cm)"
 set ylabel "{/Symbol r} (g.cm^{-2})"
 unset title
 set key right top
 set output 'density.png'

 plot solSedov  u 1:2    w l ls 3 title "{/Symbol r}"

 set autoscale x
 set autoscale y
 unset format 

######################################################################################################

 set xlabel "R (cm)"
 set ylabel "U (erg.g^{-1})"
 unset title
 set key right top
 set output 'energy.png'

 plot solSedov u 1:3 w l ls 1 title "U"

 set autoscale x
 set autoscale y
 unset format 

######################################################################################################

 set xlabel "R (cm)"
 set ylabel "P (erg.cm^{-2})"
 unset title
 set key right top
 set output 'pressure.png'

 plot solSedov u 1:4 w l ls 1 title "P"

 set autoscale x
 set autoscale y
 unset format 

######################################################################################################

 set xlabel "R (cm)"
 set ylabel "V (cm.s^{-1})"
 unset title
 set key right top
 set output 'velocity.png'

 plot solSedov u 1:5 w l ls 1 title "Velocity"

 set autoscale x
 set autoscale y
 unset format 

######################################################################################################

set xlabel "R (cm)"
set ylabel  "{/Symbol r} (g.cm^{-2}); U (erg.g^{-1}); P (erg.cm^{-2}); V (cm.s^{-1})"
unset title
set key right top
set output 'combined.png'

plot solSedov  u 1:2    w l ls 1 title "{/Symbol r}", \
     solSedov  u 1:3    w l ls 2 title "Energy",      \
     solSedov  u 1:4    w l ls 3 title "Pressure",    \
     solSedov  u 1:5    w l ls 5 title "Velocity"

set autoscale x
set autoscale y
unset format 

######################################################################################################