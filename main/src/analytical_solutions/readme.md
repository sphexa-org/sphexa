# Usage

## Setup

```
pip install -r requirements.txt
```

## evrard

```
rm -f rm dump_evrard.h5
./sphexa-cuda --glass 40c.h5 --init evrard -n 100 -s 0.77 -w 0.77 -f x,y,z,vx,vy,vz,rho,p --quiet

python3 ./compare_evrard.py --help
python3 ./compare_evrard.py -t 0.77 dump_evrard.h5
# evrard_density_0.770041.png
# evrard_pressure_0.770041.png
# evrard_velocity_0.770041.png
```

## gresho_chan

```
rm -f dump_gresho-chan.h5
./sphexa-cuda --glass 40c.h5 --init gresho-chan -n 40 -s 20 -w 10 -f x,y,z,vx,vy,vz,rho,p,temp,h --quiet
# Total execution time of 20 iterations of gresho-chan up to t = 0.000006: 4.37082s

python3 ./compare_gresho_chan.py --help
python3 ./compare_gresho_chan.py -t 0.000006 -r1 0.2 dump_gresho-chan.h5
# greshochan_colourmap_0.000.png
# greshochan_velocity_0.000.png
```

## noh

```
rm -f rm dump_noh.h5
./sphexa-cuda --glass 40c.h5 --init noh -n 100 -s 0.018 -w 0.018 -f x,y,z,vx,vy,vz,rho,p,temp --quiet

python3 ./compare_noh.py --help
python3 ./compare_noh.py -t 0.018 dump_noh.h5
# noh_pressure_0.018227.png
# noh_density_0.018227.png
# noh_velocity_0.018227.png
# noh_energy_0.018227.png
```
