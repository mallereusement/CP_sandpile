## Computational Physics WS 23/24

### Project: Sandpile

#### Project Description

To Do

#### How does the code work

The code is essentially divided into two parts. Firstly, the simulation. Secondly, the analysis of the simulated data. The parameters of a simulation can be specified in a `.txt` file.

```apache
- setting00
    - name: 2d-non_conservative-N40-abs_True_z4-closed
    - dimension:   2
    - size of grid: 40
    - crititcal value of z: 4
    - pertubation mechanism: non_conservative
    - boundary condition: closed
    - use absolute value: True
    - number of activated avalanches: 100000
    - maximum time steps: 20000000
    - track avalanches after steady state: False
    - steady state: 10000
    - save file for power spectrum calculation: True
    - save file for exponent calculation: True
    - save mean value of grid: True

- setting01
    - name: 2d-non_conservative-N80-abs_True_z4-closed
    - dimension:   2
    - size of grid: 80
    - crititcal value of z: 4
    - pertubation mechanism: non_conservative
    - boundary condition: closed
    - use absolute value: True
    - number of activated avalanches: 100000
    - maximum time steps: 20000000
    - track avalanches after steady state: False
    - steady state: 10000
    - save file for power spectrum calculation: True
    - save file for exponent calculation: True
    - save mean value of grid: True
```

#### ToDo

- Power Spectrum
- make qq plot

#### Ideas

- initialize specific starting conditions
- ability to make a shape in three dimensions and translate it into z-grid
- what happens if sand is not just added random but follows a specific PDF?
- intialize shape and look how long it takes to get detroyed by added sand
- make animations of avalanches
-

#### Questions

- Power spectrum
- heatmap: avalanches only at boundarys of grid?
