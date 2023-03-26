# NativeSARSOP

[![CI](https://github.com/JuliaPOMDP/NativeSARSOP.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/JuliaPOMDP/NativeSARSOP.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/JuliaPOMDP/NativeSARSOP.jl/branch/main/graph/badge.svg?token=sBUIhwe27n)](https://codecov.io/gh/JuliaPOMDP/NativeSARSOP.jl)

NativeSARSOP is a native julia implementation of the [SARSOP POMDP algorithm](http://www.roboticsproceedings.org/rss04/p9.pdf). It has comparable speed to the [wrapped C++ solver](https://github.com/JuliaPOMDP/SARSOP.jl), but avoids the bottleneck of writing to a pomdpx file, so it can often find a result in less total time.

## Installation

It is recommended that you have [POMDPs.jl](https://github.com/JuliaPOMDP/POMDPs.jl) installed. To install SARSOP run the following command:

```julia
] add NativeSARSOP
```

## Example Usage
```julia
using POMDPs
using NativeSARSOP
using POMDPModels

pomdp = TigerPOMDP()
solver = SARSOPSolver()
policy = solve(solver, pomdp)
```
