# TODO

- [x] Bounds initialization
  - [x] Upper bound
    - [x] Fast Informed Bound
    - [x] $V_{MDP}$
    - [x] SAWTOOTH
  - [x] Lower Bound (BAWS)
- [x] Bounds Updates
  - [x] Upper Bound
  - [x] Lower bound (in `backup!`)
- [x] Pruning
- [ ] Testing
  - [ ] Sample
  - [ ] Backup
  - [ ] Upper Bound Update
  - [ ] Prune


# PLAN

- [x] Tyler debugs tree_backup (changed to `backup!`)
- [ ] Ben debugs sample (Apparently this may not have been broken)
  - Only terminates because of ad hoc `max_steps` termination condition not present in actual algorithm. Still needs debugging.
- [ ] Ben starts tree_backup debug
- [ ] Tyler and Ben debug sample + tree_backup


# DEADLINE

- SARSOP.jl fully running by 1 Oct and providing reasonable outputs
- SARSOP.jl benchmarked with SARSOP cpp by 15 Oct
