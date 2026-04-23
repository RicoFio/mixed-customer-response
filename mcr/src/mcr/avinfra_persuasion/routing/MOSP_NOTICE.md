# MOSP Wrapper Notice

The Cython extension in this package wraps the Multiobjective Dijkstra Algorithm
implementation stored in the `mcr/avinfra_persuasion/routing/mosp/` directory.

The upstream license and citation metadata are kept with that source tree:

- `mosp/LICENSE`
- `mosp/CITATION.cff`
- `mosp/README.md`

The wrapper compiles the MOSP sources as an internal extension module and does
not link against `mosp/src/main.cpp`.
