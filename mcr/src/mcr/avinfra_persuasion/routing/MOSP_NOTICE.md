# MOD MOSP Wrapper Notice

The Cython extension in this package wraps the Multiobjective Dijkstra Algorithm
implementation stored in the repository-level `mod/` directory.

The upstream license and citation metadata are kept with that source tree:

- `mod/LICENSE`
- `mod/CITATION.cff`
- `mod/README.md`

The wrapper compiles the MOD sources as an internal extension module and does
not link against `mod/src/main.cpp`.
