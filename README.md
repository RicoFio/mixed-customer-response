# mixed-customer-response

## Installing BenPy
If you run into issues with the multi-objective solver on MacOS, try this:
```bash
# Ensure that you installed GLPK first, i.e.
# brew install glpk
GLPK_PREFIX="$(brew --prefix glpk)"

export PATH="$GLPK_PREFIX/bin:$PATH"
export CFLAGS="-I$GLPK_PREFIX/include"
export CPPFLAGS="-I$GLPK_PREFIX/include"
export LDFLAGS="-L$GLPK_PREFIX/lib -Wl,-rpath,$GLPK_PREFIX/lib"

# uv add git+https://github.com/markobud/benpy@development
uv sync
```
