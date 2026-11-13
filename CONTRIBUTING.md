# Contributing

## Build and test

```
make test
```

Binaries land in `build/bin/`.

## Documentation Conventions

### C++

Every file starts with a header comment, and every function, class and struct gets a block comment. Indent with tabs.

```cpp
/**
 * file_name.hpp
 * Author: Miguel Mochizuki Silva
 * Description: One line on what the file holds
 */

/**
 * What the function does, in a sentence
 *
 * Parameters:
 * const SnakeEnv& env: Current game
 * Mode mode: What each value means
 *
 * Returns int: What the result is
 */
int act(const SnakeEnv& env, Mode mode);

/**
 * What the type is for
 *
 * Attributes:
 * int side: Board side in cells
 */
```

A definition in a `.cpp` whose declaration in the header already has the block does not repeat it. Trivial getters need no block. Comments inside functions say why, not what.
