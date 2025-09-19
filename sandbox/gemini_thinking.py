import asyncio

import d20
import google.genai.types as gai

from kani import Kani, ai_function, print_stream
from kani.engines.google import GoogleAIEngine

DICE_SYNTAX_PROMPT = """
## Dice Syntax

This is the grammar supported by the dice parser, roughly ordered in how tightly the grammar binds.

### Numbers

These are the atoms used at the base of the syntax tree.

| Name    | Syntax                           | Description           | Examples                       |
|---------|----------------------------------|-----------------------|--------------------------------|
| literal | `INT`, `DECIMAL`                 | A literal number.     | `1`, `0.5`, `3.14`             |
| dice    | `INT? "d" (INT \| "%")`          | A set of die.         | `d20`, `3d6`                   |
| set     | `"(" (num ("," num)* ","?)? ")"` | A set of expressions. | `()`, `(2,)`, `(1, 3+3, 1d20)` |

Note that `(3d6)` is equivalent to `3d6`, but `(3d6,)` is the set containing the one element `3d6`.

### Set Operations

These operations can be performed on dice and sets.

#### Grammar

| Name     | Syntax               | Description                        | Examples        |
|----------|----------------------|------------------------------------|-----------------|
| set_op   | `operation selector` | An operation on a set (see below). | `kh3`, `ro<3`   |
| selector | `seltype INT`        | A selection on a set (see below).  | `3`, `h1`, `>2` |

#### Operators

Operators are always followed by a selector, and operate on the items in the set that match the selector.

| Syntax | Name           | Description                                                                  |
|--------|----------------|------------------------------------------------------------------------------|
| k      | keep           | Keeps all matched values.                                                    |
| p      | drop           | Drops all matched values.                                                    |
| rr     | reroll         | Rerolls all matched values until none match. (Dice only)                     |
| ro     | reroll once    | Rerolls all matched values once. (Dice only)                                 |
| ra     | reroll and add | Rerolls up to one matched value once, keeping the original roll. (Dice only) |
| e      | explode on     | Rolls another die for each matched value. (Dice only)                        |
| mi     | minimum        | Sets the minimum value of each die. (Dice only)                              |
| ma     | maximum        | Sets the maximum value of each die. (Dice only)                              |

#### Selectors

Selectors select from the remaining kept values in a set.

| Syntax | Name           | Description                                           |
|--------|----------------|-------------------------------------------------------|
| X      | literal        | All values in this set that are literally this value. |
| hX     | highest X      | The highest X values in the set.                      |
| lX     | lowest X       | The lowest X values in the set.                       |
| \>X    | greater than X | All values in this set greater than X.                |
| <X     | less than X    | All values in this set less than X.                   |

### Unary Operations

| Syntax | Name     | Description              |
|--------|----------|--------------------------|
| +X     | positive | Does nothing.            |
| -X     | negative | The negative value of X. |

### Binary Operations

| Syntax | Name           |
|--------|----------------|
| X * Y  | multiplication |
| X / Y  | division       |
| X // Y | int division   |
| X % Y  | modulo         |
| X + Y  | addition       |
| X - Y  | subtraction    |
| X == Y | equality       |
| X >= Y | greater/equal  |
| X <= Y | less/equal     |
| X > Y  | greater than   |
| X < Y  | less than      |
| X != Y | inequality     |

### Examples

```pycon
>>> from d20 import roll
>>> r = roll("4d6kh3")  # highest 3 of 4 6-sided dice
>>> r.total
14
>>> str(r)
'4d6kh3 (4, 4, **6**, ~~3~~) = `14`'

>>> r = roll("2d6ro<3")  # roll 2d6s, then reroll any 1s or 2s once
>>> r.total
9
>>> str(r)
'2d6ro<3 (**~~1~~**, 3, **6**) = `9`'

>>> r = roll("8d6mi2")  # roll 8d6s, with each die having a minimum roll of 2
>>> r.total
33
>>> str(r)
'8d6mi2 (1 -> 2, **6**, 4, 2, **6**, 2, 5, **6**) = `33`'

>>> r = roll("(1d4 + 1, 3, 2d6kl1)kh1")  # the highest of 1d4+1, 3, and the lower of 2 d6s
>>> r.total
3
>>> str(r)
'(1d4 (2) + 1, ~~3~~, ~~2d6kl1 (2, 5)~~)kh1 = `3`'
```
"""


class DiceKani(Kani):
    @ai_function()
    def roll(self, dice: str):
        """
        Roll some dice.
        """
        return str(d20.roll(dice))


async def main():
    engine = GoogleAIEngine(
        model="gemini-2.5-flash", thinking_config=gai.ThinkingConfig(thinking_budget=-1, include_thoughts=True)
    )
    ai = DiceKani(engine, system_prompt=DICE_SYNTAX_PROMPT)

    async for stream in ai.full_round_stream("Please roll me stats for a D&D character."):
        await print_stream(stream)
        print(await stream.message())


if __name__ == "__main__":
    asyncio.run(main())
