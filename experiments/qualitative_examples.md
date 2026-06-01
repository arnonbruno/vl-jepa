# Qualitative COCO retrieval: VL-JEPA vs CLIP zero-shot

Checkpoint: `experiments/exp_jepa_1024d_20ep/checkpoint_best.pt` vs CLIP ViT-L-14 zero-shot.

## Text -> Image (caption query, rank of the correct image; lower = better)

| caption | zero-shot rank | VL-JEPA rank | Δ |
|---|---|---|---|
| A bunch of plates with food on them on a table. | 124 | 42 | +82 |
| A plate of broccoli, rice, meat and other vegetables | 47 | 30 | +17 |
| Two large elephants waiting to enter their shelter. | 10 | 2 | +8 |
| Two men and woman opening boxes of pizza in a room. | 14 | 7 | +7 |
| A black and white cat relaxing inside a laptop. | 2 | 1 | +1 |
| A group of people sitting around a table with laptops and notebooks. | 3 | 2 | +1 |
| A giraffe standing in a field with a bird. | 1 | 1 | +0 |
| A cat watches cars racing on a television. | 1 | 1 | +0 |
| A man and woman posing for a picture in a sports bar. | 1 | 1 | +0 |
| Many people walk through a park as few kites fly in the air. | 1 | 1 | +0 |
| Cleaning products are on the counter of the bathroom. | 1 | 3 | -2 |
| A hand with scissors cutting into some sheets of paper. | 13 | 18 | -5 |
| there is a surfer at the beach riding waves in the ocean | 21 | 29 | -8 |
| A couple of computer monitors sitting on top of a desk. | 10 | 25 | -15 |
| A person is sitting at a keyboard near a microphone. | 53 | 75 | -22 |
| A yellow sign that is at the top of a pole. | 35 | 78 | -43 |

6/16 sampled captions improve under VL-JEPA, 6 regress (Δ = zero-shot_rank - vljepa_rank).

## Image -> Text (top retrieved captions)

**image 1675** (hit@5: True)
  - A black and white cat relaxing inside a laptop.
  - A cat is sitting in a partially closed laptop.
  - A black and white cat laying across the keyboard of a laptop.
  - A cat is laying on top of a laptop computer.
  - A dark cat hiding in between a laptop.

**image 3501** (hit@5: True)
  - A bowl holds soup with broccoli and other vegetables.
  - A bowl with rice, broccoli and a purple relish.
  - The meal in the bowl has rice and broccoli in it.
  - A bowl of a kind of vegetable stew on a table.
  - A plate of food with rice and beans, broccoli and meat.

**image 6763** (hit@5: True)
  - A man and woman posing for a picture in a sports bar.
  - A man and woman hugging in a restaurant
  - A man and a women posing next to one another in front of a table.
  - A man and a woman posing together for a picture
  - A man standing next to a woman in a gray dress.

**image 34760** (hit@5: False)
  - Blue bathroom with two white towels hanging by the shower.
  - A bathroom with a sink, tub, shower head and mirror.
  - A bathroom with a sink, shower, tub and a cabinet.
  - A bathroom sink, toilet and shower with the curtain half open.
  - A bathroom with a sink, towel rack and shower stall.

**image 92053** (hit@5: False)
  - A bunch of food sitting on a table, with three beers on it.
  - there are two plates of food and a beer in the middle of the table
  - A plate of food and a drink on a table.
  - A plate filled with food sitting next to three glasses.
  - A variety of food on a table with a few drinks.

**image 98497** (hit@5: True)
  - A sign with a person with a surfboard is near a building and palm trees.
  - a yellow sign of a person carrying a surf board
  - A traffic sign that has a picture of a man holding a surfboard on it.
  - A ROAD SIGN INDICATING SURFERS OUTSIDE A TEMPLE
  - There are a lot of surf boards leaning against the building.
