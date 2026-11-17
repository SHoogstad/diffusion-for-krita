# Country looks

With **Settings → Interface → Look: Country**, the plugin takes its colors from a country's flag
and draws that flag small (in the tab strip, on the Generate button, in the halftone). Each
country is one JSON file in this folder, named after its two letter
[ISO 3166-1 code](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2) in capitals: `NL.json`.

## Add or change a country for yourself

1. In **Settings → Interface**, click the folder button next to **Country**. It opens the
   `countries` folder in the plugin's user data folder.
2. Copy a file from here that looks like your flag (`NL.json` for three stripes, `SE.json` for a
   Nordic cross, `JP.json` for a disc) and rename it to your country code.
3. Edit it, then click the refresh button next to **Country**. Your file shows up in the list.

A file with the code of a built-in country replaces that one. A file with a mistake is skipped,
and the reason is in `client.log` in the plugin's log folder, for example
`Country look skipped, .../XK.json: flag, shape 2 (star): r is missing`.

## Add a country to the plugin (pull request)

1. Add `ai_diffusion/countries/<CODE>.json`.
2. Run `python scripts/flag_preview.py <CODE>`. It checks the file and writes
   `flag_preview.png`: the flag large, at the 24 × 16 and 16 × 11 pixels the plugin draws it,
   and the interface colors. Make sure the small sizes still read as the flag.
3. Open a pull request with the file, and put the preview image in the description.

No Python code needs to change. `python scripts/flag_preview.py` without arguments checks every
file, and `tests/test_country_files.py` does the same in CI.

Can't or don't want to make the file? Open a **Flag request** issue instead.

## Format

```json
{
  "name": "Netherlands",
  "bands": ["#ae1c28", "#ffffff", "#21468b"],
  "accent": "#ae1c28",
  "cool": "#21468b",
  "flag": [
    {"shape": "stripes", "colors": ["#ae1c28", "#ffffff", "#21468b"]}
  ]
}
```

| Field | |
|---|---|
| `name` | The name in the country list (English). |
| `bands` | Two or three flag colors, top to bottom. They make the striped bars, the halftone and the logo strokes. |
| `accent` | Color of the main action, the Generate button. Pick a strong flag color. |
| `cool` | Optional. Color for what is switched on, focused or selected. Leave it out to use a darker `accent`. |
| `flag` | The shapes that draw the flag, in order: later shapes paint over earlier ones. |
| `comment` | Optional, ignored. For notes, eg. where the colors come from. |

Colors are `"#rrggbb"`. Keep the flag simple: it has to read at 16 pixels, so leave out coats
of arms and text, and use a plain disc or star where the real flag has an emblem.

### Shapes

Positions are fractions of the flag: `x` and `y` from the top left corner (0 to 1), `w` and `h`
of its width and height. A radius `r` is a fraction of the flag's **height**. Fields with a
default can be left out.

| `shape` | Fields (default) | Draws |
|---|---|---|
| `stripes` | `colors`, `weights` (equal), `vertical` (false) | Bands across the whole flag. One color fills it. `weights: [1, 2, 1]` makes the middle band twice as wide. |
| `rect` | `color`, `x` (0), `y` (0), `w` (1), `h` (1) | A rectangle: a canton, a cross arm. |
| `star` | `color`, `x` (0.5), `y` (0.5), `r` | A five pointed star. |
| `disc` | `color`, `x` (0.5), `y` (0.5), `r` | A filled circle. |
| `ring` | `color`, `x` (0.5), `y` (0.5), `r` | A circle outline. |
| `crescent` | `color`, `x` (0.5), `y` (0.5), `r` | A crescent, open to the right. |
| `nordic-cross` | `color`, `thickness`, `x` (0.36) | A Nordic cross; `thickness` is a fraction of the height, `x` the centre of the upright. Stack two for a bordered cross (`NO.json`). |
| `triangle` | `color`, `depth` | A triangle from the hoist (left edge), `depth` of the width. |
| `diamond` | `color` | A diamond touching near the edges (`BR.json`). |
| `union-jack` | `x` (0), `y` (0), `w` (1), `h` (1) | The Union Jack in that area (`AU.json`). |
| `taegeuk` | `top`, `bottom`, `r` | A circle split in two colors (`KR.json`). |
| `star-grid` | `color`, `x` (0), `y` (0), `w` (1), `h` (1), `columns`, `rows` | Staggered small stars as dots (`US.json`). |

Need a shape that is not here? Ask in the pull request or the issue; shapes live in
`ai_diffusion/ui/flags.py` (`shapes` and `paint_flag`).
