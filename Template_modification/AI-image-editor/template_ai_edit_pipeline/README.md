# Template AI Edit Pipeline

Denne pakken gjør to ting:

1. `01_make_template_json_and_mask.py`
   - Leser `image.png` og `Prob Done v2.png`
   - Bruker diff/OpenCV til å finne boksene
   - Bruker din classifier-modell til å finne label/type
   - Lager:
     - `template_boxes.json`
     - `template_mask.png`
     - `debug_template_boxes.png`
     - `debug_grouped_mask.png`

2. `02_ai_inpaint_template.py`
   - Leser `Prob Done v2.png`
   - Leser `template_mask.png`
   - Leser `template_boxes.json`
   - Bruker Stable Diffusion Inpainting til å redigere bare maskerte områder
   - Lager:
     - `ai_edited_output.png`

## Filer du må legge i samme mappe som scripts

- `image.png`
- `Prob Done v2.png`
- `box_type_position_missing_classifier.pth`

## Installer pakker

```bash
pip install pillow numpy opencv-python torch torchvision diffusers transformers accelerate
```

## Kjør

Først:

```bash
python 01_make_template_json_and_mask.py
```

Så:

```bash
python 02_ai_inpaint_template.py
```

## Viktig

Stable Diffusion Inpainting er ekte AI image editing, men den er ikke alltid god på eksakt tekst.
Hvis du må ha nøyaktig tekst som `AB123456`, bør du senere bytte editor-delen til AnyText, AnyText2 eller TextDiffuser.

Din eksisterende classifier er fortsatt nyttig. Den finner hva slags type felt hver boks er.
