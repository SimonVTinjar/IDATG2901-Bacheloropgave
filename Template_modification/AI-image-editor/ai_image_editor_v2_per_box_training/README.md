# AI Image Editor v2: Per-box editing + enkel trenbar modell

Denne pakken er laget for prosjektet ditt der template-bildet er stort, for eksempel 7200x2983.
I stedet for å sende hele bildet til Stable Diffusion, redigerer vi én boks om gangen.

## Filer

### 03_ai_inpaint_per_box.py
Bruker:
- `Prob Done v2.png`
- `template_boxes.json`

Og lager:
- `ai_edited_output_per_box.png`
- debug-crops i `debug_ai_crops/`

Dette bruker Stable Diffusion Inpainting per boks.
Det er ekte AI-redigering, men vanlig Stable Diffusion er ikke perfekt på eksakt tekst.

### 04_build_training_crops.py
Lager treningsdata fra mange bilder.

Forventet struktur:

```text
data/
  originals/
    abc.jpg
  templates/
    abc.png
  annotations/
    abc.json
```

JSON-formatet kan være slik:

```json
{
  "image_size": {"width": 444, "height": 191},
  "boxes": [
    {"label": "serial", "bbox": [19, 42, 122, 55]}
  ]
}
```

Hvis boksen har `"value"`, brukes den.
Hvis ikke, brukes en standard testverdi basert på label.

Output:
```text
training_data/
  input/
  target/
  mask/
  text_guide/
  metadata.jsonl
```

### 05_train_local_editor.py
Trener en enkel U-Net-modell.

Input til modellen:
- RGB template crop
- mask
- text guide

Output:
- target crop

Dette er en liten forsknings/prototype-modell, ikke en stor diffusion-modell.

### 06_infer_local_editor.py
Bruker modellen fra `05_train_local_editor.py` på template-bildet.

Input:
- `Prob Done v2.png`
- `template_boxes.json`
- `local_box_editor_unet.pth`

Output:
- `local_model_edited_output.png`

## Installer

```bash
python -m pip install pillow numpy opencv-python torch torchvision diffusers transformers accelerate
```

Hvis du får problemer med Python 3.14, bruk Python 3.11.

## Anbefalt kjørerekkefølge

Først må du ha kjørt script 01 fra forrige pakke, slik at du har:

```text
template_boxes.json
```

Så kan du teste per-box AI editing:

```bash
python 03_ai_inpaint_per_box.py
```

For å trene lokal modell:

```bash
python 04_build_training_crops.py
python 05_train_local_editor.py
python 06_infer_local_editor.py
```

## Viktig

Den lokale U-Net-modellen er enkel. Den kan lære stil fra datasettet ditt, men trenger nok data.
Med 50 bilder og ca. 8 bokser per bilde får du rundt 400 crops. Det er nok til prototype, men ikke en perfekt modell.

For best mulig eksakt tekst i bilder er neste steg senere:
- AnyText
- AnyText2
- TextDiffuser
