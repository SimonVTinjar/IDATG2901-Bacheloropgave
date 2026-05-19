# LoRA-trening og bildegenerering

Denne mappen inneholder skript brukt til datasettforberedelse, bildegenerering og annotasjon i forbindelse med LoRA-finjustering av Z-Image og Z-Image-Turbo.

---

## Forutsetninger

- Python 3.10+
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) kjørende lokalt på `http://127.0.0.1:8000`
- Relevante Python-pakker: `pip install pillow transformers torch`

---

## Skriptoversikt

### `create_dataset.py`

Forbereder treningsdatasettet ved å skalere og croppe bilder fra `master_images/` til riktig format for LoRA-trening.

- Fullvisningsbilder skaleres til bredde 1024 (beholder aspektforhold)
- Detaljcrops lages som kvadratiske 1024×1024-bilder

**Kjør fra rotmappen:**
```bash
python create_dataset.py
```

---

### `create-captions.py`

Genererer `.txt`-caption-filer for hvert bilde i `lora_dataset/`. Hver caption beskriver bildeinnholdet basert på filnavnets suffiks (f.eks. `_full`, `_portrait`, `_left_pattern`).

**Kjør fra rotmappen:**
```bash
python create-captions.py
```

---

### `make_captions.py`

Alternativt caption-skript som legger inn en fast basecaption for alle bilder i en mappe. Brukt til enklere datasett uten detaljcrops.

Konfigurer `IMAGE_FOLDER` og `BASE_CAPTION` øverst i filen.

```bash
python make_captions.py
```

---

### `blip_captions.py`

Genererer automatiske bildebeskrivelser ved hjelp av Salesforce BLIP-modellen. Kan brukes til å lage prompts basert på innholdet i eksisterende seddelbilder.

```bash
python blip_captions.py <bildefil>
python blip_captions.py <mappe>   # behandler alle bilder i mappen
```

---

### `generate_batch.py`

Batch-genererer 500 seddelbilder via ComfyUI API med Z-Image-Turbo uten LoRA. Brukt til å lage det AI-genererte datasettet beskrevet i rapporten.

Konfigurer `NUM_IMAGES`, `START_SEED` og `PROMPT_TEXT` øverst i filen.

**Krav:** ComfyUI må kjøre på `http://127.0.0.1:8000`

```bash
python generate_batch.py
```

---

### `generate_batch_lora_t2_t4.py`

Batch-genererer 200 bilder via ComfyUI API med en innlastet LoRA-checkpoint. Brukt for trening 2 (USD, Z-Image-Turbo, steg 750) og trening 4 (Thai Baht, Z-Image-Turbo, steg 2000).

Kommenter inn/ut riktig `LORA_NAME` og `OUTPUT_PREFIX` øverst i filen.

```bash
python generate_batch_lora_t2_t4.py
```

---

### `generate_batch_lora_t3.py`

Batch-genererer 200 bilder med LoRA-checkpoint fra trening 3 (USD, Z-Image Base, steg 2000).

```bash
python generate_batch_lora_t3.py
```

---

### `rescale_images.py`

Skalerer bilder til 1344×576 PNG for bruk i QuickEval-brukertesten. Leser fra `rescale_bilder/` og lagrer til `normalized_bilder/`.

```bash
python rescale_images.py
```

---

### `guilloche.py`

Genererer guilloche-mønstre (spirograf-baserte kurver) som linjeart ved hjelp av matplotlib. Brukt til å lage illustrasjoner av seddelens ornamentale mønstre.

```bash
python guilloche.py
```

---

## Konfigurasjonsfiler

### `config.json`

ComfyUI workflow-konfigurasjon brukt av batch-genereringsskriptene.

---

## Relaterte filer

- LoRA YAML-konfigurasjonsfiler ligger i appendix i rapporten
- SLURM-jobskript for IDUN-klyngen ligger i appendix i rapporten
