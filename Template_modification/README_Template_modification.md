# Template Modification

Denne mappen inneholder arbeidet med template-basert redigering og feltrekonstruksjon. Målet med denne delen var å bruke en eksisterende template som grunnlag, finne områder der informasjon mangler, og forsøke å fylle inn eller rekonstruere disse områdene ved hjelp av bildebehandling, klassifikasjon og AI-basert redigering.

Dette er en eksperimentell pipeline. Resultatene ble brukt som teknisk prototype og som grunnlag for diskusjon i bachelorrapporten.

---

## Hovedidé

I stedet for å generere et helt bilde fra bunnen av, tar denne metoden utgangspunkt i en template. Bare bestemte felt skal redigeres, for eksempel:

- serienummer
- kodefelt
- serieår
- signaturfelt
- andre tekstområder

Arbeidsflyten kan forenkles slik:

```text
template
→ finn manglende/redigerbare områder
→ lag bounding boxes
→ klassifiser feltene
→ lagre feltinformasjon i JSON
→ bruk AI-editor til å redigere valgte felt
```

---

## Mappestruktur

```text
Template_modification/
├── 01_template_filling_v1/
├── 02_template_filling_v2/
├── 03_find_missing_fields_cv/
├── 04_ai_locate_fields/
├── 05_ai_image_editor_final/
├── models/
└── outputs/
```

---

## 01_template_filling_v1

Dette er en tidlig prototype for å teste om manglende områder i en template kunne fylles inn.

Denne delen viser første forsøk på å bruke en autoencoder / bildebasert metode for å finne eller rekonstruere feilområder.

Viktige filer:

```text
detect_missing.py
train_autoencoder.py
```

Modellen ligger nå i:

```text
models/autoencoders/template_autoencoder_v1.pth
```

Eksempel på kjøring:

```powershell
python Template_modification/01_template_filling_v1/detect_missing.py
```

Merk: Denne delen er en tidlig prototype og ikke sluttmetoden.

---

## 02_template_filling_v2

Dette er en forbedret versjon av første innfyllingsforsøk. Her ble arbeidet mer strukturert, spesielt rundt bruk av bokser og felt.

Denne delen ble brukt for å teste bedre kontroll over hvilke områder som skulle behandles videre.

Eksempel på relevant fil:

```text
render_fields_from_json_test.py
```

---

## 03_find_missing_fields_cv

Denne mappen handler om å finne manglende områder ved hjelp av klassisk computer vision.

Metoden sammenligner et originalbilde med en template. Forskjellen mellom bildene brukes til å finne områder der informasjon mangler.

Differanseformelen som ble brukt i rapporten:

```text
D(x, y) = |I_original(x, y) - I_template(x, y)|
```

Typiske steg:

```text
originalbilde + template
→ differansebilde
→ terskling
→ maske
→ filtrering av støy
→ bounding boxes
```

Viktige filer:

```text
check_template_boxes.py
train_box_classifier.py
box_annotator_tkinter.py
make_box_dataset.py
```

U-Net-filene:

```text
train_unet.py
predict_mask.py
```

Disse krever modellen:

```text
models/editors/unet_error_detector.pth
```

Hvis denne modellen ikke finnes, kan ikke `predict_mask.py` kjøres før modellen er trent eller lagt inn.

---

## 04_ai_locate_fields

Denne mappen handler om mer AI-basert lokalisering og klassifikasjon av felter.

Målet var å forbedre filtreringen av kandidatbokser, slik at systemet bedre kunne skille mellom relevante felt og støy.

Viktige filer:

```text
find_diff_boxes.py
make_crops_from_diff.py
train_classifier.py
train_type_with_position.py
detect_with_ai_classifier.py
full_pipeline.py
pipeline_v2.py
full_pipeline_position.py
train_yolo.py
test_yolo.py
convert_json_to_yolo.py
generate_from_fields.py
fill_missing_boxes.py
```

Modeller ligger i:

```text
models/classifiers/
models/yolo/
```

Eksempel på YOLO-test:

```powershell
python Template_modification/04_ai_locate_fields/test_yolo.py
```

`test_yolo.py` finner automatisk nyeste `best.pt` under:

```text
outputs/yolo_runs_archive/runs/detect/
```

---

## 05_ai_image_editor_final

Dette er sluttmetoden i template-delen.

Denne mappen inneholder AI-editor-pipeline-en som bruker template, feltbokser og JSON-data for å forsøke å redigere valgte områder.

Viktige filer:

```text
01_make_template_json_and_mask.py
02_ai_inpaint_template.py
03_ai_inpaint_per_box.py
04_1_build_training_crops_from_originals.py
04_2_build_training_data_with_reference.py
05_1_train_local_editor.py
05_2_train_reference_editor.py
06_1_infer_local_editor.py
06_2_infer_reference_editor.py
json_to_mask.py
manual_label_template_boxes.py
ui_add_values_to_json.py
```

Modeller ligger i:

```text
models/editors/
```

Eksempler:

```text
local_box_editor_unet.pth
local_box_editor_ref_unet.pth
local_box_editor_ref_unet_best_v1.pth
```

---

## JSON-format

Feltene lagres i JSON-format. Hver fil inneholder informasjon om bildet, bildestørrelsen og en liste med bokser.

Eksempel:

```json
{
  "image": "-20USA2010020Dollars20Banknote2C202009A2C20P-536z2C20Used2C20ReplacementStar.jpg",
  "image_size": {
    "width": 1028,
    "height": 433
  },
  "boxes": [
    {
      "label": "serial",
      "bbox": [34, 90, 272, 122],
      "value": "LH08060374 *"
    },
    {
      "label": "code",
      "bbox": [36, 132, 80, 158],
      "value": "H8"
    }
  ]
}
```

Forklaring:

```text
label = hvilken type felt det er
bbox  = koordinatene til feltet
value = teksten eller verdien som skal settes inn
```

Dette gjør at AI-editoren ikke bare vet hvor den skal redigere, men også hva slags felt det er og hvilken verdi som skal brukes.

---

## Models

Alle modellfiler er samlet under:

```text
models/
├── autoencoders/
├── classifiers/
├── editors/
└── yolo/
```

Dette gjør at kildekode og modeller ikke ligger blandet.

Eksempler:

```text
models/autoencoders/template_autoencoder_v1.pth
models/classifiers/box_classifier_v3.pth
models/classifiers/box_type_classifier_v3.pth
models/classifiers/box_type_position_missing_classifier_v3.pth
models/editors/local_box_editor_unet.pth
models/yolo/yolov8n.pt
```

---

## Outputs

Treningsresultater og generert output ligger under:

```text
outputs/
├── yolo_runs_archive/
├── ai_editor_outputs/
└── debug_outputs/
```

YOLO-runs ligger her:

```text
outputs/yolo_runs_archive/runs/
```

---

## Anbefalt kjørerekkefølge

En mulig arbeidsflyt er:

```text
1. Finn eller marker redigerbare felt
2. Lag JSON med label, bbox og value
3. Lag maske fra JSON
4. Kjør AI-editor på valgte felt
5. Evaluer output manuelt
```

Eksempel:

```powershell
python Template_modification/05_ai_image_editor_final/json_to_mask.py
python Template_modification/05_ai_image_editor_final/06_1_infer_local_editor.py
```

Filnavn og rekkefølge kan variere avhengig av hvilken del av eksperimentet som testes.

---

## Status

Denne delen av prosjektet fungerer som en teknisk prototype. Pipeline-en kan finne og strukturere felt, men sluttresultatene fra AI-redigeringen ble ikke gode nok til å regnes som en ferdig løsning.

De viktigste utfordringene var:

- støy i differanseanalyse
- feil eller upresise bounding boxes
- usikker klassifikasjon
- problemer med presis og lesbar tekst i AI-redigering
- visuell inkonsistens mellom generert innhold og template

---
