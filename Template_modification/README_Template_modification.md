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
→ lagre feltinformasjon i JSON
→ bygg reference crops og treningsdata
→ bruk image modification til å redigere valgte felt
```

Det viktigste skillet i prosjektmappen er:

```text
div_methods_to_find_missing_parts_and_make_crops/
→ tidligere forsøk på å finne felt, lage crops og teste lokalisering

image_modification/
→ sluttmetoden for lokal AI-basert redigering

image_modification/Stable Diffusion Inpainting/
→ egne Stable Diffusion-inpainting-forsøk
```

---

## Mappestruktur

```text
Template_modification/
├── div_methods_to_find_missing_parts_and_make_crops/
│   ├── 01_template_filling_v1/
│   ├── 02_template_filling_v2/
│   ├── 03_find_missing_fields_cv/
│   └── 04_locate_fields/
│
├── image_modification/
│   ├── data/
│   ├── generated_results/
│   ├── output/
│   ├── reference_crops/
│   ├── Stable Diffusion Inpainting/
│   │   ├── inpaint_template.py
│   │   └── inpaint_per_box.py
│   ├── training_data_reference/
│   ├── 01_make_template_json_and_mask.py
│   ├── 02_build_reference_crops.py
│   ├── 03_build_training_data_with_reference.py
│   ├── 04_train_reference_editor.py
│   ├── 05_infer_reference_editor.py
│   ├── json_to_mask.py
│   ├── manual_label_template_boxes.py
│   └── ui_add_values_to_json.py
│
├── models/
├── outputs/
└── docs/
```

---

# 1. `div_methods_to_find_missing_parts_and_make_crops`

Denne mappen samler tidligere forsøk og støtteverktøy for å finne manglende eller redigerbare felt i templaten. Den inneholder ikke selve sluttmetoden, men metodene som ble testet før `image_modification`.

Mappen er delt inn kronologisk:

```text
01_template_filling_v1
→ tidlig autoencoder-basert prototype

02_template_filling_v2
→ forbedret template-fylling og tidlig arbeid med bokser

03_find_missing_fields_cv
→ computer vision, differanseanalyse og ResNet18-klassifikasjon

04_locate_fields
→ mer avansert feltlokalisering, YOLO, klassifikasjon og crop-generering
```

---

## 1.1 `01_template_filling_v1`

Dette var en tidlig prototype for å teste om manglende områder i en template kunne finnes eller rekonstrueres.

Denne delen viser første forsøk på å bruke en autoencoder / bildebasert metode for å finne eller rekonstruere feilområder.

Viktige filer:

```text
detect_missing.py
train_autoencoder.py
```

Modellen ligger i:

```text
models/autoencoders/template_autoencoder_v1.pth
```

Eksempel på kjøring:

```powershell
python Template_modification/div_methods_to_find_missing_parts_and_make_crops/01_template_filling_v1/detect_missing.py
```

Merk: Denne delen er en tidlig prototype og ikke sluttmetoden.

---

## 1.2 `02_template_filling_v2`

Dette var en forbedret versjon av de tidlige template-forsøkene. Her ble arbeidet mer strukturert rundt bokser, felt og kontrollert innsetting av innhold.

Eksempel på relevant fil:

```text
render_fields_from_json_test.py
```

Denne delen handler mest om tidlig kontrollert rendering og testing av feltplassering.

---

## 1.3 `03_find_missing_fields_cv`

Denne delen handler om å finne manglende områder med klassisk computer vision.

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

Her ble det også testet ResNet18-basert klassifikasjon av crops.

Viktige typer filer:

```text
box annotation
crop dataset generation
ResNet18 training
template box checking
U-Net mask experiment
```

Merk: U-Net-eksperimentet krever en trent modell, for eksempel `unet_error_detector.pth`. Hvis den modellen ikke finnes, kan ikke prediction-scriptet kjøres uten å trene eller legge inn modellen først.

---

## 1.4 `04_locate_fields`

Denne delen inneholder mer avanserte forsøk på feltlokalisering og klassifikasjon.

Her ble det blant annet testet:

```text
YOLO-basert feltlokalisering
ResNet18-basert klassifikasjon
posisjonsbasert klassifikasjon
crop-generering
JSON til YOLO-format
generering fra feltbeskrivelser
```

Forskjellen mellom ResNet18 og YOLO i dette prosjektet var:

```text
ResNet18
→ klassifiserer crops som allerede er klippet ut

YOLO
→ prøver å finne bounding boxes og klasser direkte i hele bildet
```

YOLO-runs ligger typisk her:

```text
outputs/yolo_runs_archive/runs/
```

YOLO base-modell ligger her:

```text
models/yolo/yolov8n.pt
```

---

# 2. `image_modification`

Dette er sluttmetoden i template-delen.

Denne mappen inneholder den reference-baserte image modification-pipeline-en. Målet er å redigere bestemte felter i templaten ved hjelp av:

```text
template-bilde
JSON med label, bbox og value
reference crops
trent U-Net-basert editor
```

Den praktiske rekkefølgen er:

```text
01_make_template_json_and_mask.py
→ 02_build_reference_crops.py
→ 03_build_training_data_with_reference.py
→ 04_train_reference_editor.py
→ 05_infer_reference_editor.py
```

Hvis JSON, reference crops og modellen allerede finnes, trenger man vanligvis bare å kjøre:

```powershell
python Template_modification/image_modification/05_infer_reference_editor.py
```

---

## 2.1 `01_make_template_json_and_mask.py`

Denne filen lager eller forbereder JSON og maske for templaten.

Den brukes til å lage data som beskriver feltene som skal redigeres.

Eksempel på JSON-felt:

```json
{
  "label": "serial",
  "bbox": [34, 90, 272, 122],
  "value": "LH08060374 *"
}
```

Denne filen trenger ikke kjøres hver gang hvis JSON-filen allerede finnes, men den bør beholdes fordi den kan gjenskape inputdataene.

---

## 2.2 `manual_label_template_boxes.py`

Dette er et manuelt verktøy for å merke bokser på templaten.

Det brukes når automatisk lokalisering ikke gir gode nok bokser.

Typisk output:

```text
template_named_boxes.json
template_numbered_preview.png
```

---

## 2.3 `ui_add_values_to_json.py`

Dette programmet brukes til å legge inn eller endre `value` i JSON-filen.

Eksempel:

```text
label = serial
bbox  = [34, 90, 272, 122]
value = LH08060374 *
```

Dette gjør at brukeren kan bestemme hva som skal stå i hver boks.

---

## 2.4 `json_to_mask.py`

Denne filen lager masker fra JSON-boksene.

Den leser `bbox` fra JSON og lager maskebilder som viser hvilke områder som skal redigeres.

---

## 2.5 `02_build_reference_crops.py`

Denne filen lager reference crops.

Reference crops er små bildeutsnitt som viser hvordan ulike felttyper kan se ut.

Eksempel på struktur:

```text
reference_crops/
├── serial/
├── code/
├── series/
├── signature_1/
└── signature_2/
```

Reference crops brukes av reference-editoren som ekstra visuell kontekst.

---

## 2.6 `03_build_training_data_with_reference.py`

Denne filen lager treningsdata for reference-editoren.

Den lager typisk:

```text
input crop
target crop
mask
text guide
reference crop
metadata
```

Forklaring:

```text
input crop      = området modellen får inn
target crop     = fasiten modellen skal lære å lage
mask            = området modellen skal endre
text guide      = enkel guide basert på ønsket value
reference crop  = eksempel på samme felttype
```

Treningsmålet er:

```text
input crop + mask + text guide + reference crop
        → target crop
```

---

## 2.7 `04_train_reference_editor.py`

Denne filen trener den reference-baserte editoren.

Modellen er en U-Net-lignende image-to-image-modell. Den er ikke Stable Diffusion.

Input har typisk 8 kanaler:

```text
3 kanaler = input RGB crop
1 kanal   = mask
1 kanal   = text guide
3 kanaler = reference crop
```

Totalt:

```text
3 + 1 + 1 + 3 = 8 input-kanaler
```

Output er:

```text
3 RGB-kanaler
```

Modellen lagres typisk i:

```text
models/editors/
```

---

## 2.8 `05_infer_reference_editor.py`

Dette er slutt-scriptet.

Det bruker den trente reference-editoren til å redigere templaten.

Arbeidsflyt:

```text
les template
les JSON med bokser
for hver boks:
    hent label, bbox og value
    klipp ut crop fra template
    lag maske
    lag text guide
    hent reference crop basert på label
    send alt inn i U-Net-editoren
    lim redigert crop tilbake i templaten
lagre output
```

For at dette scriptet skal kjøre, må disse finnes:

```text
template-bilde
template_named_boxes.json
reference_crops/
trent modell
output-mappe
```

---

# 3. `image_modification/Stable Diffusion Inpainting`

Denne mappen inneholder forsøk med Stable Diffusion Inpainting.

Dette var ikke sluttmetoden, men et forsøk på å bruke en ferdigtrent generativ modell til å fylle inn maskerte områder.

Stable Diffusion-forsøkene trente ikke en ny modell. De brukte en ferdigtrent inpainting-modell til inference.

Typiske filer:

```text
inpaint_template.py
inpaint_per_box.py
```

---

## 3.1 `inpaint_template.py`

Denne filen testet inpainting på templaten med en større samlet maske.

Arbeidsflyt:

```text
les template
les maske
lag prompt
kjør Stable Diffusion Inpainting
lagre output
```

Dette forsøket ga rask testing av inpainting, men hadde begrenset kontroll over nøyaktig tekst.

---

## 3.2 `inpaint_per_box.py`

Denne filen testet inpainting én boks om gangen.

Arbeidsflyt:

```text
les template
les JSON med boxes
for hver boks:
    lag crop
    lag maske
    lag prompt fra label og value
    kjør Stable Diffusion Inpainting
    lim resultat tilbake
```

Denne metoden ga mer kontroll enn å bruke én stor maske, men modellen hadde fortsatt problemer med å generere korrekt og lesbar tekst.

Debug-crops fra denne metoden kan typisk inneholde:

```text
*_input.png
*_mask.png
*_edited.png
```

Disse viser hva modellen fikk inn, hvilken maske som ble brukt, og hva modellen genererte.

---

# 4. JSON-format

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

Dette gjør at image modification-pipeline-en ikke bare vet hvor den skal redigere, men også hva slags felt det er og hvilken verdi som skal brukes.

---

# 5. Models

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
models/editors/local_box_editor_ref_unet_best_v1.pth
models/yolo/yolov8n.pt
```

---

# 6. Outputs

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

Resultater fra image modification kan ligge i:

```text
image_modification/generated_results/
image_modification/output/
image_modification/outputs/
```

Dette kan variere etter hvilket script som kjøres.

---

# 7. Anbefalt kjørerekkefølge

Hvis alt skal bygges fra bunnen av:

```text
1. Finn eller marker redigerbare felt
2. Lag JSON med label, bbox og value
3. Lag reference crops
4. Lag treningsdata med reference crops
5. Tren reference-editoren
6. Kjør inference med reference-editoren
7. Evaluer output manuelt
```

For sluttmetoden:

```powershell
python Template_modification/image_modification/05_infer_reference_editor.py
```

Hvis modellen, JSON og reference crops allerede finnes, er det ikke nødvendig å kjøre preprocessing og trening hver gang.

---

# 8. Status

Denne delen av prosjektet fungerer som en teknisk prototype. Pipeline-en kan finne og strukturere felt, men sluttresultatene fra AI-redigeringen ble ikke gode nok til å regnes som en ferdig løsning.

De viktigste utfordringene var:

- støy i differanseanalyse
- feil eller upresise bounding boxes
- usikker klassifikasjon
- problemer med presis og lesbar tekst i AI-redigering
- visuell inkonsistens mellom generert innhold og template

---

# 9. Viktig merknad

Stable Diffusion Inpainting og reference-basert U-Net-editor er to ulike metoder:

```text
image_modification/Stable Diffusion Inpainting/
→ bruker ferdigtrent Stable Diffusion-modell
→ trener ikke ny modell
→ brukes til inpainting-forsøk

image_modification/
→ bruker egen U-Net-lignende reference-editor
→ trenes på prosjektets egne crops
→ brukes som sluttmetode
```

Det er derfor ryddig å beholde Stable Diffusion-forsøkene i egen undermappe.
