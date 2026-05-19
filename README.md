# Bacheloroppgave

Dette repositoriet inneholder kode og materiale tilknyttet en bacheloroppgave som omhandler bruk av generative maskinlæringsmodeller for realistisk bildegenerering.

Oppgaven undersøker i hvilken grad moderne generative modeller kan produsere virkelighetsnære bilder, samt hvilke etiske og sikkerhetsmessige utfordringer som følger av slik teknologi. Arbeidet inkluderer både praktiske eksperimenter og teoretiske vurderinger.

Prosjektet er gjennomført som en del av bachelorstudiet og benyttes utelukkende til akademiske og forskningsmessige formål.

---

## Mappeoversikt

---

### `Brukertest_resultater/`

Inneholder rådata fra brukertesten gjennomført i NTNU Colourlabs QuickEval. Mappen består av:

- `Quickeval - resultater.csv` – eksporterte resultater fra QuickEval, inkludert Likert-svar per bilde per deltaker og demografiske svar
- `Spørreskjema - resultater.csv` – svar fra det påfølgende spørreskjemaet i Google Forms, inkludert fritekstbesvarelser og vanskelighetsgrad

---

### `datasett copy/`

Inneholder `.txt`-filer som beskriver hvert bilde i datasettet.

Hver `.txt`-fil har samme filnavn som bildet den tilhører.


---

### `Kode/`


Denne mappen inneholde koder for:


Skalering - forskjellige operasjoner å transformere bilder på, som rotasjon, skalering eller bakgrunnskutt.
Tests - De forskjellige testene for PSNR, SSIN, LPIPS og FID. Også resultatjsons og pdfer.
Inneholder også mapper med grafer.

---

### `Template_modification/`

Inneholder ulike metoder som ble brukt for å identifisere feil eller avvik i templaten.

Mappen dokumenterer også hvordan de oppdagede områdene ble brukt videre for å generere nytt innhold i feltene der feil ble funnet.

---

### `datasett editor/`

Inneholder tre mindre Python-programmer brukt til å forberede datasettet.

Programmene ble brukt til å:

- lage `.txt`-filer for hvert bilde
- legge inn en felles tekstbeskrivelse i hver `.txt`-fil
- splitte datasettet inn i ulike mapper

---
### `draw.io/`

Denne mappen inneholde diagramer som har blit laget for å ha i raporten raporten


---

### `image overlay/`

Inneholder `.txt`-filer som ble brukt under LoRA-trening sammen med bildene i `datasett/`.

---

### `lora_dataset/`

Denne mappen inneholde filer for:

notasjon på bilder som lora treningen vår brukte

---


### `lora_training/`

Inneholder skript brukt til datasettforberedelse, bildegenerering og annotasjon i forbindelse med LoRA-finjustering av Z-Image og Z-Image-Turbo. Se egen `README.md` i mappen for detaljert beskrivelse av hvert skript.


---




