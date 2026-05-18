# Bacheloroppgave

Dette repositoriet inneholder kode og materiale tilknyttet en bacheloroppgave som omhandler bruk av generative maskinlæringsmodeller for realistisk bildegenerering.

Oppgaven undersøker i hvilken grad moderne generative modeller kan produsere virkelighetsnære bilder, samt hvilke etiske og sikkerhetsmessige utfordringer som følger av slik teknologi. Arbeidet inkluderer både praktiske eksperimenter og teoretiske vurderinger.

Prosjektet er gjennomført som en del av bachelorstudiet og benyttes utelukkende til akademiske og forskningsmessige formål.

---

## Mappeoversikt

### `datasett copy/`

Inneholder `.txt`-filer som beskriver hvert bilde i datasettet.

Hver `.txt`-fil har samme filnavn som bildet den tilhører.

---

### `datasett/`

Inneholder bilder av 100-dollar-sedler som brukes i prosjektet.

---

### `Kode/`


Denne mappen inneholde kode for:


testing og evaluering

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

### `image overlay/`

Inneholder `.txt`-filer som ble brukt under LoRA-trening sammen med bildene i `datasett/`.

---

### `thai 20 front/`

Inneholder bilder av thailandske 20 baht-sedler.

Hvert bilde har en tilhørende `.txt`-fil med samme filnavn. Tekstfilen beskriver bildet den hører til.
---




