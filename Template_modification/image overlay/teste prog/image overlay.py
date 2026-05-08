from PIL import Image

# Åpne bakgrunnsbildet
background = Image.open("VeryGoodExample.png").convert("RGBA")

# Åpne det mindre bildet som skal legges oppå
overlay = Image.open("image.png").convert("RGBA")

# Endre størrelse på det lille bildet hvis du vil
overlay = overlay.resize((200, 200))

# Velg hvor det lille bildet skal plasseres
# (x, y) = (avstand fra venstre, avstand fra toppen)
position = (50, 50)

# Legg bildet oppå bakgrunnsbildet
background.paste(overlay, position, overlay)

# Lagre resultatet
background.save("resultat.png")

print("Ferdig! Bildet er lagret som resultat.png")