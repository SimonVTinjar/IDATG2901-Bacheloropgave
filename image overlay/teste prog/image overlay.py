from PIL import Image

background = Image.open("VeryGoodExample.png").convert("RGBA")

# Åpne det mindre bildet som skal legges oppå
overlay = Image.open("image.png").convert("RGBA")


# Gjør overlay til 25% av bredden til bakgrunnsbildet
new_width = background.width // 4
ratio = new_width / overlay.width
new_height = int(overlay.height * ratio)

overlay = overlay.resize((new_width, new_height))

# Plasser nederst til høyre
x = background.width - overlay.width - 20
y = background.height - overlay.height - 20

background.paste(overlay, (x, y), overlay)
background.save("img resultat.png")

print("Ferdig!")