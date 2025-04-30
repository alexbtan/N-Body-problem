from PIL import Image

# Open both images
img1 = Image.open("energy_wh-nih.png")
img2 = Image.open("conservation_combined.png")

# Combine side by side
combined = Image.new("RGBA", (img1.width + img2.width, max(img1.height, img2.height)))
combined.paste(img1, (0, 0))
combined.paste(img2, (img1.width, 0))

# Save result
combined.save("allconsv.png")