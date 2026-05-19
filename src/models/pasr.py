def pasr(image_entropy, energy):
    if energy < 0.3 or image_entropy < 0.4:
        return "short_prompt"
    return "detailed_prompt"
