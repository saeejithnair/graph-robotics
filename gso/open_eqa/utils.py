import numpy as np
import PIL


def sample_images(image_paths, n=3):
    flattened_paths = [x for xs in image_paths.values() for x in xs]
    freqs = {i: flattened_paths.count(i) for i in set(flattened_paths)}
    n = min(n, len(freqs.keys()))
    most_freq_image = max(freqs, key=freqs.get)
    freqs.pop(most_freq_image)

    probs = np.array(list(freqs.values())) / np.sum(list(freqs.values()))
    images = np.random.choice(list(freqs.keys()), size=n - 1, replace=False, p=probs)
    return [most_freq_image] + list(images)


def subset_dict(full_dict, search_keys):
    subset_dict = {}
    for key in search_keys:
        if not key in full_dict.keys():
            print("Key", key, "not found in dict")
            continue
        subset_dict[key] = full_dict[key]
    return subset_dict


def open_image(path, new_size=(800, 480)):
    return PIL.Image.open(path).resize(new_size)
