LABELS_csgi = ['_background_', 'Bush area', 'Grass', 'Lake', 'Terrace greenery', 'Tree']

# Class to color (BGR)
LABELMAP_csgi = {
    0: (0, 0, 0),
    1: (0, 0, 128),
    2: (0, 128, 0),
    3: (0, 128, 128),
    4: (128, 0, 0),
    5: (128, 0, 128),
}

# Color (BGR) to class
INV_LABELMAP_csgi = {
    (0, 0, 0): 0,
    (0, 0, 128): 1,
    (0, 128, 0): 2,
    (0, 128, 128): 3,
    (128, 0, 0): 4,
    (128, 0, 128): 5,
}

LABELMAP_RGB_csgi = {k: (v[2], v[1], v[0]) for k, v in LABELMAP_csgi.items()}

INV_LABELMAP_RGB_cef = {v: k for k, v in LABELMAP_RGB_csgi.items()}

train_ids = [

]

val_ids = [

]

test_ids = [

]
