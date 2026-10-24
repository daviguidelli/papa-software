"""Detection against a synthetic frame, so the blue filter is actually run."""
import numpy as np

from papa.vision import presence


def frame_with_blue_square(x, y, size, hue_rgb=(28, 92, 190)):
    img = np.full((400, 600, 3), 24, dtype=np.uint8)
    img[y:y + size, x:x + size] = hue_rgb
    return img


def identity_map():
    # 1 px per mm, no rotation: keeps the test about the filter, not the maths
    return presence.CameraMap(
        points_px=[[0, 0], [100, 0], [100, 100], [0, 100]],
        points_mm=[[0, 0], [100, 0], [100, 100], [0, 100]],
    )


BAND = presence.HsvBand(h=(100, 130), s_min=80, v_min=60)


def test_a_full_film_is_present():
    img = frame_with_blue_square(50, 50, 25)
    d = presence.slide_present(img, identity_map(), (50, 50), 25, 25, BAND)
    assert d.present and d.fraction > 0.9


def test_an_empty_slot_is_absent():
    img = frame_with_blue_square(300, 300, 25)
    d = presence.slide_present(img, identity_map(), (50, 50), 25, 25, BAND)
    assert not d.present and d.fraction < 0.05


def test_a_partly_covered_square_reads_as_misaligned():
    img = frame_with_blue_square(60, 60, 12)   # offset and smaller
    d = presence.slide_present(img, identity_map(), (50, 50), 25, 25, BAND)
    assert not d.present and d.misaligned


def test_stem_disc_detects_a_loaded_swab():
    img = np.full((400, 600, 3), 24, dtype=np.uint8)
    img[98:103, 148:153] = (36, 108, 214)      # a stem seen end-on
    d = presence.swab_present(img, identity_map(), (150, 100), 4.0, BAND)
    assert d.present


def test_sampling_real_pixels_brackets_their_hue():
    img = frame_with_blue_square(50, 50, 40)
    band = presence.sample_hsv(img, (60, 60), radius_px=5)
    assert band.h[0] < band.h[1]
    d = presence.slide_present(img, identity_map(), (50, 50), 25, 25, band)
    assert d.present
