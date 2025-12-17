import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.draw import ellipse, disk
import random

def generate_stem_image(
    img_size=768,
    line_spacing=15,
    atom_spacing=15,
    line_angle_deg=1,
    atom_r_major=7,
    atom_r_minor=3,
    interstitial_r=4,
    atom_tilt_deg=-40,
    atom_intensity=0.7,
    interstitial_intensity=1.6,
    background=0.1,
    noise_level=0.1,
    n_vacancies=4,
    n_interstitials=4,
    seed=None
):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    img = np.full((img_size, img_size), background, dtype=float)

    # Angles
    line_angle = np.deg2rad(line_angle_deg)
    atom_tilt = np.deg2rad(atom_tilt_deg)

    # Lattice directions
    line_dir = np.array([np.cos(line_angle), np.sin(line_angle)])
    perp_dir = np.array([-line_dir[1], line_dir[0]])
    
    center = np.array([img_size // 2, img_size // 2])
    n_lines = img_size // line_spacing

    lattice_positions = []

    # Build lattice
    for i in range(-n_lines, n_lines):
        line_origin = center + i * line_spacing * perp_dir
        for j in range(-img_size // atom_spacing, img_size // atom_spacing):
            pos = line_origin + j * atom_spacing * line_dir
            x, y = int(pos[0]), int(pos[1])
            if 0 <= x < img_size and 0 <= y < img_size:
                lattice_positions.append(np.array([x, y]))

    lattice_positions = np.array(lattice_positions)

    # Choose vacancy sites
    vacancy_indices = random.sample(range(len(lattice_positions)), n_vacancies)
    vacancy_set = set(vacancy_indices)

    # Draw lattice atoms
    for idx, (x, y) in enumerate(lattice_positions):
        if idx in vacancy_set:
            continue

        rr, cc = ellipse(
            y, x,
            atom_r_minor,
            atom_r_major,
            rotation=atom_tilt,
            shape=img.shape
        )
        img[rr, cc] += atom_intensity

    # ---------------------------------------------------
    # Interstitials: exact midpoints between neighbors
    # ---------------------------------------------------

    interstitial_sites = []

    for pos in lattice_positions:
        # midpoint to neighbor along line direction
        mid1 = pos + 0.5 * atom_spacing * line_dir
        # midpoint to neighbor across lines
        mid2 = pos + 0.5 * line_spacing * perp_dir

        for mid in (mid1 + 0.5 * line_spacing * perp_dir,
                    mid2 + 0.5 * atom_spacing * line_dir):
            x, y = int(mid[0]), int(mid[1])
            if 0 <= x < img_size and 0 <= y < img_size:
                interstitial_sites.append((x, y))

    # Remove duplicates
    interstitial_sites = list(set(interstitial_sites))

    # Select desired number
    chosen_interstitials = random.sample(
        interstitial_sites, min(n_interstitials, len(interstitial_sites))
    )
    
    # Draw interstitial atoms (small + round)
    for x, y in chosen_interstitials:
        rr, cc = disk((y, x), radius=interstitial_r, shape=img.shape)
        img[rr, cc] += atom_intensity * interstitial_intensity

    # STEM blur + noise
    img = gaussian_filter(img, sigma=1.6)
    img += np.random.normal(0, noise_level, img.shape)

    # Normalize
    img -= img.min()
    img /= img.max()

    return img

savepath = r"C:\Users\proks\OneDrive\Documents\GitHub\2025_Hackathon\generated artificial data".replace("\\", "/")

# which random algorithm
seed = 45

if __name__ == "__main__":
    img = generate_stem_image(seed=seed)
    
    figure, axis = plt.subplots(figsize=(5,5), dpi=300)#, constrained_layout = True)#, layout="tight")#, gridspec_kw={'height_ratios': [1, 1, 1]}, sharex=True)
    # axis.figure(figsize=(6, 6))
    axis.imshow(img, cmap="gray")
    axis.axis("off")
    # plt.title("Synthetic STEM with Centered Interstitials")
    figure.savefig(savepath+'/STEM_' + str(seed) + '.jpeg', bbox_inches="tight", dpi=300)
    plt.show()
