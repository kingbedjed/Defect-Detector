import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.draw import disk
import random

def rotate_about_line(point, line_point, angle):
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    return line_point + R @ (point - line_point)

def generate_stem_image_with_continuous_gb(
        
    img_size=768,
    
    line_spacing=20,
    atom_spacing=20,
    
    lattice_angle_deg=-2,
    gb_misorientation_deg=2,
    
    gb_angle=np.deg2rad(15),
    gb_height=0.2, # 0.75 (0 to 1)
    # gb_angle=np.deg2rad(random.uniform(10,20)),
    # gb_height = random.uniform(0,1),
                
    atom_tilt_deg=-40,
    dumbbell_separation=3.0, 
    dumbbell_radius=4.5,
    atom_intensity=2,
    
    interstitial_r = 4,
    interstitial_intensity=1, # relative to atom intensities
    
    background=0.15,
    noise_level=0.2,
    gauss_sigma=2,
    
    n_vacancies=6,
    n_interstitials=6,
    
    seed=None
):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    img = np.full((img_size, img_size), background, dtype=float)

    # # Grain boundary
    # gb_angle = np.deg2rad(-15)
    m = np.tan(gb_angle)
    b = img_size * gb_height

    # Reference lattice
    lattice_angle = np.deg2rad(lattice_angle_deg)
    line_dir = np.array([np.cos(lattice_angle), np.sin(lattice_angle)])
    perp_dir = np.array([-line_dir[1], line_dir[0]])

    center = np.array([img_size // 2, img_size // 2])
    n_lines = img_size // line_spacing

    atoms = []

    for i in range(-n_lines, n_lines):
        for j in range(-img_size // atom_spacing,
                       img_size // atom_spacing):

            pos = center + j * atom_spacing * line_dir + i * line_spacing * perp_dir
            x, y = pos
            
            # # make x and y positions slightly random
            array_offset=10
            x = x + random.uniform(-array_offset, array_offset)
            y = y + random.uniform(-array_offset, array_offset)
            
            if not (0 <= x < img_size and 0 <= y < img_size):
                continue

            # Projection onto GB
            xg = (x + m * (y - b)) / (1 + m**2)
            yg = m * xg + b
            gb_point = np.array([xg, yg])
            
            if y > m * x + b:
                angle = np.deg2rad(gb_misorientation_deg)
                tilt = np.deg2rad(atom_tilt_deg)
            else:
                angle = np.deg2rad(-gb_misorientation_deg)
                tilt = np.deg2rad(-atom_tilt_deg)

            new_pos = rotate_about_line(pos, gb_point, angle)
            
            # # make x and y positions slightly random
            array_offset=1
            new_pos[0] = new_pos[0] + random.uniform(-array_offset, array_offset)
            new_pos[1] = new_pos[1] + random.uniform(-array_offset, array_offset)
            
            atoms.append((new_pos, tilt))

    # Vacancies
    vacancy_indices = random.sample(range(len(atoms)), n_vacancies)
    vacancy_set = set(vacancy_indices)

    # ---- Draw dumbbell atoms ----
    for idx, (pos, tilt) in enumerate(atoms):
        if idx in vacancy_set:
            continue

        cx, cy = pos
        dx = dumbbell_separation * np.cos(tilt)
        dy = dumbbell_separation * np.sin(tilt)

        for sx, sy in [(cx + dx, cy + dy), (cx - dx, cy - dy)]:
            x, y = int(sx), int(sy)
            # print(x,y)
            
            # # make x and y positions slightly random
            # array_offset=1
            # x = x + random.uniform(-array_offset, array_offset)
            # y = y + random.uniform(-array_offset, array_offset)
            # print(x,y)
            # print()
            if 0 <= x < img_size and 0 <= y < img_size:
                rr, cc = disk((y, x), dumbbell_radius, shape=img.shape)
                img[rr, cc] += atom_intensity * 0.5

    # Interstitials (still round)
    interstitial_sites = []
    for pos, _ in atoms:
        mid = pos + 0.5 * atom_spacing * line_dir + 0.5 * line_spacing * perp_dir
        x, y = map(int, mid)
        if 0 <= x < img_size and 0 <= y < img_size:
            interstitial_sites.append((x, y))

    interstitial_sites = list(set(interstitial_sites))
    chosen = random.sample(interstitial_sites,
                           min(n_interstitials, len(interstitial_sites)))

    for x, y in chosen:
        rr, cc = disk((y, x), radius=interstitial_r, shape=img.shape)
        img[rr, cc] += atom_intensity * interstitial_intensity

    # Blur + noise
    img = gaussian_filter(img, sigma=gauss_sigma)
    img += np.random.normal(0, noise_level, img.shape)

    img -= img.min()
    img /= img.max()

    return img

savepath = r"C:\Users\proks\OneDrive\Documents\GitHub\2025_Hackathon\generated artificial data".replace("\\", "/")

seed = 20

if __name__ == "__main__":
    img = generate_stem_image_with_continuous_gb(seed=seed)
    
    figure, axis = plt.subplots(figsize=(5,5), dpi=300)#, constrained_layout = True)#, layout="tight")#, gridspec_kw={'height_ratios': [1, 1, 1]}, sharex=True)
    # axis.figure(figsize=(6, 6))
    axis.imshow(img, cmap="gray")
    axis.axis("off")
    # plt.title("Synthetic STEM with Centered Interstitials")
    figure.savefig(savepath+'/STEM_' + str(seed) + '.jpeg', bbox_inches="tight", dpi=300)
    plt.show()
