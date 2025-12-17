import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter  # for Gaussian blurring
from skimage.draw import disk               # to draw circular atoms/interstitials
import random

# Function to rotate a point around a given line point by a specified angle
def rotate_about_line(point, line_point, angle):
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle),  np.cos(angle)]]) # 2D rotation matrix
    return line_point + R @ (point - line_point)

def generate_stem_image_with_continuous_gb(
        
    img_size=768,              # Size of the square image (pixels)
    
    line_spacing=20,           # Spacing between lattice lines
    atom_spacing=20,           # Spacing between atoms along lines
    
    lattice_angle_deg=20,       # Overall rotation of lattice (degrees)
    
    gb_misorientation_deg=5,   # Grain boundary misorientation (degrees) (don't have this too high, like 10 or less)
    
    gb_angle=np.deg2rad(20),   # For twinning: equal to lattice_angle_deg # Angle of grain boundary relative to horizontal
    gb_height=0.001,             # Relative vertical position of GB (0=bottom, 1=top)
                
    atom_tilt_deg=90,           # Tilt of dumbbell atoms (degrees)
    dumbbell_separation=4.0,   # Distance between the two "lobes" of a dumbbell atom
    dumbbell_radius=5.0,        # Radius of each dumbbell lobe
    atom_intensity=2,           # Brightness of atoms
    
    interstitial_r = 4,         # Radius of interstitial atoms
    interstitial_intensity=1,   # Brightness relative to normal atoms
    
    background=0.15,           # Base background intensity
    noise_level=0.2,           # Gaussian noise added to final image
    gauss_sigma=2.0,           # Gaussian blur applied to final image
    
    n_vacancies=6,             # Number of missing atoms
    n_interstitials=10,         # Number of extra interstitial atoms
    
    seed=None                  # Random seed for reproducibility
):
    
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        
    # Create empty image with uniform background intensity
    img = np.full((img_size, img_size), background, dtype=float)

    # Grain boundary parameters (slope m and intercept b)
    m = np.tan(gb_angle)
    b = img_size * gb_height

    # Lattice directions
    lattice_angle = np.deg2rad(lattice_angle_deg)           # Convert lattice rotation to radians
    line_dir = np.array([np.cos(lattice_angle), np.sin(lattice_angle)])  # Direction along lines
    perp_dir = np.array([-line_dir[1], line_dir[0]])       # Perpendicular direction

    center = np.array([img_size // 2, img_size // 2])      # Image center
    n_lines = img_size // line_spacing                      # Number of lattice lines to draw

    atoms = []  # list to store atom positions and tilts

    # Generate atom positions on lattice
    for i in range(-n_lines, n_lines):
        for j in range(-img_size // atom_spacing,
                       img_size // atom_spacing):

            # Compute ideal atom position on lattice
            pos = center + j * atom_spacing * line_dir + i * line_spacing * perp_dir
            x, y = pos
            
            # Small random perturbation to make lattice less perfect
            array_offset=2 # amount to randomize by
            x = x + random.uniform(-array_offset, array_offset)
            y = y + random.uniform(-array_offset, array_offset)
            
            # Skip atoms outside image bounds
            if not (0 <= x < img_size and 0 <= y < img_size):
                continue

            # Project atom onto grain boundary line
            xg = (x + m * (y - b)) / (1 + m**2)
            yg = m * xg + b
            gb_point = np.array([xg, yg])
            
            # Determine which side of GB the atom is on
            if y > m * x + b:
                angle = np.deg2rad(gb_misorientation_deg)  # rotation for this grain
                tilt = np.deg2rad(atom_tilt_deg)           # dumbbell tilt
            else:
                angle = np.deg2rad(gb_misorientation_deg)
                tilt = np.deg2rad(45-atom_tilt_deg) # this makes sure atoms tilt relative to lattice

            # Rotate atom around its projected GB point
            new_pos = rotate_about_line(pos, gb_point, angle)
            
            # Add small random jitter
            array_offset=1
            new_pos[0] = new_pos[0] + random.uniform(-array_offset, array_offset)
            new_pos[1] = new_pos[1] + random.uniform(-array_offset, array_offset)
            
            # Store atom position and tilt
            atoms.append((new_pos, tilt))

    # Randomly select indices for vacancies (missing atoms)
    vacancy_indices = random.sample(range(len(atoms)), n_vacancies)
    vacancy_set = set(vacancy_indices)

    # Draw dumbbell atoms
    for idx, (pos, tilt) in enumerate(atoms):
        if idx in vacancy_set:
            continue  # skip vacancies

        cx, cy = pos
        dx = dumbbell_separation * np.cos(tilt)
        dy = dumbbell_separation * np.sin(tilt)

        # Each dumbbell has two lobes, draw both
        for sx, sy in [(cx + dx, cy + dy), (cx - dx, cy - dy)]:
            x, y = int(sx), int(sy)
            if 0 <= x < img_size and 0 <= y < img_size:
                rr, cc = disk((y, x), dumbbell_radius, shape=img.shape)
                img[rr, cc] += atom_intensity * 0.5  # add brightness to image

    # Interstitial atoms (extra atoms at midpoints)
    interstitial_sites = []
    for pos, _ in atoms:
        mid = pos + 0.5 * atom_spacing * line_dir + 0.5 * line_spacing * perp_dir
        x, y = map(int, mid)
        if 0 <= x < img_size and 0 <= y < img_size:
            interstitial_sites.append((x, y))

    # Remove duplicates and randomly select a subset
    interstitial_sites = list(set(interstitial_sites))
    chosen = random.sample(interstitial_sites,
                           min(n_interstitials, len(interstitial_sites)))

    # Draw interstitial atoms
    for x, y in chosen:
        rr, cc = disk((y, x), radius=interstitial_r, shape=img.shape)
        img[rr, cc] += atom_intensity * interstitial_intensity

    # Apply Gaussian blur to simulate microscope resolution
    img = gaussian_filter(img, sigma=gauss_sigma)
    # Add Gaussian noise
    img += np.random.normal(0, noise_level, img.shape)

    # # Normalize image to [0,1] range
    # # I don't know if this actually works
    # img -= img.min()
    # img /= img.max()

    return img

# Path to save generated images
savepath = r"C:\Users\proks\OneDrive\Documents\GitHub\2025_Hackathon\generated artificial data".replace("\\", "/")

seed = 3  # seed for reproducibility

if __name__ == "__main__":
    # Generate synthetic STEM image
    img = generate_stem_image_with_continuous_gb(seed=seed)
    
    # Plot and save image
    figure, axis = plt.subplots(figsize=(5,5), dpi=300)
    axis.imshow(img, cmap="gray")
    axis.axis("off")
    figure.savefig(savepath+'/STEM_' + str(seed) + '.jpeg', bbox_inches="tight", dpi=300)
    plt.show()