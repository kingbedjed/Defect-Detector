import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter  # for Gaussian blurring
from skimage.draw import disk               # to draw circular atoms/interstitials
import random
from PIL import Image
import pandas as pd

# Function to rotate a point around a given line point by a specified angle
def rotate_about_line(point, line_point, angle):
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle),  np.cos(angle)]]) # 2D rotation matrix
    return line_point + R @ (point - line_point)

def generate_stem_image_with_continuous_gb(

    img_size=768,              # Size of the square image (pixels)

    line_spacing=20,           # Spacing between lattice lines
    atom_spacing=20,           # Spacing between atoms along lines

    lattice_angle_deg=50,       # Overall rotation of lattice (degrees)

    gb_misorientation_deg=0,   # Grain boundary misorientation (degrees) (don't have this too high, like 10 or less)

    gb_angle=np.deg2rad(-40),   # For twinning: equal to lattice_angle_deg # Angle of grain boundary relative to horizontal
    gb_height=0.8,             # Relative vertical position of GB (0=bottom, 1=top)

    atom_tilt_deg=8,           # Tilt of dumbbell atoms (degrees)
    dumbbell_separation=4.0,   # Distance between the two "lobes" of a dumbbell atom
    dumbbell_radius=5.0,        # Radius of each dumbbell lobe
    atom_intensity=2,           # Brightness of atoms

    interstitial_r = 4,         # Radius of interstitial atoms
    interstitial_intensity=1,   # Brightness relative to normal atoms

    background=0.15,           # Base background intensity
    noise_level=0.2,           # Gaussian noise added to final image
    gauss_sigma=2.0,           # Gaussian blur applied to final image

    n_vacancies=1,             # Number of missing atoms
    n_interstitials=2,         # Number of extra interstitial atoms

    seed=None                  # Random seed for reproducibility
):

    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    bounding_boxes = [] # to store bounding boxes of defects in format (class, x1, y1, x2, y2, x3, y3, x4, y4)
    classes = {
        'gb': 0,
        'vacancy': 1,
        'interstitial': 2
    }

    # Create empty image with uniform background intensity
    img = np.full((img_size, img_size), background, dtype=float)

    # Grain boundary parameters (slope m and intercept b)
    m = np.tan(gb_angle)
    b = img_size * gb_height

    # Add grain boundary bounding box keeping the coords within image bounds
    top_left_y = np.clip(b + line_spacing, 0, img_size)
    top_left_x = (top_left_y - (b + line_spacing)) / m
    bottom_left_y = np.clip(b - line_spacing, 0, img_size)
    bottom_left_x = (bottom_left_y - (b - line_spacing)) / m
    top_right_y = np.clip(m * img_size + b + line_spacing, 0, img_size)
    top_right_x = (top_right_y - (b + line_spacing)) / m
    bottom_right_y = np.clip(m * img_size + b - line_spacing, 0, img_size)
    bottom_right_x = (bottom_right_y - (b - line_spacing)) / m
    bounding_boxes.append(
        (classes['gb'],
         bottom_left_x, bottom_left_y,
         bottom_right_x, bottom_right_y,
         top_right_x, top_right_y,
         top_left_x, top_left_y
        )
    )

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
                # tilt = np.deg2rad(45-atom_tilt_deg) # this makes sure atoms tilt relative to lattice
                tilt = np.deg2rad(90-atom_tilt_deg) # this makes sure atoms tilt relative to lattice

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
    # vacancy_set are all the vacancy positions

    # Draw dumbbell atoms
    for idx, (pos, tilt) in enumerate(atoms):
        if idx in vacancy_set:
            bounding_boxes.append(
                (classes['vacancy'],
                 pos[0] - line_spacing, pos[1] - line_spacing,
                 pos[0] + line_spacing, pos[1] - line_spacing,
                 pos[0] + line_spacing, pos[1] + line_spacing,
                 pos[0] - line_spacing, pos[1] + line_spacing)
            )
            print('vacancy position', pos)
            continue # skip vacancies
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

    # Remove duplicates and randomly select a subset to be used as interstitial locations
    interstitial_sites = list(set(interstitial_sites))
    chosen = random.sample(interstitial_sites,
                           min(n_interstitials, len(interstitial_sites)))
    # chosen = coordinates for interstitial sites
    # Draw interstitial atoms at chosen coordinates
    box_width = line_spacing / 2
    for x, y in chosen:
        print('interstitial position:',x,y)
        bounding_boxes.append(
            (classes['interstitial'],
             x - box_width, y - box_width,
             x + box_width, y - box_width,
             x + box_width, y + box_width,
             x - box_width, y + box_width)
        )
        rr, cc = disk((y, x), radius=interstitial_r, shape=img.shape)
        img[rr, cc] += atom_intensity * interstitial_intensity

    # Apply Gaussian blur to simulate microscope resolution
    img = gaussian_filter(img, sigma=gauss_sigma)
    # Add Gaussian noise
    img += np.random.normal(0, noise_level, img.shape)

    # # Normalize image to [0,1] range
    # # I don't know if this actually works
    img -= img.min()
    img /= img.max()

    return img, bounding_boxes

def rotate_flip_image_and_bboxes(image, bounding_boxes, rot, refl):
    # Rotate image
    rotated_img = image.rotate(rot, expand=True)
    # Flip image
    if refl == 'horizontal' or refl == 'both':
        rotated_img = rotated_img.transpose(Image.FLIP_LEFT_RIGHT)
    if refl == 'vertical' or refl == 'both':
        rotated_img = rotated_img.transpose(Image.FLIP_TOP_BOTTOM)

    transformed_bboxes = []
    for box in bounding_boxes:
        cls = box[0]
        coords = np.array(box[1:]).reshape((4, 2))  # reshape to 4 (x,y) points

        # Rotate points around image center
        center = np.array(image.size) / 2
        angle_rad = np.deg2rad(-rot)
        R = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                      [np.sin(angle_rad),  np.cos(angle_rad)]])
        rotated_coords = np.dot(coords - center, R.T) + center

        # Apply reflections
        if refl == 'horizontal':
            rotated_coords[:, 0] = image.size[0] - rotated_coords[:, 0]
        elif refl == 'vertical':
            rotated_coords[:, 1] = image.size[1] - rotated_coords[:, 1]
        elif refl == 'both':
            rotated_coords[:, 0] = image.size[0] - rotated_coords[:, 0]
            rotated_coords[:, 1] = image.size[1] - rotated_coords[:, 1]

        transformed_bboxes.append(
            (cls,
             rotated_coords[0,0], rotated_coords[0,1],
             rotated_coords[1,0], rotated_coords[1,1],
             rotated_coords[2,0], rotated_coords[2,1],
             rotated_coords[3,0], rotated_coords[3,1])
        )

    return rotated_img, transformed_bboxes

# Path to save generated images
savepath = r"C:\Users\proks\OneDrive\Documents\GitHub\2025_Hackathon\generated artificial data".replace("\\", "/")

seed = 9  # seed for reproducibility

if __name__ == "__main__":
    # Generate fake STEM image
    img, bounding_boxes = generate_stem_image_with_continuous_gb(seed=seed)

    # Plot and save image
    figure, axis = plt.subplots(figsize=(5,5), dpi=300)
    axis.imshow(img, cmap="gray")
    # show the bounding boxes color coded by class
    for box in bounding_boxes:
        cls = box[0]
        x = list(box[1::2])
        y = list(box[2::2])
        if cls == 0:
            color = 'red'  # grain boundary
        elif cls == 1:
            color = 'blue' # vacancy
        elif cls == 2:
            color = 'green' # interstitial
        axis.plot(
            x + [x[0]],
            y + [y[0]],
            color=color, linewidth=.5)
    axis.axis("off")
    figure.savefig(savepath+'/STEM_' + str(seed) + '.jpeg', bbox_inches="tight", dpi=300)
    plt.show()
    image = Image.fromarray((img * 255).astype(np.uint8))
    image.save(savepath+'/STEM_' + str(seed) + '.tif')
    # Save bounding boxes to CSV
    bbox_df = pd.DataFrame(bounding_boxes, columns=['class', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4'])
    bbox_df.to_csv(savepath+'/STEM_' + str(seed) + '_bboxes.csv', index=False)

    rotations = [0, 90, 180, 270]  # degrees
    reflections = ['none', 'horizontal', 'vertical', 'both']

    # --- Apply transformations and save images ---
    for rot in rotations:
        for refl in reflections:
            if rot == 0 and refl == 'none':
                continue  # skip original image
            transformed_img, transformed_bboxes = rotate_flip_image_and_bboxes(image, bounding_boxes, rot, refl)
            # import pdb; pdb.set_trace()

            # Save the transformed image
            transformed_img.save(savepath+'/STEM_' + str(seed) + f'_rot{rot}_refl{refl}.tif')
            # Save transformed bounding boxes to CSV
            transformed_bboxes_df = pd.DataFrame(transformed_bboxes, columns=['class', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4'])
            transformed_bboxes_df.to_csv(savepath+'/STEM_' + str(seed) + f'_rot{rot}_refl{refl}_bboxes.csv', index=False)
            # plot and save transformed image with bounding boxes
            figure, axis = plt.subplots(figsize=(5,5), dpi=300)
            axis.imshow(transformed_img, cmap="gray")
            for box in transformed_bboxes:
                cls = box[0]
                x = list(box[1::2])
                y = list(box[2::2])
                if cls == 0:
                    color = 'red'  # grain boundary
                elif cls == 1:
                    color = 'blue' # vacancy
                elif cls == 2:
                    color = 'green' # interstitial
                axis.plot(
                    x + [x[0]],
                    y + [y[0]],
                    color=color, linewidth=.5)
            axis.axis("off")
            figure.savefig(savepath+'/STEM_' + str(seed) + f'_rot{rot}_refl{refl}.jpeg', bbox_inches="tight", dpi=300)
